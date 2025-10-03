""" 调用独占时间算子，分析trace的独占时间是在什么地方增加的。对于对独占时间贡献前95%的span，我们找出来。

新增功能：
1. 支持计算normal时间段内每个serviceName的平均独占时间
2. 支持从异常时间段的独占时间中减去对应serviceName的平均值(当minus_average=True时)
3. 这样可以识别出相对于正常情况，独占时间异常增加的span
4. 支持只处理每个trace中独占时间排top-1的span(当only_top1_per_trace=True时)
   - 显著减少计算量，聚焦于每个trace的最耗时操作
   - 适用于快速定位每个trace的主要瓶颈点
   - 重要：当minus_average=True时，会先对所有span减去平均值，再选择调整后时间最长的span

性能优化方案：
方案1（推荐）：直接从span_list获取serviceName和spanName
- 修改SPL查询，同时返回trace_id, span_list, span_id, span_index, exclusive_duration
- 通过span_index直接映射到span_list中的servicename和spanname
- 完全避免重新查询spans数据，性能最优
- 使用_adjust_durations_directly函数进行本地计算

方案2（备选）：采样优化
- 当方案1不可用时的备选方案
- 对span进行采样，最多处理HIGH_RT_TRACES个span
- 减少查询负载，但仍需要额外的数据库查询
- 使用_adjust_durations_with_span_average函数，内置采样机制
"""

import numpy as np
from utils.constants import HIGH_RT_TRACES, TRACES_FOR_AVG_RT, PERCENT_95, MAX_DURATION
import os

# 环境变量配置
from aliyun.log import LogClient, GetLogsRequest
# 调用自定义函数 trace_exclusive_duration 计算每个 span 的独占时间
def get_spl_query(duration_threshold: int, limit_num: int) -> str:
    # print (f"duration_threshold: {duration_threshold}, limit_num: {limit_num}")
    spl = f"""* and duration > {duration_threshold} | set session mode=scan;  set session velox_use_io_executor=true; set session presto_velox_mix_run_not_check_linked_agg_enabled=true;
set session presto_velox_mix_run_support_complex_type_enabled=true;
set session velox_sanity_limit_enabled=false;
set session enable_remote_functions=true;
with t1 as (
    select traceId as trace_id,
        array_agg(
            cast(
                row(
                    case
                        when spanId is null then ''
                        else spanId
                    end,
                    case
                        when parentSpanId is null then ''
                        else parentSpanId
                    end,
                    case
                        when statusCode is null then ''
                        else cast(statusCode as varchar)
                    end,
                    case
                        when serviceName is null then ''
                        else serviceName
                    end,
                    case
                        when spanName is null then ''
                        else spanName
                    end,
                    case
                        when hostname is null then ''
                        else hostname
                    end,
                    case
                        when kind is null then ''
                        else kind
                    end,
                    case
                        when duration is null then -1
                        else cast(duration as bigint)
                    end,
                    case
                        when startTime is null then -1
                        else cast(startTime as bigint)
                    end,
                    case
                        when endTime is null then -1
                        else cast(endTime as bigint)
                    end,
                    case
                        when resources is null then ''
                        else resources
                    end,
                    case
                        when attributes is null then ''
                        else attributes
                    end,
                    case
                        when links is null then ''
                        else links
                    end,
                    case
                        when events is null then ''
                        else events
                    end,
                    case
                        when pid is null then '{{}}'          -- extraInfo
                        else concat('{{"pid":"', pid, '"}}')
                    end
                ) as row (
                    spanid varchar,
                    parentspanid varchar,
                    statuscode varchar,
                    servicename varchar,
                    spanname varchar,
                    hostname varchar,
                    kind varchar,
                    duration bigint,
                    starttime bigint,
                    endtime bigint,
                    resources varchar,
                    attributes varchar,
                    links varchar,
                    events varchar,
                    extrainfo varchar
                )
            )
        ) as span_list
    from log
    where (
            traceId is not null
            and traceId != ''
        )
        and (
            spanId is not null
            and spanId != ''
        )
        and (parentSpanId is not null)
        and (startTime is not null)
        and (duration is not null or endTime is not null)
    group by traceId
    order by trace_id limit {limit_num}
)
select
    trace_id,
    span_list,
    trace_exclusive_duration_result.span_id as span_id,
    trace_exclusive_duration_result.span_index,
    trace_exclusive_duration_result.exclusive_duration
from (
    select trace_id, span_list, trace_exclusive_duration(trace_id, span_list) as trace_exclusive_duration_result
    from (
        select * from t1 order by trace_id limit {limit_num}
    )
)
order by span_id limit {limit_num}
"""
    return spl

# 上面这个SPL语句执行完成后，会得到类似下面的结果：每个trace_id对应一行。有三列 span_id、span_index、exclusive_duration, 每一列都是array类型，长度相等。
# 我们需要将所有span_id对应的exclusive_duration找出来，排序，然后把 exclusive_duration 占前95% 的span_id找出来
"""
span_id,    span_index,    exclusive_duration
["47536226199e3d1a","251e1d6125734573","ce05f5dcd45af66d"],    [1,2,3],    [580000,50000,6679751]
["a7ae3f4e057518c9","2c6c77123d1d76df","9951956134123686","2b3f640090a8c85d","be5dc34095786ad0","5d2b07e043008454"],    [1,2,3,4,5,6],     [919000,198000,761127,1954869,530163,1000000]
"""

class FindRootCauseSpansRT:
    def __init__(self, client, project_name: str, logstore_name: str, region: str, start_time: str, end_time: str,
                 duration_threshold: int = 1000000, limit_num: int = 1000, normal_start_time: str = None,  # type: ignore
                 normal_end_time: str = None, minus_average: bool = False, only_top1_per_trace: bool = False): # type: ignore
        """
        初始化FindRootCauseSpansRT类

        Args:
            client: SLS客户端
            project_name: SLS项目名称
            logstore_name: 日志库名称
            region: 地域
            start_time: 异常开始时间
            end_time: 异常结束时间
            duration_threshold: 持续时间阈值，默认1000000（1秒）
            limit_num: 限制处理的trace数量，默认100
            normal_start_time: 正常时间段开始时间
            normal_end_time: 正常时间段结束时间
            minus_average: 是否减去平均值，默认False
            only_top1_per_trace: 是否只处理每个trace中独占时间排top-1的span，默认False
        """
        self.client = client
        self.project_name = project_name
        self.logstore_name = logstore_name
        self.region = region
        self.start_time = start_time
        self.end_time = end_time
        self.duration_threshold = duration_threshold
        self.limit_num = limit_num
        self.normal_start_time = normal_start_time
        self.normal_end_time = normal_end_time
        self.minus_average = minus_average
        self.only_top1_per_trace = only_top1_per_trace

        # 存储每个spanName的平均独占时间
        self.span_average_durations = {}

        # 如果需要减去平均值，先计算正常时间段的平均值
        if self.minus_average and self.normal_start_time and self.normal_end_time:
            self._calculate_span_averages()

    def find_top_95_percent_spans(self) -> list[str]:
        """
        查找异常时间段内占前95%独占时间的span_id列表

        Returns:
            span_id列表
        """
        # 执行SPL查询获取独占时间数据
        query = get_spl_query(self.duration_threshold, HIGH_RT_TRACES)

        request = GetLogsRequest(
            project=self.project_name,
            logstore=self.logstore_name,
            query=query,
            fromTime=self.start_time,
            toTime=self.end_time,
            # line=HIGH_RT_TRACES # only calculate for 1000 traces
        )

        try:
            response = self.client.get_logs(request)
            logs = [log_item.get_contents() for log_item in response.get_logs()] if response else []
            print(f"查询到的日志条数: {len(logs)}")

            return self._process_exclusive_duration_data(logs)

        except Exception as e:
            print(f"查询SLS时发生错误: {e}")
            return []

    def _process_exclusive_duration_data(self, logs: list) -> list[str]:
        """
        处理独占时间数据，找出占前95%的span_id

        Args:
            logs: SLS查询返回的日志数据

        Returns:
            占前95%独占时间的span_id列表
        """
        # 输出处理模式，默认选择处理所有span
        if self.only_top1_per_trace:
            if self.minus_average and self.span_average_durations:
                mode_description = "只处理每个trace中调整后独占时间排top-1的span（先减去平均值再选择）"
            else:
                mode_description = "只处理每个trace中原始独占时间排top-1的span"
        else:
            mode_description = "处理每个trace中的所有span"
        print(f"🔧 处理模式: {mode_description}")

        # 收集所有的span_id和对应的独占时间
        span_duration_mapping = {}  # {span_id: exclusive_duration}
        span_service_mapping = {}   # {span_id: (serviceName, spanName)} - 新增：直接从span_list获取

        for log in logs:
            try:
                # 解析数组字段
                span_ids = self._parse_array_field(log.get('span_id', '[]'))
                span_indices = self._parse_array_field(log.get('span_index', '[]'))
                exclusive_durations = self._parse_array_field(log.get('exclusive_duration', '[]'))
                span_list = self._parse_array_field(log.get('span_list', '[]'))

                # 确保数组长度一致
                if len(span_ids) != len(exclusive_durations) or len(span_ids) != len(span_indices):
                    print(f"警告: 数组长度不一致 - span_id({len(span_ids)}), span_index({len(span_indices)}), exclusive_duration({len(exclusive_durations)})")
                    continue

                # 收集span_id和对应的exclusive_duration，同时从span_list获取serviceName和spanName
                if self.only_top1_per_trace:
                    # 只处理每个trace中独占时间排top-1的span
                    if self.minus_average and self.span_average_durations:
                        # 需要减去平均值的情况：先计算所有span的调整后时间，再选择top-1
                        adjusted_spans = []
                        for span_id, span_index, duration in zip(span_ids, span_indices, exclusive_durations):
                            if isinstance(duration, (int, float)) and duration > 0:
                                # 截断异常长的duration，处理outliers
                                truncated_duration = min(duration, MAX_DURATION)

                                # 获取serviceName和spanName，计算调整后的时间
                                adjusted_duration = truncated_duration
                                if span_list and 0 <= span_index < len(span_list):
                                    span_info = span_list[span_index]
                                    service_name, span_name = self._extract_service_and_span_name(span_info)
                                    if service_name and span_name:
                                        combined_key = f"{service_name}<sep>{span_name}"
                                        if combined_key in self.span_average_durations:
                                            avg_duration = self.span_average_durations[combined_key]
                                            adjusted_duration = max(0, truncated_duration - avg_duration)
                                            adjusted_duration = min(adjusted_duration, MAX_DURATION)

                                adjusted_spans.append((span_id, span_index, adjusted_duration, truncated_duration))

                        # 找到调整后独占时间最长的span
                        if adjusted_spans:
                            top_span = max(adjusted_spans, key=lambda x: x[2])  # 按调整后的duration排序
                            span_id, span_index, adjusted_duration, original_duration = top_span
                            span_duration_mapping[span_id] = original_duration  # 存储原始时间，后续会再次调整

                            # 通过span_index从span_list获取serviceName和spanName
                            if span_list and 0 <= span_index < len(span_list):
                                span_info = span_list[span_index]
                                service_name, span_name = self._extract_service_and_span_name(span_info)
                                if service_name and span_name:
                                    span_service_mapping[span_id] = (service_name, span_name)
                    else:
                        # 不需要减去平均值的情况：直接找原始独占时间最长的span
                        valid_spans = []
                        for span_id, span_index, duration in zip(span_ids, span_indices, exclusive_durations):
                            if isinstance(duration, (int, float)) and duration > 0:
                                # 截断异常长的duration，处理outliers
                                truncated_duration = min(duration, MAX_DURATION)
                                valid_spans.append((span_id, span_index, truncated_duration))

                        # 找到原始独占时间最长的span
                        if valid_spans:
                            top_span = max(valid_spans, key=lambda x: x[2])  # 按原始duration排序
                            span_id, span_index, duration = top_span
                            span_duration_mapping[span_id] = duration

                            # 通过span_index从span_list获取serviceName和spanName
                            if span_list and 0 <= span_index < len(span_list):
                                span_info = span_list[span_index]
                                service_name, span_name = self._extract_service_and_span_name(span_info)
                                if service_name and span_name:
                                    span_service_mapping[span_id] = (service_name, span_name)
                else:
                    # 处理所有span（原始逻辑）
                    for span_id, span_index, duration in zip(span_ids, span_indices, exclusive_durations):
                        if isinstance(duration, (int, float)) and duration > 0:
                            # 截断异常长的duration，处理outliers
                            truncated_duration = min(duration, MAX_DURATION)
                            span_duration_mapping[span_id] = truncated_duration

                            # 通过span_index从span_list获取serviceName和spanName
                            if span_list and 0 <= span_index < len(span_list):
                                span_info = span_list[span_index]
                                service_name, span_name = self._extract_service_and_span_name(span_info)
                                if service_name and span_name:
                                    span_service_mapping[span_id] = (service_name, span_name)

            except Exception as e:
                print(f"处理日志数据时发生错误: {e}")
                continue

        if not span_duration_mapping:
            print("没有找到有效的独占时间数据")
            return []

        print(f"总共找到 {len(span_duration_mapping)} 个有效的span独占时间数据")
        print(f"成功映射 {len(span_service_mapping)} 个span的serviceName和spanName")

        # 智能方案选择：检查方案1的成功率
        if self.minus_average and self.span_average_durations:
            if span_service_mapping:
                # 计算方案1的覆盖率
                coverage_rate = len(span_service_mapping) / len(span_duration_mapping)
                print(f"方案1覆盖率: {coverage_rate:.2%} ({len(span_service_mapping)}/{len(span_duration_mapping)})")

                # 如果覆盖率大于50%，使用方案1；否则fallback到方案2
                if coverage_rate > 0.5:
                    print("✅ 选择方案1：直接使用span_list中的serviceName和spanName（推荐）")
                    adjusted_span_durations = self._adjust_durations_directly(span_duration_mapping, span_service_mapping)
                else:
                    print("⚠️  方案1覆盖率较低，fallback到方案2：重新查询并采样")
                    adjusted_span_durations = self._adjust_durations_with_span_average(span_duration_mapping)
            else:
                print("⚠️  方案1失败：span_list中没有找到serviceName和spanName，fallback到方案2")
                adjusted_span_durations = self._adjust_durations_with_span_average(span_duration_mapping)
        else:
            # 即使不减去平均值，也要截断异常长的duration，处理outliers
            adjusted_span_durations = [(span_id, min(duration, MAX_DURATION)) for span_id, duration in span_duration_mapping.items()]

        # 按独占时间降序排序
        adjusted_span_durations.sort(key=lambda x: x[1], reverse=True)

        # 计算总独占时间
        total_duration = sum(duration for _, duration in adjusted_span_durations)
        print(f"总独占时间: {total_duration}")

        if total_duration == 0:
            print("总独占时间为0，无法计算95%")
            return []

        # 找出占前95%的span
        cumulative_duration = 0
        target_duration = total_duration * PERCENT_95 # * 0.95
        top_95_percent_spans = []

        for span_id, duration in adjusted_span_durations:
            cumulative_duration += duration
            top_95_percent_spans.append(span_id)

            if cumulative_duration >= target_duration:
                break

        print(f"占前95%独占时间的span数量: {len(top_95_percent_spans)}")
        print(f"这些span的累计独占时间: {cumulative_duration}, 占总时间的: {cumulative_duration/total_duration*100:.2f}%")

        return top_95_percent_spans

    def _adjust_durations_with_span_average(self, span_duration_mapping: dict) -> list:
        """
        根据serviceName和spanName组合的平均值调整独占时间
        优化方案2：采样HIGH_RT_TRACES这么多个span进行计算

        Args:
            span_duration_mapping: {span_id: exclusive_duration}

        Returns:
            调整后的(span_id, adjusted_duration)列表
        """
        print("🔄 [方案2] 使用采样查询方案进行调整...")
        print(f"🔄 [方案2] 采样最多 {HIGH_RT_TRACES} 个span进行查询")

        adjusted_durations = []
        span_ids = list(span_duration_mapping.keys())

        # 采样优化：如果span数量超过HIGH_RT_TRACES，随机采样
        if len(span_ids) > HIGH_RT_TRACES:
            sorted_span_ids = sorted(span_ids, key=lambda x: span_duration_mapping[x], reverse=True)
            sampled_span_ids = sorted_span_ids[:HIGH_RT_TRACES]
            print(f"从 {len(span_ids)} 个span中采样了 {len(sampled_span_ids)} 个进行查询")
        else:
            sampled_span_ids = span_ids
            print(f"span数量({len(span_ids)})不超过限制，查询所有span")

        # 分批查询，增大批次大小以提高性能
        batch_size = 500  # 增大批次大小
        span_service_mapping = {}  # 存储span_id到(serviceName, spanName)的映射

        for i in range(0, len(sampled_span_ids), batch_size):
            batch_span_ids = sampled_span_ids[i:i+batch_size]
            span_conditions = " or ".join([f"spanId='{span_id}'" for span_id in batch_span_ids])
            service_query = f"* | select spanId, serviceName, spanName from log where {span_conditions}"

            request = GetLogsRequest(
                project=self.project_name,
                logstore=self.logstore_name,
                query=service_query,
                fromTime=self.start_time,
                toTime=self.end_time,
                # line=HIGH_RT_TRACES  # 使用HIGH_RT_TRACES作为查询限制
            )

            try:
                print(f"查询第 {i//batch_size + 1} 批，共 {len(batch_span_ids)} 个span的serviceName和spanName...")
                response = self.client.get_logs(request)
                service_logs = [log_item.get_contents() for log_item in response.get_logs()] if response else []
                print(f"查询到 {len(service_logs)} 条记录")

                # 构建span_id到(serviceName, spanName)的映射
                for log in service_logs:
                    span_id = log.get('spanId')
                    service_name = log.get('serviceName')
                    span_name = log.get('spanName')
                    if span_id and service_name and span_name:
                        span_service_mapping[span_id] = (service_name, span_name)

            except Exception as e:
                print(f"查询第 {i//batch_size + 1} 批时发生错误: {e}")
                continue

        print(f"成功映射 {len(span_service_mapping)} 个span的serviceName和spanName")

        # 使用for循环在本地计算调整后的时间
        print("开始本地计算调整后的独占时间...")
        for span_id, original_duration in span_duration_mapping.items():
            # 如果这个span_id在采样范围内且有映射信息，则使用平均值调整
            if span_id in span_service_mapping:
                service_name, span_name = span_service_mapping[span_id]
                # 构建组合键
                combined_key = f"{service_name}<sep>{span_name}"

                if combined_key in self.span_average_durations:
                    avg_duration = self.span_average_durations[combined_key]
                    adjusted_duration = max(0, original_duration - avg_duration)  # 确保不为负数

                    adjusted_duration = min(adjusted_duration, MAX_DURATION)
                    # print(f"span {span_id} 服务 {service_name} spanName {span_name}: 原始时间={original_duration}, 平均时间={avg_duration:.2f}, 调整后时间={adjusted_duration:.2f}")
                else:
                    adjusted_duration = original_duration
                    # print(f"span {span_id} 服务 {service_name} spanName {span_name}: 没有平均值信息，使用原始时间={original_duration}")
            else:
                # 没有在采样范围内或没有找到serviceName和spanName，使用原始时间
                adjusted_duration = original_duration
                # print(f"span {span_id}: 没有在采样范围内或没有找到serviceName和spanName，使用原始时间={original_duration}")

            adjusted_durations.append((span_id, adjusted_duration))

        print(f"完成 {len(adjusted_durations)} 个span的时间调整计算")
        return adjusted_durations

    def _adjust_durations_directly(self, span_duration_mapping: dict, span_service_mapping: dict) -> list:
        """
        根据serviceName和spanName组合的平均值调整独占时间
        方案1：直接使用span_list中的serviceName和spanName（性能最优）

        Args:
            span_duration_mapping: {span_id: exclusive_duration}
            span_service_mapping: {span_id: (serviceName, spanName)}

        Returns:
            调整后的(span_id, adjusted_duration)列表
        """
        print("🚀 [方案1] 使用span_list中的serviceName和spanName进行调整...")
        print(f"🚀 [方案1] 无需额外查询，直接处理 {len(span_duration_mapping)} 个span")

        adjusted_durations = []
        span_ids = list(span_duration_mapping.keys())

        # 使用for循环在本地计算调整后的时间
        print("开始本地计算调整后的独占时间...")
        for span_id, original_duration in span_duration_mapping.items():
            span_info = span_service_mapping.get(span_id)

            if span_info:
                service_name, span_name = span_info
                # 构建组合键
                combined_key = f"{service_name}<sep>{span_name}"

                if combined_key in self.span_average_durations:
                    avg_duration = self.span_average_durations[combined_key]
                    adjusted_duration = max(0, original_duration - avg_duration)  # 确保不为负数

                    adjusted_duration = min(adjusted_duration, MAX_DURATION)
                    # print(f"span {span_id} 服务 {service_name} spanName {span_name}: 原始时间={original_duration}, 平均时间={avg_duration:.2f}, 调整后时间={adjusted_duration:.2f}")
                else:
                    adjusted_duration = original_duration
                    # print(f"span {span_id} 服务 {service_name} spanName {span_name}: 没有平均值信息，使用原始时间={original_duration}")
            else:
                # 没有找到serviceName和spanName，使用原始时间
                adjusted_duration = original_duration
                # print(f"span {span_id}: 没有找到serviceName和spanName，使用原始时间={original_duration}")

            adjusted_durations.append((span_id, adjusted_duration))

        print(f"完成 {len(adjusted_durations)} 个span的时间调整计算")
        return adjusted_durations

    def _extract_service_and_span_name(self, span_info):
        """
        从span_info中提取serviceName和spanName
        处理多种可能的数据格式

        Args:
            span_info: span信息，可能是dict、list、tuple等格式

        Returns:
            tuple: (service_name, span_name)
        """
        try:
            # 格式1：字典格式
            if isinstance(span_info, dict):
                service_name = span_info.get('servicename', '') or span_info.get('serviceName', '')
                span_name = span_info.get('spanname', '') or span_info.get('spanName', '')
                return service_name, span_name

            # 格式2：列表/元组格式，按照SPL查询中的字段顺序
            # 顺序：spanid, parentspanid, statuscode, servicename, spanname, hostname, kind, ...
            elif isinstance(span_info, (list, tuple)) and len(span_info) >= 5:
                service_name = span_info[3] if len(span_info) > 3 else ''
                span_name = span_info[4] if len(span_info) > 4 else ''
                return service_name, span_name

            # 格式3：字符串格式（可能是JSON字符串）
            elif isinstance(span_info, str):
                import json
                try:
                    parsed_info = json.loads(span_info)
                    return self._extract_service_and_span_name(parsed_info)  # 递归处理
                except json.JSONDecodeError:
                    pass

            print(f"警告：无法解析span_info格式: {type(span_info)}")
            return '', ''

        except Exception as e:
            print(f"提取serviceName和spanName时发生错误: {e}")
            return '', ''

    def _parse_array_field(self, field_value: str) -> list:
        """
        解析数组字段，支持JSON格式的数组

        Args:
            field_value: 字段值，可能是JSON数组格式的字符串

        Returns:
            解析后的列表
        """
        import json

        if not field_value or field_value == '[]':
            return []

        try:
            # 尝试直接解析JSON
            return json.loads(field_value)
        except json.JSONDecodeError:
            try:
                # 如果直接解析失败，尝试处理可能的格式问题
                cleaned_value = field_value.strip()
                if cleaned_value.startswith('[') and cleaned_value.endswith(']'):
                    return json.loads(cleaned_value)
                else:
                    # 如果不是数组格式，尝试按逗号分割
                    return [item.strip().strip('"\'') for item in cleaned_value.split(',') if item.strip()]
            except:
                print(f"无法解析数组字段: {field_value}")
                return []

    def get_top_95_percent_spans_query(self) -> tuple[str, str]:
        """
        获取查询前95%独占时间span详细信息的SPL查询语句

        Returns:
            (span_conditions, query): 查询条件和完整的查询语句
        """
        top_spans = self.find_top_95_percent_spans()

        if not top_spans:
            return "", "* | select * from log where false"  # 返回空结果的查询

        # 构建查询条件
        span_conditions = " or ".join([f"spanId='{span_id}'" for span_id in top_spans])
        query = f"* | select * from log where {span_conditions}"

        return span_conditions, query

    def _calculate_span_averages(self):
        """
        计算正常时间段内每个serviceName<sep>spanName组合的平均独占时间
        """

        # 首先获取独占时间数据
        print ("获取独占时间数据...")
        # exclusive_duration_query = get_spl_query(self.duration_threshold, TRACES_FOR_AVG_RT)
        exclusive_duration_query = get_spl_query(0, TRACES_FOR_AVG_RT)

        request = GetLogsRequest(
            project=self.project_name,
            logstore=self.logstore_name,
            query=exclusive_duration_query,
            fromTime=self.normal_start_time,
            toTime=self.normal_end_time,
            # line=TRACES_FOR_AVG_RT # only calculate for 1000 logs
        )

        try:
            response = self.client.get_logs(request)
            logs = [log_item.get_contents() for log_item in response.get_logs()] if response else []
            print(f"正常时间段查询到的独占时间日志条数: {len(logs)}")

            # 收集所有的span_id和对应的独占时间
            span_duration_mapping = {}  # {span_id: exclusive_duration}

            print("开始计算正常时间段的平均独占时间...")

            for log in logs:
                try:
                    span_ids = self._parse_array_field(log.get('span_id', '[]'))
                    exclusive_durations = self._parse_array_field(log.get('exclusive_duration', '[]'))

                    if len(span_ids) != len(exclusive_durations):
                        continue

                    for span_id, duration in zip(span_ids, exclusive_durations):
                        if isinstance(duration, (int, float)) and duration > 0:
                            span_duration_mapping[span_id] = duration

                except Exception as e:
                    print(f"处理独占时间日志数据时发生错误: {e}")
                    continue

            print(f"收集到 {len(span_duration_mapping)} 个span的独占时间信息")

            # 然后查询这些span的spanName信息
            if span_duration_mapping:
                self._query_span_names_for_spans(span_duration_mapping)

        except Exception as e:
            print(f"计算平均值时发生错误: {e}")

    def _query_span_names_for_spans(self, span_duration_mapping: dict):
        """
        查询指定span_id的serviceName和spanName信息
        优化：采样TRACES_FOR_AVG_RT这么多个span来计算平均值
        """
        print("查询span的serviceName和spanName信息...")

        # 采样优化：随机采样 TRACES_FOR_AVG_RT 这么多个span来计算平均值
        span_ids = list(span_duration_mapping.keys())
        if len(span_ids) > TRACES_FOR_AVG_RT:
            # span_ids = random.sample(span_ids, TRACES_FOR_AVG_RT)
            sorted_span_ids = sorted(span_ids, key=lambda x: span_duration_mapping[x], reverse=True)
            span_ids = sorted_span_ids[:TRACES_FOR_AVG_RT]
            span_duration_mapping = {span_id: span_duration_mapping[span_id] for span_id in span_ids}
            print(f"从原始span中采样了 {len(span_ids)} 个用于计算平均值")
        else:
            print(f"span数量({len(span_ids)})不超过限制，使用所有span计算平均值")

        # 构建查询条件，分批查询但增大批次大小
        batch_size = 500  # 增大批次大小以提高性能，但避免查询条件过长
        service_durations = {}  # {serviceName<sep>spanName: [duration1, duration2, ...]}

        for i in range(0, len(span_ids), batch_size):
            batch_span_ids = span_ids[i:i+batch_size]
            span_conditions = " or ".join([f"spanId='{span_id}'" for span_id in batch_span_ids])

            service_query = f"* | select spanId, serviceName, spanName from log  where {span_conditions}"

            request = GetLogsRequest(
                project=self.project_name,
                logstore=self.logstore_name,
                query=service_query,
                fromTime=self.normal_start_time,
                toTime=self.normal_end_time,
                # line=TRACES_FOR_AVG_RT  # 使用TRACES_FOR_AVG_RT作为查询限制
            )

            try:
                print(f"查询第 {i//batch_size + 1} 批，共 {len(batch_span_ids)} 个span...")
                response = self.client.get_logs(request)
                service_logs = [log_item.get_contents() for log_item in response.get_logs()] if response else []
                print(f"查询到 {len(service_logs)} 条记录")

                # 使用for循环在本地计算
                for log in service_logs:
                    span_id = log.get('spanId')
                    service_name = log.get('serviceName')
                    span_name = log.get('spanName')

                    if span_id in span_duration_mapping and service_name and span_name:
                        duration = span_duration_mapping[span_id]
                        # 构建组合键
                        combined_key = f"{service_name}<sep>{span_name}"
                        if combined_key not in service_durations:
                            service_durations[combined_key] = []
                        service_durations[combined_key].append(duration)

            except Exception as e:
                print(f"查询第 {i//batch_size + 1} 批时发生错误: {e}")
                continue

        # 计算每个serviceName<sep>spanName组合的平均值
        for combined_key, durations in service_durations.items():
            if durations:
                avg_duration = np.mean(durations)
                self.span_average_durations[combined_key] = avg_duration
                print(f"组合键 {combined_key} 的平均独占时间: {avg_duration:.2f}")

        print(f"共计算了 {len(self.span_average_durations)} 个serviceName<sep>spanName组合的平均独占时间")

# def test_optimized_versions():
#     """测试自动方案选择机制"""
#     print("="*60)
#     print("测试自动方案选择机制:")
#     print("="*60)

#     print("系统会自动选择最优方案：")
#     print("1. 首先尝试方案1：直接从span_list获取serviceName和spanName")
#     print("2. 计算方案1的覆盖率（成功映射的span比例）")
#     print("3. 如果覆盖率 > 50%，使用方案1（性能最优）")
#     print("4. 如果覆盖率 ≤ 50%，自动fallback到方案2（采样查询）")
#     print("5. 如果方案1完全失败，直接使用方案2")
#     print("\n" + "-" * 50)

#     finder = FindRootCauseSpansRT(
#         project_name="proj-xtrace-ee483ec157740929c4cb92d4ff85f-cn-qingdao",
#         logstore_name="logstore-tracing",
#         region="cn-qingdao",
#         start_time="2025-06-29 18:36:01",
#         end_time="2025-06-29 18:41:01",
#         duration_threshold=1000000,  # 1秒
#         limit_num=1000,
#         normal_start_time="2025-06-29 18:25:31",
#         normal_end_time="2025-06-29 18:35:01",
#         minus_average=True  # 启用减去平均值的功能
#     )

#     print("\n开始查找前95%独占时间的span...")
#     print("系统将自动选择最优方案:")
#     top_spans = finder.find_top_95_percent_spans()
#     print(f"\n结果：找到 {len(top_spans)} 个占前95%独占时间的span")

#     print("\n" + "="*60)
#     print("方案选择机制说明:")
#     print("✅ 方案1成功 → 直接使用span_list，无额外查询，性能最优")
#     print("⚠️  方案1部分成功 → 根据覆盖率决定是否fallback")
#     print("❌ 方案1失败 → 自动使用方案2，采样查询，保证功能")
#     print("="*60)


# def test_find_root_cause_spans_rt():
#     """测试函数"""
#     finder = FindRootCauseSpansRT(
#         project_name="proj-xtrace-ee483ec157740929c4cb92d4ff85f-cn-qingdao",
#         logstore_name="logstore-tracing",
#         region="cn-qingdao",
#         start_time="2025-06-29 18:36:01",
#         end_time="2025-06-29 18:41:01",
#         duration_threshold=1000000,  # 1秒
#         limit_num=1000,
#         normal_start_time="2025-06-29 18:25:31",
#         normal_end_time="2025-06-29 18:35:01",
#         minus_average=True  # 启用减去平均值的功能
#     )

#     # 获取前95%的span_id
#     top_spans = finder.find_top_95_percent_spans()
#     print(f"前95%独占时间的span_id: {top_spans}")

#     # 获取查询这些span详细信息的查询语句
#     span_conditions, query = finder.get_top_95_percent_spans_query()
#     print(f"查询条件: {span_conditions}")

#     # 打印服务平均值信息
#     print("\n各serviceName<sep>spanName组合的平均独占时间:")
#     for combined_key, avg_duration in finder.span_average_durations.items():
#         print(f"  {combined_key}: {avg_duration:.2f}")


# def test_without_minus_average():
#     """测试不减去平均值的情况"""
#     print("="*50)
#     print("测试不减去平均值的情况:")
#     print("="*50)

#     finder = FindRootCauseSpansRT(
#         project_name="proj-xtrace-ee483ec157740929c4cb92d4ff85f-cn-qingdao",
#         logstore_name="logstore-tracing",
#         region="cn-qingdao",
#         start_time="2025-06-29 18:36:01",
#         end_time="2025-06-29 18:41:01",
#         duration_threshold=1000000,  # 1秒
#         limit_num=1000,
#         minus_average=False  # 不减去平均值
#     )

#     top_spans = finder.find_top_95_percent_spans()
#     print(f"前95%独占时间的span_id (不减平均值): {top_spans}")


# def test_only_top1_per_trace():
#     """测试只处理每个trace中独占时间排top-1的span"""
#     print("="*60)
#     print("测试只处理每个trace中独占时间排top-1的span:")
#     print("="*60)

#     print("开关说明:")
#     print("• only_top1_per_trace=False: 处理每个trace中的所有span（默认行为）")
#     print("• only_top1_per_trace=True:  只处理每个trace中独占时间最长的span")
#     print("• 这可以显著减少计算量，聚焦于每个trace的最耗时操作")
#     print()

#     # 测试开关关闭的情况
#     print("-" * 40)
#     print("1. 开关关闭 (处理所有span):")
#     print("-" * 40)

#     finder_all = FindRootCauseSpansRT(
#         project_name="proj-xtrace-ee483ec157740929c4cb92d4ff85f-cn-qingdao",
#         logstore_name="logstore-tracing",
#         region="cn-qingdao",
#         start_time="2025-06-29 18:36:01",
#         end_time="2025-06-29 18:41:01",
#         duration_threshold=1000000,  # 1秒
#         limit_num=1000,
#         only_top1_per_trace=False  # 处理所有span
#     )

#     top_spans_all = finder_all.find_top_95_percent_spans()
#     print(f"处理所有span时找到的前95%span数量: {len(top_spans_all)}")

#     # 测试开关打开的情况
#     print()
#     print("-" * 40)
#     print("2. 开关打开 (每个trace只取top-1):")
#     print("-" * 40)

#     finder_top1 = FindRootCauseSpansRT(
#         project_name="proj-xtrace-ee483ec157740929c4cb92d4ff85f-cn-qingdao",
#         logstore_name="logstore-tracing",
#         region="cn-qingdao",
#         start_time="2025-06-29 18:36:01",
#         end_time="2025-06-29 18:41:01",
#         duration_threshold=1000000,  # 1秒
#         limit_num=1000,
#         only_top1_per_trace=True   # 只处理每个trace中top-1的span
#     )

#     top_spans_top1 = finder_top1.find_top_95_percent_spans()
#     print(f"只处理top-1 span时找到的前95%span数量: {len(top_spans_top1)}")

#     print()
#     print("=" * 60)
#     print("对比结果:")
#     print(f"• 处理所有span: {len(top_spans_all)} 个span")
#     print(f"• 只处理top-1: {len(top_spans_top1)} 个span")
#     if len(top_spans_all) > 0:
#         reduction_rate = (len(top_spans_all) - len(top_spans_top1)) / len(top_spans_all) * 100
#         print(f"• 计算量减少: {reduction_rate:.1f}%")
#     print("=" * 60)


# def test_top1_with_minus_average_logic():
#     """测试only_top1_per_trace=True且minus_average=True时的逻辑正确性"""
#     print("="*70)
#     print("测试 only_top1_per_trace + minus_average 的逻辑正确性:")
#     print("="*70)

#     print("逻辑修复说明:")
#     print("• 修复前：先找原始时间最长的span，再减去平均值")
#     print("• 修复后：先对所有span减去平均值，再找调整后时间最长的span")
#     print("• 这确保了选择的是相对于正常情况异常增加最多的span")
#     print()

#     print("场景举例:")
#     print("假设某个trace有3个span:")
#     print("  spanA: 原始时间=10s, 平均值=2s  → 调整后=8s")
#     print("  spanB: 原始时间=12s, 平均值=10s → 调整后=2s")
#     print("  spanC: 原始时间=8s,  平均值=1s  → 调整后=7s")
#     print()
#     print("修复前逻辑: 会选择spanB (原始时间最长=12s)")
#     print("修复后逻辑: 会选择spanA (调整后时间最长=8s) ✅")
#     print("修复后的选择更合理，因为spanA相对正常情况异常增加最多")
#     print()

#     # 实际测试
#     print("-" * 50)
#     print("实际测试：")
#     print("-" * 50)

#     finder = FindRootCauseSpansRT(
#         project_name="proj-xtrace-ee483ec157740929c4cb92d4ff85f-cn-qingdao",
#         logstore_name="logstore-tracing",
#         region="cn-qingdao",
#         start_time="2025-06-29 18:36:01",
#         end_time="2025-06-29 18:41:01",
#         duration_threshold=1000000,  # 1秒
#         limit_num=1000,
#         normal_start_time="2025-06-29 18:25:31",
#         normal_end_time="2025-06-29 18:35:01",
#         minus_average=True,           # 启用减去平均值
#         only_top1_per_trace=True      # 只处理每个trace的top-1
#     )

#     print("开始测试修复后的逻辑...")
#     top_spans = finder.find_top_95_percent_spans()
#     print(f"结果：找到 {len(top_spans)} 个span")
#     print("✅ 这些span是基于调整后独占时间（减去平均值后）选择的top-1")

#     print()
#     print("=" * 70)
#     print("逻辑修复完成！现在系统会正确选择相对异常增加最多的span")
#     print("=" * 70)


if __name__ == "__main__":
    pass
    # test_find_root_cause_spans_rt()
    # test_without_minus_average()
    # test_only_top1_per_trace()
    # test_top1_with_minus_average_logic()
    # test_optimized_versions()
