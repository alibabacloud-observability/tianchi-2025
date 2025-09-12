# 我们用python处理得到root cause的span

import os

from aliyun.log import LogClient, GetLogsRequest
from utils.constants import ERROR_TRACES  # 为一个超参数,表示查询的错误trace数量上限
from datetime import datetime

# 加载.env文件

class FindRootCauseSpans:
    # 核心修改点：__init__ 的参数
    def __init__(self, client, project_name: str, logstore_name: str, region: str, start_time: str, end_time: str):
        """
        构造函数。
        :param client: 一个已经配置好凭证的 aliyun.log.LogClient 实例
        :param project_name: 日志项目名称
        :param logstore_name: 日志库名称
        :param start_time: 查询开始时间 (字符串或Unix时间戳)
        :param end_time: 查询结束时间 (字符串或Unix时间戳)
        """
        # 1. 直接使用传入的 client
        self.client = client

        # 2. 保留其他属性的初始化
        self.project_name = project_name
        self.logstore_name = logstore_name
        self.region = region

        # 3. 时间转换逻辑保持不变
        if isinstance(start_time, str):
            self.start_time = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            self.start_time = int(start_time)

        if isinstance(end_time, str):
            self.end_time = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp())
        else:
            self.end_time = int(end_time)

        print(f"[FindRootCauseSpans] 初始化完成。")
        print(f"  开始时间: {datetime.fromtimestamp(self.start_time)} (时间戳: {self.start_time})")
        print(f"  结束时间: {datetime.fromtimestamp(self.end_time)} (时间戳: {self.end_time})")


    def root_cause_spans_query(self):
        # 生成最终的查询语句，将查询得到的spanID拼接成一个SLS的SQL查询语句,用于精确地查看这几个rootSpan的详细日志信息
        all_root_cause_span_ids = self.find_root_cause_spans()
        if not all_root_cause_span_ids:
            return "" # 如果没有找到，返回空字符串或特定查询
        all_root_cause_span_ids_query = "* | select * from log where " + " or ".join([f"spanId='{span_id}'" for span_id in all_root_cause_span_ids])
        return all_root_cause_span_ids_query


    def find_root_cause_spans(self):
        # 首先构建一个非常宽泛的查询 statusCode>1，在指定时间范围内，把所有状态码大于1（通常表示有错误）的Span都捞出来。ERROR_TRACES 限制了返回的日志条数为2000。
        all_spans_query = "statusCode>1"

        request = GetLogsRequest(
            project=self.project_name,
            logstore=self.logstore_name,
            query=all_spans_query,
            fromTime=self.start_time,
            toTime=self.end_time,
            line=ERROR_TRACES
        )

        print (self.project_name, self.logstore_name, self.region)
        print (f"[FindRootCauseSpans] 查询时间范围: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')} ~ {datetime.fromtimestamp(self.end_time).strftime('%Y-%m-%d %H:%M:%S')}. 查询条件: {all_spans_query}")
        response = self.client.get_logs(request)
        all_spans = [log_item.get_contents() for log_item in response.get_logs()] if response else []

        print(f"总共查询到的span数量: {len(all_spans)}")

        # 按traceId分组
        trace_groups = {}
        for span in all_spans:
            trace_id = span.get('traceId')
            if trace_id:
                if trace_id not in trace_groups:
                    trace_groups[trace_id] = []
                trace_groups[trace_id].append(span)

        print(f"涉及的trace数量: {len(trace_groups)}")

        # 对每个trace组进行处理
        all_root_cause_span_ids = []
        for trace_id, trace_logs in trace_groups.items():
            root_cause_span_ids = self.process_one_trace_log(trace_logs)
            # print(f"trace_id: {trace_id}, root_cause_span_ids: {root_cause_span_ids}")
            all_root_cause_span_ids.extend(root_cause_span_ids)

        return all_root_cause_span_ids

    def process_one_trace_log(self, trace_log: list) -> list[str]:
        """ 一个trace_log 由多个span 组成，如果某个span的status > 1而且他的parentSpanId 对应的span 的status > 1，那么这个span的parentSpan 就不是root span. root span 是 没有一个child span的status > 1的span.
        这个函数是找出来所有的root span，然后返回一个list，list 里面包含每个root span 的spanId
        """
        # 存储每个span的状态
        span_status = {}
        # 存储每个span的父span
        parent_spans = {}
        # 存储每个span的子span列表
        child_spans = {}

        # 第一遍遍历,建立映射关系
        for span in trace_log:
            span_id = span["spanId"]
            parent_span_id = span["parentSpanId"]
            status = int(span["statusCode"])

            span_status[span_id] = status

            if parent_span_id:
                parent_spans[span_id] = parent_span_id
                if parent_span_id not in child_spans:
                    child_spans[parent_span_id] = []
                child_spans[parent_span_id].append(span_id)

        # 找出所有status>0的span
        error_spans = []
        for span_id, status in span_status.items():
            if status > 1:
                error_spans.append(span_id)

        # 找出root cause spans
        root_cause_spans = []
        for error_span in error_spans:
            is_root = True
            # 检查所有子span
            if error_span in child_spans:
                for child_span in child_spans[error_span]:
                    if span_status.get(child_span, 0) > 1:
                        is_root = False
                        break
            if is_root:
                root_cause_spans.append(error_span)

        return root_cause_spans


def test(project_name: str, logstore_name: str, region: str, start_time: str, end_time: str):
    # --- 在 test 函数内部，我们需要创建 client ---
    # 这里为了简单，我们还是用AK创建，演示如何调用修改后的类
    # 在你的主脚本里，这里会替换成STS的逻辑

    access_key_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
    access_key_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
    if not access_key_id or not access_key_secret:
        raise ValueError("请设置环境变量 ALIBABA_CLOUD_ACCESS_KEY_ID 和 ALIBABA_CLOUD_ACCESS_KEY_SECRET")

    client = LogClient(endpoint=f"{region}.log.aliyuncs.com", accessKeyId=access_key_id, accessKey=access_key_secret)

    # 将创建好的 client 传进去
    find_root_cause_spans = FindRootCauseSpans(client, project_name, logstore_name, region, start_time, end_time)
    print(find_root_cause_spans.root_cause_spans_query())

if __name__ == "__main__":
    test("proj-xtrace-ee483ec157740929c4cb92d4ff85f-cn-qingdao", "logstore-tracing", "cn-qingdao", "2025-06-14 21:42:43", "2025-06-14 21:47:43")

