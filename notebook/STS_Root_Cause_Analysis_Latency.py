#!/usr/bin/env python
# coding: utf-8

"""
Latency Root Cause Analysis Module

This module implements comprehensive latency root cause analysis.
It provides a function to analyze latency issues and return root cause candidates.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta

# 将父目录添加到路径以便导入模块
sys.path.append('..')

def analyze_services_for_anomalies(services_to_check, cpu_candidates, memory_candidates, candidate_root_causes,
                                   cms_tester, from_time, to_time, analysis_type="分析"):
    """
    分析给定服务列表的CPU和内存异常

    Args:
        services_to_check: 要检查的服务列表 [(service_name, count), ...]
        cpu_candidates: CPU候选服务集合
        memory_candidates: 内存候选服务集合
        candidate_root_causes: 原始候选根因列表
        cms_tester: CMS测试客户端
        from_time, to_time: 时间范围
        analysis_type: 分析类型标识

    Returns:
        list: 根因候选数组或空列表
    """
    if not services_to_check:
        return []

    print(f"\n📊 开始{analysis_type}服务异常检测...")

    # 步骤1：收集所有服务的CPU异常信息
    print(f"🔍 收集所有服务的CPU异常信息...")
    cpu_anomalies = []

    for service_name, pattern_count in services_to_check:
        print(f"  检查 {service_name} 的CPU异常...")

        try:
            cpu_anomaly_detection_query = f"""
.entity_set with(domain='k8s', name='k8s.deployment', query=`name='{service_name}'` )
| entity-call get_metric('k8s', 'k8s.metric.high_level_metric_deployment', 'deployment_cpu_usage_total', 'range', '1m')
| extend ret = series_decompose_anomalies(__value__, '{{"confidence": 0.035}}')
| extend anomalies_score_series = ret.anomalies_score_series, anomalies_type_series = ret.anomalies_type_series, error_msg = ret.error_msg
"""

            cpu_anomaly_result = cms_tester._execute_spl_query(
                cpu_anomaly_detection_query.strip(),
                from_time=from_time,
                to_time=to_time
            )

            if cpu_anomaly_result and cpu_anomaly_result.data:
                cpu_anomaly_count = 0
                for record in cpu_anomaly_result.data:
                    if isinstance(record, (list, tuple)):
                        for item in record:
                            if isinstance(item, str):
                                exceed_upper_count = item.count('ExceedUpperBound')
                                cpu_anomaly_count += exceed_upper_count

                cpu_candidate = f"{service_name}.cpu"
                is_candidate_match = cpu_candidate in candidate_root_causes
                if not is_candidate_match:
                    is_candidate_match = service_name in cpu_candidates
                cpu_anomalies.append((service_name, cpu_anomaly_count, is_candidate_match, cpu_candidate))

                status = "✅ 匹配" if is_candidate_match else "❌ 不匹配"
                print(f"    {service_name}: {cpu_anomaly_count} 个CPU异常点 ({status})")
            else:
                print(f"    {service_name}: 无CPU数据")

        except Exception as e:
            print(f"    {service_name}: CPU查询失败 - {e}")

    # 步骤2：收集所有服务的内存异常信息
    print(f"🔍 收集所有服务的内存异常信息...")
    memory_anomalies = []

    for service_name, pattern_count in services_to_check:
        print(f"  检查 {service_name} 的内存异常...")

        try:
            memory_anomaly_query = f"""
.entity_set with(domain='k8s', name='k8s.deployment', query=`name='{service_name}'` )
| entity-call get_metric('k8s', 'k8s.metric.high_level_metric_deployment', 'deployment_memory_usage_total', 'range', '1m')
| extend ret = series_decompose_anomalies(__value__, '{{"confidence": 0.035}}')
| extend anomalies_score_series = ret.anomalies_score_series, anomalies_type_series = ret.anomalies_type_series, error_msg = ret.error_msg
"""

            memory_result = cms_tester._execute_spl_query(
                memory_anomaly_query.strip(),
                from_time=from_time,
                to_time=to_time
            )

            if memory_result and memory_result.data:
                memory_anomaly_count = 0
                for record in memory_result.data:
                    if isinstance(record, (list, tuple)):
                        for item in record:
                            if isinstance(item, str):
                                exceed_upper_count = item.count('ExceedUpperBound')
                                memory_anomaly_count += exceed_upper_count

                memory_candidate = f"{service_name}.memory"
                is_candidate_match = memory_candidate in candidate_root_causes
                if not is_candidate_match:
                    is_candidate_match = service_name in memory_candidates
                memory_anomalies.append((service_name, memory_anomaly_count, is_candidate_match, memory_candidate))

                status = "✅ 匹配" if is_candidate_match else "❌ 不匹配"
                print(f"    {service_name}: {memory_anomaly_count} 个内存异常点 ({status})")
            else:
                print(f"    {service_name}: 无内存数据")

        except Exception as e:
            print(f"    {service_name}: 内存查询失败 - {e}")

    # 步骤3：分析CPU异常，找到最高的匹配候选
    print(f"📊 分析CPU异常结果...")
    cpu_anomalies.sort(key=lambda x: x[1], reverse=True)

    print(f"CPU异常排序结果:")
    for service_name, anomaly_count, is_match, candidate in cpu_anomalies:
        status = "🎯 匹配候选" if is_match else "❌ 非候选"
        print(f"  {service_name}: {anomaly_count} 点 ({status})")

    for service_name, anomaly_count, is_match, candidate in cpu_anomalies:
        if anomaly_count >= 3 and is_match:
            print(f"🚨 找到CPU异常根因: {candidate} (异常点: {anomaly_count})")
            return [candidate]

    # 步骤4：分析内存异常，找到最高的匹配候选
    print(f"📊 分析内存异常结果...")
    memory_anomalies.sort(key=lambda x: x[1], reverse=True)

    print(f"内存异常排序结果:")
    for service_name, anomaly_count, is_match, candidate in memory_anomalies:
        status = "🎯 匹配候选" if is_match else "❌ 非候选"
        print(f"  {service_name}: {anomaly_count} 点 ({status})")

    for service_name, anomaly_count, is_match, candidate in memory_anomalies:
        if anomaly_count >= 3 and is_match:
            print(f"🚨 找到内存异常根因: {candidate} (异常点: {anomaly_count})")
            return [candidate]

    # 步骤5：没有找到符合条件的异常
    print(f"⚠️ {analysis_type}未找到符合条件的根因")
    return []


def analyze_latency_root_cause(anomaly_start_time: str, anomaly_end_time: str, candidate_root_causes: list):
    """
    Analyze latency root cause using comprehensive pattern and anomaly detection

    Args:
        anomaly_start_time: Start time of anomaly period (format: "YYYY-MM-DD HH:MM:SS")
        anomaly_end_time: End time of anomaly period (format: "YYYY-MM-DD HH:MM:SS")
        candidate_root_causes: List of candidate root causes to filter to

    Returns:
        list: Root cause candidate array (e.g., ["service.cpu"] or [])
    """

    # 导入自定义模块并进行异常处理
    try:
        from find_root_cause_spans_rt import FindRootCauseSpansRT
        from test_cms_query import TestCMSQuery
        print("✅ 成功导入自定义模块")
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保相关模块文件存在于当前目录")
        return []

    print("🚀 开始延迟根因分析...")
    print("="*60)

    print("🎯 Will analyze all services found in pattern analysis")

    # 配置信息 - 保持原始值不变
    PROJECT_NAME = "proj-xtrace-a46b97cfdc1332238f714864c014a1b-cn-qingdao"
    LOGSTORE_NAME = "logstore-tracing"
    REGION = "cn-qingdao"

    # 分析时间区间 - 使用传入的参数
    ANOMALY_START_TIME = anomaly_start_time
    ANOMALY_END_TIME = anomaly_end_time

    # 计算正常时间段（异常开始前1小时到异常开始）
    anomaly_start = datetime.strptime(ANOMALY_START_TIME, "%Y-%m-%d %H:%M:%S")
    normal_start = anomaly_start - timedelta(hours=1)
    normal_end = anomaly_start

    NORMAL_START_TIME = normal_start.strftime("%Y-%m-%d %H:%M:%S")
    NORMAL_END_TIME = normal_end.strftime("%Y-%m-%d %H:%M:%S")

    # CMS配置 - 保持原始值不变
    CMS_WORKSPACE = "quanxi-tianchi-test"
    CMS_ENDPOINT = 'metrics.cn-qingdao.aliyuncs.com'

    print(f"  SLS Project: {PROJECT_NAME}")
    print(f"  Logstore: {LOGSTORE_NAME}")
    print(f"  Region: {REGION}")
    print(f"  异常时间段: {ANOMALY_START_TIME} ~ {ANOMALY_END_TIME}")
    print(f"  正常时间段: {NORMAL_START_TIME} ~ {NORMAL_END_TIME}")
    print(f"  CMS Workspace: {CMS_WORKSPACE}")

    # 获取STS临时凭证
    def get_sts_credentials():
        """获取STS临时凭证"""
        try:
            from aliyunsdkcore.client import AcsClient
            from aliyunsdksts.request.v20150401 import AssumeRoleRequest

            # 获取环境变量中的主账号凭证
            MAIN_ACCOUNT_ACCESS_KEY_ID = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
            MAIN_ACCOUNT_ACCESS_KEY_SECRET = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
            STS_ROLE_ARN = "acs:ram::1672753017899339:role/tianchi-user-a"
            STS_SESSION_NAME = "my-sls-access"

            if not MAIN_ACCOUNT_ACCESS_KEY_ID or not MAIN_ACCOUNT_ACCESS_KEY_SECRET:
                print("❌ 环境变量未设置: ALIBABA_CLOUD_ACCESS_KEY_ID 或 ALIBABA_CLOUD_ACCESS_KEY_SECRET")
                return None, None, None

            client = AcsClient(
                MAIN_ACCOUNT_ACCESS_KEY_ID,
                MAIN_ACCOUNT_ACCESS_KEY_SECRET,
                REGION
            )

            request = AssumeRoleRequest.AssumeRoleRequest()
            request.set_RoleArn(STS_ROLE_ARN)
            request.set_RoleSessionName(STS_SESSION_NAME)
            request.set_DurationSeconds(3600)

            response = client.do_action_with_exception(request)
            response_data = json.loads(response)

            credentials = response_data['Credentials']
            return (
                credentials['AccessKeyId'],
                credentials['AccessKeySecret'],
                credentials['SecurityToken']
            )

        except Exception as e:
            print(f"❌ 获取STS临时凭证失败: {e}")
            return None, None, None

    # 获取临时凭证
    temp_access_key_id, temp_access_key_secret, security_token = get_sts_credentials()

    if not temp_access_key_id:
        print("❌ 无法获取STS临时凭证，退出分析")
        return []

    # 创建SLS客户端
    try:
        from aliyun.log import LogClient

        sls_endpoint = f"{REGION}.log.aliyuncs.com"
        log_client = LogClient(
            sls_endpoint,
            temp_access_key_id,
            temp_access_key_secret,
            security_token
        )
        print("✅ SLS客户端创建成功")

    except Exception as e:
        print(f"❌ 创建SLS客户端失败: {e}")
        return []

    # 创建FindRootCauseSpansRT实例
    try:
        DURATION_THRESHOLD = 2000000000  # 2000ms（以纳秒为单位）

        finder = FindRootCauseSpansRT(
            client=log_client,
            project_name=PROJECT_NAME,
            logstore_name=LOGSTORE_NAME,
            region=REGION,
            start_time=ANOMALY_START_TIME,
            end_time=ANOMALY_END_TIME,
            duration_threshold=DURATION_THRESHOLD,
            limit_num=1000,
            normal_start_time=NORMAL_START_TIME,
            normal_end_time=NORMAL_END_TIME,
            minus_average=True,
            only_top1_per_trace=False
        )
        print("✅ FindRootCauseSpansRT实例创建成功")

    except Exception as e:
        print(f"❌ 创建FindRootCauseSpansRT实例失败: {e}")
        return []

    # 步骤1：查找高独占时间的span
    print("\n🔍 Step 1: Finding high exclusive time spans...")
    print("="*60)

    try:
        top_95_percent_spans = finder.find_top_95_percent_spans()

        if top_95_percent_spans:
            print(f"✅ 找到 {len(top_95_percent_spans)} 个高独占时间的span")
            print(f"📝 已生成用于进一步分析的查询条件")
        else:
            print("⚠️ 未找到高独占时间的span")
            return []

    except Exception as e:
        print(f"❌ 查找高独占时间span失败: {e}")
        return []

    # 步骤2：使用 diff_patterns 进行模式分析
    print("\n🔍 Step 2: Pattern analysis with diff_patterns...")
    print("="*60)

    if top_95_percent_spans:
        # 首先将全部高独占时间的span_id拼接成一个字符串，用于diff_patterns查询条件
        span_conditions_for_patterns = " or ".join([f"spanId='{span_id}'" for span_id in top_95_percent_spans[:2000]])  # Limit for query size

        param_str = """{"minimum_support_fraction": 0.03}"""
        # 核心为 diff_patterns 算法调用，进行模式差异分析
        diff_patterns_query = f"""
duration > {DURATION_THRESHOLD} | set session enable_remote_functions=true; set session velox_support_row_constructor_enabled=true;
with t0 as (
    select spanName, serviceName, cast(duration as double) as duration,
           JSON_EXTRACT_SCALAR(resources, '$["k8s.pod.ip"]') AS pod_ip,
           JSON_EXTRACT_SCALAR(resources, '$["k8s.node.name"]') AS node_name,
           JSON_EXTRACT_SCALAR(resources, '$["service.version"]') AS service_version,
           if(({span_conditions_for_patterns}), 'true', 'false') as anomaly_label,
           cast(if((statusCode = 2 or statusCode = 3), 1, 0) as double) as error_count
    from log
),
t1 as (
    select array_agg(spanName) as spanName,
           array_agg(serviceName) as serviceName,
           array_agg(duration) as duration,
           array_agg(pod_ip) as pod_ip,
           array_agg(node_name) as node_name,
           array_agg(service_version) as service_version,
           array_agg(anomaly_label) as anomaly_label,
           array_agg(error_count) as error_count
    from t0
),
t2 as (
    select row(spanName, serviceName, anomaly_label) as table_row
    from t1
),
t3 as (
    select diff_patterns(table_row, ARRAY['spanName', 'serviceName', 'anomaly_label'], 'anomaly_label', 'true', 'false', '', '', '{param_str}') as ret
    from t2
)
select * from t3
"""

        print("📋 已生成用于模式分析的 diff_patterns 查询语句")
        print("\n🚀 正在使用SLS客户端执行 diff_patterns 查询...")

        try:
            # 创建用于 diff_patterns 查询的 GetLogsRequest
            from aliyun.log import GetLogsRequest

            request = GetLogsRequest(
                project=PROJECT_NAME,
                logstore=LOGSTORE_NAME,
                query=diff_patterns_query.strip(),
                fromTime=int(time.mktime(datetime.strptime(ANOMALY_START_TIME, "%Y-%m-%d %H:%M:%S").timetuple())),
                toTime=int(time.mktime(datetime.strptime(ANOMALY_END_TIME, "%Y-%m-%d %H:%M:%S").timetuple())),
                line=100  # Limit results
            )

            # 使用finder的SLS客户端执行查询
            patterns_result = log_client.get_logs(request)

            if patterns_result and patterns_result.get_logs():
                logs = [log_item.get_contents() for log_item in patterns_result.get_logs()]
                print(f"✅ Pattern analysis completed: {len(logs)} results")

                # 在结果中会展示同一service在正常样本和异常样本中出现的次数对比
                print("\n📊 模式分析结果:")
                for i, log_entry in enumerate(logs[:3]):  # Show first 3 results
                    print(f"  结果 {i+1}: {log_entry}")

                # Extract service patterns from diff_patterns results
                service_patterns = {}
                span_patterns = []  # Store span patterns for service inference

                for log_entry in logs:
                    if hasattr(log_entry, 'get_contents'):
                        contents = log_entry.get_contents()
                    else:
                        contents = log_entry
                    print(f"  🔍 Analyzing pattern result: {contents}")

                    # Parse structured result from diff_patterns query
                    if 'ret' in contents:
                        ret_value = contents['ret']

                        if isinstance(ret_value, str):
                            try:
                                # Replace 'null' with 'None' for Python parsing
                                data_str = ret_value.replace('null', 'None')
                                result = eval(data_str)

                                if len(result) >= 2 and isinstance(result[0], list) and isinstance(result[1], list):
                                    patterns = result[0]
                                    counts = result[1]

                                    print(f"    📊 Found {len(patterns)} patterns with counts")

                                    for i, pattern in enumerate(patterns):
                                        if i < len(counts):
                                            count = counts[i]

                                            # Parse serviceName patterns from diff_patterns results
                                            if 'serviceName' in pattern and '=' in pattern:
                                                # Handle complex patterns like "serviceName"='cart' AND "spanName"='POST'
                                                import re
                                                match = re.search(r'"serviceName"=\'([^\']+)\'', pattern)
                                                if match:
                                                    service_part = match.group(1)
                                                    service_patterns[service_part] = service_patterns.get(service_part, 0) + count
                                                    print(f"    ✅ Found serviceName pattern: '{service_part}' (count: {count})")
                                                else:
                                                    print(f"    ⚠️ Could not parse serviceName from: '{pattern}'")

                                            # Log spanName patterns for service inference
                                            elif 'spanName' in pattern:
                                                span_name = pattern.split('=')[1].strip('\'"') if '=' in pattern else pattern
                                                print(f"    ℹ️ Found spanName pattern: '{span_name}' (count: {count})")
                                                span_patterns.append((span_name, count))

                            except Exception as e:
                                print(f"    ⚠️ Error parsing ret field: {e}")
                                import traceback
                                traceback.print_exc()

                # If no serviceName patterns found, try to infer from spanName patterns
                if not service_patterns and span_patterns:
                    print(f"    📊 No serviceName patterns found - attempting service inference from spanName patterns")

                    service_candidates = {}
                    for span_name, count in span_patterns:
                        # Map spanName patterns to likely services
                        if 'CartService' in span_name or 'cart' in span_name.lower():
                            service_candidates['cart'] = service_candidates.get('cart', 0) + count
                        elif 'ProductCatalogService' in span_name or 'product' in span_name.lower():
                            service_candidates['product-catalog'] = service_candidates.get('product-catalog', 0) + count
                        elif 'PaymentService' in span_name or 'payment' in span_name.lower():
                            service_candidates['payment'] = service_candidates.get('payment', 0) + count
                        elif 'CheckoutService' in span_name or 'checkout' in span_name.lower():
                            service_candidates['checkout'] = service_candidates.get('checkout', 0) + count
                        elif 'RecommendationService' in span_name or 'recommendation' in span_name.lower():
                            service_candidates['recommendation'] = service_candidates.get('recommendation', 0) + count
                        elif 'CurrencyService' in span_name or 'currency' in span_name.lower():
                            service_candidates['currency'] = service_candidates.get('currency', 0) + count
                        elif 'frontend' in span_name.lower():
                            service_candidates['frontend'] = service_candidates.get('frontend', 0) + count
                        elif 'flagservice' in span_name.lower() or 'ad' in span_name.lower():
                            service_candidates['ad'] = service_candidates.get('ad', 0) + count
                        else:
                            print(f"    ❓ Cannot infer service from span: '{span_name}'")

                    if service_candidates:
                        print(f"    📊 Service inference from spans: {dict(service_candidates)}")
                        service_patterns = service_candidates
                    else:
                        print("    ❌ Cannot determine target service from available span patterns")

                if service_patterns:
                    print(f"\n🎯 识别出的service模式:")
                    for service, count in sorted(service_patterns.items(), key=lambda x: x[1], reverse=True):
                        print(f"  - {service}: {count} 次模式匹配")

                    # Sort all services by frequency (descending) for comprehensive analysis
                    all_service_matches = [(service, count) for service, count in service_patterns.items()]
                    all_service_matches.sort(key=lambda x: x[1], reverse=True)

                    print(f"\n💡 所有服务按模式频率排序:")
                    for service, count in all_service_matches:
                        print(f"  - {service}: {count} 次模式匹配")

                    # Store all service matches for analysis
                    globals()['CANDIDATE_SERVICES_BY_FREQUENCY'] = all_service_matches

            else:
                print("⚠️ 未返回任何模式分析结果")

        except Exception as e:
            print(f"❌ 执行 diff_patterns 查询时出错: {e}")
            print("💡 建议在SLS控制台手动执行")

    else:
        print("⚠️ 无法进行模式分析 - 未找到高独占时间的span")

    # 步骤3：首先尝试模式分析服务，失败则回退到所有候选
    print("\n📊 Step 3: 分析策略 - 优先模式服务，失败则回退全部候选")

    # 准备候选根因的CPU和内存分类
    cpu_candidates = set()
    memory_candidates = set()

    for candidate in candidate_root_causes:
        if '.' in candidate:
            if candidate.endswith('.cpu'):
                service_name = candidate.split('.')[0]
                cpu_candidates.add(service_name)
            elif candidate.endswith('.memory'):
                service_name = candidate.split('.')[0]
                memory_candidates.add(service_name)

    # Store candidates for validation
    globals()['CPU_CANDIDATES'] = cpu_candidates
    globals()['MEMORY_CANDIDATES'] = memory_candidates

    print(f"🎯 输入候选服务:")
    print(f"   CPU: {list(cpu_candidates)}")
    print(f"   内存: {list(memory_candidates)}")

    # Create CMS client for the analysis
    try:
        cms_tester = TestCMSQuery()
        cms_tester.setUp()
        print(f"✅ CMS客户端创建成功")

    except Exception as e:
        print(f"❌ 创建CMS客户端失败: {e}")
        return []

    # Convert time strings to timestamps needed for CMS queries
    from_time = int(time.mktime(datetime.strptime(NORMAL_START_TIME, "%Y-%m-%d %H:%M:%S").timetuple()))
    to_time = int(time.mktime(datetime.strptime(ANOMALY_END_TIME, "%Y-%m-%d %H:%M:%S").timetuple()))

    # 步骤3.1：首先尝试模式分析发现的服务
    pattern_result = None
    if 'CANDIDATE_SERVICES_BY_FREQUENCY' in globals():
        pattern_services = globals()['CANDIDATE_SERVICES_BY_FREQUENCY']
        print(f"✅ 优先分析模式发现的服务: {[s[0] for s in pattern_services]}")

        pattern_result = analyze_services_for_anomalies(
            pattern_services, cpu_candidates, memory_candidates, candidate_root_causes,
            cms_tester, from_time, to_time, "模式分析"
        )

        if pattern_result:
            return pattern_result

    # 步骤3.2：模式分析失败或无结果，回退到所有输入候选
    print("\n🔄 模式分析未找到根因，回退到分析所有输入候选...")
    all_input_services = cpu_candidates.union(memory_candidates)
    if all_input_services:
        fallback_services = [(service, 1) for service in all_input_services]
        print(f"🚀 回退分析服务: {[s[0] for s in fallback_services]}")

        fallback_result = analyze_services_for_anomalies(
            fallback_services, cpu_candidates, memory_candidates, candidate_root_causes,
            cms_tester, from_time, to_time, "回退分析"
        )

        return fallback_result if fallback_result else []
    else:
        print("❌ 输入候选中没有有效的服务格式")
        return []
