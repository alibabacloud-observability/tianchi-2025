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
        print("✅ 成功导入自定义模块")
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        print("请确保相关模块文件存在于当前目录")
        return []

    PROJECT_NAME = "proj-xtrace-a46b97cfdc1332238f714864c014a1b-cn-qingdao"
    LOGSTORE_NAME = "logstore-tracing"
    REGION = "cn-qingdao"
    ANOMALY_START_TIME = anomaly_start_time
    ANOMALY_END_TIME = anomaly_end_time

    anomaly_start = datetime.strptime(ANOMALY_START_TIME, "%Y-%m-%d %H:%M:%S")
    normal_start = anomaly_start - timedelta(hours=1)
    normal_end = anomaly_start
    NORMAL_START_TIME = normal_start.strftime("%Y-%m-%d %H:%M:%S")
    NORMAL_END_TIME = normal_end.strftime("%Y-%m-%d %H:%M:%S")

    def get_sts_credentials():
        try:
            from aliyunsdkcore.client import AcsClient
            from aliyunsdksts.request.v20150401 import AssumeRoleRequest

            MAIN_ACCOUNT_ACCESS_KEY_ID = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
            MAIN_ACCOUNT_ACCESS_KEY_SECRET = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
            ALIBABA_CLOUD_ROLE_ARN = os.getenv('ALIBABA_CLOUD_ROLE_ARN', 'acs:ram::1672753017899339:role/tianchi-user-a')
            STS_SESSION_NAME = os.getenv('ALIBABA_CLOUD_ROLE_SESSION_NAME', 'my-sls-access')

            if not MAIN_ACCOUNT_ACCESS_KEY_ID or not MAIN_ACCOUNT_ACCESS_KEY_SECRET:
                return None, None, None

            client = AcsClient(MAIN_ACCOUNT_ACCESS_KEY_ID, MAIN_ACCOUNT_ACCESS_KEY_SECRET, REGION)
            request = AssumeRoleRequest.AssumeRoleRequest()
            request.set_RoleArn(ALIBABA_CLOUD_ROLE_ARN)
            request.set_RoleSessionName(STS_SESSION_NAME)
            request.set_DurationSeconds(3600)

            response = client.do_action_with_exception(request)
            response_data = json.loads(response)
            credentials = response_data['Credentials']
            return (credentials['AccessKeyId'], credentials['AccessKeySecret'], credentials['SecurityToken'])
        except Exception as e:
            print(f"❌ 获取STS凭证失败: {e}")
            return None, None, None

    temp_access_key_id, temp_access_key_secret, security_token = get_sts_credentials()
    if not temp_access_key_id:
        print("❌ 无法获取STS临时凭证，分析终止")
        return []

    try:
        from aliyun.log import LogClient
        sls_endpoint = f"{REGION}.log.aliyuncs.com"
        log_client = LogClient(sls_endpoint, temp_access_key_id, temp_access_key_secret, security_token)
    except Exception as e:
        print(f"❌ 创建SLS客户端失败: {e}")
        return []

    try:
        DURATION_THRESHOLD = 2000000000
        finder = FindRootCauseSpansRT(
            client=log_client, project_name=PROJECT_NAME, logstore_name=LOGSTORE_NAME,
            region=REGION, start_time=ANOMALY_START_TIME, end_time=ANOMALY_END_TIME,
            duration_threshold=DURATION_THRESHOLD, limit_num=1000,
            normal_start_time=NORMAL_START_TIME, normal_end_time=NORMAL_END_TIME,
            minus_average=True, only_top1_per_trace=False
        )
    except Exception as e:
        print(f"❌ 创建FindRootCauseSpansRT实例失败: {e}")
        return []

    try:
        top_95_percent_spans = finder.find_top_95_percent_spans()
        if not top_95_percent_spans:
            print("❌ 未找到高独占时间的span，无法进行模式分析")
            return []
    except Exception as e:
        print(f"❌ 查找高独占时间span失败: {e}")
        return []

    if top_95_percent_spans:
        span_conditions_for_patterns = " or ".join([f"spanId='{span_id}'" for span_id in top_95_percent_spans[:2000]])
        param_str = """{"minimum_support_fraction": 0.03}"""
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

        try:
            from aliyun.log import GetLogsRequest
            request = GetLogsRequest(
                project=PROJECT_NAME, logstore=LOGSTORE_NAME, query=diff_patterns_query.strip(),
                fromTime=int(time.mktime(datetime.strptime(ANOMALY_START_TIME, "%Y-%m-%d %H:%M:%S").timetuple())),
                toTime=int(time.mktime(datetime.strptime(ANOMALY_END_TIME, "%Y-%m-%d %H:%M:%S").timetuple())),
                line=100
            )

            patterns_result = log_client.get_logs(request)
            if patterns_result and patterns_result.get_logs():
                logs = [log_item.get_contents() for log_item in patterns_result.get_logs()]
                service_patterns = {}
                span_patterns = []

                for log_entry in logs:
                    contents = log_entry if not hasattr(log_entry, 'get_contents') else log_entry.get_contents()
                    if 'ret' in contents and isinstance(contents['ret'], str):
                        try:
                            data_str = contents['ret'].replace('null', 'None')
                            result = eval(data_str)
                            if len(result) >= 2 and isinstance(result[0], list) and isinstance(result[1], list):
                                patterns, counts = result[0], result[1]
                                for i, pattern in enumerate(patterns):
                                    if i < len(counts):
                                        count = counts[i]
                                        if 'serviceName' in pattern and '=' in pattern:
                                            import re
                                            match = re.search(r'"serviceName"=\'([^\']+)\'', pattern)
                                            if match:
                                                service_patterns[match.group(1)] = service_patterns.get(match.group(1), 0) + count
                                        elif 'spanName' in pattern:
                                            span_name = pattern.split('=')[1].strip('\'"') if '=' in pattern else pattern
                                            span_patterns.append((span_name, count))
                        except Exception as e:
                            print(f"⚠️ 解析模式结果失败: {e}")
                            pass

                if not service_patterns and span_patterns:
                    service_candidates = {}
                    for span_name, count in span_patterns:
                        if 'CartService' in span_name or 'cart' in span_name.lower():
                            service_candidates['cart'] = service_candidates.get('cart', 0) + count
                        elif 'ProductCatalogService' in span_name or 'product-catalog' in span_name.lower() or 'product' in span_name.lower():
                            service_candidates['product-catalog'] = service_candidates.get('product-catalog', 0) + count
                        elif 'PaymentService' in span_name or 'payment' in span_name.lower():
                            service_candidates['payment'] = service_candidates.get('payment', 0) + count
                        elif 'CheckoutService' in span_name or 'checkout' in span_name.lower():
                            service_candidates['checkout'] = service_candidates.get('checkout', 0) + count
                        elif 'RecommendationService' in span_name or 'recommendation' in span_name.lower() or 'get_product_list' in span_name:
                            service_candidates['recommendation'] = service_candidates.get('recommendation', 0) + count
                        elif 'CurrencyService' in span_name or 'currency' in span_name.lower() or 'Currency/' in span_name:
                            service_candidates['currency'] = service_candidates.get('currency', 0) + count
                        elif 'flagservice' in span_name.lower() or 'router flagservice egress' in span_name:
                            service_candidates['ad'] = service_candidates.get('ad', 0) + count
                        elif 'InventoryService' in span_name or 'inventory' in span_name.lower():
                            service_candidates['inventory'] = service_candidates.get('inventory', 0) + count
                        elif 'ImageProviderService' in span_name or 'image-provider' in span_name.lower():
                            service_candidates['image-provider'] = service_candidates.get('image-provider', 0) + count
                        elif 'frontend' in span_name.lower() and 'proxy' not in span_name.lower():
                            service_candidates['frontend'] = service_candidates.get('frontend', 0) + count
                        elif 'load-generator' in span_name.lower():
                            service_candidates['load-generator'] = service_candidates.get('load-generator', 0) + count
                    if service_candidates:
                        service_patterns = service_candidates

                for span_name, count in span_patterns:
                    if 'RecommendationService' in span_name or 'get_product_list' in span_name:
                        service_patterns['recommendation'] = service_patterns.get('recommendation', 0) + count
                    elif 'CheckoutService' in span_name:
                        service_patterns['checkout'] = service_patterns.get('checkout', 0) + count
                    elif 'Currency/' in span_name or 'CurrencyService' in span_name:
                        service_patterns['currency'] = service_patterns.get('currency', 0) + count
                    elif 'CartService' in span_name:
                        service_patterns['cart'] = service_patterns.get('cart', 0) + count
                    elif 'router flagservice egress' in span_name:
                        service_patterns['ad'] = service_patterns.get('ad', 0) + count

                if service_patterns:
                    all_service_matches = [(service, count) for service, count in service_patterns.items()]
                    all_service_matches.sort(key=lambda x: x[1], reverse=True)
                    globals()['CANDIDATE_SERVICES_BY_FREQUENCY'] = all_service_matches

        except Exception as e:
            print(f"⚠️ diff_patterns查询执行失败: {e}")
            pass

    if 'CANDIDATE_SERVICES_BY_FREQUENCY' in globals():
        pattern_services = globals()['CANDIDATE_SERVICES_BY_FREQUENCY']
        print(f"🔍 模式分析识别的服务: {[s[0] for s in pattern_services]}")
        for service_name, pattern_count in pattern_services:
            possible_candidates = [
                f"{service_name}.cpu",
                f"{service_name}.memory",
                f"{service_name}.networkLatency",
                f"{service_name}.latency",
                f"{service_name}.Failure",
                f"{service_name}.LargeGc",
                f"{service_name}.Unreachable",
                f"{service_name}.CacheFailure",
                f"{service_name}.FloodHomepage",
                service_name
            ]
            for candidate in possible_candidates:
                if candidate in candidate_root_causes:
                    print(f"✅ 模式匹配成功: {candidate}")
                    return [candidate]
        print("⚠️ 模式分析的服务未匹配到任何候选根因")
    else:
        print("⚠️ 未识别到服务模式，尝试基于候选根因进行匹配")

    for candidate in candidate_root_causes:
        if '.' in candidate and (candidate.endswith('.cpu') or candidate.endswith('.memory')):
            service_name = candidate.split('.')[0]
            possible_candidates = [f"{service_name}.cpu", f"{service_name}.memory", service_name]
            for possible in possible_candidates:
                if possible in candidate_root_causes:
                    print(f"✅ 基于候选根因匹配成功: {possible}")
                    return [possible]
    
    print("❌ 未找到匹配的根因")
    return []
