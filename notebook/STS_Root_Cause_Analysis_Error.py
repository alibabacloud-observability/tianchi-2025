import os
import sys
from datetime import datetime

from aliyun.log import LogClient, GetLogsRequest
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_sts20150401 import models as sts_models
from alibabacloud_tea_openapi import models as open_api_models
from Tea.exceptions import TeaException

sys.path.append('..')

# SLS configuration
PROJECT_NAME = "proj-xtrace-a46b97cfdc1332238f714864c014a1b-cn-qingdao"
LOGSTORE_NAME = "logstore-tracing"
REGION = "cn-qingdao"

# Environment variables
STS_ROLE_ARN = 'acs:ram::1672753017899339:role/tianchi-user-a'
STS_SESSION_NAME = "my-sls-access"

def analyze_error_root_cause(start_time, end_time, candidate_root_causes):
    """
    Analyze error root cause for the given time period

    Args:
        start_time: Start time (YYYY-MM-DD HH:MM:SS)
        end_time: End time (YYYY-MM-DD HH:MM:SS)
        candidate_root_causes: List of candidate root causes to prioritize

    Returns:
        str: Root cause candidate
    """
    from find_root_cause_spans_error import FindRootCauseSpans

    def get_sts_credentials():
        access_key_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        access_key_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

        if not all([access_key_id, access_key_secret, STS_ROLE_ARN]):
            print("❌ 角色ARN缺失! 请在.env文件中配置 ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET")
            return None

        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            endpoint=f'sts.{REGION}.aliyuncs.com'
        )
        sts_client = StsClient(config)

        assume_role_request = sts_models.AssumeRoleRequest(
            role_arn=STS_ROLE_ARN,
            role_session_name=STS_SESSION_NAME,
            duration_seconds=3600
        )

        try:
            response = sts_client.assume_role(assume_role_request)
            print("✅ 成功获取访问权限！")
            return response.body.credentials
        except TeaException as e:
            print(f"❌ 获取STS临时凭证失败: {e.message}")
            print(f"  错误码: {e.code}")
            print("  请检查:1. 主账号AK是否正确;2. 目标角色ARN是否正确;3. 目标角色的信任策略是否已配置为信任您的主账号。")
            return None
        except Exception as e:
            print(f"❌ 发生未知错误在获取STS凭证时: {e}")
            return None

    def create_sls_client_with_sts():
        sts_credentials = get_sts_credentials()

        if not sts_credentials:
            return None

        sls_endpoint = f"{REGION}.log.aliyuncs.com"

        # aliyun-log-python-sdk 使用 securityToken 参数
        log_client = LogClient(
            endpoint=sls_endpoint,
            accessKeyId=sts_credentials.access_key_id,
            accessKey=sts_credentials.access_key_secret,
            securityToken=sts_credentials.security_token
        )

        print("✅ SLS客户端已使用临时凭证创建。")
        return log_client

    # Main analysis logic
    print("--- 开始执行根因SPAN查找任务 ---")

    # 1. 创建带有STS凭证的SLS客户端
    log_client_instance = create_sls_client_with_sts()

    # 2. 如果客户端创建成功，则开始执行查找根因span任务
    if log_client_instance:
        # 3. 创建根因span查找器，传入客户端实例
        root_cause_finder = FindRootCauseSpans(
            client=log_client_instance,
            project_name=PROJECT_NAME,
            logstore_name=LOGSTORE_NAME,
            region=REGION,
            start_time=start_time,
            end_time=end_time
        )

        print("开始查找根因spans...")
        try:
            root_cause_span_ids = root_cause_finder.find_root_cause_spans()

            print(f"找到 {len(root_cause_span_ids)} 个根因span:")
            for i, span_id in enumerate(root_cause_span_ids[:10]):
                print(f"{i+1}. {span_id}")

            if len(root_cause_span_ids) > 10:
                print(f"... 还有 {len(root_cause_span_ids) - 10} 个")
        except TeaException as e:
            print(f"查询日志时发生错误: {e.message}")
            print(f"错误码: {e.code}")
            print("请检查：1. 临时凭证是否已过期；2. 扮演的角色是否拥有对目标Project和Logstore的读权限。")
            return "unknown"
        except Exception as e:
            print(f"查询日志时发生未知错误: {e}")
            return "unknown"
    else:
        print("因无法创建SLS客户端，任务终止。")
        return "unknown"

    # Continue with the rest of the analysis...
    # Generate span conditions for further analysis
    if root_cause_span_ids:
        span_conditions = " or ".join([f"spanId='{span_id}'" for span_id in root_cause_span_ids])
        print("生成的spanId查询列表:")
        print(span_conditions[:500] + "..." if len(span_conditions) > 500 else span_conditions)
        SPAN_CONDITIONS = span_conditions
        print(f"查询条件已保存，包含 {len(root_cause_span_ids)} 个spanId")
    else:
        print("未找到根因span，无法生成查询条件")
        return {
            "root_cause": "unknown",
            "confidence": "low",
            "evidence": False,
            "error": "No root cause spans found"
        }

    # Pattern analysis
    if SPAN_CONDITIONS:
        # Build error pattern analysis query
        pattern_analysis_query = f"""
* | set session enable_remote_functions=true ;
set session velox_support_row_constructor_enabled=true;
with t0 as (
    select spanName, serviceName,
           JSON_EXTRACT_SCALAR(resources, '$["k8s.pod.ip"]') AS pod_ip,
           JSON_EXTRACT_SCALAR(resources, '$["k8s.node.name"]') AS node_name,
           JSON_EXTRACT_SCALAR(resources, '$["service.version"]') AS service_version,
           if((statusCode = 2 or statusCode = 3), 'true', 'false') as anomaly_label,
           cast(if((statusCode = 2 or statusCode = 3), 1, 0) as double) as error_count
    from log
    where {SPAN_CONDITIONS}
),
t1 as (
    select array_agg(spanName) as spanName,
           array_agg(serviceName) as serviceName,
           array_agg(pod_ip) as pod_ip,
           array_agg(node_name) as node_name,
           array_agg(service_version) as service_version,
           array_agg(anomaly_label) as anomaly_label,
           array_agg(error_count) as error_count
    from t0
),
t2 as (
    select row(spanName, serviceName) as table_row
    from t1
),
t3 as (
    select get_patterns(table_row, ARRAY['spanName', 'serviceName']) as ret
    from t2
)
select * from t3
"""

        print("执行错误特征分析查询...")

        # Convert time strings to timestamps
        start_timestamp = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S").timestamp())
        end_timestamp = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S").timestamp())

        request = GetLogsRequest(
            project=PROJECT_NAME,
            logstore=LOGSTORE_NAME,
            query=pattern_analysis_query,
            fromTime=start_timestamp,
            toTime=end_timestamp,
            line=1000
        )

        try:
            response = log_client_instance.get_logs(request)
            if response and response.get_count() > 0:
                print("错误特征分析结果:")
                get_patterns_result = None
                for log in response.get_logs():
                    contents = log.get_contents()
                    for key, value in contents.items():
                        print(f"{key}: {value}")
                        if key == "ret":
                            get_patterns_result = value
            else:
                print("未找到错误特征分析结果")
                get_patterns_result = None
        except Exception as e:
            print(f"错误特征分析查询失败: {e}")
            get_patterns_result = None

    # Continue with diff_patterns analysis...
    if SPAN_CONDITIONS:
        param_str = '{"minimum_support_fraction": 0.03}'
        diff_pattern_query = f"""
statusCode>0 | set session enable_remote_functions=true ;
set session velox_support_row_constructor_enabled=true;
with t0 as (
    select spanName, serviceName,
           JSON_EXTRACT_SCALAR(resources, '$["k8s.pod.ip"]') AS pod_ip,
           JSON_EXTRACT_SCALAR(resources, '$["k8s.node.name"]') AS node_name,
           JSON_EXTRACT_SCALAR(resources, '$["service.version"]') AS service_version,
           if(({SPAN_CONDITIONS}), 'true', 'false') as anomaly_label,
           cast(if((statusCode = 2 or statusCode = 3), 1, 0) as double) as error_count
    from log
),
t1 as (
    select array_agg(spanName) as spanName,
           array_agg(serviceName) as serviceName,
           array_agg(pod_ip) as pod_ip,
           array_agg(node_name) as node_name,
           array_agg(service_version) as service_version,
           array_agg(anomaly_label) as anomaly_label,
           array_agg(error_count) as error_count
    from t0
),
t2 as (
    select row(serviceName, anomaly_label) as table_row
    from t1
),
t3 as (
    select diff_patterns(table_row, ARRAY['serviceName', 'anomaly_label'], 'anomaly_label', 'true', 'false', '', '', '{param_str}') as ret
    from t2
)
select * from t3
"""

        print("执行diff_patterns差异模式分析查询...")

        request = GetLogsRequest(
            project=PROJECT_NAME,
            logstore=LOGSTORE_NAME,
            query=diff_pattern_query,
            fromTime=start_timestamp,
            toTime=end_timestamp,
            line=1000
        )

        try:
            response = log_client_instance.get_logs(request)
            if response and response.get_count() > 0:
                print("差异模式分析结果:")
                diff_patterns_result = None
                for log in response.get_logs():
                    contents = log.get_contents()
                    for key, value in contents.items():
                        print(f"{key}: {value}")
                        if key == "ret":
                            diff_patterns_result = value
            else:
                print("未找到差异模式分析结果")
                diff_patterns_result = None
        except Exception as e:
            print(f"差异模式分析查询失败: {e}")
            diff_patterns_result = None

    # Parse service from evidence with candidate prioritization
    def parse_service_from_evidence():
        target_service = "unknown"
        confidence_score = 0

        # Extract candidate services (without failure types)
        candidate_services = set()
        if candidate_root_causes:
            for candidate in candidate_root_causes:
                if '.' in candidate and candidate.endswith('.Failure'):
                    service_name = candidate.split('.')[0]
                    candidate_services.add(service_name)

        print(f"🎯 Limiting analysis to candidate services: {list(candidate_services)}")

        # Parse get_patterns result - ONLY consider candidates
        if 'get_patterns_result' in locals() and get_patterns_result:
            try:
                if isinstance(get_patterns_result, str):
                    data_str = get_patterns_result.replace('null', 'None')
                    result = eval(data_str)
                else:
                    result = get_patterns_result

                if len(result) >= 2 and isinstance(result[0], list) and isinstance(result[1], list):
                    service_patterns = result[0]
                    service_counts = result[1]

                    # ONLY look for candidate services - ignore all others
                    candidate_matches = []

                    for i, pattern in enumerate(service_patterns):
                        if i < len(service_counts):
                            count = service_counts[i]
                            if "serviceName=" in pattern:
                                service = pattern.split("serviceName=")[1].strip('"\'')

                                # ONLY consider if it's in candidates
                                if service in candidate_services:
                                    candidate_matches.append((service, count))
                                    print(f"🎯 Found candidate service: {service} (错误数: {count})")
                                else:
                                    print(f"❌ Ignoring non-candidate service: {service}")

                    # Use candidate with highest error count, or return empty if none
                    if candidate_matches:
                        best_candidate = max(candidate_matches, key=lambda x: x[1])
                        target_service = best_candidate[0]
                        confidence_score = best_candidate[1]
                        print(f"✅ Selected candidate service: {target_service} (错误数: {confidence_score})")
                    else:
                        print(f"❌ No candidate services found in error patterns")
                        return []  # Return empty array instead of unknown

            except Exception as e:
                print(f"⚠️ get_patterns结果解析失败: {e}")

        # Parse diff_patterns result
        if 'diff_patterns_result' in locals() and diff_patterns_result:
            try:
                if isinstance(diff_patterns_result, str):
                    data_str = diff_patterns_result.replace('null', 'None')
                    result = eval(data_str)
                else:
                    result = diff_patterns_result

                if len(result) >= 1 and isinstance(result[0], list):
                    anomaly_patterns = result[0]

                    for pattern in anomaly_patterns:
                        if "serviceName" in pattern and "=" in pattern:
                            service = pattern.split("='")[1].strip("'\"")
                            print(f"✅ diff_patterns确认异常服务: {service}")

                            if service == target_service:
                                print(f"✅ 多重证据确认: {service} 是主要根因服务")
                                return target_service, True
                            elif target_service == "unknown":
                                target_service = service
                                return target_service, True
            except Exception as e:
                print(f"⚠️ diff_patterns结果解析失败: {e}")

        if target_service != "unknown":
            return target_service, True
        else:
            print("❌ 无法从运行时证据中提取明确的目标服务")
            return "unknown", False

    # Determine final result
    if 'root_cause_span_ids' in locals() and root_cause_span_ids:
        TARGET_SERVICE, pattern_evidence = parse_service_from_evidence()
    else:
        TARGET_SERVICE = "unknown"
        pattern_evidence = False

    error_span_evidence = 'root_cause_span_ids' in locals() and len(root_cause_span_ids) > 0
    evidence = error_span_evidence and pattern_evidence

    # Simple logic: if we found a target service, return it
    if TARGET_SERVICE and TARGET_SERVICE != "unknown":
        root_cause_candidate = f"{TARGET_SERVICE}.Failure"
        print(f"\n🏆 根因候选：{root_cause_candidate}")
        return [root_cause_candidate]
    else:
        print(f"\n🏆 根因候选：unknown (no target service found)")
        return []
