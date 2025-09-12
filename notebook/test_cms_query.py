"""
测试 CMS 查询功能

去tianchi-2025-v2的workspace里面执行下面的SPL:
.entity_set with(domain='k8s', name='k8s.deployment', query=`name='recommendation'` )
| entity-call get_metric('k8s', 'k8s.metric.high_level_metric_deployment', 'deployment_cpu_usage_vs_limits', 'range', '1m')
"""

import os
import time
import unittest
from Tea.exceptions import TeaException
from alibabacloud_cms20240330.client import Client as Cms20240330Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_cms20240330 import models as cms_20240330_models
from alibabacloud_tea_util import models as util_models
from alibabacloud_sts20150401.client import Client as StsClient
from alibabacloud_sts20150401 import models as sts_models
# 加载环境变量


class TestCMSQuery(unittest.TestCase):
    """测试 CMS 查询功能"""

    def setUp(self):
        """测试设置"""
        # --- 使用账号A的凭据 ---
        self.account_a_access_key_id = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_ID')
        self.account_a_access_key_secret = os.getenv('ALIBABA_CLOUD_ACCESS_KEY_SECRET')

        # --- 账号B中角色的ARN ---
        self.role_arn_in_account_b = "acs:ram::1672753017899339:role/tianchi-user-a"

        # CMS相关配置
        self.workspace = "quanxi-tianchi-test"
        self.endpoint = 'metrics.cn-qingdao.aliyuncs.com'

        if not self.account_a_access_key_id or not self.account_a_access_key_secret:
            self.skipTest("缺少账号A的访问凭据环境变量 (ALIBABA_CLOUD_ACCESS_KEY_ID/SECRET)")

        self.cms_client = self._create_cms_client()

    def _get_sts_credentials(self):
        """使用账号A的AK，获取扮演账号B角色的临时凭证"""
        # print("🔄 正在使用账号A的凭据申请扮演账号B的角色...")
        config = open_api_models.Config(
            access_key_id=self.account_a_access_key_id, # type: ignore
            access_key_secret=self.account_a_access_key_secret, # type: ignore
            # STS的接入点可以根据需要选择，例如 'sts.cn-hangzhou.aliyuncs.com'
            endpoint='sts.cn-qingdao.aliyuncs.com'
        )
        sts_client = StsClient(config)

        assume_role_request = sts_models.AssumeRoleRequest(
            role_arn=self.role_arn_in_account_b,
            role_session_name="CmsSplQueryFromAccountA", # 会话名称，用于审计，可自定义
            duration_seconds=3600 # 临时凭证有效期，单位秒
        )

        try:
            response = sts_client.assume_role(assume_role_request)
            print("✅ 成功获取临时访问凭证！")
            return response.body.credentials
        except TeaException as e:
            print(f"❌ 获取STS临时凭证失败: {e.message}")
            print(f"  错误码: {e.code}")
            print(f"  请检查：1. 账号A的AK是否正确；2. 账号B的角色ARN是否正确；3. 账号B的角色信任策略是否正确配置为信任账号A。")
            raise

    def _create_cms_client(self) -> Cms20240330Client:
        """使用STS临时凭证创建CMS客户端"""
        # 1. 获取STS临时凭证
        sts_credentials = self._get_sts_credentials()

        # 2. 使用临时凭证配置CMS客户端
        config = open_api_models.Config(
            access_key_id=sts_credentials.access_key_id,
            access_key_secret=sts_credentials.access_key_secret,
            security_token=sts_credentials.security_token # 必须设置安全令牌
        )
        config.endpoint = self.endpoint
        return Cms20240330Client(config)

    def _execute_spl_query(self, query: str, from_time: int = None, to_time: int = None):
        """执行SPL查询"""
        max_retries = 3
        retry_count = 0

        if from_time is None:
            from_time = int(time.time()) - 60 * 60 * 1  # 24小时前（扩大时间范围）
        if to_time is None:
            to_time = int(time.time())  # 当前时间

        print(f"🔍 查询参数:")
        print(f"  Workspace: {self.workspace}")
        print(f"  时间范围: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(from_time))} 到 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(to_time))}")
        print(f"  查询语句: {query}")
        print()

        while retry_count < max_retries:
            try:
                headers = cms_20240330_models.GetEntityStoreDataHeaders()
                request = cms_20240330_models.GetEntityStoreDataRequest(
                    query=query,
                    from_=from_time,
                    to=to_time
                )
                runtime = util_models.RuntimeOptions()
                response = self.cms_client.get_entity_store_data_with_options(
                    self.workspace, request, headers, runtime
                )

                # 详细的响应调试信息
                print(f"📊 查询响应:")
                print(f"  状态码: {response.status_code if hasattr(response, 'status_code') else 'N/A'}")
                if response.body:
                    print(f"  返回header: {response.body.header}")
                    print(f"  返回data行数: {len(response.body.data) if response.body.data else 0}")
                    if hasattr(response.body, 'code'):
                        print(f"  响应code: {response.body.code}")
                    if hasattr(response.body, 'message'):
                        print(f"  响应message: {response.body.message}")
                else:
                    print(f"  响应body为空")
                print()

                return response.body
            except TeaException as e:
                print(f"❌ TeaException: code = {e.code}, message = {e.message}")
                if hasattr(e, 'data') and e.data:
                    print(f"  详细错误信息: {e.data}")
                if e.code in ["ParameterInvalid", "InvalidParameter"]:
                    break
                else:
                    time.sleep(10)
                    retry_count += 1
            except Exception as error:
                retry_count += 1
                print(f"❌ 查询失败 (尝试 {retry_count}/{max_retries}): {error}")
                if retry_count < max_retries:
                    print("等待10秒后重试...")
                    time.sleep(10)
                else:
                    raise error
        return None

    def test_environment_check(self):
        """测试环境配置"""
        print("=" * 80)
        print("环境配置检查")
        print("=" * 80)

        print(f"🔑 访问凭据检查:")
        print(f"  CMS_ACCESS_KEY_ID: {'已设置' if self.access_key_id else '❌ 未设置'}")
        print(f"  CMS_ACCESS_KEY_SECRET: {'已设置' if self.access_key_secret else '❌ 未设置'}")
        print(f"  Workspace: {self.workspace}")
        print(f"  Endpoint: {self.endpoint}")
        print()

    def test_basic_entity_query(self):
        """测试基础实体查询"""
        print("=" * 80)
        print("基础实体查询测试")
        print("=" * 80)

        # 测试不同的基础查询
        basic_queries = [
            ".entity with(domain='k8s')",  # 查询所有k8s实体
            ".entity with(domain='k8s', type='k8s.deployment')",  # 查询所有k8s deployment
        ]

        for i, query in enumerate(basic_queries, 1):
            print(f"\n--- 基础查询 {i} ---")
            result = self._execute_spl_query(query)
            if result and result.data:
                print(f"✅ 基础查询{i}成功，找到{len(result.data)}个实体")
                if len(result.data) > 0:
                    print(f"示例实体: {result.data[0]}")
            else:
                print(f"⚠️ 基础查询{i}返回空结果")

    def test_find_recommendation_deployment(self):
        """查找recommendation deployment"""
        print("=" * 80)
        print("查找 recommendation deployment")
        print("=" * 80)

        # 查询所有deployment实体，寻找recommendation
        all_deployments_query = ".entity with(domain='k8s', type='k8s.deployment')"
        all_result = self._execute_spl_query(all_deployments_query)

        if all_result and all_result.data:
            print(f"✅ 找到 {len(all_result.data)} 个deployment实体")

            # 查找名称中包含recommendation的deployment
            recommendation_deployments = []
            for deployment in all_result.data:
                # deployment[9]是name字段（根据header索引）
                if len(deployment) > 9 and 'recommendation' in str(deployment[9]).lower():
                    recommendation_deployments.append(deployment)

            if recommendation_deployments:
                print(f"✅ 找到包含'recommendation'的deployment: {len(recommendation_deployments)}个")
                for i, deployment in enumerate(recommendation_deployments):
                    print(f"  {i+1}. 名称: {deployment[9]}")
            else:
                print("❌ 没有找到名为'recommendation'的deployment")
                print("📋 当前存在的deployment名称:")
                for i, deployment in enumerate(all_result.data[:10]):  # 只显示前10个
                    print(f"  {i+1}. {deployment[9] if len(deployment) > 9 else 'N/A'}")
        else:
            print("❌ 没有找到任何k8s deployment实体")


    def test_workspace_access(self):
        """测试workspace访问权限"""
        print("=" * 80)
        print("测试 Workspace 访问权限")
        print("=" * 80)

        # 简单的查询测试workspace是否可访问
        simple_query = ".entity"  # 最简单的查询

        print(f"执行简单查询测试workspace访问: {simple_query}")
        result = self._execute_spl_query(simple_query)

        if result:
            print(f"✅ Workspace '{self.workspace}' 访问正常")
            if result.data:
                print(f"  找到 {len(result.data)} 个实体")
            else:
                print(f"  Workspace为空或时间范围内无数据")
        else:
            print(f"❌ 无法访问workspace '{self.workspace}'")
            print("💡 可能原因:")
            print("  1. Workspace名称错误")
            print("  2. 访问权限不足")
            print("  3. 网络连接问题")

    def test_recommendation_deployment_metric(self):
        """测试 recommendation deployment 的指标查询"""
        print("=" * 80)
        print("测试 recommendation deployment 的指标查询")
        print("=" * 80)

        # 根据原始需求，查询recommendation deployment的CPU使用率指标
        # 使用正确的SPL语法 - 直接使用entity-call
        query = """.entity_set with(domain='k8s', name='k8s.deployment', query=`name='recommendation'` )
| entity-call get_metric('k8s', 'k8s.metric.high_level_metric_deployment', 'deployment_memory_usage_vs_limits', 'range', '1m')"""

        print(f"查询语句: {query}")

        result = self._execute_spl_query(query)
        if result and result.data:
            print(f"🎯 ✅ recommendation deployment 指标查询成功！")
            print(f"返回数据行数: {len(result.data)}")
            if result.header:
                print(f"返回字段: {result.header}")
            if len(result.data) > 0:
                print(f"前3行数据:")
                for i, row in enumerate(result.data[:3]):
                    print(f"  行{i+1}: {row}")
        else:
            print(f"⚠️ recommendation deployment 指标查询返回空结果")
            print("💡 可能原因:")
            print("  1. 该deployment在指定时间范围内没有指标数据")
            print("  2. 指标名称不正确")
            print("  3. 需要调整查询语法")


def run_cms_query_test():
    """运行CMS查询测试的主函数"""
    print("🚀 开始执行 CMS 查询测试")
    print("=" * 80)

    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCMSQuery)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 80)
    if result.wasSuccessful():
        print("✅ 所有测试通过!")
    else:
        print("❌ 部分测试失败")
        print(f"失败数量: {len(result.failures)}")
        print(f"错误数量: {len(result.errors)}")


if __name__ == "__main__":
    run_cms_query_test()
