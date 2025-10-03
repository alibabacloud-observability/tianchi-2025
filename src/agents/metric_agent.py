"""
Real Metric Agent for RCA Data Collection - Using Real MCP Data
"""
import logging
from datetime import datetime
from typing import List, Dict, Any
from tools.paas_entity_tools import umodel_get_entities
from tools.paas_data_tools import umodel_get_golden_metrics
from ..utils.evidence_chain import EvidenceChain
import ast

class MinimalMetricAgent:
    """Minimal metric agent for data collection - 支持离线模式."""
    
    def __init__(self, debug: bool = False, offline_mode: bool = False, problem_id: str = None):
        self.debug = debug
        self.offline_mode = offline_mode
        self.problem_id = problem_id
        self.logger = logging.getLogger(__name__)
        if offline_mode:
            # 离线模式：使用本地数据加载器
            from ..utils.local_data_loader import get_local_data_loader
            self.local_loader = get_local_data_loader(debug=debug)
            self.logger.info(f"✅ MinimalMetricAgent初始化完成 (离线模式, 问题ID: {problem_id})")
        else:
            
            self.logger.info(f"✅ MinimalMetricAgent初始化完成 (在线模式)")
        
        # Service to Entity ID mapping (cached)
        self._service_entity_mapping = None
        
        # APM service list (共用)
        self.apm_services = ["accounting", "ad", "cart", "checkout", "currency", 
                            "email", "fraud-detection","frontend", "frontend-proxy", 
                            "frontend-web", "image-provider", "inventory","payment", 
                            "product-catalog", "quote", "recommendation", "shipping"]
        
    def _get_service_entity_mapping(self, start_time: datetime, end_time: datetime) -> Dict[str, str]:
        """Get service name to entity ID mapping (cached)."""
        if self._service_entity_mapping is not None:
            return self._service_entity_mapping
            
        if self.offline_mode:
            self.logger.info("⚠️ 离线模式不需要服务实体映射")
            return {}
        self._service_entity_mapping = {}
        try:
            # 使用umodel_get_entities获取APM服务实体
            # 使用问题的实际时间范围 - 必须提供！
            from_ts = int(start_time.timestamp())
            to_ts = int(end_time.timestamp())
            self.logger.info(f"🔍 使用问题时间范围查询服务实体: {start_time} ~ {end_time}")
            
            query_params = {
                "domain": "apm",
                "entity_set_name": "apm.service",
                "from_time": from_ts,
                "to_time": to_ts
            }
            
            result = umodel_get_entities.invoke(query_params)
            
            self.logger.info(f"📊 service查询结果: error={result.error if result else 'No result'}, 有data={bool(result and result.data)}")
            
            if result and not result.error and result.data:
                content = result.data
                self.logger.info(f"🔍 service响应content项数: {len(content)}")
                self.logger.info(f"🔍 找到service_records: {len(content)}条记录")                    
                for item in content:
                    self._service_entity_mapping[item['service']] = item['__entity_id__']
                    
                if self.debug:                                
                    self.logger.info(f"📊 {item['service']}: ({item['__entity_id__']})")
            
        except Exception as e:
            self.logger.error(f"❌ 获取服务实体映射失败: {e}")
            return {}

        # 返回按Pod组织的K8s指标数据
        self.logger.info(f"✅ service收集完成: {len(self._service_entity_mapping.keys())}个service")
        return self._service_entity_mapping
                    
        
    def _try_fetch_k8s_golden_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Fetch real K8s golden metrics with application pod identification."""
        if self.offline_mode:
            self.logger.info("⚠️ K8s指标在离线模式下不通过此方法获取，应由_fetch_metric_data处理")
            return {}
        
        try:
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            # 第1步：获取pod实体信息以建立应用映射
            pod_app_mapping = self._get_pod_application_mapping(start_ts, end_ts)
            
            
            # 第2步：获取K8s golden metrics
            if 'business_apps' not in pod_app_mapping:
                self.logger.warning("⚠️ 未找到业务应用映射")
                return {}
            
            self.logger.info(f"🔍 K8s数据查询")
            self.logger.info(f"   🕐 时间范围: {start_time} ~ {end_time} ({start_ts} ~ {end_ts})")
            self.logger.info(f"   🕐 时间戳差异: {end_ts - start_ts}秒 ({(end_ts - start_ts)/60:.1f}分钟)")
            
            k8s_metrics = {}
            
            for app, pods in pod_app_mapping['business_apps'].items():
                if app not in self.apm_services:
                    continue
                k8s_metrics[app] = {}
                self.logger.info(f"🔍 查找{app}的metrics_records...")
                
                for pod in pods:
                    k8s_metrics[app][pod['name']] = {}
                    k8s_metrics[app][pod['name']]['entity_id'] = pod['entity_id']
                    self.logger.info(f"🔍 查找{pod['name']}的metrics_records...")
                    
                    query_params = {    
                        "domain": "k8s",
                        "entity_set_name": "k8s.pod",
                        "entity_ids": [pod['entity_id']],
                        "from_time": start_ts,
                        "to_time": end_ts
                    }

                    result = umodel_get_golden_metrics.invoke(query_params)
                    
                    # Parse real metrics response
                    self.logger.info(f"📊 K8s查询结果: error={result.error if result else 'No result'}, 有data={bool(result and result.data)}")
                    
                    if result and not result.error and result.data:
                        content = result.data
                        self.logger.info(f"🔍 K8s响应content项数: {len(content)}")
                        self.logger.info(f"🔍 找到metrics_records: {len(content)}条记录")                    
                        
                        for item in content:
                            # 🔧 修复：直接按pod组织K8s时间序列数据，保留时间戳和pod标识
                            pod_metrics = {"values": [], "timestamps": []}
                            pod_metrics['values'] = ast.literal_eval(item.get('__value__', '[0]'))
                            pod_metrics['timestamps'] = ast.literal_eval(item.get('__ts__', '[0]'))
                            k8s_metrics[app][pod['name']][item['metric']] = pod_metrics  
                            
                            if self.debug:                                
                                self.logger.info(f"📊 {app} ({pod['name']}) - {item['metric']}: {len(pod_metrics['values'])}个数据点")

            # 返回按Pod组织的K8s指标数据
            self.logger.info(f"✅ K8s指标收集完成: {len(k8s_metrics)}个应用, {sum(len(pod_data) for pod_data in k8s_metrics.values())}个Pod")
            return k8s_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to fetch K8s metrics with app info: {e}")
            return {}
    
    def _get_pod_application_mapping(self, start_ts: int, end_ts: int) -> Dict[str, Dict]:
        """获取pod到应用的映射关系"""
        business_apps = {}
        if self.offline_mode:
            return {"business_apps": {}}
        
        try:
            
            query_params = {
                "domain": "k8s",
                "entity_set_name": "k8s.pod",
                "from_time": start_ts,
                "to_time": end_ts,
                "limit": 1000
            }

            
            result = umodel_get_entities.invoke(query_params)

            self.logger.info(f"📊 K8s查询结果: error={result.error if result else 'No result'}, 有data={bool(result and result.data)}")
            

            if result and not result.error and result.data:
                content = result.data
                self.logger.info(f"🔍 K8s响应content项数: {len(content)}")
                self.logger.info(f"🔍 找到pod实体: {len(content)}条记录")                    
                
                for item in content:
                    if item['namespace'] != 'cms-demo':
                        continue
                    pod_name = item.get('name', '')
                    entity_id = item.get('__entity_id__', '')
                    entity_type = item.get('__entity_type__', '')
                    # 使用新函数清理pod名称，提取服务名称
                    app_name = self._clean_pod_name_to_service(pod_name)
                    
                    if app_name not in business_apps:
                        business_apps[app_name] = []
                    
                    business_apps[app_name].append({
                                            'name': pod_name,
                        'entity_id': entity_id,
                        'entity_type': entity_type
                    })

                    if self.debug:                                
                        self.logger.info(f"📊 Pod映射: {pod_name} -> {app_name} (entity_id: {entity_id})")
                

                return {"business_apps": business_apps}
            else:
                return {"business_apps": {}}
            
        except Exception as e:
            self.logger.error(f"❌ 获取Pod应用映射失败: {e}")
            return {"business_apps": {}}

    def _clean_pod_name_to_service(self, pod_name: str) -> str:
        """从pod名称中提取服务名称，去除K8s生成的后缀
        
        Examples:
            'cart-ds-6kgk6' -> 'cart' (DaemonSet格式)
            'cart-7d8f6c4b5d-xyz12' -> 'cart' (Deployment格式)
            'cart-abc123' -> 'cart' (一般格式)
            'loongcollector-ds-6kgk6' -> 'loongcollector'
            'frontend-proxy-asdfaasdf-23r23r' -> 'frontend-proxy'
        """
        import re
        
        # 模式1: DaemonSet格式 'name-ds-xxxxx'
        ds_match = re.match(r'^(.+?)-ds-[a-z0-9]+$', pod_name)
        if ds_match:
            if self.debug:
                self.logger.info(f"🔍 DaemonSet模式匹配: {pod_name} -> {ds_match.group(1)}")
            return ds_match.group(1)
        
        # 模式2: Deployment格式 'name-xxxxxxxxx-xxxxx' (双层随机后缀)
        deployment_match = re.match(r'^(.+?)-[a-z0-9]{8,}-[a-z0-9]{5}$', pod_name)
        if deployment_match:
            if self.debug:
                self.logger.info(f"🔍 Deployment模式匹配: {pod_name} -> {deployment_match.group(1)}")
            return deployment_match.group(1)
        
        # 模式3: 一般格式 'name-xxxxx' (单层随机后缀，至少5个字符)
        general_match = re.match(r'^(.+?)-[a-z0-9]{5,}$', pod_name)
        if general_match:
            if self.debug:
                self.logger.info(f"🔍 一般模式匹配: {pod_name} -> {general_match.group(1)}")
            return general_match.group(1)
                
        # 如果没有匹配到任何模式，返回原名称
        if self.debug:
            self.logger.warning(f"⚠️ 无法解析pod名称: {pod_name}，返回原名称")
        return pod_name

    def _extract_service_from_pod(self, pod_name: str) -> str:
        """从pod名称中推断业务应用名称"""
        # 首先清理pod名称，去除K8s后缀
        clean_name = self._clean_pod_name_to_service(pod_name)
        
        # 业务服务映射规则 - 基于实际观察的pod命名规律
        # 优化：增加精确匹配和部分匹配的优先级
        service_patterns = {
            'payment': ['payment', 'pay'],
            'checkout': ['checkout', 'order'],
            'inventory': ['inventory', 'stock'],
            'cart': ['cart', 'shopping', 'shopping-cart'],
            'ad': ['ad', 'advertisement', 'ads'],
            'frontend': ['frontend', 'web', 'ui', 'frontend-proxy'],
            'user': ['user', 'account', 'auth'],
            'product': ['product', 'catalog'],
            'recommendation': ['recommendation', 'recommend', 'suggestion'],
            'shipping': ['shipping', 'delivery'],
            'loongcollector': ['loongcollector', 'collector'],
            'fraud-detection': ['fraud-detection', 'fraud'],
            'currency': ['currency', 'curr'],
            'email': ['email', 'mail']
        }
        
        clean_lower = clean_name.lower()
        
        # 第一轮：精确匹配优先（避免子字符串误匹配）
        for service, patterns in service_patterns.items():
            for pattern in patterns:
                if clean_lower == pattern:
                    if self.debug:
                        self.logger.info(f"🎯 精确匹配: {clean_name} -> {service}")
                    return service
        
        # 第二轮：部分匹配（保持向后兼容性）
        for service, patterns in service_patterns.items():
            for pattern in patterns:
                if pattern in clean_lower:
                    if self.debug:
                        self.logger.info(f"🔍 部分匹配: {clean_name} -> {service} (通过模式: {pattern})")
                    return service
        
        # 如果没有匹配的特定服务，使用清理后的名称
        parts = clean_name.replace('-', '_').split('_')
        for part in parts:
            if part in ['biz', 'demo', 'k8s', 'pod']:
                continue
            if len(part) > 2:  # 过滤太短的部分
                return part
        
        # 默认返回清理后的名称或generic
        return clean_name if clean_name else "generic"

    def _try_fetch_apm_golden_metrics(self, service_name: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Fetch real APM service metrics using umodel_get_golden_metrics."""
        if self.offline_mode:
            self.logger.info("⚠️ APM指标在离线模式下不通过此方法获取，应由_fetch_metric_data处理")
            return {}
        
        try:
            start_ts = int(start_time.timestamp())
            end_ts = int(end_time.timestamp())
            
            # 获取服务实体映射
            
            service_mapping = self._get_service_entity_mapping(start_time, end_time)
            if not service_mapping or service_name not in service_mapping:
                self.logger.warning(f"⚠️ 服务 '{service_name}' 未找到实体映射")
                return {}
                
            entity_id = service_mapping[service_name]
            self.logger.info(f"🔍 APM数据查询: {service_name} -> {entity_id}")
            
            # 使用umodel_get_golden_metrics工具
            query_params = {
                "domain": "apm",
                "entity_set_name": "apm.service",
                "entity_ids": [entity_id],
                "from_time": start_ts,
                "to_time": end_ts
            }
            
            result = umodel_get_golden_metrics.invoke(query_params)
            
            # Parse APM metrics response
            self.logger.info(f"📊 APM查询结果: error={result.error if result else 'No result'}")
            
            if result and not result.error and result.data:
                content = result.data
                self.logger.info(f"🔍 APM响应content项数: {len(content)}")
                
                apm_metrics = {service_name: {'entity_id': entity_id}}
                
                for item in content:
                    metric_name = item.get('metric', 'unknown')
                    apm_metrics[service_name][metric_name] = {
                        'values': ast.literal_eval(item.get('__value__', '[0]')),
                        'timestamps': ast.literal_eval(item.get('__ts__', '[0]'))
                    }
                    
                    if self.debug:
                        self.logger.info(f"📊 {service_name} - {metric_name}: APM指标收集")
                
                return apm_metrics
            else:
                self.logger.warning(f"⚠️ {service_name}: APM指标获取失败")
                return {}
            
        except Exception as e:
            self.logger.error(f"❌ APM指标获取异常 {service_name}: {e}")
            return {}

    def _fetch_metric_data(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """根据模式获取指标数据"""
        if self.offline_mode:
            # 离线模式：从本地文件加载数据
            if not self.problem_id:
                self.logger.error("❌ 离线模式需要指定problem_id")
                return {"k8s_metrics": {}, "apm_metrics": {}}
            
            # 判断是故障期还是基线期数据  
            duration_minutes = (end_time - start_time).total_seconds() / 60
            data_type = "failure" if duration_minutes <= 10 else "baseline"
            
            self.logger.info(f"📈 从本地加载 {self.problem_id} {data_type} 指标数据")
            metric_data = self.local_loader.load_metrics(self.problem_id, data_type)
            
            return {
                'k8s_golden_metrics': metric_data.get("k8s_metrics", {}),
                'apm_service_metrics': metric_data.get("apm_metrics", {})
            }
        else:
            # 在线模式：从MCP服务获取数据
        # Fetch K8s golden metrics
            k8s_metrics = self._try_fetch_k8s_golden_metrics(start_time, end_time)
        
        # Fetch APM service metrics for key services
        apm_metrics = {}
        key_services = self.apm_services
        
        for service in key_services:
            apm_metrics[service] = {}
            # 新函数一次性获取一个服务的所有指标
            service_metrics = self._try_fetch_apm_golden_metrics(service, start_time, end_time)                                
            if service_metrics and service in service_metrics:
                service_data = service_metrics[service]
                # 只提取目标指标
                target_metrics = ["request_count", "error_count", "avg_request_latency_seconds"]
                
                for metric in target_metrics:
                    if metric in service_data and isinstance(service_data[metric], dict):
                        # 提取values列表以保持原有格式兼容性
                        metric_data = service_data[metric]
                        if 'values' in metric_data:
                            apm_metrics[service][metric] = {
                                "values": metric_data['values'],
                                "timestamps": metric_data['timestamps']
                            }
        
        return {
            'k8s_golden_metrics': k8s_metrics,
            'apm_service_metrics': apm_metrics
        }
        
    def analyze(self, evidence_chain: EvidenceChain) -> Dict[str, Any]:
        """分析指标数据 - 支持离线模式"""
        mode_desc = "离线模式" if self.offline_mode else "在线模式"
        self.logger.info(f"📊 开始指标分析 ({mode_desc})")
        
        # 获取指标数据
        metric_data = self._fetch_metric_data(evidence_chain.start_time, evidence_chain.end_time)
        
        # 提取统计信息
        k8s_metrics = metric_data.get('k8s_golden_metrics', {})
        apm_metrics = metric_data.get('apm_service_metrics', {})
        
        k8s_count = sum(len(pod_data) for app_data in k8s_metrics.values() for pod_data in app_data.values() if isinstance(pod_data, dict))
        apm_count = len(apm_metrics)
        
        # 记录证据
        evidence_chain.add_evidence('metric', f'{mode_desc}_metric_analysis', metric_data, confidence=0.8)
        
        self.logger.info(f"✅ 指标分析完成 ({mode_desc})")
        self.logger.info(f"   K8s指标: {k8s_count}个")
        self.logger.info(f"   APM指标: {apm_count}个")
        
        return {
            'k8s_metrics_count': k8s_count,
            'apm_metrics_count': apm_count,
            'k8s_apps': len(k8s_metrics),
            'apm_services': len(apm_metrics),
            'total_metrics': k8s_count + apm_count
        }
