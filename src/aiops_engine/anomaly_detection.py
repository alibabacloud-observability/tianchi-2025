#!/usr/bin/env python
"""
A2: 异常检测和关联分析引擎
基于统计学方法和领域知识的智能异常检测
"""

import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics


class AnomalyType(Enum):
    """异常类型枚举 - 重构：区分根本原因和症状"""
    
    # === 根本原因类型（资源/系统层面问题）===
    CPU_SPIKE = "cpu_spike"                 # CPU问题导致的性能下降
    MEMORY_LEAK = "memory_leak"             # 内存问题（泄漏、不足、GC压力等）
    NETWORK_LATENCY = "network_latency"     # 纯粹的网络传输延迟
    DISK_IO = "disk_io"                     # 磁盘IO瓶颈
    GC_PRESSURE = "gc_pressure"             # 垃圾回收压力
    
    # === 症状类型（表现层面问题，需要进一步分析根因）===
    SERVICE_LATENCY = "service_latency"     # 服务响应延迟（症状，需查找根因）
    ERROR_BURST = "error_burst"             # 错误激增（症状，需查找根因）
    
    # === 其他 ===
    UNKNOWN = "unknown"


class SeverityLevel(Enum):
    """严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """异常检测结果"""
    service: str
    metric_name: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    confidence: float  # 0.0-1.0
    
    # 数值信息
    current_value: float
    baseline_mean: float
    baseline_std: float
    z_score: float
    percentage_change: float
    
    # 时间信息
    timestamp: datetime
    duration_minutes: int
    
    # 证据信息
    evidence: str
    raw_data: Dict[str, Any]


@dataclass  
class ServiceCorrelation:
    """服务关联性分析结果"""
    service_a: str
    service_b: str
    correlation_coefficient: float
    lag_seconds: int
    causal_direction: str  # "A->B", "B->A", "bidirectional", "none"
    confidence: float
    supporting_evidence: List[str]


class AnomalyDetectionEngine:
    """A2层：异常检测引擎
    
    基于统计学方法检测各类监控数据中的异常
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # 异常检测阈值配置
        self.thresholds = {
            # CPU相关阈值 - Phase 1优化：适度降低阈值提高敏感度
            'cpu_usage': {
                'z_score': 1.5,            # 从2.5降低到2.0，提高敏感度
                'percentage_change': 100,  # 从150%降低到100%，捕获更多CPU异常  
                'absolute_high': 0.70      # 从0.85降低到0.75，更早检测高CPU
            },
            
            # 内存相关阈值
            'memory_usage': {
                'z_score': 1.5,
                'percentage_change': 100,  # 100%变化视为异常
                'absolute_high': 0.70      # 绝对值90%以上
            },
            
            # 网络延迟阈值
            'latency': {
                'z_score': 1.5,
                'percentage_change': 200,  # 200%变化视为异常
                'absolute_high': 2.0       # 绝对值2秒以上
            },
            
            # 错误率阈值
            'error_rate': {
                'z_score': 2.0,
                'percentage_change': 300,  # 300%变化视为异常
                'absolute_high': 0.05      # 绝对值5%以上
            },
            
            # GC时间阈值
            'gc_time': {
                'z_score': 2.5,
                'percentage_change': 200,  # 200%变化视为异常
                'absolute_high': 1.0       # 绝对值1秒以上
            }
        }
        
        # 服务映射（从业务名映射到技术指标）
        self.service_mappings = {
            'ad': ['ad', 'advertisement'],
            'cart': ['cart', 'shopping-cart'], 
            'payment': ['payment', 'pay'],
            'checkout': ['checkout'],
            'inventory': ['inventory', 'stock'],
            'recommendation': ['recommendation', 'recommend'],
            'frontend': ['frontend', 'frontend-web', 'frontend-proxy']
        }
    
    def detect_all_anomalies(self, data_bundle) -> List[Anomaly]:
        """检测所有类型的异常
        
        Args:
            data_bundle: 来自ParallelDataCoordinator的数据包
            
        Returns:
            List[Anomaly]: 检测到的异常列表，按严重程度排序
        """
        
        self.logger.info("🔍 开始全面异常检测")
        
        anomalies = []
        # 移除调试断点
        
        # 1. K8s指标异常检测
        k8s_anomalies = self._detect_k8s_metrics_anomalies(
            data_bundle.k8s_metrics, 
            data_bundle.baseline_k8s_metrics,
            data_bundle.start_time
        )
        anomalies.extend(k8s_anomalies)
        
        # 2. APM指标异常检测  
        apm_anomalies = self._detect_apm_metrics_anomalies(
            data_bundle.apm_metrics,
            data_bundle.baseline_apm_metrics, 
            data_bundle.start_time
        )
        anomalies.extend(apm_anomalies)
        
        # 3. 日志异常检测
        log_anomalies = self._detect_log_anomalies(
            data_bundle.logs,
            data_bundle.baseline_logs,
            data_bundle.start_time
        )
        anomalies.extend(log_anomalies)
        
        # 4. 链路异常检测
        trace_anomalies = self._detect_trace_anomalies(
            data_bundle.traces,
            data_bundle.start_time
        )
        anomalies.extend(trace_anomalies)
        
        # 5. 🔗 执行因果分析：资源问题优先为根因
        causal_analyzed_anomalies = self._perform_causal_analysis(anomalies)
        
        if self.debug:
            self.logger.info(f"🔗 因果分析: {len(anomalies)} → {len(causal_analyzed_anomalies)} 异常")
        
        # 按严重程度和置信度排序
        anomalies = self._rank_anomalies(causal_analyzed_anomalies)
        
        self.logger.info(f"✅ 异常检测完成: 发现 {len(anomalies)} 个异常")
        self._log_anomaly_summary(anomalies)
        
        return anomalies
    
    
    
    def _detect_k8s_metrics_anomalies(self, current_metrics, baseline_metrics, timestamp: datetime) -> List[Anomaly]:
        """检测K8s异常 - 新格式：按Pod分别分析"""
        
        anomalies = []
        for service, service_current_metrics in current_metrics.items():
            service_baseline_metrics = baseline_metrics.get(service, {})
            for pod, pod_current_metrics in service_current_metrics.items():            
                pod_baseline_metrics = service_baseline_metrics.get(pod, {})
                for metric_name, metric_data in pod_current_metrics.items():
                    if metric_name == 'entity_id':
                        continue
                    if not isinstance(metric_data, dict) or 'values' not in metric_data:
                        continue
                        
                    current_values = metric_data['values']
                    if not current_values:
                        continue
                    
                    # 获取基线数据
                    baseline_metric_data = pod_baseline_metrics.get(metric_name, {})
                    baseline_values = baseline_metric_data.get('values', []) if isinstance(baseline_metric_data, dict) else []
                    
                    if len(baseline_values) < 3:  # 基线数据不足
                        continue
                    # if service == "ad":
                    #     import pdb; pdb.set_trace()
                    
                    # 执行异常检测
                    anomaly = self._analyze_single_metric_anomaly(
                        pod, metric_name, service, current_values, baseline_values, timestamp
                    )
                    
                    if anomaly:
                        anomalies.append(anomaly)
                        if self.debug:
                            self.logger.info(f"🔍 检测到K8s异常: {service} - {pod} - {metric_name}")
        
        return anomalies
    
    
    def _analyze_single_metric_anomaly(self, pod_name: str, metric_name: str, service: str,
                                     current_values: List[float], baseline_values: List[float], 
                                     timestamp: datetime) -> Optional[Anomaly]:
        """分析单个指标的异常情况"""
        
        # 计算统计信息

        current_mean = statistics.mean(current_values)
        baseline_mean = statistics.mean(baseline_values)
        
        try:
            baseline_std = statistics.stdev(baseline_values)
        except statistics.StatisticsError:
            baseline_std = 0
        
        if baseline_std == 0:  # 避免除零
            return None

        
        # Z-Score检测
        z_score = abs((current_mean - baseline_mean) / baseline_std)
        percentage_change = ((current_mean - baseline_mean) / baseline_mean) * 100
        
        # 根据指标类型判断异常
        anomaly_type, threshold_config = self._classify_k8s_anomaly_type(metric_name)
        
        if (z_score > threshold_config['z_score'] and 
            abs(percentage_change) > threshold_config['percentage_change']):
            
            # 计算严重程度
            severity = self._calculate_severity(z_score, abs(percentage_change), current_mean)
            # Phase 1优化：统一置信度计算，确保CPU异常公平竞争
            confidence = min(z_score / 4, 1.0)  # 与APM异常保持一致的置信度计算
            
            return Anomaly(
                service=service,
                metric_name=f"{pod_name}:{metric_name}",  # 包含pod信息
                anomaly_type=anomaly_type,
                severity=severity,
                confidence=confidence,
                current_value=current_mean,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                z_score=z_score,
                percentage_change=percentage_change,
                timestamp=timestamp,
                duration_minutes=5,  # K8s指标通常为5分钟窗口
                evidence=f"{service} pod {pod_name} {anomaly_type.value}: {current_mean:.2f} vs baseline {baseline_mean:.2f} ({percentage_change:+.1f}%)",
                raw_data={'current_values': current_values, 'baseline_values': baseline_values, 'pod_name': pod_name}
            )
        
        return None  # 未检测到异常
    
    def _is_error_metric(self, metric_name: str) -> bool:
        """判断是否是错误类指标"""
        error_keywords = ['error', 'failure', 'exception', 'fault', '错误', '异常', '故障']
        return any(keyword in metric_name.lower() for keyword in error_keywords)
    
    def _create_error_spike_anomaly(self, service: str, metric_name: str, 
                                  current_mean: float, baseline_mean: float,
                                  z_score: float, percentage_change: float, 
                                  timestamp: datetime, raw_data: List[float]) -> Anomaly:
        """创建错误激增异常对象"""
        
        # 根据错误数量确定严重程度
        if current_mean > 50:
            severity = SeverityLevel.CRITICAL
            confidence = 0.95
        elif current_mean > 10:
            severity = SeverityLevel.HIGH
            confidence = 0.85
        else:
            severity = SeverityLevel.MEDIUM
            confidence = 0.75
        
        # 计算持续时间（基于数据点数量，假设1秒1个点）
        duration_minutes = len(raw_data) // 60
        
        # 生成证据描述
        evidence = f"{service} error spike: from {baseline_mean:.1f} to {current_mean:.1f} errors"
        if percentage_change == float('inf'):
            evidence += " (baseline was healthy with no errors)"
        else:
            evidence += f" (+{percentage_change:.1f}% change)"
        
        return Anomaly(
            service=service,
            metric_name=metric_name,
            anomaly_type=AnomalyType.ERROR_BURST,
            severity=severity,
            confidence=confidence,
            current_value=current_mean,
            baseline_mean=baseline_mean,
            baseline_std=0.0,  # 基线为空时标准差为0
            z_score=z_score,
            percentage_change=percentage_change,
            timestamp=timestamp,
            duration_minutes=duration_minutes,
            evidence=evidence,
            raw_data={
                'current_values': raw_data,
                'baseline_values': [],
                'spike_detected': True,
                'baseline_was_healthy': baseline_mean == 0
            }
        )
    
    def _detect_apm_metrics_anomalies(self, current_metrics: Dict[str, Dict],
                                     baseline_metrics: Dict[str, Dict], 
                                     timestamp: datetime) -> List[Anomaly]:
        """检测APM应用指标异常 - 适配新数据格式
        
        新格式: {
            service_name: {
                metric_name: {
                    'values': [100, 120, ...],
                    'timestamps': [timestamp1, timestamp2, ...]  # 不使用，仅values用于异常检测
                }
            }
        }
        """
        
        anomalies = []
        
        for service_name, service_current_metrics in current_metrics.items():
            service_baseline_metrics = baseline_metrics.get(service_name, {})
            
            for metric_name, metric_data in service_current_metrics.items():
                if not isinstance(metric_data, dict) or 'values' not in metric_data:
                    continue
                    
                current_values = metric_data['values']
                if not current_values:
                    continue
                    
                # 获取基线数据
                baseline_metric_data = service_baseline_metrics.get(metric_name, {})
                baseline_values = baseline_metric_data.get('values', []) if isinstance(baseline_metric_data, dict) else []
                
                current_mean = statistics.mean(current_values)
                
                # 🔧 修复：处理基线数据缺失的情况
                if len(baseline_values) < 2:
                    # 对于error类指标，基线为空可能表示健康状态（无错误）
                    if self._is_error_metric(metric_name):
                        baseline_mean = 0.0
                        baseline_std = 0.0
                        
                        # 如果当前有错误而基线无错误，这是明显异常
                        if current_mean > 0:
                            z_score = float('inf')  # 无限大表示极端异常
                            percentage_change = float('inf') if baseline_mean == 0 else ((current_mean - baseline_mean) / baseline_mean) * 100
                            
                            # 直接标记为高风险异常
                            anomaly = self._create_error_spike_anomaly(
                                service_name, metric_name, current_mean, baseline_mean, 
                                z_score, percentage_change, timestamp, current_values
                            )
                            anomalies.append(anomaly)
                        
                    continue  # 其他类型指标仍然跳过
                
                # 统计计算（有充分基线数据的情况）
                baseline_mean = statistics.mean(baseline_values) 
                baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
                
                if baseline_std == 0:
                    # 基线标准差为0但有数据，可能是稳定期
                    if baseline_mean == 0 and current_mean > 0:
                        # 从0到有值的突变，特别是error类指标
                        z_score = float('inf')
                        percentage_change = float('inf')
                    else:
                        percentage_change = ((current_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean != 0 else 0
                        z_score = 0  # 稳定期无法计算z_score
                        continue
                else:
                    z_score = abs((current_mean - baseline_mean) / baseline_std)
                    percentage_change = ((current_mean - baseline_mean) / baseline_mean) * 100
                
                # APM指标异常类型识别
                anomaly_type = self._classify_apm_anomaly_type(metric_name, percentage_change)
                threshold_config = self._get_threshold_config(anomaly_type, metric_name)
                
                if (z_score > threshold_config['z_score'] and 
                    abs(percentage_change) > threshold_config['percentage_change']):
                    
                    severity = self._calculate_severity(z_score, abs(percentage_change), current_mean)
                    confidence = min(z_score / 4, 1.0)
                    
                    anomaly = Anomaly(
                        service=service_name,
                        metric_name=metric_name,
                        anomaly_type=anomaly_type,
                        severity=severity, 
                        confidence=confidence,
                        current_value=current_mean,
                        baseline_mean=baseline_mean,
                        baseline_std=baseline_std,
                        z_score=z_score,
                        percentage_change=percentage_change,
                        timestamp=timestamp,
                        duration_minutes=5,
                        evidence=f"{service_name} {metric_name}: {current_mean:.3f} vs baseline {baseline_mean:.3f} ({percentage_change:+.1f}%)",
                        raw_data={
                            'current_values': current_values,
                            'baseline_values': baseline_values,
                            'metric_type': 'apm'
                        }
                    )
                    
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_log_anomalies(self, current_logs: List[Dict[str, Any]], 
                             baseline_logs: List[Dict[str, Any]],
                             timestamp: datetime) -> List[Anomaly]:
        """检测日志异常模式"""
        
        anomalies = []
        
        # 统计当前日志的错误模式
        current_patterns = self._analyze_log_patterns(current_logs)
        baseline_patterns = self._analyze_log_patterns(baseline_logs) if baseline_logs else {}
        
        for service, patterns in current_patterns.items():
            baseline_service_patterns = baseline_patterns.get(service, {})
            
            # 检查错误率异常
            current_error_rate = patterns.get('error_rate', 0)
            baseline_error_rate = baseline_service_patterns.get('error_rate', 0)
            
            if current_error_rate > baseline_error_rate * 3 and current_error_rate > 0.1:
                anomaly = Anomaly(
                    service=service,
                    metric_name='log_error_rate',
                    anomaly_type=AnomalyType.ERROR_BURST,
                    severity=self._calculate_log_severity(current_error_rate),
                    confidence=0.8,
                    current_value=current_error_rate,
                    baseline_mean=baseline_error_rate,
                    baseline_std=0,
                    z_score=3.0,  # 简化处理
                    percentage_change=((current_error_rate - baseline_error_rate) / baseline_error_rate * 100) if baseline_error_rate > 0 else 300,
                    timestamp=timestamp,
                    duration_minutes=5,
                    evidence=f"{service} log error rate: {current_error_rate:.1%} vs baseline {baseline_error_rate:.1%}",
                    raw_data={
                        'current_patterns': patterns,
                        'baseline_patterns': baseline_service_patterns
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_trace_anomalies(self, traces: List[Dict[str, Any]], 
                               timestamp: datetime) -> List[Anomaly]:
        """检测链路追踪异常"""
        
        anomalies = []
        
        if not traces:
            return anomalies
        
        # 按服务分组分析延迟
        service_latencies = {}
        for trace in traces:
            service = trace.get('service_name', 'unknown')
            duration_ms = trace.get('duration_ms', 0)
            
            if service not in service_latencies:
                service_latencies[service] = []
            service_latencies[service].append(duration_ms)
        
        # 检测高延迟异常
        for service, latencies in service_latencies.items():
            if len(latencies) < 3:
                continue
            
            # 计算延迟统计
            mean_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            # 高延迟阈值判断（简化版）
            if mean_latency > 1000 or p95_latency > 5000:  # 1秒平均或5秒P95
                severity = SeverityLevel.HIGH if p99_latency > 10000 else SeverityLevel.MEDIUM
                
                anomaly = Anomaly(
                    service=service,
                    metric_name='trace_latency',
                    anomaly_type=AnomalyType.SERVICE_LATENCY,
                    severity=severity,
                    confidence=0.7,
                    current_value=mean_latency,
                    baseline_mean=500,  # 假设正常基线500ms
                    baseline_std=100,
                    z_score=(mean_latency - 500) / 100,
                    percentage_change=((mean_latency - 500) / 500) * 100,
                    timestamp=timestamp,
                    duration_minutes=5,
                    evidence=f"{service} high latency: avg {mean_latency:.0f}ms, p95 {p95_latency:.0f}ms",
                    raw_data={
                        'latencies': latencies,
                        'stats': {
                            'mean': mean_latency,
                            'p95': p95_latency,
                            'p99': p99_latency
                        }
                    }
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _classify_k8s_anomaly_type(self, metric_name: str) -> Tuple[AnomalyType, Dict]:
        """分类K8s指标异常类型"""
        metric_lower = metric_name.lower()
        
        if 'cpu' in metric_lower:
            return AnomalyType.CPU_SPIKE, self.thresholds['cpu_usage']
        elif 'memory' in metric_lower:
            return AnomalyType.MEMORY_LEAK, self.thresholds['memory_usage']
        elif 'network' in metric_lower:
            return AnomalyType.NETWORK_LATENCY, self.thresholds['latency']
        else:
            return AnomalyType.UNKNOWN, self.thresholds['cpu_usage']  # 默认
    
    def _classify_apm_anomaly_type(self, metric_name: str, percentage_change: float) -> AnomalyType:
        """分类APM指标异常类型 - 重构：正确区分根因和症状"""
        metric_lower = metric_name.lower()
        
        if 'error' in metric_lower:
            return AnomalyType.ERROR_BURST  # 症状：错误激增
        elif 'latency' in metric_lower or 'response_time' in metric_lower:
            return AnomalyType.SERVICE_LATENCY  # 症状：服务响应延迟（不是网络延迟！）
        elif 'request' in metric_lower and percentage_change < 0:
            return AnomalyType.SERVICE_LATENCY  # 请求下降可能导致服务响应问题
        elif 'gc' in metric_lower:
            return AnomalyType.GC_PRESSURE  # 根因：GC压力
        else:
            return AnomalyType.UNKNOWN
    
    def _get_threshold_config(self, anomaly_type: AnomalyType, metric_name: str) -> Dict:
        """获取阈值配置 - 更新：支持新的异常类型"""
        if anomaly_type == AnomalyType.CPU_SPIKE:
            return self.thresholds['cpu_usage']
        elif anomaly_type == AnomalyType.MEMORY_LEAK:
            return self.thresholds['memory_usage']
        elif anomaly_type == AnomalyType.NETWORK_LATENCY:
            return self.thresholds['latency']
        elif anomaly_type == AnomalyType.SERVICE_LATENCY:
            return self.thresholds['latency']  # 服务延迟使用延迟阈值
        elif anomaly_type == AnomalyType.ERROR_BURST:
            return self.thresholds['error_rate']
        elif anomaly_type == AnomalyType.GC_PRESSURE:
            return self.thresholds['gc_time']
        else:
            return self.thresholds['cpu_usage']  # 默认
    
    def _is_root_cause_type(self, anomaly_type: AnomalyType) -> bool:
        """判断异常类型是否为根本原因类型"""
        root_cause_types = {
            AnomalyType.CPU_SPIKE,
            AnomalyType.MEMORY_LEAK,
            AnomalyType.NETWORK_LATENCY,
            AnomalyType.DISK_IO,
            AnomalyType.GC_PRESSURE
        }
        return anomaly_type in root_cause_types
    
    def _is_symptom_type(self, anomaly_type: AnomalyType) -> bool:
        """判断异常类型是否为症状类型"""
        symptom_types = {
            AnomalyType.SERVICE_LATENCY,
            AnomalyType.ERROR_BURST
        }
        return anomaly_type in symptom_types
    
    def _perform_causal_analysis(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """执行因果分析：资源问题优先为根因，症状次之
        
        核心逻辑：
        1. 如果同时存在资源异常和症状异常，优先选择资源异常为根因
        2. 如果症状异常无对应资源异常，症状本身为根因
        3. 资源异常始终保留为根因
        """
        
        if not anomalies:
            return anomalies
            
        # 按服务分组分析
        service_anomalies = {}
        for anomaly in anomalies:
            service = anomaly.service
            if service not in service_anomalies:
                service_anomalies[service] = []
            service_anomalies[service].append(anomaly)
        
        final_anomalies = []
        
        for service, service_anomaly_list in service_anomalies.items():
            # 分离根因异常和症状异常
            root_cause_anomalies = [a for a in service_anomaly_list if self._is_root_cause_type(a.anomaly_type)]
            symptom_anomalies = [a for a in service_anomaly_list if self._is_symptom_type(a.anomaly_type)]
            other_anomalies = [a for a in service_anomaly_list if not self._is_root_cause_type(a.anomaly_type) and not self._is_symptom_type(a.anomaly_type)]
            
            # 1. 资源异常始终保留（根因优先）
            final_anomalies.extend(root_cause_anomalies)
            
            # 2. 处理症状异常
            if root_cause_anomalies:
                # 有根因异常时，症状异常被解释为由根因导致，不单独报告
                if self.debug:
                    root_causes = [a.anomaly_type.value for a in root_cause_anomalies]
                    symptoms = [a.anomaly_type.value for a in symptom_anomalies]
                    self.logger.info(f"🔗 服务 {service} 因果分析: 根因 {root_causes} 解释症状 {symptoms}")
            else:
                # 无根因异常时，症状本身为根因
                final_anomalies.extend(symptom_anomalies)
                if self.debug and symptom_anomalies:
                    self.logger.info(f"🔍 服务 {service} 症状异常无对应根因，症状本身为根因")
            
            # 3. 其他未分类异常保留
            final_anomalies.extend(other_anomalies)
        
        return final_anomalies
    
    def _calculate_severity(self, z_score: float, percentage_change: float, current_value: float) -> SeverityLevel:
        """计算异常严重程度"""
        
        # 综合评分
        severity_score = 0
        
        # Z-score贡献
        if z_score > 5:
            severity_score += 3
        elif z_score > 3:
            severity_score += 2
        elif z_score > 2:
            severity_score += 1
        
        # 百分比变化贡献
        if percentage_change > 500:
            severity_score += 3
        elif percentage_change > 200:
            severity_score += 2
        elif percentage_change > 100:
            severity_score += 1
        
        # 绝对值贡献（针对特定指标）
        if current_value > 0.9:  # 如CPU/内存使用率超过90%
            severity_score += 2
        elif current_value > 0.8:
            severity_score += 1
        
        # 映射到严重程度级别
        if severity_score >= 6:
            return SeverityLevel.CRITICAL
        elif severity_score >= 4:
            return SeverityLevel.HIGH
        elif severity_score >= 2:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _calculate_log_severity(self, error_rate: float) -> SeverityLevel:
        """计算日志异常严重程度"""
        if error_rate > 0.5:
            return SeverityLevel.CRITICAL
        elif error_rate > 0.3:
            return SeverityLevel.HIGH
        elif error_rate > 0.1:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _analyze_log_patterns(self, logs: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """分析日志模式"""
        patterns = {}
        
        for log in logs:
            if not isinstance(log, dict):
                continue
                
            service = log.get('service_name', 'unknown')
            
            if service not in patterns:
                patterns[service] = {
                    'total_logs': 0,
                    'error_logs': 0,
                    'error_rate': 0.0
                }
            
            patterns[service]['total_logs'] += 1
            
            # 检查是否为错误日志
            log_text = str(log.get('raw_log_text', '')).lower()
            if any(keyword in log_text for keyword in ['error', 'exception', 'fail', 'timeout']):
                patterns[service]['error_logs'] += 1
        
        # 计算错误率
        for service, pattern in patterns.items():
            if pattern['total_logs'] > 0:
                pattern['error_rate'] = pattern['error_logs'] / pattern['total_logs']
        
        return patterns
    
    def _extract_service_from_k8s_metric(self, metric_name: str) -> str:
        """从K8s指标名称中提取服务名"""
        # 分析metric_name格式，提取业务应用信息
        # 例: "cpu_usage_percent[ad+cart+paymentpods×12]" -> "multi-service"
        
        if '[' in metric_name:
            service_info = metric_name.split('[')[1].split(']')[0]
            if '+' in service_info:
                return "multi-service"  # 多服务指标
            else:
                # 提取单一服务名
                for app in service_info.split('pods')[0].split('+'):
                    return app.strip()
        
        return "k8s-cluster"
    
    def _rank_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """对异常进行排序"""
        
        # 定义严重程度权重
        severity_weights = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.HIGH: 3, 
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 1
        }
        
        # 按严重程度和置信度排序 - 重构：根因优先于症状
        def sort_key(anomaly):
            # 基础排序权重
            base_score = (severity_weights[anomaly.severity], anomaly.confidence, anomaly.z_score)
            
            # 根本原因类型获得优先权重（资源问题优先为根因）
            if self._is_root_cause_type(anomaly.anomaly_type):
                root_cause_bonus = 0.1  # 根因获得显著优先权
            else:
                root_cause_bonus = 0.0
            
            return (base_score[0], base_score[1] + root_cause_bonus, base_score[2])
        
        return sorted(anomalies, key=sort_key, reverse=True)
    
    def _log_anomaly_summary(self, anomalies: List[Anomaly]):
        """记录异常摘要"""
        
        if not anomalies:
            return
        
        # 按严重程度统计
        severity_counts = {}
        type_counts = {}
        service_counts = {}
        
        for anomaly in anomalies:
            severity_counts[anomaly.severity] = severity_counts.get(anomaly.severity, 0) + 1
            type_counts[anomaly.anomaly_type] = type_counts.get(anomaly.anomaly_type, 0) + 1
            service_counts[anomaly.service] = service_counts.get(anomaly.service, 0) + 1
        
        self.logger.info("📊 异常检测摘要:")
        self.logger.info(f"   严重程度分布: {dict(severity_counts)}")
        self.logger.info(f"   异常类型分布: {dict(type_counts)}")
        self.logger.info(f"   受影响服务: {dict(service_counts)}")
        
        # 显示前5个最严重的异常
        # import pdb; pdb.set_trace()
        self.logger.info("🔥 最严重的异常:")
        for i, anomaly in enumerate(anomalies[:30], 1):
            self.logger.info(f"   {i}. {anomaly.service} - {anomaly.anomaly_type.value} ({anomaly.severity.value}) - {anomaly.evidence}")


class CorrelationAnalysisEngine:
    """关联分析引擎"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
    
    def analyze_service_correlations(self, anomalies: List[Anomaly], 
                                   data_bundle) -> List[ServiceCorrelation]:
        """分析服务间的关联性"""
        
        self.logger.info("🔗 开始服务关联分析")
        
        correlations = []
        
        # 1. 基于异常传播的关联分析
        propagation_correlations = self._analyze_anomaly_propagation(anomalies)
        correlations.extend(propagation_correlations)
        
        # 2. 基于时序数据的关联分析
        metric_correlations = self._analyze_metric_correlations(data_bundle)
        correlations.extend(metric_correlations)
        
        self.logger.info(f"✅ 关联分析完成: 发现 {len(correlations)} 个服务关联")
        
        return correlations
    
    def _analyze_anomaly_propagation(self, anomalies: List[Anomaly]) -> List[ServiceCorrelation]:
        """基于异常传播分析服务关联"""
        # 简化实现：基于异常时间序列分析服务间的影响关系
        correlations = []
        
        # 按服务分组异常
        service_anomalies = {}
        for anomaly in anomalies:
            if anomaly.service not in service_anomalies:
                service_anomalies[anomaly.service] = []
            service_anomalies[anomaly.service].append(anomaly)
        
        # 寻找时间相关的异常模式
        services = list(service_anomalies.keys())
        for i, service_a in enumerate(services):
            for service_b in services[i+1:]:
                # 简化的关联度计算
                correlation_score = 0.5  # 占位符
                
                if correlation_score > 0.6:
                    correlation = ServiceCorrelation(
                        service_a=service_a,
                        service_b=service_b,
                        correlation_coefficient=correlation_score,
                        lag_seconds=0,
                        causal_direction="unknown",
                        confidence=0.6,
                        supporting_evidence=[f"Both services show anomalies in similar timeframe"]
                    )
                    correlations.append(correlation)
        
        return correlations
    
    def _analyze_metric_correlations(self, data_bundle) -> List[ServiceCorrelation]:
        """基于指标数据分析关联性"""
        # 简化实现，返回空列表
        # 在完整版本中，这里会分析时间序列的相关性
        return []
