#!/usr/bin/env python
"""
A2: 专家规则引擎和综合评分系统
基于运维专家经验的故障诊断规则库
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import yaml

from .anomaly_detection import Anomaly, AnomalyType, SeverityLevel, ServiceCorrelation


@dataclass
class ExpertRule:
    """专家规则定义"""
    name: str
    description: str
    target_candidates: List[str]  # 适用的候选根因，如 ["ad.LargeGc", "ad.memory"]
    
    # 触发条件
    required_anomaly_types: List[AnomalyType]
    required_services: List[str]  # 可选，为空表示任意服务
    min_anomaly_count: int
    min_severity: SeverityLevel
    
    # 评分权重
    base_score: float  # 基础分数 (0.0-1.0)
    confidence_multiplier: float  # 置信度乘数
    
    # 附加条件
    additional_conditions: Dict[str, Any]
    supporting_evidence: List[str]


@dataclass
class RootCauseScore:
    """根因评分结果"""
    candidate: str
    total_score: float
    confidence: float
    
    # 分维度评分
    anomaly_score: float
    correlation_score: float
    expert_rule_score: float
    temporal_score: float
    business_impact_score: float
    
    # 支持证据
    supporting_anomalies: List[Anomaly]
    supporting_correlations: List[ServiceCorrelation]
    matched_rules: List[str]
    evidence_summary: List[str]
    
    # 推理链
    reasoning_chain: List[str]


class ExpertRulesEngine:
    """专家规则引擎
    
    编码运维专家的故障诊断经验和模式识别规则
    """
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # 加载专家规则库
        self.rules = self._initialize_expert_rules()
        
        # 服务依赖拓扑（简化版）
        self.service_topology = {
            'ad': {
                'downstream': ['recommendation', 'user-profile'],
                'upstream': ['frontend', 'load-balancer'],
                'criticality': 'high'
            },
            'cart': {
                'downstream': ['inventory', 'payment', 'checkout'],
                'upstream': ['frontend', 'recommendation'],
                'criticality': 'high'
            },
            'payment': {
                'downstream': ['risk-control', 'bank-gateway'],
                'upstream': ['cart', 'checkout'],
                'criticality': 'critical'
            },
            'checkout': {
                'downstream': ['inventory', 'payment', 'shipping'],
                'upstream': ['cart', 'frontend'],
                'criticality': 'critical'
            },
            'inventory': {
                'downstream': ['database', 'cache'],
                'upstream': ['cart', 'checkout'],
                'criticality': 'medium'
            },
            'recommendation': {
                'downstream': ['ml-engine', 'user-profile'],
                'upstream': ['frontend', 'ad'],
                'criticality': 'medium'
            }
        }
        
        self.logger.info(f"✅ 专家规则引擎初始化完成: {len(self.rules)} 条规则")
    
    def evaluate_candidates(self, candidates: List[str], anomalies: List[Anomaly], 
                           correlations: List[ServiceCorrelation], 
                           data_bundle) -> List[RootCauseScore]:
        """评估所有候选根因并评分排序
        
        Args:
            candidates: 候选根因列表，如 ["ad.Failure", "ad.LargeGc", "cart.Failure"]
            anomalies: 检测到的异常列表
            correlations: 服务关联分析结果
            data_bundle: 原始数据包
            
        Returns:
            List[RootCauseScore]: 按总分排序的评分结果
        """
        
        self.logger.info(f"🎯 开始评估 {len(candidates)} 个候选根因")
        
        scored_candidates = []
        
        for candidate in candidates:
            score = self._evaluate_single_candidate(
                candidate, anomalies, correlations, data_bundle
            )
            scored_candidates.append(score)
        
        # 按总分排序
        scored_candidates.sort(key=lambda x: x.total_score, reverse=True)
        
        self._log_scoring_summary(scored_candidates)
        
        return scored_candidates
    
    def _evaluate_single_candidate(self, candidate: str, anomalies: List[Anomaly],
                                  correlations: List[ServiceCorrelation], 
                                  data_bundle) -> RootCauseScore:
        """评估单个候选根因"""
        
        service, failure_type = self._parse_candidate(candidate)
        
        # 1. 异常匹配评分
        anomaly_score, supporting_anomalies = self._score_anomaly_match(
            candidate, service, failure_type, anomalies
        )
        
        # 2. 关联性评分
        correlation_score, supporting_correlations = self._score_correlations(
            service, correlations
        )
        
        # 3. 专家规则评分
        expert_rule_score, matched_rules = self._score_expert_rules(
            candidate, service, failure_type, anomalies
        )
        
        # 4. 时序评分
        temporal_score = self._score_temporal_patterns(
            service, anomalies
        )
        
        # 5. 业务影响评分
        business_impact_score = self._score_business_impact(
            service, failure_type
        )
        
        # 6. 综合评分
        total_score, confidence = self._calculate_comprehensive_score(
            anomaly_score, correlation_score, expert_rule_score, 
            temporal_score, business_impact_score, len(supporting_anomalies)
        )
        
        # 7. 生成推理链
        reasoning_chain = self._generate_reasoning_chain(
            candidate, anomaly_score, correlation_score, expert_rule_score,
            supporting_anomalies, matched_rules
        )
        
        # 8. 生成证据摘要
        evidence_summary = self._generate_evidence_summary(
            supporting_anomalies, supporting_correlations, matched_rules
        )
        
        return RootCauseScore(
            candidate=candidate,
            total_score=total_score,
            confidence=confidence,
            anomaly_score=anomaly_score,
            correlation_score=correlation_score,
            expert_rule_score=expert_rule_score,
            temporal_score=temporal_score,
            business_impact_score=business_impact_score,
            supporting_anomalies=supporting_anomalies,
            supporting_correlations=supporting_correlations,
            matched_rules=matched_rules,
            evidence_summary=evidence_summary,
            reasoning_chain=reasoning_chain
        )
    
    def _score_anomaly_match(self, candidate: str, service: str, failure_type: str, 
                            anomalies: List[Anomaly]) -> Tuple[float, List[Anomaly]]:
        """基于异常匹配度评分"""
        
        score = 0.0
        supporting_anomalies = []
        
        # 筛选相关服务的异常
        service_anomalies = [a for a in anomalies if a.service == service or a.service == 'multi-service']
        
        for anomaly in service_anomalies:
            match_score = 0.0
            
            # 根据故障类型和异常类型的匹配度评分
            if failure_type == "LargeGc":
                if anomaly.anomaly_type in [AnomalyType.MEMORY_LEAK, AnomalyType.GC_PRESSURE]:
                    match_score = 0.4
                elif anomaly.anomaly_type == AnomalyType.CPU_SPIKE:
                    match_score = 0.3  # CPU高通常伴随GC压力
            
            elif failure_type == "memory":
                if anomaly.anomaly_type == AnomalyType.MEMORY_LEAK:
                    match_score = 0.5
                elif anomaly.anomaly_type == AnomalyType.GC_PRESSURE:
                    match_score = 0.3
            
            elif failure_type == "cpu":
                if anomaly.anomaly_type == AnomalyType.CPU_SPIKE:
                    match_score = 0.5
            
            elif failure_type == "networkLatency":
                # 只有真正的网络延迟才匹配networkLatency候选根因
                if anomaly.anomaly_type == AnomalyType.NETWORK_LATENCY:
                    match_score = 0.5
            
            elif failure_type == "Failure":
                # 通用故障可以匹配多种异常类型
                if anomaly.anomaly_type == AnomalyType.ERROR_BURST:
                    match_score = 0.4
                elif anomaly.anomaly_type == AnomalyType.SERVICE_LATENCY:
                    # 服务延迟症状匹配服务故障候选根因
                    match_score = 0.4  
                elif anomaly.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                    match_score = 0.3
            
            # 考虑异常严重程度的加权
            severity_weight = {
                SeverityLevel.CRITICAL: 1.0,
                SeverityLevel.HIGH: 0.8,
                SeverityLevel.MEDIUM: 0.6,
                SeverityLevel.LOW: 0.3
            }
            
            match_score *= severity_weight[anomaly.severity]
            match_score *= anomaly.confidence  # 乘以异常检测置信度
            
            if match_score > 0.1:  # 只考虑有意义的匹配
                score += match_score
                supporting_anomalies.append(anomaly)
        
        # 归一化分数
        score = min(score, 1.0)
        
        return score, supporting_anomalies
    
    def _score_correlations(self, service: str, 
                           correlations: List[ServiceCorrelation]) -> Tuple[float, List[ServiceCorrelation]]:
        """基于服务关联性评分"""
        
        score = 0.0
        supporting_correlations = []
        
        for correlation in correlations:
            if service in [correlation.service_a, correlation.service_b]:
                # 关联强度评分
                correlation_strength = abs(correlation.correlation_coefficient)
                
                # 因果关系加权
                causal_weight = 1.2 if service in correlation.causal_direction else 1.0
                
                contribution = correlation_strength * correlation.confidence * causal_weight * 0.2
                score += contribution
                supporting_correlations.append(correlation)
        
        return min(score, 1.0), supporting_correlations
    
    def _score_expert_rules(self, candidate: str, service: str, failure_type: str,
                           anomalies: List[Anomaly]) -> Tuple[float, List[str]]:
        """基于专家规则评分"""
        
        score = 0.0
        matched_rules = []
        
        for rule in self.rules:
            if candidate in rule.target_candidates:
                # 检查规则是否匹配
                if self._check_rule_match(rule, service, anomalies):
                    rule_score = rule.base_score
                    
                    # 根据匹配的异常数量调整分数
                    matching_anomalies = self._count_matching_anomalies(rule, anomalies)
                    if matching_anomalies >= rule.min_anomaly_count:
                        rule_score *= rule.confidence_multiplier
                    
                    score += rule_score
                    matched_rules.append(rule.name)
        
        return min(score, 1.0), matched_rules
    
    def _score_temporal_patterns(self, service: str, anomalies: List[Anomaly]) -> float:
        """基于时序模式评分"""
        
        service_anomalies = [a for a in anomalies if a.service == service]
        
        if len(service_anomalies) <= 1:
            return 0.1  # 单个异常的时序分数较低
        
        # 异常持续性评分
        duration_score = 0.3 if len(service_anomalies) >= 3 else 0.1
        
        # 异常严重程度趋势评分
        severity_trend_score = 0.2 if any(a.severity == SeverityLevel.CRITICAL for a in service_anomalies) else 0.1
        
        return min(duration_score + severity_trend_score, 1.0)
    
    def _score_business_impact(self, service: str, failure_type: str) -> float:
        """基于业务影响评分"""
        
        # 获取服务关键程度
        service_info = self.service_topology.get(service, {'criticality': 'low'})
        criticality = service_info['criticality']
        
        # 关键程度评分
        criticality_scores = {
            'critical': 0.4,
            'high': 0.3,
            'medium': 0.2,
            'low': 0.1
        }
        
        # 故障类型影响评分
        failure_impact_scores = {
            'Failure': 0.3,      # 完全故障影响最大
            'LargeGc': 0.25,     # GC问题影响性能
            'memory': 0.2,       # 内存问题可能导致OOM
            'cpu': 0.15,         # CPU问题影响响应
            'networkLatency': 0.2 # 网络延迟影响用户体验
        }
        
        base_score = criticality_scores.get(criticality, 0.1)
        impact_score = failure_impact_scores.get(failure_type, 0.1)
        
        return base_score + impact_score
    
    def _calculate_comprehensive_score(self, anomaly_score: float, correlation_score: float,
                                     expert_rule_score: float, temporal_score: float,
                                     business_impact_score: float, anomaly_count: int) -> Tuple[float, float]:
        """计算综合评分和置信度"""
        
        # 加权综合评分
        weights = {
            'anomaly': 0.35,      # 异常匹配权重35%
            'expert_rules': 0.30, # 专家规则权重30%
            'correlation': 0.15,  # 关联分析权重15%
            'temporal': 0.10,     # 时序模式权重10%
            'business': 0.10      # 业务影响权重10%
        }
        
        total_score = (
            anomaly_score * weights['anomaly'] +
            expert_rule_score * weights['expert_rules'] +
            correlation_score * weights['correlation'] +
            temporal_score * weights['temporal'] +
            business_impact_score * weights['business']
        )
        
        # 置信度计算
        confidence_factors = [
            anomaly_score,
            expert_rule_score,
            min(anomaly_count / 5, 1.0),  # 异常数量因子
            1.0 if correlation_score > 0.3 else 0.5  # 关联性因子
        ]
        
        confidence = sum(confidence_factors) / len(confidence_factors)
        confidence = min(confidence, 1.0)
        
        return total_score, confidence
    
    def _generate_reasoning_chain(self, candidate: str, anomaly_score: float, 
                                correlation_score: float, expert_rule_score: float,
                                supporting_anomalies: List[Anomaly], 
                                matched_rules: List[str]) -> List[str]:
        """生成推理链"""
        
        reasoning = []
        service, failure_type = self._parse_candidate(candidate)
        
        reasoning.append(f"评估候选根因: {candidate}")
        
        # 异常证据
        if supporting_anomalies:
            reasoning.append(f"发现 {len(supporting_anomalies)} 个相关异常:")
            for i, anomaly in enumerate(supporting_anomalies[:3], 1):
                reasoning.append(f"  {i}. {anomaly.evidence} ({anomaly.severity.value})")
        
        # 专家规则匹配
        if matched_rules:
            reasoning.append(f"匹配专家规则: {', '.join(matched_rules)}")
        
        # 评分推理
        if anomaly_score > 0.5:
            reasoning.append(f"异常匹配度高 ({anomaly_score:.2f})")
        if expert_rule_score > 0.5:
            reasoning.append(f"专家规则强支持 ({expert_rule_score:.2f})")
        if correlation_score > 0.3:
            reasoning.append(f"服务关联性明显 ({correlation_score:.2f})")
        
        return reasoning
    
    def _generate_evidence_summary(self, supporting_anomalies: List[Anomaly],
                                 supporting_correlations: List[ServiceCorrelation],
                                 matched_rules: List[str]) -> List[str]:
        """生成证据摘要"""
        
        evidence = []
        
        # 异常证据
        if supporting_anomalies:
            critical_anomalies = [a for a in supporting_anomalies if a.severity == SeverityLevel.CRITICAL]
            high_anomalies = [a for a in supporting_anomalies if a.severity == SeverityLevel.HIGH]
            
            if critical_anomalies:
                evidence.append(f"发现 {len(critical_anomalies)} 个严重异常")
            if high_anomalies:
                evidence.append(f"发现 {len(high_anomalies)} 个高级异常")
        
        # 关联证据
        if supporting_correlations:
            strong_correlations = [c for c in supporting_correlations if abs(c.correlation_coefficient) > 0.7]
            if strong_correlations:
                evidence.append(f"与 {len(strong_correlations)} 个服务存在强关联")
        
        # 规则证据
        if matched_rules:
            evidence.append(f"匹配 {len(matched_rules)} 条专家规则")
        
        return evidence
    
    def _initialize_expert_rules(self) -> List[ExpertRule]:
        """初始化专家规则库"""
        
        rules = [
            # ad.LargeGc 专家规则
            ExpertRule(
                name="ad_large_gc_pattern",
                description="Ad服务大GC模式识别",
                target_candidates=["ad.LargeGc"],
                required_anomaly_types=[AnomalyType.MEMORY_LEAK, AnomalyType.GC_PRESSURE],
                required_services=["ad"],
                min_anomaly_count=2,
                min_severity=SeverityLevel.MEDIUM,
                base_score=0.4,
                confidence_multiplier=1.2,
                additional_conditions={
                    "memory_threshold": 0.85,
                    "gc_time_threshold": 1000  # ms
                },
                supporting_evidence=[
                    "Memory usage exceeds 85%",
                    "GC time over 1 second",
                    "CPU spike during GC events"
                ]
            ),
            
            # 内存泄漏模式
            ExpertRule(
                name="memory_leak_pattern",
                description="内存泄漏模式识别",
                target_candidates=["ad.memory", "cart.memory", "payment.memory"],
                required_anomaly_types=[AnomalyType.MEMORY_LEAK],
                required_services=[],  # 任意服务
                min_anomaly_count=1,
                min_severity=SeverityLevel.HIGH,
                base_score=0.35,
                confidence_multiplier=1.1,
                additional_conditions={},
                supporting_evidence=[
                    "Memory usage continuously increasing",
                    "No memory release after requests"
                ]
            ),
            
            # 网络超时模式
            ExpertRule(
                name="network_timeout_pattern", 
                description="网络超时级联故障模式",
                target_candidates=["payment.networkLatency", "checkout.networkLatency", "inventory.networkLatency"],
                required_anomaly_types=[AnomalyType.NETWORK_LATENCY, AnomalyType.SERVICE_LATENCY],
                required_services=[],
                min_anomaly_count=1,
                min_severity=SeverityLevel.MEDIUM,
                base_score=0.3,
                confidence_multiplier=1.0,
                additional_conditions={},
                supporting_evidence=[
                    "Network latency significantly increased",
                    "Timeout errors in logs"
                ]
            ),
            
            # 错误爆发模式
            ExpertRule(
                name="error_burst_pattern",
                description="错误爆发导致的服务故障",
                target_candidates=["ad.Failure", "cart.Failure", "payment.Failure", "checkout.Failure"],
                required_anomaly_types=[AnomalyType.ERROR_BURST],
                required_services=[],
                min_anomaly_count=1,
                min_severity=SeverityLevel.HIGH,
                base_score=0.35,
                confidence_multiplier=1.1,
                additional_conditions={},
                supporting_evidence=[
                    "Error rate significantly increased",
                    "Service availability degraded"
                ]
            ),
            
            # 复合故障模式 - CPU + Memory
            ExpertRule(
                name="cpu_memory_compound_failure",
                description="CPU和内存复合资源耗尽",
                target_candidates=["ad.LargeGc", "cart.Failure", "payment.Failure"],
                required_anomaly_types=[AnomalyType.CPU_SPIKE, AnomalyType.MEMORY_LEAK],
                required_services=[],
                min_anomaly_count=2,
                min_severity=SeverityLevel.MEDIUM,
                base_score=0.4,
                confidence_multiplier=1.3,  # 复合模式置信度更高
                additional_conditions={},
                supporting_evidence=[
                    "Both CPU and memory anomalies detected",
                    "Resource contention pattern"
                ]
            )
        ]
        
        return rules
    
    def _check_rule_match(self, rule: ExpertRule, service: str, anomalies: List[Anomaly]) -> bool:
        """检查规则是否匹配当前情况"""
        
        # 检查服务要求
        if rule.required_services and service not in rule.required_services:
            return False
        
        # 检查异常类型要求
        anomaly_types_found = [a.anomaly_type for a in anomalies if a.service == service]
        for required_type in rule.required_anomaly_types:
            if required_type not in anomaly_types_found:
                return False
        
        # 检查异常数量要求
        matching_anomalies = self._count_matching_anomalies(rule, anomalies)
        if matching_anomalies < rule.min_anomaly_count:
            return False
        
        # 检查严重程度要求
        severity_levels = [SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        
        # 获取该服务的所有异常严重程度索引
        service_anomalies = [a for a in anomalies if a.service == service]
        if not service_anomalies:
            return False
        
        # 找到最高严重程度的索引（使用索引比较而不是枚举值比较）
        max_severity_index = max(severity_levels.index(a.severity) for a in service_anomalies)
        min_required_index = severity_levels.index(rule.min_severity)
        
        if max_severity_index < min_required_index:
            return False
        
        return True
    
    def _count_matching_anomalies(self, rule: ExpertRule, anomalies: List[Anomaly]) -> int:
        """计算匹配规则的异常数量"""
        
        count = 0
        for anomaly in anomalies:
            if (not rule.required_services or anomaly.service in rule.required_services) and \
               anomaly.anomaly_type in rule.required_anomaly_types:
                count += 1
        
        return count
    
    def _parse_candidate(self, candidate: str) -> Tuple[str, str]:
        """解析候选根因字符串"""
        
        if '.' in candidate:
            service, failure_type = candidate.split('.', 1)
            return service, failure_type
        else:
            return candidate, "Failure"
    
    def _log_scoring_summary(self, scored_candidates: List[RootCauseScore]):
        """记录评分摘要"""
        
        self.logger.info("🏆 根因评分结果:")
        
        for i, score in enumerate(scored_candidates[:5], 1):  # 显示前5名
            self.logger.info(f"  {i}. {score.candidate}: {score.total_score:.3f} (置信度: {score.confidence:.3f})")
            self.logger.info(f"     - 异常匹配: {score.anomaly_score:.3f}")
            self.logger.info(f"     - 专家规则: {score.expert_rule_score:.3f}")
            self.logger.info(f"     - 服务关联: {score.correlation_score:.3f}")
            if score.matched_rules:
                self.logger.info(f"     - 匹配规则: {', '.join(score.matched_rules)}")
