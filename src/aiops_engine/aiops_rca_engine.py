#!/usr/bin/env python
"""
AIOps根因分析引擎 - 方案A渐进式优化的完整集成
结合并行数据获取、异常检测、专家规则的智能决策引擎
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from .parallel_data_coordinator import ParallelDataCoordinator, DataBundle
from .anomaly_detection import AnomalyDetectionEngine, CorrelationAnalysisEngine, Anomaly, ServiceCorrelation
from .expert_rules import ExpertRulesEngine, RootCauseScore


class AIOpsRCAEngine:
    """AIOps根因分析引擎
    
    方案A的完整实现：数据驱动 + 算法智能 + LLM辅助
    支持离线模式和在线模式
    """
    
    def __init__(self, debug: bool = False, offline_mode: bool = False, problem_id: str = None):
        self.debug = debug
        self.offline_mode = offline_mode
        self.problem_id = problem_id
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个组件
        try:
            # 数据协调器需要传递离线模式参数
            self.data_coordinator = ParallelDataCoordinator(
                debug=debug, offline_mode=offline_mode, problem_id=problem_id
            )
            
            # 其他组件不需要离线模式参数（它们处理的是已收集的数据）
            self.anomaly_detector = AnomalyDetectionEngine(debug=debug)
            self.correlation_analyzer = CorrelationAnalysisEngine(debug=debug)
            self.expert_rules = ExpertRulesEngine(debug=debug)
            
            mode_desc = f"离线模式 (问题ID: {problem_id})" if offline_mode else "在线模式"
            self.logger.info(f"✅ AIOps RCA引擎初始化完成 ({mode_desc})")
            
        except Exception as e:
            self.logger.error(f"❌ AIOps RCA引擎初始化失败: {e}")
            raise
        
        # 性能统计
        self.performance_stats = {
            'total_analyses': 0,
            'successful_analyses': 0,
            'avg_execution_time': 0.0,
            'avg_data_quality_score': 0.0,
            'avg_confidence_score': 0.0
        }
    
    async def analyze_root_cause(self, time_range: str, 
                                candidates: List[str]) -> List[RootCauseScore]:
        """执行完整的根因分析
        
        Args:
            time_range: 时间范围，如 "2025-08-28 15:08:03 ~ 2025-08-28 15:13:03"
            candidates: 候选根因，如 ["ad.Failure", "ad.LargeGc", "cart.Failure"]
            
        Returns:
            List[RootCauseScore]: 按置信度排序的根因评分结果
        """
        
        start_time = time.time()
        analysis_id = f"RCA_{int(start_time)}"
        
        self.logger.info(f"🚀 开始AIOps根因分析 [{analysis_id}]")
        self.logger.info(f"   📅 时间范围: {time_range}")
        self.logger.info(f"   🎯 候选根因: {candidates}")
        
        try:
            # 1. 并行数据收集 (A1层)
            self.logger.info("🔄 Phase 1: 并行数据收集")
            data_bundle = await self.data_coordinator.collect_comprehensive_data(time_range, candidates)
            
            data_summary = self.data_coordinator.get_data_summary(data_bundle)
            self.logger.info(f"   📊 数据质量: {data_bundle.data_quality_score:.2f}/1.0")
            self.logger.info(f"   🏢 发现服务: {data_summary['services_discovered']}")
            
            # 2. 异常检测分析 (A2层)
            self.logger.info("🔄 Phase 2: 异常检测分析")
            anomalies = self.anomaly_detector.detect_all_anomalies(data_bundle)
            
            # 3. 服务关联分析
            self.logger.info("🔄 Phase 3: 服务关联分析")
            correlations = self.correlation_analyzer.analyze_service_correlations(anomalies, data_bundle)
            
            # 4. 专家规则评分 (A2层)
            self.logger.info("🔄 Phase 4: 专家规则评分")
            scored_results = self.expert_rules.evaluate_candidates(
                candidates, anomalies, correlations, data_bundle
            )
            
            # 5. 结果优化和验证
            optimized_results = self._optimize_results(scored_results, data_bundle)
            
            # 6. 记录性能统计
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time, data_bundle.data_quality_score, optimized_results)
            
            self.logger.info(f"✅ AIOps根因分析完成 [{analysis_id}]")
            self.logger.info(f"   ⏱️  总耗时: {execution_time:.2f}秒")
            self.logger.info(f"   🎯 最佳候选: {optimized_results[0].candidate if optimized_results else 'None'}")
            self.logger.info(f"   📈 置信度: {optimized_results[0].confidence:.3f} if optimized_results else 0")
            
            return optimized_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ AIOps根因分析失败 [{analysis_id}]: {e}")
            self.logger.error(f"   ⏱️  执行时间: {execution_time:.2f}秒")
            
            # 返回空结果或基于部分数据的降级结果
            return self._create_fallback_results(candidates, str(e))
    
    def _optimize_results(self, results: List[RootCauseScore], 
                         data_bundle: DataBundle) -> List[RootCauseScore]:
        """优化和验证分析结果"""
        
        if not results:
            return results
        
        # 1. 置信度阈值过滤
        min_confidence = 0.3
        filtered_results = [r for r in results if r.confidence >= min_confidence]
        
        if not filtered_results:
            # 如果所有结果置信度都太低，保留原始结果但标记低置信度
            self.logger.warning(f"⚠️ 所有候选根因置信度低于阈值 {min_confidence}")
            return results
        
        # 2. 数据质量调整
        data_quality_factor = max(data_bundle.data_quality_score, 0.5)  # 最低0.5的质量因子
        
        for result in filtered_results:
            # 根据数据质量调整置信度
            result.confidence *= data_quality_factor
            result.total_score *= data_quality_factor
        
        # 3. 重新排序
        filtered_results.sort(key=lambda x: (x.confidence, x.total_score), reverse=True)
        
        # 4. 添加决策解释
        for i, result in enumerate(filtered_results[:3]):  # 前3名添加额外解释
            result.reasoning_chain.append(f"经数据质量调整后排名第 {i+1}")
            if result.confidence > 0.8:
                result.reasoning_chain.append("高置信度推荐")
            elif result.confidence > 0.6:
                result.reasoning_chain.append("中等置信度，建议进一步验证")
            else:
                result.reasoning_chain.append("低置信度，仅供参考")
        
        return filtered_results
    
    def _create_fallback_results(self, candidates: List[str], error_message: str) -> List[RootCauseScore]:
        """创建降级结果（分析失败时）"""
        
        fallback_results = []
        
        for candidate in candidates:
            # 基于候选名称的简单启发式评分
            base_score = 0.1  # 很低的基础分数
            
            # 根据常见故障模式给予一些分数
            if "Failure" in candidate:
                base_score += 0.2
            if "LargeGc" in candidate:
                base_score += 0.15
            
            result = RootCauseScore(
                candidate=candidate,
                total_score=base_score,
                confidence=0.2,  # 很低的置信度
                anomaly_score=0.0,
                correlation_score=0.0,
                expert_rule_score=0.0,
                temporal_score=0.0,
                business_impact_score=base_score,
                supporting_anomalies=[],
                supporting_correlations=[],
                matched_rules=[],
                evidence_summary=[f"分析失败，降级结果: {error_message}"],
                reasoning_chain=[
                    f"AIOps引擎分析失败: {error_message}",
                    "使用基础启发式评分",
                    "建议手动排查或重试分析"
                ]
            )
            
            fallback_results.append(result)
        
        # 简单排序
        fallback_results.sort(key=lambda x: x.total_score, reverse=True)
        
        return fallback_results
    
    def _update_performance_stats(self, execution_time: float, 
                                 data_quality_score: float,
                                 results: List[RootCauseScore]):
        """更新性能统计"""
        
        self.performance_stats['total_analyses'] += 1
        
        if results and results[0].confidence > 0.5:
            self.performance_stats['successful_analyses'] += 1
        
        # 滑动平均更新
        alpha = 0.1  # 平滑因子
        self.performance_stats['avg_execution_time'] = (
            alpha * execution_time + (1 - alpha) * self.performance_stats['avg_execution_time']
        )
        
        self.performance_stats['avg_data_quality_score'] = (
            alpha * data_quality_score + (1 - alpha) * self.performance_stats['avg_data_quality_score']
        )
        
        if results:
            avg_confidence = sum(r.confidence for r in results) / len(results)
            self.performance_stats['avg_confidence_score'] = (
                alpha * avg_confidence + (1 - alpha) * self.performance_stats['avg_confidence_score']
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        
        stats = self.performance_stats.copy()
        
        if stats['total_analyses'] > 0:
            stats['success_rate'] = stats['successful_analyses'] / stats['total_analyses']
        else:
            stats['success_rate'] = 0.0
        
        # 计算性能等级
        performance_grade = self._calculate_performance_grade(stats)
        stats['performance_grade'] = performance_grade
        
        # 添加改进建议
        stats['improvement_suggestions'] = self._generate_improvement_suggestions(stats)
        
        return stats
    
    def _calculate_performance_grade(self, stats: Dict[str, Any]) -> str:
        """计算性能等级"""
        
        # 综合评分
        score = 0
        
        # 成功率评分 (40%)
        success_rate = stats.get('success_rate', 0)
        score += success_rate * 40
        
        # 执行时间评分 (30%)
        avg_time = stats.get('avg_execution_time', 120)  # 默认120秒
        if avg_time <= 30:
            score += 30
        elif avg_time <= 60:
            score += 25
        elif avg_time <= 90:
            score += 20
        else:
            score += 10
        
        # 数据质量评分 (20%)
        data_quality = stats.get('avg_data_quality_score', 0.5)
        score += data_quality * 20
        
        # 置信度评分 (10%)
        confidence = stats.get('avg_confidence_score', 0.5)
        score += confidence * 10
        
        # 等级划分
        if score >= 90:
            return "A+"
        elif score >= 80:
            return "A"
        elif score >= 70:
            return "B+"
        elif score >= 60:
            return "B"
        elif score >= 50:
            return "C"
        else:
            return "D"
    
    def _generate_improvement_suggestions(self, stats: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        
        suggestions = []
        
        success_rate = stats.get('success_rate', 0)
        avg_time = stats.get('avg_execution_time', 0)
        data_quality = stats.get('avg_data_quality_score', 0)
        confidence = stats.get('avg_confidence_score', 0)
        
        if success_rate < 0.8:
            suggestions.append("提高分析成功率：检查数据源连接稳定性")
        
        if avg_time > 60:
            suggestions.append("优化执行时间：考虑增加并行度或缓存机制")
        
        if data_quality < 0.7:
            suggestions.append("改善数据质量：完善数据源配置和清洗逻辑")
        
        if confidence < 0.6:
            suggestions.append("提升置信度：丰富专家规则库和异常检测算法")
        
        if not suggestions:
            suggestions.append("性能表现良好，继续保持")
        
        return suggestions
    
    async def batch_analyze(self, analysis_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量分析多个问题"""
        
        self.logger.info(f"🔄 开始批量分析 {len(analysis_requests)} 个问题")
        
        # 创建并行任务
        tasks = []
        for i, request in enumerate(analysis_requests):
            task = asyncio.create_task(
                self._single_batch_analysis(i, request)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch_results.append({
                    'problem_id': analysis_requests[i].get('problem_id', f'problem_{i}'),
                    'status': 'failed',
                    'error': str(result),
                    'results': []
                })
            else:
                batch_results.append(result)
        
        self.logger.info(f"✅ 批量分析完成")
        
        return batch_results
    
    async def _single_batch_analysis(self, index: int, request: Dict[str, Any]) -> Dict[str, Any]:
        """单个批量分析任务"""
        
        try:
            time_range = request['time_range']
            candidates = request['candidates']
            problem_id = request.get('problem_id', f'problem_{index}')
            
            results = await self.analyze_root_cause(time_range, candidates)
            
            return {
                'problem_id': problem_id,
                'status': 'success',
                'results': [
                    {
                        'candidate': r.candidate,
                        'total_score': r.total_score,
                        'confidence': r.confidence,
                        'evidence_summary': r.evidence_summary,
                        'top_reasoning': r.reasoning_chain[:3]  # 前3条推理
                    }
                    for r in results[:5]  # 前5个结果
                ]
            }
            
        except Exception as e:
            return {
                'problem_id': request.get('problem_id', f'problem_{index}'),
                'status': 'failed', 
                'error': str(e),
                'results': []
            }
