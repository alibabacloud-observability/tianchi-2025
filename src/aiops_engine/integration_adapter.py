#!/usr/bin/env python
"""
集成适配器 - 连接AIOps引擎与现有LangGraph框架
实现新旧系统的平滑过渡和集成
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..syntax import create_analysis_class
from .aiops_rca_engine import AIOpsRCAEngine
from .expert_rules import RootCauseScore


class AIOpsIntegrationAdapter:
    """AIOps引擎集成适配器
    
    负责将AIOps引擎的结果转换为现有框架期望的格式
    """
    
    def __init__(self, debug: bool = False, offline_mode: bool = False, problem_id: str = None):
        self.debug = debug
        self.offline_mode = offline_mode
        self.problem_id = problem_id
        self.logger = logging.getLogger(__name__)
        
        # 初始化AIOps引擎
        try:
            self.aiops_engine = AIOpsRCAEngine(
                debug=debug, 
                offline_mode=offline_mode, 
                problem_id=problem_id
            )
            mode_desc = f"离线模式 (问题ID: {problem_id})" if offline_mode else "在线模式"
            self.logger.info(f"✅ AIOps集成适配器初始化完成 ({mode_desc})")
        except Exception as e:
            self.logger.error(f"❌ AIOps引擎初始化失败: {e}")
            raise
        
        # 配置参数
        self.config = {
            'min_confidence_threshold': 0.3,     # 最低置信度阈值
            'enable_fallback': True,             # 启用降级到原始框架
            'max_aiops_execution_time': 120,     # AIOps最大执行时间(秒)
            'result_count_limit': 5,             # 最多返回结果数
        }
    
    async def analyze(self, time_range: str, 
                     candidate_root_causes: List[str],
                     input_description: str = "",
                     problem_id: str = "unknown") -> Any:
        """根因分析接口 - 兼容enhanced_agent.py的调用
        
        Args:
            time_range: 时间范围字符串
            candidate_root_causes: 候选根因列表
            input_description: 输入描述（兼容现有接口）
            problem_id: 问题ID
            
        Returns:
            Analysis对象 (兼容现有框架的返回格式)
        """
        return await self.enhanced_root_cause_analysis(
            time_range=time_range,
            candidate_root_causes=candidate_root_causes,
            input_description=input_description,
            problem_id=problem_id
        )

    async def enhanced_root_cause_analysis(self, time_range: str, 
                                          candidate_root_causes: List[str],
                                          input_description: str = "",
                                          problem_id: str = "unknown") -> Any:
        """增强版根因分析 - 核心实现
        
        Args:
            time_range: 时间范围字符串
            candidate_root_causes: 候选根因列表
            input_description: 输入描述（兼容现有接口）
            problem_id: 问题ID
            
        Returns:
            Analysis对象 (兼容现有框架的返回格式)
        """
        
        self.logger.info(f"🚀 开始增强版根因分析")
        self.logger.info(f"   Problem ID: {problem_id}")
        self.logger.info(f"   时间范围: {time_range}")
        self.logger.info(f"   候选根因: {candidate_root_causes}")
        
        try:
            # 1. 使用AIOps引擎进行分析
            aiops_results = await asyncio.wait_for(
                self.aiops_engine.analyze_root_cause(time_range, candidate_root_causes),
                timeout=self.config['max_aiops_execution_time']
            )
            
            # 2. 检查结果质量
            if self._is_high_quality_result(aiops_results):
                self.logger.info("✅ AIOps引擎返回高质量结果")
                return self._convert_to_analysis_format(aiops_results, candidate_root_causes)
            
            else:
                self.logger.warning("⚠️ AIOps引擎结果质量不足，需要降级处理")
                if self.config['enable_fallback']:
                    return self._create_fallback_analysis(candidate_root_causes, aiops_results)
                else:
                    return self._convert_to_analysis_format(aiops_results, candidate_root_causes)
        
        except asyncio.TimeoutError:
            self.logger.error(f"❌ AIOps引擎执行超时 ({self.config['max_aiops_execution_time']}s)")
            if self.config['enable_fallback']:
                return self._create_timeout_fallback_analysis(candidate_root_causes)
            else:
                raise
        
        except Exception as e:
            self.logger.error(f"❌ AIOps引擎执行失败: {e}")
            if self.config['enable_fallback']:
                return self._create_error_fallback_analysis(candidate_root_causes, str(e))
            else:
                raise
    
    def _is_high_quality_result(self, results: List[RootCauseScore]) -> bool:
        """检查结果是否为高质量"""
        
        if not results:
            return False
        
        # 检查最佳结果的置信度
        best_result = results[0]
        if best_result.confidence < self.config['min_confidence_threshold']:
            return False
        
        # 检查是否有足够的支撑证据
        if len(best_result.supporting_anomalies) == 0 and len(best_result.matched_rules) == 0:
            return False
        
        # 检查评分的合理性
        if best_result.total_score < 0.2:
            return False
        
        return True
    
    def _convert_to_analysis_format(self, aiops_results: List[RootCauseScore], 
                                   candidate_root_causes: List[str]) -> Any:
        """将AIOps结果转换为现有框架的Analysis格式"""
        
        # 创建Analysis类
        Analysis = create_analysis_class(candidate_root_causes)
        
        if not aiops_results:
            # 空结果情况
            return Analysis(
                root_causes=candidate_root_causes[:1] if candidate_root_causes else [],
                evidences=["AIOps引擎未能找到确定的根因"],
                evidence_chain=self._create_minimal_evidence_chain()
            )
        
        # 取最佳结果
        best_result = aiops_results[0]
        
        # 构建根因列表（取前3个高置信度的）
        root_causes = []
        for result in aiops_results[:3]:
            if result.confidence >= self.config['min_confidence_threshold']:
                root_causes.append(result.candidate)
        
        if not root_causes:
            root_causes = [best_result.candidate]  # 至少返回一个
        
        # 构建证据列表
        evidences = self._build_evidences_from_aiops_results(aiops_results[:3])
        
        # 构建证据链
        evidence_chain = self._build_evidence_chain_from_aiops_results(best_result)
        
        self.logger.info(f"✅ 转换完成: 根因={root_causes}, 证据数={len(evidences)}")
        
        return Analysis(
            root_causes=root_causes,
            evidences=evidences,
            evidence_chain=evidence_chain
        )
    
    def _build_evidences_from_aiops_results(self, results: List[RootCauseScore]) -> List[str]:
        """从AIOps结果构建证据列表"""
        
        evidences = []
        
        for result in results:
            # 添加主要证据
            main_evidence = f"{result.candidate} (置信度: {result.confidence:.2f})"
            evidences.append(main_evidence)
            
            # 添加异常证据
            for anomaly in result.supporting_anomalies[:2]:  # 最多2个异常
                evidence = f"检测到{anomaly.service}服务{anomaly.anomaly_type.value}: {anomaly.evidence}"
                evidences.append(evidence)
            
            # 添加专家规则证据
            if result.matched_rules:
                rule_evidence = f"匹配专家规则: {', '.join(result.matched_rules[:2])}"
                evidences.append(rule_evidence)
        
        # 去重并限制数量
        unique_evidences = list(dict.fromkeys(evidences))  # 去重保持顺序
        return unique_evidences[:10]  # 最多10个证据
    
    def _build_evidence_chain_from_aiops_results(self, best_result: RootCauseScore) -> Any:
        """从最佳AIOps结果构建证据链"""
        
        from ..syntax import EvidenceChain, Action
        
        # 创建行动列表
        actions = []
        
        # 1. 数据收集行动
        actions.append(Action(
            tool_name="aiops_parallel_data_collection",
            tool_args={"method": "parallel", "data_sources": ["logs", "metrics", "traces"]},
            result_summary=f"并行收集多维度监控数据，发现{len(best_result.supporting_anomalies)}个异常"
        ))
        
        # 2. 异常检测行动
        if best_result.supporting_anomalies:
            actions.append(Action(
                tool_name="aiops_anomaly_detection",
                tool_args={"algorithm": "statistical_ml_fusion"},
                result_summary=f"检测到{len(best_result.supporting_anomalies)}个异常：{', '.join([a.anomaly_type.value for a in best_result.supporting_anomalies[:3]])}"
            ))
        
        # 3. 专家规则匹配行动
        if best_result.matched_rules:
            actions.append(Action(
                tool_name="aiops_expert_rules_matching",
                tool_args={"rules_count": len(best_result.matched_rules)},
                result_summary=f"匹配专家规则: {', '.join(best_result.matched_rules)}"
            ))
        
        # 4. 综合评分行动
        actions.append(Action(
            tool_name="aiops_comprehensive_scoring",
            tool_args={
                "anomaly_score": best_result.anomaly_score,
                "expert_score": best_result.expert_rule_score,
                "correlation_score": best_result.correlation_score
            },
            result_summary=f"综合评分: {best_result.total_score:.3f}, 置信度: {best_result.confidence:.3f}"
        ))
        
        # 构建证据链
        evidence_chain = EvidenceChain(
            motivation=["基于数据驱动的AIOps智能分析", "多维度异常检测和专家规则融合"],
            actions=actions,
            observations=best_result.evidence_summary + best_result.reasoning_chain[:3],
            decision=[f"推荐根因: {best_result.candidate}", f"置信度: {best_result.confidence:.2f}"]
        )
        
        return evidence_chain
    
    def _create_minimal_evidence_chain(self) -> Any:
        """创建最小证据链"""
        
        from ..syntax import EvidenceChain, Action
        
        return EvidenceChain(
            motivation=["AIOps引擎分析"],
            actions=[Action(
                tool_name="aiops_analysis",
                tool_args={"status": "no_clear_result"},
                result_summary="未能找到明确的根因证据"
            )],
            observations=["数据不足或异常不明显"],
            decision=["需要更多信息或手动分析"]
        )
    
    def _create_fallback_analysis(self, candidate_root_causes: List[str], 
                                 aiops_results: List[RootCauseScore]) -> Any:
        """创建降级分析结果"""
        
        Analysis = create_analysis_class(candidate_root_causes)
        
        # 使用置信度最高的结果，但标记为低置信度
        if aiops_results:
            best_candidate = aiops_results[0].candidate
            evidences = [
                f"AIOps分析结果置信度较低 ({aiops_results[0].confidence:.2f})",
                "建议结合人工分析进行验证",
                f"检测到{len(aiops_results[0].supporting_anomalies)}个相关异常"
            ]
        else:
            best_candidate = candidate_root_causes[0] if candidate_root_causes else "unknown"
            evidences = ["AIOps引擎未能确定根因", "建议人工排查"]
        
        return Analysis(
            root_causes=[best_candidate],
            evidences=evidences,
            evidence_chain=self._create_minimal_evidence_chain()
        )
    
    def _create_timeout_fallback_analysis(self, candidate_root_causes: List[str]) -> Any:
        """创建超时降级分析结果"""
        
        Analysis = create_analysis_class(candidate_root_causes)
        
        return Analysis(
            root_causes=[candidate_root_causes[0]] if candidate_root_causes else ["unknown"],
            evidences=[
                f"AIOps引擎分析超时 (>{self.config['max_aiops_execution_time']}s)",
                "建议缩小时间范围或减少候选根因数量重试",
                "或使用传统分析方法"
            ],
            evidence_chain=self._create_minimal_evidence_chain()
        )
    
    def _create_error_fallback_analysis(self, candidate_root_causes: List[str], 
                                       error_message: str) -> Any:
        """创建错误降级分析结果"""
        
        Analysis = create_analysis_class(candidate_root_causes)
        
        return Analysis(
            root_causes=[candidate_root_causes[0]] if candidate_root_causes else ["unknown"],
            evidences=[
                f"AIOps引擎分析失败: {error_message}",
                "建议检查数据源连接和配置",
                "或使用传统分析方法"
            ],
            evidence_chain=self._create_minimal_evidence_chain()
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        
        return {
            'aiops_engine_stats': self.aiops_engine.get_performance_report(),
            'integration_config': self.config,
            'status': 'active'
        }
