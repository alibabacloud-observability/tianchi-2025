#!/usr/bin/env python3
"""
Enhanced Agent - 直接基于异常检测结果进行智能RCA分析
避免复杂的专家规则，让LLM直接推理异常数据
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Any, Dict
import json

from src.react_agent import compute_anomaly_based_rca
from src.aiops_engine.anomaly_detection import AnomalyDetectionEngine
from src.aiops_engine.aiops_rca_engine import AIOpsRCAEngine
from src.syntax import ProblemData

logger = logging.getLogger(__name__)


async def enhanced_compute_plan_execute(
    time_range: str,
    candidate_root_causes: List[str],
    input_description: str = "Investigate system failure and identify root causes",
    problem_id: str = "unknown",
    use_aiops: bool = True,
    debug: bool = False,
    offline_mode: bool = False,
    use_react_agent: bool = True  # 新参数：是否使用基于异常的ReactAgent
) -> Any:
    """Enhanced计算分析 - 支持传统AIOps和基于异常的ReactAgent两种模式
    
    Args:
        time_range: 时间范围
        candidate_root_causes: 候选根因列表
        input_description: 问题描述
        problem_id: 问题ID
        use_aiops: 是否使用AIOps引擎
        debug: 是否调试模式
        offline_mode: 是否离线模式
        use_react_agent: 是否使用基于异常的ReactAgent
        
    Returns:
        分析结果
    """
    
    logger = logging.getLogger(__name__)
    
    mode_desc = "离线模式" if offline_mode else "在线模式"
    rca_mode = "ReactAgent异常分析" if use_react_agent else ("AIOps引擎" if use_aiops else "传统方法")
    
    logger.info(f"🚀 Enhanced计算分析开始")
    logger.info(f"   RCA方式: {rca_mode}")
    logger.info(f"   Data Mode: {mode_desc}")
    logger.info(f"   Problem: {problem_id}")
    logger.info(f"   Time Range: {time_range}")
    logger.info(f"   Candidates: {candidate_root_causes}")
    
    try:
        if use_react_agent:
            # 新模式：基于异常的ReactAgent分析
            return await enhanced_anomaly_based_rca(
                time_range=time_range,
                candidate_root_causes=candidate_root_causes,
                problem_id=problem_id,
                debug=debug,
                offline_mode=offline_mode
            )
        
        elif use_aiops:
            # 传统AIOps引擎模式
            from src.aiops_engine.integration_adapter import AIOpsIntegrationAdapter
            
            adapter = AIOpsIntegrationAdapter(debug=debug, offline_mode=offline_mode, problem_id=problem_id)
            analysis = await adapter.analyze(
                time_range=time_range,
                candidate_root_causes=candidate_root_causes,
                input_description=input_description,
                problem_id=problem_id
            )
            
            logger.info("✅ AIOps分析完成")
            return analysis
        
        else:
            # 传统计划执行方法
            from src.agent import compute_plan_execute
            
            return await compute_plan_execute(
                time_range=time_range,
                candidate_root_causes=candidate_root_causes,
                input_description=input_description,
                problem_id=problem_id
            )
            
    except Exception as e:
        logger.error(f"❌ Enhanced分析失败: {str(e)}")
        raise


async def enhanced_anomaly_based_rca(
    time_range: str,
    candidate_root_causes: List[str],
    problem_id: str = "unknown",
    debug: bool = False,
    offline_mode: bool = False
) -> Any:
    """基于异常检测的智能RCA分析
    
    核心流程：
    1. 使用异常检测引擎获取所有异常
    2. 将异常数据直接输入到ReactAgent
    3. 让LLM基于异常进行智能推理，避免复杂规则
    
    Args:
        time_range: 时间范围
        candidate_root_causes: 候选根因列表
        problem_id: 问题ID
        debug: 调试模式
        offline_mode: 离线模式
        
    Returns:
        分析结果
    """
    
    logger.info(f"🧠 开始基于异常的智能RCA分析")
    logger.info(f"   问题ID: {problem_id}")
    logger.info(f"   时间范围: {time_range}")
    
    try:
        # Step 1: 解析时间范围
        start_time_str, end_time_str = time_range.split(' ~ ')
        start_time = datetime.strptime(start_time_str.strip(), "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time_str.strip(), "%Y-%m-%d %H:%M:%S")
        
        # Step 2: 运行异常检测
        logger.info("🔍 Step 1: 运行异常检测...")
        
        # 创建AIOps引擎仅用于数据收集和异常检测
        aiops_engine = AIOpsRCAEngine(debug=debug, offline_mode=offline_mode, problem_id=problem_id)
        
        # 收集数据
        data_bundle = await aiops_engine.data_coordinator.collect_comprehensive_data(
            time_range=time_range,
            candidates=candidate_root_causes
        )
        
        # 执行异常检测
        anomalies = aiops_engine.anomaly_detector.detect_all_anomalies(data_bundle)
        
        logger.info(f"✅ 异常检测完成: 发现 {len(anomalies)} 个异常")
        
        # Step 3: 检查异常数量
        if not anomalies:
            logger.warning("⚠️  未检测到任何异常，返回空的分析结果")
            return {
                "root_causes": [],
                "evidences": ["未检测到系统异常，可能系统运行正常或数据不足"],
                "analysis_method": "enhanced_anomaly_based_rca",
                "anomaly_count": 0,
                "confidence": 0.0
            }
        
        # Step 4: 使用ReactAgent进行智能分析
        logger.info("🧠 Step 2: 使用ReactAgent进行智能分析...")
        
        # 使用新的基于异常的RCA分析
        analysis = await compute_anomaly_based_rca(
            anomalies=anomalies,
            time_range=time_range,
            candidate_root_causes=candidate_root_causes,
            problem_id=problem_id
        )
        
        logger.info("✅ 基于异常的智能RCA分析完成")
        logger.info(f"🎯 识别根因: {analysis['root_causes']}")
        
        return analysis
        
    except Exception as e:
        logger.error(f"❌ 基于异常的智能RCA分析失败: {str(e)}")
        raise


# 为了保持向后兼容性，保留原有函数签名
async def enhanced_compute_plan_execute_legacy(
    time_range: str,
    candidate_root_causes: List[str], 
    input_description: str = "Investigate system failure and identify root causes",
    problem_id: str = "unknown",
    use_aiops: bool = True,
    debug: bool = False,
    offline_mode: bool = False
) -> Any:
    """Legacy接口，调用新的enhanced_compute_plan_execute"""
    return await enhanced_compute_plan_execute(
        time_range=time_range,
        candidate_root_causes=candidate_root_causes,
        input_description=input_description,
        problem_id=problem_id,
        use_aiops=use_aiops,
        debug=debug,
        offline_mode=offline_mode,
        use_react_agent=True  # 默认使用新的ReactAgent模式
    )


if __name__ == "__main__":
    # 测试示例
    async def test_enhanced_rca():
        """测试基于异常的智能RCA分析"""
        
        time_range = "2025-08-28 16:14:30 ~ 2025-08-28 16:19:30"
        candidate_root_causes = [
            "ad.Failure", "ad.cpu", "ad.memory", "ad.networkLatency",
            "cart.Failure", "cart.cpu", "checkout.cpu", "checkout.Failure"
        ]
        
        try:
            result = await enhanced_compute_plan_execute(
                time_range=time_range,
                candidate_root_causes=candidate_root_causes,
                problem_id="test_005",
                use_react_agent=True,  # 使用新的基于异常的分析
                debug=True,
                offline_mode=False
            )
            
            print(f"🎯 分析结果: {result.root_causes}")
            print(f"📄 证据: {result.evidences}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 运行测试
    asyncio.run(test_enhanced_rca())
