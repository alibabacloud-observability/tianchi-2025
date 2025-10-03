#!/usr/bin/env python
"""
AIOps RCA运行脚本
基于调试好的agents运行完整的根因分析
"""

import asyncio
import logging
import json
import sys
import os
import argparse
from datetime import datetime
from typing import List

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.enhanced_agent import enhanced_compute_plan_execute
from src.simple_tracing import rca_problem_agent

# 配置日志 - 同时输出到控制台和文件
def setup_logging():
    """设置日志配置，同时输出到控制台和文件"""
    
    # 创建logs目录
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    log_filename = f"aiops_rca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 1. 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. 文件处理器
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # 文件中保存更详细的日志
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print(f"📝 日志将同时输出到控制台和文件: {log_filepath}")
    return log_filepath

# 设置日志
log_file_path = setup_logging()

def _extract_result_data(result):
    """
    格式适配函数：统一处理ReactAgent和AIOps引擎的不同输出格式
    
    Args:
        result: ReactAgent的字典格式或AIOps引擎的Analysis对象格式
        
    Returns:
        tuple: (root_causes, evidences)
    """
    
    if result is None:
        return [], []
    
    # 检测结果类型并使用相应的访问方式
    if isinstance(result, dict):
        # ReactAgent格式：字典访问
        root_causes = result.get('root_causes', [])
        evidences = result.get('evidences', [])
        
        if log_file_path:  # 避免在测试时的日志问题
            print(f"🔧 检测到ReactAgent字典格式")
            
    elif hasattr(result, 'root_causes') and hasattr(result, 'evidences'):
        # AIOps引擎格式：Analysis对象访问
        root_causes = result.root_causes if result.root_causes else []
        evidences = result.evidences if result.evidences else []
        
        if log_file_path:  # 避免在测试时的日志问题
            print(f"🔧 检测到AIOps引擎Analysis对象格式")
            
    else:
        # 未知格式，尝试通用处理
        print(f"⚠️ 未知结果格式: {type(result)}")
        
        # 尝试字典访问
        try:
            root_causes = result.get('root_causes', []) if hasattr(result, 'get') else []
            evidences = result.get('evidences', []) if hasattr(result, 'get') else []
        except:
            # 尝试对象属性访问
            try:
                root_causes = getattr(result, 'root_causes', [])
                evidences = getattr(result, 'evidences', [])
            except:
                # 最后兜底
                root_causes = []
                evidences = [f"无法解析结果格式: {type(result)}"]
    
    # 确保返回的是列表类型
    if not isinstance(root_causes, list):
        root_causes = [root_causes] if root_causes else []
    if not isinstance(evidences, list):
        evidences = [evidences] if evidences else []
    
    return root_causes, evidences

async def run_single_rca(problem_id: str, time_range: str, candidates: List[str], description: str = "", 
                        offline_mode: bool = False, use_react_agent: bool = True):
    """运行单个RCA分析
    
    Args:
        problem_id: 问题ID
        time_range: 时间范围
        candidates: 候选根因列表
        description: 问题描述
        offline_mode: 是否使用离线模式
        use_react_agent: True=ReactAgent模式, False=传统AIOps引擎模式
    """
    
    mode_desc = "离线模式" if offline_mode else "在线模式"
    rca_method = "ReactAgent智能推理" if use_react_agent else "传统AIOps专家规则"
    
    print(f"\n🎯 开始RCA分析: {problem_id} ({mode_desc})")
    print(f"   📅 时间范围: {time_range}")
    print(f"   🔍 候选根因: {len(candidates)}个")
    print(f"   📝 问题描述: {description}")
    print(f"   🧠 分析方法: {rca_method}")
    if offline_mode:
        print(f"   💾 数据源: 离线缓存文件 (data/problem_{problem_id}/)")
    print("-" * 50)
    
    try:
        start_time = datetime.now()
        
        # Create RCA agent following ARMS documentation pattern
        async with rca_problem_agent(
            problem_id=problem_id,
            time_range=time_range,
            candidates=candidates,
            input_description=description or f"RCA analysis for problem {problem_id}",
            use_aiops=True,  # 启用数据收集和异常检测
            use_react_agent=use_react_agent,  # 🔑 关键参数：选择分析方法
            debug=True,
            offline_mode=offline_mode  # 传递离线模式参数
        ) as agent:
            # Execute the analysis using the agent (like agent.ainvoke() in docs)
            result = await agent.analyze()
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            print(f"\n✅ RCA分析完成 ({execution_time:.2f}秒)")
            
            # 🔧 格式适配：检测结果类型并使用正确的访问方式
            root_causes, evidences = _extract_result_data(result)
            
            print(f"🎯 推荐根因: {root_causes}")
            print(f"📊 证据数量: {len(evidences)}")
            
            # 显示详细结果
            print("\n📋 详细分析结果:")
            for i, root_cause in enumerate(root_causes[:5], 1):
                print(f"   {i}. {root_cause}")
            
            if evidences:
                print(f"\n🔍 关键证据:")
                for i, evidence in enumerate(evidences[:3], 1):
                    print(f"   {i}. {evidence[:100]}..." if len(evidence) > 100 else f"   {i}. {evidence}")
            
            # 保存结果
            result_file = f"results/rca_result_{problem_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            
            # Get log file path for the current session
            log_file_path = os.path.join(project_root, f"logs/aiops_rca_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            
            result_data = {
                'problem_id': problem_id,
                'time_range': time_range,
                'execution_time': execution_time,
                'root_causes': root_causes,
                'evidences': evidences,
                'analysis_timestamp': end_time.isoformat(),
                'log_file_path': log_file_path
            }
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 结果已保存至: {result_file}")
            print(f"📝 完整日志已保存至: {log_file_path}")
            
            return result
        
    except Exception as e:
        print(f"❌ RCA分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

async def run_batch_rca(problems_file: str, offline_mode: bool = False, use_react_agent: bool = True, single_problem_id: str = None):
    """批量运行RCA分析
    
    Args:
        problems_file: 问题文件路径
        offline_mode: 是否使用离线模式
        use_react_agent: True=ReactAgent模式, False=传统AIOps引擎模式
        single_problem_id: 只运行指定的问题ID (可选)
    """
    
    mode_desc = "离线模式" if offline_mode else "在线模式"
    rca_method = "ReactAgent智能推理" if use_react_agent else "传统AIOps专家规则"
    
    print("🚀 批量RCA分析")
    print(f"📊 数据模式: {mode_desc}")
    print(f"🧠 分析方法: {rca_method}")
    print("=" * 60)
    
    try:
        with open(problems_file, 'r', encoding='utf-8') as f:
            problems = []
            for line in f:
                if line.strip():
                    problems.append(json.loads(line.strip()))
        
        # 如果指定了单个问题ID，过滤出该问题
        if single_problem_id:
            problems = [p for p in problems if p.get('problem_id') == single_problem_id]
            if not problems:
                print(f"❌ 未找到问题ID: {single_problem_id}")
                return
            print(f"🎯 只运行单个问题: {single_problem_id}")
        else:
            print(f"📋 加载了 {len(problems)} 个问题")
        
        successful_count = 0
        
        for i, problem in enumerate(problems, 1):
            print(f"\n📍 处理问题 {i}/{len(problems)}")
            
            result = await run_single_rca(
                problem_id=problem['problem_id'],
                time_range=problem['time_range'], 
                candidates=problem['candidate_root_causes'],
                description=f"Problem {problem['problem_id']} - {problem.get('alarm_rules', [])}",
                offline_mode=offline_mode,
                use_react_agent=use_react_agent  # 传递分析方法参数
            )
            
            if result:
                successful_count += 1
        
        print(f"\n🎉 批量分析完成!")
        print(f"✅ 成功: {successful_count}/{len(problems)} ({successful_count/len(problems)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ 批量分析失败: {e}")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='AIOps RCA运行脚本 - 基于调试好的agents运行完整的根因分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🔧 模式说明:
  在线模式  : 实时从MCP服务器获取监控数据 (默认)
  离线模式 : 从预下载的本地文件加载数据

🧠 分析方法说明:
  ReactAgent模式 : 基于异常检测+LLM智能推理 (默认)
  AIOps引擎模式  : 基于异常检测+专家规则评分 (--aiops-engine)

💡 推荐工作流程:
  1. 首先运行数据下载脚本：
     python scripts/download_data.py
  2. 然后使用离线模式分析：
     python run_aiops_rca.py dataset/problems.jsonl --offline-mode

⚡ 离线模式优势:
  - 分析速度更快 (无网络延迟)
  - 数据一致性更好 (避免实时查询差异)
  - 支持离线分析

示例:
  python run_aiops_rca.py dataset/B榜题目.jsonl
  python run_aiops_rca.py dataset/problems.jsonl --offline-mode
  python run_aiops_rca.py dataset/problems.jsonl --offline-mode --aiops-engine
  python run_aiops_rca.py dataset/problems.jsonl --offline-mode --single-problem 004
"""
    )
    
    parser.add_argument(
        'problems_file',
        type=str,
        help='问题文件路径，如 dataset/B榜题目.jsonl'
    )
    
    parser.add_argument(
        '--offline-mode',
        action='store_true',
        dest='offline_mode',
        help='使用离线缓存数据而非实时在线查询'
    )
    
    parser.add_argument(
        '--aiops-engine',
        action='store_true',
        dest='use_aiops_engine',
        help='使用传统AIOps专家规则引擎 (默认使用ReactAgent)'
    )
    
    parser.add_argument(
        '--single-problem',
        type=str,
        dest='single_problem_id',
        help='只运行指定的问题ID (例如: 004)'
    )
    
    return parser.parse_args()

async def main():
    """主函数"""
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查文件是否存在
    if not os.path.exists(args.problems_file):
        print(f"❌ 文件不存在: {args.problems_file}")
        return
    
    # 设置分析方法
    use_react_agent = not args.use_aiops_engine
    
    # 显示配置信息
    mode_desc = "离线模式" if args.offline_mode else "在线模式"
    rca_method = "ReactAgent智能推理" if use_react_agent else "传统AIOps专家规则"
    print(f"📊 数据模式: {mode_desc}")
    print(f"🧠 分析方法: {rca_method}")
    print(f"📁 问题文件: {args.problems_file}")
    if args.single_problem_id:
        print(f"🎯 单个问题: {args.single_problem_id}")
    print("=" * 60)
    
    # 执行批量分析
    await run_batch_rca(args.problems_file, args.offline_mode, use_react_agent, args.single_problem_id)

if __name__ == "__main__":
    asyncio.run(main())
