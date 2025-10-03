#!/usr/bin/env python
"""
A1: 并行数据获取协调器 - 基于现有三个agents实现
统一协调 log_agent, metric_agent, trace_agent 并行工作
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time

# 导入现有的三个专业agents
from ..agents.log_agent import MinimalLogAgent
from ..agents.metric_agent import MinimalMetricAgent  
from ..agents.trace_agent import MinimalTraceAgent
from ..utils.evidence_chain import EvidenceChain


@dataclass
class DataBundle:
    """统一的数据包格式"""
    # 原始数据
    logs: List[Dict[str, Any]]
    k8s_metrics: Dict[str, List[float]]
    apm_metrics: Dict[str, List[float]]  
    traces: List[Dict[str, Any]]
    
    # 元信息
    time_range: str
    start_time: datetime
    end_time: datetime
    baseline_start: datetime
    baseline_end: datetime
    
    # 统计信息
    collection_stats: Dict[str, Any]
    data_quality_score: float
    
    # 基线数据（用于异常检测）
    baseline_logs: List[Dict[str, Any]] = None
    baseline_k8s_metrics: Dict[str, List[float]] = None
    baseline_apm_metrics: Dict[str, List[float]] = None


@dataclass
class CollectionTask:
    """数据收集任务"""
    name: str
    agent_type: str  # 'log', 'metric', 'trace'
    start_time: datetime
    end_time: datetime
    expected_duration_seconds: int
    status: str = 'pending'  # pending, running, completed, failed
    result: Any = None
    error_message: str = None
    execution_time: float = 0.0


class ParallelDataCoordinator:
    """A1层：并行数据获取协调器
    
    基于现有的三个专业agents，实现并行数据采集和统一协调
    支持离线模式和在线模式
    """
    
    def __init__(self, debug: bool = False, offline_mode: bool = False, problem_id: str = None):
        self.debug = debug
        self.offline_mode = offline_mode
        self.problem_id = problem_id
        self.logger = logging.getLogger(__name__)
        
        # 初始化三个专业agents - 根据模式传递参数
        try:
            if offline_mode:
                self.log_agent = MinimalLogAgent(debug=debug, offline_mode=True, problem_id=problem_id)
                self.metric_agent = MinimalMetricAgent(debug=debug, offline_mode=True, problem_id=problem_id)
                self.trace_agent = MinimalTraceAgent(debug=debug, offline_mode=True, problem_id=problem_id)
                
                mode_desc = f"离线模式 (问题ID: {problem_id})"
            else:
                self.log_agent = MinimalLogAgent(debug=debug)
                self.metric_agent = MinimalMetricAgent(debug=debug)
                self.trace_agent = MinimalTraceAgent(debug=debug)
                
                mode_desc = "在线模式"
                
            self.logger.info(f"✅ 三个数据采集agents初始化完成 ({mode_desc})")
        except Exception as e:
            self.logger.error(f"❌ Agents初始化失败: {e}")
            raise
        
        # 配置参数
        self.config = {
            'baseline_hours_before': 0.15,      # 基线期：故障前2小时
            'baseline_buffer_minutes': 1,   # 基线缓冲：故障前10分钟
            'timeout_seconds': 120,          # 单个agent超时时间
            'parallel_timeout_seconds': 150, # 并行总超时时间
            'retry_attempts': 2,             # 重试次数
        }
    
    async def collect_comprehensive_data(self, time_range: str, candidates: List[str]) -> DataBundle:
        """并行收集所有类型的监控数据
        
        Args:
            time_range: 时间范围，格式 "2025-08-28 15:08:03 ~ 2025-08-28 15:13:03"
            candidates: 候选根因列表，如 ["ad.Failure", "ad.LargeGc"]
            
        Returns:
            DataBundle: 统一格式的数据包
        """
        
        start_total = time.time()
        self.logger.info(f"🚀 开始并行数据收集")
        self.logger.info(f"   📅 时间范围: {time_range}")
        self.logger.info(f"   🎯 候选根因: {candidates}")
        
        # 1. 解析时间范围
        start_time, end_time = self._parse_time_range(time_range)
        baseline_start, baseline_end = self._calculate_baseline_period(start_time)
        
        # 2. 创建并行收集任务
        tasks = [
            # 故障期数据收集
            CollectionTask('failure_logs', 'log', start_time, end_time, 30),
            CollectionTask('failure_metrics', 'metric', start_time, end_time, 45),
            CollectionTask('failure_traces', 'trace', start_time, end_time, 20),
            
            # 基线期数据收集  
            CollectionTask('baseline_logs', 'log', baseline_start, baseline_end, 25),
            CollectionTask('baseline_metrics', 'metric', baseline_start, baseline_end, 40),
        ]
        
        # 3. 并行执行所有任务
        results = await self._execute_parallel_tasks(tasks)
        
        # 4. 整理和验证数据
        data_bundle = self._assemble_data_bundle(
            results, time_range, start_time, end_time, 
            baseline_start, baseline_end
        )
        
        # 5. 计算数据质量评分
        data_bundle.data_quality_score = self._calculate_data_quality_score(data_bundle)
        
        total_time = time.time() - start_total
        self.logger.info(f"✅ 并行数据收集完成")
        self.logger.info(f"   ⏱️  总耗时: {total_time:.2f}秒")
        self.logger.info(f"   📊 数据质量评分: {data_bundle.data_quality_score:.2f}/1.0")
        self.logger.info(f"   📈 性能提升: 预计比串行快 {self._estimate_speedup(tasks, total_time):.1f}倍")
        
        return data_bundle
    
    async def _execute_parallel_tasks(self, tasks: List[CollectionTask]) -> Dict[str, CollectionTask]:
        """并行执行所有数据收集任务"""
        
        self.logger.info(f"🔄 并行执行 {len(tasks)} 个数据收集任务")
        
        # 创建异步任务
        async_tasks = []
        for task in tasks:
            async_task = asyncio.create_task(
                self._execute_single_task(task)
            )
            async_tasks.append(async_task)
        
        # 并行等待所有任务完成
        try:
            completed_tasks = await asyncio.wait_for(
                asyncio.gather(*async_tasks, return_exceptions=True),
                timeout=self.config['parallel_timeout_seconds']
            )
            
            # 整理结果
            results = {}
            for i, result in enumerate(completed_tasks):
                task = tasks[i]
                if isinstance(result, Exception):
                    task.status = 'failed'
                    task.error_message = str(result)
                    self.logger.error(f"❌ 任务 {task.name} 失败: {result}")
                else:
                    task.status = 'completed'
                    self.logger.info(f"✅ 任务 {task.name} 完成: {task.execution_time:.1f}s")
                
                results[task.name] = task
            
            return results
            
        except asyncio.TimeoutError:
            self.logger.error(f"❌ 并行任务超时 ({self.config['parallel_timeout_seconds']}s)")
            # 返回部分完成的结果
            results = {}
            for task in tasks:
                if task.status not in ['completed', 'failed']:
                    task.status = 'timeout'
                results[task.name] = task
            return results
    
    async def _execute_single_task(self, task: CollectionTask) -> CollectionTask:
        """执行单个数据收集任务"""
        
        start_time = time.time()
        task.status = 'running'
        
        try:
            # 创建临时证据链
            evidence_chain = EvidenceChain(task.start_time, task.end_time)
            
            # 根据agent类型执行对应任务
            if task.agent_type == 'log':
                result = self.log_agent.analyze(evidence_chain)
                # 从证据链中提取日志数据
                log_evidence = None
                for evidence in evidence_chain.evidence:
                    if evidence.evidence_type == 'log':
                        log_evidence = evidence.data
                        break
                task.result = {
                    'analysis': result,
                    'raw_logs': log_evidence or []
                }
                
            elif task.agent_type == 'metric':
                result = self.metric_agent.analyze(evidence_chain)
                # 从证据链中提取指标数据
                metric_evidence = None
                for evidence in evidence_chain.evidence:
                    if evidence.evidence_type == 'metric':
                        metric_evidence = evidence.data
                        break
                task.result = {
                    'analysis': result,
                    'metrics_data': metric_evidence or {}
                }
                
            elif task.agent_type == 'trace':
                result = self.trace_agent.analyze(evidence_chain)
                # 从证据链中提取链路数据
                trace_evidence = None
                for evidence in evidence_chain.evidence:
                    if evidence.evidence_type == 'trace':
                        trace_evidence = evidence.data
                        break
                task.result = {
                    'analysis': result,
                    'trace_data': trace_evidence or []
                }
            
            task.execution_time = time.time() - start_time
            task.status = 'completed'
            
        except Exception as e:
            task.execution_time = time.time() - start_time
            task.status = 'failed'
            task.error_message = str(e)
            self.logger.error(f"❌ 任务 {task.name} 执行失败: {e}")
        
        return task
    
    def _assemble_data_bundle(self, results: Dict[str, CollectionTask], 
                             time_range: str, start_time: datetime, end_time: datetime,
                             baseline_start: datetime, baseline_end: datetime) -> DataBundle:
        """组装数据包"""
        
        # 提取故障期数据
        logs = []
        k8s_metrics = {}
        apm_metrics = {}
        traces = []
        
        # 提取基线期数据
        baseline_logs = []
        baseline_k8s_metrics = {}
        baseline_apm_metrics = {}
        
        # 统计信息
        collection_stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_timeout': 0,
            'total_execution_time': 0,
            'data_sources': {
                'logs': False,
                'k8s_metrics': False,
                'apm_metrics': False,
                'traces': False
            }
        }
        
        for task_name, task in results.items():
            collection_stats['total_execution_time'] += task.execution_time
            
            if task.status == 'completed':
                collection_stats['tasks_completed'] += 1
                
                if 'logs' in task_name:
                    if task.result and 'raw_logs' in task.result:
                        if 'failure' in task_name:
                            logs = task.result['raw_logs']
                            collection_stats['data_sources']['logs'] = True
                        elif 'baseline' in task_name:
                            baseline_logs = task.result['raw_logs']
                
                elif 'metrics' in task_name:
                    if task.result and 'metrics_data' in task.result:
                        metrics_data = task.result['metrics_data']
                        if 'failure' in task_name:
                            k8s_metrics = metrics_data.get('k8s_golden_metrics', {})
                            apm_metrics = metrics_data.get('apm_service_metrics', {})
                            collection_stats['data_sources']['k8s_metrics'] = bool(k8s_metrics)
                            collection_stats['data_sources']['apm_metrics'] = bool(apm_metrics)
                        elif 'baseline' in task_name:
                            baseline_k8s_metrics = metrics_data.get('k8s_golden_metrics', {})
                            baseline_apm_metrics = metrics_data.get('apm_service_metrics', {})
                
                elif 'traces' in task_name:
                    if task.result and 'trace_data' in task.result:
                        traces = task.result['trace_data']
                        collection_stats['data_sources']['traces'] = bool(traces)
            
            elif task.status == 'failed':
                collection_stats['tasks_failed'] += 1
            elif task.status == 'timeout':
                collection_stats['tasks_timeout'] += 1
        
        return DataBundle(
            logs=logs,
            k8s_metrics=k8s_metrics,
            apm_metrics=apm_metrics,
            traces=traces,
            time_range=time_range,
            start_time=start_time,
            end_time=end_time,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            baseline_logs=baseline_logs,
            baseline_k8s_metrics=baseline_k8s_metrics,
            baseline_apm_metrics=baseline_apm_metrics,
            collection_stats=collection_stats,
            data_quality_score=0.0  # Will be calculated later
        )
    
    def _calculate_data_quality_score(self, data_bundle: DataBundle) -> float:
        """计算数据质量评分 (0.0 - 1.0)"""
        
        score = 0.0
        max_score = 1.0
        
        # 数据完整性评分 (40%)
        completeness_score = 0.0
        data_sources = data_bundle.collection_stats['data_sources']
        
        if data_sources['logs']:
            completeness_score += 0.1
        if data_sources['k8s_metrics']:
            completeness_score += 0.15
        if data_sources['apm_metrics']:
            completeness_score += 0.1
        if data_sources['traces']:
            completeness_score += 0.05
        
        # 任务成功率评分 (30%)
        stats = data_bundle.collection_stats
        total_tasks = stats['tasks_completed'] + stats['tasks_failed'] + stats['tasks_timeout']
        success_rate = stats['tasks_completed'] / total_tasks if total_tasks > 0 else 0
        success_score = success_rate * 0.3
        
        # 数据量评分 (20%)
        volume_score = 0.0
        if len(data_bundle.logs) > 0:
            volume_score += 0.05
        if len(data_bundle.k8s_metrics) > 0:
            volume_score += 0.08
        if len(data_bundle.apm_metrics) > 0:
            volume_score += 0.05
        if len(data_bundle.traces) > 0:
            volume_score += 0.02
        
        # 基线数据可用性评分 (10%)
        baseline_score = 0.0
        if data_bundle.baseline_logs:
            baseline_score += 0.03
        if data_bundle.baseline_k8s_metrics:
            baseline_score += 0.04
        if data_bundle.baseline_apm_metrics:
            baseline_score += 0.03
        
        total_score = completeness_score + success_score + volume_score + baseline_score
        return min(total_score, 1.0)
    
    def _parse_time_range(self, time_range: str) -> Tuple[datetime, datetime]:
        """解析时间范围字符串"""
        try:
            start_str, end_str = time_range.split(' ~ ')
            start_time = datetime.strptime(start_str, '%Y-%m-%d %H:%M:%S')
            end_time = datetime.strptime(end_str, '%Y-%m-%d %H:%M:%S')
            return start_time, end_time
        except Exception as e:
            self.logger.error(f"❌ 时间范围解析失败: {e}")
            raise ValueError(f"Invalid time range format: {time_range}")
    
    def _calculate_baseline_period(self, failure_start: datetime) -> Tuple[datetime, datetime]:
        """计算基线期时间范围
        
        逻辑：获取故障前一段时间的正常基线数据
        - baseline_end: 故障前buffer_minutes分钟（避免故障前兆影响）
        - baseline_start: 再往前推baseline_hours_before小时的数据作为基线窗口
        """
        # 故障前10分钟作为基线结束时间（避免故障前兆）
        baseline_end = failure_start - timedelta(minutes=self.config['baseline_buffer_minutes'])
        
        # 从基线结束时间往前推0.15小时(9分钟)作为基线窗口
        baseline_start = baseline_end - timedelta(hours=self.config['baseline_hours_before'])
        
        return baseline_start, baseline_end
    
    def _estimate_speedup(self, tasks: List[CollectionTask], actual_total_time: float) -> float:
        """估算相对于串行执行的加速比"""
        serial_time = sum(task.expected_duration_seconds for task in tasks)
        return serial_time / actual_total_time if actual_total_time > 0 else 1.0
    
    def get_data_summary(self, data_bundle: DataBundle) -> Dict[str, Any]:
        """获取数据摘要"""
        
        # 提取服务列表
        services = set()
        
        # 从日志中提取服务
        for log in data_bundle.logs:
            if isinstance(log, dict) and 'service_name' in log:
                services.add(log['service_name'])
        
        # 从指标中提取服务
        for metric_name in data_bundle.apm_metrics.keys():
            if ':' in metric_name:
                service = metric_name.split(':')[0]
                services.add(service)
        
        # 从链路中提取服务
        for trace in data_bundle.traces:
            if isinstance(trace, dict) and 'service_name' in trace:
                services.add(trace['service_name'])
        
        return {
            'data_quality_score': data_bundle.data_quality_score,
            'services_discovered': sorted(list(services)),
            'data_counts': {
                'logs': len(data_bundle.logs),
                'k8s_metrics': len(data_bundle.k8s_metrics),
                'apm_metrics': len(data_bundle.apm_metrics),
                'traces': len(data_bundle.traces)
            },
            'baseline_available': {
                'logs': bool(data_bundle.baseline_logs),
                'k8s_metrics': bool(data_bundle.baseline_k8s_metrics),
                'apm_metrics': bool(data_bundle.baseline_apm_metrics)
            },
            'collection_performance': {
                'total_time': data_bundle.collection_stats['total_execution_time'],
                'tasks_completed': data_bundle.collection_stats['tasks_completed'],
                'tasks_failed': data_bundle.collection_stats['tasks_failed']
            }
        }
