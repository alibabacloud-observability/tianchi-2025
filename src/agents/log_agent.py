"""
Real Log Agent for RCA Data Collection - FIXED VERSION
修复了日志无法下载的问题，使用验证有效的apm.log.agent_info配置
"""
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from tools.paas_data_tools import umodel_get_logs
from ..utils.evidence_chain import EvidenceChain


class MinimalLogAgent:
    """修复后的最小日志代理 - 专注于有效的配置，支持离线模式"""
    
    def __init__(self, debug: bool = False, offline_mode: bool = False, problem_id: str = None):
        self.debug = debug
        self.offline_mode = offline_mode
        self.problem_id = problem_id
        self.logger = logging.getLogger(__name__)
        
        if offline_mode:
            # 离线模式：使用本地数据加载器
            from ..utils.local_data_loader import get_local_data_loader
            self.local_loader = get_local_data_loader(debug=debug)
            self.logger.info(f"✅ MinimalLogAgent初始化完成 (离线模式, 问题ID: {problem_id})")
        else:
            
            # 使用排查验证的有效配置 (移除其他无效的log_sets)
            self.effective_log_config = {
                'domain': 'apm',
                'entity_set_name': 'apm.service',
                'log_set_name': 'apm.log.agent_info',  # 唯一有效的log_set
                'log_set_domain': 'apm'
            }
            
            self.logger.info(f"✅ MinimalLogAgent初始化完成 (在线模式)")
            if self.debug:
                self.logger.info(f"🔧 使用有效配置: {self.effective_log_config}")
    
    def _fetch_mcp_log_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """使用排查验证的有效配置获取日志数据"""
        try:
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(end_time.timestamp())
            
            self.logger.info(f"🔍 获取日志数据 - 使用修复后的配置")
            self.logger.info(f"   时间范围: {start_time} ~ {end_time}")
            
            # 直接调用LangChain工具
            result = umodel_get_logs.invoke({
                "domain": self.effective_log_config['domain'],
                "entity_set_name": self.effective_log_config['entity_set_name'],
                "log_set_name": self.effective_log_config['log_set_name'],
                "log_set_domain": self.effective_log_config['log_set_domain'],
                "from_time": start_timestamp,
                "to_time": end_timestamp
            })
            
            logs = result.data if result and not result.error else []
            
            if logs:
                self.logger.info(f"✅ 成功获取 {len(logs)} 条日志")
                
                # 统计服务分布（基于实际字段结构）
                services = set()
                log_types = set()
                for log in logs:
                    if isinstance(log, dict):
                        if 'service_name' in log and log['service_name']:
                            services.add(log['service_name'])
                        if 'log_type' in log and log['log_type']:
                            log_types.add(log['log_type'])
                
                if services:
                    self.logger.info(f"📊 涉及服务 ({len(services)}个): {sorted(services)}")
                else:
                    self.logger.warning(f"⚠️ 无法从日志中提取服务信息")
                
                if log_types:
                    self.logger.info(f"📊 日志类型: {sorted(log_types)}")
                
                return logs
            else:
                self.logger.warning("⚠️ 日志获取结果为空")
                return []
            
        except Exception as e:
            self.logger.error(f"❌ 获取日志数据失败: {e}")
            return []
    
    def _parse_log_response(self, result: Dict) -> List[Dict[str, Any]]:
        """解析MCP日志响应 - 优化版本"""
        logs = []
        
        if 'content' not in result:
            self.logger.warning("⚠️ MCP响应中无content字段")
            return logs
        
        content = result['content']
        if not isinstance(content, list):
            self.logger.warning(f"⚠️ content格式异常: {type(content)}")
            return logs
        
        self.logger.debug(f"🔍 解析 {len(content)} 个content项")
        
        for i, item in enumerate(content):
            if isinstance(item, dict) and item.get('type') == 'text':
                try:
                    text_content = item.get('text', '')
                    if not text_content.strip():
                        continue
                        
                    # 解析JSON内容
                    text_data = json.loads(text_content)
                    
                    if isinstance(text_data, dict) and 'data' in text_data:
                        log_data = text_data['data']
                        
                        if isinstance(log_data, list):
                            # 处理日志数组
                            for log_entry in log_data:
                                if isinstance(log_entry, dict):
                                    enhanced_log = self._enhance_log_entry(log_entry)
                                    logs.append(enhanced_log)
                        elif isinstance(log_data, dict):
                            # 处理单个日志对象
                            enhanced_log = self._enhance_log_entry(log_data)
                            logs.append(enhanced_log)
                            
                        self.logger.debug(f"✅ 解析content项 [{i}]: 获得 {len(log_data) if isinstance(log_data, list) else 1} 条日志")
                        
                except json.JSONDecodeError as e:
                    self.logger.debug(f"⚠️ JSON解析失败 [{i}]: {e}")
                    # 保留无法解析的原始文本
                    raw_text = item.get('text', '').strip()
                    if raw_text:
                        logs.append({
                            'raw_log_text': raw_text,
                            'parse_status': 'failed',
                            'content_item_index': i
                        })
            else:
                self.logger.debug(f"⚠️ 跳过非文本项 [{i}]: {item.get('type', 'unknown')}")
        
        return logs
    
    def _enhance_log_entry(self, log_entry: Dict) -> Dict[str, Any]:
        """增强日志条目 - 标准化字段和添加元信息"""
        enhanced = log_entry.copy()
        
        # 确保关键字段存在
        enhanced['service_name'] = enhanced.get('service_name', 'unknown')
        enhanced['log_type'] = enhanced.get('log_type', 'agent_info')
        
        # 清理和标准化字段
        for field in ['language', 'version', 'source']:
            if field in enhanced:
                value = enhanced[field]
                if value == 'null' or value is None:
                    enhanced[field] = 'unknown'
        
        # 添加解析元信息
        enhanced['_parsed_timestamp'] = datetime.now().isoformat()
        enhanced['_source_config'] = 'apm.log.agent_info'
        enhanced['_parsing_version'] = 'fixed_v1.0'
        
        return enhanced
    
    def _fetch_log_data(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """根据模式获取日志数据"""
        if self.offline_mode:
            # 离线模式：从本地文件加载数据
            if not self.problem_id:
                self.logger.error("❌ 离线模式需要指定problem_id")
                return []
            
            # 判断是故障期还是基线期数据
            # 这里简单判断：如果时间段较短（<=5分钟）认为是故障期，否则是基线期
            duration_minutes = (end_time - start_time).total_seconds() / 60
            data_type = "failure" if duration_minutes <= 10 else "baseline"
            
            self.logger.info(f"📂 从本地加载 {self.problem_id} {data_type} 日志数据")
            return self.local_loader.load_logs(self.problem_id, data_type)
        else:
            # 在线模式：从MCP服务获取数据  
            return self._fetch_mcp_log_data(start_time, end_time)
    
    def analyze(self, evidence_chain: EvidenceChain) -> Dict[str, Any]:
        """分析日志数据 - 支持离线模式"""
        
        mode_desc = "离线模式" if self.offline_mode else "在线模式"
        self.logger.info(f"🔍 开始日志分析 ({mode_desc})")
        
        # 获取日志数据 - 根据模式选择数据源
        log_data = self._fetch_log_data(evidence_chain.start_time, evidence_chain.end_time)
        
        # 执行日志分析
        analysis_result = self._analyze_logs(log_data, evidence_chain.start_time, evidence_chain.end_time)
        
        # 添加到证据链
        confidence = 0.9 if len(log_data) > 0 else 0.1
        source_desc = f"{mode_desc}_log_analysis"
        evidence_chain.add_evidence(
            'log', 
            source_desc, 
            log_data,  # 保存原始日志数据到证据链
            confidence=confidence
        )
        
        self.logger.info(f"✅ 日志分析完成 ({mode_desc})")
        self.logger.info(f"   日志总数: {len(log_data)}")
        self.logger.info(f"   服务数量: {len(analysis_result.get('services', []))}")
        self.logger.info(f"   置信度: {confidence}")
        
        return analysis_result
    
    def _analyze_logs(self, logs: List[Dict], start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """分析日志数据 - 提取关键信息"""
        
        # 初始化统计
        services = set()
        log_types = set()
        languages = set()
        versions = set()
        
        service_distribution = {}
        parse_stats = {'successful': 0, 'failed': 0}
        
        for log in logs:
            if isinstance(log, dict):
                # 统计解析状态
                if log.get('parse_status') == 'failed':
                    parse_stats['failed'] += 1
                else:
                    parse_stats['successful'] += 1
                
                # 提取服务信息
                service = log.get('service_name', 'unknown')
                if service != 'unknown':
                    services.add(service)
                    service_distribution[service] = service_distribution.get(service, 0) + 1
                
                # 提取其他字段
                log_types.add(log.get('log_type', 'unknown'))
                languages.add(log.get('language', 'unknown'))
                versions.add(log.get('version', 'unknown'))
        
        # 构建分析结果
        analysis = {
            'summary': {
                'total_logs': len(logs),
                'unique_services': len(services),
                'time_range': f"{start_time.isoformat()} ~ {end_time.isoformat()}",
                'collection_method': 'fixed_apm.log.agent_info',
                'parsing_success_rate': (parse_stats['successful'] / len(logs) * 100) if logs else 0
            },
            'services': sorted(services),
            'service_distribution': service_distribution,
            'log_types': sorted(log_types),
            'languages': sorted(languages),
            'versions': sorted(versions),
            'parse_statistics': parse_stats,
            'sample_logs': logs[:5] if logs else [],  # 保留前5条作为样本
            'raw_data_available': len(logs) > 0
        }
        
        return analysis
