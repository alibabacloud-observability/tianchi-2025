"""
Minimal MCP Client for RCA Data Collection
"""
import requests
import json
import logging
from typing import Dict, List, Any, Optional


class MinimalMCPClient:
    """Minimal MCP Client for log/trace/metrics data collection."""
    
    def __init__(self, debug: bool = False):
        # MCP server configuration
        self.server_url = "http://8.153.195.170:8080/mcp"
        self.user_id = "1819385687343877"
        self.workspace = "tianchi-2025"
        self.region_id = "cn-qingdao"
        
        # Session state
        self.session_id = None
        self.initialized = False
        
        # HTTP headers
        self.headers = {
            "X-User-ID": self.user_id,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Add properties expected by agent code
        self.workspace = self.workspace
        self.region_id = self.region_id
        
    def _get_session_id(self) -> bool:
        """Get session ID using OPTIONS request."""
        try:
            response = requests.options(self.server_url, headers=self.headers, timeout=10)
            session_id = response.headers.get('mcp-session-id')
            
            if session_id:
                self.session_id = session_id
                self.headers['mcp-session-id'] = session_id
                if self.debug:
                    self.logger.info(f"✅ Session ID obtained: {session_id[:16]}...")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to get session ID: {e}")
            return False
    
    def _initialize_connection(self) -> bool:
        """Initialize MCP connection with complete handshake."""
        if not self.session_id and not self._get_session_id():
            return False
        
        try:
            # 1. Initialize请求
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "clientInfo": {
                        "name": "minimal-rca-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = requests.post(self.server_url, headers=self.headers, 
                                   json=init_request, timeout=30)
            
            if response.status_code != 200:
                return False
                
            # 2. 🔑 关键修复：发送initialized通知完成握手
            notify_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            requests.post(self.server_url, headers=self.headers,
                         json=notify_request, timeout=10)
            
            self.initialized = True
            if self.debug:
                self.logger.info("✅ MCP connection initialized with complete handshake")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize connection: {e}")
            return False
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Public method to call MCP tool."""
        return self._call_tool(tool_name, arguments)
    
    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call MCP tool."""
        if not self.initialized and not self._initialize_connection():
            return None
        
        try:
            tool_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            response = requests.post(self.server_url, headers=self.headers,
                                   json=tool_request, timeout=30)
            
            if response.status_code == 200:
                try:
                    # Handle SSE format response
                    response_text = response.text.strip()
                    if response_text.startswith("event: message"):
                        # Extract JSON from SSE format
                        lines = response_text.split('\n')
                        for line in lines:
                            if line.startswith("data: "):
                                json_data = line[6:]  # Remove "data: " prefix
                                result = json.loads(json_data)
                                # 🔑 关键修复：提取实际结果而不是整个JSON-RPC响应
                                if 'result' in result:
                                    return result['result']
                                elif 'error' in result:
                                    if self.debug:
                                        self.logger.error(f"MCP tool error: {result['error']}")
                                    return result  # 返回错误信息供调用方处理
                                return result
                    else:
                        # Normal JSON response
                        result = response.json()
                        # 🔑 关键修复：提取实际结果而不是整个JSON-RPC响应
                        if 'result' in result:
                            return result['result']
                        elif 'error' in result:
                            if self.debug:
                                self.logger.error(f"MCP tool error: {result['error']}")
                            return result  # 返回错误信息供调用方处理
                        return result
                except (json.JSONDecodeError, Exception) as e:
                    if self.debug:
                        self.logger.error(f"Failed to parse response: {e}")
                        self.logger.error(f"Response text: {response.text[:200]}")
                    return None
            else:
                if self.debug:
                    self.logger.error(f"HTTP {response.status_code}: {response.text[:200]}")
                return None
        except Exception as e:
            self.logger.error(f"❌ Tool call failed: {e}")
            return None
    
    def get_logs(self, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """Get logs from MCP server."""
        result = self._call_tool("umodel_get_logs", {
            "domain": "apm",
            "entity_set_name": "apm.service",
            "log_set_name": "apm.log.agent_info",
            "log_set_domain": "apm",
            "workspace": self.workspace,
            "regionId": self.region_id,
            "from_time": start_time,
            "to_time": end_time,
            "limit": 50
        })
        if self.debug:
            self.logger.info(f"🔍 get_logs 原始响应类型: {type(result)}")
            if result:
                self.logger.info(f"🔍 get_logs 响应字段: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    
        # 🔑 修复：根据实际响应结构提取数据
        if result and isinstance(result, dict):
            if not "structuredContent" in result:
                self.logger.info(f"structuredContent 字段 not found")
                return []
            if not "result" in result['structuredContent']:
                self.logger.info(f"result 字段 not found")
                return []

            if not "data" in result['structuredContent']['result']:
                self.logger.info(f"data 字段 not found")
                return []

            logs = result['structuredContent']['result']['data']
            
            if self.debug:
                self.logger.info(f"✅ get_logs 成功提取 {len(logs)} 条日志")
                if logs:
                    first_log = logs[0]
                    self.logger.info(f"🔑 首条日志字段: {list(first_log.keys()) if isinstance(first_log, dict) else 'N/A'}")
            return logs

        
        if self.debug:
            self.logger.warning("⚠️ get_logs 未获取到有效数据")
        return []
    
    def get_traces(self, start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """Get traces from MCP server."""
        # First search for trace IDs
        search_result = self._call_tool("umodel_search_traces", {
            "domain": "apm", 
            "entity_set_name": "apm.service",
            "trace_set_domain": "apm",
            "trace_set_name": "apm.trace",
            "workspace": self.workspace,
            "from_time": start_time,
            "to_time": end_time,
            "limit": 50
        })
        
        traces = []
        if search_result and "result" in search_result:
            trace_ids = search_result["result"].get("trace_ids", [])
            
            # Get detailed traces
            for trace_id in trace_ids[:20]:  # Limit to 20 traces
                trace_result = self._call_tool("umodel_get_traces", {
                    "domain": "apm",
                    "entity_set_name": "apm.service", 
                    "trace_ids": [trace_id],
                    "trace_set_domain": "apm",
                    "trace_set_name": "apm.trace",
                    "workspace": self.workspace
                })
                
                if trace_result and "result" in trace_result:
                    spans = trace_result["result"].get("spans", [])
                    traces.extend(spans)
        
        return traces
    
    def get_metrics(self, entity_ids: List[str], metric_name: str, 
                   start_time: int, end_time: int) -> List[Dict[str, Any]]:
        """Get metrics from MCP server."""
        result = self._call_tool("umodel_search_entities", {
            "domain": "k8s",
            "entity_set_name": "k8s.pod",
            "workspace": self.workspace,
            "search_text": "",
            "from_time": start_time,
            "to_time": end_time,
            "limit": 100
        })
        
        if result and "result" in result:
            return result["result"].get("entities", [])
        return []

    # 兼容性方法 - 与原有接口保持一致
    def initialize_connection(self) -> bool:
        """Public method to initialize connection."""
        return self._initialize_connection()
    
    def get_stats(self) -> Dict[str, int]:
        """Get client statistics for compatibility."""
        return {
            "total_requests": 0,
            "successful_requests": 0 if not self.initialized else 1,
            "failed_requests": 0,
            "logs_retrieved": 0
        }
