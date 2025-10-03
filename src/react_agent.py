#!/usr/bin/env python

from contextlib import asynccontextmanager
import logging as std_logging
import os
import random
import sys
import threading
import traceback
from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
# from langchain_core.messages.utils import count_tokens_approximately  # Not available in this version

def count_tokens_approximately(messages) -> int:
    """简单的token数量估算"""
    if not messages:
        return 0
    
    total_chars = 0
    for message in messages:
        if hasattr(message, 'content') and message.content:
            total_chars += len(str(message.content))
        elif isinstance(message, str):
            total_chars += len(message)
    
    # 粗略估算：平均每个token约4个字符
    return total_chars // 4
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langmem.short_term import SummarizationNode
from pydantic import BaseModel, Field, validator

from src.tools import provide_diagnostic_response, create_analysis_tool
from src.config import model, tracer
from src.syntax import create_analysis_class
from src.prompts import create_prompt
from src.aiops_engine.anomaly_detection import Anomaly, AnomalyType, SeverityLevel
import numpy as np


class State(AgentState):
    context: Dict[str, Any]


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif hasattr(obj, '__dict__'):  # Handle dataclass objects like Anomaly
        return convert_numpy_types(obj.__dict__)
    else:
        return obj


def sanitize_anomaly_for_serialization(anomaly: Anomaly) -> Anomaly:
    """Create a copy of anomaly with numpy types converted to Python native types."""
    from dataclasses import replace
    return replace(
        anomaly,
        confidence=float(anomaly.confidence),
        current_value=float(anomaly.current_value),
        baseline_mean=float(anomaly.baseline_mean),
        baseline_std=float(anomaly.baseline_std),
        z_score=float(anomaly.z_score),
        percentage_change=float(anomaly.percentage_change),
        raw_data=convert_numpy_types(anomaly.raw_data) if anomaly.raw_data else {}
    )


class AnomalyBasedRCAState(AgentState):
    """State for anomaly-based RCA workflow."""
    # 直接存储Anomaly对象（因为禁用了checkpointer，不需要序列化）
    anomalies: List[Anomaly] = Field(default_factory=list, description="List of detected anomalies")
    analysis: Optional[Any] = Field(default=None, description="The extracted analysis object")
    error: str = Field(default="", description="Error message from any failures")


def create_react_diagnostic_state_class(candidate_root_causes: List[str]):
    """Create a ReactDiagnosticState class with proper Analysis typing."""
    Analysis = create_analysis_class(candidate_root_causes)
    
    class ReactDiagnosticState(AgentState):
        """State for the React diagnostic workflow with strong typing."""

        analysis: Optional[Analysis] = Field(default=None, description="The extracted analysis object")
        error: str = Field(default="", description="Error message from any failures")
    
    return ReactDiagnosticState


def setup_logger():
    logger = std_logging.getLogger(__name__)
    logger.setLevel(std_logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "react_agent.log"
    )
    file_handler = std_logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(std_logging.DEBUG)

    console_handler = std_logging.StreamHandler()
    console_handler.setLevel(std_logging.INFO)

    formatter = std_logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


def anomaly_to_dict(anomaly: Anomaly) -> Dict[str, Any]:
    """将Anomaly对象转换为可序列化的字典"""
    return {
        "service": anomaly.service,
        "metric_name": anomaly.metric_name,
        "anomaly_type": anomaly.anomaly_type.value,
        "severity": anomaly.severity.value,
        "confidence": anomaly.confidence,
        "current_value": anomaly.current_value,
        "baseline_mean": anomaly.baseline_mean,
        "baseline_std": anomaly.baseline_std,
        "z_score": anomaly.z_score,
        "percentage_change": anomaly.percentage_change,
        "timestamp": anomaly.timestamp.isoformat(),
        "duration_minutes": anomaly.duration_minutes,
        "evidence": anomaly.evidence,
        "raw_data": anomaly.raw_data
    }


def dict_to_anomaly(anomaly_dict: Dict[str, Any]) -> Anomaly:
    """将字典转换回Anomaly对象"""
    from datetime import datetime
    return Anomaly(
        service=anomaly_dict["service"],
        metric_name=anomaly_dict["metric_name"],
        anomaly_type=AnomalyType(anomaly_dict["anomaly_type"]),
        severity=SeverityLevel(anomaly_dict["severity"]),
        confidence=anomaly_dict["confidence"],
        current_value=anomaly_dict["current_value"],
        baseline_mean=anomaly_dict["baseline_mean"],
        baseline_std=anomaly_dict["baseline_std"],
        z_score=anomaly_dict["z_score"],
        percentage_change=anomaly_dict["percentage_change"],
        timestamp=datetime.fromisoformat(anomaly_dict["timestamp"]),
        duration_minutes=anomaly_dict["duration_minutes"],
        evidence=anomaly_dict["evidence"],
        raw_data=anomaly_dict["raw_data"]
    )


def format_anomalies_for_analysis(anomalies: List[Anomaly]) -> str:
    """Format anomalies data for LLM analysis."""
    if not anomalies:
        return "No anomalies detected."
    
    formatted_sections = []
    
    # Group anomalies by service
    service_anomalies = {}
    for anomaly in anomalies:
        if anomaly.service not in service_anomalies:
            service_anomalies[anomaly.service] = []
        service_anomalies[anomaly.service].append(anomaly)
    
    # Format by service
    for service, service_anomaly_list in service_anomalies.items():
        section = f"\n🔍 **Service: {service}**\n"
        for i, anomaly in enumerate(service_anomaly_list, 1):
            # Determine if this is a root cause or symptom
            is_root_cause = anomaly.anomaly_type in [
                AnomalyType.CPU_SPIKE, AnomalyType.MEMORY_LEAK, 
                AnomalyType.NETWORK_LATENCY, AnomalyType.GC_PRESSURE, AnomalyType.DISK_IO
            ]
            cause_type = "🔴 ROOT CAUSE" if is_root_cause else "🟡 SYMPTOM"
            
            section += f"  {i}. **{anomaly.anomaly_type.value}** ({cause_type})\n"
            section += f"     • Severity: {anomaly.severity.value}\n"
            section += f"     • Confidence: {anomaly.confidence:.2f}\n"
            section += f"     • Metric: {anomaly.metric_name}\n"
            section += f"     • Current: {anomaly.current_value:.2f} vs Baseline: {anomaly.baseline_mean:.2f}\n"
            section += f"     • Change: {anomaly.percentage_change:+.1f}% (Z-score: {anomaly.z_score:.2f})\n"
            section += f"     • Evidence: {anomaly.evidence}\n"
            section += f"     • Duration: {anomaly.duration_minutes} minutes\n\n"
        
        formatted_sections.append(section)
    
    # Summary statistics
    root_causes = [a for a in anomalies if a.anomaly_type in [
        AnomalyType.CPU_SPIKE, AnomalyType.MEMORY_LEAK, 
        AnomalyType.NETWORK_LATENCY, AnomalyType.GC_PRESSURE, AnomalyType.DISK_IO
    ]]
    symptoms = [a for a in anomalies if a.anomaly_type in [
        AnomalyType.SERVICE_LATENCY, AnomalyType.ERROR_BURST
    ]]
    
    summary = f"""📊 **Anomaly Summary:**
• Total anomalies: {len(anomalies)}
• Root causes detected: {len(root_causes)}
• Symptoms detected: {len(symptoms)}
• Services affected: {len(service_anomalies)}
• Severity distribution: {dict(zip([s.value for s in SeverityLevel], [sum(1 for a in anomalies if a.severity == s) for s in SeverityLevel]))}

"""
    
    return summary + "\n".join(formatted_sections)


@tool
def analyze_anomaly_correlations(anomalies_summary: str) -> str:
    """Analyze correlations and patterns in the detected anomalies.
    
    Args:
        anomalies_summary: Summary of all detected anomalies
        
    Returns:
        Analysis of correlations and patterns
    """
    return """Based on the anomalies, I can analyze the following correlation patterns:

1. **Resource-Symptom Correlations**: Look for resource issues (CPU/Memory) that correlate with service symptoms (latency/errors)
2. **Service Dependencies**: Identify if upstream service issues cascade to downstream services
3. **Timeline Analysis**: Check if anomalies appeared in a specific sequence
4. **Severity Patterns**: Determine if critical anomalies in one service cause medium/low anomalies elsewhere

This tool helps identify causal relationships between detected anomalies."""


# create_analysis_tool is now imported from src.tools


def create_react_execution_agent(tools: List[BaseTool]) -> Any:
    """Create a react agent for step execution in plan-execute workflow."""
    with tracer.start_as_current_span("ReactExecutionAgent"):
        system_prompt = """You are a DevOps diagnostic expert executing a specific investigation step.

**Your Role:**
- Execute the given diagnostic step thoroughly
- Use available tools to gather evidence
- Provide detailed findings and observations
- Focus on the specific step assigned to you

**Guidelines:**
- Be thorough but focused on the current step
- Use tools systematically to gather evidence
- Document all findings clearly
- If a step cannot be completed, explain why and suggest alternatives
"""

        summarization_node = SummarizationNode(
            model=model,
            token_counter=count_tokens_approximately,
            max_tokens=262144,
            output_messages_key="messages",
        )

        checkpointer = InMemorySaver()
        react_agent = create_react_agent(
            model,
            tools,
            prompt=SystemMessage(content=system_prompt),
            pre_model_hook=summarization_node,
            state_schema=State,
            checkpointer=checkpointer,
        )

        return react_agent


def extract_analysis_from_messages(messages: List) -> Optional[Dict[str, Any]]:
    """从消息历史中提取工具调用结果。
    
    Args:
        messages: LangChain消息列表
        
    Returns:
        分析结果字典，如果未找到则返回None
    """
    if not messages:
        logger.warning(f"⚠️ 消息列表为空")
        return None
    
    logger.info(f"🔍 开始从{len(messages)}条消息中提取分析结果...")
    
    # 从后往前搜索，找到最近的工具调用结果
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        message_type = type(message).__name__
        logger.debug(f"🔍 检查消息 {i}: {message_type}")
        
        # 检查ToolMessage（工具调用结果）
        if hasattr(message, 'tool_call_id') and hasattr(message, 'content'):
            logger.info(f"🔧 发现ToolMessage: tool_call_id={getattr(message, 'tool_call_id', 'N/A')}")
            
            # 这是一个工具回复消息
            try:
                content = message.content
                logger.info(f"🔍 工具回复内容类型: {type(content)}")
                logger.info(f"🔍 工具回复内容预览: {str(content)[:300]}...")
                
                # 如果内容是字符串，尝试解析为JSON
                if isinstance(content, str):
                    import json
                    try:
                        result = json.loads(content)
                        if isinstance(result, dict) and 'root_causes' in result and result.get('success'):
                            logger.info(f"✅ 成功解析JSON工具结果: {result.get('root_causes')}")
                            return result
                        else:
                            logger.debug(f"🔍 JSON解析成功但不是有效的分析结果: {result}")
                    except json.JSONDecodeError as e:
                        logger.debug(f"🔍 JSON解析失败: {e}")
                        
                        # 尝试查找JSON结构（有时候工具返回包含额外文本的内容）
                        json_start = content.find('{')
                        json_end = content.rfind('}')
                        if json_start != -1 and json_end != -1:
                            try:
                                json_part = content[json_start:json_end + 1]
                                result = json.loads(json_part)
                                if isinstance(result, dict) and 'root_causes' in result:
                                    logger.info(f"✅ 从部分内容解析JSON工具结果: {result.get('root_causes')}")
                                    return result
                            except:
                                pass
                
                # 如果内容已经是字典
                elif isinstance(content, dict) and 'root_causes' in content:
                    logger.info(f"✅ 发现字典格式工具结果: {content.get('root_causes')}")
                    return content
                    
            except Exception as e:
                logger.warning(f"⚠️ 解析工具消息失败: {e}")
                continue
        
        # 检查AIMessage的tool_calls属性
        elif hasattr(message, 'tool_calls') and message.tool_calls:
            logger.info(f"🔧 发现AIMessage包含{len(message.tool_calls)}个工具调用")
            for j, tool_call in enumerate(message.tool_calls):
                tool_name = tool_call.get('name') if isinstance(tool_call, dict) else getattr(tool_call, 'name', 'unknown')
                logger.info(f"   工具调用 {j}: {tool_name}")
                if tool_name == 'simple_provide_final_analysis':
                    logger.info(f"🎯 发现simple_provide_final_analysis工具调用，继续寻找对应结果...")
                    # 工具调用本身不包含结果，需要继续寻找对应的ToolMessage
    
    logger.error(f"❌ 未在{len(messages)}条消息中找到有效的分析结果")
    logger.error("🔍 消息类型分布:")
    message_types = {}
    for msg in messages:
        msg_type = type(msg).__name__
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    for msg_type, count in message_types.items():
        logger.error(f"   {msg_type}: {count}条")
    
    return None


async def create_anomaly_based_rca_workflow(
    anomalies: List[Anomaly], 
    time_range: str, 
    candidate_root_causes: List[str]
) -> StateGraph:
    """Create a ReAct-based RCA workflow that analyzes detected anomalies."""
    with tracer.start_as_current_span("AnomalyBasedRCA"):
        logger.info(f"🚀 创建基于异常的RCA工作流，处理 {len(anomalies)} 个异常")
        
        # Create the analysis tool for the specific candidates
        analysis_tool = create_analysis_tool(candidate_root_causes)
        
        # Create a simpler analysis tool for debugging
        from langchain_core.tools import tool
        @tool
        def simple_provide_final_analysis(root_causes: List[str], reasoning: str) -> str:
            """Provide final analysis with root causes and reasoning.
            
            Args:
                root_causes: List of identified root causes from the candidate list
                reasoning: Explanation of why these root causes were selected
                
            Returns:
                JSON string of analysis result with root causes and evidence
            """
            logger.info(f"🎯 Agent调用简化分析工具")
            logger.info(f"   Root Causes: {root_causes}")
            logger.info(f"   Reasoning: {reasoning[:200]}...")
            
            # Validate root causes
            invalid_causes = [rc for rc in root_causes if rc not in candidate_root_causes]
            if invalid_causes:
                logger.error(f"❌ 无效根因: {invalid_causes}")
                raise ValueError(f"Invalid root causes: {invalid_causes}. Must be from: {candidate_root_causes}")
            
            # Create a simple analysis object
            import json
            analysis_result = {
                'root_causes': root_causes,
                'evidences': [reasoning],
                'evidence_chain': {},  # 简化evidence_chain为空字典
                'success': True
            }
            
            # 返回JSON字符串，便于提取
            result_json = json.dumps(analysis_result, ensure_ascii=False, indent=2)
            logger.info(f"✅ 工具返回分析结果: {len(result_json)}字符")
            return result_json
        
        # Format anomalies for LLM understanding
        anomalies_formatted = format_anomalies_for_analysis(anomalies)
        
        # Create tools specific for anomaly analysis - 使用简化工具进行调试
        tools = [simple_provide_final_analysis, analyze_anomaly_correlations]
        
        # Create enhanced system prompt for anomaly-based analysis
        system_prompt = f"""You are an expert AIOps engineer specializing in root cause analysis. You have been provided with a comprehensive set of anomalies detected by the system's monitoring infrastructure.

**Incident Information:**
• Time Range: {time_range}
• Candidate Root Causes: {candidate_root_causes}
• Total Anomalies Detected: {len(anomalies)}

**Your Mission:**
Analyze the provided anomalies to identify the most likely root cause(s) from the candidate list. Focus on logical causality rather than complex rules.

**Analysis Strategy:**
1. **Understand the Anomaly Landscape**: Review all detected anomalies and their relationships
2. **Apply Causal Reasoning**: 
   - Resource issues (CPU/Memory spikes) typically CAUSE service symptoms (latency/errors)
   - Service symptoms without corresponding resource issues suggest service-specific problems
   - Network issues affect multiple services simultaneously
   - Time correlations matter - what happened first?

3. **Identify Primary vs Secondary Effects**:
   - 🔴 ROOT CAUSES: CPU_SPIKE, MEMORY_LEAK, NETWORK_LATENCY, GC_PRESSURE, DISK_IO
   - 🟡 SYMPTOMS: SERVICE_LATENCY, ERROR_BURST (these are EFFECTS, not root causes unless no resource issues found)

4. **Apply Logical Priority**:
   - If CPU spike detected → likely CPU-related root cause
   - If memory issues detected → likely memory-related root cause  
   - If only service symptoms → likely service failure root cause
   - Consider severity and confidence levels

**Key Principles:**
- Symptoms (latency, errors) are usually CAUSED BY resource issues, not root causes themselves
- Multiple symptoms from one service usually indicate a single root cause in that service
- Cross-service impact suggests shared infrastructure or dependency issues
- Higher confidence and severity anomalies are more reliable indicators

**DETECTED ANOMALIES:**
{anomalies_formatted}

**Final Analysis Requirements:**
- Select root cause(s) from the candidate list based on logical analysis
- Explain your reasoning based on the anomaly evidence  
- Focus on the most direct causal relationships
- Use the `simple_provide_final_analysis` tool when ready to submit your conclusion
- The tool takes two parameters: root_causes (list) and reasoning (string)

**CRITICAL:** Base your analysis on the provided anomaly data. Don't use tools to gather additional data - all necessary information is already provided in the anomaly detection results.

**EXAMPLE TOOL CALL:**
```
simple_provide_final_analysis(
    root_causes=["ad.CPUAnomalyFailure"],
    reasoning="Based on the detected anomalies, I found multiple CPU spikes in the 'ad' service with high confidence (0.85-0.90). The CPU usage increased by 45-50% from baseline during the incident window, indicating a clear CPU resource bottleneck causing the service degradation."
)
```
"""

        # Create the ReAct agent (禁用checkpointer避免序列化问题)
        react_agent = create_react_agent(
            model,
            tools,
            prompt=SystemMessage(content=system_prompt),
            state_schema=AnomalyBasedRCAState,
            checkpointer=None,  # 禁用checkpointer避免Anomaly序列化问题
        )

        return react_agent


async def create_react_diagnostic_workflow(
    tools: List[BaseTool], time_range: str, candidate_root_causes: List[str]
) -> StateGraph:
    """Create a ReAct-based diagnostic workflow using LangGraph nodes."""
    with tracer.start_as_current_span("ReactRCA"):
        # Create the properly typed state class
        ReactDiagnosticState = create_react_diagnostic_state_class(candidate_root_causes)
        
        # Create the analysis tool
        analysis_tool = create_analysis_tool(candidate_root_causes)

        # Add the analysis tool to the available tools
        enhanced_tools = tools + [analysis_tool]

        system_prompt = f"""You are a DevOps diagnostic expert investigating system failures.

**Time Range:** {time_range}
**Candidate Root Causes:** {candidate_root_causes}

**Your Mission:**
1. Investigate the system failure using available tools
2. Gather evidence systematically
3. Identify the most likely root causes from the candidate list
4. When you have sufficient evidence, use the `provide_final_analysis` tool to submit your structured findings

**Investigation Guidelines:**
- Start with high-level system health checks
- Drill down into specific components showing issues
- Compare with baseline/normal behavior when possible，using strictly no more than 10 minutes before the incident as the baseline period
- Look for correlations between symptoms and timing
- Gather concrete evidence for each suspected root cause

**Final Analysis Requirements:**
- Root causes must be from the provided candidate list
- Provide clear evidence supporting each root cause
- Document your investigation steps and tool usage
- Use the `provide_final_analysis` tool when ready to conclude

**IMPORTANT:** You must use the `provide_final_analysis` tool to submit your final structured analysis. Do not attempt to format the response manually.
"""

        summarization_node = SummarizationNode(
            model=model,
            token_counter=count_tokens_approximately,
            max_tokens=262144,
            output_messages_key="messages",
        )

        checkpointer = InMemorySaver()
        react_agent = create_react_agent(
            model,
            enhanced_tools,
            prompt=SystemMessage(content=system_prompt),
            pre_model_hook=summarization_node,
            state_schema=ReactDiagnosticState,
            checkpointer=checkpointer,
        )

        # Define workflow nodes
        async def run_investigation(state: ReactDiagnosticState) -> Dict[str, Any]:
            """Run the React agent investigation."""
            logger.info("Starting React agent investigation...")

            result = await react_agent.ainvoke(
                state,
                RunnableConfig(
                    recursion_limit=sys.maxsize,
                    configurable={"thread_id": threading.get_ident()},
                ),
                print_mode="tasks",
            )

            messages: List[BaseMessage] = result["messages"]
            for i, message in enumerate(messages):
                if isinstance(message, AIMessage):
                    logger.debug(
                        f"Message #{i} / {len(messages)}:\n{message.pretty_repr()}\n"
                    )

            return {"messages": result["messages"]}

        async def extract_analysis(state: ReactDiagnosticState) -> ReactDiagnosticState:
            """Extract the structured analysis from tool calls with strong typing."""
            logger.info("Extracting structured analysis from tool calls...")

            messages: List[BaseMessage] = state.get("messages", [])

            # Look for ToolMessage with provide_final_analysis result
            for message in reversed(messages):
                if (
                    isinstance(message, ToolMessage)
                    and message.name == "provide_final_analysis"
                ):
                    logger.info("Found provide_final_analysis tool result")
                    
                    # Extract the analysis content - Pydantic will validate when assigned to typed field
                    try:
                        analysis_content = message.content
                        logger.info("Successfully extracted analysis object")
                        # Create state copy without conflicting fields
                        clean_state = {k: v for k, v in state.items() if k not in ['analysis', 'last_error']}
                        return ReactDiagnosticState(
                            **clean_state,
                            analysis=analysis_content,  # Pydantic will validate this is an Analysis object
                            last_error=""
                        )
                    except ValueError as e:
                        # Catch Pydantic validation errors when assigning to typed field
                        logger.warning(f"Pydantic validation failed: {e}")
                        # Create state copy without conflicting fields
                        clean_state = {k: v for k, v in state.items() if k != 'last_error'}
                        return ReactDiagnosticState(
                            **clean_state,
                            last_error=f"Pydantic validation error: {str(e)}"
                        )

            # No tool result found - set error and let the graph continue stepping
            logger.warning("No analysis tool result found")
            # Create state copy without conflicting fields
            clean_state = {k: v for k, v in state.items() if k != 'last_error'}
            return ReactDiagnosticState(
                **clean_state,
                last_error="No provide_final_analysis tool result found"
            )

        def should_extract_analysis(state: ReactDiagnosticState) -> str:
            """Determine if analysis should be extracted or if we should retry with strong typing."""
            # Check if analysis was already successfully extracted
            if state.get("analysis"):
                return END
            
            # Check if we have a provide_final_analysis tool result
            has_analysis_tool = False
            for message in reversed(state.get("messages", [])):
                if (
                    isinstance(message, ToolMessage)
                    and message.name == "provide_final_analysis"
                ):
                    has_analysis_tool = True
                    break
            
            if has_analysis_tool:
                return "extract_analysis"
            
            # No analysis tool found - continue with error context
            logger.info("No analysis tool found, continuing investigation with error context")
            return "continue_with_error_context"

        def should_continue_investigation(state: ReactDiagnosticState) -> str:
            """Determine if we should continue the investigation with error context."""
            # Check if analysis was successfully extracted
            if state.get("analysis"):
                return END
            else:
                return "investigate"  # Continue investigating

        async def continue_with_error_context(state: ReactDiagnosticState) -> ReactDiagnosticState:
            """Continue investigation with error context for better reasoning."""
            logger.info("Continuing investigation with error context")
            
            # Add an error context message to guide the agent's reasoning
            error_context_message = f"""
Previous analysis attempt encountered an issue: {state.get('last_error', 'Unknown error')}

Please continue your investigation with this context in mind:
1. The previous attempt failed - consider what might have gone wrong
2. Try a different diagnostic approach or use different tools
3. Gather more comprehensive evidence before attempting final analysis
4. When ready, use the provide_final_analysis tool with valid root causes from: {candidate_root_causes}
5. Ensure your analysis is complete and well-supported by evidence

Continue your investigation now, taking into account the previous error.
"""
            
            updated_messages = state.get("messages", []) + [HumanMessage(content=error_context_message)]
            
            # Create state copy without conflicting fields
            clean_state = {k: v for k, v in state.items() if k not in ['messages', 'last_error']}
            return ReactDiagnosticState(
                **clean_state,
                messages=updated_messages,
                last_error=""  # Clear error since we're providing context
            )

        # Create the workflow
        workflow = StateGraph(ReactDiagnosticState)

        # Add nodes
        workflow.add_node("investigate", run_investigation)
        workflow.add_node("extract_analysis", extract_analysis)
        workflow.add_node("continue_with_error_context", continue_with_error_context)

        # Add edges
        workflow.add_edge(START, "investigate")
        workflow.add_conditional_edges(
            "investigate",
            should_extract_analysis,
            {
                "extract_analysis": "extract_analysis", 
                "continue_with_error_context": "continue_with_error_context",
                END: END
            },
        )
        workflow.add_conditional_edges(
            "continue_with_error_context",
            should_continue_investigation,
            {
                "investigate": "investigate",
                END: END
            },
        )
        workflow.add_conditional_edges(
            "extract_analysis",
            should_continue_investigation,
            {
                "continue_with_error_context": "continue_with_error_context",
                END: END
            },
        )

        # Compile the workflow
        app = workflow.compile()
        return app


@asynccontextmanager
async def create_react_diagnostic_agent(
    tools: List[BaseTool], time_range: str, candidate_root_causes: List[str]
) -> AsyncGenerator[Any, None]:
    """Create a ReAct-based diagnostic agent using LangGraph workflow."""
    workflow = await create_react_diagnostic_workflow(
        tools, time_range, candidate_root_causes
    )

    async def run_diagnostic_workflow(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the diagnostic workflow and return structured results."""
        # Initialize state with input messages
        initial_state = {
            "messages": input_data.get("messages", []),
            "structured_response": None,
            "analysis_extracted": False,
        }

        # Run the workflow
        final_state = None
        async for event in workflow.astream(initial_state):
            for k, v in event.items():
                if k != "__end__":
                    logger.debug(f"Done: {k}")
                    final_state = v

        # Extract the structured response
        if final_state and final_state.get("analysis_extracted"):
            return {
                "structured_response": final_state["structured_response"],
                "messages": final_state["messages"],
            }
        else:
            raise ValueError("Workflow did not produce a structured analysis")

    yield RunnableLambda(run_diagnostic_workflow)


async def compute_react(
    chain: RunnableLambda,
    time_range: str,
    candidate_root_causes: List[str],
    max_retries: int = 3,
) -> BaseModel:
    """Execute ReAct diagnostic workflow."""
    logger.info(f"Time range: {time_range}")
    logger.info(f"Candidate root causes: {candidate_root_causes}")

    try:
        response = await chain.ainvoke(
            {"messages": [HumanMessage(content="Let's begin.")]},
            RunnableConfig(recursion_limit=sys.maxsize),
        )

        analysis = response.get("analysis")
        if analysis:
            logger.info(
                f"Successfully obtained analysis with root_causes: {analysis.root_causes}"
            )
            return analysis
        else:
            # No analysis produced
            raise RuntimeError("React diagnostic workflow failed to produce analysis")

    except Exception as e:
        logger.error(f"React diagnostic workflow failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Never create fallback analysis - let the driver handle the exception
        raise RuntimeError(f"React diagnostic workflow failed to produce analysis: {str(e)}") from e


async def compute_anomaly_based_rca(
    anomalies: List[Anomaly],
    time_range: str,
    candidate_root_causes: List[str],
    problem_id: str = "unknown"
) -> Any:
    """Execute anomaly-based root cause analysis using ReAct agent.
    
    Args:
        anomalies: List of detected anomalies from anomaly detection
        time_range: Time range of the incident
        candidate_root_causes: List of candidate root causes to consider
        problem_id: Problem ID for tracking
        
    Returns:
        Analysis object with root causes and evidence
    """
    
    try:
        logger.info(f"🚀 开始基于异常的RCA分析")
        logger.info(f"   Problem ID: {problem_id}")
        logger.info(f"   Time Range: {time_range}")
        logger.info(f"   异常数量: {len(anomalies)}")
        logger.info(f"   候选根因数量: {len(candidate_root_causes)}")
        
        # Log anomaly summary
        service_count = len(set(a.service for a in anomalies))
        root_causes_detected = [a for a in anomalies if a.anomaly_type in [
            AnomalyType.CPU_SPIKE, AnomalyType.MEMORY_LEAK, 
            AnomalyType.NETWORK_LATENCY, AnomalyType.GC_PRESSURE, AnomalyType.DISK_IO
        ]]
        symptoms_detected = [a for a in anomalies if a.anomaly_type in [
            AnomalyType.SERVICE_LATENCY, AnomalyType.ERROR_BURST
        ]]
        
        logger.info(f"   🔴 根因型异常: {len(root_causes_detected)}")
        logger.info(f"   🟡 症状型异常: {len(symptoms_detected)}")
        logger.info(f"   📊 影响服务数: {service_count}")
        
        # Create the anomaly-based RCA workflow
        react_agent = await create_anomaly_based_rca_workflow(
            anomalies=anomalies,
            time_range=time_range,
            candidate_root_causes=candidate_root_causes
        )
        
        # Execute the workflow
        logger.info("🔄 执行基于异常的RCA分析...")
        
        # 清理异常对象，确保没有numpy类型
        sanitized_anomalies = [sanitize_anomaly_for_serialization(anomaly) for anomaly in anomalies]
        
        initial_state = AnomalyBasedRCAState(
            messages=[HumanMessage(content=f"请分析问题 {problem_id} 的根因。基于提供的异常检测结果进行分析。")],
            anomalies=sanitized_anomalies
        )
        
        # Configure execution
        config = RunnableConfig({
            "thread_id": f"anomaly_rca_{problem_id}",
            "recursion_limit": 50  # Limit iterations to prevent infinite loops
        })
        
        # Run the workflow
        response = await react_agent.ainvoke(initial_state, config)
        
        logger.info("✅ 基于异常的RCA分析完成")
        
        # Extract the analysis - 修复：从工具调用消息中提取结果
        analysis = extract_analysis_from_messages(response.get("messages", []))
        if analysis:
            logger.info(f"🎯 识别根因: {analysis.get('root_causes', [])}")
            logger.info(f"📄 证据数量: {len(analysis.get('evidences', []))}")
            return analysis
        else:
            # 调试信息：显示响应的所有键和消息
            logger.error(f"🔍 响应中的键: {list(response.keys()) if response else 'None'}")
            if response:
                messages = response.get("messages", [])
                logger.error(f"🔍 消息数量: {len(messages)}")
                
                # 详细分析最后几条消息，寻找工具调用
                logger.error("🔍 最后几条消息详情:")
                for i, msg in enumerate(messages[-5:]):  # 增加到最后5条消息
                    msg_type = type(msg).__name__
                    msg_content = str(msg)[:300]
                    logger.error(f"   消息 {i}: {msg_type}")
                    logger.error(f"   内容: {msg_content}...")
                    
                    # 检查是否有工具调用
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        logger.error(f"   🔧 发现工具调用: {len(msg.tool_calls)}个")
                        for j, tool_call in enumerate(msg.tool_calls):
                            logger.error(f"      工具{j}: {tool_call}")
                    
                    # 检查是否是工具回复消息
                    if hasattr(msg, 'content') and 'simple_provide_final_analysis' in str(msg.content):
                        logger.error(f"   🎯 发现工具回复消息")
                        
            raise RuntimeError("基于异常的RCA分析未能产生分析结果")
            
    except Exception as e:
        logger.error(f"❌ 基于异常的RCA分析失败: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise RuntimeError(f"基于异常的RCA分析失败: {str(e)}") from e
