#!/usr/bin/env python

from contextlib import asynccontextmanager
import logging as std_logging
import operator
import os
import random
import sys
import threading
import traceback
from typing import Any, AsyncGenerator, Dict, List, Tuple, Union, Optional, TYPE_CHECKING
from typing_extensions import TypedDict


from langchain.output_parsers.retry import NAIVE_RETRY_WITH_ERROR_PROMPT
from langchain_core.exceptions import OutputParserException
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
# from langchain_core.output_parsers import PydanticOutputParser  # Not available in this version
try:
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:
    # 使用基础的输出解析器作为替代
    from langchain_core.output_parsers import JsonOutputParser
    PydanticOutputParser = JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field, validator

from src.config import model, tracer, PERFORMANCE_CONFIG, DEBUG_CONFIG, RESPONSE_OUTPUT_CONFIG
from src.syntax import create_analysis_class
from src.tools import DiagnosticPlan, DiagnosticResponse, create_diagnostic_plan, provide_diagnostic_response
from src.prompts import create_prompt

# create_react_execution_agent imported locally to avoid circular import


class State(AgentState):
    context: Dict[str, Any]


# Plan-and-Execute Models are now imported from src.tools


def create_plan_execute_state_class(candidate_root_causes: List[str]):
    """Create a PlanExecuteState class with proper Analysis typing."""
    Analysis = create_analysis_class(candidate_root_causes)
    
    class PlanExecuteState(AgentState):
        """State for plan-and-execute diagnostic workflow with strong typing."""

        input: str = Field(description="The diagnostic problem description")
        time_range: str = Field(description="Time range for the investigation")
        plan: List[str] = Field(default_factory=list, description="Current diagnostic plan steps")
        past_steps: List[Tuple[str, str]] = Field(default_factory=list, description="Completed steps with results")
        response: str = Field(default="", description="Final diagnostic response")
        analysis: Optional[Analysis] = Field(default=None, description="Parsed structured analysis object")
        error: str = Field(default="", description="Error message from any failures")
        problem_id: str = Field(default="unknown", description="Problem ID for tracking and output organization")
    
    return PlanExecuteState


def setup_logger():
    logger = std_logging.getLogger(__name__)
    logger.setLevel(std_logging.DEBUG)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent.log")
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


def debug_print_messages(messages: List[BaseMessage], step_description: str, phase: str = "PROCESSING") -> None:
    """Print detailed message information for debugging purposes."""
    debug_level = DEBUG_CONFIG["debug_level"]
    
    if debug_level == "NONE":
        return
    
    # Prepare debug output
    debug_output = []
    debug_output.append(f"\n{'='*80}")
    debug_output.append(f"🐛 DEBUG [{phase}] Step: {step_description}")
    debug_output.append(f"📊 Total Messages: {len(messages)}")
    
    if debug_level == "BASIC":
        # Just show message type statistics
        msg_types = {}
        for msg in messages:
            msg_type = type(msg).__name__
            msg_types[msg_type] = msg_types.get(msg_type, 0) + 1
        debug_output.append(f"📋 Message Types: {dict(msg_types)}")
    
    elif debug_level in ["DETAILED", "FULL", "COMPARISON"]:
        max_preview = DEBUG_CONFIG["max_message_preview"]
        show_tool_results = DEBUG_CONFIG["show_tool_results"]
        show_ai_responses = DEBUG_CONFIG["show_ai_responses"]
        
        for i, message in enumerate(messages):
            msg_type = type(message).__name__
            debug_output.append(f"\n--- Message {i+1}: {msg_type} ---")
            
            if isinstance(message, HumanMessage):
                content = message.content[:max_preview] if debug_level != "FULL" else message.content
                debug_output.append(f"🤔 Human: {content}{'...' if len(message.content) > max_preview and debug_level != 'FULL' else ''}")
            
            elif isinstance(message, AIMessage) and show_ai_responses:
                if message.content:
                    content = message.content[:max_preview] if debug_level != "FULL" else message.content
                    debug_output.append(f"🤖 AI: {content}{'...' if len(message.content) > max_preview and debug_level != 'FULL' else ''}")
                
                # Show tool calls if any
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    debug_output.append(f"🔧 Tool Calls: {len(message.tool_calls)}")
                    for tool_call in message.tool_calls:
                        debug_output.append(f"   └─ {tool_call.get('name', 'unknown')}({str(tool_call.get('args', {}))[:100]})")
            
            elif isinstance(message, ToolMessage) and show_tool_results:
                tool_name = message.name
                content = message.content[:max_preview] if debug_level != "FULL" else message.content
                debug_output.append(f"🛠️  Tool[{tool_name}]: {content}{'...' if len(message.content) > max_preview and debug_level != 'FULL' else ''}")
            
            else:
                debug_output.append(f"📝 Other: {str(message)[:max_preview]}")
    
    debug_output.append(f"{'='*80}\n")
    
    # Output to console and/or file
    debug_text = "\n".join(debug_output)
    
    if DEBUG_CONFIG["debug_to_file"]:
        with open(DEBUG_CONFIG["debug_file_path"], "a", encoding="utf-8") as f:
            f.write(debug_text + "\n")
    
    # Always show in console for debugging
    print(debug_text)


def print_debug_startup_info() -> None:
    """Print debug configuration info at startup."""
    debug_level = DEBUG_CONFIG["debug_level"]
    if debug_level != "NONE":
        print(f"\n🐛 DEBUG MODE ENABLED - Level: {debug_level}")
        print(f"📝 Message Preview: {DEBUG_CONFIG['max_message_preview']} chars")
        print(f"💾 Save to File: {DEBUG_CONFIG['debug_to_file']}")
        if DEBUG_CONFIG["debug_to_file"]:
            print(f"📁 Debug File: {DEBUG_CONFIG['debug_file_path']}")
        print(f"🛠️  Show Tool Results: {DEBUG_CONFIG['show_tool_results']}")
        print(f"🤖 Show AI Responses: {DEBUG_CONFIG['show_ai_responses']}")
        print("="*60)


def save_final_response(response_text: str, problem_info: Dict[str, Any] = None) -> None:
    """Save the final diagnostic response in formatted JSON to file."""
    if not RESPONSE_OUTPUT_CONFIG["save_final_response"]:
        return
    
    import json
    from datetime import datetime
    import os
    
    try:
        # Parse the response text as JSON
        response_data = json.loads(response_text)
        
        # Prepare the output structure
        output_data = {}
        
        # Add metadata if enabled
        if RESPONSE_OUTPUT_CONFIG["include_metadata"]:
            output_data["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "saved_by": "diagnostic_agent_v1.0",
                "response_length_chars": len(response_text),
            }
            
            # Add problem information if provided
            if problem_info:
                output_data["metadata"].update(problem_info)
        
        # Add the actual response content
        output_data["diagnostic_result"] = response_data
        
        # Determine output format
        format_type = RESPONSE_OUTPUT_CONFIG["response_file_format"]
        if format_type == "pretty":
            # Beautiful formatting with indentation and colors
            formatted_json = json.dumps(output_data, indent=2, ensure_ascii=False, sort_keys=False)
        elif format_type == "compact":
            # Compact single-line format
            formatted_json = json.dumps(output_data, ensure_ascii=False, separators=(',', ':'))
        elif format_type == "timestamped":
            # Pretty format with timestamp prefix
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_json = f"// Generated at {timestamp}\n" + json.dumps(output_data, indent=2, ensure_ascii=False)
        else:
            # Default to pretty
            formatted_json = json.dumps(output_data, indent=2, ensure_ascii=False)
        
        # Determine file path
        file_path = RESPONSE_OUTPUT_CONFIG["response_file_path"]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
        
        # Write to file
        mode = "a" if RESPONSE_OUTPUT_CONFIG["append_mode"] else "w"
        with open(file_path, mode, encoding="utf-8") as f:
            if RESPONSE_OUTPUT_CONFIG["append_mode"] and os.path.getsize(file_path) > 0:
                f.write("\n" + "="*100 + "\n")  # Separator for multiple entries
            f.write(formatted_json)
            f.write("\n")  # Ensure ending newline
        
        # Log success
        logger.info(f"💾 Final response saved to: {file_path} ({len(formatted_json)} chars)")
        
        # Optional: Print a brief confirmation to console
        if RESPONSE_OUTPUT_CONFIG["include_metadata"] and problem_info:
            problem_id = problem_info.get("problem_id", "unknown")
            root_causes = response_data.get("root_causes", [])
            print(f"💾 Final diagnosis saved for Problem {problem_id}: {root_causes}")
            
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse response as JSON: {e}")
        logger.error(f"Raw response text: {response_text[:500]}...")
        
        # Save raw text as fallback
        file_path = RESPONSE_OUTPUT_CONFIG["response_file_path"] + ".raw"
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*50} RAW RESPONSE (PARSE FAILED) {'='*50}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n")
            f.write(f"Content:\n{response_text}\n")
            f.write("="*100 + "\n")
        logger.warning(f"💾 Raw response saved to: {file_path}")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error saving final response: {e}")


def debug_compare_compression(original_messages: List[BaseMessage], compressed_summary: str, step_description: str) -> None:
    """Compare original messages with compressed summary for verification."""
    if DEBUG_CONFIG["debug_level"] != "COMPARISON":
        return
    
    # Calculate sizes
    original_size = sum(len(str(msg)) for msg in original_messages)
    compressed_size = len(compressed_summary)
    compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"🔍 COMPRESSION ANALYSIS - Step: {step_description}")
    print(f"📏 Original Size: {original_size:,} chars ({len(original_messages)} messages)")
    print(f"🗜️  Compressed Size: {compressed_size:,} chars")
    print(f"📊 Compression Ratio: {compression_ratio:.1f}%")
    print(f"{'='*80}")
    print(f"📄 ORIGINAL MESSAGES:")
    for i, msg in enumerate(original_messages):
        msg_type = type(msg).__name__
        content_preview = str(msg)[:200] + "..." if len(str(msg)) > 200 else str(msg)
        print(f"   {i+1}. [{msg_type}] {content_preview}")
    
    print(f"\n🗜️  COMPRESSED SUMMARY:")
    print(f"   {compressed_summary}")
    print(f"{'='*80}\n")
    
    if DEBUG_CONFIG["debug_to_file"]:
        with open(DEBUG_CONFIG["debug_file_path"], "a", encoding="utf-8") as f:
            f.write(f"COMPRESSION ANALYSIS - {step_description}\n")
            f.write(f"Original: {original_size} chars, Compressed: {compressed_size} chars, Ratio: {compression_ratio:.1f}%\n")
            f.write(f"Compressed Summary: {compressed_summary}\n\n")


def has_strong_evidence(past_steps: List[Tuple[str, str]]) -> bool:
    """Check if we have strong enough evidence to consider early termination."""
    
    if len(past_steps) < 3:
        return False
    
    # Look for strong evidence keywords in recent steps
    strong_evidence_keywords = [
        'critical', 'severe', 'definitive', 'confirmed', 'obvious', 
        'clear evidence', 'root cause identified', 'conclusive'
    ]
    
    recent_results = [result.lower() for _, result in past_steps[-3:]]
    combined_results = " ".join(recent_results)
    
    # Check if multiple strong indicators are present
    evidence_count = sum(1 for keyword in strong_evidence_keywords if keyword in combined_results)
    
    return evidence_count >= PERFORMANCE_CONFIG["early_termination_threshold"]


def get_relevant_context(past_steps: List[Tuple[str, str]], current_step: str, max_steps: int = None) -> str:
    """Get most relevant context for current step, avoiding overwhelming information."""
    
    if not past_steps:
        return "No previous investigation completed."
    
    # Use configurable max steps for context
    if max_steps is None:
        max_steps = PERFORMANCE_CONFIG["max_context_steps"]
    
    # Limit to recent steps to prevent context explosion
    recent_steps = past_steps[-max_steps:]
    
    # Create concise context summary
    context_lines = []
    for step_desc, step_result in recent_steps:
        # Extract key insights from each step (first 100 chars)
        key_insight = step_result.split('\n')[0][:100] + "..." if len(step_result) > 100 else step_result.split('\n')[0]
        context_lines.append(f"• {step_desc[:50]}{'...' if len(step_desc) > 50 else ''}: {key_insight}")
    
    return "\n".join(context_lines)


async def create_step_summary(messages: List[BaseMessage], step_description: str) -> str:
    """Create a concise summary of step execution instead of storing full conversation."""
    
    # Extract key findings from the conversation
    key_findings = []
    tool_results = []
    
    for message in messages:
        if isinstance(message, ToolMessage):
            # Summarize tool results instead of storing full content
            tool_results.append(f"Tool {message.name}: {message.content[:200]}..." if len(message.content) > 200 else f"Tool {message.name}: {message.content}")
        elif isinstance(message, AIMessage) and message.content:
            # Extract key insights from AI messages, focusing on conclusions
            if any(keyword in message.content.lower() for keyword in ['found', 'identified', 'indicates', 'shows', 'evidence', 'conclusion']):
                max_length = PERFORMANCE_CONFIG["summary_length_limit"] // 3  # Divide among findings
                key_findings.append(message.content[:max_length] + "..." if len(message.content) > max_length else message.content)
    
    # Create structured summary
    summary_parts = [
        f"**Step:** {step_description}",
        f"**Tools Used:** {len(tool_results)} tools",
    ]
    
    if key_findings:
        summary_parts.append(f"**Key Findings:** {' | '.join(key_findings[:3])}")  # Limit to top 3 findings
    
    if tool_results:
        summary_parts.append(f"**Tool Results:** {' | '.join(tool_results[:2])}")  # Limit to top 2 tool results
    
    return "\n".join(summary_parts)


# Tools are now imported from src.tools


def create_planner(time_range: str, candidate_root_causes: List[str]) -> Any:
    """Create a planner for diagnostic investigation."""

    planner_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=f"""You are a DevOps diagnostic expert. For the given failure scenario, create a systematic step-by-step investigation plan.

**Failure Context:**
- Time Range: {time_range}
- Candidate Root Causes: {candidate_root_causes}

**Planning Guidelines:**
1. Start with high-level symptom identification
2. Progress to resource-level investigation (CPU, memory, network)
3. Drill down to specific components (pods, services, databases)
4. Include baseline comparison steps
5. End with root cause correlation and evidence gathering

Each step should be specific, actionable, and focused on gathering evidence.
Do not add superfluous steps. Make sure each step builds on previous findings.
The final step should synthesize findings into root cause identification.

Use the create_diagnostic_plan tool to create your diagnostic plan with the ordered steps.
"""
            ),
            ("placeholder", "{messages}"),
        ]
    )

    return planner_prompt | model.bind_tools(
        [create_diagnostic_plan], tool_choice="create_diagnostic_plan"
    )


def create_replanner(time_range: str, candidate_root_causes: List[str]) -> Any:
    """Create a replanner for diagnostic investigation."""

    # Create the analysis class and parser to get format instructions
    from src.syntax import create_analysis_class

    Analysis = create_analysis_class(candidate_root_causes)
    pydantic_parser = PydanticOutputParser(pydantic_object=Analysis)
    format_instructions = pydantic_parser.get_format_instructions()

    # Escape braces in format_instructions to prevent template variable conflicts
    escaped_format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
    replanner_prompt = ChatPromptTemplate.from_template(
        f"""You are a DevOps diagnostic expert. Based on investigation progress, update the diagnostic plan or provide final diagnosis.

**Original Objective:**
{{input}}

**Failure Context:**
- Time Range: {time_range}
- Candidate Root Causes: {candidate_root_causes}

**Original Plan:**
{{plan}}

**Completed Investigation Steps:**
{{past_steps}}

**Instructions:**
- If you have sufficient evidence to identify root causes, provide a final diagnosis using provide_diagnostic_response with response_type="response"
- If more investigation is needed, update the plan using provide_diagnostic_response with response_type="plan"
- Only include steps that still NEED to be done
- Do not repeat completed steps
- Focus on the most promising leads from completed steps

**Decision Criteria for Final Response:**
- Clear evidence pointing to specific root cause(s)
- Baseline comparisons showing significant anomalies
- Resource exhaustion patterns identified
- Service dependency issues confirmed

**When providing final diagnosis, format the content field as structured analysis in this format:**
{escaped_format_instructions}

Use the provide_diagnostic_response tool to provide your response.
"""
    )

    return replanner_prompt | model.bind_tools(
        [provide_diagnostic_response], tool_choice="provide_diagnostic_response"
    )


async def create_plan_execute_workflow(
    tools: List[BaseTool], time_range: str, candidate_root_causes: List[str]
) -> StateGraph:
    """Create a plan-and-execute diagnostic workflow."""

    # Create the properly typed state class
    PlanExecuteState = create_plan_execute_state_class(candidate_root_causes)

    # Create planner and replanner
    planner = create_planner(time_range, candidate_root_causes)
    replanner = create_replanner(time_range, candidate_root_causes)

    # Create execution agent using the react agent from react_agent.py
    from src.react_agent import create_react_execution_agent
    execution_agent = create_react_execution_agent(tools)

    # Define workflow steps
    async def plan_step(state: PlanExecuteState) -> Dict[str, List[str]]:
        """Create initial diagnostic plan."""
        logger.info("Creating diagnostic plan...")
        response = await planner.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Create a diagnostic plan for: {state['input']}"
                    )
                ]
            }
        )

        # Extract the tool call result (guaranteed by tool_choice)
        tool_call = response.tool_calls[0]
        steps = tool_call["args"]["steps"]
        logger.info(f"Created plan with {len(steps)} steps")
        return {"plan": steps}

    async def execute_step(
        state: PlanExecuteState,
    ) -> Dict[str, Union[List[str], List[Tuple[str, str]]]]:
        """Execute the next step in the diagnostic plan."""
        plan = state["plan"]
        if not plan:
            # No more steps to execute, maintain current past_steps
            return {"past_steps": state.get("past_steps", [])}

        current_step = plan[0]
        logger.info(f"Executing step: {current_step}")

        # Format the step with optimized context
        plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
        
        # Limit context to most recent and relevant steps
        past_steps = state.get("past_steps", [])
        relevant_context = get_relevant_context(past_steps, current_step)
        
        task_formatted = f"""
**Current Diagnostic Plan:**
{plan_str}

**You are executing step 1: {current_step}**

**Relevant Previous Findings:**
{relevant_context}

Focus on this specific step and provide detailed findings.
"""

        # 🐛 DEBUG: Show input message for this step
        input_messages = [HumanMessage(content=task_formatted)]
        debug_print_messages(input_messages, current_step, "INPUT")

        # Execute with the agent
        agent_response = await execution_agent.ainvoke(
            {"messages": input_messages},
            RunnableConfig(
                # recursion_limit=sys.maxsize,
                configurable={"thread_id": threading.get_ident()},
            ),
        )

        # Extract all messages from the conversation
        messages = agent_response["messages"]

        # 🐛 DEBUG: Show all messages from agent execution
        debug_print_messages(messages, current_step, "OUTPUT")

        # Create a concise summary instead of full conversation
        result_content = await create_step_summary(messages, current_step)

        # 🐛 DEBUG: Compare original vs compressed
        debug_compare_compression(messages, result_content, current_step)

        # Calculate compression ratio for performance monitoring
        if PERFORMANCE_CONFIG["enable_context_compression"]:
            original_length = sum(len(msg.content) if hasattr(msg, 'content') else len(str(msg)) for msg in messages)
            compression_ratio = (1 - len(result_content) / original_length) * 100 if original_length > 0 else 0
            logger.info(
                f"✅ Step completed: {len(messages)} messages, compressed {len(result_content)} chars "
                f"(🗜️ {compression_ratio:.1f}% compression)"
            )
        else:
            logger.info(
                f"Step completed with {len(messages)} messages, total content length: {len(result_content)}"
            )

        # Remove the completed step from the plan and add it to past_steps
        remaining_plan = plan[1:]  # Remove the first step that we just executed
        updated_past_steps = state.get("past_steps", []) + [
            (current_step, result_content)
        ]

        return {
            "plan": remaining_plan,
            "past_steps": updated_past_steps,
        }

    async def replan_step(state: PlanExecuteState) -> Dict[str, Union[str, List[str]]]:
        """Replan based on investigation progress."""
        logger.info("Replanning based on investigation progress...")

        # 🐛 DEBUG: Show state content going into replanner
        if DEBUG_CONFIG["debug_level"] in ["DETAILED", "FULL"]:
            print(f"\n🧠 REPLANNER INPUT STATE:")
            print(f"   Input: {state.get('input', 'N/A')[:200]}...")
            print(f"   Plan: {state.get('plan', [])}")
            print(f"   Past Steps: {len(state.get('past_steps', []))} completed")
            print(f"   Error: {state.get('error', 'None')}")

        response = await replanner.ainvoke(state)

        # Extract the tool call result (guaranteed by tool_choice)
        tool_call = response.tool_calls[0]
        response_type = tool_call["args"]["response_type"]
        content = tool_call["args"]["content"]

        if response_type == "response":
            logger.info("Investigation complete - providing final diagnosis")
            return {"response": content}
        else:  # response_type == "plan"
            # Parse the comma-separated steps
            steps = [step.strip() for step in content.split(",") if step.strip()]
            logger.info(f"Continuing investigation with {len(steps)} remaining steps")
            return {"plan": steps}

    async def parse_response(state: PlanExecuteState) -> PlanExecuteState:
        """Parse the response into structured analysis with strong typing."""
        if not state.get("response"):
            logger.warning("No response to parse")
            # Create state copy without conflicting fields
            clean_state = {k: v for k, v in state.items() if k != 'error'}
            return PlanExecuteState(
                **clean_state,
                error="No response to parse"
            )

        # Create the analysis class and parser
        from src.syntax import create_analysis_class
        Analysis = create_analysis_class(candidate_root_causes)
        pydantic_parser = PydanticOutputParser(pydantic_object=Analysis)

        try:
            # Try to parse the response text as structured output - Pydantic will validate
            analysis = pydantic_parser.parse(state["response"])
            logger.info(f"Successfully parsed structured response with root_causes: {analysis.root_causes}")
            
            # 💾 SAVE FINAL RESPONSE: Save the complete diagnostic result to file
            problem_info = {
                "problem_id": state.get("problem_id", "unknown"),
                "time_range": state.get("time_range", "unknown"),
                "input_description": state.get("input", "unknown"),
                "total_investigation_steps": len(state.get("past_steps", [])),
                "root_causes_count": len(analysis.root_causes) if hasattr(analysis, 'root_causes') else 0
            }
            save_final_response(state["response"], problem_info)
            
            # Create state copy without conflicting fields
            clean_state = {k: v for k, v in state.items() if k not in ['analysis', 'error']}
            return PlanExecuteState(
                **clean_state,
                analysis=analysis,  # Pydantic will validate this is an Analysis object
                error=""  # Clear any previous errors
            )

        except Exception as e:
            # Catch all exceptions and put them in the error field
            logger.warning(f"Parsing failed: {e}")
            logger.warning(f"Response text: {state['response']}")

            # Prepare error context for replanning
            error_context = f"Parsing failed: {str(e)}"
            if isinstance(e, OutputParserException):
                # Add format instructions for output parser errors
                retry_message = NAIVE_RETRY_WITH_ERROR_PROMPT.format(
                    prompt=pydantic_parser.get_format_instructions(),
                    completion=state["response"],
                    error=str(e),
                )
                updated_past_steps = state.get("past_steps", []) + [("Parsing error context", retry_message)]
            else:
                updated_past_steps = state.get("past_steps", [])

            # Create state copy without conflicting fields
            clean_state = {k: v for k, v in state.items() if k not in ['error', 'past_steps', 'plan', 'response']}
            return PlanExecuteState(
                **clean_state,
                error=error_context,
                past_steps=updated_past_steps,
                plan=[],  # Reset plan to force replanning
                response=""  # Clear response to force new generation
            )

    def should_end_after_agent(state: PlanExecuteState) -> str:
        """Determine next step after agent execution with strong typing."""
        past_steps = state.get("past_steps", [])
        
        # Early termination: if we have strong evidence, skip remaining steps
        if PERFORMANCE_CONFIG["enable_early_termination"] and len(past_steps) >= 3 and has_strong_evidence(past_steps):
            logger.info("🚀 OPTIMIZATION: Strong evidence found, triggering early termination to save time")
            return "replan"
        
        if state["plan"]:
            return "agent"  # Continue executing remaining plan steps
        else:
            return "replan"  # No more steps, go to replanning

    def should_end_after_replan(state: PlanExecuteState) -> str:
        """Determine next step after replanning with strong typing."""
        if state["response"]:
            return "parse"  # Go to parsing if we have a final response
        else:
            return "agent"  # Continue with agent steps if we have more plan

    def should_end_after_parse(state: PlanExecuteState) -> str:
        """Determine next step after parsing with strong typing."""
        if state["analysis"]:
            # Successfully parsed analysis - we're done
            return END
        else:
            # No analysis - replan immediately
            return "replan"

    # Build the workflow graph
    workflow = StateGraph(PlanExecuteState)

    # Add nodes
    workflow.add_node("planner", plan_step)
    workflow.add_node("agent", execute_step)
    workflow.add_node("replan", replan_step)
    workflow.add_node("parse", parse_response)

    # Add edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_end_after_agent,
        ["replan", "agent"],  # Either go to replan or continue with more agent steps
    )
    workflow.add_conditional_edges(
        "replan",
        should_end_after_replan,
        [
            "parse",
            "agent",
        ],  # Either go to parse (final response) or continue investigation
    )
    workflow.add_conditional_edges(
        "parse",
        should_end_after_parse,
        ["replan", END],  # Either retry via replan or end successfully
    )

    # Compile the workflow
    app = workflow.compile()
    return app


@asynccontextmanager
async def create_plan_execute_agent(
    tools: List[BaseTool], time_range: str, candidate_root_causes: List[str]
) -> AsyncGenerator[Any, None]:
    """Create a plan-and-execute diagnostic agent."""
    with tracer.start_as_current_span("PlanExecuteRCA"):
        workflow = await create_plan_execute_workflow(
            tools, time_range, candidate_root_causes
        )
        yield workflow


async def compute(
    chain: RunnableLambda,
    time_range: str,
    candidate_root_causes: List[str],
    max_retries: int = 3,
) -> BaseModel:
    logger.info(f"Time range: {time_range}")
    logger.info(f"Candidate root causes: {candidate_root_causes}")

    try:
        response = await chain.ainvoke(
            {"messages": [HumanMessage(content="Let's begin.")]}
        )

        analysis = response["structured_response"]
        logger.info(
            f"Successfully obtained valid analysis with root_causes: {analysis.root_causes}"
        )
        return analysis

    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        from src.syntax import create_analysis_class

        Analysis = create_analysis_class(candidate_root_causes)
        return Analysis(root_causes=random.choices(candidate_root_causes))


async def compute_plan_execute(
    workflow: Any,
    time_range: str,
    candidate_root_causes: List[str],
    input_description: str = "Investigate system failure and identify root causes",
    problem_id: str = "unknown",
) -> BaseModel:
    """Execute plan-and-execute diagnostic workflow."""
    logger.info(f"Starting plan-and-execute diagnosis for time range: {time_range}")
    logger.info(f"Candidate root causes: {candidate_root_causes}")
    
    # 🐛 Show debug configuration at startup
    print_debug_startup_info()

    try:
        # Create the properly typed state class
        PlanExecuteState = create_plan_execute_state_class(candidate_root_causes)
        
        # Initialize the workflow state
        initial_state = PlanExecuteState(
            input=input_description,
            time_range=time_range,
            plan=[],
            past_steps=[],
            response="",
            analysis=None,
            error="",
            messages=[],  # Required by AgentState
            # Add problem_id to state for response saving
            problem_id=problem_id
        )

        # Execute the workflow - retry logic is built into the graph
        config = RunnableConfig(recursion_limit=sys.maxsize)  # Allow infinite retries
        final_state = None

        async for event in workflow.astream(initial_state, config=config):
            for k, v in event.items():
                if k != "__end__":
                    logger.debug(f"Finished: {k}")
                    final_state = v

        # Extract the analysis from the final state
        if final_state and final_state.get("analysis"):
            analysis = final_state["analysis"]
            logger.info(
                f"Plan-and-execute diagnosis completed successfully with root_causes: {analysis.root_causes}"
            )
            return analysis
        else:
            # Check if we have error information
            error_msg = "Plan-execute workflow failed to produce analysis"
            if final_state and final_state.get("error"):
                error_msg += f": {final_state['error']}"
            raise RuntimeError(error_msg)

    except Exception as e:
        logger.error(f"Plan-and-execute diagnosis failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Re-raise the exception instead of falling back - let it fail properly
        raise e
