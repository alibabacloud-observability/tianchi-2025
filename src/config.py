#!/usr/bin/env python

import os
import logging
# from langchain_core.rate_limiters import InMemoryRateLimiter  # Not available in older versions
from langchain_openai import ChatOpenAI
# from langchain_mcp_adapters.client import MultiServerMCPClient  # Not available in Python 3.8
from src.mcp.minimal_mcp_client import MinimalMCPClient
from opentelemetry import trace

# Suppress OpenTelemetry attribute warnings to reduce noise
logging.getLogger('opentelemetry.attributes').setLevel(logging.ERROR)

tracer = trace.get_tracer(__name__)

# Performance optimization settings
PERFORMANCE_CONFIG = {
    "max_context_steps": int(os.getenv("MAX_CONTEXT_STEPS", "5")),  # Limit context to recent N steps
    "summary_length_limit": int(os.getenv("SUMMARY_LENGTH_LIMIT", "500")),  # Max chars per step summary
    "early_termination_threshold": int(os.getenv("EARLY_TERM_THRESHOLD", "2")),  # Evidence count for early stop
    "enable_context_compression": os.getenv("ENABLE_COMPRESSION", "true").lower() == "true",
    "enable_early_termination": os.getenv("ENABLE_EARLY_TERM", "true").lower() == "true",
}

# Debug configuration for message inspection
DEBUG_CONFIG = {
    "debug_level": os.getenv("DEBUG_LEVEL", "NONE").upper(),  # NONE, BASIC, DETAILED, FULL, COMPARISON
    "max_message_preview": int(os.getenv("DEBUG_MESSAGE_PREVIEW", "300")),  # Max chars to show per message
    "debug_to_file": os.getenv("DEBUG_TO_FILE", "false").lower() == "true",  # Save debug to file
    "debug_file_path": os.getenv("DEBUG_FILE_PATH", "debug_messages.log"),
    "show_tool_results": os.getenv("DEBUG_TOOL_RESULTS", "true").lower() == "true",
    "show_ai_responses": os.getenv("DEBUG_AI_RESPONSES", "true").lower() == "true",
}

# Final response output configuration - independent from debug system
RESPONSE_OUTPUT_CONFIG = {
    "save_final_response": os.getenv("SAVE_FINAL_RESPONSE", "false").lower() == "true",  # Enable/disable response saving
    "response_file_path": os.getenv("RESPONSE_FILE_PATH", "final_responses.json"),  # Output file path
    "response_file_format": os.getenv("RESPONSE_FILE_FORMAT", "pretty").lower(),  # pretty, compact, or timestamped
    "append_mode": os.getenv("RESPONSE_APPEND_MODE", "true").lower() == "true",  # Append vs overwrite
    "include_metadata": os.getenv("RESPONSE_INCLUDE_META", "true").lower() == "true",  # Include timestamp and problem info
}

model = ChatOpenAI(
    model="qwen3-coder-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv(
        "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        # "OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    extra_body={"max_input_tokens": 997952},
    max_retries=10,
)
