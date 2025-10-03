# Alibaba Cloud ARMS Python Agent

Pre-built wheel files for ARMS (Application Real-Time Monitoring Service) instrumentation.

## Installation

```bash
# Install all packages
pip install aliyun-python-agent/*.whl

# Or install specific ones
pip install aliyun-python-agent/aliyun_instrumentation_langchain-*.whl
pip install aliyun-python-agent/aliyun_instrumentation_openai-*.whl
```

## Usage

```bash
# Set ARMS credentials
export ARMS_APP_NAME=aiops-rca
export ARMS_REGION_ID=cn-hangzhou
export ARMS_LICENSE_KEY=your-license-key

# Run with instrumentation
aliyun-instrument python run_aiops_rca.py dataset/B榜题目.jsonl --local-mode
```

## Packages Included

- **Core**: OpenTelemetry SDK, ARMS extension, semantic conventions
- **LLM**: LangChain, OpenAI, DashScope instrumentation
- **Web**: FastAPI, Flask, Django support
- **Database**: SQLAlchemy, Redis
- **HTTP**: Requests, HTTPX, AIOHTTP
- **AI/ML**: LlamaIndex, Dify, vLLM, SGLang, MCP

## Documentation

- [ARMS Python Agent](https://help.aliyun.com/zh/arms/application-monitoring/user-guide/manually-install-the-python-probe)
- [LLM Monitoring](https://help.aliyun.com/zh/arms/application-monitoring/user-guide/use-the-arms-agent-for-python-to-monitor-llm-applications)

