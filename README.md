# AIOps Root Cause Analysis (RCA) Agent

An intelligent AIOps system for automated root cause analysis using LangChain and LLMs.

## Quick Start

### Environment Variables

The system uses standard Alibaba Cloud SDK environment variables:

- `OPENAI_API_KEY` - OpenAI API key for LLM
- `OPENAI_BASE_URL` - API endpoint
- `ALIBABA_CLOUD_ACCESS_KEY_ID` - Alibaba Cloud access key ID
- `ALIBABA_CLOUD_ACCESS_KEY_SECRET` - Alibaba Cloud access key secret
- `ALIBABA_CLOUD_ROLE_ARN` - RAM role ARN for AssumeRole (see: https://developer.aliyun.com/article/1679516)
- `ALIBABA_CLOUD_ROLE_SESSION_NAME` - Session name for AssumeRole

These are read by the Alibaba Cloud SDK via `alibabacloud_credentials.utils.auth_util`.

```bash
# Configure LLM
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"

# Configure Alibaba Cloud credentials
export ALIBABA_CLOUD_ACCESS_KEY_ID="your-access-key-id"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your-access-key-secret"
# If you registered _before_ Mon 22 Sep 2025:
export ALIBABA_CLOUD_ROLE_ARN="acs:ram::1672753017899339:role/tianchi-user-a"
# If you registered _after_  Mon 22 Sep 2025:
# export ALIBABA_CLOUD_ROLE_ARN="acs:ram::1672753017899339:role/tianchi-user-1"
export ALIBABA_CLOUD_ROLE_SESSION_NAME="my-sls-access"
```

### Setup and Run

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run analysis on single problem with online mode
python run_aiops_rca.py dataset/B榜题目.jsonl --single-problem 002

# Results saved to: results/rca_result_002_<timestamp>.json
```

## Usage

### Single Problem Examples

```bash
# Single problem with online mode (recommended for testing)
python run_aiops_rca.py dataset/B榜题目.jsonl --single-problem 002

# Rule-based analysis: Use expert rules instead of LLM
python run_aiops_rca.py dataset/B榜题目.jsonl --aiops-engine --single-problem 002

# Offline mode: Use cached data (requires downloading first)
python scripts/download_data.py --single-problem 002
python run_aiops_rca.py dataset/B榜题目.jsonl --offline-mode --single-problem 002
```

### Batch Processing (Entire Dataset)

```bash
# Offline mode: Download all problems first, then analyze
python scripts/download_data.py
python run_aiops_rca.py dataset/B榜题目.jsonl --offline-mode

# Default: Online mode with real-time SDK queries (entire dataset)
python run_aiops_rca.py dataset/B榜题目.jsonl
```

## Project Structure

```
agent-example/
├── run_aiops_rca.py        # Main entry point
├── requirements.txt        # Dependencies
├── src/                    # Source code
│   ├── agents/            # Data collection agents
│   ├── aiops_engine/      # Analysis engine
│   ├── utils/             # Utilities
│   └── ...
├── tools/                  # Observability tools (Alibaba Cloud SDK)
├── scripts/               # Helper scripts
├── dataset/               # Sample problems
└── aliyun-python-agent/   # ARMS instrumentation packages (optional)
```

## Output

Results are saved to:
- `results/rca_result_<problem_id>_<timestamp>.json` - Root cause analysis with evidence
- `logs/aiops_rca_<timestamp>.log` - Detailed execution logs

## License

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
