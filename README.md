# Tianchi 2025 AIOps Challenge

本repo是2025 AI原生编程挑战赛的示例代码仓库 （不定期更新）

https://tianchi.aliyun.com/specials/promotion/2025aicloudnativecompetition

## Root Cause Analysis Tool

### Input Data

Place your input data in `dataset/input.jsonl` following the required format.

### Environment Variables

```bash
export ALIBABA_CLOUD_ACCESS_KEY_ID="your-access-key-id"
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="your-access-key-secret"
# If you registered _before_ Mon 22 Sep 2025:
export ALIBABA_CLOUD_ROLE_ARN="acs:ram::1672753017899339:role/tianchi-user-a"
# If you registered _after_  Mon 22 Sep 2025:
# export ALIBABA_CLOUD_ROLE_ARN="acs:ram::1672753017899339:role/tianchi-user-1"
export ALIBABA_CLOUD_ROLE_SESSION_NAME="my-sls-access"
```

### Running the Analysis

Execute the analysis script:

```bash
# For test suite B, update `dataset/input.jsonl`, and uncomment the following line:
# perl -i -pe 's/quanxi-tianchi-test/tianchi-workspace/g' notebook/test_cms_query.py
./run_analysis.sh
```

### Output Results

Results will be generated in `dataset/output.jsonl` with the identified root causes for each problem case.
