#!/usr/bin/env python

"""Global constants for tools configuration.

These constants specify the project, region, and workspace
parameters across all tools in the system.
"""

# Hardcoded configuration values
import os


PROJECT_NAME = "proj-xtrace-a46b97cfdc1332238f714864c014a1b-cn-qingdao"
REGION_ID = "cn-qingdao"
WORKSPACE_NAME = "tianchi-workspace"

ALIBABA_CLOUD_ACCESS_KEY_ID = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_ID')
ALIBABA_CLOUD_ACCESS_KEY_SECRET = os.environ.get('ALIBABA_CLOUD_ACCESS_KEY_SECRET')
ALIBABA_CLOUD_ROLE_ARN = os.environ.get('ALIBABA_CLOUD_ROLE_ARN', 'acs:ram::1672753017899339:role/tianchi-user-a')
ALIBABA_CLOUD_ROLE_SESSION_NAME = os.environ.get('ALIBABA_CLOUD_ROLE_SESSION_NAME', 'aiops-rca-session')
