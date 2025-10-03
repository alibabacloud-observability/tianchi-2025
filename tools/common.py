"""Common utilities and models for Aliyun Observability Tools

This module contains shared functionality used across all toolkit implementations,
including client creation, time parsing, and query execution utilities.
All functions use strict typing without Dict or Any types.
"""

import re
from datetime import datetime, timedelta
from typing import List, Optional, Union
from pydantic import BaseModel, Field

# Aliyun SDK imports
from alibabacloud_cms20240330.client import Client as CmsClient
from alibabacloud_cms20240330.models import GetEntityStoreDataRequest
from alibabacloud_sls20201230.client import Client as SLSClient
from alibabacloud_credentials.client import Client as CredClient
from alibabacloud_credentials.models import Config as CredConfig
from alibabacloud_credentials.utils import auth_util
from alibabacloud_tea_openapi import models as open_api_models

from .constants import ALIBABA_CLOUD_ACCESS_KEY_ID, ALIBABA_CLOUD_ACCESS_KEY_SECRET, ALIBABA_CLOUD_ROLE_ARN, ALIBABA_CLOUD_ROLE_SESSION_NAME


# Common response models for helper functions
# Specific data structure models
class EntityData(BaseModel):
    """Individual entity information"""

    entity_id: Optional[str] = Field(
        None, description="Entity unique identifier")
    entity_name: Optional[str] = Field(None, description="Entity display name")
    entity_type: Optional[str] = Field(None, description="Entity type")
    domain: Optional[str] = Field(None, description="Entity domain")
    attributes: Optional[dict] = Field(None, description="Entity attributes")
    properties: Optional[dict] = Field(None, description="Entity properties")
    metadata: Optional[dict] = Field(None, description="Entity metadata")
    timestamp: Optional[int] = Field(None, description="Data timestamp")


class MetricData(BaseModel):
    """Individual metric data point"""

    timestamp: Optional[int] = Field(None, description="Metric timestamp")
    value: Optional[Union[float, int, str]] = Field(
        None, description="Metric value")
    metric_name: Optional[str] = Field(None, description="Metric name")
    entity_id: Optional[str] = Field(None, description="Associated entity ID")
    labels: Optional[dict] = Field(None, description="Metric labels")
    tags: Optional[dict] = Field(None, description="Metric tags")


class LogData(BaseModel):
    """Individual log entry"""

    timestamp: Optional[int] = Field(None, description="Log timestamp")
    message: Optional[str] = Field(None, description="Log message")
    level: Optional[str] = Field(None, description="Log level")
    source: Optional[str] = Field(None, description="Log source")
    entity_id: Optional[str] = Field(None, description="Associated entity ID")
    fields: Optional[dict] = Field(None, description="Additional log fields")


class TraceData(BaseModel):
    """Individual trace or span data"""

    trace_id: Optional[str] = Field(None, description="Trace ID")
    span_id: Optional[str] = Field(None, description="Span ID")
    operation_name: Optional[str] = Field(None, description="Operation name")
    duration: Optional[int] = Field(
        None, description="Duration in microseconds")
    start_time: Optional[int] = Field(None, description="Start timestamp")
    end_time: Optional[int] = Field(None, description="End timestamp")
    status_code: Optional[str] = Field(None, description="Status code")
    entity_id: Optional[str] = Field(None, description="Associated entity ID")
    tags: Optional[dict] = Field(None, description="Span tags")


class EventData(BaseModel):
    """Individual event data"""

    event_id: Optional[str] = Field(None, description="Event ID")
    timestamp: Optional[int] = Field(None, description="Event timestamp")
    event_type: Optional[str] = Field(None, description="Event type")
    title: Optional[str] = Field(None, description="Event title")
    description: Optional[str] = Field(None, description="Event description")
    entity_id: Optional[str] = Field(None, description="Associated entity ID")
    severity: Optional[str] = Field(None, description="Event severity")
    attributes: Optional[dict] = Field(None, description="Event attributes")


class WorkspaceData(BaseModel):
    """Individual workspace information"""

    id: Optional[str] = Field(None, description="Workspace ID")
    name: Optional[str] = Field(None, description="Workspace name")
    description: Optional[str] = Field(
        None, description="Workspace description")
    region: Optional[str] = Field(None, description="Workspace region")
    status: Optional[str] = Field(None, description="Workspace status")


class EntitySetData(BaseModel):
    """Individual entity set information"""

    domain: Optional[str] = Field(None, description="Entity set domain")
    name: Optional[str] = Field(None, description="Entity set name")
    display_name: Optional[str] = Field(
        None, description="Entity set display name")
    description: Optional[str] = Field(
        None, description="Entity set description")


class QueryResult(BaseModel):
    """Result from CMS query execution"""

    error: bool = Field(..., description="Whether an error occurred")
    data: List[dict] = Field(
        default_factory=list,
        description="Query result data")
    query: str = Field(..., description="Executed query")
    workspace: str = Field(..., description="Workspace name")
    headers: List[str] = Field(
        default_factory=list,
        description="Column headers")
    row_count: int = Field(0, description="Number of rows returned")
    message: Optional[str] = Field(
        None, description="Success or error message")


class TimeRange(BaseModel):
    """Time range for queries"""

    from_time: Union[str, int] = Field(..., description="Start time")
    to_time: Union[str, int] = Field(..., description="End time")


class EntityContext(BaseModel):
    """Entity context information"""

    domain: str = Field(..., description="Entity domain")
    type: str = Field(..., description="Entity type")
    id: Optional[str] = Field(None, description="Entity ID")


# Common utility functions
def create_unified_config() -> open_api_models.Config:
    """Create unified Alibaba Cloud client configuration"""
    role_arn = auth_util.environment_role_arn or 'acs:ram::1672753017899339:role/tianchi-user-a'
    access_key_id = auth_util.environment_access_key_id
    access_key_secret = auth_util.environment_access_key_secret
    role_session_name = auth_util.environment_role_session_name or 'my-sls-access'

    if role_arn:
        # Use Role ARN for authentication
        credentials_client = CredClient(
            CredConfig(
                type="ram_role_arn",
                access_key_id=access_key_id,
                access_key_secret=access_key_secret,
                role_arn=role_arn,
                role_session_name=role_session_name,
            )
        )
        return open_api_models.Config(credential=credentials_client)

    # Use Access Key authentication
    return open_api_models.Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
    )


def create_cms_client(region: str) -> CmsClient:
    """Create CMS client for a specific region"""
    config = create_unified_config()
    config.endpoint = f"cms.{region}.aliyuncs.com"
    return CmsClient(config)


def create_sls_client(region: str) -> SLSClient:
    """Create SLS client for a specific region"""
    config = create_unified_config()
    config.endpoint = f"{region}.log.aliyuncs.com"
    return SLSClient(config)


def parse_time_expression(time_expr: Union[str, int]) -> int:
    """Parse time expression to Unix timestamp (seconds)"""
    if isinstance(time_expr, int):
        # Already a timestamp
        if time_expr > 1e10:  # Likely milliseconds
            return int(time_expr / 1000)
        return time_expr

    if isinstance(time_expr, str):
        if time_expr.startswith("now"):
            current = datetime.now()
            if time_expr == "now":
                return int(current.timestamp())

            # Parse relative time like "now-5m", "now-1h", etc.
            match = re.match(r"now-(\d+)([smhd])", time_expr)
            if match:
                amount, unit = int(match.group(1)), match.group(2)
                if unit == "s":
                    delta = timedelta(seconds=amount)
                elif unit == "m":
                    delta = timedelta(minutes=amount)
                elif unit == "h":
                    delta = timedelta(hours=amount)
                elif unit == "d":
                    delta = timedelta(days=amount)
                else:
                    delta = timedelta(minutes=amount)  # Default

                return int((current - delta).timestamp())
        else:
            # Try to parse as direct timestamp string
            try:
                timestamp = int(float(time_expr))
                if timestamp > 1e10:  # Milliseconds
                    return int(timestamp / 1000)
                return timestamp
            except ValueError:
                return int(datetime.now().timestamp())

    return int(datetime.now().timestamp())


def execute_cms_query(
    cms_client: CmsClient,
    workspace_name: str,
    query: str,
    from_time: Union[str, int],
    to_time: Union[str, int],
    limit: int = 1000,
) -> QueryResult:
    """Execute CMS SPL query and return structured result

    Enhanced version that handles time expression parsing and provides
    standardized response format matching the original MCP implementation.

    Args:
        cms_client: CMS client instance
        workspace_name: CMS workspace name
        query: SPL query statement
        from_time: Start time - Unix timestamp or relative expression (now-5m)
        to_time: End time - Unix timestamp or relative expression (now)
        limit: Maximum number of results to return

    Returns:
        QueryResult with standardized response format
    """
    try:
        # Import locally to avoid circular imports
        from .time_utils import TimeRangeParser

        # Parse time expressions to timestamps
        time_range = TimeRangeParser.parse_time_range(from_time, to_time)

        request = GetEntityStoreDataRequest(
            query=query,
            from_=time_range.from_timestamp,
            to=time_range.to_timestamp
        )

        # Get the response body directly as shown in original MCP
        # implementation
        response_body = cms_client.get_entity_store_data(
            workspace_name, request).body

        # Extract data and header from the response body
        # data is List[List[str]] - rows of data
        # header is List[str] - column names
        data = response_body.data or []
        header = response_body.header or []

        # Convert to list of dictionaries using headers as keys
        result_data = []
        if data and header:
            for row in data:
                row_dict = {}
                for i, value in enumerate(row):
                    if i < len(header):
                        row_dict[header[i]] = value
                result_data.append(row_dict)

        # Apply limit if needed
        if len(result_data) > limit:
            result_data = result_data[:limit]

        return QueryResult(
            error=False,
            data=result_data,
            query=query,
            workspace=workspace_name,
            headers=header,
            row_count=len(result_data),
            message=(
                f"success, returned {len(result_data)} records"
                if result_data
                else "No data found"
            ),
        )

    except Exception as e:
        return QueryResult(
            error=True,
            data=[],
            query=query,
            workspace=workspace_name,
            headers=[],
            row_count=0,
            message=f"Query execution failed: {str(e)}",
        )


def build_entity_filter(entity_ids: Optional[str]) -> str:
    """Build entity ID filter for SPL queries"""
    if not entity_ids:
        return ""

    ids = [f"'{id_.strip()}'" for id_ in entity_ids.split(",") if id_.strip()]
    if not ids:
        return ""

    if len(ids) == 1:
        return f" | where __id__ = {ids[0]}"
    else:
        return f" | where __id__ in ({', '.join(ids)})"


def get_current_time_context() -> str:
    """Get current time context for queries"""
    current_time = datetime.now()
    return f"当前时间: {
        current_time.strftime('%Y-%m-%d %H:%M:%S')}, 当前时间戳: {
        int(current_time.timestamp())} "
