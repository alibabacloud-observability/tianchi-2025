"""PaaS Data Tools - Observability Data Query Toolkit

Provides structured data query tools for metrics, logs, and traces.
All functions use strict typing with Pydantic models and use SDK return values directly.
"""

from typing import Optional, Union, List
from langchain.tools import tool

from .common import create_cms_client, execute_cms_query, QueryResult
from .constants import REGION_ID, WORKSPACE_NAME


def _build_entity_ids_param(entity_ids: Optional[List[str]]) -> str:
    """Build entity IDs parameter for SPL queries"""
    if not entity_ids:
        return ""

    # Filter out empty strings and strip whitespace
    clean_ids = [id.strip() for id in entity_ids if id and id.strip()]
    if not clean_ids:
        return ""
    
    quoted = [f"'{id}'" for id in clean_ids]
    return f", ids=[{','.join(quoted)}]"


# Tool implementations
@tool
def umodel_get_golden_metrics(
    domain: str,
    entity_set_name: str,
    from_time: Union[str, int],
    to_time: Union[str, int],
    entity_ids: Optional[List[str]] = None,
) -> QueryResult:
    """获取实体的黄金指标（关键性能指标）数据。包括延迟、吞吐量、错误率、饱和度等核心指标。
    ## 参数获取: 1)搜索实体集→ 2)获取实体ID(可选) → 3)执行查询
    - domain,entity_set_name: `umodel_search_entity_set(search_text="apm")`
    - entity_ids: `umodel_get_entities()` (可选)

    Args:
        domain: 实体域, cannot be '*'
        entity_set_name: 实体类型, cannot be '*'
        entity_ids: List of entity IDs
        from_time: 开始时间: Unix时间戳
        to_time: 结束时间: Unix时间戳

    Returns:
        包含黄金指标数据的响应对象
    """
    entity_ids_param = _build_entity_ids_param(entity_ids)

    query = f".entity_set with(domain='{domain}', name='{entity_set_name}'{entity_ids_param}) | entity-call get_golden_metrics()"

    cms_client = create_cms_client(REGION_ID)
    return execute_cms_query(
        cms_client,
        WORKSPACE_NAME,
        query,
        from_time,
        to_time,
        1000,
    )


@tool
def umodel_get_logs(
    domain: str,
    entity_set_name: str,
    log_set_domain: str,
    log_set_name: str,
    from_time: Union[str, int],
    to_time: Union[str, int],
    entity_ids: Optional[List[str]] = None,
) -> QueryResult:
    """获取实体相关的日志数据，用于故障诊断、性能分析、审计等场景。
    ## 参数获取: 1)搜索实体集→ 2)列出LogSet→ 3)获取实体ID(可选) → 4)执行查询
    - domain,entity_set_name: `umodel_search_entity_set(search_text="apm")`
    - log_set_domain,log_set_name: `umodel_list_data_set(data_set_types="log_set")`返回domain/name
    - entity_ids: `umodel_get_entities()` (可选)

    Args:
        domain: 实体域, cannot be '*'
        entity_set_name: 实体类型, cannot be '*'
        log_set_domain: LogSet domain
        log_set_name: LogSet name
        entity_ids: List of entity IDs
        from_time: 开始时间: Unix时间戳
        to_time: 结束时间: Unix时间戳

    Returns:
        包含日志数据的响应对象
    """
    entity_ids_param = _build_entity_ids_param(entity_ids)

    query = f".entity_set with(domain='{domain}', name='{entity_set_name}'{entity_ids_param}) | entity-call get_log('{log_set_domain}', '{log_set_name}')"

    cms_client = create_cms_client(REGION_ID)
    return execute_cms_query(
        cms_client,
        WORKSPACE_NAME,
        query,
        from_time,
        to_time,
        1000,
    )


@tool
def umodel_get_traces(
    domain: str,
    entity_set_name: str,
    trace_set_domain: str,
    trace_set_name: str,
    trace_ids: List[str],
    from_time: Union[str, int],
    to_time: Union[str, int],
) -> QueryResult:
    """获取指定trace ID的详细trace数据，包括所有span、时序数据和元数据。用于深入分析慢trace和错误trace。
    ## 参数获取: 1)搜索trace → 2)获取详细信息
    - trace_ids: 通常从`umodel_search_traces()`工具输出中获得
    - domain,entity_set_name: `umodel_search_entity_set(search_text="apm")`
    - trace_set_domain,trace_set_name: `umodel_list_data_set(data_set_types="trace_set")`返回domain/name

    Args:
        domain: EntitySet域名，如'apm'
        entity_set_name: EntitySet名称，如'apm.service'
        trace_set_domain: TraceSet域名，如'apm'
        trace_set_name: TraceSet名称，如'apm.trace.common'
        trace_ids: trace ID列表
        from_time: 开始时间: Unix时间戳
        to_time: 结束时间: Unix时间戳

    Returns:
        包含详细trace数据的响应对象
    """
    # Build trace_ids parameter (following MCP server implementation)
    if not trace_ids:
        raise ValueError("trace_ids is required and cannot be empty")

    clean_trace_ids = [tid.strip() for tid in trace_ids if tid and tid.strip()]
    if not clean_trace_ids:
        raise ValueError("trace_ids is required and cannot be empty")

    quoted_filters = [f"traceId='{tid}'" for tid in clean_trace_ids]
    trace_ids_param = " or ".join(quoted_filters)

    # Implementation based on trace_ids filtering (following MCP server implementation)
    query = f".entity_set with(domain='{domain}', name='{entity_set_name}') | entity-call get_trace('{trace_set_domain}', '{trace_set_name}') | where {trace_ids_param} | extend duration_ms = cast(duration as double) / 1000000 | project-away duration | sort traceId desc, duration_ms desc | limit 1000"

    cms_client = create_cms_client(REGION_ID)
    return execute_cms_query(
        cms_client,
        WORKSPACE_NAME,
        query,
        from_time,
        to_time,
        1000,
    )


@tool
def umodel_search_traces(
    domain: str,
    entity_set_name: str,
    trace_set_domain: str,
    trace_set_name: str,
    from_time: Union[str, int],
    to_time: Union[str, int],
    entity_ids: Optional[List[str]] = None,
    has_error: Optional[bool] = None,
    min_duration_ms: Optional[float] = None,
    max_duration_ms: Optional[float] = None,
    limit: Optional[int] = 100,
) -> QueryResult:
    """基于过滤条件搜索trace并返回摘要信息。支持按持续时间、错误状态、实体ID过滤，返回traceID用于详细分析。
    ## 参数获取: 1)搜索实体集→ 2)列出TraceSet→ 3)获取实体ID(可选) → 4)执行搜索
    - domain,entity_set_name: `umodel_search_entity_set(search_text="apm")`
    - trace_set_domain,trace_set_name: `umodel_list_data_set(data_set_types="trace_set")`返回domain/name
    - entity_ids: `umodel_get_entities()` (可选)
    - 过滤条件: min_duration_ms(慢trace)、has_error(错误trace)、max_duration_ms等

    Args:
        domain: EntitySet域名，如'apm'
        entity_set_name: EntitySet名称，如'apm.service'
        trace_set_domain: TraceSet域名，如'apm'
        trace_set_name: TraceSet名称，如'apm.trace.common'
        entity_ids: 实体ID列表
        has_error: 按错误状态过滤（true表示错误trace，false表示成功trace）
        min_duration_ms: 最小trace持续时间（毫秒）
        max_duration_ms: 最大trace持续时间（毫秒）
        limit: 返回的最大trace摘要数量
        from_time: 开始时间: Unix时间戳(秒/毫秒)或相对时间(now-5m)
        to_time: 结束时间: Unix时间戳(秒/毫秒)或相对时间(now)

    Returns:
        包含trace搜索结果的响应对象
    """
    # Build entity_ids parameter
    entity_ids_param = _build_entity_ids_param(entity_ids)

    # Build filter conditions (following MCP server implementation)
    filter_params = []

    if min_duration_ms is not None:
        filter_params.append(
            f"cast(duration as bigint) > {int(min_duration_ms * 1000000)}"
        )

    if max_duration_ms is not None:
        filter_params.append(
            f"cast(duration as bigint) < {int(max_duration_ms * 1000000)}"
        )

    if has_error is not None:
        filter_params.append("cast(statusCode as varchar) = '2'")

    limit_value = 100
    if limit is not None and limit > 0:
        limit_value = int(limit)

    filter_param_str = ""
    if filter_params:
        filter_param_str = "| where " + " and ".join(filter_params)

    stats_str = "| extend duration_ms = cast(duration as double) / 1000000, is_error = case when cast(statusCode as varchar) = '2' then 1 else 0 end |  stats span_count = count(1), error_span_count = sum(is_error), duration_ms = max(duration_ms) by traceId | sort duration_ms desc, error_span_count desc | project traceId, duration_ms, span_count, error_span_count"

    # Use get_trace with filtering logic (following MCP server implementation)
    query = f".entity_set with(domain='{domain}', name='{entity_set_name}'{entity_ids_param}) | entity-call get_trace('{trace_set_domain}', '{trace_set_name}') {filter_param_str} {stats_str} | limit {limit_value}"

    cms_client = create_cms_client(REGION_ID)
    return execute_cms_query(
        cms_client,
        WORKSPACE_NAME,
        query,
        from_time,
        to_time,
        1000,
    )
