"""PaaS Entity Tools - Entity Management Toolkit

Provides structured entity query tools ported from umodel entity handlers.
All functions use strict typing with Pydantic models and use SDK return values directly.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain.tools import tool

from .common import create_cms_client, execute_cms_query, QueryResult
from .constants import REGION_ID, WORKSPACE_NAME


# Response models
class GetEntitiesResponse(BaseModel):
    """Response model for getting entities"""

    data: List[Dict[str, Any]] = Field(..., description="Entity data")
    message: str = Field(..., description="Success or error message")
    count: int = Field(..., description="Number of entities returned")
    error: bool = Field(..., description="Whether an error occurred")


def _build_entity_ids_param(entity_ids: Optional[str]) -> str:
    """Build entity IDs parameter for SPL queries"""
    if not entity_ids:
        return ""

    ids = [f"'{id_.strip()}'" for id_ in entity_ids.split(",") if id_.strip()]
    if not ids:
        return ""

    if len(ids) == 1:
        return f", entity_ids=[{ids[0]}]"
    else:
        return f", entity_ids=[{', '.join(ids)}]"


# Tool implementations
@tool
def umodel_get_entities(
    domain: str,
    entity_set_name: str,
    from_time: Union[str, int],
    to_time: Union[str, int],
    entity_ids: Optional[str] = None,
    limit: int = 20,
) -> QueryResult:
    """获取实体信息的PaaS API工具。

    ## 功能概述

    该工具用于检索实体信息，支持分页查询和精确ID查询。专注于实体的基础信息获取。

    ## 功能特点

    - **数量控制**: 默认返回20个实体，支持通过limit参数控制返回数量
    - **全量查询**: 支持获取指定实体集合下的所有实体（分页返回）
    - **精确查询**: 支持根据实体ID列表进行精确查询
    - **职责清晰**: 专注于基础实体信息获取，不包含复杂过滤逻辑

    ## 使用场景

    - **分页浏览**: 分页获取实体列表，适用于大量实体的展示场景
    - **精确查询**: 根据已知的实体ID列表批量获取实体详细信息
    - **全量获取**: 获取指定实体集合下的所有实体信息
    - **基础数据**: 为其他分析工具提供基础实体数据

    ## 参数说明

    - domain: 实体集合的域，如 'apm'、'infrastructure' 等
    - entity_set_name: 实体集合名称，如 'apm.service'、'host.instance' 等
    - entity_ids: 可选的逗号分隔实体ID字符串，用于精确查询指定实体
    - from_time/to_time: 查询时间范围，支持时间戳和相对时间表达式
    - limit: 返回多少个实体，默认20个，最大1000个

    ## 示例用法

    ```
    # 获取前20个服务实体（默认数量）
    umodel_get_entities(
        domain="apm",
        entity_set_name="apm.service"
    )

    # 获取前100个服务实体
    umodel_get_entities(
        domain="apm",
        entity_set_name="apm.service",
        limit=100
    )

    # 根据实体ID批量查询
    umodel_get_entities(
        domain="apm",
        entity_set_name="apm.service",
        entity_ids="service-1,service-2,service-3"
    )
    ```

    Args:
        domain: 实体域, cannot be '*'
        entity_set_name: 实体类型, cannot be '*'
        entity_ids: 可选的逗号分隔实体ID列表，用于精确查询指定实体
        from_time: 开始时间: Unix时间戳(秒/毫秒)或相对时间(now-5m)
        to_time: 结束时间: Unix时间戳(秒/毫秒)或相对时间(now)
        limit: 返回多少个实体，默认20个

    Returns:
        包含实体信息的响应对象，包括实体列表和查询元数据
    """
    # Build entity IDs parameter if provided
    entity_ids_param = _build_entity_ids_param(entity_ids)

    # 验证domain和entity_set_name不能为通配符
    if domain == "*":
        raise ValueError(
            "domain parameter cannot be '*', must be a specific domain like 'apm'"
        )
    if entity_set_name == "*":
        raise ValueError(
            "entity_set_name parameter cannot be '*', must be a specific entity type like 'apm.service'"
        )

    # 简化查询，只使用limit参数
    query = f".entity_set with(domain='{domain}', name='{entity_set_name}'{
        entity_ids_param}) | entity-call get_entities() | limit {limit} "

    cms_client = create_cms_client(REGION_ID)
    return execute_cms_query(cms_client, WORKSPACE_NAME,
                             query, from_time, to_time, limit)
