"""Alibaba Cloud Observability Tools

Minimal set of tools actually used by the RCA agents.
"""

from .paas_entity_tools import umodel_get_entities
from .paas_data_tools import (
    umodel_get_logs,
    umodel_get_golden_metrics,
    umodel_get_traces,
    umodel_search_traces,
)

# All tools combined
ALL_TOOLS = [
    umodel_get_entities,
    umodel_get_logs,
    umodel_get_golden_metrics,
    umodel_get_traces,
    umodel_search_traces,
]

__all__ = [
    # Entity tools
    "umodel_get_entities",
    # Data tools
    "umodel_get_logs",
    "umodel_get_golden_metrics",
    "umodel_get_traces",
    "umodel_search_traces",
    # Tool group
    "ALL_TOOLS",
]
