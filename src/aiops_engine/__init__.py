"""
AIOps RCA Engine - 方案A渐进式优化实现
基于数据驱动的智能根因分析引擎
"""

from .parallel_data_coordinator import ParallelDataCoordinator, DataBundle, CollectionTask
from .anomaly_detection import (
    AnomalyDetectionEngine, 
    CorrelationAnalysisEngine,
    Anomaly,
    ServiceCorrelation,
    AnomalyType,
    SeverityLevel
)
from .expert_rules import ExpertRulesEngine, ExpertRule, RootCauseScore
from .aiops_rca_engine import AIOpsRCAEngine

__all__ = [
    # Main engine
    'AIOpsRCAEngine',
    
    # Data coordination
    'ParallelDataCoordinator',
    'DataBundle',
    'CollectionTask',
    
    # Anomaly detection
    'AnomalyDetectionEngine',
    'CorrelationAnalysisEngine', 
    'Anomaly',
    'ServiceCorrelation',
    'AnomalyType',
    'SeverityLevel',
    
    # Expert rules
    'ExpertRulesEngine',
    'ExpertRule',
    'RootCauseScore'
]

__version__ = "1.0.0"
__author__ = "AIOps Team"
__description__ = "Data-driven intelligent root cause analysis engine"
