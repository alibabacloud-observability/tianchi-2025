#!/usr/bin/env python

from pydantic import BaseModel, Field, validator
import warnings
import os

# 设置Pydantic验证器重用
os.environ['PYDANTIC_ALLOW_REUSE'] = 'True'
warnings.filterwarnings("ignore", message="duplicate validator function")
from typing import List, Dict, Any, Type


class Action(BaseModel):
    tool_name: str = Field(description="Name of the tool that was called")
    tool_args: Dict[str, Any] = Field(description="Arguments passed to the tool")
    result_summary: str = Field(description="Summary of the tool's result")


class EvidenceChain(BaseModel):
    motivation: List[str] = Field(
        default_factory=list,
        description="List of questions the agent wants to address and investigate"
    )
    actions: List[Action] = Field(
        default_factory=list,
        description="List of actions taken during the analysis"
    )
    observations: List[str] = Field(
        default_factory=list,
        description="List of strings describing observations made during analysis"
    )
    decision: List[str] = Field(
        default_factory=list,
        description="List of decisions made - either the next investigation step or that the root cause has been located"
    )


class ProblemData(BaseModel):
    problem_id: str = Field(description="Unique identifier for the problem")
    time_range: str = Field(description="Time range when the problem occurred")
    candidate_root_causes: List[str] = Field(
        description="List of potential root causes to investigate")


def create_analysis_class(candidate_root_causes: List[str]) -> Type[BaseModel]:

    class Analysis(BaseModel):
        root_causes: List[str] = Field(
            description=f"The root causes of the issue. Must be a non-empty subset of the candidate root causes: {candidate_root_causes}."
        )
        
        @validator('root_causes')
        def validate_root_causes(cls, v):
            if not v or len(v) == 0:
                raise ValueError("root_causes must not be empty")

            invalid_causes = [
                cause for cause in v if cause not in candidate_root_causes]
            if invalid_causes:
                raise ValueError(
                    f"root_causes contains invalid entries not in candidate_root_causes: {invalid_causes}. "
                    f"Valid candidates are: {candidate_root_causes}"
                )

            return v
        evidences: List[str] = Field(
            default_factory=list,
            description="""Natural language descriptions of evidence collected during analysis. Each evidence should be:
            - A clear, factual statement about observed anomalies, metrics, or system behavior
            - Include specific values, timestamps, and percentage changes when available
            - Focus on concrete observations rather than interpretations
            - Provide factual data that supports or refutes the candidate root causes
            """
        )
        evidence_chain: EvidenceChain = Field(
            default_factory=EvidenceChain,
            description="The evidence chain from the analysis structured as motivation, actions, observations, and decisions"
        )

    return Analysis
