#!/usr/bin/env python

from typing import List
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.syntax import create_analysis_class


# Plan-and-Execute Models
class DiagnosticPlan(BaseModel):
    """Plan for failure diagnosis with step-by-step investigation tasks."""

    steps: List[str] = Field(
        description="Ordered steps for investigating the failure, each step should be a specific diagnostic task"
    )


class DiagnosticResponse(BaseModel):
    """Response containing either an updated plan or final diagnosis."""

    response_type: str = Field(
        description="Type of response: 'plan' for updated investigation plan, 'response' for final diagnosis"
    )
    content: str = Field(
        description="Response content - either comma-separated steps for plan or structured analysis for response"
    )


@tool
def create_diagnostic_plan(steps: List[str]) -> DiagnosticPlan:
    """Create a diagnostic plan with ordered investigation steps.

    Args:
        steps: List of specific diagnostic tasks to be performed in order.
               Each step should be a clear, actionable investigation task.

    Returns:
        DiagnosticPlan: A structured plan with the provided steps.
    """
    return DiagnosticPlan(steps=steps)


@tool
def provide_diagnostic_response(response_type: str, content: str) -> DiagnosticResponse:
    """Provide either an updated plan or final diagnosis response.

    Args:
        response_type: Either "plan" for updated investigation plan or "response" for final diagnosis
        content: The response content - either comma-separated steps for plan or structured analysis for response

    Returns:
        DiagnosticResponse: Structured response with type and content.
    """
    return DiagnosticResponse(response_type=response_type, content=content)


def create_analysis_tool(candidate_root_causes: List[str]):
    """Create a tool that produces structured analysis."""
    Analysis = create_analysis_class(candidate_root_causes)

    @tool
    def provide_final_analysis(
        root_causes: List[str],
        evidences: List[str],
        evidence_chain_actions: List[str],
        evidence_chain_tool_names: List[str],
        evidence_chain_tool_args: List[str],
        evidence_chain_result_summaries: List[str],
    ) -> Analysis:
        """Provide the final structured analysis of the system failure.

        Args:
            root_causes: List of identified root causes from the candidate list
            evidences: List of evidence supporting the root causes
            evidence_chain_actions: List of investigation actions taken
            evidence_chain_tool_names: List of tools used in investigation
            evidence_chain_tool_args: List of tool arguments used
            evidence_chain_result_summaries: List of result summaries from each action

        Returns:
            Analysis: Structured analysis with root causes and evidence
        """
        # Validate root causes are from candidates
        invalid_causes = [rc for rc in root_causes if rc not in candidate_root_causes]
        if invalid_causes:
            raise ValueError(f"Invalid root causes: {invalid_causes}. Must be from: {candidate_root_causes}")

        from src.syntax import Action, EvidenceChain

        # Build evidence chain from the provided data
        actions = []
        for i in range(len(evidence_chain_actions)):
            action = Action(
                tool_name=(
                    evidence_chain_tool_names[i]
                    if i < len(evidence_chain_tool_names)
                    else ""
                ),
                tool_args=(
                    {"args": evidence_chain_tool_args[i]}
                    if i < len(evidence_chain_tool_args)
                    else {}
                ),
                result_summary=(
                    evidence_chain_result_summaries[i]
                    if i < len(evidence_chain_result_summaries)
                    else ""
                ),
            )
            actions.append(action)

        evidence_chain = EvidenceChain(actions=actions)

        return Analysis(
            root_causes=root_causes, evidences=evidences, evidence_chain=evidence_chain
        )

    return provide_final_analysis
