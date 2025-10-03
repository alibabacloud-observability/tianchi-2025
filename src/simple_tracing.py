#!/usr/bin/env python3
"""
Simple ARMS Tracing following the official documentation pattern
https://help.aliyun.com/zh/arms/application-monitoring/use-cases/end-to-end-full-link-tracing-based-on-mcp-protocol
"""

from contextlib import asynccontextmanager
from opentelemetry import trace
from src.enhanced_agent import enhanced_compute_plan_execute

# Get tracer exactly like in the documentation
tracer = trace.get_tracer(__name__)

class RCAAgent:
    """RCA Agent that encapsulates the computation logic"""
    
    def __init__(self, problem_id: str, time_range: str, candidates: list, **kwargs):
        self.problem_id = problem_id
        self.time_range = time_range
        self.candidates = candidates
        self.kwargs = kwargs
    
    async def analyze(self):
        """Run the RCA analysis - this is like ainvoke() in the documentation example"""
        return await enhanced_compute_plan_execute(
            time_range=self.time_range,
            candidate_root_causes=self.candidates,
            problem_id=self.problem_id,
            **self.kwargs
        )

@asynccontextmanager
async def rca_problem_agent(problem_id: str, time_range: str, candidates: list, **kwargs):
    """
    Create a traced RCA agent following ARMS documentation pattern.
    Exactly like the sls_agent example from the documentation.
    
    Args:
        problem_id: The problem ID
        time_range: Time range for analysis  
        candidates: List of candidate root causes
        **kwargs: Additional parameters for enhanced_compute_plan_execute
    """
    with tracer.start_as_current_span(f"rca-problem-{problem_id}") as span:
        # Set attributes for the problem context
        span.set_attribute("problem.id", problem_id)
        span.set_attribute("problem.time_range", time_range)
        span.set_attribute("problem.candidates_count", len(candidates))
        span.set_attribute("problem.candidates", str(candidates)[:500])  # Truncate if too long
        span.set_attribute("rca.method", "enhanced_agent")
        
        # Create the RCA agent object (like create_react_agent in the docs)
        agent = RCAAgent(problem_id, time_range, candidates, **kwargs)
        
        # Yield the agent object - just like in the documentation
        yield agent
