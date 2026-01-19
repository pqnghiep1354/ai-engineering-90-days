"""
Quick Research Workflow - Fast research for basic queries.
"""

from typing import Optional

from loguru import logger

from .base import BaseWorkflow, WorkflowResult
from ..agents import ResearcherAgent, WriterAgent
from ..agents.base import AgentState


class QuickResearchWorkflow(BaseWorkflow):
    """
    Quick research workflow for fast, basic research.
    
    Flow: Research → Write
    Time: 2-5 minutes
    """
    
    def __init__(self, max_iterations: int = 5):
        """Initialize quick research workflow."""
        super().__init__(
            name="quick_research",
            description="Fast research for basic queries (2-5 min)",
            max_iterations=max_iterations,
        )
        
        # Initialize agents
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
    
    async def execute(
        self,
        topic: str,
        max_sources: int = 5,
        report_format: str = "brief",
        **kwargs,
    ) -> WorkflowResult:
        """
        Execute quick research workflow.
        
        Args:
            topic: Research topic
            max_sources: Maximum sources to collect
            report_format: Output format (brief, executive)
            
        Returns:
            Workflow result
        """
        logger.info(f"Starting quick research: {topic}")
        self._start_timer()
        
        # Create initial state
        state = self.create_state(topic)
        
        try:
            # Step 1: Research
            logger.info("Step 1: Gathering information...")
            research_response = await self.researcher.process(
                state,
                max_sources=max_sources,
            )
            
            if not research_response.success:
                return self._create_result(
                    state,
                    success=False,
                    error=research_response.error,
                )
            
            # Update state with research results
            state = self.researcher.update_state(state, research_response)
            state["iteration"] += 1
            
            # Step 2: Write report
            logger.info("Step 2: Writing report...")
            writer_response = await self.writer.process(
                state,
                report_format=report_format,
                include_citations=True,
            )
            
            if not writer_response.success:
                return self._create_result(
                    state,
                    success=False,
                    error=writer_response.error,
                )
            
            # Update state with report
            state["final_report"] = writer_response.content
            state["iteration"] += 1
            
            logger.info(f"Quick research complete in {self._get_elapsed_time():.1f}s")
            
            return self._create_result(state, success=True)
            
        except Exception as e:
            logger.error(f"Quick research failed: {e}")
            return self._create_result(
                state,
                success=False,
                error=str(e),
            )
