"""
Deep Dive Workflow - Comprehensive research with fact-checking.
"""

from typing import Optional

from loguru import logger

from .base import BaseWorkflow, WorkflowResult
from ..agents import (
    ResearcherAgent,
    AnalystAgent,
    WriterAgent,
    FactCheckerAgent,
)
from ..agents.base import AgentState
from ..config import settings


class DeepDiveWorkflow(BaseWorkflow):
    """
    Deep dive workflow for comprehensive research.
    
    Flow: Research → Analyze → Research (refinement) → Fact-Check → Write
    Time: 10-20 minutes
    """
    
    def __init__(self, max_iterations: int = 10):
        """Initialize deep dive workflow."""
        super().__init__(
            name="deep_dive",
            description="Comprehensive research with analysis and fact-checking (10-20 min)",
            max_iterations=max_iterations,
        )
        
        # Initialize agents
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.fact_checker = FactCheckerAgent()
    
    async def execute(
        self,
        topic: str,
        max_sources: int = 15,
        enable_fact_checking: Optional[bool] = None,
        report_format: str = "full",
        **kwargs,
    ) -> WorkflowResult:
        """
        Execute deep dive workflow.
        
        Args:
            topic: Research topic
            max_sources: Maximum sources to collect
            enable_fact_checking: Override fact-checking setting
            report_format: Output format (full, executive)
            
        Returns:
            Workflow result
        """
        logger.info(f"Starting deep dive research: {topic}")
        self._start_timer()
        
        # Determine if fact-checking is enabled
        do_fact_check = enable_fact_checking if enable_fact_checking is not None else settings.enable_fact_checking
        
        # Create initial state
        state = self.create_state(topic)
        
        try:
            # Phase 1: Initial Research
            logger.info("Phase 1: Initial research...")
            state = await self._phase_research(state, max_sources=max_sources // 2)
            
            if state.get("errors"):
                return self._create_result(
                    state,
                    success=False,
                    error=state["errors"][-1],
                )
            
            # Phase 2: Analysis
            logger.info("Phase 2: Analysis...")
            state = await self._phase_analysis(state)
            
            # Phase 3: Follow-up Research
            logger.info("Phase 3: Follow-up research...")
            state = await self._phase_followup_research(state, max_sources=max_sources // 2)
            
            # Phase 4: Fact-Checking (optional)
            if do_fact_check:
                logger.info("Phase 4: Fact-checking...")
                state = await self._phase_fact_check(state)
            
            # Phase 5: Report Writing
            logger.info("Phase 5: Writing report...")
            state = await self._phase_writing(state, report_format=report_format)
            
            elapsed = self._get_elapsed_time()
            logger.info(f"Deep dive complete in {elapsed:.1f}s")
            
            return self._create_result(state, success=True)
            
        except Exception as e:
            logger.error(f"Deep dive failed: {e}")
            return self._create_result(
                state,
                success=False,
                error=str(e),
            )
    
    async def _phase_research(
        self,
        state: AgentState,
        max_sources: int = 10,
    ) -> AgentState:
        """Execute initial research phase."""
        response = await self.researcher.process(
            state,
            max_sources=max_sources,
        )
        
        if response.success:
            state = self.researcher.update_state(state, response)
            state["iteration"] += 1
        else:
            state["errors"].append(response.error or "Research failed")
        
        return state
    
    async def _phase_analysis(
        self,
        state: AgentState,
        analysis_type: str = "comprehensive",
    ) -> AgentState:
        """Execute analysis phase."""
        response = await self.analyst.process(
            state,
            analysis_type=analysis_type,
        )
        
        if response.success:
            state = self.analyst.update_state(state, response)
            state["analysis"] = response.content
            state["iteration"] += 1
        else:
            state["errors"].append(response.error or "Analysis failed")
        
        return state
    
    async def _phase_followup_research(
        self,
        state: AgentState,
        max_sources: int = 10,
    ) -> AgentState:
        """Execute follow-up research based on analysis."""
        # Generate follow-up queries based on analysis
        analysis = state.get("analysis", "")
        
        if not analysis:
            return state
        
        # The researcher will generate queries based on gaps identified
        response = await self.researcher.process(
            state,
            max_sources=max_sources,
        )
        
        if response.success:
            state = self.researcher.update_state(state, response)
            state["iteration"] += 1
        
        return state
    
    async def _phase_fact_check(
        self,
        state: AgentState,
    ) -> AgentState:
        """Execute fact-checking phase."""
        response = await self.fact_checker.process(
            state,
            max_claims=10,
        )
        
        if response.success:
            state = self.fact_checker.update_state(state, response)
            state["fact_check_results"] = response.metadata.get("details", [])
            state["iteration"] += 1
        
        return state
    
    async def _phase_writing(
        self,
        state: AgentState,
        report_format: str = "full",
    ) -> AgentState:
        """Execute report writing phase."""
        response = await self.writer.process(
            state,
            report_format=report_format,
            include_citations=True,
        )
        
        if response.success:
            state["final_report"] = response.content
            state["draft"] = response.content
            state["iteration"] += 1
        else:
            state["errors"].append(response.error or "Writing failed")
        
        return state
