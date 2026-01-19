"""
Research Orchestrator - Main coordination for multi-agent research.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from .config import settings
from .workflows import (
    BaseWorkflow,
    WorkflowResult,
    QuickResearchWorkflow,
    DeepDiveWorkflow,
)
from .tools.citation import CitationManager


class ResearchOrchestrator:
    """
    Main orchestrator for the multi-agent research system.
    Coordinates workflows and manages research sessions.
    """
    
    # Available workflows
    WORKFLOWS: Dict[str, Type[BaseWorkflow]] = {
        "quick": QuickResearchWorkflow,
        "deep": DeepDiveWorkflow,
        "deep_dive": DeepDiveWorkflow,
    }
    
    def __init__(self):
        """Initialize research orchestrator."""
        self.citation_manager = CitationManager()
        self.history: List[WorkflowResult] = []
        
        logger.info("Research Orchestrator initialized")
    
    async def research(
        self,
        topic: str,
        workflow: Optional[str] = "quick",
        **kwargs,
    ) -> WorkflowResult:
        """
        Execute a research task.
        
        Args:
            topic: Research topic
            workflow: Workflow type (quick, deep)
            **kwargs: Additional workflow arguments
            
        Returns:
            Workflow result
        """
        logger.info(f"Starting research on: {topic}")
        logger.info(f"Workflow: {workflow}")
        
        # Get workflow class
        workflow_cls = self.WORKFLOWS.get(workflow, QuickResearchWorkflow)
        workflow_instance = workflow_cls()
        
        # Execute workflow
        result = await workflow_instance.execute(topic, **kwargs)
        
        # Store in history
        self.history.append(result)
        
        # Add sources to citation manager
        for source in result.sources:
            self.citation_manager.add_from_source(source)
        
        return result
    
    async def quick_research(
        self,
        topic: str,
        **kwargs,
    ) -> WorkflowResult:
        """
        Execute quick research workflow.
        
        Args:
            topic: Research topic
            **kwargs: Additional arguments
            
        Returns:
            Workflow result
        """
        return await self.research(topic, workflow="quick", **kwargs)
    
    async def deep_research(
        self,
        topic: str,
        **kwargs,
    ) -> WorkflowResult:
        """
        Execute deep dive research workflow.
        
        Args:
            topic: Research topic
            **kwargs: Additional arguments
            
        Returns:
            Workflow result
        """
        return await self.research(topic, workflow="deep", **kwargs)
    
    def save_report(
        self,
        result: WorkflowResult,
        filename: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> str:
        """
        Save research report to file.
        
        Args:
            result: Workflow result
            filename: Output filename
            directory: Output directory
            
        Returns:
            Path to saved file
        """
        directory = directory or str(settings.reports_path)
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            # Generate filename from topic
            safe_topic = "".join(
                c if c.isalnum() or c in "- " else "_"
                for c in result.topic[:50]
            ).strip().replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_topic}_{timestamp}.md"
        
        filepath = dir_path / filename
        result.save_report(str(filepath))
        
        return str(filepath)
    
    def get_citations(self, style: str = "markdown") -> str:
        """Get formatted citations."""
        return self.citation_manager.format_reference_list(style)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get research history."""
        return [
            {
                "topic": r.topic,
                "success": r.success,
                "sources": len(r.sources),
                "time": r.execution_time_seconds,
            }
            for r in self.history
        ]
    
    def clear_history(self):
        """Clear research history."""
        self.history.clear()
        self.citation_manager.clear()
    
    @staticmethod
    def list_workflows() -> List[Dict[str, str]]:
        """List available workflows."""
        return [
            {"name": "quick", "description": "Fast research (2-5 min)"},
            {"name": "deep", "description": "Comprehensive research with fact-checking (10-20 min)"},
        ]


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_research(
    topic: str,
    workflow: str = "quick",
    save_report: bool = True,
    **kwargs,
) -> WorkflowResult:
    """
    Convenience function to run research.
    
    Args:
        topic: Research topic
        workflow: Workflow type
        save_report: Whether to save report to file
        **kwargs: Additional arguments
        
    Returns:
        Workflow result
    """
    orchestrator = ResearchOrchestrator()
    result = await orchestrator.research(topic, workflow=workflow, **kwargs)
    
    if save_report and result.success:
        filepath = orchestrator.save_report(result)
        logger.info(f"Report saved: {filepath}")
    
    return result


def run_research_sync(
    topic: str,
    workflow: str = "quick",
    **kwargs,
) -> WorkflowResult:
    """
    Synchronous wrapper for run_research.
    
    Args:
        topic: Research topic
        workflow: Workflow type
        **kwargs: Additional arguments
        
    Returns:
        Workflow result
    """
    return asyncio.run(run_research(topic, workflow=workflow, **kwargs))
