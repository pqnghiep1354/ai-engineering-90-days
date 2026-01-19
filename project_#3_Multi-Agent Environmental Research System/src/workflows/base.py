"""
Base workflow class for research orchestration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from ..agents.base import AgentState, create_initial_state


@dataclass
class WorkflowResult:
    """Result from a workflow execution."""
    topic: str
    report: str
    sources: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "report": self.report,
            "sources": self.sources,
            "findings": self.findings,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
            "execution_time_seconds": self.execution_time_seconds,
        }
    
    def save_report(self, filepath: str):
        """Save report to file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.report)
        logger.info(f"Report saved to: {filepath}")


class BaseWorkflow(ABC):
    """
    Base class for research workflows.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        max_iterations: int = 10,
    ):
        """
        Initialize workflow.
        
        Args:
            name: Workflow name
            description: Workflow description
            max_iterations: Maximum iterations allowed
        """
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self._start_time: Optional[datetime] = None
    
    @abstractmethod
    async def execute(
        self,
        topic: str,
        **kwargs,
    ) -> WorkflowResult:
        """
        Execute the workflow.
        
        Args:
            topic: Research topic
            **kwargs: Additional arguments
            
        Returns:
            Workflow result
        """
        pass
    
    def create_state(self, topic: str) -> AgentState:
        """Create initial state for workflow."""
        return create_initial_state(topic)
    
    def _start_timer(self):
        """Start execution timer."""
        self._start_time = datetime.now()
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time:
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0
    
    def _create_result(
        self,
        state: AgentState,
        success: bool = True,
        error: Optional[str] = None,
    ) -> WorkflowResult:
        """Create workflow result from state."""
        return WorkflowResult(
            topic=state["topic"],
            report=state.get("final_report", ""),
            sources=state.get("sources", []),
            findings=state.get("findings", []),
            metadata={
                "workflow": self.name,
                "iterations": state.get("iteration", 0),
                "fact_checks": len(state.get("fact_check_results", [])),
            },
            success=success,
            error=error,
            execution_time_seconds=self._get_elapsed_time(),
        )
