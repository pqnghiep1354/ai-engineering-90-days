"""
Workflow implementations for the research system.
"""

from .base import BaseWorkflow, WorkflowResult
from .quick_research import QuickResearchWorkflow
from .deep_dive import DeepDiveWorkflow

__all__ = [
    "BaseWorkflow",
    "WorkflowResult",
    "QuickResearchWorkflow",
    "DeepDiveWorkflow",
]
