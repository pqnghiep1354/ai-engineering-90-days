"""
EIA Generator - Environmental Impact Assessment Report Generation System

A comprehensive AI-powered system for generating professional EIA reports
compliant with Vietnamese and international environmental regulations.
"""

__version__ = "1.0.0"
__author__ = "AI Engineer Portfolio"

from .config import Settings, ProjectInput, EIAConfig
from .orchestrator import EIAOrchestrator

__all__ = [
    "Settings",
    "ProjectInput", 
    "EIAConfig",
    "EIAOrchestrator",
]
