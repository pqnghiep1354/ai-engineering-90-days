"""
Base agent class for EIA Generator.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from loguru import logger

from ..config import ProjectInput, EIAConfig, get_settings
from ..llm import get_llm


# =============================================================================
# Agent State
# =============================================================================

class AgentState(TypedDict):
    """State shared between agents in the workflow."""
    
    # Input
    project: Dict[str, Any]
    config: Dict[str, Any]
    
    # Messages
    messages: List[BaseMessage]
    
    # Generated Content
    sections: Dict[str, str]
    tables: Dict[str, List[Dict]]
    figures: Dict[str, List[Dict]]
    
    # Research Results
    regulations: List[Dict[str, str]]
    baseline_data: Dict[str, Any]
    impact_matrix: Dict[str, Dict[str, Any]]
    
    # Validation
    validation_results: Dict[str, Any]
    compliance_score: float
    
    # Control
    current_step: str
    errors: List[str]


def create_initial_state(
    project: ProjectInput,
    config: EIAConfig,
) -> AgentState:
    """Create initial state for workflow."""
    return AgentState(
        project=project.model_dump(),
        config=config.model_dump(),
        messages=[],
        sections={},
        tables={},
        figures={},
        regulations=[],
        baseline_data={},
        impact_matrix={},
        validation_results={},
        compliance_score=0.0,
        current_step="init",
        errors=[],
    )


# =============================================================================
# Base Agent
# =============================================================================

class BaseAgent(ABC):
    """Base class for all EIA agents."""
    
    def __init__(
        self,
        name: str,
        description: str,
        model: Optional[str] = None,
        temperature: float = 0.4,
    ):
        self.name = name
        self.description = description
        
        settings = get_settings()
        self.model = model or settings.default_model
        self.temperature = temperature
        
        # Use LLM factory to get appropriate LLM instance
        self.llm = get_llm(
            model_name=self.model,
            temperature=temperature,
        )
        
        self.system_prompt = self._build_system_prompt()
        
        logger.info(f"Initialized agent: {name} with model: {self.model}")
    
    @abstractmethod
    def _build_system_prompt(self) -> str:
        """Build system prompt for the agent."""
        pass
    
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent's task."""
        pass
    
    async def _generate(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response from LLM."""
        messages = [
            SystemMessage(content=system_prompt or self.system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Agent {self.name} generation error: {e}")
            raise
    
    def _format_project_context(self, project: Dict[str, Any]) -> str:
        """Format project information for prompts."""
        return f"""
DỰ ÁN: {project.get('name', 'N/A')}
LOẠI: {project.get('type', 'N/A')}
VỊ TRÍ: {project.get('location', 'N/A')}
QUY MÔ: {project.get('area_hectares', 0)} ha
CÔNG SUẤT: {project.get('capacity', 'N/A')}
VỐN ĐẦU TƯ: {project.get('investment_usd', 0):,.0f} USD
THỜI GIAN XÂY DỰNG: {project.get('construction_months', 0)} tháng
THỜI GIAN VẬN HÀNH: {project.get('operation_years', 0)} năm
""".strip()


# =============================================================================
# Agent Registry
# =============================================================================

class AgentRegistry:
    """Registry for managing agents."""
    
    _agents: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        """Register an agent."""
        cls._agents[agent.name] = agent
        logger.debug(f"Registered agent: {agent.name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        """Get agent by name."""
        return cls._agents.get(name)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agents."""
        return list(cls._agents.keys())
