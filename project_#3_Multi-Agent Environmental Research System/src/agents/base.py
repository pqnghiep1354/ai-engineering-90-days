"""
Base agent class for the research system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from loguru import logger

from ..config import settings


# =============================================================================
# Enums and Types
# =============================================================================

class AgentRole(str, Enum):
    """Agent roles in the system."""
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    FACT_CHECKER = "fact_checker"


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Source:
    """Represents a research source."""
    title: str
    url: str
    content: str
    snippet: str = ""
    domain: str = ""
    relevance_score: float = 0.0
    retrieved_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "domain": self.domain,
            "relevance_score": self.relevance_score,
        }


@dataclass
class Finding:
    """Represents a research finding."""
    content: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.8
    category: str = "general"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "category": self.category,
        }


@dataclass
class AgentResponse:
    """Response from an agent."""
    agent: AgentRole
    content: str
    findings: List[Finding] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent.value,
            "content": self.content,
            "findings": [f.to_dict() for f in self.findings],
            "sources": [s.to_dict() for s in self.sources],
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
        }


class AgentState(TypedDict):
    """State passed between agents in the workflow."""
    topic: str
    messages: List[BaseMessage]
    sources: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    analysis: str
    draft: str
    final_report: str
    fact_check_results: List[Dict[str, Any]]
    current_agent: str
    iteration: int
    status: str
    errors: List[str]


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """
    Base class for all agents in the research system.
    """
    
    def __init__(
        self,
        role: AgentRole,
        system_prompt: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Any]] = None,
    ):
        """
        Initialize base agent.
        
        Args:
            role: Agent role
            system_prompt: System prompt for the agent
            model_name: OpenAI model name
            temperature: Model temperature
            tools: List of tools available to agent
        """
        self.role = role
        self.system_prompt = system_prompt
        self.model_name = model_name or settings.openai_model
        self.temperature = temperature if temperature is not None else settings.default_temperature
        self.tools = tools or []
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=settings.openai_api_key,
        )
        
        # Bind tools if provided
        if self.tools:
            self.llm = self.llm.bind_tools(self.tools)
        
        logger.info(f"Initialized {role.value} agent with model {self.model_name}")
    
    def _create_messages(
        self,
        user_message: str,
        context: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Create message list for LLM call.
        
        Args:
            user_message: User's message
            context: Additional context
            
        Returns:
            List of messages
        """
        messages = [SystemMessage(content=self.system_prompt)]
        
        if context:
            messages.append(HumanMessage(content=f"Context:\n{context}"))
        
        messages.append(HumanMessage(content=user_message))
        
        return messages
    
    async def _call_llm(
        self,
        messages: List[BaseMessage],
    ) -> AIMessage:
        """
        Call LLM with messages.
        
        Args:
            messages: Messages to send
            
        Returns:
            AI response message
        """
        try:
            response = await self.llm.ainvoke(messages)
            return response
        except Exception as e:
            logger.error(f"LLM call failed for {self.role.value}: {e}")
            raise
    
    @abstractmethod
    async def process(
        self,
        state: AgentState,
        **kwargs,
    ) -> AgentResponse:
        """
        Process the current state and return response.
        
        Args:
            state: Current agent state
            **kwargs: Additional arguments
            
        Returns:
            Agent response
        """
        pass
    
    def update_state(
        self,
        state: AgentState,
        response: AgentResponse,
    ) -> AgentState:
        """
        Update state with agent response.
        
        Args:
            state: Current state
            response: Agent response
            
        Returns:
            Updated state
        """
        # Add response to messages
        state["messages"].append(
            AIMessage(content=response.content, name=self.role.value)
        )
        
        # Update sources
        for source in response.sources:
            state["sources"].append(source.to_dict())
        
        # Update findings
        for finding in response.findings:
            state["findings"].append(finding.to_dict())
        
        # Update current agent
        state["current_agent"] = self.role.value
        
        return state
    
    def get_context_from_state(self, state: AgentState) -> str:
        """
        Extract relevant context from state.
        
        Args:
            state: Current state
            
        Returns:
            Context string
        """
        context_parts = []
        
        # Add topic
        context_parts.append(f"Research Topic: {state['topic']}")
        
        # Add existing findings summary
        if state["findings"]:
            findings_text = "\n".join(
                f"- {f['content'][:200]}..." 
                for f in state["findings"][:5]
            )
            context_parts.append(f"\nExisting Findings:\n{findings_text}")
        
        # Add source count
        if state["sources"]:
            context_parts.append(f"\nSources collected: {len(state['sources'])}")
        
        return "\n".join(context_parts)


# =============================================================================
# Utility Functions
# =============================================================================

def create_initial_state(topic: str) -> AgentState:
    """
    Create initial state for a research task.
    
    Args:
        topic: Research topic
        
    Returns:
        Initial agent state
    """
    return AgentState(
        topic=topic,
        messages=[HumanMessage(content=topic)],
        sources=[],
        findings=[],
        analysis="",
        draft="",
        final_report="",
        fact_check_results=[],
        current_agent="",
        iteration=0,
        status=TaskStatus.PENDING.value,
        errors=[],
    )
