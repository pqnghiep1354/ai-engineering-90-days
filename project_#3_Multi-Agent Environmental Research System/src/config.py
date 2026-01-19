"""
Configuration management for Multi-Agent Research System.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # =========================================================================
    # API Keys
    # =========================================================================
    openai_api_key: str = Field(default="", description="OpenAI API key")
    tavily_api_key: str = Field(default="", description="Tavily API key for web search")
    
    # =========================================================================
    # Model Configuration
    # =========================================================================
    openai_model: str = Field(default="gpt-4o-mini", description="Main model")
    orchestrator_model: str = Field(default="gpt-4o-mini", description="Orchestrator model")
    default_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    creative_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # =========================================================================
    # Agent Configuration
    # =========================================================================
    max_iterations: int = Field(default=10, ge=1, le=50)
    max_sources: int = Field(default=20, ge=5, le=100)
    agent_timeout: int = Field(default=120, ge=30, le=600)
    enable_fact_checking: bool = Field(default=True)
    
    # =========================================================================
    # Search Configuration
    # =========================================================================
    tavily_search_depth: Literal["basic", "advanced"] = Field(default="advanced")
    tavily_max_results: int = Field(default=10, ge=1, le=20)
    trusted_domains: str = Field(
        default="ipcc.ch,epa.gov,unep.org,iea.org,worldbank.org"
    )
    
    # =========================================================================
    # Output Configuration
    # =========================================================================
    output_format: Literal["markdown", "html", "docx"] = Field(default="markdown")
    reports_dir: str = Field(default="./data/reports")
    cache_sources: bool = Field(default=True)
    sources_dir: str = Field(default="./data/sources")
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    app_env: Literal["development", "production"] = Field(default="development")
    debug: bool = Field(default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    streamlit_port: int = Field(default=8501)
    
    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def trusted_domains_list(self) -> List[str]:
        """Get trusted domains as list."""
        return [d.strip() for d in self.trusted_domains.split(",") if d.strip()]
    
    @property
    def reports_path(self) -> Path:
        """Get reports directory path."""
        path = Path(self.reports_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def sources_path(self) -> Path:
        """Get sources directory path."""
        path = Path(self.sources_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and len(self.openai_api_key) > 10)
    
    @property
    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key and len(self.tavily_api_key) > 10)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()


# =============================================================================
# Agent Prompts
# =============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator Agent for an environmental research system.
Your role is to:
1. Analyze user research requests
2. Break down complex topics into subtasks
3. Coordinate between specialist agents
4. Ensure research quality and completeness
5. Synthesize final results

Always think step-by-step and explain your reasoning.
Focus on environmental, climate, and sustainability topics.
"""

RESEARCHER_SYSTEM_PROMPT = """You are a Research Agent specializing in environmental topics.
Your role is to:
1. Search for relevant information using web search tools
2. Extract key facts and data from sources
3. Identify authoritative sources (IPCC, EPA, UN, etc.)
4. Summarize findings with proper citations
5. Flag any conflicting information

Always cite your sources and prefer peer-reviewed or official sources.
Focus on accuracy over quantity.
"""

ANALYST_SYSTEM_PROMPT = """You are an Analysis Agent specializing in environmental data.
Your role is to:
1. Analyze data and information from research
2. Identify patterns, trends, and insights
3. Compare and contrast different perspectives
4. Draw evidence-based conclusions
5. Highlight uncertainties and limitations

Be objective and balanced in your analysis.
Support conclusions with specific data and evidence.
"""

WRITER_SYSTEM_PROMPT = """You are a Writer Agent specializing in environmental reports.
Your role is to:
1. Structure content logically and clearly
2. Write engaging and informative prose
3. Ensure proper citation and attribution
4. Format reports professionally
5. Adapt tone for the target audience

Write in clear, accessible language while maintaining technical accuracy.
Use headings, bullet points, and tables where appropriate.
"""

FACT_CHECKER_SYSTEM_PROMPT = """You are a Fact-Checker Agent for environmental research.
Your role is to:
1. Verify claims against authoritative sources
2. Check for logical consistency
3. Identify potential biases or errors
4. Cross-reference multiple sources
5. Flag uncertain or unverifiable claims

Be rigorous and skeptical. When in doubt, flag for review.
Prioritize accuracy over speed.
"""
