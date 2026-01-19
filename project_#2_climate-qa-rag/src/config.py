"""
Configuration management for Climate Q&A RAG System.

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

import os
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # API Keys
    # ==========================================================================
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google AI API key")

    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model to use",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses",
    )
    llm_max_tokens: int = Field(
        default=2048,
        ge=100,
        le=8192,
        description="Maximum tokens for LLM response",
    )

    # ==========================================================================
    # Vector Database
    # ==========================================================================
    vector_db_type: Literal["chroma", "pinecone"] = Field(
        default="chroma",
        description="Vector database type",
    )
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="ChromaDB persistence directory",
    )
    chroma_collection_name: str = Field(
        default="climate_documents",
        description="ChromaDB collection name",
    )
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_environment: str = Field(default="us-east-1")
    pinecone_index_name: str = Field(default="climate-qa")

    # ==========================================================================
    # RAG Configuration
    # ==========================================================================
    retriever_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve",
    )
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Document chunk size in characters",
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Chunk overlap in characters",
    )
    use_reranker: bool = Field(
        default=True,
        description="Whether to use reranking",
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Reranker model name",
    )
    reranker_top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of documents after reranking",
    )

    # ==========================================================================
    # LangSmith Monitoring
    # ==========================================================================
    langchain_api_key: Optional[str] = Field(default=None)
    langchain_tracing_v2: bool = Field(default=False)
    langchain_project: str = Field(default="climate-qa-rag")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com")

    # ==========================================================================
    # Application Settings
    # ==========================================================================
    app_env: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    streamlit_port: int = Field(default=8501)

    # ==========================================================================
    # Feature Flags
    # ==========================================================================
    enable_memory: bool = Field(default=True)
    enable_query_expansion: bool = Field(default=True)
    enable_hybrid_search: bool = Field(default=True)
    enable_streaming: bool = Field(default=True)

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @field_validator("reranker_top_k")
    @classmethod
    def validate_reranker_top_k(cls, v: int, info) -> int:
        """Ensure reranker_top_k is less than or equal to retriever_top_k."""
        retriever_top_k = info.data.get("retriever_top_k", 5)
        if v > retriever_top_k:
            raise ValueError("reranker_top_k must be <= retriever_top_k")
        return v

    # ==========================================================================
    # Properties
    # ==========================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI API key is configured."""
        return self.openai_api_key is not None and len(self.openai_api_key) > 0

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic API key is configured."""
        return self.anthropic_api_key is not None and len(self.anthropic_api_key) > 0

    @property
    def has_langsmith(self) -> bool:
        """Check if LangSmith is configured."""
        return (
            self.langchain_api_key is not None
            and len(self.langchain_api_key) > 0
            and self.langchain_tracing_v2
        )

    def setup_langsmith(self) -> None:
        """Configure LangSmith environment variables."""
        if self.has_langsmith:
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
            os.environ["LANGCHAIN_ENDPOINT"] = self.langchain_endpoint


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    return Settings()


# Create a global settings instance for easy access
settings = get_settings()
