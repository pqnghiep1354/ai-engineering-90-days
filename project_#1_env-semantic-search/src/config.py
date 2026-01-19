"""
Configuration management for Environmental Semantic Search Tool.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

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
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for embeddings"
    )
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini embeddings"
    )
    
    # =========================================================================
    # Embedding Provider
    # =========================================================================
    embedding_provider: Literal["openai", "gemini"] = Field(
        default="gemini",
        description="Embedding provider: openai or gemini"
    )
    
    # =========================================================================
    # Embedding Configuration
    # =========================================================================
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model to use"
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=256,
        le=3072,
        description="Embedding vector dimensions"
    )
    
    # =========================================================================
    # Vector Database
    # =========================================================================
    vector_db_type: Literal["chroma", "pinecone"] = Field(
        default="chroma",
        description="Vector database type"
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        description="ChromaDB persistence directory"
    )
    chroma_collection_name: str = Field(
        default="environmental_docs",
        description="ChromaDB collection name"
    )
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_environment: str = Field(default="us-east-1")
    pinecone_index_name: str = Field(default="env-search")
    
    # =========================================================================
    # Document Processing
    # =========================================================================
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Document chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between chunks"
    )
    min_chunk_size: int = Field(
        default=100,
        ge=10,
        description="Minimum chunk size to keep"
    )
    
    # =========================================================================
    # Search Configuration
    # =========================================================================
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default number of search results"
    )
    min_similarity_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )
    enable_hybrid_search: bool = Field(
        default=False,
        description="Enable hybrid vector + keyword search"
    )
    
    # =========================================================================
    # Application Settings
    # =========================================================================
    app_env: Literal["development", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=True)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO"
    )
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    streamlit_port: int = Field(default=8501)
    
    # =========================================================================
    # Validators
    # =========================================================================
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    # =========================================================================
    # Properties
    # =========================================================================
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"
    
    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key and len(self.openai_api_key) > 10)
    
    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
