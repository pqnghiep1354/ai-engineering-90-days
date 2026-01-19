"""
Embedding models for Climate Q&A RAG System.

Supports OpenAI, Gemini, and local embedding models.
"""

from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger

from .config import settings
from .utils import Timer, EmbeddingError


# =============================================================================
# Embedding Factory
# =============================================================================

def get_embedding_model(
    model_name: Optional[str] = None,
    provider: str = "openai",
) -> Embeddings:
    """
    Get embedding model instance.
    
    Args:
        model_name: Model name (default depends on provider)
        provider: Provider name ("openai", "gemini", or "local")
        
    Returns:
        Embeddings instance
    """
    if provider == "openai":
        # For OpenAI, use settings.embedding_model as default if no model_name provided
        model_name = model_name or settings.embedding_model
        return get_openai_embeddings(model_name)
    elif provider == "gemini":
        # For Gemini, use function default (text-embedding-004) if no model_name provided
        return get_gemini_embeddings(model_name) if model_name else get_gemini_embeddings()
    elif provider == "local":
        # For local, use function default if no model_name provided
        return get_local_embeddings(model_name) if model_name else get_local_embeddings()
    else:
        raise EmbeddingError(f"Unknown embedding provider: {provider}")


def get_openai_embeddings(model_name: str = "text-embedding-3-small") -> Embeddings:
    """
    Get OpenAI embedding model.
    
    Args:
        model_name: OpenAI embedding model name
        
    Returns:
        OpenAIEmbeddings instance
    """
    if not settings.has_openai:
        raise EmbeddingError("OpenAI API key not configured")
    
    logger.info(f"Initializing OpenAI embeddings with model: {model_name}")
    
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=settings.openai_api_key,
    )


def get_gemini_embeddings(model_name: str = "text-embedding-004") -> Embeddings:
    """
    Get Google Gemini embedding model.
    
    Args:
        model_name: Gemini embedding model name
        
    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    if not settings.google_api_key:
        raise EmbeddingError("Google API key not configured")
    
    logger.info(f"Initializing Gemini embeddings with model: {model_name}")
    
    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=settings.google_api_key,
    )


def get_local_embeddings(model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    """
    Get local embedding model using HuggingFace.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        HuggingFaceEmbeddings instance
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise EmbeddingError(
            "HuggingFace embeddings not available. "
            "Install with: pip install sentence-transformers"
        )
    
    logger.info(f"Initializing local embeddings with model: {model_name}")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# =============================================================================
# Embedding Utilities
# =============================================================================

class EmbeddingWrapper:
    """
    Wrapper for embedding model with caching and error handling.
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        cache_enabled: bool = True,
    ):
        """
        Initialize embedding wrapper.
        
        Args:
            embeddings: Base embedding model
            cache_enabled: Whether to enable caching
        """
        self.embeddings = embeddings
        self.cache_enabled = cache_enabled
        self._cache = {} if cache_enabled else None
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        if self.cache_enabled and text in self._cache:
            return self._cache[text]
        
        try:
            with Timer("Embedding query"):
                embedding = self.embeddings.embed_query(text)
                
            if self.cache_enabled:
                self._cache[text] = embedding
                
            return embedding
            
        except Exception as e:
            raise EmbeddingError(f"Error embedding query: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Check cache for all texts
        if self.cache_enabled:
            cached = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._cache:
                    cached.append((i, self._cache[text]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if not uncached_texts:
                # All texts were cached
                return [emb for _, emb in sorted(cached)]
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached = []
        
        try:
            with Timer(f"Embedding {len(uncached_texts)} documents"):
                new_embeddings = self.embeddings.embed_documents(uncached_texts)
            
            # Update cache
            if self.cache_enabled:
                for text, emb in zip(uncached_texts, new_embeddings):
                    self._cache[text] = emb
            
            # Combine cached and new embeddings
            if cached:
                all_embeddings = [None] * len(texts)
                for i, emb in cached:
                    all_embeddings[i] = emb
                for i, emb in zip(uncached_indices, new_embeddings):
                    all_embeddings[i] = emb
                return all_embeddings
            else:
                return new_embeddings
                
        except Exception as e:
            raise EmbeddingError(f"Error embedding documents: {e}")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self._cache is not None:
            self._cache.clear()
            logger.info("Embedding cache cleared")
    
    @property
    def cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache) if self._cache else 0


# =============================================================================
# Model Information
# =============================================================================

EMBEDDING_MODELS = {
    "openai": {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1k": 0.00002,
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8191,
            "cost_per_1k": 0.00013,
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1k": 0.0001,
        },
    },
    "local": {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_tokens": 512,
            "cost_per_1k": 0,
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "max_tokens": 512,
            "cost_per_1k": 0,
        },
    },
}


def get_model_info(model_name: str, provider: str = "openai") -> dict:
    """
    Get information about an embedding model.
    
    Args:
        model_name: Model name
        provider: Provider name
        
    Returns:
        Model information dictionary
    """
    provider_models = EMBEDDING_MODELS.get(provider, {})
    return provider_models.get(model_name, {})


def list_available_models() -> dict:
    """
    List all available embedding models.
    
    Returns:
        Dictionary of providers and their models
    """
    return EMBEDDING_MODELS
