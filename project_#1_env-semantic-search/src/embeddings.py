"""
Embedding model wrapper for Environmental Semantic Search Tool.

Handles text embedding using OpenAI's embedding models.
"""

from typing import List, Optional, Dict
from pathlib import Path

import openai
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings
from .utils import Timer, EmbeddingError


# =============================================================================
# OpenAI Embedding Model
# =============================================================================

class EmbeddingModel:
    """
    Wrapper for OpenAI embedding model with caching and error handling.
    """
    
    # Model information
    MODELS = {
        "text-embedding-3-small": {
            "max_tokens": 8191,
            "dimensions": 1536,
            "cost_per_1m": 0.02,
        },
        "text-embedding-3-large": {
            "max_tokens": 8191,
            "dimensions": 3072,
            "cost_per_1m": 0.13,
        },
        "text-embedding-ada-002": {
            "max_tokens": 8191,
            "dimensions": 1536,
            "cost_per_1m": 0.10,
        },
    }
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        cache_file: str = ".embedding_cache.json",
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: OpenAI model name
            dimensions: Output dimensions (for text-embedding-3-* models)
            api_key: OpenAI API key
            cache_file: Path to cache file
        """
        self.model_name = model_name or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.api_key = api_key or settings.openai_api_key
        
        # Validate
        if not self.api_key:
            raise EmbeddingError("OpenAI API key not configured")
        
        if self.model_name not in self.MODELS:
            logger.warning(f"Unknown model {self.model_name}, using defaults")
        
        # Initialize client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Stats
        self._total_tokens = 0
        self._total_requests = 0
        
        # Cache setup
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        logger.info(f"Loaded {len(self.cache)} cached embeddings")
        
        logger.info(f"EmbeddingModel initialized: {self.model_name} ({self.dimensions}D)")
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cache from disk."""
        import json
        from typing import Dict, List
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        import json
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_text_hash(self, text: str) -> str:
        """Get hash for text + model config."""
        import hashlib
        # Include model info in hash so changing model invalidates cache
        key = f"{self.model_name}_{self.dimensions}_{text}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=20),
    )
    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Call OpenAI API with retry logic.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # Build request params
        params = {
            "model": self.model_name,
            "input": texts,
        }
        
        # Add dimensions for text-embedding-3-* models
        if self.model_name.startswith("text-embedding-3"):
            params["dimensions"] = self.dimensions
        
        response = self.client.embeddings.create(**params)
        
        # Update stats
        self._total_tokens += response.usage.total_tokens
        self._total_requests += 1
        
        # Extract embeddings (sorted by index)
        embeddings = [None] * len(texts)
        for item in response.data:
            embeddings[item.index] = item.embedding
        
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text with caching.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")
            
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        with Timer("Single embedding"):
            embeddings = self._call_api([text])
            vector = embeddings[0]
            
            # Update cache
            self.cache[text_hash] = vector
            self._save_cache()
            return vector
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Embed multiple texts with batching and caching.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            show_progress: Show progress logs
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter empty texts & check cache
        valid_indices = []
        valid_texts = []
        cached_embeddings = {}
        to_embed_indices = []
        to_embed_texts = []
        
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            
            valid_indices.append(i)
            valid_texts.append(text)
            
            text_hash = self._get_text_hash(text)
            if text_hash in self.cache:
                cached_embeddings[i] = self.cache[text_hash]
            else:
                to_embed_indices.append(i)
                to_embed_texts.append(text)
        
        if not valid_indices:
            raise EmbeddingError("All texts are empty")
            
        # Embed missing texts in batches
        if to_embed_texts:
            logger.info(f"Embedding {len(to_embed_texts)}/{len(valid_texts)} texts (others cached)")
            
            total_batches = (len(to_embed_texts) + batch_size - 1) // batch_size
            
            with Timer(f"Embedding {len(to_embed_texts)} new texts"):
                for batch_num in range(total_batches):
                    start_idx = batch_num * batch_size
                    end_idx = min(start_idx + batch_size, len(to_embed_texts))
                    
                    batch_texts = to_embed_texts[start_idx:end_idx]
                    batch_orig_indices = to_embed_indices[start_idx:end_idx]
                    
                    if show_progress:
                        logger.info(f"Embedding batch {batch_num + 1}/{total_batches}")
                    
                    try:
                        batch_embeddings = self._call_api(batch_texts)
                        
                        for orig_idx, text, emb in zip(batch_orig_indices, batch_texts, batch_embeddings):
                            cached_embeddings[orig_idx] = emb
                            # Update cache
                            self.cache[self._get_text_hash(text)] = emb
                            
                    except Exception as e:
                        raise EmbeddingError(f"Batch embedding failed: {e}")
            
            # Save cache after all batches
            self._save_cache()
        
        # Reconstruct result list
        all_embeddings = [None] * len(texts)
        for idx in valid_indices:
            all_embeddings[idx] = cached_embeddings[idx]
            
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed a search query.
        
        Alias for embed_text, can be extended for query-specific preprocessing.
        
        Args:
            query: Search query
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed document texts.
        
        Alias for embed_texts, can be extended for document-specific preprocessing.
        
        Args:
            documents: List of document texts
            
        Returns:
            List of document embeddings
        """
        return self.embed_texts(documents)
    
    @property
    def total_tokens_used(self) -> int:
        """Total tokens used."""
        return self._total_tokens
    
    @property
    def total_requests(self) -> int:
        """Total API requests made."""
        return self._total_requests
    
    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD."""
        model_info = self.MODELS.get(self.model_name, {})
        cost_per_1m = model_info.get("cost_per_1m", 0.02)
        return (self._total_tokens / 1_000_000) * cost_per_1m
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "total_tokens": self._total_tokens,
            "total_requests": self._total_requests,
            "estimated_cost_usd": round(self.estimated_cost, 4),
            "cache_size": len(self.cache),
        }


# =============================================================================
# Gemini Embedding Model
# =============================================================================

class GeminiEmbeddingModel:
    """
    Wrapper for Google Gemini embedding model with caching.
    """
    
    MODELS = {
        "text-embedding-004": {
            "dimensions": 768,
            "max_tokens": 2048,
        },
        "embedding-001": {
            "dimensions": 768,
            "max_tokens": 2048,
        },
    }
    
    def __init__(
        self,
        model_name: str = "text-embedding-004",
        api_key: Optional[str] = None,
        cache_file: str = ".embedding_cache.json",
    ):
        """
        Initialize Gemini embedding model.
        
        Args:
            model_name: Gemini model name
            api_key: Google API key
            cache_file: Path to cache file
        """
        import google.generativeai as genai
        
        self.model_name = model_name
        self.api_key = api_key or settings.google_api_key
        self.dimensions = self.MODELS.get(model_name, {}).get("dimensions", 768)
        
        if not self.api_key:
            raise EmbeddingError("Google API key not configured")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Stats
        self._total_requests = 0
        
        # Cache setup
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        logger.info(f"Loaded {len(self.cache)} cached embeddings")
        
        logger.info(f"GeminiEmbeddingModel initialized: {self.model_name} ({self.dimensions}D)")
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load cache from disk."""
        import json
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        import json
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _get_text_hash(self, text: str) -> str:
        """Get hash for text + model config."""
        import hashlib
        key = f"gemini_{self.model_name}_{text}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _call_api(self, text: str) -> List[float]:
        """Call Gemini API to get embedding."""
        import google.generativeai as genai
        
        result = genai.embed_content(
            model=f"models/{self.model_name}",
            content=text,
            task_type="retrieval_document",
        )
        self._total_requests += 1
        return result['embedding']
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text with caching."""
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")
            
        text_hash = self._get_text_hash(text)
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        with Timer("Single embedding"):
            vector = self._call_api(text)
            self.cache[text_hash] = vector
            self._save_cache()
            return vector
    
    def embed_texts(self, texts: List[str], batch_size: int = 100, show_progress: bool = True) -> List[List[float]]:
        """Embed multiple texts with caching."""
        if not texts:
            return []
        
        all_embeddings = [None] * len(texts)
        to_embed = []
        
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            text_hash = self._get_text_hash(text)
            if text_hash in self.cache:
                all_embeddings[i] = self.cache[text_hash]
            else:
                to_embed.append((i, text))
        
        if to_embed:
            logger.info(f"Embedding {len(to_embed)}/{len(texts)} texts (others cached)")
            for idx, text in to_embed:
                if show_progress:
                    logger.debug(f"Embedding text {idx}")
                vector = self._call_api(text)
                all_embeddings[idx] = vector
                self.cache[self._get_text_hash(text)] = vector
            self._save_cache()
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a search query using retrieval_query task_type."""
        import google.generativeai as genai
        
        if not query.strip():
            raise EmbeddingError("Cannot embed empty query")
        
        # Use retrieval_query for queries (different from retrieval_document for docs)
        result = genai.embed_content(
            model=f"models/{self.model_name}",
            content=query,
            task_type="retrieval_query",  # Important: different from documents!
        )
        self._total_requests += 1
        return result['embedding']
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed document texts."""
        return self.embed_texts(documents)
    
    @property
    def total_requests(self) -> int:
        """Total API requests made."""
        return self._total_requests
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        return {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "total_requests": self._total_requests,
            "cache_size": len(self.cache),
        }


# =============================================================================
# Factory Function
# =============================================================================

def get_embedding_model(
    model_name: Optional[str] = None,
    dimensions: Optional[int] = None,
):
    """
    Get embedding model instance based on configured provider.
    
    Args:
        model_name: Model name (optional)
        dimensions: Output dimensions (optional, OpenAI only)
        
    Returns:
        EmbeddingModel or GeminiEmbeddingModel instance
    """
    provider = settings.embedding_provider
    
    if provider == "gemini":
        logger.info("Using Gemini embedding provider")
        return GeminiEmbeddingModel(
            model_name=model_name or "text-embedding-004",
        )
    else:
        logger.info("Using OpenAI embedding provider")
        return EmbeddingModel(
            model_name=model_name,
            dimensions=dimensions,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    import math
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def list_available_models() -> dict:
    """List available embedding models with info."""
    return EmbeddingModel.MODELS
