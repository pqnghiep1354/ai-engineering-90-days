"""
Search engine for Environmental Semantic Search Tool.

Provides high-level search interface with various search strategies.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from .config import settings
from .vector_store import VectorStore, load_index
from .utils import detect_language, truncate_text


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """Single search result."""
    
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def source(self) -> str:
        return self.metadata.get("source", "Unknown")
    
    @property
    def chunk_index(self) -> int:
        return self.metadata.get("chunk_index", 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_time_ms": self.search_time_ms,
            "language": self.language,
        }


# =============================================================================
# Search Engine
# =============================================================================

class SemanticSearchEngine:
    """
    Semantic search engine for environmental documents.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        default_top_k: int = None,
        min_score: float = None,
    ):
        """
        Initialize search engine.
        
        Args:
            vector_store: Vector store instance
            default_top_k: Default number of results
            min_score: Minimum similarity score
        """
        self.vector_store = vector_store or load_index()
        self.default_top_k = default_top_k or settings.default_top_k
        self.min_score = min_score or settings.min_similarity_score
        
        logger.info("SemanticSearchEngine initialized")
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """
        Perform semantic search.
        
        Args:
            query: Search query
            top_k: Number of results
            threshold: Minimum similarity score
            filter: Metadata filter
            
        Returns:
            SearchResponse with results
        """
        if not query.strip():
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time_ms=0,
            )
        
        top_k = top_k or self.default_top_k
        threshold = threshold if threshold is not None else self.min_score
        
        # Detect language
        language = detect_language(query)
        
        # Time the search
        start_time = time.time()
        
        # Perform search
        raw_results = self.vector_store.search(
            query=query,
            top_k=top_k,
            threshold=threshold,
            filter=filter,
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        # Convert to SearchResult objects
        results = [
            SearchResult(
                content=r["content"],
                score=r["score"],
                metadata=r["metadata"],
            )
            for r in raw_results
        ]
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time_ms=round(search_time_ms, 2),
            language=language,
        )
    
    def search_by_source(
        self,
        query: str,
        source: str,
        top_k: int = 10,
    ) -> SearchResponse:
        """
        Search within a specific source.
        
        Args:
            query: Search query
            source: Source filename to filter
            top_k: Number of results
            
        Returns:
            SearchResponse with results from specific source
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter={"source": source},
        )
    
    def find_similar(
        self,
        text: str,
        top_k: int = 5,
        exclude_self: bool = True,
    ) -> SearchResponse:
        """
        Find documents similar to given text.
        
        Args:
            text: Reference text
            top_k: Number of results
            exclude_self: Exclude exact matches
            
        Returns:
            SearchResponse with similar documents
        """
        response = self.search(text, top_k=top_k + 1 if exclude_self else top_k)
        
        if exclude_self and response.results:
            # Remove exact match (score ~1.0)
            response.results = [r for r in response.results if r.score < 0.99]
            response.results = response.results[:top_k]
            response.total_results = len(response.results)
        
        return response
    
    def multi_query_search(
        self,
        queries: List[str],
        top_k_per_query: int = 3,
        total_results: int = 10,
    ) -> SearchResponse:
        """
        Search with multiple queries and combine results.
        
        Args:
            queries: List of search queries
            top_k_per_query: Results per query
            total_results: Total results to return
            
        Returns:
            Combined SearchResponse
        """
        if not queries:
            return SearchResponse(
                query="",
                results=[],
                total_results=0,
                search_time_ms=0,
            )
        
        start_time = time.time()
        
        # Collect results from all queries
        all_results: Dict[str, SearchResult] = {}
        
        for query in queries:
            response = self.search(query, top_k=top_k_per_query)
            
            for result in response.results:
                # Use content hash as key to deduplicate
                key = hash(result.content)
                
                if key not in all_results or result.score > all_results[key].score:
                    all_results[key] = result
        
        # Sort by score and take top results
        sorted_results = sorted(
            all_results.values(),
            key=lambda r: r.score,
            reverse=True,
        )[:total_results]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=" | ".join(queries),
            results=sorted_results,
            total_results=len(sorted_results),
            search_time_ms=round(search_time_ms, 2),
        )
    
    def get_sources(self) -> List[str]:
        """Get list of indexed sources."""
        return self.vector_store.list_sources()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        store_stats = self.vector_store.get_stats()
        
        return {
            **store_stats,
            "default_top_k": self.default_top_k,
            "min_score_threshold": self.min_score,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def get_search_engine(
    vector_store: Optional[VectorStore] = None,
) -> SemanticSearchEngine:
    """
    Get search engine instance.
    
    Args:
        vector_store: Optional vector store
        
    Returns:
        SemanticSearchEngine instance
    """
    return SemanticSearchEngine(vector_store=vector_store)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_search(
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Quick search function for simple use cases.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        List of result dictionaries
    """
    engine = get_search_engine()
    response = engine.search(query, top_k=top_k)
    return [r.to_dict() for r in response.results]


def format_results_for_display(
    results: List[SearchResult],
    max_content_length: int = 200,
) -> str:
    """
    Format search results for display.
    
    Args:
        results: Search results
        max_content_length: Maximum content length
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    lines = []
    for i, result in enumerate(results, 1):
        content = truncate_text(result.content, max_content_length)
        lines.append(
            f"\n[{i}] Score: {result.score:.3f} | Source: {result.source}\n"
            f"    {content}"
        )
    
    return "\n".join(lines)
