"""
Reranking models for Climate Q&A RAG System.

Supports cross-encoder reranking for improved retrieval accuracy.
"""

from typing import List, Optional, Tuple

from langchain_core.documents import Document
from loguru import logger

from .config import settings
from .utils import Timer


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================

class CrossEncoderReranker:
    """
    Reranker using cross-encoder models for accurate relevance scoring.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        Initialize reranker.
        
        Args:
            model_name: Cross-encoder model name
            top_k: Number of documents to return after reranking
            device: Device to run model on
        """
        self.model_name = model_name or settings.reranker_model
        self.top_k = top_k or settings.reranker_top_k
        self.device = device
        self._model = None
        
        logger.info(f"CrossEncoderReranker initialized with model: {self.model_name}")
    
    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Loaded cross-encoder model: {self.model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not available. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (document, score) tuples, sorted by relevance
        """
        if not documents:
            return []
        
        top_k = top_k or self.top_k
        
        with Timer(f"Reranking {len(documents)} documents"):
            # Create query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending) and return top_k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked = doc_scores[:top_k]
        
        logger.debug(
            f"Reranked to top {len(reranked)} documents. "
            f"Score range: {reranked[-1][1]:.3f} - {reranked[0][1]:.3f}"
        )
        
        return reranked
    
    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents and filter by score threshold.
        
        Args:
            query: Query text
            documents: Documents to rerank
            threshold: Minimum score to include
            top_k: Maximum number of documents
            
        Returns:
            Filtered and reranked documents
        """
        reranked = self.rerank(query, documents, top_k=len(documents))
        
        # Filter by threshold
        filtered = [(doc, score) for doc, score in reranked if score >= threshold]
        
        # Apply top_k after filtering
        if top_k:
            filtered = filtered[:top_k]
        
        logger.debug(
            f"After threshold {threshold}: {len(filtered)} documents remain"
        )
        
        return filtered


# =============================================================================
# LLM-based Reranker
# =============================================================================

class LLMReranker:
    """
    Reranker using LLM for relevance assessment.
    
    Slower but can provide explanations for rankings.
    """
    
    def __init__(self, llm, top_k: Optional[int] = None):
        """
        Initialize LLM reranker.
        
        Args:
            llm: Language model instance
            top_k: Number of documents to return
        """
        self.llm = llm
        self.top_k = top_k or settings.reranker_top_k
    
    def _score_document(self, query: str, document: Document) -> float:
        """
        Score a single document's relevance.
        
        Args:
            query: Query text
            document: Document to score
            
        Returns:
            Relevance score (0-1)
        """
        prompt = f"""Rate the relevance of the following document to the query on a scale of 0 to 1.
Only respond with a single number.

Query: {query}

Document: {document.page_content[:1000]}

Relevance score:"""
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return max(0, min(1, score))  # Clamp to [0, 1]
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse LLM score: {e}")
            return 0.5  # Default score
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using LLM scoring.
        
        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of top documents
            
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        top_k = top_k or self.top_k
        
        with Timer(f"LLM reranking {len(documents)} documents"):
            # Score each document
            doc_scores = []
            for doc in documents:
                score = self._score_document(query, doc)
                doc_scores.append((doc, score))
            
            # Sort by score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return doc_scores[:top_k]


# =============================================================================
# Cohere Reranker (Cloud-based)
# =============================================================================

class CohereReranker:
    """
    Reranker using Cohere's rerank API.
    
    High quality but requires API key and has usage costs.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
        top_k: Optional[int] = None,
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key
            model: Rerank model name
            top_k: Number of documents to return
        """
        self.api_key = api_key
        self.model = model
        self.top_k = top_k or settings.reranker_top_k
        self._client = None
    
    @property
    def client(self):
        """Lazy load Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError(
                    "Cohere not available. Install with: pip install cohere"
                )
        return self._client
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Rerank documents using Cohere API.
        
        Args:
            query: Query text
            documents: Documents to rerank
            top_k: Number of top documents
            
        Returns:
            List of (document, score) tuples
        """
        if not documents:
            return []
        
        top_k = top_k or self.top_k
        
        with Timer(f"Cohere reranking {len(documents)} documents"):
            # Extract document texts
            doc_texts = [doc.page_content for doc in documents]
            
            # Call Cohere rerank API
            results = self.client.rerank(
                query=query,
                documents=doc_texts,
                model=self.model,
                top_n=top_k,
            )
            
            # Map results back to documents
            reranked = []
            for result in results.results:
                doc = documents[result.index]
                reranked.append((doc, result.relevance_score))
        
        return reranked


# =============================================================================
# Reranker Factory
# =============================================================================

def get_reranker(
    reranker_type: str = "cross-encoder",
    **kwargs,
):
    """
    Get reranker instance based on type.
    
    Args:
        reranker_type: Type of reranker
        **kwargs: Additional arguments
        
    Returns:
        Reranker instance
    """
    if reranker_type == "cross-encoder":
        return CrossEncoderReranker(
            model_name=kwargs.get("model_name"),
            top_k=kwargs.get("top_k"),
            device=kwargs.get("device", "cpu"),
        )
    elif reranker_type == "llm":
        if "llm" not in kwargs:
            raise ValueError("LLM instance required for LLM reranker")
        return LLMReranker(
            llm=kwargs["llm"],
            top_k=kwargs.get("top_k"),
        )
    elif reranker_type == "cohere":
        return CohereReranker(
            api_key=kwargs.get("api_key"),
            model=kwargs.get("model", "rerank-english-v3.0"),
            top_k=kwargs.get("top_k"),
        )
    else:
        raise ValueError(f"Unknown reranker type: {reranker_type}")


# =============================================================================
# Utility Functions
# =============================================================================

def rerank_documents(
    query: str,
    documents: List[Document],
    reranker_type: str = "cross-encoder",
    top_k: Optional[int] = None,
    **kwargs,
) -> List[Document]:
    """
    Convenience function to rerank documents.
    
    Args:
        query: Query text
        documents: Documents to rerank
        reranker_type: Type of reranker
        top_k: Number of documents to return
        **kwargs: Additional reranker arguments
        
    Returns:
        Reranked documents (without scores)
    """
    if not settings.use_reranker:
        logger.debug("Reranking disabled, returning original documents")
        return documents[:top_k] if top_k else documents
    
    reranker = get_reranker(reranker_type, top_k=top_k, **kwargs)
    reranked = reranker.rerank(query, documents, top_k=top_k)
    
    return [doc for doc, _ in reranked]
