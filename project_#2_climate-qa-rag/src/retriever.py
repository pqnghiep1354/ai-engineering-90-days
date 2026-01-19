"""
Document retrieval strategies for Climate Q&A RAG System.

Supports basic retrieval, hybrid search, and query expansion.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from loguru import logger
from pydantic import Field

from .config import settings
from .utils import Timer, detect_language


# =============================================================================
# Base Retriever
# =============================================================================

class ClimateRetriever(BaseRetriever):
    """
    Custom retriever for climate documents with enhanced features.
    """
    
    vector_store: VectorStore = Field(description="Vector store for retrieval")
    search_type: str = Field(default="similarity", description="Search type")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: Optional[float] = Field(default=None, description="Minimum score threshold")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Query text
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        with Timer(f"Retrieval ({self.search_type})"):
            if self.search_type == "similarity":
                docs = self.vector_store.similarity_search(
                    query=query,
                    k=self.top_k,
                    filter=self.filter,
                )
            elif self.search_type == "mmr":
                docs = self.vector_store.max_marginal_relevance_search(
                    query=query,
                    k=self.top_k,
                    fetch_k=self.top_k * 4,
                    filter=self.filter,
                )
            elif self.search_type == "similarity_score":
                docs_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=self.top_k,
                    filter=self.filter,
                )
                # Filter by score threshold if set
                if self.score_threshold is not None:
                    docs_with_scores = [
                        (doc, score) for doc, score in docs_with_scores
                        if score >= self.score_threshold
                    ]
                docs = [doc for doc, _ in docs_with_scores]
            else:
                raise ValueError(f"Unknown search type: {self.search_type}")
        
        logger.debug(f"Retrieved {len(docs)} documents")
        return docs


# =============================================================================
# Hybrid Retriever
# =============================================================================

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector and keyword search.
    """
    
    vector_store: VectorStore = Field(description="Vector store")
    top_k: int = Field(default=5, description="Number of results")
    vector_weight: float = Field(default=0.7, description="Weight for vector search")
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Query text
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Get vector search results
        vector_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=self.top_k * 2,
        )
        
        # Simple keyword matching for BM25-like scoring
        keyword_scores = self._keyword_search(query, [doc for doc, _ in vector_results])
        
        # Combine scores
        combined_results = []
        for i, (doc, vector_score) in enumerate(vector_results):
            keyword_score = keyword_scores.get(i, 0)
            combined_score = (
                self.vector_weight * vector_score +
                self.keyword_weight * keyword_score
            )
            combined_results.append((doc, combined_score))
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in combined_results[:self.top_k]]
        
        return docs
    
    def _keyword_search(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[int, float]:
        """
        Simple keyword matching for scoring.
        
        Args:
            query: Query text
            documents: Documents to score
            
        Returns:
            Dictionary mapping document index to score
        """
        query_terms = set(query.lower().split())
        scores = {}
        
        for i, doc in enumerate(documents):
            doc_terms = set(doc.page_content.lower().split())
            overlap = len(query_terms & doc_terms)
            scores[i] = overlap / max(len(query_terms), 1)
        
        return scores


# =============================================================================
# Query Expansion Retriever
# =============================================================================

class QueryExpansionRetriever(BaseRetriever):
    """
    Retriever with automatic query expansion.
    """
    
    vector_store: VectorStore = Field(description="Vector store")
    llm: Any = Field(description="LLM for query expansion")
    top_k: int = Field(default=5, description="Number of results")
    num_expansions: int = Field(default=3, description="Number of expanded queries")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query using LLM.
        
        Args:
            query: Original query
            
        Returns:
            List of expanded queries
        """
        from .prompts import get_query_expansion_prompt
        
        language = detect_language(query)
        prompt = get_query_expansion_prompt(language)
        
        try:
            response = self.llm.invoke(prompt.format(question=query))
            expanded = response.content.strip().split("\n")
            expanded = [q.strip() for q in expanded if q.strip()]
            return [query] + expanded[:self.num_expansions]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return [query]
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve documents with query expansion.
        
        Args:
            query: Query text
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Expand query
        queries = self._expand_query(query)
        logger.debug(f"Expanded to {len(queries)} queries")
        
        # Retrieve for each query
        all_docs = {}
        for q in queries:
            results = self.vector_store.similarity_search_with_score(
                query=q,
                k=self.top_k,
            )
            for doc, score in results:
                doc_id = hash(doc.page_content)
                if doc_id not in all_docs or score > all_docs[doc_id][1]:
                    all_docs[doc_id] = (doc, score)
        
        # Sort by score and return top_k
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
        docs = [doc for doc, _ in sorted_docs[:self.top_k]]
        
        return docs


# =============================================================================
# Contextual Compression Retriever
# =============================================================================

class ContextualCompressionRetriever(BaseRetriever):
    """
    Retriever that compresses retrieved context to most relevant parts.
    """
    
    base_retriever: BaseRetriever = Field(description="Base retriever")
    llm: Any = Field(description="LLM for compression")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _compress_document(self, document: Document, query: str) -> Document:
        """
        Compress document to relevant parts.
        
        Args:
            document: Document to compress
            query: Query for relevance
            
        Returns:
            Compressed document
        """
        compress_prompt = f"""Extract only the parts of the following text that are relevant to answering the question.
If no parts are relevant, return "NOT_RELEVANT".

Question: {query}

Text: {document.page_content}

Relevant parts:"""
        
        try:
            response = self.llm.invoke(compress_prompt)
            compressed_content = response.content.strip()
            
            if compressed_content == "NOT_RELEVANT":
                return None
            
            return Document(
                page_content=compressed_content,
                metadata={**document.metadata, "compressed": True},
            )
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            return document
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Retrieve and compress documents.
        
        Args:
            query: Query text
            run_manager: Callback manager
            
        Returns:
            List of compressed documents
        """
        # Get base documents
        docs = self.base_retriever.invoke(query)
        
        # Compress each document
        compressed_docs = []
        for doc in docs:
            compressed = self._compress_document(doc, query)
            if compressed is not None:
                compressed_docs.append(compressed)
        
        return compressed_docs


# =============================================================================
# Retriever Factory
# =============================================================================

def get_retriever(
    vector_store: VectorStore,
    retriever_type: str = "basic",
    llm: Any = None,
    **kwargs,
) -> BaseRetriever:
    """
    Get retriever based on type.
    
    Args:
        vector_store: Vector store instance
        retriever_type: Type of retriever
        llm: LLM for advanced retrievers
        **kwargs: Additional arguments
        
    Returns:
        Retriever instance
    """
    top_k = kwargs.get("top_k", settings.retriever_top_k)
    
    if retriever_type == "basic":
        return ClimateRetriever(
            vector_store=vector_store,
            search_type="similarity",
            top_k=top_k,
        )
    elif retriever_type == "mmr":
        return ClimateRetriever(
            vector_store=vector_store,
            search_type="mmr",
            top_k=top_k,
        )
    elif retriever_type == "hybrid":
        return HybridRetriever(
            vector_store=vector_store,
            top_k=top_k,
            vector_weight=kwargs.get("vector_weight", 0.7),
            keyword_weight=kwargs.get("keyword_weight", 0.3),
        )
    elif retriever_type == "expansion":
        if llm is None:
            raise ValueError("LLM required for query expansion retriever")
        return QueryExpansionRetriever(
            vector_store=vector_store,
            llm=llm,
            top_k=top_k,
            num_expansions=kwargs.get("num_expansions", 3),
        )
    elif retriever_type == "compression":
        if llm is None:
            raise ValueError("LLM required for compression retriever")
        base_retriever = ClimateRetriever(
            vector_store=vector_store,
            search_type="similarity",
            top_k=top_k * 2,  # Retrieve more for compression
        )
        return ContextualCompressionRetriever(
            base_retriever=base_retriever,
            llm=llm,
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


# =============================================================================
# Multi-Query Retriever
# =============================================================================

def multi_query_retrieve(
    queries: List[str],
    vector_store: VectorStore,
    top_k: int = 5,
) -> List[Document]:
    """
    Retrieve documents for multiple queries and deduplicate.
    
    Args:
        queries: List of queries
        vector_store: Vector store
        top_k: Results per query
        
    Returns:
        Deduplicated list of documents
    """
    all_docs = {}
    
    for query in queries:
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
        )
        for doc, score in results:
            doc_id = hash(doc.page_content)
            if doc_id not in all_docs or score > all_docs[doc_id][1]:
                all_docs[doc_id] = (doc, score)
    
    # Sort by score and return
    sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:top_k]]
