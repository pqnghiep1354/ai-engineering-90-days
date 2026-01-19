"""
Vector store operations for Climate Q&A RAG System.

Supports ChromaDB (local) and Pinecone (cloud).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from loguru import logger

from .config import settings
from .embeddings import get_embedding_model
from .utils import Timer


# =============================================================================
# Vector Store Factory
# =============================================================================

def get_vector_store(
    embeddings: Optional[Embeddings] = None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Get vector store instance based on configuration.
    
    Args:
        embeddings: Embedding model (default from settings)
        collection_name: Collection name
        persist_directory: Persistence directory for local stores
        
    Returns:
        VectorStore instance
    """
    if embeddings is None:
        embeddings = get_embedding_model()
    
    if settings.vector_db_type == "chroma":
        return get_chroma_store(
            embeddings=embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
    elif settings.vector_db_type == "pinecone":
        return get_pinecone_store(
            embeddings=embeddings,
            index_name=collection_name,
        )
    else:
        raise ValueError(f"Unknown vector DB type: {settings.vector_db_type}")


def get_chroma_store(
    embeddings: Embeddings,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> Chroma:
    """
    Get ChromaDB vector store.
    
    Args:
        embeddings: Embedding model
        collection_name: Collection name
        persist_directory: Directory for persistence
        
    Returns:
        Chroma instance
    """
    collection_name = collection_name or settings.chroma_collection_name
    persist_directory = persist_directory or settings.chroma_persist_directory
    
    # Ensure directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    logger.info(
        f"Initializing ChromaDB: collection={collection_name}, "
        f"persist_dir={persist_directory}"
    )
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def get_pinecone_store(
    embeddings: Embeddings,
    index_name: Optional[str] = None,
) -> VectorStore:
    """
    Get Pinecone vector store.
    
    Args:
        embeddings: Embedding model
        index_name: Pinecone index name
        
    Returns:
        Pinecone instance
    """
    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone
    except ImportError:
        raise ImportError(
            "Pinecone not available. Install with: pip install pinecone-client langchain-pinecone"
        )
    
    if not settings.pinecone_api_key:
        raise ValueError("Pinecone API key not configured")
    
    index_name = index_name or settings.pinecone_index_name
    
    logger.info(f"Initializing Pinecone: index={index_name}")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    return PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=embeddings,
    )


# =============================================================================
# Vector Store Manager
# =============================================================================

class VectorStoreManager:
    """
    Manager for vector store operations.
    """
    
    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize vector store manager.
        
        Args:
            embeddings: Embedding model
            collection_name: Collection name
        """
        self.embeddings = embeddings or get_embedding_model()
        self.collection_name = collection_name or settings.chroma_collection_name
        self._vector_store: Optional[VectorStore] = None
    
    @property
    def vector_store(self) -> VectorStore:
        """Get or create vector store."""
        if self._vector_store is None:
            self._vector_store = get_vector_store(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
            )
        return self._vector_store
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            documents: Documents to add
            batch_size: Batch size for insertion
            show_progress: Whether to show progress
            
        Returns:
            List of document IDs
        """
        all_ids = []
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        with Timer(f"Adding {len(documents)} documents"):
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                if show_progress:
                    logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                ids = self.vector_store.add_documents(batch)
                all_ids.extend(ids)
        
        logger.info(f"Added {len(all_ids)} documents to vector store")
        return all_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results (default from settings)
            filter: Metadata filter
            
        Returns:
            List of similar documents
        """
        k = k or settings.retriever_top_k
        
        with Timer(f"Similarity search (k={k})"):
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )
        
        logger.debug(f"Found {len(results)} similar documents")
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query text
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        k = k or settings.retriever_top_k
        
        with Timer(f"Similarity search with scores (k={k})"):
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter,
            )
        
        logger.debug(f"Found {len(results)} similar documents with scores")
        return results
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> List[Document]:
        """
        Perform MMR search for diverse results.
        
        Args:
            query: Query text
            k: Number of results
            fetch_k: Number of candidates to fetch
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            
        Returns:
            List of diverse documents
        """
        k = k or settings.retriever_top_k
        
        with Timer(f"MMR search (k={k}, fetch_k={fetch_k})"):
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        
        logger.debug(f"Found {len(results)} diverse documents")
        return results
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        if isinstance(self.vector_store, Chroma):
            self.vector_store.delete_collection()
            self._vector_store = None
            logger.info(f"Deleted collection: {self.collection_name}")
        else:
            logger.warning("Delete collection not supported for this vector store type")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Statistics dictionary
        """
        if isinstance(self.vector_store, Chroma):
            collection = self.vector_store._collection
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "type": "chroma",
            }
        else:
            return {
                "name": self.collection_name,
                "type": settings.vector_db_type,
            }


# =============================================================================
# Indexing Functions
# =============================================================================

def create_index(
    documents: List[Document],
    embeddings: Optional[Embeddings] = None,
    collection_name: Optional[str] = None,
    batch_size: int = 100,
) -> VectorStoreManager:
    """
    Create a new index from documents.
    
    Args:
        documents: Documents to index
        embeddings: Embedding model
        collection_name: Collection name
        batch_size: Batch size for insertion
        
    Returns:
        VectorStoreManager instance
    """
    manager = VectorStoreManager(
        embeddings=embeddings,
        collection_name=collection_name,
    )
    
    manager.add_documents(documents, batch_size=batch_size)
    
    return manager


def load_existing_index(
    embeddings: Optional[Embeddings] = None,
    collection_name: Optional[str] = None,
) -> VectorStoreManager:
    """
    Load an existing index.
    
    Args:
        embeddings: Embedding model
        collection_name: Collection name
        
    Returns:
        VectorStoreManager instance
    """
    manager = VectorStoreManager(
        embeddings=embeddings,
        collection_name=collection_name,
    )
    
    # Access vector_store to initialize it
    _ = manager.vector_store
    
    stats = manager.get_collection_stats()
    logger.info(f"Loaded existing index: {stats}")
    
    return manager
