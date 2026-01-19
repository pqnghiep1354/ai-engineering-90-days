"""
Vector store operations for Environmental Semantic Search Tool.

Handles indexing and retrieval using ChromaDB.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from .config import settings
from .document_processor import Document
from .embeddings import EmbeddingModel, get_embedding_model
from .utils import Timer


# =============================================================================
# ChromaDB Vector Store
# =============================================================================

class VectorStore:
    """
    Vector store using ChromaDB for document storage and retrieval.
    """
    
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistence
            embedding_model: Embedding model instance
        """
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_dir
        
        # Initialize embedding model
        self.embedding_model = embedding_model or get_embedding_model()
        
        # Create persist directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        
        logger.info(
            f"VectorStore initialized: collection={self.collection_name}, "
            f"persist_dir={self.persist_directory}"
        )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        show_progress: bool = True,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for processing
            show_progress: Show progress logs
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        total_added = 0
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        with Timer(f"Indexing {len(documents)} documents"):
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(documents))
                batch = documents[start_idx:end_idx]
                
                if show_progress:
                    logger.info(f"Indexing batch {batch_num + 1}/{total_batches}")
                
                try:
                    # Extract data
                    ids = [doc.id for doc in batch]
                    texts = [doc.content for doc in batch]
                    metadatas = [doc.metadata for doc in batch]
                    
                    # Generate embeddings
                    embeddings = self.embedding_model.embed_documents(texts)
                    
                    # Add to collection
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=texts,
                        metadatas=metadatas,
                    )
                    
                    total_added += len(batch)
                    
                except Exception as e:
                    logger.error(f"Error indexing batch: {e}")
                    continue
        
        logger.info(f"Added {total_added} documents to vector store")
        return total_added
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filter: Metadata filter
            
        Returns:
            List of search results with content, score, and metadata
        """
        threshold = threshold if threshold is not None else settings.min_similarity_score
        
        with Timer(f"Search: '{query[:50]}...'"):
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter,
                include=["documents", "metadatas", "distances"],
            )
        
        # Format results
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB returns distances (0 = identical), convert to similarity
                distance = results["distances"][0][i]
                similarity = 1 - distance  # For cosine distance
                
                # Apply threshold
                if similarity < threshold:
                    continue
                
                search_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "score": round(similarity, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })
        
        logger.debug(f"Found {len(search_results)} results above threshold {threshold}")
        return search_results
    
    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Search and return documents with scores.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        results = self.search(query, top_k=top_k, threshold=0)
        
        return [
            (
                Document(
                    content=r["content"],
                    metadata=r["metadata"],
                ),
                r["score"]
            )
            for r in results
        ]
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.warning(f"Deleted collection: {self.collection_name}")
            
            # Recreate empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Statistics dictionary
        """
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimensions": self.embedding_model.dimensions,
        }
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"],
            )
            
            if result["ids"]:
                return {
                    "id": result["ids"][0],
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0] if result["metadatas"] else {},
                }
        except Exception:
            pass
        
        return None
    
    def list_sources(self) -> List[str]:
        """
        List all unique document sources.
        
        Returns:
            List of source names
        """
        # Get all documents (limited for performance)
        result = self.collection.get(
            limit=10000,
            include=["metadatas"],
        )
        
        sources = set()
        if result["metadatas"]:
            for metadata in result["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
        
        return sorted(sources)


# =============================================================================
# Factory Functions
# =============================================================================

def get_vector_store(
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
) -> VectorStore:
    """
    Get vector store instance.
    
    Args:
        collection_name: Collection name
        persist_directory: Persistence directory
        
    Returns:
        VectorStore instance
    """
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


def create_index(
    documents: List[Document],
    collection_name: Optional[str] = None,
    clear_existing: bool = False,
) -> VectorStore:
    """
    Create a new index from documents.
    
    Args:
        documents: Documents to index
        collection_name: Collection name
        clear_existing: Clear existing collection
        
    Returns:
        VectorStore instance
    """
    store = get_vector_store(collection_name=collection_name)
    
    if clear_existing:
        store.delete_collection()
    
    store.add_documents(documents)
    
    return store


def load_index(
    collection_name: Optional[str] = None,
) -> VectorStore:
    """
    Load existing index.
    
    Args:
        collection_name: Collection name
        
    Returns:
        VectorStore instance
    """
    store = get_vector_store(collection_name=collection_name)
    stats = store.get_stats()
    
    logger.info(f"Loaded index with {stats['document_count']} documents")
    
    return store
