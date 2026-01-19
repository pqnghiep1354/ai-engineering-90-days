#!/usr/bin/env python3
"""
Document indexing script for Climate Q&A RAG System.

Usage:
    python scripts/index_documents.py --data-dir data/sample
    python scripts/index_documents.py --data-dir /path/to/docs --collection my_collection
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import settings
from src.document_loader import ClimateDocumentLoader, get_document_stats, deduplicate_documents
from src.embeddings import get_embedding_model
from src.vector_store import VectorStoreManager
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index documents for Climate Q&A RAG System"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing documents to index",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name (default from settings)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for splitting (default from settings)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap (default from settings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before indexing",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=True,
        help="Remove duplicate documents",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="gemini",
        choices=["openai", "gemini", "local"],
        help="Embedding provider (default: gemini)",
    )
    return parser.parse_args()


def main():
    """Main indexing function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    logger.info(f"Starting document indexing from: {data_dir}")
    
    # Initialize document loader
    loader = ClimateDocumentLoader(
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
    )
    
    # Load and split documents
    logger.info("Loading documents...")
    documents = loader.load_and_split(data_dir, show_progress=True)
    
    if not documents:
        logger.warning("No documents found to index")
        sys.exit(0)
    
    # Deduplicate if enabled
    if args.deduplicate:
        documents = deduplicate_documents(documents)
    
    # Print statistics
    stats = get_document_stats(documents)
    logger.info(f"Document statistics:")
    logger.info(f"  - Total chunks: {stats['count']}")
    logger.info(f"  - Total characters: {stats['total_characters']:,}")
    logger.info(f"  - Average chunk size: {stats['avg_length']:.0f}")
    logger.info(f"  - Unique sources: {stats['unique_sources']}")
    
    # Initialize embeddings and vector store
    logger.info(f"Initializing embedding model with provider: {args.embedding_provider}")
    embeddings = get_embedding_model(provider=args.embedding_provider)
    
    collection_name = args.collection or settings.chroma_collection_name
    manager = VectorStoreManager(
        embeddings=embeddings,
        collection_name=collection_name,
    )
    
    # Clear existing collection if requested
    if args.clear:
        logger.warning(f"Clearing existing collection: {collection_name}")
        try:
            manager.delete_collection()
            manager = VectorStoreManager(
                embeddings=embeddings,
                collection_name=collection_name,
            )
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")
    
    # Index documents
    logger.info(f"Indexing {len(documents)} documents...")
    ids = manager.add_documents(
        documents,
        batch_size=args.batch_size,
        show_progress=True,
    )
    
    # Print final statistics
    final_stats = manager.get_collection_stats()
    logger.info(f"Indexing complete!")
    logger.info(f"  - Collection: {final_stats['name']}")
    logger.info(f"  - Total documents: {final_stats.get('count', len(ids))}")
    
    logger.info(f"\nTo use the indexed documents, run:")
    logger.info(f"  streamlit run src/app.py")


if __name__ == "__main__":
    main()
