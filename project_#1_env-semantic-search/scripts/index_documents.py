#!/usr/bin/env python3
"""
Document indexing script for Environmental Semantic Search Tool.

Usage:
    python scripts/index_documents.py --data-dir data/documents
    python scripts/index_documents.py --data-dir /path/to/docs --clear
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import settings
from src.document_processor import (
    DocumentProcessor,
    deduplicate_documents,
    get_document_stats,
)
from src.vector_store import VectorStore, load_index
from src.utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Index documents for semantic search"
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
        help="Chunk size (default from settings)",
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap (default from settings)",
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before adding",
    )
    
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main indexing function."""
    args = parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    
    # Validate directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        sys.exit(1)
    
    logger.info(f"📂 Indexing documents from: {data_dir}")
    
    # Initialize processor
    processor = DocumentProcessor(
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
    )
    
    # Process documents
    logger.info("📄 Loading and processing documents...")
    result = processor.process_directory(data_dir, show_progress=True)
    
    if not result.documents:
        logger.warning("No documents found to index")
        sys.exit(0)
    
    # Deduplicate
    documents = result.documents
    if not args.no_dedup:
        documents = deduplicate_documents(documents)
    
    # Print statistics
    stats = get_document_stats(documents)
    logger.info("📊 Document Statistics:")
    logger.info(f"   Total chunks: {stats['count']}")
    logger.info(f"   Total characters: {stats['total_characters']:,}")
    logger.info(f"   Average chunk size: {stats['avg_length']:.0f}")
    logger.info(f"   Unique sources: {stats['unique_sources']}")
    
    # Initialize vector store
    logger.info("🔧 Initializing vector store...")
    collection_name = args.collection or settings.chroma_collection_name
    
    vector_store = VectorStore(collection_name=collection_name)
    
    # Clear if requested
    if args.clear:
        logger.warning(f"🗑️  Clearing existing collection: {collection_name}")
        vector_store.delete_collection()
    
    # Index documents
    logger.info(f"📥 Indexing {len(documents)} document chunks...")
    added = vector_store.add_documents(documents, show_progress=True)
    
    # Final statistics
    final_stats = vector_store.get_stats()
    
    logger.info("✅ Indexing complete!")
    logger.info(f"   Collection: {final_stats['collection_name']}")
    logger.info(f"   Total documents: {final_stats['document_count']}")
    logger.info(f"   Embedding model: {final_stats['embedding_model']}")
    
    # Print usage instructions
    logger.info("\n🚀 To start searching:")
    logger.info("   Web App:  streamlit run src/app.py")
    logger.info("   API:      uvicorn src.api:app --reload")
    logger.info("   CLI:      python src/cli.py 'your query'")


if __name__ == "__main__":
    main()
