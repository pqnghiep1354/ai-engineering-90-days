"""
Document loading and processing for Climate Q&A RAG System.

Supports multiple document formats: PDF, TXT, MD, DOCX
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document
from loguru import logger

from .config import settings
from .utils import (
    clean_text,
    get_file_hash,
    get_file_extension,
    is_supported_file,
    list_files_recursive,
    Timer,
    DocumentLoadError,
)


# =============================================================================
# Document Loaders
# =============================================================================

class DocumentLoader:
    """
    Universal document loader supporting multiple formats.
    """
    
    LOADER_MAPPING = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": Docx2txtLoader,
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        clean_documents: bool = True,
    ):
        """
        Initialize document loader.
        
        Args:
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            clean_documents: Whether to clean text content
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.clean_documents = clean_documents
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        logger.info(
            f"DocumentLoader initialized with chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )
    
    def load_file(self, file_path: Path) -> List[Document]:
        """
        Load a single file.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentLoadError(f"File not found: {file_path}")
        
        if not is_supported_file(file_path):
            raise DocumentLoadError(f"Unsupported file type: {file_path.suffix}")
        
        extension = get_file_extension(file_path)
        loader_class = self.LOADER_MAPPING.get(extension)
        
        if loader_class is None:
            raise DocumentLoadError(f"No loader for extension: {extension}")
        
        try:
            with Timer(f"Loading {file_path.name}"):
                loader = loader_class(str(file_path))
                documents = loader.load()
                
                # Add metadata
                file_hash = get_file_hash(file_path)
                for doc in documents:
                    doc.metadata.update({
                        "source": file_path.name,
                        "source_path": str(file_path),
                        "file_type": extension,
                        "file_hash": file_hash,
                    })
                
                logger.info(f"Loaded {len(documents)} pages from {file_path.name}")
                return documents
                
        except Exception as e:
            raise DocumentLoadError(f"Error loading {file_path}: {e}")
    
    def load_directory(
        self,
        directory: Path,
        glob_pattern: str = "**/*",
        show_progress: bool = True,
    ) -> List[Document]:
        """
        Load all supported files from a directory.
        
        Args:
            directory: Directory path
            glob_pattern: Glob pattern for file matching
            show_progress: Whether to show progress
            
        Returns:
            List of Document objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise DocumentLoadError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise DocumentLoadError(f"Not a directory: {directory}")
        
        # Get all supported files
        files = list_files_recursive(directory)
        logger.info(f"Found {len(files)} supported files in {directory}")
        
        all_documents = []
        errors = []
        
        for i, file_path in enumerate(files):
            if show_progress:
                logger.info(f"Processing [{i+1}/{len(files)}]: {file_path.name}")
            
            try:
                documents = self.load_file(file_path)
                all_documents.extend(documents)
            except DocumentLoadError as e:
                logger.warning(f"Skipping {file_path.name}: {e}")
                errors.append((file_path, str(e)))
        
        if errors:
            logger.warning(f"Failed to load {len(errors)} files")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents
        """
        with Timer("Splitting documents"):
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["chunk_size"] = len(chunk.page_content)
                
                # Clean content if enabled
                if self.clean_documents:
                    chunk.page_content = clean_text(chunk.page_content)
            
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
    
    def load_and_split(
        self,
        source: Path,
        show_progress: bool = True,
    ) -> List[Document]:
        """
        Load and split documents in one step.
        
        Args:
            source: File or directory path
            show_progress: Whether to show progress
            
        Returns:
            List of chunked documents
        """
        source = Path(source)
        
        if source.is_file():
            documents = self.load_file(source)
        else:
            documents = self.load_directory(source, show_progress=show_progress)
        
        return self.split_documents(documents)


# =============================================================================
# Specialized Loaders
# =============================================================================

class ClimateDocumentLoader(DocumentLoader):
    """
    Specialized loader for climate/environmental documents.
    
    Adds domain-specific metadata extraction and processing.
    """
    
    # Keywords for categorization
    CATEGORIES = {
        "climate_change": [
            "climate change", "global warming", "greenhouse gas",
            "carbon dioxide", "temperature rise", "sea level",
            "biến đổi khí hậu", "nóng lên toàn cầu", "khí nhà kính"
        ],
        "air_quality": [
            "air quality", "pollution", "particulate matter", "PM2.5",
            "emissions", "smog", "chất lượng không khí", "ô nhiễm"
        ],
        "biodiversity": [
            "biodiversity", "ecosystem", "species", "conservation",
            "habitat", "đa dạng sinh học", "hệ sinh thái", "bảo tồn"
        ],
        "energy": [
            "renewable energy", "solar", "wind", "hydropower",
            "fossil fuel", "năng lượng tái tạo", "năng lượng mặt trời"
        ],
        "policy": [
            "regulation", "policy", "agreement", "protocol", "law",
            "quy định", "chính sách", "luật", "nghị định"
        ],
        "esg": [
            "ESG", "sustainability", "environmental social governance",
            "carbon footprint", "phát triển bền vững", "dấu chân carbon"
        ],
    }
    
    def _categorize_document(self, content: str) -> List[str]:
        """
        Categorize document based on content.
        
        Args:
            content: Document content
            
        Returns:
            List of categories
        """
        content_lower = content.lower()
        categories = []
        
        for category, keywords in self.CATEGORIES.items():
            if any(keyword in content_lower for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ["general"]
    
    def _extract_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Extract additional metadata from document.
        
        Args:
            document: Document to process
            
        Returns:
            Additional metadata
        """
        content = document.page_content
        
        metadata = {
            "categories": self._categorize_document(content),
            "word_count": len(content.split()),
            "has_numbers": bool(any(char.isdigit() for char in content)),
            "has_citations": "[" in content or "(" in content,
        }
        
        return metadata
    
    def load_and_split(
        self,
        source: Path,
        show_progress: bool = True,
    ) -> List[Document]:
        """
        Load, split, and enhance documents with climate-specific metadata.
        
        Args:
            source: File or directory path
            show_progress: Whether to show progress
            
        Returns:
            List of enhanced documents
        """
        chunks = super().load_and_split(source, show_progress)
        
        # Add climate-specific metadata
        for chunk in chunks:
            extra_metadata = self._extract_metadata(chunk)
            chunk.metadata.update(extra_metadata)
        
        return chunks


# =============================================================================
# Document Processing Utilities
# =============================================================================

def deduplicate_documents(
    documents: List[Document],
    similarity_threshold: float = 0.95,
) -> List[Document]:
    """
    Remove near-duplicate documents based on content hash.
    
    Args:
        documents: List of documents
        similarity_threshold: Not used (reserved for future fuzzy dedup)
        
    Returns:
        Deduplicated documents
    """
    seen_hashes = set()
    unique_documents = []
    
    for doc in documents:
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_documents.append(doc)
    
    removed = len(documents) - len(unique_documents)
    if removed > 0:
        logger.info(f"Removed {removed} duplicate documents")
    
    return unique_documents


def filter_documents_by_length(
    documents: List[Document],
    min_length: int = 50,
    max_length: int = 10000,
) -> List[Document]:
    """
    Filter documents by content length.
    
    Args:
        documents: List of documents
        min_length: Minimum content length
        max_length: Maximum content length
        
    Returns:
        Filtered documents
    """
    filtered = [
        doc for doc in documents
        if min_length <= len(doc.page_content) <= max_length
    ]
    
    removed = len(documents) - len(filtered)
    if removed > 0:
        logger.info(f"Filtered out {removed} documents by length")
    
    return filtered


def get_document_stats(documents: List[Document]) -> Dict[str, Any]:
    """
    Calculate statistics about documents.
    
    Args:
        documents: List of documents
        
    Returns:
        Statistics dictionary
    """
    if not documents:
        return {"count": 0}
    
    lengths = [len(doc.page_content) for doc in documents]
    sources = set(doc.metadata.get("source", "unknown") for doc in documents)
    
    return {
        "count": len(documents),
        "total_characters": sum(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "unique_sources": len(sources),
        "sources": list(sources),
    }
