"""
Utility functions for Climate Q&A RAG System.
"""

import hashlib
import re
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import tiktoken
from loguru import logger

T = TypeVar("T")


# =============================================================================
# Text Processing Utilities
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep Vietnamese
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:\-()"\']', '', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        model: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def split_text_by_tokens(
    text: str,
    max_tokens: int = 500,
    model: str = "gpt-4o-mini"
) -> List[str]:
    """
    Split text into chunks by token count.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk
        model: Model name for tokenizer
        
    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    
    return chunks


def detect_language(text: str) -> str:
    """
    Simple language detection (Vietnamese vs English).
    
    Args:
        text: Input text
        
    Returns:
        Language code: "vi" or "en"
    """
    # Vietnamese-specific characters
    vietnamese_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    vietnamese_chars.update(c.upper() for c in vietnamese_chars)
    
    text_lower = text.lower()
    vietnamese_count = sum(1 for char in text_lower if char in vietnamese_chars)
    
    # If more than 5% Vietnamese characters, consider it Vietnamese
    if len(text) > 0 and vietnamese_count / len(text) > 0.05:
        return "vi"
    return "en"


# =============================================================================
# File Utilities
# =============================================================================

def get_file_hash(file_path: Path) -> str:
    """
    Calculate MD5 hash of a file.
    
    Args:
        file_path: Path to file
        
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_extension(file_path: Path) -> str:
    """
    Get file extension in lowercase.
    
    Args:
        file_path: Path to file
        
    Returns:
        File extension (e.g., ".pdf")
    """
    return file_path.suffix.lower()


def is_supported_file(file_path: Path) -> bool:
    """
    Check if file type is supported for indexing.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if supported
    """
    supported_extensions = {".pdf", ".txt", ".md", ".docx", ".doc", ".html"}
    return get_file_extension(file_path) in supported_extensions


def list_files_recursive(directory: Path, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    List all files in directory recursively.
    
    Args:
        directory: Directory path
        extensions: Optional list of extensions to filter
        
    Returns:
        List of file paths
    """
    if extensions is None:
        extensions = [".pdf", ".txt", ".md", ".docx"]
    
    files = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source documents for display.
    
    Args:
        sources: List of source document dictionaries
        
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources found."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        content = source.get("content", source.get("page_content", ""))
        metadata = source.get("metadata", {})
        
        source_name = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")
        
        formatted.append(
            f"**Source {i}**: {source_name} (Page: {page})\n"
            f"{truncate_text(content, 300)}\n"
        )
    
    return "\n".join(formatted)


def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """
    Format chat history for display or prompt.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        
    Returns:
        Formatted string
    """
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format datetime as string.
    
    Args:
        dt: Datetime object (default: now)
        
    Returns:
        Formatted string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# =============================================================================
# Timing and Performance Utilities
# =============================================================================

def timer_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.3f}s")
        return result
    return wrapper


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Block"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args) -> None:
        self.end_time = time.time()
        logger.debug(f"{self.name} took {self.elapsed:.3f}s")
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


# =============================================================================
# Validation Utilities
# =============================================================================

def validate_api_key(api_key: Optional[str], provider: str = "OpenAI") -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        provider: Provider name for logging
        
    Returns:
        True if valid format
    """
    if not api_key:
        logger.warning(f"{provider} API key not set")
        return False
    
    if len(api_key) < 10:
        logger.warning(f"{provider} API key appears too short")
        return False
    
    return True


def validate_model_name(model: str, provider: str = "openai") -> bool:
    """
    Validate model name.
    
    Args:
        model: Model name
        provider: Provider name
        
    Returns:
        True if valid
    """
    valid_models = {
        "openai": [
            "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
            "gpt-3.5-turbo", "text-embedding-3-small", "text-embedding-3-large"
        ],
        "anthropic": [
            "claude-3-opus-20240229", "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"
        ],
    }
    
    provider_models = valid_models.get(provider.lower(), [])
    return model in provider_models


# =============================================================================
# Error Handling Utilities
# =============================================================================

class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass


class DocumentLoadError(RAGError):
    """Error loading documents."""
    pass


class EmbeddingError(RAGError):
    """Error generating embeddings."""
    pass


class RetrievalError(RAGError):
    """Error during document retrieval."""
    pass


class GenerationError(RAGError):
    """Error during response generation."""
    pass


def safe_execute(
    func: Callable[..., T],
    *args,
    default: Optional[T] = None,
    error_message: str = "Operation failed",
    **kwargs
) -> Optional[T]:
    """
    Execute function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        error_message: Error message to log
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return default


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging with loguru.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    from loguru import logger
    import sys
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    
    # Add file handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
        )
