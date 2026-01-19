"""
Utility functions for Environmental Semantic Search Tool.
"""

import hashlib
import re
import sys
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import tiktoken
from loguru import logger

T = TypeVar("T")


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
    # Remove default handler
    logger.remove()
    
    # Console handler with colors
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    
    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip",
        )


# =============================================================================
# Text Processing
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
    text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF.,!?;:\-()"\'\[\]]', '', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix for truncated text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text.
    
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


def detect_language(text: str) -> str:
    """
    Simple language detection (Vietnamese vs English).
    
    Args:
        text: Input text
        
    Returns:
        Language code: "vi" or "en"
    """
    vietnamese_chars = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")
    vietnamese_chars.update(list(c.upper() for c in vietnamese_chars))
    
    text_lower = text.lower()
    vietnamese_count = sum(1 for char in text_lower if char in vietnamese_chars)
    
    if len(text) > 0 and vietnamese_count / len(text) > 0.03:
        return "vi"
    return "en"


# =============================================================================
# File Operations
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
    """Get file extension in lowercase."""
    return file_path.suffix.lower()


def is_supported_file(file_path: Path) -> bool:
    """Check if file type is supported."""
    supported = {".pdf", ".txt", ".md", ".docx", ".html", ".htm"}
    return get_file_extension(file_path) in supported


def list_documents(directory: Path, recursive: bool = True) -> List[Path]:
    """
    List all supported documents in directory.
    
    Args:
        directory: Directory path
        recursive: Search recursively
        
    Returns:
        List of document paths
    """
    if not directory.exists():
        return []
    
    documents = []
    pattern = "**/*" if recursive else "*"
    
    for path in directory.glob(pattern):
        if path.is_file() and is_supported_file(path):
            documents.append(path)
    
    return sorted(documents)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


# =============================================================================
# Timing Utilities
# =============================================================================

def timer_decorator(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.3f}s")
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
        logger.debug(f"{self.name} took {self.elapsed_ms:.1f}ms")
    
    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed * 1000


# =============================================================================
# Formatting
# =============================================================================

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime as string."""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_search_results(results: List[Dict[str, Any]]) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    lines = []
    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        content = truncate_text(result.get("content", ""), 200)
        source = result.get("metadata", {}).get("source", "Unknown")
        
        lines.append(
            f"[{i}] Score: {score:.3f} | Source: {source}\n"
            f"    {content}\n"
        )
    
    return "\n".join(lines)


# =============================================================================
# Validation
# =============================================================================

def validate_api_key(api_key: Optional[str], provider: str = "OpenAI") -> bool:
    """Validate API key format."""
    if not api_key or len(api_key) < 10:
        logger.warning(f"{provider} API key not configured or too short")
        return False
    return True


# =============================================================================
# Error Handling
# =============================================================================

class SearchError(Exception):
    """Base exception for search errors."""
    pass


class DocumentError(SearchError):
    """Error processing documents."""
    pass


class EmbeddingError(SearchError):
    """Error generating embeddings."""
    pass


class IndexError(SearchError):
    """Error with vector index."""
    pass
