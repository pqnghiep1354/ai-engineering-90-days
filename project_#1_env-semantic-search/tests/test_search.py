"""
Tests for search functionality.

Run with: pytest tests/test_search.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    clean_text,
    truncate_text,
    detect_language,
    is_supported_file,
    get_file_extension,
)
from src.document_processor import (
    Document,
    DocumentProcessor,
    deduplicate_documents,
    get_document_stats,
)


# =============================================================================
# Utility Tests
# =============================================================================

class TestUtils:
    """Tests for utility functions."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "  Hello   World  \n\n  Test  "
        cleaned = clean_text(text)
        assert cleaned == "Hello World Test"
    
    def test_clean_text_preserves_vietnamese(self):
        """Test that Vietnamese characters are preserved."""
        text = "Biến đổi khí hậu ảnh hưởng đến Việt Nam"
        cleaned = clean_text(text)
        assert "đổi" in cleaned
        assert "ảnh" in cleaned
    
    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a long text that needs truncation."
        truncated = truncate_text(text, max_length=20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")
    
    def test_truncate_text_short(self):
        """Test truncation of short text."""
        text = "Short"
        assert truncate_text(text, max_length=20) == "Short"
    
    def test_detect_language_english(self):
        """Test English detection."""
        text = "What causes climate change?"
        assert detect_language(text) == "en"
    
    def test_detect_language_vietnamese(self):
        """Test Vietnamese detection."""
        text = "Biến đổi khí hậu là gì?"
        assert detect_language(text) == "vi"
    
    def test_is_supported_file(self):
        """Test file type support checking."""
        assert is_supported_file(Path("test.pdf")) == True
        assert is_supported_file(Path("test.txt")) == True
        assert is_supported_file(Path("test.md")) == True
        assert is_supported_file(Path("test.docx")) == True
        assert is_supported_file(Path("test.exe")) == False
        assert is_supported_file(Path("test.jpg")) == False
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        assert get_file_extension(Path("test.PDF")) == ".pdf"
        assert get_file_extension(Path("test.TXT")) == ".txt"


# =============================================================================
# Document Tests
# =============================================================================

class TestDocument:
    """Tests for Document class."""
    
    def test_document_creation(self):
        """Test document creation."""
        doc = Document(
            content="Test content",
            metadata={"source": "test.txt"}
        )
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test.txt"
    
    def test_document_id(self):
        """Test document ID generation."""
        doc = Document(content="Test content")
        assert len(doc.id) == 16
    
    def test_document_length(self):
        """Test document length."""
        doc = Document(content="Test content")
        assert len(doc) == len("Test content")


class TestDocumentProcessor:
    """Tests for DocumentProcessor."""
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
    
    def test_chunk_text(self):
        """Test text chunking."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        text = "A" * 250  # 250 characters
        
        chunks = processor.chunk_text(text, metadata={"source": "test"})
        
        assert len(chunks) > 1
        assert all(isinstance(c, Document) for c in chunks)
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        processor = DocumentProcessor()
        chunks = processor.chunk_text("")
        assert chunks == []


class TestDeduplication:
    """Tests for document deduplication."""
    
    def test_deduplicate_documents(self):
        """Test duplicate removal."""
        docs = [
            Document(content="Same content"),
            Document(content="Same content"),
            Document(content="Different content"),
        ]
        
        unique = deduplicate_documents(docs)
        assert len(unique) == 2
    
    def test_get_document_stats(self):
        """Test document statistics."""
        docs = [
            Document(content="Short"),
            Document(content="Longer content here"),
        ]
        
        stats = get_document_stats(docs)
        
        assert stats["count"] == 2
        assert stats["total_characters"] > 0
        assert "avg_length" in stats


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration."""
    
    def test_settings_load(self):
        """Test settings loading."""
        from src.config import settings
        
        assert settings is not None
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
    
    def test_settings_validation(self):
        """Test settings validation."""
        from src.config import settings
        
        assert settings.chunk_overlap < settings.chunk_size


# =============================================================================
# Search Engine Tests (Mocked)
# =============================================================================

class TestSearchEngineMocked:
    """Tests for search engine with mocks."""
    
    @patch('src.search_engine.load_index')
    def test_search_engine_init(self, mock_load_index):
        """Test search engine initialization."""
        mock_store = MagicMock()
        mock_load_index.return_value = mock_store
        
        from src.search_engine import SemanticSearchEngine
        
        engine = SemanticSearchEngine(vector_store=mock_store)
        assert engine is not None
    
    @patch('src.search_engine.load_index')
    def test_search_empty_query(self, mock_load_index):
        """Test search with empty query."""
        mock_store = MagicMock()
        mock_load_index.return_value = mock_store
        
        from src.search_engine import SemanticSearchEngine
        
        engine = SemanticSearchEngine(vector_store=mock_store)
        response = engine.search("")
        
        assert response.total_results == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
