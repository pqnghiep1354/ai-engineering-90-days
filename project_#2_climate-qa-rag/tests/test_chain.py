"""
Tests for RAG chain functionality.

Run with: pytest tests/test_chain.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prompts import (
    get_rag_prompt,
    get_query_expansion_prompt,
    get_standalone_question_prompt,
    SAMPLE_QUESTIONS_EN,
    SAMPLE_QUESTIONS_VI,
)
from src.utils import (
    clean_text,
    truncate_text,
    detect_language,
    format_sources,
)


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtils:
    """Tests for utility functions."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "  Hello   World  \n\n  Test  "
        cleaned = clean_text(text)
        assert "  " not in cleaned
        assert cleaned == "Hello World Test"
    
    def test_truncate_text(self):
        """Test text truncation."""
        text = "This is a long text that needs to be truncated."
        
        # Truncate to shorter length
        truncated = truncate_text(text, max_length=20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")
        
        # Text shorter than max length
        short_text = "Short"
        assert truncate_text(short_text, max_length=20) == "Short"
    
    def test_detect_language_english(self):
        """Test English language detection."""
        english_text = "What are the main causes of climate change?"
        assert detect_language(english_text) == "en"
    
    def test_detect_language_vietnamese(self):
        """Test Vietnamese language detection."""
        vietnamese_text = "Nguyên nhân chính gây ra biến đổi khí hậu là gì?"
        assert detect_language(vietnamese_text) == "vi"
    
    def test_format_sources_empty(self):
        """Test formatting empty sources."""
        result = format_sources([])
        assert result == "No sources found."
    
    def test_format_sources_with_data(self):
        """Test formatting sources with data."""
        sources = [
            {
                "content": "Test content here",
                "metadata": {"source": "test.pdf", "page": 1}
            }
        ]
        result = format_sources(sources)
        assert "Source 1" in result
        assert "test.pdf" in result


# =============================================================================
# Prompt Tests
# =============================================================================

class TestPrompts:
    """Tests for prompt templates."""
    
    def test_get_rag_prompt_english(self):
        """Test English RAG prompt."""
        prompt = get_rag_prompt(language="en")
        assert prompt is not None
        # Check it has the expected variables
        assert "context" in str(prompt.input_variables) or "context" in str(prompt)
    
    def test_get_rag_prompt_vietnamese(self):
        """Test Vietnamese RAG prompt."""
        prompt = get_rag_prompt(language="vi")
        assert prompt is not None
    
    def test_get_rag_prompt_with_history(self):
        """Test RAG prompt with history."""
        prompt = get_rag_prompt(language="en", with_history=True)
        assert prompt is not None
    
    def test_query_expansion_prompt(self):
        """Test query expansion prompt."""
        prompt = get_query_expansion_prompt(language="en")
        assert prompt is not None
    
    def test_standalone_question_prompt(self):
        """Test standalone question prompt."""
        prompt = get_standalone_question_prompt(language="en")
        assert prompt is not None
    
    def test_sample_questions_exist(self):
        """Test that sample questions are defined."""
        assert len(SAMPLE_QUESTIONS_EN) > 0
        assert len(SAMPLE_QUESTIONS_VI) > 0
        assert len(SAMPLE_QUESTIONS_EN) == len(SAMPLE_QUESTIONS_VI)


# =============================================================================
# Document Loader Tests
# =============================================================================

class TestDocumentLoader:
    """Tests for document loading functionality."""
    
    def test_supported_file_check(self):
        """Test file type support checking."""
        from src.utils import is_supported_file
        
        assert is_supported_file(Path("test.pdf")) == True
        assert is_supported_file(Path("test.txt")) == True
        assert is_supported_file(Path("test.md")) == True
        assert is_supported_file(Path("test.docx")) == True
        assert is_supported_file(Path("test.exe")) == False
        assert is_supported_file(Path("test.jpg")) == False
    
    def test_get_file_extension(self):
        """Test file extension extraction."""
        from src.utils import get_file_extension
        
        assert get_file_extension(Path("test.PDF")) == ".pdf"
        assert get_file_extension(Path("test.TXT")) == ".txt"
        assert get_file_extension(Path("file.name.md")) == ".md"


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration."""
    
    def test_settings_load(self):
        """Test that settings can be loaded."""
        from src.config import settings
        
        assert settings is not None
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.retriever_top_k > 0
    
    def test_settings_validation(self):
        """Test settings validation."""
        from src.config import settings
        
        # Chunk overlap should be less than chunk size
        assert settings.chunk_overlap < settings.chunk_size
        
        # Reranker top_k should be <= retriever top_k
        assert settings.reranker_top_k <= settings.retriever_top_k


# =============================================================================
# Mock Tests for Chain (without API keys)
# =============================================================================

class TestChainMocked:
    """Tests for chain with mocked components."""
    
    @patch('src.chain.get_embedding_model')
    @patch('src.chain.load_existing_index')
    @patch('src.chain.get_llm')
    def test_rag_chain_initialization(
        self,
        mock_llm,
        mock_index,
        mock_embeddings
    ):
        """Test RAG chain initialization."""
        # Setup mocks
        mock_embeddings.return_value = MagicMock()
        mock_index.return_value = MagicMock()
        mock_index.return_value.vector_store = MagicMock()
        mock_llm.return_value = MagicMock()
        
        from src.chain import RAGChain
        
        # Should not raise
        chain = RAGChain(
            vector_store=mock_index.return_value.vector_store,
            llm=mock_llm.return_value,
        )
        
        assert chain is not None


# =============================================================================
# Integration Tests (require API key)
# =============================================================================

@pytest.mark.skipif(
    not pytest.importorskip("openai", reason="OpenAI not installed"),
    reason="OpenAI not installed"
)
class TestIntegration:
    """Integration tests requiring API keys."""
    
    @pytest.fixture
    def api_key_available(self):
        """Check if API key is available."""
        import os
        return os.getenv("OPENAI_API_KEY") is not None
    
    def test_embedding_model(self, api_key_available):
        """Test embedding model initialization."""
        if not api_key_available:
            pytest.skip("OpenAI API key not available")
        
        from src.embeddings import get_embedding_model
        
        model = get_embedding_model()
        assert model is not None
    
    def test_llm_model(self, api_key_available):
        """Test LLM model initialization."""
        if not api_key_available:
            pytest.skip("OpenAI API key not available")
        
        from src.llm import get_llm
        
        llm = get_llm()
        assert llm is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
