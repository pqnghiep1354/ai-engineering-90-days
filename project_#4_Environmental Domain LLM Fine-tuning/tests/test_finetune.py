"""
Tests for Environmental LLM Fine-tuning components.

Run with: pytest tests/test_finetune.py -v
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    Settings,
    LoRAConfig,
    TrainingConfig,
    format_instruction,
    get_instruction_template,
)
from src.data_processor import (
    clean_text,
    is_valid_example,
    convert_qa_to_instruction,
    augment_instruction,
)
from src.dataset import get_dataset_stats


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration."""
    
    def test_settings_defaults(self):
        """Test default settings."""
        settings = Settings()
        assert settings.default_model == "microsoft/phi-2"
        assert settings.log_level == "INFO"
    
    def test_lora_config(self):
        """Test LoRA config."""
        config = LoRAConfig(r=16, lora_alpha=32)
        assert config.r == 16
        assert config.lora_alpha == 32
        
        data = config.to_dict()
        assert "r" in data
        assert "target_modules" in data
    
    def test_training_config(self):
        """Test training config."""
        config = TrainingConfig(num_train_epochs=3, learning_rate=2e-4)
        assert config.num_train_epochs == 3
        assert config.learning_rate == 2e-4
    
    def test_instruction_template(self):
        """Test instruction template."""
        template = get_instruction_template("alpaca")
        assert "prompt_input" in template
        assert "prompt_no_input" in template
    
    def test_format_instruction(self):
        """Test instruction formatting."""
        formatted = format_instruction(
            instruction="What is climate change?",
            input_text="",
            output="Climate change is...",
            template_name="alpaca",
        )
        
        assert "What is climate change?" in formatted
        assert "Climate change is..." in formatted
        assert "### Instruction:" in formatted


# =============================================================================
# Data Processor Tests
# =============================================================================

class TestDataProcessor:
    """Tests for data processing."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "  Hello   world  \n\t  "
        cleaned = clean_text(text)
        assert cleaned == "Hello world"
    
    def test_is_valid_example(self):
        """Test example validation."""
        valid = {
            "instruction": "What is climate change?",
            "output": "Climate change refers to long-term shifts..."
        }
        assert is_valid_example(valid)
        
        invalid = {
            "instruction": "Hi",
            "output": "Hello"
        }
        assert not is_valid_example(invalid, min_length=10)
    
    def test_convert_qa_to_instruction(self):
        """Test Q&A conversion."""
        qa_data = [
            {"question": "What is ESG?", "answer": "ESG stands for..."}
        ]
        
        result = convert_qa_to_instruction(
            qa_data,
            question_key="question",
            answer_key="answer",
        )
        
        assert len(result) == 1
        assert result[0]["instruction"] == "What is ESG?"
    
    def test_augment_instruction(self):
        """Test instruction augmentation."""
        instruction = "What is climate change?"
        variations = augment_instruction(instruction)
        
        assert len(variations) > 1
        assert instruction in variations


# =============================================================================
# Dataset Tests
# =============================================================================

class TestDataset:
    """Tests for dataset handling."""
    
    def test_get_dataset_stats(self):
        """Test dataset statistics."""
        data = [
            {
                "instruction": "What is climate change?",
                "input": "",
                "output": "Climate change refers to long-term shifts in temperatures."
            },
            {
                "instruction": "Explain ESG.",
                "input": "Focus on environmental aspects.",
                "output": "ESG stands for Environmental, Social, Governance."
            },
        ]
        
        stats = get_dataset_stats(data)
        
        assert stats["total_examples"] == 2
        assert stats["with_input"] == 1
        assert stats["avg_instruction_words"] > 0
        assert stats["avg_output_words"] > 0


# =============================================================================
# Model Utils Tests (Mock)
# =============================================================================

class TestModelUtils:
    """Tests for model utilities (with mocks)."""
    
    def test_get_compute_dtype(self):
        """Test dtype conversion."""
        from src.model_utils import get_compute_dtype
        import torch
        
        assert get_compute_dtype("float16") == torch.float16
        assert get_compute_dtype("bfloat16") == torch.bfloat16
        assert get_compute_dtype("float32") == torch.float32


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_data_pipeline(self):
        """Test full data preparation pipeline."""
        # Create sample data
        sample_data = [
            {
                "instruction": "What causes global warming?",
                "input": "",
                "output": "Global warming is caused by greenhouse gas emissions."
            }
        ]
        
        # Validate
        for item in sample_data:
            assert is_valid_example(item)
        
        # Get stats
        stats = get_dataset_stats(sample_data)
        assert stats["total_examples"] == 1
    
    def test_config_loading(self):
        """Test config can be imported."""
        from src.config import get_settings
        
        settings = get_settings()
        assert settings is not None


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
