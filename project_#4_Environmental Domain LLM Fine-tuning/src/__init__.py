"""
Environmental Domain LLM Fine-tuning Package

Fine-tune Large Language Models for environmental, climate, and ESG tasks
using LoRA and QLoRA techniques.
"""

__version__ = "1.0.0"
__author__ = "AI Engineer Portfolio"

from .config import Settings, get_settings
from .model_utils import load_model, load_tokenizer
from .dataset import InstructionDataset
from .trainer import EnvironmentalTrainer
from .inference import EnvironmentalLLM

__all__ = [
    "Settings",
    "get_settings",
    "load_model",
    "load_tokenizer",
    "InstructionDataset",
    "EnvironmentalTrainer",
    "EnvironmentalLLM",
]
