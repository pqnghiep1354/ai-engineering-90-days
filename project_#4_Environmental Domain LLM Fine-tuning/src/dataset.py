"""
Dataset classes for Environmental LLM Fine-tuning.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from loguru import logger
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from .config import format_instruction, get_instruction_template


# =============================================================================
# Instruction Dataset
# =============================================================================

class InstructionDataset(TorchDataset):
    """
    Dataset for instruction-tuning format.
    
    Expected data format:
    {
        "instruction": "What causes climate change?",
        "input": "",  # Optional context
        "output": "Climate change is caused by..."
    }
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        template_name: str = "alpaca",
    ):
        """
        Initialize instruction dataset.
        
        Args:
            data: List of instruction dictionaries
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            template_name: Instruction template to use
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_name = template_name
        
        logger.info(f"Created dataset with {len(data)} examples")
        logger.info(f"Max length: {max_length}, Template: {template_name}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format instruction
        full_text = format_instruction(
            instruction=item.get("instruction", ""),
            input_text=item.get("input", ""),
            output=item.get("output", ""),
            template_name=self.template_name,
        )
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Prepare labels (same as input_ids for causal LM)
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    @classmethod
    def from_json(
        cls,
        json_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        template_name: str = "alpaca",
    ) -> "InstructionDataset":
        """Load dataset from JSON file."""
        logger.info(f"Loading dataset from: {json_path}")
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(data, tokenizer, max_length, template_name)
    
    @classmethod
    def from_jsonl(
        cls,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        template_name: str = "alpaca",
    ) -> "InstructionDataset":
        """Load dataset from JSONL file."""
        logger.info(f"Loading dataset from: {jsonl_path}")
        
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        return cls(data, tokenizer, max_length, template_name)


# =============================================================================
# HuggingFace Dataset Wrapper
# =============================================================================

def create_hf_dataset(
    data: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    template_name: str = "alpaca",
    train_split: float = 0.9,
) -> Dict[str, Dataset]:
    """
    Create HuggingFace Dataset for training with Trainer.
    
    Args:
        data: List of instruction dictionaries
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        template_name: Instruction template
        train_split: Train/validation split ratio
        
    Returns:
        Dictionary with 'train' and 'eval' datasets
    """
    
    def tokenize_function(examples):
        """Tokenize examples."""
        texts = []
        
        for i in range(len(examples["instruction"])):
            text = format_instruction(
                instruction=examples["instruction"][i],
                input_text=examples.get("input", [""] * len(examples["instruction"]))[i],
                output=examples["output"][i],
                template_name=template_name,
            )
            texts.append(text)
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        
        # Set labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Split
    split_dataset = dataset.train_test_split(
        test_size=1 - train_split,
        seed=42,
    )
    
    # Tokenize
    tokenized_train = split_dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    tokenized_eval = split_dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    logger.info(f"Train samples: {len(tokenized_train)}")
    logger.info(f"Eval samples: {len(tokenized_eval)}")
    
    return {
        "train": tokenized_train,
        "eval": tokenized_eval,
    }


def load_hf_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    template_name: str = "alpaca",
    train_split: float = 0.9,
) -> Dict[str, Dataset]:
    """
    Load dataset from file and prepare for training.
    
    Args:
        dataset_path: Path to JSON or JSONL file
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        template_name: Instruction template
        train_split: Train/validation split
        
    Returns:
        Dictionary with train and eval datasets
    """
    path = Path(dataset_path)
    
    logger.info(f"Loading dataset: {dataset_path}")
    
    # Load data
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(f"Loaded {len(data)} examples")
    
    return create_hf_dataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        template_name=template_name,
        train_split=train_split,
    )


# =============================================================================
# Data Collator
# =============================================================================

class DataCollatorForInstructionTuning:
    """
    Data collator for instruction tuning.
    Handles padding and label masking.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        pad_to_multiple_of: int = 8,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
    
    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        
        # Get max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Pad to multiple
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1) 
                // self.pad_to_multiple_of 
                * self.pad_to_multiple_of
            )
        
        max_len = min(max_len, self.max_length)
        
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }
        
        for feature in features:
            input_ids = feature["input_ids"][:max_len]
            attention_mask = feature["attention_mask"][:max_len]
            labels = feature["labels"][:max_len]
            
            # Pad
            padding_length = max_len - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                labels = labels + [-100] * padding_length
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Convert to tensors
        return {
            k: torch.tensor(v, dtype=torch.long)
            for k, v in batch.items()
        }


# =============================================================================
# Dataset Statistics
# =============================================================================

def get_dataset_stats(data: List[Dict[str, str]]) -> Dict[str, Any]:
    """Get statistics for a dataset."""
    
    instructions = [d.get("instruction", "") for d in data]
    inputs = [d.get("input", "") for d in data]
    outputs = [d.get("output", "") for d in data]
    
    instruction_lengths = [len(i.split()) for i in instructions]
    input_lengths = [len(i.split()) for i in inputs if i]
    output_lengths = [len(o.split()) for o in outputs]
    
    stats = {
        "total_examples": len(data),
        "with_input": sum(1 for i in inputs if i),
        "avg_instruction_words": sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
        "avg_output_words": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
        "max_instruction_words": max(instruction_lengths) if instruction_lengths else 0,
        "max_output_words": max(output_lengths) if output_lengths else 0,
    }
    
    if input_lengths:
        stats["avg_input_words"] = sum(input_lengths) / len(input_lengths)
        stats["max_input_words"] = max(input_lengths)
    
    return stats


def print_dataset_stats(data: List[Dict[str, str]]) -> None:
    """Print dataset statistics."""
    stats = get_dataset_stats(data)
    
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"Total examples: {stats['total_examples']}")
    print(f"Examples with input context: {stats['with_input']}")
    print(f"Avg instruction length: {stats['avg_instruction_words']:.1f} words")
    print(f"Avg output length: {stats['avg_output_words']:.1f} words")
    print(f"Max instruction length: {stats['max_instruction_words']} words")
    print(f"Max output length: {stats['max_output_words']} words")
    print("=" * 50 + "\n")
