"""
Data processing utilities for creating training datasets.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# =============================================================================
# Data Cleaning
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters but keep punctuation
    text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)
    # Strip
    text = text.strip()
    return text


def is_valid_example(example: Dict[str, str], min_length: int = 10) -> bool:
    """Check if example is valid."""
    instruction = example.get("instruction", "")
    output = example.get("output", "")
    
    # Check minimum length
    if len(instruction) < min_length or len(output) < min_length:
        return False
    
    # Check not empty
    if not instruction.strip() or not output.strip():
        return False
    
    return True


# =============================================================================
# Format Converters
# =============================================================================

def convert_qa_to_instruction(
    qa_data: List[Dict[str, str]],
    question_key: str = "question",
    answer_key: str = "answer",
    context_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Convert Q&A format to instruction format.
    
    Args:
        qa_data: List of Q&A pairs
        question_key: Key for question field
        answer_key: Key for answer field
        context_key: Optional key for context field
        
    Returns:
        List in instruction format
    """
    instruction_data = []
    
    for item in qa_data:
        instruction = item.get(question_key, "")
        output = item.get(answer_key, "")
        input_text = item.get(context_key, "") if context_key else ""
        
        if instruction and output:
            instruction_data.append({
                "instruction": clean_text(instruction),
                "input": clean_text(input_text),
                "output": clean_text(output),
            })
    
    return instruction_data


def convert_chat_to_instruction(
    chat_data: List[Dict[str, Any]],
    user_key: str = "user",
    assistant_key: str = "assistant",
) -> List[Dict[str, str]]:
    """
    Convert chat format to instruction format.
    
    Args:
        chat_data: List of chat conversations
        user_key: Key for user message
        assistant_key: Key for assistant message
        
    Returns:
        List in instruction format
    """
    instruction_data = []
    
    for conv in chat_data:
        messages = conv.get("messages", conv)
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                user_msg = msg.get(user_key, "")
                assistant_msg = msg.get(assistant_key, "")
            else:
                continue
            
            if user_msg and assistant_msg:
                instruction_data.append({
                    "instruction": clean_text(user_msg),
                    "input": "",
                    "output": clean_text(assistant_msg),
                })
    
    return instruction_data


# =============================================================================
# Environmental Domain Data
# =============================================================================

CLIMATE_QA_TEMPLATES = [
    {
        "instruction": "What is {topic}?",
        "output_prefix": "{topic} refers to",
    },
    {
        "instruction": "Explain {topic} in simple terms.",
        "output_prefix": "{topic} is",
    },
    {
        "instruction": "What are the main causes of {topic}?",
        "output_prefix": "The main causes of {topic} include",
    },
    {
        "instruction": "What are the effects of {topic}?",
        "output_prefix": "{topic} leads to several effects:",
    },
    {
        "instruction": "How can we address {topic}?",
        "output_prefix": "To address {topic}, we can",
    },
]

ENVIRONMENTAL_TOPICS = [
    "climate change",
    "global warming",
    "greenhouse gases",
    "carbon emissions",
    "renewable energy",
    "deforestation",
    "biodiversity loss",
    "ocean acidification",
    "sea level rise",
    "extreme weather events",
    "carbon footprint",
    "sustainable development",
    "circular economy",
    "ESG investing",
    "carbon neutrality",
]


def generate_environmental_qa(
    topics: Optional[List[str]] = None,
    templates: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Generate environmental Q&A pairs using templates.
    
    Args:
        topics: List of environmental topics
        templates: Q&A templates
        
    Returns:
        List of Q&A pairs
    """
    topics = topics or ENVIRONMENTAL_TOPICS
    templates = templates or CLIMATE_QA_TEMPLATES
    
    qa_pairs = []
    
    for topic in topics:
        for template in templates:
            qa_pairs.append({
                "instruction": template["instruction"].format(topic=topic),
                "input": "",
                "output": template["output_prefix"].format(topic=topic) + " [PLACEHOLDER - Add actual content]",
            })
    
    return qa_pairs


# =============================================================================
# Data Augmentation
# =============================================================================

def augment_instruction(instruction: str) -> List[str]:
    """
    Generate variations of an instruction.
    
    Args:
        instruction: Original instruction
        
    Returns:
        List of instruction variations
    """
    variations = [instruction]
    
    # Question to command
    if instruction.endswith("?"):
        variations.append(instruction[:-1].replace("What is", "Explain").replace("How", "Describe how"))
    
    # Add politeness
    if not instruction.startswith("Please"):
        variations.append(f"Please {instruction[0].lower()}{instruction[1:]}")
    
    # Add context request
    variations.append(f"{instruction} Provide specific examples.")
    
    return variations


def augment_dataset(
    data: List[Dict[str, str]],
    augmentation_factor: int = 2,
) -> List[Dict[str, str]]:
    """
    Augment dataset with instruction variations.
    
    Args:
        data: Original dataset
        augmentation_factor: How many times to augment
        
    Returns:
        Augmented dataset
    """
    augmented = []
    
    for item in data:
        # Original
        augmented.append(item)
        
        # Variations
        variations = augment_instruction(item["instruction"])
        for var in variations[:augmentation_factor]:
            if var != item["instruction"]:
                augmented.append({
                    "instruction": var,
                    "input": item.get("input", ""),
                    "output": item["output"],
                })
    
    return augmented


# =============================================================================
# Data Splitting
# =============================================================================

def split_dataset(
    data: List[Dict[str, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Split dataset into train/val/test.
    
    Args:
        data: Dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train, val, test)
    """
    import random
    random.seed(seed)
    
    # Shuffle
    data = data.copy()
    random.shuffle(data)
    
    # Split
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]
    
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    
    return train, val, test


# =============================================================================
# File Operations
# =============================================================================

def load_data(filepath: str) -> List[Dict[str, str]]:
    """Load data from JSON or JSONL file."""
    path = Path(filepath)
    
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif path.suffix == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def save_data(data: List[Dict[str, str]], filepath: str) -> None:
    """Save data to JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(data)} examples to: {filepath}")


def save_jsonl(data: List[Dict[str, str]], filepath: str) -> None:
    """Save data to JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    logger.info(f"Saved {len(data)} examples to: {filepath}")


# =============================================================================
# Data Pipeline
# =============================================================================

def prepare_training_data(
    input_path: str,
    output_path: str,
    format_type: str = "instruction",
    augment: bool = False,
    split: bool = True,
    min_length: int = 10,
) -> None:
    """
    Full data preparation pipeline.
    
    Args:
        input_path: Input data path
        output_path: Output data path
        format_type: Input format type (qa, chat, instruction)
        augment: Whether to augment data
        split: Whether to split data
        min_length: Minimum example length
    """
    logger.info(f"Loading data from: {input_path}")
    raw_data = load_data(input_path)
    
    # Convert format
    if format_type == "qa":
        data = convert_qa_to_instruction(raw_data)
    elif format_type == "chat":
        data = convert_chat_to_instruction(raw_data)
    else:
        data = raw_data
    
    logger.info(f"Converted {len(data)} examples")
    
    # Filter
    data = [d for d in data if is_valid_example(d, min_length)]
    logger.info(f"After filtering: {len(data)} examples")
    
    # Augment
    if augment:
        data = augment_dataset(data)
        logger.info(f"After augmentation: {len(data)} examples")
    
    # Split and save
    if split:
        train, val, test = split_dataset(data)
        
        base_path = Path(output_path)
        save_data(train, str(base_path.parent / f"{base_path.stem}_train.json"))
        save_data(val, str(base_path.parent / f"{base_path.stem}_val.json"))
        save_data(test, str(base_path.parent / f"{base_path.stem}_test.json"))
    else:
        save_data(data, output_path)
