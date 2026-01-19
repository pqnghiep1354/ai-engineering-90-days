"""
Model loading utilities for Environmental LLM Fine-tuning.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from loguru import logger
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import (
    LoRAConfig,
    QuantizationConfig,
    get_settings,
)


# =============================================================================
# Tokenizer Loading
# =============================================================================

def load_tokenizer(
    model_name: str,
    padding_side: str = "right",
    add_eos_token: bool = True,
    add_bos_token: bool = False,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """
    Load tokenizer for a model.
    
    Args:
        model_name: HuggingFace model name or path
        padding_side: Padding side for batching
        add_eos_token: Add EOS token to sequences
        add_bos_token: Add BOS token to sequences
        trust_remote_code: Trust remote code for custom models
        
    Returns:
        Loaded tokenizer
    """
    settings = get_settings()
    
    logger.info(f"Loading tokenizer: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
        add_eos_token=add_eos_token,
        add_bos_token=add_bos_token,
        token=settings.hf_token if settings.has_hf_token else None,
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Tokenizer loaded. Vocab size: {len(tokenizer)}")
    
    return tokenizer


# =============================================================================
# Model Loading
# =============================================================================

def get_compute_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, torch.float16)


def load_model(
    model_name: str,
    torch_dtype: str = "float16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    quantization_config: Optional[QuantizationConfig] = None,
    use_gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """
    Load base model for fine-tuning.
    
    Args:
        model_name: HuggingFace model name or path
        torch_dtype: Model precision
        device_map: Device mapping strategy
        trust_remote_code: Trust remote code
        quantization_config: QLoRA quantization config
        use_gradient_checkpointing: Enable gradient checkpointing
        
    Returns:
        Loaded model
    """
    settings = get_settings()
    compute_dtype = get_compute_dtype(torch_dtype)
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Device map: {device_map}, Dtype: {torch_dtype}")
    
    # Prepare load kwargs
    load_kwargs = {
        "pretrained_model_name_or_path": model_name,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "token": settings.hf_token if settings.has_hf_token else None,
    }
    
    # Add quantization config for QLoRA
    if quantization_config is not None:
        logger.info("Using QLoRA quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.load_in_4bit,
            bnb_4bit_quant_type=quantization_config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=get_compute_dtype(
                quantization_config.bnb_4bit_compute_dtype
            ),
            bnb_4bit_use_double_quant=quantization_config.bnb_4bit_use_double_quant,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = compute_dtype
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    
    # Prepare for k-bit training if quantized
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
    elif use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Print model info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Total params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model


def apply_lora(
    model: PreTrainedModel,
    lora_config: LoRAConfig,
) -> PeftModel:
    """
    Apply LoRA to model.
    
    Args:
        model: Base model
        lora_config: LoRA configuration
        
    Returns:
        Model with LoRA applied
    """
    logger.info("Applying LoRA configuration")
    logger.info(f"LoRA rank: {lora_config.r}, alpha: {lora_config.lora_alpha}")
    logger.info(f"Target modules: {lora_config.target_modules}")
    
    # Create PEFT config
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
        target_modules=lora_config.target_modules,
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model


def load_model_for_training(
    model_name: str,
    lora_config: LoRAConfig,
    quantization_config: Optional[QuantizationConfig] = None,
    torch_dtype: str = "float16",
    device_map: str = "auto",
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer ready for training.
    
    Args:
        model_name: Model name or path
        lora_config: LoRA configuration
        quantization_config: Optional QLoRA quantization config
        torch_dtype: Model precision
        device_map: Device mapping
        
    Returns:
        Tuple of (model with LoRA, tokenizer)
    """
    # Load tokenizer
    tokenizer = load_tokenizer(model_name)
    
    # Load base model
    model = load_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
    )
    
    # Apply LoRA
    model = apply_lora(model, lora_config)
    
    return model, tokenizer


# =============================================================================
# Model Saving and Loading
# =============================================================================

def save_lora_model(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
):
    """
    Save LoRA adapter weights.
    
    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        output_dir: Output directory
    """
    logger.info(f"Saving LoRA adapter to: {output_dir}")
    
    # Save adapter
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model saved successfully")


def load_lora_model(
    base_model_name: str,
    lora_path: str,
    device_map: str = "auto",
    torch_dtype: str = "float16",
) -> Tuple[PeftModel, PreTrainedTokenizer]:
    """
    Load a saved LoRA model.
    
    Args:
        base_model_name: Base model name
        lora_path: Path to LoRA adapter
        device_map: Device mapping
        torch_dtype: Model precision
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading LoRA model from: {lora_path}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(lora_path)
    
    # Load base model
    model = load_model(
        model_name=base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        use_gradient_checkpointing=False,
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, lora_path)
    
    logger.info("LoRA model loaded successfully")
    
    return model, tokenizer


def merge_lora_weights(
    base_model_name: str,
    lora_path: str,
    output_dir: str,
    torch_dtype: str = "float16",
) -> None:
    """
    Merge LoRA weights into base model.
    
    Args:
        base_model_name: Base model name
        lora_path: Path to LoRA adapter
        output_dir: Output directory for merged model
        torch_dtype: Model precision
    """
    logger.info("Merging LoRA weights into base model")
    
    # Load base model and LoRA
    model, tokenizer = load_lora_model(
        base_model_name=base_model_name,
        lora_path=lora_path,
        device_map="cpu",  # Use CPU for merging
        torch_dtype=torch_dtype,
    )
    
    # Merge and unload
    model = model.merge_and_unload()
    
    # Save merged model
    logger.info(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model merged and saved successfully")


# =============================================================================
# Model Information
# =============================================================================

def get_model_info(model: PreTrainedModel) -> Dict[str, Any]:
    """Get model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_percent": 100 * trainable_params / total_params,
        "model_size_mb": total_params * 2 / (1024 * 1024),  # Assuming fp16
    }


def print_model_info(model: PreTrainedModel) -> None:
    """Print model information."""
    info = get_model_info(model)
    
    print("\n" + "=" * 50)
    print("Model Information")
    print("=" * 50)
    print(f"Total Parameters: {info['total_parameters']:,}")
    print(f"Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"Trainable Percent: {info['trainable_percent']:.2f}%")
    print(f"Model Size (FP16): {info['model_size_mb']:.2f} MB")
    print("=" * 50 + "\n")
