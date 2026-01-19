"""
Configuration management for Environmental LLM Fine-tuning.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger


# =============================================================================
# Environment Settings
# =============================================================================

class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # HuggingFace
    hf_token: str = Field(default="", description="HuggingFace API token")
    hf_home: str = Field(default="~/.cache/huggingface")
    
    # Weights & Biases
    wandb_api_key: str = Field(default="")
    wandb_project: str = Field(default="env-llm-finetune")
    wandb_disabled: bool = Field(default=False)
    
    # Model
    default_model: str = Field(default="microsoft/phi-2")
    model_cache_dir: str = Field(default="./models/cache")
    
    # Training
    output_dir: str = Field(default="./models/output")
    data_dir: str = Field(default="./data/processed")
    use_fp16: bool = Field(default=True)
    gradient_checkpointing: bool = Field(default=True)
    
    # Hardware
    cuda_visible_devices: str = Field(default="0")
    num_gpus: int = Field(default=1)
    
    # Logging
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="./logs")
    
    @property
    def has_hf_token(self) -> bool:
        return bool(self.hf_token and len(self.hf_token) > 10)
    
    @property
    def has_wandb_key(self) -> bool:
        return bool(self.wandb_api_key and len(self.wandb_api_key) > 10)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# =============================================================================
# Training Configuration
# =============================================================================

class LoRAConfig:
    """LoRA configuration."""
    
    def __init__(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        target_modules: Optional[List[str]] = None,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj"
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "task_type": self.task_type,
            "target_modules": self.target_modules,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoRAConfig":
        return cls(**data)


class QuantizationConfig:
    """Quantization configuration for QLoRA."""
    
    def __init__(
        self,
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_use_double_quant: bool = True,
    ):
        self.load_in_4bit = load_in_4bit
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }


class TrainingConfig:
    """Training configuration."""
    
    def __init__(
        self,
        output_dir: str = "./models/output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine",
        max_grad_norm: float = 1.0,
        fp16: bool = True,
        bf16: bool = False,
        gradient_checkpointing: bool = True,
        optim: str = "paged_adamw_32bit",
        logging_steps: int = 10,
        save_steps: int = 100,
        save_total_limit: int = 3,
        evaluation_strategy: str = "steps",
        eval_steps: int = 100,
        load_best_model_at_end: bool = True,
        report_to: str = "wandb",
        seed: int = 42,
        **kwargs,
    ):
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.max_grad_norm = max_grad_norm
        self.fp16 = fp16
        self.bf16 = bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.optim = optim
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.load_best_model_at_end = load_best_model_at_end
        self.report_to = report_to
        self.seed = seed
        self.extra = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "load_best_model_at_end": self.load_best_model_at_end,
            "report_to": self.report_to,
            "seed": self.seed,
        }


# =============================================================================
# Config Loading
# =============================================================================

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded config from: {config_path}")
    return config


def load_training_config(config_path: str) -> Dict[str, Any]:
    """Load complete training configuration."""
    raw_config = load_yaml_config(config_path)
    
    # Parse sub-configs
    config = {
        "model": raw_config.get("model", {}),
        "lora": LoRAConfig.from_dict(raw_config.get("lora", {})),
        "training": TrainingConfig(**raw_config.get("training", {})),
        "dataset": raw_config.get("dataset", {}),
        "generation": raw_config.get("generation", {}),
    }
    
    # Add quantization if present (QLoRA)
    if "quantization" in raw_config:
        config["quantization"] = QuantizationConfig(**raw_config["quantization"])
    
    return config


# =============================================================================
# Instruction Templates
# =============================================================================

INSTRUCTION_TEMPLATES = {
    "alpaca": {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Input:\n{input}\n\n"
            "### Response:\n"
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:\n"
        ),
    },
    "chatml": {
        "prompt_input": (
            "<|im_start|>system\nYou are a helpful environmental expert.<|im_end|>\n"
            "<|im_start|>user\n{instruction}\n\nContext: {input}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "prompt_no_input": (
            "<|im_start|>system\nYou are a helpful environmental expert.<|im_end|>\n"
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
    },
    "llama2": {
        "prompt_input": (
            "[INST] <<SYS>>\nYou are a helpful environmental expert.\n<</SYS>>\n\n"
            "{instruction}\n\nContext: {input} [/INST] "
        ),
        "prompt_no_input": (
            "[INST] <<SYS>>\nYou are a helpful environmental expert.\n<</SYS>>\n\n"
            "{instruction} [/INST] "
        ),
    },
    "phi": {
        "prompt_input": (
            "Instruct: {instruction}\nInput: {input}\nOutput: "
        ),
        "prompt_no_input": (
            "Instruct: {instruction}\nOutput: "
        ),
    },
}


def get_instruction_template(template_name: str = "alpaca") -> Dict[str, str]:
    """Get instruction template by name."""
    if template_name not in INSTRUCTION_TEMPLATES:
        logger.warning(f"Unknown template: {template_name}, using alpaca")
        template_name = "alpaca"
    return INSTRUCTION_TEMPLATES[template_name]


def format_instruction(
    instruction: str,
    input_text: str = "",
    output: str = "",
    template_name: str = "alpaca",
) -> str:
    """Format a single instruction example."""
    template = get_instruction_template(template_name)
    
    if input_text:
        prompt = template["prompt_input"].format(
            instruction=instruction,
            input=input_text,
        )
    else:
        prompt = template["prompt_no_input"].format(
            instruction=instruction,
        )
    
    if output:
        return prompt + output
    return prompt
