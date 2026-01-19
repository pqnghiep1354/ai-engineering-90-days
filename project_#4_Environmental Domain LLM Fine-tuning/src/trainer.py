"""
Training utilities for Environmental LLM Fine-tuning.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset
from loguru import logger
from peft import PeftModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer

from .config import TrainingConfig, get_settings
from .dataset import DataCollatorForInstructionTuning


# =============================================================================
# Training Arguments Builder
# =============================================================================

def build_training_arguments(
    config: TrainingConfig,
    run_name: Optional[str] = None,
) -> TrainingArguments:
    """
    Build TrainingArguments from config.
    
    Args:
        config: Training configuration
        run_name: Optional run name for logging
        
    Returns:
        TrainingArguments instance
    """
    settings = get_settings()
    
    # Prepare output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine report_to
    report_to = config.report_to
    if report_to == "wandb" and not settings.has_wandb_key:
        logger.warning("W&B key not found, using tensorboard")
        report_to = "tensorboard"
    
    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        logging_steps=config.logging_steps,
        logging_dir=str(output_dir / "logs"),
        save_strategy=config.extra.get("save_strategy", "steps"),
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        evaluation_strategy=config.evaluation_strategy,
        eval_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.extra.get("metric_for_best_model", "eval_loss"),
        greater_is_better=config.extra.get("greater_is_better", False),
        report_to=report_to,
        seed=config.seed,
        dataloader_num_workers=config.extra.get("dataloader_num_workers", 4),
        remove_unused_columns=False,
    )


# =============================================================================
# Environmental Trainer
# =============================================================================

class EnvironmentalTrainer:
    """
    Trainer wrapper for environmental domain fine-tuning.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_config: Optional[TrainingConfig] = None,
        data_collator: Optional[Any] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_config: Training configuration
            data_collator: Optional data collator
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config or TrainingConfig()
        
        # Create data collator
        if data_collator is None:
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
        else:
            self.data_collator = data_collator
        
        # Build training arguments
        self.training_args = build_training_arguments(self.config)
        
        # Create trainer
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )
        
        logger.info("Trainer initialized")
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Run training.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training metrics
        """
        logger.info("Starting training...")
        
        # Train
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        logger.info(f"Training complete. Loss: {result.training_loss:.4f}")
        
        return {
            "training_loss": result.training_loss,
            "global_step": result.global_step,
            "metrics": result.metrics,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        logger.info("Running evaluation...")
        
        metrics = self.trainer.evaluate()
        
        logger.info(f"Eval loss: {metrics.get('eval_loss', 'N/A')}")
        
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save model and tokenizer."""
        output_dir = output_dir or self.config.output_dir
        
        logger.info(f"Saving model to: {output_dir}")
        
        # Save model
        self.trainer.save_model(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Model saved successfully")


# =============================================================================
# SFT Trainer Wrapper
# =============================================================================

class SFTEnvironmentalTrainer:
    """
    SFT Trainer wrapper using TRL library.
    Better for instruction tuning.
    """
    
    def __init__(
        self,
        model: Union[PreTrainedModel, PeftModel],
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_config: Optional[TrainingConfig] = None,
        max_seq_length: int = 2048,
        dataset_text_field: str = "text",
        packing: bool = False,
    ):
        """
        Initialize SFT trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_config: Training configuration
            max_seq_length: Maximum sequence length
            dataset_text_field: Field containing text
            packing: Whether to use packing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config or TrainingConfig()
        self.max_seq_length = max_seq_length
        
        # Build training arguments
        self.training_args = build_training_arguments(self.config)
        
        # Create SFT trainer
        self.trainer = SFTTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            dataset_text_field=dataset_text_field,
            packing=packing,
        )
        
        logger.info("SFT Trainer initialized")
    
    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run training."""
        logger.info("Starting SFT training...")
        
        result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        logger.info(f"Training complete. Loss: {result.training_loss:.4f}")
        
        return {
            "training_loss": result.training_loss,
            "global_step": result.global_step,
            "metrics": result.metrics,
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        return self.trainer.evaluate()
    
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save model and tokenizer."""
        output_dir = output_dir or self.config.output_dir
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)


# =============================================================================
# Training Callbacks
# =============================================================================

from transformers import TrainerCallback


class LoggingCallback(TrainerCallback):
    """Custom logging callback."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                logger.info(f"Step {state.global_step}: loss={logs['loss']:.4f}")
            if "eval_loss" in logs:
                logger.info(f"Eval loss: {logs['eval_loss']:.4f}")


class EarlyStoppingCallback(TrainerCallback):
    """Early stopping callback."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            eval_loss = metrics.get("eval_loss", float("inf"))
            
            if eval_loss < self.best_loss - self.min_delta:
                self.best_loss = eval_loss
                self.counter = 0
            else:
                self.counter += 1
                
                if self.counter >= self.patience:
                    logger.info(f"Early stopping triggered at step {state.global_step}")
                    control.should_training_stop = True


# =============================================================================
# Training Utilities
# =============================================================================

def get_trainable_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """Get trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        "trainable": trainable,
        "total": total,
        "percent": 100 * trainable / total,
    }


def estimate_training_time(
    num_examples: int,
    batch_size: int,
    gradient_accumulation: int,
    num_epochs: int,
    time_per_step: float = 1.0,  # seconds
) -> float:
    """Estimate training time in hours."""
    steps_per_epoch = num_examples // (batch_size * gradient_accumulation)
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps * time_per_step
    return total_seconds / 3600


def estimate_memory_usage(
    model_params: int,
    batch_size: int,
    seq_length: int,
    precision: str = "fp16",
) -> float:
    """Estimate GPU memory usage in GB."""
    bytes_per_param = 2 if precision == "fp16" else 4
    
    # Model weights
    model_memory = model_params * bytes_per_param
    
    # Activations (rough estimate)
    activation_memory = batch_size * seq_length * 4096 * bytes_per_param * 3
    
    # Optimizer states (AdamW has 2 states per param)
    optimizer_memory = model_params * bytes_per_param * 2
    
    total_bytes = model_memory + activation_memory + optimizer_memory
    return total_bytes / (1024 ** 3)
