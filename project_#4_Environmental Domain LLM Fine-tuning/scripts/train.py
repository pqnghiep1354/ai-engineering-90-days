#!/usr/bin/env python3
"""
Training script for Environmental Domain LLM Fine-tuning.

Usage:
    python scripts/train.py \
        --model_name microsoft/phi-2 \
        --dataset data/processed/climate_qa_train.json \
        --output_dir models/phi2-climate-lora \
        --config configs/lora_config.yaml
        
    # QLoRA for limited GPU
    python scripts/train.py \
        --model_name microsoft/phi-2 \
        --dataset data/processed/climate_qa_train.json \
        --output_dir models/phi2-climate-qlora \
        --config configs/qlora_config.yaml \
        --use_qlora
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from src.config import (
    LoRAConfig,
    QuantizationConfig,
    TrainingConfig,
    load_training_config,
    get_settings,
)
from src.dataset import load_hf_dataset, print_dataset_stats
from src.model_utils import (
    load_model,
    load_tokenizer,
    apply_lora,
    save_lora_model,
    print_model_info,
)
from src.trainer import (
    EnvironmentalTrainer,
    build_training_arguments,
    get_trainable_parameters,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM for environmental domain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/phi-2",
        help="Base model name or path",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset (JSON/JSONL)",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for model",
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/lora_config.yaml",
        help="Path to config file",
    )
    
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantization)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (override config)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (override config)",
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (override config)",
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    
    parser.add_argument(
        "--template",
        type=str,
        default="alpaca",
        choices=["alpaca", "chatml", "llama2", "phi"],
        help="Instruction template",
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    # Print banner
    print("\n" + "=" * 60)
    print("🧠 Environmental Domain LLM Fine-tuning")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available, training will be slow")
    
    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_training_config(args.config)
    
    # Override config with CLI arguments
    if args.epochs:
        config["training"].num_train_epochs = args.epochs
    if args.batch_size:
        config["training"].per_device_train_batch_size = args.batch_size
    if args.learning_rate:
        config["training"].learning_rate = args.learning_rate
    
    config["training"].output_dir = args.output_dir
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = load_tokenizer(args.model_name)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    datasets = load_hf_dataset(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        template_name=args.template,
        train_split=config["dataset"].get("train_split", 0.9),
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    
    quantization_config = None
    if args.use_qlora and "quantization" in config:
        logger.info("Using QLoRA quantization")
        quantization_config = config["quantization"]
    
    model = load_model(
        model_name=args.model_name,
        torch_dtype=config["model"].get("torch_dtype", "float16"),
        device_map=config["model"].get("device_map", "auto"),
        quantization_config=quantization_config,
        use_gradient_checkpointing=config["training"].gradient_checkpointing,
    )
    
    # Apply LoRA
    model = apply_lora(model, config["lora"])
    print_model_info(model)
    
    # Create trainer
    trainer = EnvironmentalTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        training_config=config["training"],
    )
    
    # Train
    logger.info("Starting training...")
    print("\n" + "=" * 60)
    
    result = trainer.train(resume_from_checkpoint=args.resume)
    
    print("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"Final loss: {result['training_loss']:.4f}")
    
    # Save model
    logger.info(f"Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # Final evaluation
    if datasets["eval"]:
        eval_metrics = trainer.evaluate()
        logger.info(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60 + "\n")
    
    # Usage instructions
    print("Next steps:")
    print(f"  1. Merge LoRA: python scripts/merge_lora.py --base_model {args.model_name} --lora_path {args.output_dir}")
    print(f"  2. Evaluate: python scripts/evaluate.py --model_path {args.output_dir}")
    print(f"  3. Test: python -c \"from src.inference import EnvironmentalLLM; llm = EnvironmentalLLM('{args.output_dir}', '{args.model_name}'); print(llm.generate('What is climate change?'))\"")


if __name__ == "__main__":
    main()
