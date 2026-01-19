#!/usr/bin/env python3
"""
Merge LoRA weights into base model.

Usage:
    python scripts/merge_lora.py \
        --base_model microsoft/phi-2 \
        --lora_path models/phi2-climate-lora \
        --output models/phi2-climate-merged
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.model_utils import merge_lora_weights, print_model_info


def setup_logging(verbose: bool = False):
    """Setup logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path",
    )
    
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA adapter",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged model",
    )
    
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model precision",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    print("\n" + "=" * 60)
    print("🔀 LoRA Weight Merging")
    print("=" * 60 + "\n")
    
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"LoRA path: {args.lora_path}")
    logger.info(f"Output: {args.output}")
    
    # Merge
    merge_lora_weights(
        base_model_name=args.base_model,
        lora_path=args.lora_path,
        output_dir=args.output,
        torch_dtype=args.torch_dtype,
    )
    
    print("\n" + "=" * 60)
    print("✅ Merge complete!")
    print(f"Merged model saved to: {args.output}")
    print("=" * 60 + "\n")
    
    print("Next steps:")
    print(f"  1. Test: python -c \"from src.inference import EnvironmentalLLM; llm = EnvironmentalLLM('{args.output}'); print(llm.generate('What is climate change?'))\"")
    print(f"  2. Export: python scripts/export_model.py --model_path {args.output} --output_format gguf")


if __name__ == "__main__":
    main()
