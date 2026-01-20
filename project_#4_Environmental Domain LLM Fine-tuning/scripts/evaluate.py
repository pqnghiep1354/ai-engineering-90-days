#!/usr/bin/env python3
"""
Evaluation script for Environmental LLM.

Usage:
    python scripts/evaluate.py \
        --model_path models/phi2-climate-lora \
        --base_model microsoft/phi-2 \
        --test_data data/processed/climate_qa_test.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger
from tqdm import tqdm

from src.inference import EnvironmentalLLM
from src.config import format_instruction


def setup_logging(verbose: bool = False):
    """Setup logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def load_test_data(filepath: str) -> List[Dict[str, str]]:
    """Load test data."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    from collections import Counter
    
    # Exact match
    exact_matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    
    # Token overlap (simple F1)
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
            continue
        
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        f1_scores.append(f1)
    
    # Length ratio
    length_ratios = [
        len(p) / max(len(r), 1)
        for p, r in zip(predictions, references)
    ]
    
    return {
        "exact_match": exact_matches / len(predictions),
        "token_f1": sum(f1_scores) / len(f1_scores),
        "avg_length_ratio": sum(length_ratios) / len(length_ratios),
        "total_examples": len(predictions),
    }


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Environmental LLM")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model name (required for LoRA)",
    )
    
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/eval_results.json",
        help="Output file for results",
    )
    
    parser.add_argument(
        "--template",
        type=str,
        default="alpaca",
        help="Instruction template",
    )
    
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum examples to evaluate",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    print("\n" + "=" * 60)
    print("Environmental LLM Evaluation")
    print("=" * 60 + "\n")
    
    # Load model
    logger.info(f"Loading model: {args.model_path}")
    llm = EnvironmentalLLM(
        model_path=args.model_path,
        base_model=args.base_model,
        template_name=args.template,
    )
    
    # Load test data
    logger.info(f"Loading test data: {args.test_data}")
    test_data = load_test_data(args.test_data)
    
    if args.max_examples:
        test_data = test_data[:args.max_examples]
    
    logger.info(f"Evaluating on {len(test_data)} examples")
    
    # Generate predictions
    predictions = []
    references = []
    
    for example in tqdm(test_data, desc="Generating"):
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        reference = example.get("output", "")
        
        # Generate
        prediction = llm.generate(
            instruction=instruction,
            input_text=input_text,
            max_new_tokens=256,
            temperature=0.1,  # Low for evaluation
            do_sample=False,
        )
        
        predictions.append(prediction)
        references.append(reference)
        
        if args.verbose:
            logger.debug(f"Q: {instruction[:50]}...")
            logger.debug(f"Pred: {prediction[:100]}...")
            logger.debug(f"Ref: {reference[:100]}...")
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    metrics = calculate_metrics(predictions, references)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total Examples: {metrics['total_examples']}")
    print(f"Exact Match: {metrics['exact_match']:.2%}")
    print(f"Token F1: {metrics['token_f1']:.2%}")
    print(f"Avg Length Ratio: {metrics['avg_length_ratio']:.2f}")
    print("=" * 60 + "\n")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "metrics": metrics,
        "model_path": args.model_path,
        "test_data": args.test_data,
        "examples": [
            {
                "instruction": test_data[i].get("instruction", ""),
                "prediction": predictions[i],
                "reference": references[i],
            }
            for i in range(min(10, len(predictions)))  # Save first 10 examples
        ],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
