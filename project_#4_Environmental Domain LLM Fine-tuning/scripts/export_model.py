#!/usr/bin/env python3
"""
Export fine-tuned model to various formats.

Usage:
    # Export to GGUF (for llama.cpp)
    python scripts/export_model.py \
        --model_path models/phi2-climate-merged \
        --output_format gguf \
        --quantization q4_k_m

    # Export to ONNX
    python scripts/export_model.py \
        --model_path models/phi2-climate-merged \
        --output_format onnx
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def setup_logging(verbose: bool = False):
    """Setup logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "q4_k_m",
) -> None:
    """
    Export model to GGUF format for llama.cpp.
    
    Args:
        model_path: Path to HuggingFace model
        output_path: Output directory
        quantization: Quantization type (q4_0, q4_k_m, q5_k_m, q8_0, f16)
    """
    logger.info(f"Exporting to GGUF with {quantization} quantization")
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if llama.cpp conversion script exists
    try:
        import llama_cpp
        logger.info("llama-cpp-python is installed")
    except ImportError:
        logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    
    # The actual conversion would require llama.cpp tools
    # This is a placeholder showing the workflow
    
    conversion_script = """
# To convert to GGUF, you need llama.cpp:
# 1. Clone: git clone https://github.com/ggerganov/llama.cpp
# 2. Build: cd llama.cpp && make
# 3. Convert: python convert.py {model_path} --outtype f16 --outfile {output_path}/model-f16.gguf
# 4. Quantize: ./quantize {output_path}/model-f16.gguf {output_path}/model-{quant}.gguf {quant}
""".format(model_path=model_path, output_path=output_path, quant=quantization)
    
    # Save conversion instructions
    instructions_path = output_dir / "GGUF_CONVERSION.md"
    with open(instructions_path, "w") as f:
        f.write(f"# GGUF Conversion Instructions\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Quantization: {quantization}\n\n")
        f.write("## Steps\n\n")
        f.write("```bash\n")
        f.write("# 1. Clone llama.cpp\n")
        f.write("git clone https://github.com/ggerganov/llama.cpp\n")
        f.write("cd llama.cpp\n\n")
        f.write("# 2. Build\n")
        f.write("make\n\n")
        f.write("# 3. Convert to GGUF (F16 first)\n")
        f.write(f"python convert-hf-to-gguf.py {model_path} --outfile {output_path}/model-f16.gguf\n\n")
        f.write("# 4. Quantize\n")
        f.write(f"./llama-quantize {output_path}/model-f16.gguf {output_path}/model-{quantization}.gguf {quantization}\n")
        f.write("```\n\n")
        f.write("## Quantization Options\n\n")
        f.write("| Type | Size | Quality |\n")
        f.write("|------|------|--------|\n")
        f.write("| q4_0 | Smallest | Lower |\n")
        f.write("| q4_k_m | Small | Good |\n")
        f.write("| q5_k_m | Medium | Better |\n")
        f.write("| q8_0 | Large | High |\n")
        f.write("| f16 | Largest | Best |\n")
    
    logger.info(f"Conversion instructions saved to: {instructions_path}")


def export_to_onnx(
    model_path: str,
    output_path: str,
) -> None:
    """
    Export model to ONNX format.
    
    Args:
        model_path: Path to HuggingFace model
        output_path: Output directory
    """
    logger.info("Exporting to ONNX format")
    
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and convert
        logger.info("Loading model for ONNX conversion...")
        model = ORTModelForCausalLM.from_pretrained(
            model_path,
            export=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"ONNX model saved to: {output_dir}")
        
    except ImportError:
        logger.warning("optimum not installed. Install with: pip install optimum[onnxruntime]")
        
        # Save instructions
        instructions_path = Path(output_path) / "ONNX_CONVERSION.md"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        with open(instructions_path, "w") as f:
            f.write("# ONNX Conversion Instructions\n\n")
            f.write("```bash\n")
            f.write("# Install optimum\n")
            f.write("pip install optimum[onnxruntime]\n\n")
            f.write("# Convert\n")
            f.write("python -c \"\n")
            f.write("from optimum.onnxruntime import ORTModelForCausalLM\n")
            f.write(f"model = ORTModelForCausalLM.from_pretrained('{model_path}', export=True)\n")
            f.write(f"model.save_pretrained('{output_path}')\n")
            f.write("\"\n")
            f.write("```\n")
        
        logger.info(f"Instructions saved to: {instructions_path}")


def export_to_safetensors(
    model_path: str,
    output_path: str,
) -> None:
    """
    Convert model to SafeTensors format.
    
    Args:
        model_path: Path to model
        output_path: Output directory
    """
    logger.info("Converting to SafeTensors format")
    
    try:
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Save as safetensors
        state_dict = model.state_dict()
        save_file(state_dict, output_dir / "model.safetensors")
        
        # Copy config
        model.config.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info(f"SafeTensors model saved to: {output_dir}")
        
    except ImportError:
        logger.error("safetensors not installed. Install with: pip install safetensors")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export fine-tuned model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output directory (default: model_path-{format})",
    )
    
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["gguf", "onnx", "safetensors"],
        default="gguf",
        help="Output format",
    )
    
    parser.add_argument(
        "--quantization",
        type=str,
        default="q4_k_m",
        help="Quantization type for GGUF (q4_0, q4_k_m, q5_k_m, q8_0, f16)",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    
    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()
    setup_logging(args.verbose)
    
    print("\n" + "=" * 60)
    print("📦 Model Export Tool")
    print("=" * 60 + "\n")
    
    # Set default output path
    output_path = args.output_path
    if output_path is None:
        output_path = f"{args.model_path}-{args.output_format}"
    
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Format: {args.output_format}")
    logger.info(f"Output: {output_path}")
    
    # Export based on format
    if args.output_format == "gguf":
        export_to_gguf(args.model_path, output_path, args.quantization)
    elif args.output_format == "onnx":
        export_to_onnx(args.model_path, output_path)
    elif args.output_format == "safetensors":
        export_to_safetensors(args.model_path, output_path)
    
    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print(f"Output: {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
