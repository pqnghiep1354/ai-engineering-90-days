# System Architecture

## Overview

This project implements a complete pipeline for fine-tuning Large Language Models (LLMs) on environmental domain data using Parameter-Efficient Fine-Tuning (PEFT) methods.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FINE-TUNING PIPELINE                              │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA PREPARATION                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Raw Data    │───▶│  Processor   │───▶│  Formatted   │                   │
│  │  (Q&A, Chat) │    │  (Clean,     │    │  Dataset     │                   │
│  │              │    │   Convert)   │    │  (Alpaca)    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL LOADING                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Base Model  │───▶│ Quantization │───▶│  LoRA/QLoRA  │                   │
│  │  (Phi-2,     │    │  (4-bit for  │    │  Adaptation  │                   │
│  │   Llama)     │    │   QLoRA)     │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRAINING                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  HF Trainer  │───▶│  Gradient    │───▶│  Checkpoint  │                   │
│  │  (SFTTrainer)│    │  Updates     │    │  Saving      │                   │
│  │              │    │  (FP16/BF16) │    │              │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Merge       │───▶│  Export      │───▶│  Deploy      │                   │
│  │  LoRA        │    │  (GGUF,      │    │  (API,       │                   │
│  │  Weights     │    │   ONNX)      │    │   Gradio)    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Preparation (`src/data_processor.py`)

**Responsibilities:**
- Load data from various formats (JSON, JSONL, CSV)
- Convert Q&A and Chat formats to instruction format
- Clean and validate examples
- Augment data with variations
- Split into train/val/test sets

**Input Formats Supported:**
```python
# Q&A Format
{"question": "...", "answer": "..."}

# Chat Format
{"messages": [{"user": "...", "assistant": "..."}]}

# Instruction Format (target)
{"instruction": "...", "input": "...", "output": "..."}
```

### 2. Dataset Handling (`src/dataset.py`)

**Key Classes:**
- `InstructionDataset`: PyTorch Dataset for instruction tuning
- `DataCollatorForInstructionTuning`: Custom data collator

**Features:**
- Instruction templates (Alpaca, ChatML, Llama2, Phi)
- Dynamic padding and truncation
- Label masking for causal LM

### 3. Model Utilities (`src/model_utils.py`)

**Functions:**
- `load_model()`: Load base model with optional quantization
- `load_tokenizer()`: Load tokenizer with proper pad token
- `apply_lora()`: Apply LoRA adapters to model
- `merge_lora_weights()`: Merge LoRA into base model

**Quantization Support:**
```python
# 4-bit quantization for QLoRA
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

### 4. Training (`src/trainer.py`)

**Trainer Classes:**
- `EnvironmentalTrainer`: Standard HF Trainer wrapper
- `SFTEnvironmentalTrainer`: TRL SFTTrainer wrapper

**Features:**
- Mixed precision training (FP16/BF16)
- Gradient checkpointing
- Weights & Biases integration
- Early stopping callback

### 5. Inference (`src/inference.py`)

**Key Classes:**
- `EnvironmentalLLM`: High-level inference API
- `BatchInference`: Efficient batch processing

**Features:**
- Automatic LoRA merging
- Streaming generation
- Chat interface with history

## LoRA Architecture

### Standard LoRA
```
┌─────────────────────────────────────────┐
│           Original Layer                 │
│    W (d_in × d_out) - Frozen            │
└─────────────────────────────────────────┘
              │
              │  h = Wx
              ▼
       ┌──────┴──────┐
       │             │
       ▼             ▼
┌────────────┐ ┌────────────┐
│   W × x    │ │  B × A × x │
│  (frozen)  │ │ (trainable)│
│            │ │  r << d    │
└────────────┘ └────────────┘
       │             │
       └──────┬──────┘
              │  h' = Wx + BAx
              ▼
       ┌────────────┐
       │   Output   │
       └────────────┘
```

### QLoRA (4-bit)
```
┌─────────────────────────────────────────┐
│      Quantized Base (4-bit NF4)         │
│    W_q = quantize(W) - Frozen           │
└─────────────────────────────────────────┘
              │
              │  dequantize for compute
              ▼
┌─────────────────────────────────────────┐
│           LoRA Adapters (FP16)          │
│    A (d_in × r), B (r × d_out)          │
└─────────────────────────────────────────┘
```

## Memory Optimization

| Technique | Memory Savings | Speed Impact |
|-----------|---------------|--------------|
| 4-bit Quantization | ~75% | -10% |
| Gradient Checkpointing | ~50% | -20% |
| Mixed Precision (FP16) | ~50% | +10% |
| LoRA (r=16) | ~99% params frozen | +50% |

## Configuration System

### Environment Variables (`.env`)
```bash
HF_TOKEN           # HuggingFace access token
WANDB_API_KEY      # Weights & Biases
CUDA_VISIBLE_DEVICES  # GPU selection
```

### YAML Configs (`configs/`)
```yaml
# lora_config.yaml
model:
  name: microsoft/phi-2
  torch_dtype: float16
lora:
  r: 16
  lora_alpha: 32
  target_modules: [q_proj, v_proj]
training:
  learning_rate: 2e-4
  batch_size: 4
```

## Error Handling

1. **GPU Memory**: Auto-fallback to smaller batch sizes
2. **API Limits**: Retry with exponential backoff
3. **Data Errors**: Skip invalid examples with logging
4. **Checkpoint**: Resume from last saved state

## Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| Exact Match | Exact string match | EM = (exact_matches) / total |
| Token F1 | Token-level overlap | F1 = 2PR / (P+R) |
| BLEU | N-gram overlap | BLEU = BP × exp(∑log(pn)) |
| ROUGE-L | Longest common subsequence | - |

## Deployment Options

1. **Local Inference**: Load merged model directly
2. **API Server**: FastAPI + Uvicorn
3. **Gradio Demo**: Interactive web interface
4. **GGUF Export**: For llama.cpp deployment
5. **ONNX Export**: For optimized inference
