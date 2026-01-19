# =============================================================================
# Environmental Domain LLM Fine-tuning - Dockerfile
# =============================================================================

FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA
RUN pip3 install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY data/ ./data/

# Create directories
RUN mkdir -p models logs evaluation/results

# Create non-root user
RUN useradd --create-home --shell /bin/bash trainer && \
    chown -R trainer:trainer /app

USER trainer

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python3", "-c", "print('Environmental LLM Fine-tuning Container Ready')"]

# Example training command:
# docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
#   env-llm-finetune python3 scripts/train.py \
#   --model_name microsoft/phi-2 \
#   --dataset data/processed/climate_qa_train.json \
#   --output_dir models/phi2-climate-lora \
#   --config configs/qlora_config.yaml \
#   --use_qlora
