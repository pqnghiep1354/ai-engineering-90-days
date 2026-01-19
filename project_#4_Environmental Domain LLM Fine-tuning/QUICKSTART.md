# 🚀 Hướng Dẫn Nhanh - Fine-tune LLM Môi Trường

## Bước 1: Cài đặt môi trường

```bash
# Tạo conda environment
conda create -n env-llm python=3.10
conda activate env-llm

# Cài đặt PyTorch với CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài đặt dependencies
pip install -r requirements.txt
```

## Bước 2: Cấu hình

```bash
cp .env.example .env
```

Thêm vào `.env`:
```
HF_TOKEN=hf_your_huggingface_token  # Nếu dùng Llama
WANDB_API_KEY=your_wandb_key        # Optional
```

## Bước 3: Chuẩn bị dữ liệu

```bash
# Tạo dataset mẫu
python scripts/prepare_data.py \
    --generate_sample \
    --output data/processed/climate_qa.json \
    --split

# Hoặc từ file có sẵn
python scripts/prepare_data.py \
    --input data/raw/your_data.json \
    --output data/processed/climate_qa.json \
    --format qa \
    --split
```

## Bước 4: Fine-tune

### Option A: LoRA (GPU 16GB+)
```bash
python scripts/train.py \
    --model_name microsoft/phi-2 \
    --dataset data/processed/climate_qa_train.json \
    --output_dir models/phi2-climate-lora \
    --config configs/lora_config.yaml \
    --epochs 3
```

### Option B: QLoRA (GPU 8GB)
```bash
python scripts/train.py \
    --model_name microsoft/phi-2 \
    --dataset data/processed/climate_qa_train.json \
    --output_dir models/phi2-climate-qlora \
    --config configs/qlora_config.yaml \
    --use_qlora \
    --epochs 3
```

## Bước 5: Đánh giá

```bash
python scripts/evaluate.py \
    --model_path models/phi2-climate-lora \
    --base_model microsoft/phi-2 \
    --test_data data/processed/climate_qa_test.json
```

## Bước 6: Sử dụng

```python
from src.inference import EnvironmentalLLM

# Load model
llm = EnvironmentalLLM(
    model_path="models/phi2-climate-lora",
    base_model="microsoft/phi-2"
)

# Generate
response = llm.generate("What is climate change?")
print(response)
```

## 📊 So sánh LoRA vs QLoRA

| Aspect | LoRA | QLoRA |
|--------|------|-------|
| **VRAM cần thiết** | 16GB+ | 8GB |
| **Tốc độ training** | Nhanh hơn | Chậm hơn ~20% |
| **Chất lượng** | Tốt nhất | Gần bằng LoRA |
| **Models phù hợp** | 7B-13B | 7B-13B (quantized) |

## 🔧 Các tham số quan trọng

### LoRA Config
- `r`: Rank (8-64, cao hơn = nhiều params hơn)
- `lora_alpha`: Scaling factor (thường = 2*r)
- `target_modules`: Layers để fine-tune

### Training Config
- `batch_size`: Điều chỉnh theo VRAM
- `learning_rate`: 1e-4 đến 3e-4
- `epochs`: 2-5 cho datasets nhỏ

## ❓ Troubleshooting

### "CUDA out of memory"
- Giảm `batch_size`
- Dùng QLoRA thay LoRA
- Bật `gradient_checkpointing`

### "Model not loading"
- Kiểm tra HF_TOKEN cho gated models
- Kiểm tra internet connection

### Training không converge
- Giảm learning_rate
- Tăng epochs
- Kiểm tra data quality

---

**🎯 Portfolio Project #4** - Environmental Domain LLM Fine-tuning
