# 🚀 Hướng dẫn Bắt đầu Nhanh

## Bước 1: Cài đặt

```bash
# Clone hoặc giải nén dự án
cd climate-qa-rag

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Bước 2: Cấu hình

```bash
# Copy file cấu hình mẫu
cp .env.example .env
```

**Cấu hình API Key (chọn 1 trong các options):**

### Option A: Ollama Local (Khuyến nghị - Miễn phí, không giới hạn)
```bash
# Cài Ollama: https://ollama.ai
ollama pull gemma3:4b

# Trong .env:
LLM_MODEL=gemma3:4b
```

### Option B: Google Gemini (Free tier)
```bash
# Lấy API key: https://aistudio.google.com/
# Trong .env:
GOOGLE_API_KEY=your-google-api-key
LLM_MODEL=gemini-2.0-flash
```

### Option C: OpenAI
```bash
# Lấy API key: https://platform.openai.com/api-keys
# Trong .env:
OPENAI_API_KEY=sk-your-api-key
LLM_MODEL=gpt-4o-mini
```

## Bước 3: Index tài liệu mẫu

```bash
# Với Gemini embeddings (khuyến nghị)
python scripts/index_documents.py --data-dir data/sample --embedding-provider gemini

# Hoặc với OpenAI embeddings
python scripts/index_documents.py --data-dir data/sample --embedding-provider openai
```

## Bước 4: Chạy ứng dụng

### Option 1: Web App (Streamlit)
```bash
streamlit run src/app.py
# Mở browser: http://localhost:8501
```

### Option 2: API Server (FastAPI)
```bash
uvicorn src.api:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Option 3: Command Line
```bash
python src/cli.py "What causes climate change?"
# Hoặc chế độ interactive:
python src/cli.py --interactive
```

## Bước 5: Thêm tài liệu của bạn

```bash
# Thêm file PDF, TXT, MD vào thư mục data/
python scripts/index_documents.py --data-dir data/your_docs --clear --embedding-provider gemini
```

## 🎉 Xong!

Bây giờ bạn có thể hỏi các câu hỏi về khí hậu và môi trường!

---

## Câu hỏi mẫu để thử

**Tiếng Anh:**
- What are the main causes of climate change?
- How does the Paris Agreement work?
- What is carbon footprint?

**Tiếng Việt:**
- Biến đổi khí hậu ảnh hưởng đến Việt Nam như thế nào?
- ESG là gì và tại sao quan trọng?
- Làm thế nào để giảm khí thải carbon?

---

## Troubleshooting

**Lỗi "API key not configured":**
- Kiểm tra file `.env` đã có API key phù hợp
- Hoặc dùng Ollama local để không cần API key

**Lỗi "Rate limit exceeded":**
- Chờ 1 phút và thử lại
- Hoặc chuyển sang Ollama local (không giới hạn)

**Lỗi "No documents indexed":**
- Chạy lại: `python scripts/index_documents.py --data-dir data/sample --embedding-provider gemini`

**Lỗi "Ollama connection refused":**
- Đảm bảo Ollama đang chạy: `ollama serve`

**Lỗi import:**
- Đảm bảo đang ở thư mục gốc của dự án
- Đảm bảo đã activate virtual environment
