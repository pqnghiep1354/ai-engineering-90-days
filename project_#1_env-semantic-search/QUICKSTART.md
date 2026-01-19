# 🚀 Hướng Dẫn Nhanh - Environmental Semantic Search Tool

## Bước 1: Cài đặt Dependencies

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc: venv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

## Bước 2: Cấu hình API Key

```bash
# Copy file cấu hình mẫu
cp .env.example .env

# Mở file .env và thêm OpenAI API key
# OPENAI_API_KEY=sk-your-api-key-here
```

**💡 Lấy API Key miễn phí:**
- [OpenAI](https://platform.openai.com/api-keys) - Tài khoản mới được $5 credits
- [Anthropic](https://console.anthropic.com/) - Free tier có sẵn

## Bước 3: Index tài liệu mẫu

```bash
python scripts/index_documents.py --data-dir data/documents
```

**Output mong đợi:**
```
📂 Indexing documents from: data/documents
📄 Loading and processing documents...
📊 Document Statistics:
   Total chunks: ~50
   Unique sources: 5
📥 Indexing document chunks...
✅ Indexing complete!
```

## Bước 4: Chạy ứng dụng

### Option A: Web Interface (Streamlit)
```bash
streamlit run src/app.py
```
Mở browser tại: http://localhost:8501

### Option B: REST API (FastAPI)
```bash
uvicorn src.api:app --reload --port 8000
```
API docs tại: http://localhost:8000/docs

### Option C: Command Line
```bash
# Tìm kiếm đơn giản
python src/cli.py "What causes climate change?"

# Chế độ tương tác
python src/cli.py --interactive
```

## Bước 5: Thử nghiệm

### Câu hỏi mẫu (Tiếng Anh)
- "What causes global warming?"
- "How does solar energy work?"
- "What is ESG reporting?"
- "Air pollution health effects"
- "Renewable energy benefits"

### Câu hỏi mẫu (Tiếng Việt)
- "Biến đổi khí hậu ảnh hưởng đến Việt Nam như thế nào?"
- "Nguyên nhân nước biển dâng"
- "Năng lượng tái tạo là gì?"

## 📁 Thêm tài liệu của bạn

1. Copy tài liệu vào thư mục `data/documents/`
   - Hỗ trợ: PDF, TXT, MD, DOCX, HTML

2. Chạy lại indexing:
```bash
python scripts/index_documents.py --data-dir data/documents --clear
```

## 🐳 Chạy với Docker

```bash
# Build và chạy
docker-compose up -d

# Web app: http://localhost:8501
# API: http://localhost:8000
```

## ❓ Xử lý lỗi thường gặp

### "OpenAI API key not configured"
→ Kiểm tra file `.env` đã có `OPENAI_API_KEY`

### "No documents indexed"
→ Chạy `python scripts/index_documents.py --data-dir data/documents`

### "Module not found"
→ Kiểm tra đã activate virtual environment

### Import errors
→ Chạy `pip install -r requirements.txt`

## 📊 Đánh giá chất lượng tìm kiếm

```bash
python scripts/evaluate_search.py
```

## 📚 Tài liệu thêm

- [README.md](README.md) - Tài liệu đầy đủ
- [API Reference](docs/api_reference.md) - Chi tiết API
- [Architecture](docs/architecture.md) - Kiến trúc hệ thống

---

**🎯 Portfolio Project #1** - Environmental Semantic Search Tool

Dự án này demonstrate:
- ✅ Semantic search với AI embeddings
- ✅ Vector database (ChromaDB)
- ✅ Document processing pipeline
- ✅ Multiple interfaces (Web, API, CLI)
- ✅ Bilingual support (EN/VI)
