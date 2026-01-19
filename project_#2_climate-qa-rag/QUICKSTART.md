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

## Bước 2: Cấu hình API Key

```bash
# Copy file cấu hình mẫu
cp .env.example .env

# Mở file .env và thêm API key của bạn
# OPENAI_API_KEY=sk-your-api-key-here
```

**Lấy API Key miễn phí:**
- OpenAI: https://platform.openai.com/api-keys (có $5 credits miễn phí)
- Anthropic: https://console.anthropic.com/ (có free tier)

## Bước 3: Index tài liệu mẫu

```bash
python scripts/index_documents.py --data-dir data/sample
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
python scripts/index_documents.py --data-dir data/your_docs --clear
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
- Kiểm tra file `.env` đã có `OPENAI_API_KEY`
- Đảm bảo API key bắt đầu bằng `sk-`

**Lỗi "No documents indexed":**
- Chạy lại: `python scripts/index_documents.py --data-dir data/sample`

**Lỗi import:**
- Đảm bảo đang ở thư mục gốc của dự án
- Đảm bảo đã activate virtual environment
