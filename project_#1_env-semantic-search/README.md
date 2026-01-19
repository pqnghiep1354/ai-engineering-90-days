# 🔍 Environmental Semantic Search Tool

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Công cụ tìm kiếm ngữ nghĩa thông minh cho tài liệu môi trường sử dụng AI Embeddings

## 🎯 Giới thiệu

Environmental Semantic Search Tool là ứng dụng AI giúp tìm kiếm thông tin trong các tài liệu môi trường một cách thông minh. Thay vì tìm kiếm từ khóa truyền thống, công cụ này hiểu **ý nghĩa** của câu hỏi và tìm các đoạn văn bản liên quan nhất.

### Tại sao cần Semantic Search?

| Tìm kiếm truyền thống | Semantic Search |
|----------------------|-----------------|
| Khớp từ khóa chính xác | Hiểu ý nghĩa câu hỏi |
| Bỏ lỡ từ đồng nghĩa | Tìm nội dung tương tự về nghĩa |
| Cần biết từ khóa chính xác | Hỏi bằng ngôn ngữ tự nhiên |
| "carbon emission" ≠ "CO2 release" | "carbon emission" ≈ "CO2 release" |

### Use Cases

- 🔬 **Nghiên cứu**: Tìm kiếm nhanh trong báo cáo IPCC, EPA
- 📊 **Phân tích ESG**: Tra cứu tiêu chuẩn và metrics
- 📋 **Compliance**: Tìm quy định môi trường liên quan
- 📚 **Học tập**: Khám phá tài liệu khoa học khí hậu

## ✨ Tính năng

### Core Features
- ✅ **Semantic Search**: Tìm kiếm theo ý nghĩa, không chỉ từ khóa
- ✅ **Multi-format Support**: PDF, TXT, MD, DOCX, HTML
- ✅ **Bilingual**: Hỗ trợ tiếng Anh và tiếng Việt
- ✅ **Relevance Scoring**: Xếp hạng kết quả theo độ liên quan
- ✅ **Source Citation**: Trích dẫn nguồn rõ ràng

### User Interface
- ✅ **Web App**: Giao diện Streamlit thân thiện
- ✅ **REST API**: FastAPI cho tích hợp hệ thống
- ✅ **CLI**: Command line cho automation

### Technical Features
- ✅ **Vector Database**: ChromaDB (local) / Pinecone (cloud)
- ✅ **Embeddings**: OpenAI text-embedding-3-small
- ✅ **Chunking**: Smart document splitting
- ✅ **Caching**: Embedding cache để tiết kiệm API calls

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interfaces                             │
│         ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│         │ Web App  │  │ REST API │  │   CLI    │               │
│         │(Streamlit)│  │(FastAPI) │  │          │               │
│         └────┬─────┘  └────┬─────┘  └────┬─────┘               │
└──────────────┼─────────────┼─────────────┼──────────────────────┘
               │             │             │
               ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Search Engine                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  Query Processing                        │   │
│  │   [User Query] → [Embedding] → [Vector Search]          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Vector Database                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ChromaDB / Pinecone                                     │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │   │
│  │  │ Doc 1   │ │ Doc 2   │ │ Doc 3   │ │ Doc N   │       │   │
│  │  │[vector] │ │[vector] │ │[vector] │ │[vector] │       │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- 2GB RAM minimum
- 1GB disk space

### Bước 1: Clone/Download

```bash
git clone https://github.com/yourusername/env-semantic-search.git
cd env-semantic-search
```

### Bước 2: Tạo Virtual Environment

```bash
# Với venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Hoặc với conda
conda create -n env-search python=3.11
conda activate env-search
```

### Bước 3: Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình Environment

```bash
cp .env.example .env
# Mở .env và thêm OpenAI API key
```

### Bước 5: Index tài liệu mẫu

```bash
python scripts/index_documents.py --data-dir data/documents
```

### Bước 6: Chạy ứng dụng

```bash
# Web App
streamlit run src/app.py

# Hoặc API Server
uvicorn src.api:app --reload --port 8000

# Hoặc CLI
python src/cli.py "climate change impacts"
```

## 📖 Hướng dẫn sử dụng

### Web Interface

1. Mở browser tại `http://localhost:8501`
2. Nhập câu hỏi tìm kiếm vào ô search
3. Chọn số lượng kết quả muốn hiển thị
4. Xem kết quả với relevance score và source

### CLI Mode

```bash
# Tìm kiếm đơn giản
python src/cli.py "What is carbon footprint?"

# Tìm kiếm với nhiều kết quả
python src/cli.py "renewable energy benefits" --top-k 10

# Output JSON
python src/cli.py "ESG reporting" --format json

# Interactive mode
python src/cli.py --interactive
```

### API Mode

```bash
# Start server
uvicorn src.api:app --reload

# Search request
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "greenhouse gas emissions", "top_k": 5}'
```

## 📁 Cấu trúc dự án

```
env-semantic-search/
├── src/
│   ├── __init__.py
│   ├── app.py              # Streamlit web application
│   ├── api.py              # FastAPI REST endpoints
│   ├── cli.py              # Command line interface
│   ├── config.py           # Configuration management
│   ├── embeddings.py       # Embedding model wrapper
│   ├── document_processor.py # Document loading & chunking
│   ├── vector_store.py     # Vector database operations
│   ├── search_engine.py    # Core search logic
│   └── utils.py            # Utility functions
├── data/
│   ├── documents/          # Documents to index
│   └── sample_queries/     # Example queries for testing
├── scripts/
│   ├── index_documents.py  # Document indexing script
│   ├── evaluate_search.py  # Search quality evaluation
│   └── download_samples.py # Download sample documents
├── tests/
│   ├── test_embeddings.py
│   ├── test_search.py
│   └── test_api.py
├── docs/
│   ├── architecture.md
│   └── api_reference.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔌 API Reference

### POST /search

Tìm kiếm semantic trong documents.

**Request:**
```json
{
  "query": "What causes global warming?",
  "top_k": 5,
  "threshold": 0.5,
  "filter": {
    "source_type": "pdf"
  }
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Global warming is primarily caused by...",
      "score": 0.92,
      "metadata": {
        "source": "ipcc_ar6_summary.pdf",
        "page": 15,
        "chunk_id": 42
      }
    }
  ],
  "query": "What causes global warming?",
  "total_results": 5,
  "search_time_ms": 125
}
```

### POST /index

Index new documents.

### GET /stats

Get index statistics.

### GET /health

Health check endpoint.

## 📊 Sample Documents

Dự án bao gồm các tài liệu mẫu từ:

| Source | Description | Topics |
|--------|-------------|--------|
| **IPCC** | Climate science reports | Climate change, impacts |
| **EPA** | Environmental regulations | Air quality, emissions |
| **UN SDGs** | Sustainable development | Goals, indicators |
| **ESG Guides** | Corporate sustainability | Reporting, metrics |

## 🐳 Docker Deployment

```bash
# Build image
docker build -t env-semantic-search .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key env-semantic-search

# Or use docker-compose
docker-compose up -d
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Search latency | < 200ms |
| Indexing speed | ~100 docs/min |
| Embedding dimensions | 1536 |
| Supported file size | Up to 50MB |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) - Embedding models
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [LangChain](https://langchain.com/) - Document processing
- [Streamlit](https://streamlit.io/) - Web framework

## 📧 Contact

- **Author**: [Your Name]
- **Email**: your.email@example.com
- **Portfolio**: [Your Portfolio URL]

---

⭐ Nếu dự án hữu ích, hãy cho một star!

**Đây là Portfolio Project #1 trong lộ trình AI Engineer**
