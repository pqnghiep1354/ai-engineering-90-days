# 🌍 Climate Science Q&A System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Hệ thống hỏi đáp thông minh về khoa học khí hậu và môi trường sử dụng RAG (Retrieval-Augmented Generation)

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Contributing](#-contributing)

## 🎯 Giới thiệu

Climate Science Q&A System là một ứng dụng AI sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) để trả lời các câu hỏi về:

- 🌡️ **Biến đổi khí hậu**: Nguyên nhân, tác động, dự báo
- 🏭 **Khí thải carbon**: Tính toán, giảm thiểu, offset
- 🌿 **Môi trường**: Ô nhiễm, bảo tồn, phát triển bền vững
- 📊 **Báo cáo ESG**: Tiêu chuẩn, metrics, compliance
- 📜 **Quy định**: Luật môi trường Việt Nam và quốc tế

### Tại sao cần dự án này?

1. **Thông tin chính xác**: Trích xuất từ các nguồn uy tín (IPCC, EPA, UN)
2. **Cập nhật**: Dễ dàng thêm tài liệu mới
3. **Truy xuất nguồn**: Mỗi câu trả lời đều có citation
4. **Tiếng Việt**: Hỗ trợ câu hỏi và trả lời bằng tiếng Việt

## ✨ Tính năng

### Core Features
- ✅ RAG với semantic search
- ✅ Multi-document support (PDF, TXT, MD, DOCX)
- ✅ Conversation memory
- ✅ Source citation
- ✅ Hybrid search (vector + keyword)

### Advanced Features
- ✅ Reranking với Cross-Encoder
- ✅ Query expansion
- ✅ Streaming responses
- ✅ Multi-language support (EN/VI)
- ✅ Export chat history

### Monitoring & Observability
- ✅ LangSmith integration
- ✅ Token usage tracking
- ✅ Response latency metrics
- ✅ Error logging

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                    (Streamlit Web App)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Query Processing                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Query      │  │   Query      │  │   Language   │          │
│  │   Rewriter   │──▶│   Expansion  │──▶│   Detection  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Retrieval Pipeline                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Vector     │  │   Hybrid     │  │   Reranker   │          │
│  │   Search     │──▶│   Fusion     │──▶│   (Cross-    │          │
│  │   (Chroma)   │  │              │  │   Encoder)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Generation Pipeline                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Context    │  │   LLM        │  │   Response   │          │
│  │   Formatter  │──▶│   (GPT-4/   │──▶│   Formatter  │          │
│  │              │  │   Claude)    │  │   + Citation │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | OpenAI GPT-4o-mini / Claude Haiku |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector DB** | ChromaDB (local) / Pinecone (cloud) |
| **Framework** | LangChain 0.1+ |
| **UI** | Streamlit |
| **Monitoring** | LangSmith |

## 🚀 Cài đặt

### Yêu cầu hệ thống

- Python 3.10+
- 4GB RAM minimum
- 2GB disk space

### Bước 1: Clone repository

```bash
git clone https://github.com/yourusername/climate-qa-rag.git
cd climate-qa-rag
```

### Bước 2: Tạo virtual environment

```bash
# Với conda
conda create -n climate-qa python=3.11
conda activate climate-qa

# Hoặc với venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### Bước 3: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình environment variables

```bash
cp .env.example .env
```

Chỉnh sửa file `.env`:

```env
# Required
OPENAI_API_KEY=sk-your-openai-api-key

# Optional - for advanced features
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key
LANGCHAIN_API_KEY=ls__your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=climate-qa-rag

# Optional - for cloud vector DB
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=climate-qa
```

### Bước 5: Index documents

```bash
# Index sample documents
python scripts/index_documents.py --data-dir data/sample

# Hoặc index custom documents
python scripts/index_documents.py --data-dir /path/to/your/documents
```

### Bước 6: Chạy ứng dụng

```bash
streamlit run src/app.py
```

Truy cập: http://localhost:8501

## 📖 Sử dụng

### Web Interface

1. Mở browser tại `http://localhost:8501`
2. Upload documents hoặc sử dụng sample data
3. Nhập câu hỏi vào chat box
4. Xem câu trả lời với citations

### CLI Mode

```bash
# Single question
python src/cli.py "Biến đổi khí hậu ảnh hưởng như thế nào đến Việt Nam?"

# Interactive mode
python src/cli.py --interactive
```

### API Mode

```bash
# Start API server
uvicorn src.api:app --reload --port 8000

# Query API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What causes global warming?"}'
```

## 📁 Cấu trúc dự án

```
climate-qa-rag/
├── src/
│   ├── __init__.py
│   ├── app.py              # Streamlit web app
│   ├── api.py              # FastAPI endpoints
│   ├── cli.py              # Command line interface
│   ├── config.py           # Configuration management
│   ├── document_loader.py  # Document processing
│   ├── embeddings.py       # Embedding models
│   ├── vector_store.py     # Vector database operations
│   ├── retriever.py        # Retrieval logic
│   ├── reranker.py         # Reranking models
│   ├── llm.py              # LLM interface
│   ├── chain.py            # RAG chain composition
│   ├── prompts.py          # Prompt templates
│   └── utils.py            # Utility functions
├── data/
│   ├── sample/             # Sample documents
│   └── chroma_db/          # Vector database storage
├── tests/
│   ├── test_retriever.py
│   ├── test_chain.py
│   └── test_api.py
├── scripts/
│   ├── index_documents.py  # Document indexing script
│   ├── evaluate.py         # RAG evaluation
│   └── export_data.py      # Data export utilities
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 🔌 API Reference

### POST /query

Query the RAG system.

**Request:**
```json
{
  "question": "What is climate change?",
  "language": "en",
  "top_k": 5,
  "use_reranker": true
}
```

**Response:**
```json
{
  "answer": "Climate change refers to...",
  "sources": [
    {
      "content": "...",
      "metadata": {
        "source": "ipcc_ar6_summary.pdf",
        "page": 12
      },
      "relevance_score": 0.92
    }
  ],
  "tokens_used": 1250,
  "latency_ms": 2340
}
```

### POST /documents/upload

Upload and index new documents.

### GET /health

Health check endpoint.

## 🐳 Deployment

### Docker

```bash
# Build image
docker build -t climate-qa-rag .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  climate-qa-rag
```

### Docker Compose

```bash
docker-compose up -d
```

### Hugging Face Spaces

1. Fork repository
2. Create new Space (Streamlit SDK)
3. Add secrets: `OPENAI_API_KEY`
4. Deploy

### Streamlit Cloud

1. Connect GitHub repository
2. Set environment variables
3. Deploy

## 📊 Evaluation

Run RAG evaluation:

```bash
python scripts/evaluate.py \
  --test-file data/test_questions.json \
  --output-file results/evaluation.json
```

Metrics:
- **Answer Relevance**: Độ liên quan của câu trả lời
- **Faithfulness**: Câu trả lời có dựa trên context không
- **Context Precision**: Độ chính xác của retrieval
- **Context Recall**: Độ đầy đủ của retrieval

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [IPCC](https://www.ipcc.ch/) - Climate science reports
- [EPA](https://www.epa.gov/) - Environmental data
- [LangChain](https://langchain.com/) - RAG framework
- [OpenAI](https://openai.com/) - Language models

## 📧 Contact

- **Author**: [Your Name]
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn]
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

⭐ Nếu dự án hữu ích, hãy cho một star!
