# Deployment Guide - Climate Q&A RAG System

## Mục Lục

1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Hugging Face Spaces](#hugging-face-spaces)
4. [Streamlit Cloud](#streamlit-cloud)
5. [Production Considerations](#production-considerations)

---

## Local Development

### Yêu cầu

- Python 3.10+
- 4GB RAM minimum
- OpenAI API key

### Cài đặt

```bash
# Clone repository
git clone https://github.com/yourusername/climate-qa-rag.git
cd climate-qa-rag

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Copy và cấu hình .env
cp .env.example .env
# Chỉnh sửa .env với API keys của bạn
```

### Index Documents

```bash
# Index sample documents
python scripts/index_documents.py --data-dir data/sample

# Hoặc index custom documents
python scripts/index_documents.py --data-dir /path/to/your/documents
```

### Chạy ứng dụng

```bash
# Streamlit Web App
streamlit run src/app.py

# FastAPI Server
uvicorn src.api:app --reload --port 8000

# CLI
python src/cli.py --interactive
```

---

## Docker Deployment

### Build Image

```bash
# Build
docker build -t climate-qa-rag .

# Hoặc với docker-compose
docker-compose build
```

### Run với Docker

```bash
# Streamlit
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  climate-qa-rag

# API Server
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  climate-qa-rag uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Run với Docker Compose

```bash
# Tạo file .env với API keys
echo "OPENAI_API_KEY=your-key" > .env

# Start services
docker-compose up -d

# Index documents (first time)
docker-compose run --rm indexer

# View logs
docker-compose logs -f streamlit

# Stop
docker-compose down
```

---

## Hugging Face Spaces

### Bước 1: Chuẩn bị Repository

1. Fork repository về GitHub của bạn
2. Đảm bảo có file `requirements.txt`
3. Tạo file `app.py` ở root (copy từ `src/app.py`)

### Bước 2: Tạo Space

1. Đi đến [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Chọn:
   - **SDK**: Streamlit
   - **Hardware**: CPU Basic (Free) hoặc nâng cấp nếu cần
   - **Visibility**: Public/Private

### Bước 3: Cấu hình Secrets

Trong Space Settings → Repository secrets:

```
OPENAI_API_KEY=sk-your-key
LANGCHAIN_API_KEY=ls__your-key  # Optional
```

### Bước 4: Push Code

```bash
# Add HF remote
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/climate-qa-rag

# Push
git push space main
```

### File cấu hình cho HF Spaces

Tạo `README.md` với frontmatter:

```yaml
---
title: Climate Science Q&A
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.30.0
app_file: src/app.py
pinned: false
---
```

---

## Streamlit Cloud

### Bước 1: Push code lên GitHub

```bash
git add .
git commit -m "Prepare for Streamlit Cloud"
git push origin main
```

### Bước 2: Deploy trên Streamlit Cloud

1. Đi đến [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Chọn repository, branch, và file path (`src/app.py`)
4. Click "Deploy"

### Bước 3: Cấu hình Secrets

Trong App Settings → Secrets:

```toml
OPENAI_API_KEY = "sk-your-key"
LANGCHAIN_API_KEY = "ls__your-key"
```

---

## Production Considerations

### 1. Security

```python
# Validate API inputs
from pydantic import BaseModel, Field, validator

class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000)
    
    @validator('question')
    def sanitize_question(cls, v):
        # Remove potential injection patterns
        return v.strip()

# Rate limiting
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/query")
@limiter.limit("10/minute")
async def query(request: Request, ...):
    ...
```

### 2. Monitoring

```python
# LangSmith integration
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "climate-qa-production"

# Custom metrics
from prometheus_client import Counter, Histogram

query_counter = Counter('queries_total', 'Total queries processed')
latency_histogram = Histogram('query_latency_seconds', 'Query latency')
```

### 3. Caching

```python
from functools import lru_cache
from cachetools import TTLCache

# Response caching
response_cache = TTLCache(maxsize=1000, ttl=3600)

def cached_query(question: str):
    if question in response_cache:
        return response_cache[question]
    
    result = chain.invoke(question)
    response_cache[question] = result
    return result
```

### 4. Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def query_with_retry(question: str):
    return await chain.ainvoke(question)
```

### 5. Scaling

#### Horizontal Scaling với Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: climate-qa-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: climate-qa-api
  template:
    metadata:
      labels:
        app: climate-qa-api
    spec:
      containers:
      - name: api
        image: climate-qa-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: climate-qa-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### 6. Vector Database for Production

#### Pinecone (Recommended for Production)

```python
# config.py
VECTOR_DB_TYPE = "pinecone"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "climate-qa-production"

# Initialize
from pinecone import Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
```

### 7. Backup and Recovery

```bash
# Backup ChromaDB
tar -czvf chroma_backup_$(date +%Y%m%d).tar.gz data/chroma_db/

# Restore
tar -xzvf chroma_backup_20240101.tar.gz
```

---

## Checklist Trước Khi Deploy

- [ ] API keys được cấu hình qua environment variables
- [ ] Không có secrets trong code
- [ ] Rate limiting được enable
- [ ] Error handling đầy đủ
- [ ] Logging được cấu hình
- [ ] Health check endpoint hoạt động
- [ ] Documents đã được index
- [ ] Tests pass
- [ ] Documentation đầy đủ

---

## Troubleshooting

### Lỗi thường gặp

**1. "OpenAI API key not configured"**
```bash
export OPENAI_API_KEY=sk-your-key
# Hoặc thêm vào .env file
```

**2. "No documents indexed"**
```bash
python scripts/index_documents.py --data-dir data/sample
```

**3. "Out of memory"**
- Giảm `chunk_size` trong config
- Giảm `retriever_top_k`
- Sử dụng smaller embedding model

**4. Docker container không start**
```bash
# Check logs
docker logs climate-qa-streamlit

# Check health
curl http://localhost:8501/_stcore/health
```

---

## Support

- GitHub Issues: [Link to issues]
- Documentation: [Link to docs]
- Email: your.email@example.com
