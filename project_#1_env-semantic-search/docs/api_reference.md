# API Reference

## REST API Endpoints

Base URL: `http://localhost:8000`

### Health Check

#### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "document_count": 150,
  "embedding_model": "text-embedding-3-small"
}
```

---

### Search

#### POST /search

Perform semantic search.

**Request Body:**
```json
{
  "query": "What causes climate change?",
  "top_k": 5,
  "threshold": 0.3,
  "filter": {
    "source": "climate_report.pdf"
  }
}
```

**Parameters:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| query | string | Yes | - | Search query (1-1000 chars) |
| top_k | integer | No | 5 | Number of results (1-50) |
| threshold | float | No | 0.3 | Min similarity (0-1) |
| filter | object | No | null | Metadata filter |

**Response:**
```json
{
  "query": "What causes climate change?",
  "results": [
    {
      "content": "Climate change is primarily caused by...",
      "score": 0.8923,
      "metadata": {
        "source": "climate_overview.md",
        "chunk_index": 5
      }
    }
  ],
  "total_results": 5,
  "search_time_ms": 125.5
}
```

#### GET /search

Simple search via query parameters.

**Query Parameters:**
- `q` (required): Search query
- `top_k` (optional): Number of results
- `threshold` (optional): Minimum similarity

**Example:**
```
GET /search?q=renewable%20energy&top_k=10
```

---

### Index Management

#### GET /stats

Get index statistics.

**Response:**
```json
{
  "collection_name": "environmental_docs",
  "document_count": 150,
  "embedding_model": "text-embedding-3-small",
  "embedding_dimensions": 1536,
  "sources": ["climate.pdf", "energy.md", "esg.docx"]
}
```

#### GET /sources

List indexed document sources.

**Response:**
```json
{
  "sources": [
    "climate_change_overview.md",
    "renewable_energy_guide.md",
    "esg_sustainability_guide.md"
  ],
  "count": 3
}
```

#### POST /index/texts

Index raw text content.

**Request Body:**
```json
{
  "texts": [
    "Climate change refers to long-term shifts...",
    "Renewable energy comes from sources..."
  ],
  "sources": ["doc1", "doc2"]
}
```

**Response:**
```json
{
  "status": "success",
  "documents_indexed": 15
}
```

#### POST /index/files

Upload and index files.

**Request:** `multipart/form-data`
- `files`: One or more files (PDF, TXT, MD, DOCX)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/index/files" \
  -F "files=@document1.pdf" \
  -F "files=@document2.md"
```

**Response:**
```json
{
  "status": "success",
  "documents_indexed": 25
}
```

#### DELETE /index

Clear all indexed documents.

**Response:**
```json
{
  "status": "cleared"
}
```

---

## Python SDK Usage

### Basic Search

```python
from src.search_engine import get_search_engine

# Initialize
engine = get_search_engine()

# Search
response = engine.search(
    query="climate change impacts",
    top_k=5,
    threshold=0.3
)

# Access results
for result in response.results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
    print(f"Source: {result.source}")
```

### Document Processing

```python
from src.document_processor import DocumentProcessor

# Initialize processor
processor = DocumentProcessor(
    chunk_size=1000,
    chunk_overlap=200
)

# Process directory
result = processor.process_directory("data/documents")

print(f"Processed {result.total_chunks} chunks")
print(f"From {len(result.sources)} files")
```

### Vector Store Operations

```python
from src.vector_store import VectorStore, load_index

# Load existing index
store = load_index()

# Search
results = store.search(
    query="What is ESG?",
    top_k=5,
    threshold=0.3
)

# Get statistics
stats = store.get_stats()
print(f"Documents: {stats['document_count']}")

# List sources
sources = store.list_sources()
```

### Embedding Operations

```python
from src.embeddings import get_embedding_model

# Initialize
model = get_embedding_model()

# Embed single text
embedding = model.embed_text("climate change")
print(f"Dimensions: {len(embedding)}")

# Embed multiple texts
embeddings = model.embed_texts([
    "text one",
    "text two"
])

# Get usage stats
stats = model.get_stats()
print(f"Total tokens: {stats['total_tokens']}")
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 500 | Internal server error |
| 503 | Service unavailable (not initialized) |

### Error Response Format

```json
{
  "detail": "Error message describing the issue"
}
```

---

## Rate Limits

Default configuration has no rate limits. For production:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/search")
@limiter.limit("100/minute")
async def search(request: SearchRequest):
    ...
```

---

## Examples

### cURL Examples

**Search:**
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "renewable energy benefits", "top_k": 5}'
```

**Get Stats:**
```bash
curl "http://localhost:8000/stats"
```

**Index Text:**
```bash
curl -X POST "http://localhost:8000/index/texts" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Sample document content..."], "sources": ["sample.txt"]}'
```

### Python Requests Examples

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "What is carbon footprint?",
        "top_k": 5
    }
)
results = response.json()

# Upload files
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/index/files",
        files={"files": f}
    )
```

### JavaScript/Fetch Examples

```javascript
// Search
const response = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'climate change impacts',
    top_k: 5
  })
});

const data = await response.json();
console.log(data.results);
```
