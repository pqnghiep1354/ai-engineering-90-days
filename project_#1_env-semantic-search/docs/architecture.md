# System Architecture

## Overview

Environmental Semantic Search Tool is built using a modular architecture that separates concerns and enables easy extension.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           User Interfaces                                │
├─────────────────┬─────────────────┬─────────────────────────────────────┤
│   Web App       │   REST API      │   CLI                               │
│   (Streamlit)   │   (FastAPI)     │   (argparse)                        │
│   :8501         │   :8000         │                                     │
└────────┬────────┴────────┬────────┴────────┬────────────────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Search Engine Layer                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SemanticSearchEngine                                            │   │
│  │  - search(query, top_k, threshold, filter)                      │   │
│  │  - multi_query_search(queries)                                  │   │
│  │  - find_similar(text)                                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Vector Store Layer                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  VectorStore (ChromaDB)                                          │   │
│  │  - add_documents(documents)                                      │   │
│  │  - search(query_embedding, top_k)                               │   │
│  │  - delete_collection()                                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────────┐    ┌─────────────────────────────────────────┐
│    Embedding Layer      │    │         Document Processing              │
│  ┌───────────────────┐  │    │  ┌─────────────────────────────────┐   │
│  │  EmbeddingModel   │  │    │  │  DocumentProcessor              │   │
│  │  (OpenAI API)     │  │    │  │  - load_file(path)              │   │
│  │  - embed_text()   │  │    │  │  - chunk_text(text)             │   │
│  │  - embed_texts()  │  │    │  │  - process_directory()          │   │
│  └───────────────────┘  │    │  └─────────────────────────────────┘   │
└─────────────────────────┘    └─────────────────────────────────────────┘
```

## Component Details

### 1. User Interfaces

#### Web App (Streamlit)
- Interactive search interface
- Real-time results display
- Document upload capability
- Search history tracking

#### REST API (FastAPI)
- RESTful endpoints
- OpenAPI documentation
- JSON request/response
- CORS support

#### CLI
- Command-line search
- Interactive mode
- JSON output option
- Scriptable

### 2. Search Engine

The `SemanticSearchEngine` class provides high-level search functionality:

```python
class SemanticSearchEngine:
    def search(query, top_k, threshold, filter) -> SearchResponse
    def multi_query_search(queries) -> SearchResponse
    def find_similar(text) -> SearchResponse
    def get_sources() -> List[str]
    def get_stats() -> Dict
```

### 3. Vector Store

ChromaDB-based vector storage with:
- Cosine similarity search
- Metadata filtering
- Persistent storage
- Batch operations

### 4. Embedding Layer

OpenAI embedding models:
- `text-embedding-3-small` (default)
- `text-embedding-3-large`
- `text-embedding-ada-002`

Features:
- Batch processing
- Retry logic
- Cost tracking

### 5. Document Processing

Supported formats:
- PDF (pypdf)
- DOCX (python-docx)
- TXT/MD (plain text)
- HTML (BeautifulSoup)

Processing pipeline:
1. Load file
2. Extract text
3. Clean and normalize
4. Split into chunks
5. Add metadata

## Data Flow

### Indexing Flow

```
Documents → Load → Extract Text → Clean → Chunk → Embed → Store
    │                                                        │
    └──────────────────────────────────────────────────────►│
                                                    ChromaDB/Pinecone
```

### Search Flow

```
Query → Embed → Vector Search → Rank → Filter → Format → Return
                     │
                     ▼
              ChromaDB/Pinecone
```

## Configuration

Configuration is managed via Pydantic Settings:

```python
class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    
    # Embedding
    embedding_model: str
    embedding_dimensions: int
    
    # Vector DB
    vector_db_type: str
    chroma_persist_dir: str
    
    # Document Processing
    chunk_size: int
    chunk_overlap: int
    
    # Search
    default_top_k: int
    min_similarity_score: float
```

## Extensibility

### Adding New Document Formats

1. Create loader function in `document_processor.py`:
```python
def load_new_format(file_path: Path) -> str:
    # Implementation
    pass
```

2. Register in LOADERS dict:
```python
LOADERS[".new"] = load_new_format
```

### Adding New Embedding Providers

1. Create new embedding class implementing same interface
2. Update factory function in `embeddings.py`

### Adding New Vector Stores

1. Create new store class implementing `VectorStore` interface
2. Update factory functions in `vector_store.py`

## Performance Considerations

### Embedding Caching
- Consider caching embeddings for repeated queries
- Use batch processing for multiple documents

### Search Optimization
- Adjust `top_k` based on needs
- Use metadata filters to reduce search space
- Consider HNSW parameters for large datasets

### Memory Management
- ChromaDB loads index into memory
- For large datasets, consider Pinecone or Qdrant
- Implement pagination for large result sets

## Security

### API Key Management
- Store keys in environment variables
- Never commit keys to version control
- Use `.env` file for local development

### Input Validation
- Validate query length
- Sanitize file uploads
- Rate limiting in production

## Monitoring

### Logging
- Loguru for structured logging
- Configurable log levels
- Optional file logging

### Metrics
- Search latency tracking
- Embedding API usage
- Error rates

## Deployment Options

1. **Local Development**: ChromaDB + Streamlit
2. **Docker**: Containerized deployment
3. **Cloud**: 
   - Streamlit Cloud
   - Hugging Face Spaces
   - AWS/GCP/Azure with Pinecone
