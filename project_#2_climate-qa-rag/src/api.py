"""
FastAPI REST API for Climate Q&A RAG System.

Run with: uvicorn src.api:app --reload --port 8000
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.chain import RAGChain, AdvancedRAGChain
from src.vector_store import VectorStoreManager, load_existing_index
from src.embeddings import get_embedding_model
from src.document_loader import ClimateDocumentLoader
from src.utils import detect_language


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for queries."""
    question: str = Field(..., description="Question to ask", min_length=1, max_length=2000)
    language: str = Field(default="auto", description="Response language (en/vi/auto)")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")
    use_reranker: bool = Field(default=True, description="Whether to use reranking")
    return_sources: bool = Field(default=True, description="Whether to return source documents")


class QueryResponse(BaseModel):
    """Response model for queries."""
    answer: str = Field(..., description="Generated answer")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Source documents")
    language: str = Field(..., description="Detected/used language")
    latency_ms: int = Field(..., description="Response latency in milliseconds")
    num_sources: int = Field(..., description="Number of sources used")


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    status: str
    message: str
    num_chunks: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    vector_store: str
    document_count: Optional[int]


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request with history."""
    message: str = Field(..., description="Current message")
    history: List[ChatMessage] = Field(default=[], description="Chat history")
    language: str = Field(default="auto")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Climate Science Q&A API",
    description="RAG-based API for answering climate science questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_chain: Optional[AdvancedRAGChain] = None
_vector_store_manager: Optional[VectorStoreManager] = None


# =============================================================================
# Startup and Shutdown
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    global _chain, _vector_store_manager
    
    logger.info("Starting Climate Q&A API...")
    
    # Setup LangSmith if configured
    settings.setup_langsmith()
    
    try:
        # Initialize embeddings and vector store
        embeddings = get_embedding_model()
        _vector_store_manager = VectorStoreManager(embeddings=embeddings)
        
        # Initialize RAG chain
        _chain = AdvancedRAGChain(
            vector_store=_vector_store_manager.vector_store,
            enable_memory=False,  # Stateless for API
            language="auto",
        )
        
        logger.info("API initialization complete")
    except Exception as e:
        logger.warning(f"Could not initialize chain: {e}")
        logger.info("API started without pre-loaded documents")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Climate Q&A API...")


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    doc_count = None
    vs_status = "not_initialized"
    
    if _vector_store_manager:
        try:
            stats = _vector_store_manager.get_collection_stats()
            doc_count = stats.get("count")
            vs_status = "healthy"
        except Exception:
            vs_status = "error"
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store=vs_status,
        document_count=doc_count,
    )


@app.get("/", tags=["System"])
async def root():
    """API root endpoint."""
    return {
        "name": "Climate Science Q&A API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# Query Endpoints
# =============================================================================

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the RAG system with a question.
    
    Returns an answer based on indexed climate science documents.
    """
    if _chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please upload documents first."
        )
    
    start_time = time.time()
    
    try:
        # Detect language
        language = request.language
        if language == "auto":
            language = detect_language(request.question)
        
        # Query the chain
        result = _chain.invoke(
            request.question,
            return_sources=request.return_sources,
        )
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources") if request.return_sources else None,
            language=language,
            latency_ms=latency_ms,
            num_sources=result.get("num_sources", 0),
        )
        
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Stream query response.
    
    Returns a streaming response with answer chunks.
    """
    if _chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized."
        )
    
    async def generate():
        try:
            for chunk in _chain.stream(request.question):
                if chunk.get("type") == "answer":
                    yield f"data: {chunk['content']}\n\n"
                elif chunk.get("type") == "done":
                    yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )


# =============================================================================
# Document Management
# =============================================================================

@app.post("/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """
    Upload and index documents.
    
    Supports PDF, TXT, MD, and DOCX files.
    """
    global _chain, _vector_store_manager
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file types
    allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}"
            )
    
    try:
        # Save files temporarily
        temp_dir = Path("/tmp/climate_qa_api_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        saved_files = []
        for file in files:
            file_path = temp_dir / file.filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(file_path)
        
        # Load and index documents
        loader = ClimateDocumentLoader()
        documents = loader.load_and_split(temp_dir)
        
        # Initialize or update vector store
        if _vector_store_manager is None:
            embeddings = get_embedding_model()
            _vector_store_manager = VectorStoreManager(embeddings=embeddings)
        
        _vector_store_manager.add_documents(documents)
        
        # Reinitialize chain
        _chain = AdvancedRAGChain(
            vector_store=_vector_store_manager.vector_store,
            enable_memory=False,
            language="auto",
        )
        
        # Cleanup temp files in background
        def cleanup():
            for f in saved_files:
                try:
                    f.unlink()
                except:
                    pass
        
        background_tasks.add_task(cleanup)
        
        return DocumentUploadResponse(
            status="success",
            message=f"Indexed {len(files)} files",
            num_chunks=len(documents),
        )
        
    except Exception as e:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/stats", tags=["Documents"])
async def document_stats():
    """Get statistics about indexed documents."""
    if _vector_store_manager is None:
        return {"status": "no_documents", "count": 0}
    
    try:
        stats = _vector_store_manager.get_collection_stats()
        return {
            "status": "ok",
            "collection": stats.get("name"),
            "count": stats.get("count", 0),
            "type": stats.get("type"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents", tags=["Documents"])
async def clear_documents():
    """Clear all indexed documents."""
    global _chain, _vector_store_manager
    
    if _vector_store_manager is None:
        return {"status": "ok", "message": "No documents to clear"}
    
    try:
        _vector_store_manager.delete_collection()
        _vector_store_manager = None
        _chain = None
        
        return {"status": "ok", "message": "All documents cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Sample Questions
# =============================================================================

@app.get("/samples", tags=["Samples"])
async def get_sample_questions():
    """Get sample questions for testing."""
    return {
        "english": [
            "What are the main causes of climate change?",
            "How does deforestation contribute to global warming?",
            "What is the Paris Agreement and what are its goals?",
            "How can individuals reduce their carbon footprint?",
            "What are the effects of climate change on biodiversity?",
        ],
        "vietnamese": [
            "Nguyên nhân chính gây ra biến đổi khí hậu là gì?",
            "Phá rừng góp phần vào sự nóng lên toàn cầu như thế nào?",
            "Hiệp định Paris là gì và mục tiêu của nó là gì?",
            "Cá nhân có thể giảm dấu chân carbon như thế nào?",
            "Tác động của biến đổi khí hậu đối với đa dạng sinh học?",
        ],
    }


# =============================================================================
# Run Application
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.is_development,
    )
