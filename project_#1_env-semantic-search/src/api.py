"""
FastAPI REST API for Environmental Semantic Search Tool.

Run with: uvicorn src.api:app --reload --port 8000
"""

import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from .config import settings
from .search_engine import SemanticSearchEngine, get_search_engine
from .vector_store import load_index, VectorStore
from .document_processor import DocumentProcessor


# =============================================================================
# API Models
# =============================================================================

class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


class SearchResultItem(BaseModel):
    """Single search result."""
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResultItem]
    total_results: int
    search_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    document_count: int
    embedding_model: str


class StatsResponse(BaseModel):
    """Statistics response."""
    collection_name: str
    document_count: int
    embedding_model: str
    embedding_dimensions: int
    sources: List[str]


class IndexRequest(BaseModel):
    """Document indexing request."""
    texts: List[str] = Field(..., min_length=1, description="Texts to index")
    sources: Optional[List[str]] = Field(default=None, description="Source names")


class IndexResponse(BaseModel):
    """Indexing response."""
    status: str
    documents_indexed: int


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Environmental Semantic Search API",
    description="AI-powered semantic search for environmental documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Application state."""
    search_engine: Optional[SemanticSearchEngine] = None
    initialized: bool = False


state = AppState()


# =============================================================================
# Startup/Shutdown Events
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("Starting Environmental Semantic Search API...")
    
    try:
        state.search_engine = get_search_engine()
        state.initialized = True
        logger.info("API initialized successfully")
    except Exception as e:
        logger.warning(f"Could not initialize search engine: {e}")
        logger.info("API starting without pre-loaded index")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "Environmental Semantic Search API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "healthy" if state.initialized else "not_initialized",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    doc_count = 0
    embedding_model = settings.embedding_model
    
    if state.search_engine:
        try:
            stats = state.search_engine.get_stats()
            doc_count = stats.get("document_count", 0)
            embedding_model = stats.get("embedding_model", embedding_model)
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if state.initialized else "not_initialized",
        version="1.0.0",
        document_count=doc_count,
        embedding_model=embedding_model,
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Perform semantic search.
    
    Search for documents similar to the query using semantic understanding.
    """
    if not state.search_engine:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Please index documents first.",
        )
    
    try:
        response = state.search_engine.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            filter=request.filter,
        )
        
        return SearchResponse(
            query=response.query,
            results=[
                SearchResultItem(
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in response.results
            ],
            total_results=response.total_results,
            search_time_ms=response.search_time_ms,
        )
        
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of results"),
    threshold: float = Query(default=0.3, ge=0.0, le=1.0, description="Minimum similarity"),
):
    """
    Perform semantic search (GET method).
    
    Simple GET endpoint for quick searches.
    """
    request = SearchRequest(query=q, top_k=top_k, threshold=threshold)
    return await search(request)


@app.get("/stats", response_model=StatsResponse, tags=["Index"])
async def get_stats():
    """Get index statistics."""
    if not state.search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    try:
        stats = state.search_engine.get_stats()
        sources = state.search_engine.get_sources()
        
        return StatsResponse(
            collection_name=stats.get("collection_name", "unknown"),
            document_count=stats.get("document_count", 0),
            embedding_model=stats.get("embedding_model", "unknown"),
            embedding_dimensions=stats.get("embedding_dimensions", 0),
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources", tags=["Index"])
async def list_sources():
    """List all indexed document sources."""
    if not state.search_engine:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    sources = state.search_engine.get_sources()
    return {"sources": sources, "count": len(sources)}


@app.post("/index/texts", response_model=IndexResponse, tags=["Index"])
async def index_texts(request: IndexRequest):
    """
    Index text documents.
    
    Index raw text content into the search engine.
    """
    try:
        processor = DocumentProcessor()
        documents = processor.process_texts(
            texts=request.texts,
            sources=request.sources,
        )
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid content to index")
        
        # Get or create vector store
        vector_store = load_index()
        vector_store.add_documents(documents)
        
        # Update search engine
        state.search_engine = SemanticSearchEngine(vector_store=vector_store)
        state.initialized = True
        
        return IndexResponse(
            status="success",
            documents_indexed=len(documents),
        )
        
    except Exception as e:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/files", response_model=IndexResponse, tags=["Index"])
async def index_files(files: List[UploadFile] = File(...)):
    """
    Index uploaded files.
    
    Upload and index document files (PDF, TXT, MD, DOCX).
    """
    import tempfile
    from pathlib import Path
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save files
            for file in files:
                file_path = temp_path / file.filename
                content = await file.read()
                with open(file_path, "wb") as f:
                    f.write(content)
            
            # Process
            processor = DocumentProcessor()
            result = processor.process_directory(temp_path)
            
            if not result.documents:
                raise HTTPException(status_code=400, detail="No valid content found")
            
            # Index
            vector_store = load_index()
            vector_store.add_documents(result.documents)
            
            # Update state
            state.search_engine = SemanticSearchEngine(vector_store=vector_store)
            state.initialized = True
            
            return IndexResponse(
                status="success",
                documents_indexed=result.total_chunks,
            )
            
    except Exception as e:
        logger.exception("File indexing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index", tags=["Index"])
async def clear_index():
    """Clear all indexed documents."""
    if not state.search_engine:
        return {"status": "no_index"}
    
    try:
        state.search_engine.vector_store.delete_collection()
        state.initialized = False
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Run with Uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
    )
