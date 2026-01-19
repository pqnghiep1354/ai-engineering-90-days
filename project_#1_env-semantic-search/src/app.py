"""
Streamlit Web Application for Environmental Semantic Search Tool.

Run with: streamlit run src/app.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

from src.config import settings
from src.search_engine import SemanticSearchEngine, SearchResponse, format_results_for_display
from src.vector_store import load_index, VectorStore
from src.document_processor import DocumentProcessor
from src.utils import setup_logging, format_file_size


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Environmental Semantic Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #2E7D32;
    }
    .score-badge {
        background-color: #2E7D32;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    .source-badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
    .metric-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "search_engine" not in st.session_state:
        st.session_state.search_engine = None
    
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    if "total_searches" not in st.session_state:
        st.session_state.total_searches = 0


def load_search_engine():
    """Load or create search engine."""
    if st.session_state.search_engine is None:
        try:
            vector_store = load_index()
            st.session_state.search_engine = SemanticSearchEngine(vector_store=vector_store)
            return True
        except Exception as e:
            logger.error(f"Failed to load search engine: {e}")
            return False
    return True


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.markdown("## ⚙️ Search Settings")
        
        # Number of results
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of search results to return",
        )
        
        # Similarity threshold
        threshold = st.slider(
            "Minimum Similarity",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Minimum similarity score (0-1)",
        )
        
        st.markdown("---")
        
        # Index statistics
        st.markdown("## 📊 Index Statistics")
        
        if st.session_state.search_engine:
            stats = st.session_state.search_engine.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get("document_count", 0))
            with col2:
                st.metric("Searches", st.session_state.total_searches)
            
            # Sources
            sources = st.session_state.search_engine.get_sources()
            if sources:
                with st.expander(f"📁 Sources ({len(sources)})"):
                    for source in sources:
                        st.text(f"• {source}")
        else:
            st.warning("Index not loaded")
        
        st.markdown("---")
        
        # Document upload
        st.markdown("## 📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files to index",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )
        
        if uploaded_files and st.button("📥 Index Documents"):
            index_uploaded_files(uploaded_files)
        
        st.markdown("---")
        
        # About
        st.markdown("## ℹ️ About")
        st.markdown("""
        **Environmental Semantic Search**
        
        AI-powered search for environmental documents using semantic understanding.
        
        🌿 Built for Portfolio Project #1
        """)
        
        return top_k, threshold


# =============================================================================
# Document Indexing
# =============================================================================

def index_uploaded_files(uploaded_files):
    """Index uploaded files."""
    import tempfile
    
    with st.spinner("Processing documents..."):
        try:
            # Create temp directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save uploaded files
                for file in uploaded_files:
                    file_path = temp_path / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # Process documents
                processor = DocumentProcessor()
                result = processor.process_directory(temp_path)
                
                if result.documents:
                    # Add to vector store
                    vector_store = load_index()
                    vector_store.add_documents(result.documents)
                    
                    # Reload search engine
                    st.session_state.search_engine = SemanticSearchEngine(vector_store=vector_store)
                    
                    st.success(f"✅ Indexed {result.total_chunks} chunks from {len(uploaded_files)} files!")
                else:
                    st.warning("No content found in uploaded files")
                    
        except Exception as e:
            st.error(f"Error indexing documents: {e}")


# =============================================================================
# Search Interface
# =============================================================================

def render_search_interface(top_k: int, threshold: float):
    """Render main search interface."""
    
    # Header
    st.markdown('<p class="main-header">🔍 Environmental Semantic Search</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Search environmental documents using AI-powered semantic understanding</p>',
        unsafe_allow_html=True
    )
    
    # Search input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter your search query... (e.g., 'What causes climate change?')",
            label_visibility="collapsed",
        )
    
    with col2:
        search_clicked = st.button("🔍 Search", use_container_width=True)
    
    # Sample queries
    st.markdown("**💡 Try these queries:**")
    sample_queries = [
        "greenhouse gas emissions",
        "renewable energy benefits",
        "carbon footprint reduction",
        "climate change impacts Vietnam",
        "ESG reporting standards",
    ]
    
    cols = st.columns(len(sample_queries))
    for i, (col, sample) in enumerate(zip(cols, sample_queries)):
        with col:
            if st.button(sample, key=f"sample_{i}", use_container_width=True):
                query = sample
                search_clicked = True
    
    st.markdown("---")
    
    # Perform search
    if search_clicked and query:
        perform_search(query, top_k, threshold)
    
    # Show search history
    if st.session_state.search_history:
        with st.expander("📜 Recent Searches"):
            for i, past_query in enumerate(reversed(st.session_state.search_history[-5:])):
                if st.button(f"🔄 {past_query}", key=f"history_{i}"):
                    perform_search(past_query, top_k, threshold)


def perform_search(query: str, top_k: int, threshold: float):
    """Perform search and display results."""
    
    if not st.session_state.search_engine:
        st.error("Search engine not initialized. Please index some documents first.")
        return
    
    # Add to history
    if query not in st.session_state.search_history:
        st.session_state.search_history.append(query)
    st.session_state.total_searches += 1
    
    # Perform search
    with st.spinner("Searching..."):
        try:
            response = st.session_state.search_engine.search(
                query=query,
                top_k=top_k,
                threshold=threshold,
            )
        except Exception as e:
            if "RateLimitError" in str(e) or "429" in str(e):
                st.error("⚠️ Quá giới hạn API (Rate Limit). Vui lòng thử lại sau vài giây.")
            else:
                st.error(f"❌ Có lỗi xảy ra: {str(e)}")
            return
    
    # Display results header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Results Found", response.total_results)
    with col2:
        st.metric("Search Time", f"{response.search_time_ms:.0f}ms")
    with col3:
        st.metric("Language", response.language.upper())
    
    st.markdown("---")
    
    # Display results
    if response.results:
        for i, result in enumerate(response.results, 1):
            render_result_card(i, result)
    else:
        st.info("No results found. Try a different query or lower the similarity threshold.")


def render_result_card(index: int, result):
    """Render a single result card."""
    
    with st.container():
        st.markdown(f"""
        <div class="result-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span><strong>Result #{index}</strong></span>
                <span>
                    <span class="score-badge">Score: {result.score:.3f}</span>
                    <span class="source-badge">📄 {result.source}</span>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Content
        st.markdown(result.content)
        
        # Metadata expander
        with st.expander("📋 Metadata"):
            for key, value in result.metadata.items():
                st.text(f"{key}: {value}")
        
        st.markdown("---")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    
    # Initialize
    init_session_state()
    setup_logging(level="WARNING")
    
    # Load search engine
    engine_loaded = load_search_engine()
    
    # Render sidebar and get settings
    top_k, threshold = render_sidebar()
    
    # Main content
    if not engine_loaded:
        st.warning(
            "⚠️ No documents indexed yet. "
            "Please upload documents in the sidebar or run the indexing script."
        )
        st.code("python scripts/index_documents.py --data-dir data/documents")
    
    # Render search interface
    render_search_interface(top_k, threshold)


if __name__ == "__main__":
    main()
