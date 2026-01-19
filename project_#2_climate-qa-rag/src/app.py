"""
Streamlit Web Application for Climate Q&A RAG System.

Run with: streamlit run src/app.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

from src.config import settings
from src.chain import RAGChain, ConversationalRAGChain, AdvancedRAGChain
from src.vector_store import VectorStoreManager, load_existing_index
from src.embeddings import get_embedding_model
from src.document_loader import ClimateDocumentLoader
from src.utils import detect_language, format_timestamp


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Climate Science Q&A",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E4D78;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #1E4D78;
        color: #1a1a1a !important;
    }
    .source-card strong {
        color: #1E4D78 !important;
    }
    .source-card small {
        color: #333 !important;
    }
    .metric-card {
        background-color: #e8f4f8;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-align: center;
    }
    .stButton>button {
        background-color: #1E4D78;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chain" not in st.session_state:
        st.session_state.chain = None
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    
    if "language" not in st.session_state:
        st.session_state.language = "auto"
    
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Language selection
        language = st.selectbox(
            "Response Language",
            options=["auto", "en", "vi"],
            format_func=lambda x: {
                "auto": "🔄 Auto-detect",
                "en": "🇬🇧 English",
                "vi": "🇻🇳 Tiếng Việt"
            }[x],
            index=0,
        )
        st.session_state.language = language
        
        # Model selection
        model = st.selectbox(
            "LLM Model",
            options=["gemma3:4b", "gemini-2.0-flash", "claude-3-haiku-20240307", "gpt-4o-mini"],
            index=0,
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            use_reranker = st.checkbox("Enable Reranking", value=True)
            use_memory = st.checkbox("Enable Conversation Memory", value=True)
            top_k = st.slider("Documents to Retrieve", 1, 10, 5)
        
        st.markdown("---")
        
        # Document upload
        st.markdown("## 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or MD files",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            if st.button("📥 Index Documents"):
                index_uploaded_documents(uploaded_files)
        
        st.markdown("---")
        
        # Statistics
        st.markdown("## 📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.total_queries)
        with col2:
            if st.session_state.vector_store:
                try:
                    count = st.session_state.vector_store._collection.count()
                    st.metric("Documents", count)
                except:
                    st.metric("Documents", "N/A")
        
        # Clear chat
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if hasattr(st.session_state.chain, "clear_history"):
                st.session_state.chain.clear_history()
            st.rerun()
        
        st.markdown("---")
        
        # Document Sources
        st.markdown("## 📁 Indexed Sources")
        if st.session_state.vector_store:
            try:
                # Get unique sources from metadata
                collection = st.session_state.vector_store._collection
                results = collection.get(include=["metadatas"])
                sources = set()
                for meta in results.get("metadatas", []):
                    if meta and "source" in meta:
                        # Get just the filename
                        source_name = meta["source"].split("/")[-1].split("\\")[-1]
                        sources.add(source_name)
                
                if sources:
                    for source in sorted(sources):
                        st.markdown(f"📄 {source}")
                else:
                    st.markdown("*No sources found*")
            except Exception as e:
                st.markdown(f"*Unable to load sources*")
        
        st.markdown("---")
        
        # About
        st.markdown("## ℹ️ About")
        st.markdown("""
        **Climate Science Q&A** is a RAG-based system 
        for answering questions about climate change 
        and environmental topics.
        
        Built with:
        - 🦜 LangChain
        - 🗄️ ChromaDB
        - 🤖 OpenAI GPT
        - 🎈 Streamlit
        """)


# =============================================================================
# Document Indexing
# =============================================================================

def index_uploaded_documents(uploaded_files):
    """Index uploaded documents."""
    import tempfile
    with st.spinner("Indexing documents..."):
        try:
            # Save uploaded files temporarily (cross-platform temp dir)
            temp_dir = Path(tempfile.gettempdir()) / "climate_qa_uploads"
            temp_dir.mkdir(exist_ok=True)
            
            for file in uploaded_files:
                file_path = temp_dir / file.name
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            # Load and index
            loader = ClimateDocumentLoader()
            documents = loader.load_and_split(temp_dir)
            
            # Get or create vector store
            embeddings = get_embedding_model(provider="gemini")
            manager = VectorStoreManager(embeddings=embeddings)
            manager.add_documents(documents)
            
            st.session_state.vector_store = manager.vector_store
            
            # Cleanup
            for file in temp_dir.iterdir():
                file.unlink()
            
            st.success(f"✅ Indexed {len(documents)} document chunks!")
            
        except Exception as e:
            st.error(f"Error indexing documents: {e}")
            logger.exception("Document indexing failed")


# =============================================================================
# Chat Interface
# =============================================================================

def render_chat():
    """Render chat interface."""
    # Header
    st.markdown('<p class="main-header">🌍 Climate Science Q&A</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about climate change, environmental science, and sustainability</p>',
        unsafe_allow_html=True
    )
    
    # Initialize chain if needed
    if st.session_state.chain is None:
        try:
            with st.spinner("Loading knowledge base..."):
                embeddings = get_embedding_model(provider="gemini")
                manager = load_existing_index(embeddings=embeddings)
                st.session_state.vector_store = manager.vector_store
                
                st.session_state.chain = AdvancedRAGChain(
                    vector_store=manager.vector_store,
                    enable_memory=True,
                    language=st.session_state.language,
                )
        except Exception as e:
            st.warning(
                "⚠️ No documents indexed yet. Please upload documents in the sidebar "
                "or run the indexing script."
            )
            logger.warning(f"Failed to load index: {e}")
            return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source {i}:</strong> {source.get('metadata', {}).get('source', 'Unknown')}<br>
                            <small>{source.get('content', '')[:200]}...</small>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Check for pending question from sample buttons
    prompt = None
    if st.session_state.pending_question:
        prompt = st.session_state.pending_question
        st.session_state.pending_question = None
    
    # Chat input (or use pending question)
    if prompt is None:
        prompt = st.chat_input("Ask a question about climate science...")
    
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chain.invoke(
                        prompt,
                        return_sources=True,
                    )
                    
                    response = result["answer"]
                    sources = result.get("sources", [])
                    
                    st.markdown(response)
                    
                    # Show sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"""
                                <div class="source-card">
                                    <strong>Source {i}:</strong> {source.get('metadata', {}).get('source', 'Unknown')}<br>
                                    <small>{source.get('content', '')[:200]}...</small>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Update session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources,
                    })
                    st.session_state.total_queries += 1
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.exception("Response generation failed")


# =============================================================================
# Sample Questions
# =============================================================================

def render_sample_questions():
    """Render sample questions section."""
    st.markdown("### 💡 Sample Questions")
    
    sample_questions = [
        "What are the main causes of climate change?",
        "How does deforestation contribute to global warming?",
        "What is the Paris Agreement?",
        "How can individuals reduce their carbon footprint?",
        "What are the effects of climate change on Vietnam?",
    ]
    
    cols = st.columns(len(sample_questions))
    for i, (col, question) in enumerate(zip(cols, sample_questions)):
        with col:
            if st.button(f"Q{i+1}", help=question):
                # Set pending question to be processed
                st.session_state.pending_question = question
                st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Setup LangSmith if configured
    settings.setup_langsmith()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    render_chat()
    
    # Sample questions (only show if no messages)
    if not st.session_state.messages:
        render_sample_questions()


if __name__ == "__main__":
    main()
