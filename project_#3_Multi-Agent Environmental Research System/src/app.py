"""
Streamlit Web Application for Multi-Agent Research System.

Run with: streamlit run src/app.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from loguru import logger

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.orchestrator import ResearchOrchestrator


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="AI Research System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .status-error {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .agent-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = ResearchOrchestrator()
    
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    
    if "is_researching" not in st.session_state:
        st.session_state.is_researching = False


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar with settings."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Workflow selection
        workflow = st.selectbox(
            "Research Workflow",
            options=["quick", "deep"],
            index=0,
            help="Quick: 2-5 min, Deep: 10-20 min with fact-checking",
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            max_sources = st.slider(
                "Max Sources",
                min_value=5,
                max_value=30,
                value=10 if workflow == "quick" else 20,
            )
            
            enable_fact_check = st.checkbox(
                "Enable Fact-Checking",
                value=workflow == "deep",
            )
            
            report_format = st.selectbox(
                "Report Format",
                options=["full", "executive", "brief"],
                index=0 if workflow == "deep" else 2,
            )
        
        st.markdown("---")
        
        # API Status
        st.markdown("## 🔑 API Status")
        
        if settings.has_openai_key:
            st.success("✓ OpenAI API configured")
        else:
            st.error("✗ OpenAI API key missing")
        
        if settings.has_tavily_key:
            st.success("✓ Tavily API configured")
        else:
            st.warning("⚠ Tavily API key missing (mock search)")
        
        st.markdown("---")
        
        # Research History
        st.markdown("## 📜 History")
        
        history = st.session_state.orchestrator.get_history()
        if history:
            for i, h in enumerate(history[-5:], 1):
                icon = "✓" if h["success"] else "✗"
                st.text(f"{icon} {h['topic'][:30]}...")
        else:
            st.text("No research yet")
        
        return {
            "workflow": workflow,
            "max_sources": max_sources,
            "enable_fact_check": enable_fact_check,
            "report_format": report_format,
        }


# =============================================================================
# Main Content
# =============================================================================

def render_main():
    """Render main content."""
    
    # Header
    st.markdown(
        '<p class="main-header">🤖 Multi-Agent Environmental Research</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">AI-powered research with specialized agents</p>',
        unsafe_allow_html=True
    )
    
    # Research input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        topic = st.text_input(
            "Research Topic",
            placeholder="Enter your research topic (e.g., 'Climate change impacts in Vietnam')",
            label_visibility="collapsed",
        )
    
    with col2:
        research_clicked = st.button(
            "🔍 Research",
            use_container_width=True,
            disabled=st.session_state.is_researching,
        )
    
    # Sample topics
    st.markdown("**💡 Sample topics:**")
    sample_topics = [
        "Climate change impacts in Southeast Asia",
        "ESG investing trends 2024",
        "Renewable energy transition challenges",
        "Carbon capture technologies",
    ]
    
    cols = st.columns(len(sample_topics))
    for col, sample in zip(cols, sample_topics):
        with col:
            if st.button(sample[:25] + "...", key=f"sample_{sample[:10]}", use_container_width=True):
                topic = sample
                research_clicked = True
    
    return topic, research_clicked


async def run_research_async(topic: str, settings_dict: dict):
    """Run research asynchronously."""
    orchestrator = st.session_state.orchestrator
    
    result = await orchestrator.research(
        topic=topic,
        workflow=settings_dict["workflow"],
        max_sources=settings_dict["max_sources"],
        enable_fact_checking=settings_dict["enable_fact_check"],
        report_format=settings_dict["report_format"],
    )
    
    return result


def render_results():
    """Render research results."""
    result = st.session_state.current_result
    
    if result is None:
        return
    
    st.markdown("---")
    
    if result.success:
        # Success metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sources", len(result.sources))
        with col2:
            st.metric("Findings", len(result.findings))
        with col3:
            st.metric("Time", f"{result.execution_time_seconds:.1f}s")
        with col4:
            st.metric("Status", "✓ Complete")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 Report", "📚 Sources", "🔍 Findings"])
        
        with tab1:
            st.markdown(result.report)
            
            # Download button
            st.download_button(
                "📥 Download Report",
                data=result.report,
                file_name=f"research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
        
        with tab2:
            if result.sources:
                for i, source in enumerate(result.sources, 1):
                    with st.expander(f"{i}. {source.get('title', 'Unknown')[:60]}..."):
                        st.write(f"**URL:** {source.get('url', 'N/A')}")
                        st.write(f"**Domain:** {source.get('domain', 'N/A')}")
                        st.write(f"**Snippet:** {source.get('snippet', 'N/A')[:300]}...")
            else:
                st.info("No sources collected")
        
        with tab3:
            if result.findings:
                for finding in result.findings:
                    confidence = finding.get("confidence", 0.5)
                    st.markdown(f"""
                    <div class="agent-card">
                        <strong>Finding:</strong> {finding.get('content', 'N/A')}<br>
                        <small>Confidence: {confidence:.0%}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No specific findings extracted")
    
    else:
        st.error(f"Research failed: {result.error}")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application."""
    init_session_state()
    
    # Render sidebar and get settings
    settings_dict = render_sidebar()
    
    # Render main content
    topic, research_clicked = render_main()
    
    # Handle research
    if research_clicked and topic:
        st.session_state.is_researching = True
        
        with st.spinner(f"🔍 Researching: {topic}..."):
            # Run async research
            result = asyncio.run(run_research_async(topic, settings_dict))
            st.session_state.current_result = result
            st.session_state.research_history.append({
                "topic": topic,
                "success": result.success,
                "timestamp": datetime.now(),
            })
        
        st.session_state.is_researching = False
        st.rerun()
    
    # Render results
    render_results()
    
    # Agent information
    with st.expander("ℹ️ About the Agents"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🔍 Research Agent**
            - Searches web for information
            - Extracts key facts from sources
            - Prioritizes authoritative sources
            
            **📊 Analysis Agent**
            - Analyzes collected information
            - Identifies patterns and trends
            - Draws evidence-based conclusions
            """)
        
        with col2:
            st.markdown("""
            **✍️ Writer Agent**
            - Structures content logically
            - Writes clear, informative prose
            - Formats reports professionally
            
            **✓ Fact-Checker Agent**
            - Verifies claims against sources
            - Cross-references information
            - Flags uncertain claims
            """)


if __name__ == "__main__":
    main()
