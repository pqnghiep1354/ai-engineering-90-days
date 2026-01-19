"""
Tests for agent functionality.

Run with: pytest tests/test_agents.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.base import (
    AgentRole,
    AgentState,
    AgentResponse,
    Source,
    Finding,
    create_initial_state,
)
from src.tools.citation import Citation, CitationManager


# =============================================================================
# Base Agent Tests
# =============================================================================

class TestAgentState:
    """Tests for AgentState."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        state = create_initial_state("Climate change")
        
        assert state["topic"] == "Climate change"
        assert state["sources"] == []
        assert state["findings"] == []
        assert state["iteration"] == 0
        assert state["status"] == "pending"
    
    def test_state_has_required_fields(self):
        """Test state has all required fields."""
        state = create_initial_state("Test topic")
        
        required_fields = [
            "topic", "messages", "sources", "findings",
            "analysis", "draft", "final_report",
            "fact_check_results", "current_agent",
            "iteration", "status", "errors"
        ]
        
        for field in required_fields:
            assert field in state


class TestSource:
    """Tests for Source class."""
    
    def test_source_creation(self):
        """Test source creation."""
        source = Source(
            title="Test Title",
            url="https://example.com",
            content="Test content",
            snippet="Test snippet",
            domain="example.com",
            relevance_score=0.9,
        )
        
        assert source.title == "Test Title"
        assert source.url == "https://example.com"
        assert source.relevance_score == 0.9
    
    def test_source_to_dict(self):
        """Test source serialization."""
        source = Source(
            title="Test",
            url="https://test.com",
            content="Content",
        )
        
        data = source.to_dict()
        
        assert "title" in data
        assert "url" in data
        assert data["title"] == "Test"


class TestFinding:
    """Tests for Finding class."""
    
    def test_finding_creation(self):
        """Test finding creation."""
        finding = Finding(
            content="Test finding",
            confidence=0.8,
            category="test",
        )
        
        assert finding.content == "Test finding"
        assert finding.confidence == 0.8
    
    def test_finding_with_sources(self):
        """Test finding with sources."""
        source = Source(title="S1", url="https://s1.com", content="")
        finding = Finding(
            content="Finding with source",
            sources=[source],
        )
        
        assert len(finding.sources) == 1


class TestAgentResponse:
    """Tests for AgentResponse."""
    
    def test_response_success(self):
        """Test successful response."""
        response = AgentResponse(
            agent=AgentRole.RESEARCHER,
            content="Test content",
            success=True,
        )
        
        assert response.success
        assert response.error is None
    
    def test_response_failure(self):
        """Test failed response."""
        response = AgentResponse(
            agent=AgentRole.RESEARCHER,
            content="",
            success=False,
            error="Test error",
        )
        
        assert not response.success
        assert response.error == "Test error"


# =============================================================================
# Citation Manager Tests
# =============================================================================

class TestCitationManager:
    """Tests for CitationManager."""
    
    def test_add_citation(self):
        """Test adding citation."""
        manager = CitationManager()
        citation = manager.add_citation(
            title="Test Article",
            url="https://example.com/article",
        )
        
        assert citation.id == 1
        assert citation.title == "Test Article"
    
    def test_no_duplicate_citations(self):
        """Test duplicate prevention."""
        manager = CitationManager()
        
        c1 = manager.add_citation("Title", "https://example.com")
        c2 = manager.add_citation("Title", "https://example.com")
        
        assert c1.id == c2.id
        assert len(manager.get_all_citations()) == 1
    
    def test_citation_formatting(self):
        """Test citation formatting."""
        citation = Citation(
            id=1,
            title="Climate Report",
            url="https://ipcc.ch/report",
            publisher="IPCC",
        )
        
        apa = citation.to_apa()
        assert "Climate Report" in apa
        
        md = citation.to_markdown()
        assert "[1]" in md
    
    def test_format_reference_list(self):
        """Test reference list formatting."""
        manager = CitationManager()
        manager.add_citation("Article 1", "https://a.com")
        manager.add_citation("Article 2", "https://b.com")
        
        refs = manager.format_reference_list(style="markdown")
        
        assert "[1]" in refs
        assert "[2]" in refs


# =============================================================================
# Workflow Tests
# =============================================================================

class TestWorkflows:
    """Tests for workflows."""
    
    def test_workflow_result_creation(self):
        """Test workflow result creation."""
        from src.workflows.base import WorkflowResult
        
        result = WorkflowResult(
            topic="Test topic",
            report="# Report",
            sources=[],
            findings=[],
            success=True,
        )
        
        assert result.success
        assert result.topic == "Test topic"
    
    def test_workflow_result_to_dict(self):
        """Test workflow result serialization."""
        from src.workflows.base import WorkflowResult
        
        result = WorkflowResult(
            topic="Test",
            report="Report",
            sources=[{"title": "S1"}],
            findings=[{"content": "F1"}],
        )
        
        data = result.to_dict()
        
        assert "topic" in data
        assert "sources" in data
        assert len(data["sources"]) == 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
