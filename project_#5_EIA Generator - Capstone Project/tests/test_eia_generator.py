"""
Tests for EIA Generator.

Run with: pytest tests/test_eia_generator.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ProjectInput,
    ProjectType,
    EIAConfig,
    EIAReport,
    EIASection,
    ImpactCategory,
    PROJECT_REGULATIONS,
    IMPACT_FACTORS_BY_PROJECT,
)
from src.agents.base import AgentState, create_initial_state


# =============================================================================
# Config Tests
# =============================================================================

class TestConfig:
    """Tests for configuration."""
    
    def test_project_input_basic(self):
        """Test basic project input creation."""
        project = ProjectInput(
            name="Test Solar Plant",
            type=ProjectType.ENERGY_SOLAR,
            location="Ninh Thuận",
            area_hectares=100,
        )
        
        assert project.name == "Test Solar Plant"
        assert project.type == ProjectType.ENERGY_SOLAR
        assert project.area_hectares == 100
    
    def test_project_input_with_investment(self):
        """Test project input with investment."""
        project = ProjectInput(
            name="Test Project",
            type=ProjectType.INDUSTRIAL_MANUFACTURING,
            location="Binh Duong",
            area_hectares=50,
            investment_usd=10_000_000,
        )
        
        assert project.investment_usd == 10_000_000
        # VND should be calculated
        assert project.investment_vnd > 0
    
    def test_eia_config_defaults(self):
        """Test EIA config defaults."""
        config = EIAConfig()
        
        assert config.language == "vi"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.4
    
    def test_eia_section(self):
        """Test EIA section creation."""
        section = EIASection(
            id="1",
            title="Mô tả dự án",
            title_en="Project Description",
            content="Test content",
        )
        
        assert section.id == "1"
        assert section.title == "Mô tả dự án"
        assert len(section.subsections) == 0
    
    def test_project_regulations_mapping(self):
        """Test project regulations mapping."""
        regs = PROJECT_REGULATIONS.get(ProjectType.ENERGY_SOLAR, {})
        
        assert "primary" in regs
        assert "technical" in regs
        assert "environmental" in regs
    
    def test_impact_factors_mapping(self):
        """Test impact factors mapping."""
        factors = IMPACT_FACTORS_BY_PROJECT.get(ProjectType.ENERGY_SOLAR, [])
        
        assert len(factors) > 0
        assert ImpactCategory.LAND_USE in factors


# =============================================================================
# Agent State Tests
# =============================================================================

class TestAgentState:
    """Tests for agent state management."""
    
    def test_create_initial_state(self):
        """Test initial state creation."""
        project = ProjectInput(
            name="Test",
            type=ProjectType.ENERGY_SOLAR,
            location="Test Location",
            area_hectares=50,
        )
        config = EIAConfig()
        
        state = create_initial_state(project, config)
        
        assert state["project"]["name"] == "Test"
        assert state["current_step"] == "init"
        assert len(state["sections"]) == 0
        assert len(state["errors"]) == 0
    
    def test_state_has_required_keys(self):
        """Test state has all required keys."""
        project = ProjectInput(
            name="Test",
            type=ProjectType.ENERGY_SOLAR,
            location="Test",
            area_hectares=10,
        )
        config = EIAConfig()
        
        state = create_initial_state(project, config)
        
        required_keys = [
            "project", "config", "messages", "sections",
            "tables", "figures", "regulations", "baseline_data",
            "impact_matrix", "validation_results", "compliance_score",
            "current_step", "errors"
        ]
        
        for key in required_keys:
            assert key in state, f"Missing key: {key}"


# =============================================================================
# Report Tests
# =============================================================================

class TestEIAReport:
    """Tests for EIA report."""
    
    def test_report_creation(self):
        """Test report creation."""
        project = ProjectInput(
            name="Test Project",
            type=ProjectType.ENERGY_SOLAR,
            location="Test Location",
            area_hectares=100,
        )
        
        report = EIAReport(
            project=project,
            generated_at="2024-01-01T00:00:00",
            executive_summary="Test summary",
            compliance_score=85.0,
        )
        
        assert report.project.name == "Test Project"
        assert report.compliance_score == 85.0
    
    def test_report_to_dict(self):
        """Test report serialization."""
        project = ProjectInput(
            name="Test",
            type=ProjectType.ENERGY_SOLAR,
            location="Test",
            area_hectares=50,
        )
        
        report = EIAReport(
            project=project,
            generated_at="2024-01-01",
        )
        
        data = report.to_dict()
        
        assert isinstance(data, dict)
        assert "project" in data
        assert "sections" in data


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests (with mocks)."""
    
    def test_project_type_enum(self):
        """Test all project types are valid."""
        for ptype in ProjectType:
            assert ptype.value is not None
    
    def test_impact_category_enum(self):
        """Test all impact categories are valid."""
        for category in ImpactCategory:
            assert category.value is not None
    
    def test_regulations_for_all_project_types(self):
        """Test regulations exist for common project types."""
        common_types = [
            ProjectType.ENERGY_SOLAR,
            ProjectType.INDUSTRIAL_MANUFACTURING,
        ]
        
        for ptype in common_types:
            regs = PROJECT_REGULATIONS.get(ptype, {})
            assert len(regs) > 0, f"No regulations for {ptype}"


# =============================================================================
# Validation Tests
# =============================================================================

class TestValidation:
    """Tests for validation logic."""
    
    def test_completeness_check_empty(self):
        """Test completeness check with empty sections."""
        sections = {}
        
        required = ["regulations", "baseline_natural", "impact_construction"]
        present = [s for s in required if s in sections]
        
        assert len(present) == 0
    
    def test_completeness_check_partial(self):
        """Test completeness check with partial sections."""
        sections = {
            "regulations": "x" * 200,  # Long enough
            "baseline_natural": "x" * 50,  # Too short
        }
        
        min_length = 100
        valid = [k for k, v in sections.items() if len(v) >= min_length]
        
        assert len(valid) == 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
