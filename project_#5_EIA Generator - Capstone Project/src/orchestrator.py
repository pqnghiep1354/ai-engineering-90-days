"""
EIA Generator Orchestrator - Main workflow coordination.

Uses LangGraph to orchestrate multi-agent workflow for EIA generation.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from loguru import logger

from .config import (
    ProjectInput,
    EIAConfig,
    EIAReport,
    EIASection,
    EIA_SECTIONS,
    get_settings,
)
from .agents.base import AgentState, create_initial_state
from .agents.research_agent import ResearchAgent
from .agents.baseline_agent import BaselineAgent
from .agents.impact_agent import ImpactAgent
from .agents.mitigation_agent import MitigationAgent
from .agents.monitoring_agent import MonitoringAgent
from .agents.validator_agent import ValidatorAgent


# =============================================================================
# Workflow Definition
# =============================================================================

class EIAOrchestrator:
    """
    Main orchestrator for EIA report generation.
    
    Coordinates multiple specialized agents through a LangGraph workflow.
    """
    
    def __init__(self, config: Optional[EIAConfig] = None):
        """Initialize orchestrator with configuration."""
        self.config = config or EIAConfig()
        self.settings = get_settings()
        
        # Initialize agents
        self.agents = {
            "research": ResearchAgent(model=self.config.model),
            "baseline": BaselineAgent(model=self.config.model),
            "impact": ImpactAgent(model=self.config.model),
            "mitigation": MitigationAgent(model=self.config.model),
            "monitoring": MonitoringAgent(model=self.config.model),
            "validator": ValidatorAgent(model=self.config.model),
        }
        
        # Build workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("EIA Orchestrator initialized")
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("research", self._run_research)
        workflow.add_node("baseline", self._run_baseline)
        workflow.add_node("impact", self._run_impact)
        workflow.add_node("mitigation", self._run_mitigation)
        workflow.add_node("monitoring", self._run_monitoring)
        workflow.add_node("validate", self._run_validation)
        workflow.add_node("compile", self._compile_report)
        
        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "baseline")
        workflow.add_edge("baseline", "impact")
        workflow.add_edge("impact", "mitigation")
        workflow.add_edge("mitigation", "monitoring")
        workflow.add_edge("monitoring", "validate")
        workflow.add_edge("validate", "compile")
        workflow.add_edge("compile", END)
        
        return workflow.compile()
    
    async def _run_research(self, state: AgentState) -> AgentState:
        """Run research agent."""
        logger.info("Step 1/7: Research regulations")
        return await self.agents["research"].execute(state)
    
    async def _run_baseline(self, state: AgentState) -> AgentState:
        """Run baseline agent."""
        logger.info("Step 2/7: Generate baseline assessment")
        return await self.agents["baseline"].execute(state)
    
    async def _run_impact(self, state: AgentState) -> AgentState:
        """Run impact agent."""
        logger.info("Step 3/7: Assess environmental impacts")
        return await self.agents["impact"].execute(state)
    
    async def _run_mitigation(self, state: AgentState) -> AgentState:
        """Run mitigation agent."""
        logger.info("Step 4/7: Propose mitigation measures")
        return await self.agents["mitigation"].execute(state)
    
    async def _run_monitoring(self, state: AgentState) -> AgentState:
        """Run monitoring agent."""
        logger.info("Step 5/7: Design monitoring program")
        return await self.agents["monitoring"].execute(state)
    
    async def _run_validation(self, state: AgentState) -> AgentState:
        """Run validation agent."""
        logger.info("Step 6/7: Validate report")
        return await self.agents["validator"].execute(state)
    
    async def _compile_report(self, state: AgentState) -> AgentState:
        """Compile final report."""
        logger.info("Step 7/7: Compile final report")
        
        # Generate executive summary
        summary = await self._generate_executive_summary(state)
        state["sections"]["executive_summary"] = summary
        
        state["current_step"] = "complete"
        return state
    
    async def _generate_executive_summary(self, state: AgentState) -> str:
        """Generate executive summary from all sections."""
        project = state["project"]
        validation = state.get("validation_results", {})
        
        summary = f"""
# TÓM TẮT BÁO CÁO ĐÁNH GIÁ TÁC ĐỘNG MÔI TRƯỜNG

## 1. THÔNG TIN DỰ ÁN

- **Tên dự án:** {project.get('name', 'N/A')}
- **Chủ đầu tư:** {project.get('investor_name', 'N/A')}
- **Vị trí:** {project.get('location', 'N/A')}
- **Quy mô:** {project.get('area_hectares', 0)} ha
- **Tổng vốn đầu tư:** {project.get('investment_usd', 0):,.0f} USD

## 2. CÁC TÁC ĐỘNG CHÍNH

### Giai đoạn xây dựng:
- Tác động đến chất lượng không khí do bụi, khí thải
- Tác động tiếng ồn từ máy móc thi công
- Tác động đến giao thông khu vực

### Giai đoạn vận hành:
- Tác động từ khí thải quá trình sản xuất
- Tác động từ nước thải
- Tác động từ chất thải rắn

## 3. BIỆN PHÁP GIẢM THIỂU

- Lắp đặt hệ thống xử lý khí thải đạt QCVN 19:2009/BTNMT
- Xây dựng hệ thống xử lý nước thải đạt QCVN 40:2011/BTNMT
- Thực hiện phân loại và xử lý chất thải rắn theo quy định

## 4. CHƯƠNG TRÌNH GIÁM SÁT

- Giám sát môi trường không khí: 3 tháng/lần
- Giám sát nước thải: 3 tháng/lần
- Giám sát tiếng ồn: 6 tháng/lần

## 5. KẾT LUẬN

Dự án đã được đánh giá tác động môi trường theo quy định của Luật Bảo vệ 
môi trường 2020 và Nghị định 08/2022/NĐ-CP. Với các biện pháp giảm thiểu 
và chương trình giám sát đề xuất, dự án có thể được triển khai đảm bảo 
các yêu cầu về bảo vệ môi trường.

**Điểm đánh giá chất lượng báo cáo:** {validation.get('overall_score', 0):.1f}/100
"""
        return summary.strip()
    
    async def generate(
        self,
        project: ProjectInput,
        progress_callback: Optional[callable] = None,
    ) -> EIAReport:
        """
        Generate complete EIA report for a project.
        
        Args:
            project: Project input data
            progress_callback: Optional callback for progress updates
            
        Returns:
            Complete EIA report
        """
        logger.info(f"Starting EIA generation for: {project.name}")
        
        # Create initial state
        state = create_initial_state(project, self.config)
        
        # Run workflow
        try:
            final_state = await self.workflow.ainvoke(state)
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            raise
        
        # Create report
        report = self._create_report(project, final_state)
        
        logger.info(f"EIA generation complete. Score: {report.compliance_score:.1f}")
        return report
    
    def _create_report(
        self,
        project: ProjectInput,
        state: AgentState,
    ) -> EIAReport:
        """Create EIAReport from final state."""
        
        sections = []
        
        # Map generated content to EIA sections
        section_mapping = {
            "1": ["regulations"],
            "2": ["baseline_natural", "baseline_socio", "baseline_env"],
            "3": ["impact_preparation", "impact_construction", "impact_operation", "impact_decommission"],
            "4": ["mitigation_preparation", "mitigation_construction", "mitigation_operation"],
            "5": ["management", "monitoring", "emergency_response"],
            "6": ["consultation"],
        }
        
        for section_def in EIA_SECTIONS:
            section_id = section_def["id"]
            content_keys = section_mapping.get(section_id, [])
            
            content_parts = []
            for key in content_keys:
                if key in state["sections"]:
                    content_parts.append(state["sections"][key])
            
            section = EIASection(
                id=section_id,
                title=section_def["title"],
                title_en=section_def["title_en"],
                content="\n\n".join(content_parts),
                subsections=[
                    EIASection(
                        id=sub["id"],
                        title=sub["title"],
                        title_en=sub["title_en"],
                    )
                    for sub in section_def.get("subsections", [])
                ],
            )
            sections.append(section)
        
        validation = state.get("validation_results", {})
        
        return EIAReport(
            project=project,
            generated_at=datetime.now().isoformat(),
            executive_summary=state["sections"].get("executive_summary", ""),
            sections=sections,
            compliance_score=validation.get("overall_score", 0),
            completeness_score=validation.get("completeness", {}).get("score", 0),
            validation_notes=validation.get("recommendations", []),
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def generate_eia(
    project: ProjectInput,
    config: Optional[EIAConfig] = None,
) -> EIAReport:
    """
    Convenience function to generate EIA report.
    
    Args:
        project: Project input data
        config: Optional configuration
        
    Returns:
        Generated EIA report
    """
    orchestrator = EIAOrchestrator(config)
    return await orchestrator.generate(project)


def generate_eia_sync(
    project: ProjectInput,
    config: Optional[EIAConfig] = None,
) -> EIAReport:
    """Synchronous wrapper for generate_eia."""
    return asyncio.run(generate_eia(project, config))
