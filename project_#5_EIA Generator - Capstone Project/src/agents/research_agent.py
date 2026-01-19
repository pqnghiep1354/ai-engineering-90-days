"""
Research Agent - Retrieves regulations, standards, and reference materials.
"""

from typing import Any, Dict, List

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import PROJECT_REGULATIONS, ProjectType


class ResearchAgent(BaseAgent):
    """Agent responsible for researching regulations and standards."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        super().__init__(
            name="research",
            description="Research relevant regulations, standards, and guidelines for EIA",
            model=model,
            temperature=temperature,
        )
    
    def _build_system_prompt(self) -> str:
        return """Bạn là chuyên gia về pháp luật môi trường Việt Nam và quốc tế.

NHIỆM VỤ:
- Xác định các văn bản pháp luật áp dụng cho dự án
- Liệt kê các quy chuẩn kỹ thuật môi trường (QCVN) liên quan
- Xác định các tiêu chuẩn quốc tế áp dụng (IFC, World Bank)
- Tổng hợp các yêu cầu cần tuân thủ

KIẾN THỨC CHÍNH:
1. Luật Bảo vệ môi trường 2020 (Luật số 72/2020/QH14)
2. Nghị định 08/2022/NĐ-CP về đánh giá tác động môi trường
3. Các QCVN về môi trường (không khí, nước thải, tiếng ồn...)
4. IFC Environmental and Social Performance Standards
5. World Bank Environmental and Social Framework

OUTPUT: Trả về danh sách có cấu trúc các quy định áp dụng."""
    
    async def execute(self, state: AgentState) -> AgentState:
        """Research regulations for the project."""
        logger.info(f"Research Agent: Researching regulations for project")
        
        project = state["project"]
        project_type = ProjectType(project.get("type", "industrial_manufacturing"))
        
        # Get predefined regulations
        regs = PROJECT_REGULATIONS.get(project_type, {})
        
        # Generate comprehensive regulation analysis
        prompt = f"""Phân tích các quy định pháp luật áp dụng cho dự án sau:

{self._format_project_context(project)}

Quy định sơ bộ đã xác định:
- Quy định chính: {', '.join(regs.get('primary', []))}
- Quy định kỹ thuật: {', '.join(regs.get('technical', []))}
- Quy định môi trường: {', '.join(regs.get('environmental', []))}

YÊU CẦU:
1. Xác nhận và bổ sung các văn bản pháp luật áp dụng
2. Liệt kê các điều khoản cụ thể cần tuân thủ
3. Xác định ngưỡng/giới hạn cho phép theo QCVN
4. Ghi chú các yêu cầu đặc biệt (nếu có)

Trả lời bằng tiếng Việt, có cấu trúc rõ ràng."""

        research_result = await self._generate(prompt)
        
        # Update state
        state["regulations"] = [
            {"category": "primary", "items": regs.get("primary", [])},
            {"category": "technical", "items": regs.get("technical", [])},
            {"category": "environmental", "items": regs.get("environmental", [])},
        ]
        state["sections"]["regulations"] = research_result
        state["current_step"] = "research_complete"
        
        logger.info("Research Agent: Completed regulation research")
        return state
    
    def get_emission_standards(self, project_type: ProjectType) -> Dict[str, Any]:
        """Get emission standards for project type."""
        standards = {
            "air": {
                "dust": {"limit": 200, "unit": "mg/Nm³", "standard": "QCVN 19:2009/BTNMT"},
                "CO": {"limit": 1000, "unit": "mg/Nm³", "standard": "QCVN 19:2009/BTNMT"},
                "SO2": {"limit": 500, "unit": "mg/Nm³", "standard": "QCVN 19:2009/BTNMT"},
                "NOx": {"limit": 850, "unit": "mg/Nm³", "standard": "QCVN 19:2009/BTNMT"},
            },
            "wastewater": {
                "BOD5": {"limit": 50, "unit": "mg/L", "standard": "QCVN 40:2011/BTNMT Cột B"},
                "COD": {"limit": 150, "unit": "mg/L", "standard": "QCVN 40:2011/BTNMT Cột B"},
                "TSS": {"limit": 100, "unit": "mg/L", "standard": "QCVN 40:2011/BTNMT Cột B"},
                "pH": {"limit": "5.5-9", "unit": "-", "standard": "QCVN 40:2011/BTNMT"},
            },
            "noise": {
                "day_industrial": {"limit": 70, "unit": "dBA", "standard": "QCVN 26:2010/BTNMT"},
                "night_industrial": {"limit": 55, "unit": "dBA", "standard": "QCVN 26:2010/BTNMT"},
                "day_residential": {"limit": 55, "unit": "dBA", "standard": "QCVN 26:2010/BTNMT"},
                "night_residential": {"limit": 45, "unit": "dBA", "standard": "QCVN 26:2010/BTNMT"},
            },
        }
        return standards
