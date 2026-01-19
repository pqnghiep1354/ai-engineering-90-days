"""
Impact Agent - Generates environmental impact assessment sections.
"""

from typing import Any, Dict, List

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import ImpactCategory, IMPACT_FACTORS_BY_PROJECT, ProjectType


class ImpactAgent(BaseAgent):
    """Agent responsible for environmental impact assessment."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.4):
        super().__init__(
            name="impact",
            description="Generate environmental impact assessment for all project phases",
            model=model,
            temperature=temperature,
        )
    
    def _build_system_prompt(self) -> str:
        return """Bạn là chuyên gia đánh giá tác động môi trường (Environmental Impact Assessment Expert).

NHIỆM VỤ: Xây dựng mục "ĐÁNH GIÁ TÁC ĐỘNG MÔI TRƯỜNG" trong báo cáo ĐTM.

PHƯƠNG PHÁP ĐÁNH GIÁ:
1. Ma trận tác động (Impact Matrix)
2. Phương pháp liệt kê (Checklist)
3. Phương pháp chuyên gia (Expert Judgment)
4. Mô hình hóa (nếu có số liệu)

CÁC GIAI ĐOẠN ĐÁNH GIÁ:
1. Giai đoạn chuẩn bị (giải phóng mặt bằng)
2. Giai đoạn xây dựng
3. Giai đoạn vận hành
4. Giai đoạn kết thúc dự án

CÁC YẾU TỐ MÔI TRƯỜNG:
- Môi trường không khí
- Môi trường nước
- Môi trường đất
- Tiếng ồn, độ rung
- Chất thải rắn
- Hệ sinh thái
- Kinh tế - xã hội

YÊU CẦU:
- Định lượng tác động khi có thể
- Phân loại mức độ tác động (không đáng kể, nhỏ, trung bình, lớn)
- Xác định tác động tích cực và tiêu cực
- Đề xuất biện pháp giảm thiểu sơ bộ"""
    
    async def execute(self, state: AgentState) -> AgentState:
        """Generate impact assessment sections."""
        logger.info("Impact Agent: Generating impact assessment")
        
        project = state["project"]
        project_type = ProjectType(project.get("type", "industrial_manufacturing"))
        
        # Get relevant impact factors
        impact_factors = IMPACT_FACTORS_BY_PROJECT.get(
            project_type, 
            list(ImpactCategory)[:5]
        )
        
        # Generate impact assessment for each phase
        phases = [
            ("preparation", "Giai đoạn chuẩn bị và giải phóng mặt bằng"),
            ("construction", "Giai đoạn xây dựng"),
            ("operation", "Giai đoạn vận hành"),
            ("decommission", "Giai đoạn kết thúc dự án"),
        ]
        
        impact_sections = {}
        impact_matrix = {}
        
        for phase_key, phase_name in phases:
            prompt = f"""Đánh giá tác động môi trường {phase_name} của dự án:

{self._format_project_context(project)}

Các yếu tố môi trường cần đánh giá:
{self._format_impact_factors(impact_factors)}

YÊU CẦU:
1. Xác định nguồn gây tác động chính
2. Đánh giá mức độ và phạm vi tác động
3. Phân loại: Tích cực (+) / Tiêu cực (-)
4. Mức độ: Không đáng kể / Nhỏ / Trung bình / Lớn
5. Lập bảng ma trận tác động

Viết chi tiết bằng tiếng Việt."""

            content = await self._generate(prompt)
            impact_sections[phase_key] = content
            
            # Build impact matrix
            impact_matrix[phase_key] = self._build_impact_matrix(
                phase_key, impact_factors, project
            )
        
        # Update state
        state["sections"]["impact_preparation"] = impact_sections.get("preparation", "")
        state["sections"]["impact_construction"] = impact_sections.get("construction", "")
        state["sections"]["impact_operation"] = impact_sections.get("operation", "")
        state["sections"]["impact_decommission"] = impact_sections.get("decommission", "")
        
        state["impact_matrix"] = impact_matrix
        state["current_step"] = "impact_complete"
        
        logger.info("Impact Agent: Completed impact assessment")
        return state
    
    def _format_impact_factors(self, factors: List[ImpactCategory]) -> str:
        """Format impact factors for prompt."""
        factor_names = {
            ImpactCategory.AIR_QUALITY: "Chất lượng không khí",
            ImpactCategory.WATER_QUALITY: "Chất lượng nước",
            ImpactCategory.SOIL: "Môi trường đất",
            ImpactCategory.NOISE_VIBRATION: "Tiếng ồn và độ rung",
            ImpactCategory.BIODIVERSITY: "Đa dạng sinh học",
            ImpactCategory.LAND_USE: "Sử dụng đất",
            ImpactCategory.VISUAL_LANDSCAPE: "Cảnh quan",
            ImpactCategory.SOCIOECONOMIC: "Kinh tế - xã hội",
            ImpactCategory.WASTE: "Chất thải",
            ImpactCategory.TRAFFIC: "Giao thông",
            ImpactCategory.HEALTH_SAFETY: "Sức khỏe và an toàn",
            ImpactCategory.CLIMATE: "Biến đổi khí hậu",
        }
        
        lines = []
        for i, factor in enumerate(factors, 1):
            name = factor_names.get(factor, factor.value)
            lines.append(f"{i}. {name}")
        
        return "\n".join(lines)
    
    def _build_impact_matrix(
        self,
        phase: str,
        factors: List[ImpactCategory],
        project: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Build impact matrix for a phase."""
        
        # Default impact levels by phase and factor
        impact_levels = {
            "preparation": {
                ImpactCategory.LAND_USE: {"level": "large", "type": "negative"},
                ImpactCategory.SOIL: {"level": "medium", "type": "negative"},
                ImpactCategory.BIODIVERSITY: {"level": "medium", "type": "negative"},
                ImpactCategory.SOCIOECONOMIC: {"level": "medium", "type": "mixed"},
            },
            "construction": {
                ImpactCategory.AIR_QUALITY: {"level": "medium", "type": "negative"},
                ImpactCategory.NOISE_VIBRATION: {"level": "large", "type": "negative"},
                ImpactCategory.TRAFFIC: {"level": "medium", "type": "negative"},
                ImpactCategory.WASTE: {"level": "medium", "type": "negative"},
            },
            "operation": {
                ImpactCategory.AIR_QUALITY: {"level": "small", "type": "negative"},
                ImpactCategory.WATER_QUALITY: {"level": "medium", "type": "negative"},
                ImpactCategory.SOCIOECONOMIC: {"level": "large", "type": "positive"},
                ImpactCategory.CLIMATE: {"level": "medium", "type": "positive"},
            },
            "decommission": {
                ImpactCategory.WASTE: {"level": "medium", "type": "negative"},
                ImpactCategory.LAND_USE: {"level": "medium", "type": "positive"},
            },
        }
        
        matrix = {}
        phase_impacts = impact_levels.get(phase, {})
        
        for factor in factors:
            if factor in phase_impacts:
                matrix[factor.value] = phase_impacts[factor]
            else:
                matrix[factor.value] = {"level": "small", "type": "negative"}
        
        return matrix
