"""
Mitigation Agent - Generates mitigation measures for identified impacts.
"""

from typing import Any, Dict, List

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import ImpactCategory, ProjectType


class MitigationAgent(BaseAgent):
    """Agent responsible for proposing mitigation measures."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.4):
        super().__init__(
            name="mitigation",
            description="Generate mitigation measures for environmental impacts",
            model=model,
            temperature=temperature,
        )
    
    def _build_system_prompt(self) -> str:
        return """Bạn là chuyên gia về biện pháp giảm thiểu tác động môi trường (Mitigation Expert).

NHIỆM VỤ: Xây dựng mục "BIỆN PHÁP PHÒNG NGỪA, GIẢM THIỂU TÁC ĐỘNG" trong báo cáo ĐTM.

NGUYÊN TẮC GIẢM THIỂU (Mitigation Hierarchy):
1. TRÁNH (Avoid) - Thay đổi thiết kế để tránh tác động
2. GIẢM THIỂU (Minimize) - Giảm mức độ tác động
3. PHỤC HỒI (Restore) - Khôi phục môi trường bị ảnh hưởng  
4. BÙ ĐẮP (Offset) - Bồi thường tác động không thể tránh

LOẠI BIỆN PHÁP:
1. Biện pháp kỹ thuật (công nghệ, thiết bị xử lý)
2. Biện pháp quản lý (quy trình, đào tạo)
3. Biện pháp sinh học (trồng cây, tạo vùng đệm)
4. Biện pháp kinh tế (bồi thường, hỗ trợ)

YÊU CẦU:
- Biện pháp phải khả thi về kỹ thuật và kinh tế
- Tuân thủ QCVN và tiêu chuẩn hiện hành
- Ước tính chi phí thực hiện
- Xác định trách nhiệm thực hiện
- Đề xuất chỉ tiêu đánh giá hiệu quả"""
    
    async def execute(self, state: AgentState) -> AgentState:
        """Generate mitigation measures."""
        logger.info("Mitigation Agent: Generating mitigation measures")
        
        project = state["project"]
        impact_matrix = state.get("impact_matrix", {})
        
        # Generate mitigation for each phase
        phases = [
            ("preparation", "Giai đoạn chuẩn bị"),
            ("construction", "Giai đoạn xây dựng"),
            ("operation", "Giai đoạn vận hành"),
        ]
        
        mitigation_sections = {}
        
        for phase_key, phase_name in phases:
            phase_impacts = impact_matrix.get(phase_key, {})
            
            prompt = f"""Đề xuất biện pháp giảm thiểu tác động môi trường {phase_name}:

{self._format_project_context(project)}

Các tác động đã xác định trong giai đoạn này:
{self._format_impacts(phase_impacts)}

YÊU CẦU:
1. Đề xuất biện pháp cụ thể cho từng loại tác động
2. Mô tả chi tiết kỹ thuật/thiết bị sử dụng
3. Ước tính chi phí thực hiện
4. Xác định đơn vị chịu trách nhiệm
5. Đề xuất chỉ tiêu đánh giá hiệu quả

Viết theo format:

## [Loại tác động]
### Biện pháp 1: [Tên biện pháp]
- Mô tả: ...
- Thiết bị/công nghệ: ...
- Chi phí ước tính: ...
- Đơn vị thực hiện: ...
- Hiệu quả dự kiến: ...

Viết bằng tiếng Việt."""

            content = await self._generate(prompt)
            mitigation_sections[phase_key] = content
        
        # Generate cost summary
        cost_summary = await self._generate_cost_summary(project, mitigation_sections)
        
        # Update state
        state["sections"]["mitigation_preparation"] = mitigation_sections.get("preparation", "")
        state["sections"]["mitigation_construction"] = mitigation_sections.get("construction", "")
        state["sections"]["mitigation_operation"] = mitigation_sections.get("operation", "")
        state["sections"]["mitigation_cost"] = cost_summary
        
        state["current_step"] = "mitigation_complete"
        
        logger.info("Mitigation Agent: Completed mitigation measures")
        return state
    
    def _format_impacts(self, impacts: Dict[str, Dict]) -> str:
        """Format impacts for prompt."""
        if not impacts:
            return "Chưa có thông tin cụ thể"
        
        lines = []
        for factor, details in impacts.items():
            level = details.get("level", "unknown")
            impact_type = details.get("type", "negative")
            
            level_vn = {
                "large": "Lớn",
                "medium": "Trung bình",
                "small": "Nhỏ",
            }.get(level, level)
            
            type_vn = {
                "positive": "Tích cực",
                "negative": "Tiêu cực",
                "mixed": "Hỗn hợp",
            }.get(impact_type, impact_type)
            
            lines.append(f"- {factor}: Mức độ {level_vn}, Loại {type_vn}")
        
        return "\n".join(lines)
    
    async def _generate_cost_summary(
        self,
        project: Dict[str, Any],
        mitigation_sections: Dict[str, str],
    ) -> str:
        """Generate cost summary for mitigation measures."""
        prompt = f"""Tổng hợp chi phí thực hiện các biện pháp bảo vệ môi trường:

DỰ ÁN: {project.get('name', 'N/A')}
TỔNG VỐN ĐẦU TƯ: {project.get('investment_usd', 0):,.0f} USD

Lập bảng tổng hợp chi phí theo format:

| STT | Hạng mục | Giai đoạn | Chi phí (USD) | Tỷ lệ/Tổng ĐT |
|-----|----------|-----------|---------------|---------------|
| 1   | Xử lý khí thải | Vận hành | ... | ...% |
| 2   | Xử lý nước thải | Vận hành | ... | ...% |
| ... | ... | ... | ... | ... |
| | TỔNG CỘNG | | ... | ...% |

Ghi chú: Chi phí BVMT thông thường chiếm 1-5% tổng vốn đầu tư.

Viết bằng tiếng Việt."""

        return await self._generate(prompt)
    
    def get_standard_measures(self, impact_category: ImpactCategory) -> List[Dict[str, str]]:
        """Get standard mitigation measures by impact category."""
        measures = {
            ImpactCategory.AIR_QUALITY: [
                {
                    "name": "Hệ thống lọc bụi túi vải",
                    "description": "Lắp đặt hệ thống lọc bụi công suất phù hợp",
                    "efficiency": "95-99%",
                },
                {
                    "name": "Phun nước dập bụi",
                    "description": "Phun nước định kỳ tại khu vực phát sinh bụi",
                    "efficiency": "60-80%",
                },
            ],
            ImpactCategory.WATER_QUALITY: [
                {
                    "name": "Hệ thống xử lý nước thải",
                    "description": "XLNT bằng công nghệ sinh học + hóa lý",
                    "efficiency": "Đạt QCVN 40:2011 Cột B",
                },
            ],
            ImpactCategory.NOISE_VIBRATION: [
                {
                    "name": "Tường cách âm",
                    "description": "Xây dựng tường/barrier cách âm",
                    "efficiency": "10-15 dBA",
                },
                {
                    "name": "Trồng cây xanh",
                    "description": "Trồng đai cây xanh cách ly",
                    "efficiency": "3-5 dBA",
                },
            ],
            ImpactCategory.WASTE: [
                {
                    "name": "Phân loại chất thải tại nguồn",
                    "description": "Bố trí thùng phân loại rác",
                    "efficiency": "Tái chế 50-70%",
                },
            ],
        }
        
        return measures.get(impact_category, [])
