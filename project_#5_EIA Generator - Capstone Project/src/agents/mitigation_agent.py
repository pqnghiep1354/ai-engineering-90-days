"""
Mitigation Agent - Generates mitigation measures for identified impacts.
"""

from typing import Any, Dict, List

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import ImpactCategory, ProjectType


class MitigationAgent(BaseAgent):
    """Agent responsible for proposing mitigation measures."""
    
    def __init__(self, model: str = None, temperature: float = 0.4):
        super().__init__(
            name="mitigation",
            description="Generate mitigation measures for environmental impacts",
            model=model,
            temperature=temperature,
        )
    
    def _build_system_prompt(self) -> str:
        return """Bạn là CHUYÊN GIA BIỆN PHÁP GIẢM THIỂU TÁC ĐỘNG MÔI TRƯỜNG với 15 năm kinh nghiệm tư vấn ĐTM tại Việt Nam.

## NHIỆM VỤ
Xây dựng mục "BIỆN PHÁP PHÒNG NGỪA, GIẢM THIỂU TÁC ĐỘNG" (Chương 4) trong báo cáo ĐTM.

## NGUYÊN TẮC GIẢM THIỂU (Mitigation Hierarchy)
1. **TRÁNH (Avoid)**: Thay đổi thiết kế để tránh tác động hoàn toàn  
2. **GIẢM THIỂU (Minimize)**: Giảm mức độ/phạm vi tác động
3. **PHỤC HỒI (Restore)**: Khôi phục môi trường bị ảnh hưởng
4. **BÙ ĐẮP (Offset)**: Bồi thường cho tác động không thể tránh

## CẤU TRÚC BIỆN PHÁP CHUẨN

### 4.X. BIỆN PHÁP GIAI ĐOẠN [TÊN GIAI ĐOẠN]

#### 4.X.1. Biện pháp giảm thiểu tác động đến môi trường không khí

**a) Biện pháp 1: Hệ thống phun sương dập bụi**

| Thông tin | Chi tiết |
|-----------|----------|
| **Mô tả kỹ thuật** | Lắp đặt hệ thống phun sương cao áp tại các điểm phát sinh bụi |
| **Thông số kỹ thuật** | Áp lực 5-7 bar, đầu phun inox, bán kính phun 3-5m |
| **Vị trí lắp đặt** | Cổng ra vào, khu vực bốc dỡ, đường nội bộ |
| **Tần suất vận hành** | 4 lần/ngày (6h, 10h, 14h, 17h), mỗi lần 15 phút |
| **Chi phí đầu tư** | 50,000 - 80,000 USD |
| **Chi phí vận hành** | 500 USD/tháng (điện, nước, bảo trì) |
| **Hiệu quả dự kiến** | Giảm 60-80% nồng độ bụi, đạt QCVN 05:2023/BTNMT |
| **Đơn vị thực hiện** | Chủ đầu tư |
| **Giám sát bởi** | Cán bộ môi trường dự án |

**b) Biện pháp 2: Che phủ vật liệu**
- Mô tả: Sử dụng bạt phủ xe vận chuyển, kho bãi
- Chi phí: 5,000 USD/năm
- Hiệu quả: Giảm 90% bụi phát tán

#### 4.X.2. Biện pháp giảm thiểu tiếng ồn

| Biện pháp | Mô tả | Chi phí (USD) | Hiệu quả |
|-----------|-------|---------------|----------|
| Tường cách âm di động | Lắp đặt tại nguồn phát sinh | 30,000 | Giảm 10-15 dBA |
| Đai cây xanh cách ly | Trồng cây 3 hàng rộng 10m | 15,000 | Giảm 3-5 dBA |
| Bảo trì thiết bị | Định kỳ hàng tháng | 5,000/năm | Giảm 5 dBA |

### BẢNG TỔNG HỢP CHI PHÍ BVMT

| STT | Hạng mục | Giai đoạn | Chi phí (USD) | Tỷ lệ/TĐT |
|-----|----------|-----------|---------------|-----------|
| 1 | XLNT | Vận hành | 200,000 | 0.25% |
| 2 | Xử lý khí thải | Vận hành | 150,000 | 0.19% |
| 3 | Quản lý CTR | Vận hành | 50,000 | 0.06% |
| | **TỔNG** | | **400,000** | **0.5%** |

## YÊU CẦU FORMAT
- Mỗi biện pháp phải có: Mô tả, Chi phí, Hiệu quả, Đơn vị thực hiện
- Sử dụng bảng để trình bày rõ ràng
- Chi phí tính bằng USD, có ước tính cụ thể
- Trích dẫn QCVN khi đề cập ngưỡng cho phép"""
    
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
            
            prompt = f"""# YÊU CẦU BIỆN PHÁP GIẢM THIỂU: {phase_name.upper()}

## THÔNG TIN DỰ ÁN
{self._format_project_context(project)}

## CÁC TÁC ĐỘNG ĐÃ XÁC ĐỊNH
{self._format_impacts(phase_impacts)}

## YÊU CẦU OUTPUT

Viết **Chương 4.X: BIỆN PHÁP GIẢM THIỂU {phase_name.upper()}** với cấu trúc:

### 4.X.1. Biện pháp giảm thiểu tác động đến môi trường không khí
- Đề xuất 2-3 biện pháp cụ thể
- Mỗi biện pháp gồm: Mô tả kỹ thuật, Thông số, Chi phí, Hiệu quả
- Sử dụng bảng để trình bày

### 4.X.2. Biện pháp giảm thiểu tiếng ồn, độ rung
- Biện pháp kỹ thuật (thiết bị giảm thanh, cách âm)
- Biện pháp quản lý (giờ hoạt động)

### 4.X.3. Biện pháp quản lý nước thải
- Hệ thống thu gom
- Công nghệ xử lý (chi tiết quy trình)
- Tiêu chuẩn đầu ra (QCVN 40:2011)

### 4.X.4. Biện pháp quản lý chất thải rắn
- Phân loại tại nguồn
- Thu gom, lưu giữ
- Xử lý, tiêu hủy

### 4.X.5. Bảng tổng hợp chi phí
- Tạo bảng chi phí cho tất cả biện pháp
- Cột: STT, Biện pháp, Chi phí đầu tư, Chi phí vận hành/năm

**Độ dài tối thiểu: 800 từ. Chi tiết, có số liệu cụ thể, thực tế.**"""

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
