"""
Research Agent - Retrieves regulations, standards, and reference materials.
"""

from typing import Any, Dict, List

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import PROJECT_REGULATIONS, ProjectType
from ..knowledge.regulations_kb import get_regulations_kb


class ResearchAgent(BaseAgent):
    """Agent responsible for researching regulations and standards."""
    
    def __init__(self, model: str = None, temperature: float = 0.3):
        super().__init__(
            name="research",
            description="Research relevant regulations, standards, and guidelines for EIA",
            model=model,
            temperature=temperature,
        )
        # Load knowledge base
        self.kb = get_regulations_kb()
    
    def _build_system_prompt(self) -> str:
        return """Bạn là CHUYÊN GIA PHÁP LUẬT MÔI TRƯỜNG với 20 năm kinh nghiệm về Đánh giá Tác động Môi trường (ĐTM) tại Việt Nam.

## NHIỆM VỤ
Nghiên cứu và tổng hợp đầy đủ các văn bản pháp luật, quy chuẩn kỹ thuật áp dụng cho dự án.

## KIẾN THỨC BẮT BUỘC ÁP DỤNG

### 1. Luật và Nghị định chính:
- **Luật Bảo vệ môi trường 2020** (Luật số 72/2020/QH14)
  + Điều 30-34: Quy định về ĐTM
  + Điều 39-52: Giấy phép môi trường
- **Nghị định 08/2022/NĐ-CP**: Quy định chi tiết một số điều của Luật BVMT
  + Phụ lục II: Danh mục dự án phải lập ĐTM
  + Phụ lục IV: Nội dung chính của báo cáo ĐTM

### 2. Quy chuẩn kỹ thuật quốc gia (QCVN):
- QCVN 05:2023/BTNMT: Chất lượng không khí xung quanh
- QCVN 19:2009/BTNMT: Khí thải công nghiệp
- QCVN 40:2011/BTNMT: Nước thải công nghiệp
- QCVN 14:2008/BTNMT: Nước thải sinh hoạt
- QCVN 26:2010/BTNMT: Tiếng ồn
- QCVN 27:2010/BTNMT: Độ rung

### 3. Tiêu chuẩn quốc tế:
- IFC Environmental and Social Performance Standards
- World Bank Environmental and Social Framework
- TCVN ISO 14001 - Hệ thống quản lý môi trường

## VÍ DỤ OUTPUT CHUẨN

### I. VĂN BẢN PHÁP LUẬT ÁP DỤNG

#### 1.1. Luật và Nghị định
| STT | Văn bản | Nội dung áp dụng |
|-----|---------|------------------|
| 1 | Luật BVMT 2020 | Điều 30-34 về ĐTM, Điều 39 về giấy phép môi trường |
| 2 | Nghị định 08/2022/NĐ-CP | Phụ lục II - Nhóm I.A.1: Dự án năng lượng |

#### 1.2. Quy chuẩn kỹ thuật quốc gia
| QCVN | Phạm vi áp dụng | Giới hạn chính |
|------|-----------------|----------------|
| QCVN 05:2023/BTNMT | Không khí xung quanh | PM10 ≤ 150 µg/m³ (24h) |
| QCVN 26:2010/BTNMT | Tiếng ồn | ≤ 70 dBA (khu công nghiệp) |

### II. YÊU CẦU CỤ THỂ CẦN TUÂN THỦ
1. **Lập báo cáo ĐTM** trước khi khởi công (Điều 30 Luật BVMT 2020)
2. **Xin giấy phép môi trường** sau khi ĐTM được phê duyệt (Điều 39)
3. **Quan trắc môi trường** theo quy định tại Điều 99-100

## YÊU CẦU FORMAT
- Sử dụng bảng để trình bày thông tin
- Trích dẫn chính xác số điều, khoản
- Ghi rõ giới hạn cho phép với đơn vị
- Viết văn bản chính thức, trang trọng"""
    
    async def execute(self, state: AgentState) -> AgentState:
        """Research regulations for the project."""
        logger.info(f"Research Agent: Researching regulations for project")
        
        project = state["project"]
        project_type = ProjectType(project.get("type", "industrial_manufacturing"))
        
        # Get predefined regulations
        regs = PROJECT_REGULATIONS.get(project_type, {})
        
        # Generate comprehensive regulation analysis
        prompt = f"""# YÊU CẦU NGHIÊN CỨU QUY ĐỊNH PHÁP LUẬT

## THÔNG TIN DỰ ÁN
{self._format_project_context(project)}

## QUY ĐỊNH SƠ BỘ ĐÃ XÁC ĐỊNH
- **Quy định chính**: {', '.join(regs.get('primary', ['Luật BVMT 2020', 'Nghị định 08/2022']))}
- **Quy định kỹ thuật**: {', '.join(regs.get('technical', []))}
- **Quy định môi trường**: {', '.join(regs.get('environmental', []))}

## YÊU CẦU OUTPUT

Hãy viết phần **"CHƯƠNG 1: CƠ SỞ PHÁP LÝ VÀ KỸ THUẬT"** của báo cáo ĐTM với cấu trúc:

### 1.1. Cơ sở pháp lý
- Liệt kê đầy đủ các văn bản pháp luật áp dụng (Luật, Nghị định, Thông tư)
- Trích dẫn cụ thể điều khoản liên quan đến dự án
- Sử dụng bảng để trình bày

### 1.2. Quy chuẩn kỹ thuật quốc gia
- Liệt kê các QCVN áp dụng cho dự án
- Ghi rõ giới hạn cho phép với đơn vị
- Bao gồm: không khí, nước thải, tiếng ồn, chất thải rắn

### 1.3. Tiêu chuẩn quốc tế tham khảo
- IFC Performance Standards
- World Bank Guidelines

### 1.4. Yêu cầu cụ thể cần tuân thủ
- Liệt kê 5-10 yêu cầu quan trọng nhất

**Độ dài tối thiểu: 800 từ. Viết chi tiết, chuyên nghiệp, có trích dẫn cụ thể.**"""

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
