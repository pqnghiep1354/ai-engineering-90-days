"""
Impact Agent - Generates environmental impact assessment sections.
"""

from typing import Any, Dict, List

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import ImpactCategory, IMPACT_FACTORS_BY_PROJECT, ProjectType


class ImpactAgent(BaseAgent):
    """Agent responsible for environmental impact assessment."""
    
    def __init__(self, model: str = None, temperature: float = 0.4):
        super().__init__(
            name="impact",
            description="Generate environmental impact assessment for all project phases",
            model=model,
            temperature=temperature,
        )
    
    def _build_system_prompt(self) -> str:
        return """Bạn là CHUYÊN GIA ĐÁNH GIÁ TÁC ĐỘNG MÔI TRƯỜNG với 15 năm kinh nghiệm thẩm định ĐTM tại Việt Nam.

## NHIỆM VỤ
Xây dựng mục "ĐÁNH GIÁ TÁC ĐỘNG MÔI TRƯỜNG" (Chương 3) trong báo cáo ĐTM theo Nghị định 08/2022/NĐ-CP.

## PHƯƠNG PHÁP ĐÁNH GIÁ
1. **Ma trận tác động (Leopold Matrix)**: Đánh giá cường độ và phạm vi
2. **Phương pháp liệt kê (Checklist)**: Xác định nguồn và đối tượng tác động
3. **Phương pháp so sánh**: So sánh với QCVN/TCVN
4. **Mô hình phát tán**: Tính toán nồng độ khí thải, tiếng ồn

## CẤU TRÚC ĐÁNH GIÁ CHO MỖI GIAI ĐOẠN

### 3.X. TÁC ĐỘNG GIAI ĐOẠN [TÊN GIAI ĐOẠN]

#### 3.X.1. Nguồn gây tác động
| STT | Nguồn tác động | Đối tượng bị tác động | Thành phần môi trường |
|-----|----------------|----------------------|----------------------|
| 1 | San lấp mặt bằng | Đất, cây xanh | Môi trường đất, sinh thái |
| 2 | Hoạt động xe máy móc | Công nhân, dân cư | Không khí, tiếng ồn |

#### 3.X.2. Đánh giá tác động môi trường không khí
**a) Nguồn phát sinh:**
- Khí thải từ phương tiện vận chuyển
- Bụi từ hoạt động san lấp

**b) Tải lượng phát thải:**
- Bụi: Q = n × m × EF = 50 tấn/tháng
- CO: Qco = 0.5 kg/km × 100 lượt/ngày = 50 kg/ngày

**c) Đánh giá mức độ tác động:**
| Thông số | Nồng độ dự báo | QCVN 05:2023 | Đánh giá |
|----------|---------------|--------------|----------|
| Bụi TSP | 0.25 mg/m³ | 0.3 mg/m³ | Đạt |
| CO | 15 mg/m³ | 30 mg/m³ | Đạt |

**Kết luận:** Tác động đến môi trường không khí ở mức TRUNG BÌNH, có thể kiểm soát.

#### 3.X.3. Ma trận tổng hợp tác động
| Thành phần MT | Mức độ | Phạm vi | Thời gian | Khả năng phục hồi |
|---------------|--------|---------|-----------|-------------------|
| Không khí | Trung bình (-) | Cục bộ | Ngắn hạn | Có |
| Tiếng ồn | Lớn (-) | Cục bộ | Ngắn hạn | Có |
| Kinh tế-XH | Lớn (+) | Khu vực | Dài hạn | - |

## THANG ĐÁNH GIÁ
- **Không đáng kể**: Tác động nhỏ, tự phục hồi
- **Nhỏ**: Tác động có thể chấp nhận, không cần biện pháp đặc biệt
- **Trung bình**: Cần biện pháp giảm thiểu thông thường
- **Lớn**: Cần biện pháp giảm thiểu đặc biệt, giám sát chặt chẽ"""
    
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
            prompt = f"""# YÊU CẦU ĐÁNH GIÁ TÁC ĐỘNG: {phase_name.upper()}

## THÔNG TIN DỰ ÁN
{self._format_project_context(project)}

## CÁC YẾU TỐ MÔI TRƯỜNG CẦN ĐÁNH GIÁ
{self._format_impact_factors(impact_factors)}

## YÊU CẦU NỘI DUNG

Viết **mục 3.X: ĐÁNH GIÁ TÁC ĐỘNG {phase_name.upper()}** với cấu trúc:

### 3.X.1. Nguồn gây tác động trong {phase_name}
- Liệt kê tất cả nguồn tác động (ít nhất 5 nguồn)
- Sử dụng bảng để trình bày
- Xác định đối tượng bị tác động

### 3.X.2. Đánh giá tác động đến từng thành phần môi trường
Với mỗi thành phần (không khí, nước, đất, tiếng ồn, sinh thái...):
- **Nguồn phát sinh**: Cụ thể hoạt động nào
- **Tải lượng/Khối lượng**: Ước tính số liệu cụ thể (kg/ngày, m³/ngày, dB)
- **So sánh với QCVN**: Đối chiếu với quy chuẩn
- **Kết luận mức độ**: Không đáng kể / Nhỏ / Trung bình / Lớn

### 3.X.3. Ma trận tổng hợp tác động
- Tạo bảng ma trận với các cột: Thành phần, Mức độ, Tính chất (+/-), Phạm vi, Thời gian
- Đánh giá khách quan, cân bằng

### 3.X.4. Tác động tích cực
- Liệt kê các tác động tích cực đến kinh tế, xã hội, môi trường

**Độ dài tối thiểu: 600 từ cho mỗi giai đoạn. Sử dụng bảng, số liệu cụ thể.**"""

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
