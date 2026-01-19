"""
Baseline Agent - Generates environmental and socioeconomic baseline sections.
"""

from typing import Any, Dict

from loguru import logger

from .base import BaseAgent, AgentState


class BaselineAgent(BaseAgent):
    """Agent responsible for environmental baseline analysis."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.4):
        super().__init__(
            name="baseline",
            description="Generate environmental and socioeconomic baseline assessment",
            model=model,
            temperature=temperature,
        )
    
    def _build_system_prompt(self) -> str:
        return """Bạn là chuyên gia đánh giá môi trường nền (Environmental Baseline Assessment).

NHIỆM VỤ: Xây dựng mục "ĐIỀU KIỆN TỰ NHIÊN, KINH TẾ - XÃ HỘI" trong báo cáo ĐTM.

NỘI DUNG CẦN PHÂN TÍCH:

1. ĐIỀU KIỆN TỰ NHIÊN:
   - Vị trí địa lý, địa hình, địa chất
   - Khí hậu, thủy văn
   - Hiện trạng sử dụng đất
   - Tài nguyên sinh vật (động vật, thực vật)

2. ĐIỀU KIỆN KINH TẾ - XÃ HỘI:
   - Dân số, lao động
   - Cơ cấu kinh tế
   - Cơ sở hạ tầng (giao thông, điện, nước)
   - Các công trình nhạy cảm lân cận

3. HIỆN TRẠNG MÔI TRƯỜNG:
   - Chất lượng không khí
   - Chất lượng nước mặt, nước ngầm
   - Chất lượng đất
   - Mức độ tiếng ồn, độ rung

YÊU CẦU:
- Sử dụng số liệu cụ thể, có nguồn trích dẫn
- Phân tích phù hợp với vị trí dự án
- Xác định các vùng/đối tượng nhạy cảm
- Đề xuất vị trí quan trắc môi trường nền"""
    
    async def execute(self, state: AgentState) -> AgentState:
        """Generate baseline assessment section."""
        logger.info("Baseline Agent: Generating environmental baseline")
        
        project = state["project"]
        
        # Generate natural conditions
        prompt_natural = f"""Phân tích điều kiện tự nhiên khu vực dự án:

{self._format_project_context(project)}

Yêu cầu viết mục 2.1 "Điều kiện tự nhiên" bao gồm:
a) Vị trí địa lý và địa hình
b) Đặc điểm khí hậu (nhiệt độ, mưa, gió, độ ẩm)
c) Thủy văn (sông suối, mực nước ngầm)
d) Địa chất, thổ nhưỡng
e) Tài nguyên sinh vật

Viết chi tiết, có số liệu minh họa. Tiếng Việt."""

        natural_content = await self._generate(prompt_natural)
        
        # Generate socioeconomic conditions
        prompt_socio = f"""Phân tích điều kiện kinh tế - xã hội khu vực dự án:

{self._format_project_context(project)}

Yêu cầu viết mục 2.2 "Điều kiện kinh tế - xã hội" bao gồm:
a) Dân số và lao động trong khu vực
b) Cơ cấu kinh tế địa phương
c) Cơ sở hạ tầng hiện có
d) Các công trình nhạy cảm lân cận (trường học, bệnh viện, khu dân cư)
e) Di tích văn hóa, lịch sử (nếu có)

Viết chi tiết, có số liệu minh họa. Tiếng Việt."""

        socio_content = await self._generate(prompt_socio)
        
        # Generate environmental baseline
        prompt_env = f"""Đánh giá hiện trạng chất lượng môi trường khu vực dự án:

{self._format_project_context(project)}

Yêu cầu viết mục 2.3 "Hiện trạng môi trường" bao gồm:
a) Chất lượng không khí xung quanh
b) Chất lượng nước mặt
c) Chất lượng nước ngầm
d) Chất lượng đất
e) Tiếng ồn, độ rung hiện trạng

Lập bảng tổng hợp kết quả so với QCVN. Tiếng Việt."""

        env_content = await self._generate(prompt_env)
        
        # Update state
        state["sections"]["baseline_natural"] = natural_content
        state["sections"]["baseline_socio"] = socio_content
        state["sections"]["baseline_env"] = env_content
        
        state["baseline_data"] = {
            "natural": {"generated": True},
            "socioeconomic": {"generated": True},
            "environmental": {"generated": True},
        }
        state["current_step"] = "baseline_complete"
        
        logger.info("Baseline Agent: Completed baseline assessment")
        return state
    
    def _get_climate_data(self, province: str) -> Dict[str, Any]:
        """Get typical climate data for Vietnamese provinces."""
        # Simplified climate data for major regions
        climate_data = {
            "ninh_thuan": {
                "temperature_avg": 27.0,
                "temperature_max": 40.0,
                "temperature_min": 18.0,
                "rainfall_annual": 700,
                "humidity_avg": 75,
                "sunshine_hours": 2800,
                "wind_dominant": "Đông - Đông Bắc",
            },
            "binh_duong": {
                "temperature_avg": 27.5,
                "temperature_max": 38.0,
                "temperature_min": 20.0,
                "rainfall_annual": 1800,
                "humidity_avg": 80,
                "sunshine_hours": 2200,
                "wind_dominant": "Tây Nam",
            },
            "ha_noi": {
                "temperature_avg": 24.0,
                "temperature_max": 38.0,
                "temperature_min": 8.0,
                "rainfall_annual": 1700,
                "humidity_avg": 82,
                "sunshine_hours": 1500,
                "wind_dominant": "Đông Nam",
            },
            "default": {
                "temperature_avg": 26.0,
                "temperature_max": 38.0,
                "temperature_min": 15.0,
                "rainfall_annual": 1500,
                "humidity_avg": 78,
                "sunshine_hours": 2000,
                "wind_dominant": "Đông",
            },
        }
        
        province_key = province.lower().replace(" ", "_")
        return climate_data.get(province_key, climate_data["default"])
