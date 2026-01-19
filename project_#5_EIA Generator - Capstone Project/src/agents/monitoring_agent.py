"""
Monitoring Agent - Generates environmental monitoring program.
"""

from typing import Any, Dict, List

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import ImpactCategory, ProjectType


class MonitoringAgent(BaseAgent):
    """Agent responsible for environmental monitoring program."""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.4):
        super().__init__(
            name="monitoring",
            description="Generate environmental management and monitoring program",
            model=model,
            temperature=temperature,
        )
    
    def _build_system_prompt(self) -> str:
        return """Bạn là chuyên gia quản lý và giám sát môi trường (Environmental Management Expert).

NHIỆM VỤ: Xây dựng mục "CHƯƠNG TRÌNH QUẢN LÝ VÀ GIÁM SÁT MÔI TRƯỜNG" trong báo cáo ĐTM.

NỘI DUNG:

1. CHƯƠNG TRÌNH QUẢN LÝ MÔI TRƯỜNG:
   - Cơ cấu tổ chức quản lý môi trường
   - Phân công trách nhiệm
   - Quy trình vận hành, bảo dưỡng thiết bị BVMT
   - Kế hoạch đào tạo
   - Kế hoạch ứng phó sự cố

2. CHƯƠNG TRÌNH GIÁM SÁT MÔI TRƯỜNG:
   - Vị trí quan trắc
   - Thông số quan trắc
   - Tần suất quan trắc
   - Phương pháp/thiết bị
   - Đơn vị thực hiện
   - Chi phí quan trắc

TIÊU CHUẨN ÁP DỤNG:
- QCVN 05:2023/BTNMT (Không khí xung quanh)
- QCVN 08:2023/BTNMT (Nước mặt)  
- QCVN 09:2023/BTNMT (Nước ngầm)
- QCVN 19:2009/BTNMT (Khí thải công nghiệp)
- QCVN 40:2011/BTNMT (Nước thải công nghiệp)
- QCVN 26:2010/BTNMT (Tiếng ồn)"""
    
    async def execute(self, state: AgentState) -> AgentState:
        """Generate monitoring program."""
        logger.info("Monitoring Agent: Generating monitoring program")
        
        project = state["project"]
        
        # Generate management program
        prompt_management = f"""Xây dựng chương trình quản lý môi trường cho dự án:

{self._format_project_context(project)}

YÊU CẦU:
1. Đề xuất cơ cấu tổ chức quản lý môi trường
2. Phân công trách nhiệm cụ thể
3. Xây dựng quy trình vận hành thiết bị BVMT
4. Kế hoạch đào tạo nhân viên
5. Kế hoạch ứng phó sự cố môi trường

Viết chi tiết bằng tiếng Việt."""

        management_content = await self._generate(prompt_management)
        
        # Generate monitoring program
        prompt_monitoring = f"""Xây dựng chương trình giám sát môi trường cho dự án:

{self._format_project_context(project)}

YÊU CẦU lập bảng theo format:

| Giai đoạn | Loại giám sát | Vị trí | Thông số | Tần suất | QCVN áp dụng | Chi phí/năm |
|-----------|---------------|--------|----------|----------|--------------|-------------|

Các loại giám sát:
1. Giám sát chất lượng không khí xung quanh
2. Giám sát khí thải nguồn
3. Giám sát nước mặt
4. Giám sát nước ngầm
5. Giám sát nước thải
6. Giám sát tiếng ồn
7. Giám sát chất thải rắn

Viết chi tiết bằng tiếng Việt."""

        monitoring_content = await self._generate(prompt_monitoring)
        
        # Generate emergency response plan
        prompt_emergency = f"""Xây dựng kế hoạch ứng phó sự cố môi trường:

{self._format_project_context(project)}

YÊU CẦU:
1. Nhận diện các sự cố môi trường tiềm ẩn
2. Kịch bản ứng phó cho từng loại sự cố
3. Phân công trách nhiệm khi xảy ra sự cố
4. Trang thiết bị ứng phó cần thiết
5. Quy trình báo cáo sự cố
6. Kế hoạch diễn tập

Viết chi tiết bằng tiếng Việt."""

        emergency_content = await self._generate(prompt_emergency)
        
        # Update state
        state["sections"]["management"] = management_content
        state["sections"]["monitoring"] = monitoring_content
        state["sections"]["emergency_response"] = emergency_content
        
        # Add monitoring tables
        state["tables"]["monitoring_schedule"] = self._create_monitoring_schedule(project)
        
        state["current_step"] = "monitoring_complete"
        
        logger.info("Monitoring Agent: Completed monitoring program")
        return state
    
    def _create_monitoring_schedule(self, project: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create monitoring schedule table."""
        schedule = [
            {
                "phase": "Xây dựng",
                "type": "Không khí xung quanh",
                "location": "4 vị trí xung quanh công trường",
                "parameters": "Bụi TSP, PM10, CO, NO2, SO2",
                "frequency": "3 tháng/lần",
                "standard": "QCVN 05:2023/BTNMT",
            },
            {
                "phase": "Xây dựng",
                "type": "Tiếng ồn",
                "location": "4 vị trí ranh giới",
                "parameters": "LAeq, Lmax",
                "frequency": "3 tháng/lần",
                "standard": "QCVN 26:2010/BTNMT",
            },
            {
                "phase": "Vận hành",
                "type": "Khí thải nguồn",
                "location": "Ống khói/nguồn thải",
                "parameters": "Bụi, CO, SO2, NOx",
                "frequency": "6 tháng/lần",
                "standard": "QCVN 19:2009/BTNMT",
            },
            {
                "phase": "Vận hành",
                "type": "Nước thải",
                "location": "Cửa xả sau HTXLNT",
                "parameters": "pH, BOD5, COD, TSS, Coliform",
                "frequency": "3 tháng/lần",
                "standard": "QCVN 40:2011/BTNMT",
            },
            {
                "phase": "Vận hành",
                "type": "Nước ngầm",
                "location": "2 giếng quan trắc",
                "parameters": "pH, độ cứng, Fe, Mn, Coliform",
                "frequency": "6 tháng/lần",
                "standard": "QCVN 09:2023/BTNMT",
            },
        ]
        return schedule
