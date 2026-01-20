"""
Environmental Regulations Knowledge Base for EIA Generator.

Provides structured knowledge about Vietnamese environmental regulations,
standards, and guidelines for RAG-enhanced report generation.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Regulation:
    """Environmental regulation entry."""
    code: str
    name: str
    description: str
    applicable_to: List[str]
    key_requirements: List[str]
    effective_date: str = ""
    

@dataclass
class Standard:
    """Environmental standard/limit."""
    code: str
    name: str
    parameters: Dict[str, Dict]  # parameter -> {limit, unit, condition}
    

# =============================================================================
# Vietnamese Environmental Regulations Database
# =============================================================================

LAWS = [
    Regulation(
        code="72/2020/QH14",
        name="Luật Bảo vệ môi trường 2020",
        description="Luật khung về bảo vệ môi trường, quy định về ĐTM, giấy phép môi trường",
        applicable_to=["all"],
        key_requirements=[
            "Điều 30: Đối tượng phải lập ĐTM",
            "Điều 31: Nội dung báo cáo ĐTM",
            "Điều 32: Tham vấn trong quá trình lập ĐTM",
            "Điều 33: Thẩm định báo cáo ĐTM",
            "Điều 34: Phê duyệt kết quả thẩm định ĐTM",
            "Điều 39: Đối tượng phải có giấy phép môi trường",
            "Điều 99: Quan trắc môi trường",
        ],
        effective_date="2022-01-01",
    ),
]

DECREES = [
    Regulation(
        code="08/2022/NĐ-CP",
        name="Nghị định quy định chi tiết một số điều của Luật BVMT",
        description="Hướng dẫn chi tiết về ĐTM, giấy phép môi trường",
        applicable_to=["all"],
        key_requirements=[
            "Phụ lục II: Danh mục dự án phải lập ĐTM",
            "Phụ lục III: Nội dung báo cáo ĐTM sơ bộ",
            "Phụ lục IV: Nội dung chính của báo cáo ĐTM",
            "Mục 1-6: Nội dung chi tiết 6 chương báo cáo ĐTM",
        ],
        effective_date="2022-01-10",
    ),
]

CIRCULARS = [
    Regulation(
        code="02/2022/TT-BTNMT",
        name="Thông tư hướng dẫn Luật BVMT",
        description="Quy định chi tiết về mẫu báo cáo, trình tự thủ tục",
        applicable_to=["all"],
        key_requirements=[
            "Phụ lục II: Mẫu văn bản ĐTM",
            "Phụ lục V: Mẫu báo cáo ĐTM",
        ],
        effective_date="2022-01-10",
    ),
]


# =============================================================================
# Environmental Standards (QCVN)
# =============================================================================

QCVN_AIR_AMBIENT = Standard(
    code="QCVN 05:2023/BTNMT",
    name="Chất lượng không khí xung quanh",
    parameters={
        "SO2_1h": {"limit": 350, "unit": "µg/m³", "condition": "Trung bình 1 giờ"},
        "SO2_24h": {"limit": 125, "unit": "µg/m³", "condition": "Trung bình 24 giờ"},
        "SO2_year": {"limit": 50, "unit": "µg/m³", "condition": "Trung bình năm"},
        "NO2_1h": {"limit": 200, "unit": "µg/m³", "condition": "Trung bình 1 giờ"},
        "NO2_24h": {"limit": 100, "unit": "µg/m³", "condition": "Trung bình 24 giờ"},
        "NO2_year": {"limit": 40, "unit": "µg/m³", "condition": "Trung bình năm"},
        "CO_1h": {"limit": 30000, "unit": "µg/m³", "condition": "Trung bình 1 giờ"},
        "CO_8h": {"limit": 10000, "unit": "µg/m³", "condition": "Trung bình 8 giờ"},
        "O3_1h": {"limit": 180, "unit": "µg/m³", "condition": "Trung bình 1 giờ"},
        "O3_8h": {"limit": 120, "unit": "µg/m³", "condition": "Trung bình 8 giờ"},
        "PM10_24h": {"limit": 150, "unit": "µg/m³", "condition": "Trung bình 24 giờ"},
        "PM10_year": {"limit": 50, "unit": "µg/m³", "condition": "Trung bình năm"},
        "PM2.5_24h": {"limit": 50, "unit": "µg/m³", "condition": "Trung bình 24 giờ"},
        "PM2.5_year": {"limit": 25, "unit": "µg/m³", "condition": "Trung bình năm"},
        "TSP_24h": {"limit": 300, "unit": "µg/m³", "condition": "Trung bình 24 giờ"},
        "TSP_year": {"limit": 100, "unit": "µg/m³", "condition": "Trung bình năm"},
        "Pb_24h": {"limit": 1.5, "unit": "µg/m³", "condition": "Trung bình 24 giờ"},
    },
)

QCVN_AIR_INDUSTRIAL = Standard(
    code="QCVN 19:2009/BTNMT",
    name="Khí thải công nghiệp đối với bụi và các chất vô cơ",
    parameters={
        "Bui": {"limit": 200, "unit": "mg/Nm³", "condition": "Cột B"},
        "CO": {"limit": 1000, "unit": "mg/Nm³", "condition": "Cột B"},
        "SO2": {"limit": 500, "unit": "mg/Nm³", "condition": "Cột B"},
        "NOx": {"limit": 850, "unit": "mg/Nm³", "condition": "Cột B"},
        "H2S": {"limit": 7.5, "unit": "mg/Nm³", "condition": "Cột B"},
        "NH3": {"limit": 50, "unit": "mg/Nm³", "condition": "Cột B"},
        "HCl": {"limit": 50, "unit": "mg/Nm³", "condition": "Cột B"},
        "HF": {"limit": 20, "unit": "mg/Nm³", "condition": "Cột B"},
        "Pb": {"limit": 5, "unit": "mg/Nm³", "condition": "Cột B"},
        "Cd": {"limit": 1, "unit": "mg/Nm³", "condition": "Cột B"},
        "As": {"limit": 5, "unit": "mg/Nm³", "condition": "Cột B"},
    },
)

QCVN_WASTEWATER_INDUSTRIAL = Standard(
    code="QCVN 40:2011/BTNMT",
    name="Nước thải công nghiệp",
    parameters={
        "pH": {"limit": "5.5-9", "unit": "-", "condition": "Cột B"},
        "BOD5": {"limit": 50, "unit": "mg/L", "condition": "Cột B - 20°C"},
        "COD": {"limit": 150, "unit": "mg/L", "condition": "Cột B"},
        "TSS": {"limit": 100, "unit": "mg/L", "condition": "Cột B"},
        "Coliform": {"limit": 5000, "unit": "MPN/100mL", "condition": "Cột B"},
        "Amoni": {"limit": 10, "unit": "mg/L", "condition": "Cột B, tính theo N"},
        "Tong_N": {"limit": 40, "unit": "mg/L", "condition": "Cột B"},
        "Tong_P": {"limit": 6, "unit": "mg/L", "condition": "Cột B"},
        "Dau_mo": {"limit": 10, "unit": "mg/L", "condition": "Cột B"},
        "Clo_du": {"limit": 2, "unit": "mg/L", "condition": "Cột B"},
        "Fe": {"limit": 5, "unit": "mg/L", "condition": "Cột B"},
        "Pb": {"limit": 0.5, "unit": "mg/L", "condition": "Cột B"},
        "Cd": {"limit": 0.1, "unit": "mg/L", "condition": "Cột B"},
        "As": {"limit": 0.1, "unit": "mg/L", "condition": "Cột B"},
        "Hg": {"limit": 0.01, "unit": "mg/L", "condition": "Cột B"},
    },
)

QCVN_WASTEWATER_DOMESTIC = Standard(
    code="QCVN 14:2008/BTNMT",
    name="Nước thải sinh hoạt",
    parameters={
        "pH": {"limit": "5-9", "unit": "-", "condition": "Cột B"},
        "BOD5": {"limit": 50, "unit": "mg/L", "condition": "Cột B"},
        "TSS": {"limit": 100, "unit": "mg/L", "condition": "Cột B"},
        "TDS": {"limit": 1000, "unit": "mg/L", "condition": "Cột B"},
        "Amoni": {"limit": 10, "unit": "mg/L", "condition": "Cột B"},
        "Nitrat": {"limit": 50, "unit": "mg/L", "condition": "Cột B"},
        "Phosphat": {"limit": 10, "unit": "mg/L", "condition": "Cột B"},
        "Dau_mo": {"limit": 20, "unit": "mg/L", "condition": "Cột B"},
        "Coliform": {"limit": 5000, "unit": "MPN/100mL", "condition": "Cột B"},
    },
)

QCVN_NOISE = Standard(
    code="QCVN 26:2010/BTNMT",
    name="Tiếng ồn",
    parameters={
        "Khu_dan_cu_ngay": {"limit": 70, "unit": "dBA", "condition": "6h-21h"},
        "Khu_dan_cu_dem": {"limit": 55, "unit": "dBA", "condition": "21h-6h"},
        "Khu_CN_ngay": {"limit": 70, "unit": "dBA", "condition": "6h-21h, khu công nghiệp"},
        "Khu_CN_dem": {"limit": 70, "unit": "dBA", "condition": "21h-6h, khu công nghiệp"},
        "Benh_vien_ngay": {"limit": 55, "unit": "dBA", "condition": "6h-21h"},
        "Benh_vien_dem": {"limit": 45, "unit": "dBA", "condition": "21h-6h"},
        "Truong_hoc": {"limit": 55, "unit": "dBA", "condition": "Trong giờ học"},
    },
)

QCVN_VIBRATION = Standard(
    code="QCVN 27:2010/BTNMT",
    name="Độ rung",
    parameters={
        "Khu_dan_cu_ngay": {"limit": 75, "unit": "dB", "condition": "6h-18h"},
        "Khu_dan_cu_toi": {"limit": 70, "unit": "dB", "condition": "18h-22h"},
        "Khu_dan_cu_dem": {"limit": 65, "unit": "dB", "condition": "22h-6h"},
        "Benh_vien_ngay": {"limit": 70, "unit": "dB", "condition": "6h-18h"},
        "Benh_vien_dem": {"limit": 60, "unit": "dB", "condition": "22h-6h"},
    },
)


# =============================================================================
# Knowledge Retrieval Functions
# =============================================================================

class RegulationsKB:
    """Knowledge base for environmental regulations."""
    
    def __init__(self):
        self.laws = LAWS
        self.decrees = DECREES
        self.circulars = CIRCULARS
        self.standards = {
            "QCVN 05:2023": QCVN_AIR_AMBIENT,
            "QCVN 19:2009": QCVN_AIR_INDUSTRIAL,
            "QCVN 40:2011": QCVN_WASTEWATER_INDUSTRIAL,
            "QCVN 14:2008": QCVN_WASTEWATER_DOMESTIC,
            "QCVN 26:2010": QCVN_NOISE,
            "QCVN 27:2010": QCVN_VIBRATION,
        }
    
    def get_all_regulations(self) -> str:
        """Get formatted list of all regulations."""
        lines = ["## VĂN BẢN PHÁP LUẬT VỀ MÔI TRƯỜNG\n"]
        
        lines.append("### 1. Luật")
        for reg in self.laws:
            lines.append(f"- **{reg.code}**: {reg.name}")
            for req in reg.key_requirements[:3]:
                lines.append(f"  + {req}")
        
        lines.append("\n### 2. Nghị định")
        for reg in self.decrees:
            lines.append(f"- **{reg.code}**: {reg.name}")
            for req in reg.key_requirements[:3]:
                lines.append(f"  + {req}")
        
        lines.append("\n### 3. Thông tư")
        for reg in self.circulars:
            lines.append(f"- **{reg.code}**: {reg.name}")
        
        return "\n".join(lines)
    
    def get_standard_limits(self, standard_code: str) -> str:
        """Get formatted limits for a specific standard."""
        std = self.standards.get(standard_code)
        if not std:
            return f"Không tìm thấy quy chuẩn: {standard_code}"
        
        lines = [f"## {std.code}: {std.name}\n"]
        lines.append("| Thông số | Giới hạn | Đơn vị | Điều kiện |")
        lines.append("|----------|----------|--------|-----------|")
        
        for param, info in std.parameters.items():
            limit = info["limit"]
            unit = info["unit"]
            condition = info.get("condition", "")
            lines.append(f"| {param} | {limit} | {unit} | {condition} |")
        
        return "\n".join(lines)
    
    def get_air_limits(self) -> str:
        """Get air quality limits."""
        return (
            self.get_standard_limits("QCVN 05:2023") + 
            "\n\n" + 
            self.get_standard_limits("QCVN 19:2009")
        )
    
    def get_water_limits(self) -> str:
        """Get water quality limits."""
        return (
            self.get_standard_limits("QCVN 40:2011") + 
            "\n\n" + 
            self.get_standard_limits("QCVN 14:2008")
        )
    
    def get_noise_limits(self) -> str:
        """Get noise limits."""
        return (
            self.get_standard_limits("QCVN 26:2010") + 
            "\n\n" + 
            self.get_standard_limits("QCVN 27:2010")
        )
    
    def get_eia_requirements(self) -> str:
        """Get EIA requirements according to Decree 08/2022."""
        return """## YÊU CẦU NỘI DUNG BÁO CÁO ĐTM
(Theo Phụ lục IV, Nghị định 08/2022/NĐ-CP)

### CHƯƠNG 1: MÔ TẢ TÓM TẮT DỰ ÁN
- Thông tin chung về dự án
- Các hạng mục công trình
- Công nghệ sản xuất
- Nguyên, nhiên liệu đầu vào
- Tiến độ thực hiện

### CHƯƠNG 2: ĐIỀU KIỆN TỰ NHIÊN, KINH TẾ - XÃ HỘI
- Điều kiện tự nhiên (địa lý, khí hậu, thủy văn)
- Điều kiện kinh tế - xã hội
- Hiện trạng chất lượng môi trường

### CHƯƠNG 3: ĐÁNH GIÁ TÁC ĐỘNG MÔI TRƯỜNG
- Tác động giai đoạn chuẩn bị
- Tác động giai đoạn xây dựng
- Tác động giai đoạn vận hành
- Tác động giai đoạn kết thúc

### CHƯƠNG 4: BIỆN PHÁP PHÒNG NGỪA, GIẢM THIỂU
- Biện pháp giảm thiểu giai đoạn xây dựng
- Biện pháp giảm thiểu giai đoạn vận hành
- Chi phí các công trình BVMT

### CHƯƠNG 5: CHƯƠNG TRÌNH QUẢN LÝ VÀ GIÁM SÁT
- Chương trình quản lý môi trường
- Chương trình giám sát môi trường
- Kế hoạch ứng phó sự cố

### CHƯƠNG 6: THAM VẤN CỘNG ĐỒNG
- Tham vấn UBND cấp xã
- Tham vấn cộng đồng dân cư
- Tổng hợp ý kiến và phản hồi"""


# =============================================================================
# Singleton Instance
# =============================================================================

_kb_instance = None

def get_regulations_kb() -> RegulationsKB:
    """Get singleton knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = RegulationsKB()
    return _kb_instance
