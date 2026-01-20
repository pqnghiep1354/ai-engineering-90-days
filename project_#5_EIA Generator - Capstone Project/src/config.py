"""
Configuration management for EIA Generator.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Environment Settings
# =============================================================================

class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # API Keys
    openai_api_key: str = Field(default="")
    tavily_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    
    # Model Settings
    default_model: str = Field(default="gemini-2.0-flash")
    embedding_model: str = Field(default="text-embedding-3-small")
    ollama_base_url: str = Field(default="http://localhost:11434")
    
    # Paths
    data_dir: str = Field(default="./data")
    output_dir: str = Field(default="./outputs")
    template_dir: str = Field(default="./data/templates")
    
    # Generation Settings
    max_section_tokens: int = Field(default=4000)
    temperature: float = Field(default=0.4)
    
    # Logging
    log_level: str = Field(default="INFO")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# =============================================================================
# Project Types
# =============================================================================

class ProjectType(str, Enum):
    """Types of projects requiring EIA."""
    
    # Energy
    ENERGY_SOLAR = "energy_solar"
    ENERGY_WIND = "energy_wind"
    ENERGY_THERMAL = "energy_thermal"
    ENERGY_HYDRO = "energy_hydro"
    
    # Industrial
    INDUSTRIAL_MANUFACTURING = "industrial_manufacturing"
    INDUSTRIAL_CHEMICAL = "industrial_chemical"
    INDUSTRIAL_FOOD = "industrial_food"
    INDUSTRIAL_TEXTILE = "industrial_textile"
    
    # Infrastructure
    INFRA_ROAD = "infrastructure_road"
    INFRA_PORT = "infrastructure_port"
    INFRA_AIRPORT = "infrastructure_airport"
    INFRA_RAILWAY = "infrastructure_railway"
    
    # Urban Development
    URBAN_RESIDENTIAL = "urban_residential"
    URBAN_COMMERCIAL = "urban_commercial"
    URBAN_INDUSTRIAL_ZONE = "urban_industrial_zone"
    
    # Other
    MINING = "mining"
    AGRICULTURE = "agriculture"
    TOURISM = "tourism"
    WASTE_TREATMENT = "waste_treatment"


# =============================================================================
# Impact Categories
# =============================================================================

class ImpactCategory(str, Enum):
    """Environmental impact categories."""
    
    AIR_QUALITY = "air_quality"
    WATER_QUALITY = "water_quality"
    SOIL = "soil"
    NOISE_VIBRATION = "noise_vibration"
    BIODIVERSITY = "biodiversity"
    LAND_USE = "land_use"
    VISUAL_LANDSCAPE = "visual_landscape"
    SOCIOECONOMIC = "socioeconomic"
    CULTURAL_HERITAGE = "cultural_heritage"
    CLIMATE = "climate"
    WASTE = "waste"
    TRAFFIC = "traffic"
    HEALTH_SAFETY = "health_safety"


# =============================================================================
# Project Input
# =============================================================================

class ProjectInput(BaseModel):
    """Input data for EIA generation."""
    
    # Basic Information
    name: str = Field(..., description="Project name")
    name_en: Optional[str] = Field(None, description="Project name in English")
    type: ProjectType = Field(..., description="Project type")
    description: str = Field(default="", description="Project description")
    
    # Location
    location: str = Field(..., description="Project location")
    province: str = Field(default="", description="Province/City")
    district: str = Field(default="", description="District")
    commune: str = Field(default="", description="Commune/Ward")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Lat/Long")
    
    # Scale
    area_hectares: float = Field(..., description="Project area in hectares")
    capacity: str = Field(default="", description="Production capacity")
    investment_usd: float = Field(default=0, description="Investment in USD")
    investment_vnd: float = Field(default=0, description="Investment in VND")
    
    # Timeline
    construction_months: int = Field(default=24, description="Construction duration")
    operation_years: int = Field(default=20, description="Operation duration")
    
    # Investor
    investor_name: str = Field(default="", description="Investor/Owner name")
    investor_address: str = Field(default="", description="Investor address")
    investor_contact: str = Field(default="", description="Contact information")
    
    # Additional Data
    existing_land_use: str = Field(default="", description="Current land use")
    nearby_sensitive_areas: List[str] = Field(default_factory=list)
    expected_emissions: Optional[Dict[str, float]] = Field(None)
    water_usage_m3_day: Optional[float] = Field(None)
    wastewater_m3_day: Optional[float] = Field(None)
    workers_construction: Optional[int] = Field(None)
    workers_operation: Optional[int] = Field(None)
    
    @model_validator(mode="after")
    def calculate_vnd(self) -> "ProjectInput":
        if self.investment_vnd == 0 and self.investment_usd > 0:
            self.investment_vnd = self.investment_usd * 24000  # Approximate rate
        return self


# =============================================================================
# EIA Report Structure
# =============================================================================

class EIASection(BaseModel):
    """EIA report section."""
    
    id: str
    title: str
    title_en: str
    content: str = ""
    subsections: List["EIASection"] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    figures: List[Dict[str, Any]] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)


class EIAReport(BaseModel):
    """Complete EIA report."""
    
    # Metadata
    project: ProjectInput
    generated_at: str
    version: str = "1.0"
    language: str = "vi"
    
    # Content
    executive_summary: str = ""
    sections: List[EIASection] = Field(default_factory=list)
    appendices: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Validation
    compliance_score: float = 0.0
    completeness_score: float = 0.0
    validation_notes: List[str] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# =============================================================================
# EIA Configuration
# =============================================================================

class EIAConfig(BaseModel):
    """Configuration for EIA generation."""
    
    # Report Settings
    language: str = Field(default="vi", description="Primary language (vi/en)")
    include_english: bool = Field(default=True, description="Include English sections")
    format: str = Field(default="full", description="Report format (full/summary)")
    
    # Agent Settings
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.4)
    max_tokens_per_section: int = Field(default=4000)
    
    # Content Settings
    include_appendices: bool = Field(default=True)
    include_figures: bool = Field(default=True)
    include_cost_estimates: bool = Field(default=True)
    
    # Validation Settings
    validate_compliance: bool = Field(default=True)
    min_compliance_score: float = Field(default=0.8)


# =============================================================================
# Regulation Mappings
# =============================================================================

PROJECT_REGULATIONS = {
    ProjectType.ENERGY_SOLAR: {
        "primary": ["QCVN 01:2021/BTNMT", "Nghị định 08/2022/NĐ-CP"],
        "technical": ["TCVN 9481:2012", "IEC 61215"],
        "environmental": ["QCVN 05:2023/BTNMT", "QCVN 26:2010/BTNMT"],
    },
    ProjectType.ENERGY_WIND: {
        "primary": ["QCVN 01:2021/BTNMT", "Nghị định 08/2022/NĐ-CP"],
        "technical": ["IEC 61400"],
        "environmental": ["QCVN 26:2010/BTNMT", "QCVN 05:2023/BTNMT"],
    },
    ProjectType.INDUSTRIAL_MANUFACTURING: {
        "primary": ["Luật BVMT 2020", "Nghị định 08/2022/NĐ-CP"],
        "technical": ["TCVN ISO 14001", "TCVN 5939:2005"],
        "environmental": [
            "QCVN 19:2009/BTNMT",  # Air emissions
            "QCVN 40:2011/BTNMT",  # Wastewater
            "QCVN 26:2010/BTNMT",  # Noise
        ],
    },
    ProjectType.INFRA_ROAD: {
        "primary": ["Luật BVMT 2020", "Luật Giao thông đường bộ"],
        "technical": ["TCVN 4054:2005", "22TCN 273-01"],
        "environmental": ["QCVN 05:2023/BTNMT", "QCVN 26:2010/BTNMT"],
    },
    ProjectType.URBAN_RESIDENTIAL: {
        "primary": ["Luật BVMT 2020", "Luật Xây dựng"],
        "technical": ["TCVN 4449:1987", "QCVN 01:2021/BXD"],
        "environmental": ["QCVN 14:2008/BTNMT", "QCVN 26:2010/BTNMT"],
    },
}

IMPACT_FACTORS_BY_PROJECT = {
    ProjectType.ENERGY_SOLAR: [
        ImpactCategory.LAND_USE,
        ImpactCategory.VISUAL_LANDSCAPE,
        ImpactCategory.BIODIVERSITY,
        ImpactCategory.SOIL,
        ImpactCategory.CLIMATE,
    ],
    ProjectType.ENERGY_WIND: [
        ImpactCategory.NOISE_VIBRATION,
        ImpactCategory.VISUAL_LANDSCAPE,
        ImpactCategory.BIODIVERSITY,
        ImpactCategory.LAND_USE,
    ],
    ProjectType.INDUSTRIAL_MANUFACTURING: [
        ImpactCategory.AIR_QUALITY,
        ImpactCategory.WATER_QUALITY,
        ImpactCategory.NOISE_VIBRATION,
        ImpactCategory.WASTE,
        ImpactCategory.TRAFFIC,
        ImpactCategory.HEALTH_SAFETY,
    ],
    ProjectType.INFRA_ROAD: [
        ImpactCategory.AIR_QUALITY,
        ImpactCategory.NOISE_VIBRATION,
        ImpactCategory.LAND_USE,
        ImpactCategory.TRAFFIC,
        ImpactCategory.SOCIOECONOMIC,
    ],
    ProjectType.URBAN_RESIDENTIAL: [
        ImpactCategory.WATER_QUALITY,
        ImpactCategory.WASTE,
        ImpactCategory.TRAFFIC,
        ImpactCategory.SOCIOECONOMIC,
        ImpactCategory.VISUAL_LANDSCAPE,
    ],
}


# =============================================================================
# EIA Section Templates
# =============================================================================

EIA_SECTIONS = [
    {
        "id": "1",
        "title": "MÔ TẢ TÓM TẮT DỰ ÁN",
        "title_en": "Project Description",
        "subsections": [
            {"id": "1.1", "title": "Tên dự án", "title_en": "Project Name"},
            {"id": "1.2", "title": "Chủ dự án", "title_en": "Project Owner"},
            {"id": "1.3", "title": "Vị trí địa lý", "title_en": "Location"},
            {"id": "1.4", "title": "Quy mô/công suất", "title_en": "Scale/Capacity"},
            {"id": "1.5", "title": "Công nghệ sản xuất", "title_en": "Technology"},
        ],
    },
    {
        "id": "2",
        "title": "ĐIỀU KIỆN TỰ NHIÊN, KINH TẾ - XÃ HỘI",
        "title_en": "Environmental and Socioeconomic Baseline",
        "subsections": [
            {"id": "2.1", "title": "Điều kiện tự nhiên", "title_en": "Natural Conditions"},
            {"id": "2.2", "title": "Điều kiện kinh tế - xã hội", "title_en": "Socioeconomic Conditions"},
            {"id": "2.3", "title": "Hiện trạng môi trường", "title_en": "Environmental Baseline"},
        ],
    },
    {
        "id": "3",
        "title": "ĐÁNH GIÁ TÁC ĐỘNG MÔI TRƯỜNG",
        "title_en": "Environmental Impact Assessment",
        "subsections": [
            {"id": "3.1", "title": "Đánh giá tác động giai đoạn chuẩn bị", "title_en": "Pre-construction Phase"},
            {"id": "3.2", "title": "Đánh giá tác động giai đoạn xây dựng", "title_en": "Construction Phase"},
            {"id": "3.3", "title": "Đánh giá tác động giai đoạn vận hành", "title_en": "Operation Phase"},
            {"id": "3.4", "title": "Đánh giá tác động khi kết thúc dự án", "title_en": "Decommissioning Phase"},
        ],
    },
    {
        "id": "4",
        "title": "BIỆN PHÁP PHÒNG NGỪA, GIẢM THIỂU TÁC ĐỘNG",
        "title_en": "Mitigation Measures",
        "subsections": [
            {"id": "4.1", "title": "Biện pháp giai đoạn chuẩn bị", "title_en": "Pre-construction Measures"},
            {"id": "4.2", "title": "Biện pháp giai đoạn xây dựng", "title_en": "Construction Measures"},
            {"id": "4.3", "title": "Biện pháp giai đoạn vận hành", "title_en": "Operation Measures"},
        ],
    },
    {
        "id": "5",
        "title": "CHƯƠNG TRÌNH QUẢN LÝ VÀ GIÁM SÁT MÔI TRƯỜNG",
        "title_en": "Environmental Management and Monitoring Program",
        "subsections": [
            {"id": "5.1", "title": "Chương trình quản lý môi trường", "title_en": "Environmental Management"},
            {"id": "5.2", "title": "Chương trình giám sát môi trường", "title_en": "Environmental Monitoring"},
        ],
    },
    {
        "id": "6",
        "title": "THAM VẤN CỘNG ĐỒNG",
        "title_en": "Public Consultation",
        "subsections": [
            {"id": "6.1", "title": "Kết quả tham vấn UBND xã", "title_en": "Consultation with Local Authorities"},
            {"id": "6.2", "title": "Kết quả tham vấn cộng đồng dân cư", "title_en": "Community Consultation"},
        ],
    },
]
