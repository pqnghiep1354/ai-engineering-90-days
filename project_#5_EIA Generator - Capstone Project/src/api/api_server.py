"""
FastAPI server for EIA Generator.

Provides REST API for generating EIA reports.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from loguru import logger

from ..config import ProjectInput, ProjectType, EIAConfig, EIAReport
from ..orchestrator import EIAOrchestrator
from ..generators.docx_generator import DocxGenerator


# =============================================================================
# Request/Response Models
# =============================================================================

class ProjectRequest(BaseModel):
    """Request model for project input."""
    
    name: str = Field(..., description="Project name")
    type: str = Field(..., description="Project type")
    location: str = Field(..., description="Project location")
    province: str = Field(default="", description="Province")
    area_hectares: float = Field(..., description="Area in hectares")
    capacity: str = Field(default="", description="Production capacity")
    investment_usd: float = Field(default=0, description="Investment in USD")
    construction_months: int = Field(default=24, description="Construction duration")
    operation_years: int = Field(default=20, description="Operation duration")
    investor_name: str = Field(default="", description="Investor name")
    description: str = Field(default="", description="Project description")


class GenerateRequest(BaseModel):
    """Request model for EIA generation."""
    
    project: ProjectRequest
    language: str = Field(default="vi", description="Report language")
    format: str = Field(default="full", description="Report format")


class ReportSummary(BaseModel):
    """Summary of generated report."""
    
    project_name: str
    generated_at: str
    compliance_score: float
    completeness_score: float
    sections_count: int
    download_url: Optional[str] = None


class GenerateResponse(BaseModel):
    """Response model for EIA generation."""
    
    success: bool
    message: str
    report_id: str
    summary: Optional[ReportSummary] = None
    errors: List[str] = []


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    timestamp: str


class ProjectTypesResponse(BaseModel):
    """Available project types response."""
    
    types: List[Dict[str, str]]


# =============================================================================
# Report Storage (In-memory for demo)
# =============================================================================

class ReportStorage:
    """Simple in-memory storage for generated reports."""
    
    def __init__(self):
        self.reports: Dict[str, EIAReport] = {}
        self.files: Dict[str, str] = {}
    
    def save(self, report_id: str, report: EIAReport, file_path: str = None):
        self.reports[report_id] = report
        if file_path:
            self.files[report_id] = file_path
    
    def get(self, report_id: str) -> Optional[EIAReport]:
        return self.reports.get(report_id)
    
    def get_file(self, report_id: str) -> Optional[str]:
        return self.files.get(report_id)


storage = ReportStorage()


# =============================================================================
# API Router
# =============================================================================

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["EIA Generator"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


@router.get("/project-types", response_model=ProjectTypesResponse)
async def get_project_types():
    """Get available project types."""
    types = [
        {"value": "energy_solar", "label": "Điện mặt trời", "label_en": "Solar Power"},
        {"value": "energy_wind", "label": "Điện gió", "label_en": "Wind Power"},
        {"value": "industrial_manufacturing", "label": "Sản xuất công nghiệp", "label_en": "Industrial Manufacturing"},
        {"value": "infrastructure_road", "label": "Đường giao thông", "label_en": "Road Infrastructure"},
        {"value": "urban_residential", "label": "Khu đô thị", "label_en": "Urban Residential"},
        {"value": "urban_industrial_zone", "label": "Khu công nghiệp", "label_en": "Industrial Zone"},
    ]
    return ProjectTypesResponse(types=types)


@router.post("/generate", response_model=GenerateResponse)
async def generate_eia(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate EIA report.
    
    This endpoint starts the EIA generation process.
    """
    try:
        # Validate project type
        try:
            project_type = ProjectType(request.project.type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid project type: {request.project.type}",
            )
        
        # Create project input
        project = ProjectInput(
            name=request.project.name,
            type=project_type,
            location=request.project.location,
            province=request.project.province,
            area_hectares=request.project.area_hectares,
            capacity=request.project.capacity,
            investment_usd=request.project.investment_usd,
            construction_months=request.project.construction_months,
            operation_years=request.project.operation_years,
            investor_name=request.project.investor_name,
            description=request.project.description,
        )
        
        # Create config
        config = EIAConfig(
            language=request.language,
            format=request.format,
        )
        
        # Generate report ID
        report_id = f"eia_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate report
        logger.info(f"Starting EIA generation: {report_id}")
        orchestrator = EIAOrchestrator(config)
        report = await orchestrator.generate(project)
        
        # Generate DOCX
        output_path = f"outputs/{report_id}.docx"
        generator = DocxGenerator()
        generator.generate(report, output_path)
        
        # Save to storage
        storage.save(report_id, report, output_path)
        
        # Create response
        summary = ReportSummary(
            project_name=project.name,
            generated_at=report.generated_at,
            compliance_score=report.compliance_score,
            completeness_score=report.completeness_score,
            sections_count=len(report.sections),
            download_url=f"/api/v1/download/{report_id}",
        )
        
        return GenerateResponse(
            success=True,
            message="EIA report generated successfully",
            report_id=report_id,
            summary=summary,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return GenerateResponse(
            success=False,
            message="Failed to generate EIA report",
            report_id="",
            errors=[str(e)],
        )


@router.get("/report/{report_id}")
async def get_report(report_id: str):
    """Get report details."""
    report = storage.get(report_id)
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {
        "report_id": report_id,
        "project_name": report.project.name,
        "generated_at": report.generated_at,
        "compliance_score": report.compliance_score,
        "executive_summary": report.executive_summary[:500] + "...",
        "sections": [
            {"id": s.id, "title": s.title}
            for s in report.sections
        ],
    }


@router.get("/download/{report_id}")
async def download_report(report_id: str):
    """Download generated report."""
    file_path = storage.get_file(report_id)
    
    if not file_path:
        raise HTTPException(status_code=404, detail="Report file not found")
    
    return FileResponse(
        path=file_path,
        filename=f"EIA_Report_{report_id}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# =============================================================================
# FastAPI Application
# =============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application."""
    
    app = FastAPI(
        title="EIA Generator API",
        description="""
        API for generating Environmental Impact Assessment (EIA) reports.
        
        ## Features
        - Generate comprehensive EIA reports
        - Support multiple project types
        - Vietnamese and English output
        - Download as DOCX
        
        ## Usage
        1. POST /api/v1/generate with project details
        2. GET /api/v1/download/{report_id} to download
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include router
    app.include_router(router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "name": "EIA Generator API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health",
        }
    
    return app


# =============================================================================
# Run Server
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
