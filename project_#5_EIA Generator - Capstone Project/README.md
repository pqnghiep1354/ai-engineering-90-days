# 🌍 EIA Generator - Environmental Impact Assessment System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜-LangChain-green.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Capstone Project**: AI-powered system for generating comprehensive Environmental Impact Assessment (EIA) reports compliant with Vietnamese and international regulations.

## 🎯 Overview

EIA Generator là hệ thống AI tự động tạo Báo cáo Đánh giá Tác động Môi trường (ĐTM), tích hợp tất cả kỹ năng từ các dự án portfolio trước:

| Component | Source Project | Function |
|-----------|---------------|----------|
| Knowledge RAG | Project 2 | Retrieve regulations & standards |
| Multi-Agent | Project 3 | Specialized section writers |
| Document Gen | Project 1 | Semantic search & templates |
| Validation | Project 4 | Compliance checking |

### What is EIA?

Environmental Impact Assessment (Đánh giá tác động môi trường - ĐTM) là quy trình đánh giá tác động tiềm tàng của dự án đến môi trường, bắt buộc theo pháp luật Việt Nam (Luật Bảo vệ Môi trường 2020) và quốc tế.

## ✨ Features

### Core Capabilities
- 🤖 **Multi-Agent Architecture**: 6 specialized agents for different EIA sections
- 📚 **RAG-powered Research**: Retrieve relevant regulations, standards, and case studies
- 📝 **Template Engine**: Generate professional Word documents
- ✅ **Compliance Validator**: Check against Vietnamese regulations (Luật BVMT 2020)
- 🌐 **Bilingual Support**: Vietnamese and English output

### EIA Report Sections Generated
1. **Mô tả dự án** (Project Description)
2. **Điều kiện tự nhiên & KT-XH** (Baseline Environment)
3. **Đánh giá tác động** (Impact Assessment)
4. **Biện pháp giảm thiểu** (Mitigation Measures)
5. **Chương trình quản lý & giám sát** (Monitoring Program)
6. **Tham vấn cộng đồng** (Public Consultation)

### Supported Project Types
- 🏭 Industrial facilities (Manufacturing, Processing)
- 🏗️ Construction projects (Buildings, Infrastructure)
- ⚡ Energy projects (Solar, Wind, Thermal)
- 🛣️ Transportation (Roads, Ports, Airports)
- 🏥 Urban development (Residential, Commercial)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EIA GENERATOR SYSTEM                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   User Input    │      │  Knowledge Base │      │    Templates    │
│  ─────────────  │      │  ─────────────  │      │  ─────────────  │
│ • Project info  │      │ • Regulations   │      │ • EIA format    │
│ • Location      │      │ • Standards     │      │ • DOCX styles   │
│ • Scale/Type    │      │ • Case studies  │      │ • Charts/Tables │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR AGENT                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        LangGraph Workflow                               │ │
│  │                                                                         │ │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐  │ │
│  │  │Research │──▶│Baseline │──▶│ Impact  │──▶│Mitiga-  │──▶│Monitor- │  │ │
│  │  │ Agent   │   │ Agent   │   │ Agent   │   │ tion    │   │  ing    │  │ │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘  │ │
│  │       │             │             │             │             │        │ │
│  │       ▼             ▼             ▼             ▼             ▼        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    VALIDATOR AGENT                               │  │ │
│  │  │  • Regulation compliance  • Completeness check  • Quality score │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DOCUMENT GENERATOR                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │    DOCX      │    │     PDF      │    │    JSON      │                   │
│  │   Report     │    │   Export     │    │   Metadata   │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
eia-generator/
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── orchestrator.py         # Main workflow orchestrator
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py             # Base agent class
│   │   ├── research_agent.py   # Regulation research
│   │   ├── baseline_agent.py   # Environmental baseline
│   │   ├── impact_agent.py     # Impact assessment
│   │   ├── mitigation_agent.py # Mitigation measures
│   │   ├── monitoring_agent.py # Monitoring program
│   │   └── validator_agent.py  # Compliance validation
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── rag_tool.py         # Knowledge retrieval
│   │   ├── web_search.py       # Web research
│   │   ├── calculator.py       # Impact calculations
│   │   └── gis_tool.py         # Location analysis
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── docx_generator.py   # Word document generation
│   │   ├── pdf_generator.py    # PDF export
│   │   └── chart_generator.py  # Charts and figures
│   ├── templates/
│   │   ├── __init__.py
│   │   └── eia_template.py     # EIA structure templates
│   └── validators/
│       ├── __init__.py
│       ├── compliance.py       # Regulation compliance
│       └── quality.py          # Quality scoring
├── data/
│   ├── regulations/            # Vietnamese regulations
│   ├── templates/              # Document templates
│   ├── examples/               # Sample EIAs
│   └── knowledge_base/         # RAG knowledge base
├── configs/
│   ├── agents.yaml             # Agent configurations
│   └── regulations.yaml        # Regulation mappings
├── tests/
│   └── test_eia_generator.py
├── docs/
│   ├── architecture.md
│   └── user_guide.md
├── outputs/                    # Generated reports
├── app.py                      # Streamlit web interface
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/eia-generator.git
cd eia-generator

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Add your OPENAI_API_KEY and TAVILY_API_KEY
```

### Generate EIA Report

```bash
# CLI
python -m src.main \
    --project "Solar Power Plant" \
    --location "Ninh Thuận, Vietnam" \
    --capacity "100MW" \
    --output outputs/eia_report.docx

# Web Interface
streamlit run app.py
```

### Python API

```python
from src.orchestrator import EIAOrchestrator
from src.config import ProjectInput

# Define project
project = ProjectInput(
    name="Nhà máy điện mặt trời ABC",
    type="energy_solar",
    location="Xã X, Huyện Y, Tỉnh Ninh Thuận",
    capacity="100 MW",
    area_hectares=200,
    investment_usd=80_000_000,
    construction_months=18,
    operation_years=25,
)

# Generate EIA
orchestrator = EIAOrchestrator()
report = await orchestrator.generate(project)

# Export
report.to_docx("outputs/eia_solar_plant.docx")
report.to_pdf("outputs/eia_solar_plant.pdf")
```

## 📊 Sample Output

### Executive Summary (Generated)

```
BÁO CÁO ĐÁNH GIÁ TÁC ĐỘNG MÔI TRƯỜNG
DỰ ÁN NHÀ MÁY ĐIỆN MẶT TRỜI ABC

1. GIỚI THIỆU
   Dự án Nhà máy điện mặt trời ABC với công suất 100 MW, 
   tổng vốn đầu tư 80 triệu USD, được đề xuất xây dựng 
   tại xã X, huyện Y, tỉnh Ninh Thuận...

2. TÁC ĐỘNG CHÍNH
   ✓ Tác động tích cực: Giảm 150,000 tấn CO2/năm
   ⚠ Tác động cần giảm thiểu: Sử dụng đất, cảnh quan

3. BIỆN PHÁP GIẢM THIỂU
   • Bảo tồn lớp đất mặt trong giai đoạn thi công
   • Trồng cây xanh xung quanh khu vực dự án
   • Lắp đặt hệ thống thoát nước mưa...

4. KẾT LUẬN
   Dự án đáp ứng các yêu cầu về bảo vệ môi trường 
   theo Luật BVMT 2020 và các quy chuẩn liên quan.
```

## 🔧 Configuration

### Project Types

```yaml
# configs/project_types.yaml
project_types:
  energy_solar:
    name: "Điện mặt trời"
    regulations: ["QCVN 01:2021/BTNMT", "TCVN 9481:2012"]
    impact_factors: ["land_use", "visual", "biodiversity"]
    
  industrial_manufacturing:
    name: "Sản xuất công nghiệp"
    regulations: ["QCVN 19:2009/BTNMT", "QCVN 40:2011/BTNMT"]
    impact_factors: ["air", "water", "waste", "noise"]
```

### Agent Settings

```yaml
# configs/agents.yaml
agents:
  research:
    model: "gpt-4o"
    temperature: 0.3
    tools: ["rag", "web_search"]
    
  impact:
    model: "gpt-4o"
    temperature: 0.4
    tools: ["calculator", "rag"]
```

## 📚 Knowledge Base

The system includes a RAG-powered knowledge base with:

| Category | Content |
|----------|---------|
| **Regulations** | Luật BVMT 2020, Nghị định 08/2022, QCVN |
| **Standards** | TCVN, IFC EHS Guidelines, World Bank |
| **Templates** | Sample EIA sections, tables, figures |
| **Case Studies** | Approved EIA reports for reference |

## 🎯 Compliance Validation

The Validator Agent checks against:

- ✅ **Legal Requirements**: Luật BVMT 2020, Nghị định 08/2022
- ✅ **Technical Standards**: QCVN, TCVN
- ✅ **Structure Completeness**: All required sections present
- ✅ **Data Validity**: Calculations and references
- ✅ **Format Standards**: BTNMT template compliance

## 🐳 Docker Deployment

```bash
# Build
docker build -t eia-generator .

# Run
docker run -p 8501:8501 \
    -e OPENAI_API_KEY=your_key \
    -v $(pwd)/outputs:/app/outputs \
    eia-generator
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Report Generation Time | 5-10 minutes |
| Sections Generated | 6 main + appendices |
| Pages (typical) | 50-100 pages |
| Compliance Score | 85-95% |
| Languages | Vietnamese, English |

## 🛣️ Roadmap

- [ ] GIS integration for location analysis
- [ ] Historical data comparison
- [ ] Automatic permit tracking
- [ ] Multi-project management
- [ ] AI-powered revision suggestions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Vietnamese Ministry of Natural Resources and Environment (MONRE)
- World Bank EIA Guidelines
- IFC Environmental and Social Performance Standards

---

⭐ **Portfolio Capstone Project** - EIA Generator

**Demonstrates:**
- Multi-agent AI system design
- RAG-powered knowledge retrieval
- Professional document generation
- Regulatory compliance validation
- Full-stack AI application development
- Environmental domain expertise
