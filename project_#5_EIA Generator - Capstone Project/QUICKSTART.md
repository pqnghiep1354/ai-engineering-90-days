# 🚀 Hướng Dẫn Nhanh - EIA Generator (Capstone Project)

## Tổng Quan

EIA Generator là hệ thống AI tự động tạo Báo cáo Đánh giá Tác động Môi trường (ĐTM) theo quy định Việt Nam.

## Yêu Cầu

- Python 3.10+
- OpenAI API Key
- 8GB RAM trở lên

## Cài Đặt

```bash
# 1. Clone repository
git clone https://github.com/yourusername/eia-generator.git
cd eia-generator

# 2. Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Cấu hình
cp .env.example .env
# Mở file .env và thêm OPENAI_API_KEY
```

## Sử Dụng

### Cách 1: Web Interface (Streamlit)

```bash
streamlit run app.py
```

Truy cập: http://localhost:8501

### Cách 2: Command Line

```bash
# Chế độ tương tác
python -m src.main --interactive

# Chế độ tham số
python -m src.main \
    --project "Nhà máy điện mặt trời ABC" \
    --location "Ninh Thuận" \
    --type energy_solar \
    --area 100 \
    --output outputs/eia_report.docx
```

### Cách 3: Python API

```python
from src.orchestrator import EIAOrchestrator
from src.config import ProjectInput, ProjectType

# Định nghĩa dự án
project = ProjectInput(
    name="Nhà máy điện mặt trời ABC",
    type=ProjectType.ENERGY_SOLAR,
    location="Xã A, Huyện B, Tỉnh Ninh Thuận",
    area_hectares=100,
    capacity="50 MW",
    investment_usd=40_000_000,
)

# Tạo báo cáo
import asyncio
orchestrator = EIAOrchestrator()
report = asyncio.run(orchestrator.generate(project))

# Xuất file
from src.generators.docx_generator import DocxGenerator
generator = DocxGenerator()
generator.generate(report, "outputs/eia_report.docx")
```

### Cách 4: REST API

```bash
# Khởi động server
uvicorn src.api.api_server:create_app --factory --host 0.0.0.0 --port 8000
```

```bash
# Tạo báo cáo
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "project": {
      "name": "Nhà máy điện mặt trời ABC",
      "type": "energy_solar",
      "location": "Ninh Thuận",
      "area_hectares": 100
    }
  }'
```

## Loại Dự Án Hỗ Trợ

| Loại | Mã | Quy định chính |
|------|-----|----------------|
| Điện mặt trời | energy_solar | QCVN 05, 26 |
| Điện gió | energy_wind | QCVN 26, 05 |
| Sản xuất công nghiệp | industrial_manufacturing | QCVN 19, 40, 26 |
| Đường giao thông | infrastructure_road | QCVN 05, 26 |
| Khu đô thị | urban_residential | QCVN 14, 26 |

## Cấu Trúc Báo Cáo

1. **Chương 1**: Mô tả dự án
2. **Chương 2**: Điều kiện tự nhiên, KT-XH
3. **Chương 3**: Đánh giá tác động môi trường
4. **Chương 4**: Biện pháp giảm thiểu
5. **Chương 5**: Chương trình giám sát
6. **Chương 6**: Tham vấn cộng đồng

## Docker

```bash
# Build
docker build -t eia-generator .

# Run Streamlit
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  eia-generator

# Docker Compose
docker-compose up -d
```

## Troubleshooting

### "OpenAI API error"
- Kiểm tra OPENAI_API_KEY trong file .env
- Kiểm tra quota API

### "Generation timeout"
- Tăng timeout trong configs/agents.yaml
- Giảm MAX_SECTION_TOKENS

### "Low compliance score"
- Kiểm tra thông tin dự án đầy đủ
- Chạy lại với model gpt-4o

## Lưu Ý Quan Trọng

⚠️ **Báo cáo được tạo tự động chỉ mang tính tham khảo.**

Trước khi nộp cơ quan thẩm định:
- Bổ sung số liệu quan trắc thực tế
- Kiểm tra và chỉnh sửa nội dung
- Tham vấn chuyên gia môi trường
- Hoàn thiện phụ lục theo quy định

---

📧 Hỗ trợ: support@eia-generator.vn

🌍 **Portfolio Project #5** - AI Engineer Environmental Specialization
