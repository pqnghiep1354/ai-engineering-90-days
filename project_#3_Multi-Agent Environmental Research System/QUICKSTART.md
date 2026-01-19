# 🚀 Hướng Dẫn Nhanh - Multi-Agent Research System

## Bước 1: Cài đặt

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## Bước 2: Cấu hình API Keys

```bash
cp .env.example .env
```

Mở file `.env` và thêm:
```
OPENAI_API_KEY=sk-your-openai-key
TAVILY_API_KEY=tvly-your-tavily-key  # Optional, dùng cho web search
```

**Lấy API Keys:**
- OpenAI: https://platform.openai.com/api-keys
- Tavily (free): https://tavily.com/

## Bước 3: Chạy thử

### Option A: Command Line
```bash
# Quick research
python -m src.main "Climate change impacts in Vietnam"

# Deep research
python -m src.main "ESG trends 2024" --workflow deep

# Interactive mode
python -m src.main --interactive
```

### Option B: Web Interface
```bash
streamlit run src/app.py
```
Mở browser: http://localhost:8501

## Bước 4: Các topic mẫu

**English:**
- "What are the latest developments in carbon capture technology?"
- "Compare renewable energy policies: EU vs US"
- "Impact of microplastics on marine ecosystems"

**Vietnamese:**
- "Tác động của biến đổi khí hậu đến Đồng bằng sông Cửu Long"
- "Xu hướng ESG tại Việt Nam năm 2024"

## Workflows

| Workflow | Thời gian | Mô tả |
|----------|-----------|-------|
| `quick` | 2-5 phút | Nghiên cứu nhanh, 5-10 nguồn |
| `deep` | 10-20 phút | Phân tích sâu + fact-checking |

## Agents trong hệ thống

1. **🔍 Researcher**: Thu thập thông tin từ web
2. **📊 Analyst**: Phân tích và tìm insights
3. **✍️ Writer**: Viết báo cáo có cấu trúc
4. **✓ Fact-Checker**: Xác minh thông tin

## Output

Reports được lưu trong `data/reports/` với format Markdown.

## Troubleshooting

### "API key not configured"
→ Kiểm tra file `.env` có đúng key không

### "Rate limit exceeded"
→ Chờ một lúc hoặc giảm `max_sources`

### Import errors
→ Chạy lại `pip install -r requirements.txt`

---

**🎯 Portfolio Project #3** - Multi-Agent Environmental Research System
