# 🌍 Lộ Trình 90 Ngày AI Engineering - Environmental Solutions

Chào mừng bạn đến với kho lưu trữ tổng hợp các dự án trong **Lộ trình 90 ngày trở thành AI Engineer** chuyên sâu về lĩnh vực **Môi trường**. Đây là nơi tập hợp các giải pháp AI thực tế từ tìm kiếm ngữ nghĩa, RAG, Multi-Agent cho đến Fine-tuning mô hình ngôn ngữ lớn (LLM).

---

## 🚀 Danh Sách Các Dự Án

Kho lưu trữ này bao gồm 5 dự án trọng điểm, được sắp xếp theo cấp độ tăng dần về kỹ thuật:

### [1. Environmental Semantic Search](file:///d:/Nghiep_works/0.%20Agentic%20AI/0_lo_trinh_90_ngay_AI_enginering_env/project_#1_env-semantic-search)
- **Mô tả**: Hệ thống tìm kiếm ngữ nghĩa chuyên sâu cho dữ liệu môi trường.
- **Công nghệ**: Gemini API, Sentence Transformers, ChromaDB.
- **Tính năng**: Tìm kiếm văn bản dựa trên ý nghĩa thay vì từ khóa, hỗ trợ đa ngôn ngữ.

### [2. Climate QA RAG System](file:///d:/Nghiep_works/0.%20Agentic%20AI/0_lo_trinh_90_ngay_AI_enginering_env/project_#2_climate-qa-rag)
- **Mô tả**: Hệ thống hỏi đáp thông minh về biến đổi khí hậu sử dụng kỹ thuật RAG (Retrieval-Augmented Generation).
- **Công nghệ**: LangChain, OpenAI/Gemini, Vector Database.
- **Tính năng**: Truy xuất thông tin chính xác từ các tài liệu khoa học về khí hậu.

### [3. Multi-Agent Environmental Research System](file:///d:/Nghiep_works/0.%20Agentic%20AI/0_lo_trinh_90_ngay_AI_enginering_env/project_#3_Multi-Agent%20Environmental%20Research%20System)
- **Mô tả**: Hệ thống đại lý đa tác nhân thực hiện các nghiên cứu môi trường phức tạp.
- **Công nghệ**: LangGraph, CrewAI, Tavily Search.
- **Tính năng**: Phân chia nhiệm vụ giữa các AI Agent (Research, Writer, Reviewer) để tạo báo cáo nghiên cứu.

### [4. Environmental Domain LLM Fine-tuning](file:///d:/Nghiep_works/0.%20Agentic%20AI/0_lo_trinh_90_ngay_AI_enginering_env/project_#4_Environmental%20Domain%20LLM%20Fine-tuning)
- **Mô tả**: Tinh chỉnh mô hình ngôn ngữ lớn trên tập dữ liệu chuyên ngành môi trường.
- **Công nghệ**: Hugging Face, PyTorch, LoRA/QLoRA, Phi-1/Mistral.
- **Tính năng**: Mô hình hiểu sâu các khái niệm và thuật ngữ môi trường Việt Nam.

### [5. EIA Generator - Capstone Project](file:///d:/Nghiep_works/0.%20Agentic%20AI/0_lo_trinh_90_ngay_AI_enginering_env/project_#5_EIA%20Generator%20-%20Capstone%20Project) ⭐
- **Mô tả**: Hệ thống tự động khởi tạo báo cáo Đánh giá Tác động Môi trường (ĐTM).
- **Công nghệ**: PhoBERT, Qwen2.5 (Local), RAG, Streamlit.
- **Tính năng**: Tự động hóa quy trình viết báo cáo ĐTM tuân thủ quy định pháp luật Việt Nam.

---

## 🛠️ Cấu Trình CI/CD Monorepo

Dự án này sử dụng GitHub Actions với cấu trúc **Monorepo CI/CD** tối ưu:
- **Tách biệt Jobs**: Mỗi dự án chạy CI riêng biệt khi có sự thay đổi trong thư mục tương ứng.
- **Auto PR & Merge**: Tự động tạo Pull Request và Merge vào nhánh `main` khi vượt qua các bài kiểm tra chất lượng.
- **Validation**: Tích hợp kiểm tra logic (tests) và định dạng mã nguồn (linting).

---

## 💻 Hướng Dẫn Cài Đặt Chung

1. **Clone repository**:
   ```bash
   git clone https://github.com/pqnghiep1354/ai-engineering-90-days.git
   cd ai-engineering-90-days
   ```

2. **Cấu hình môi trường**:
   - Tạo file `.env` tại thư mục gốc hoặc từng thư mục dự án.
   - Cung cấp các API Key cần thiết (`GOOGLE_API_KEY`, `OPENAI_API_KEY`, v.v.).

3. **Cài đặt thư viện**:
   Truy cập vào từng folder dự án và cài đặt theo `README.md` riêng:
   ```bash
   pip install -r project_#X_.../requirements.txt
   ```

---

## 📬 Liên Hệ
- **Tác giả**: [pqnghiep1354](https://github.com/pqnghiep1354)
- **Lộ trình**: 90 Ngày AI Engineering cho ngành Môi trường.

---
*Dự án được phát triển với mục đích học thuật và ứng dụng AI vào bảo vệ môi trường Việt Nam.*