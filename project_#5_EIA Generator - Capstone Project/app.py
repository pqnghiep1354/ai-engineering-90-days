"""
Streamlit Web Interface for EIA Generator.

Run with: streamlit run app.py
"""

import asyncio
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

# Page config
st.set_page_config(
    page_title="EIA Generator - Tạo Báo cáo ĐTM",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import after page config
from src.config import ProjectInput, ProjectType, EIAConfig
from src.orchestrator import EIAOrchestrator
from src.generators.docx_generator import DocxGenerator


# =============================================================================
# Session State
# =============================================================================

if "report" not in st.session_state:
    st.session_state.report = None
if "generating" not in st.session_state:
    st.session_state.generating = False


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/environment-care.png", width=80)
    st.title("EIA Generator")
    st.markdown("*Hệ thống tạo Báo cáo Đánh giá Tác động Môi trường*")
    
    st.divider()
    
    # API Key (optional for Ollama)
    api_key = st.text_input(
        "Google API Key (không cần nếu dùng Ollama)",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Nhập Google API key cho Gemini, hoặc để trống nếu dùng Ollama local",
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.divider()
    
    # Config
    st.subheader("⚙️ Cấu hình")
    
    language = st.selectbox(
        "Ngôn ngữ báo cáo",
        ["Tiếng Việt", "English", "Song ngữ"],
        index=0,
    )
    
    # Model selection with provider groups
    model_options = [
        # Gemini (Cloud - requires API key)
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        # Ollama (Local - free, no API key)
        "qwen2.5:7b",  # Best for Vietnamese
        "gemma3:4b",
        "gemma3:12b",
        "llama3.2:3b",
        "mistral:7b",
        "phi3:mini",
    ]
    
    model = st.selectbox(
        "Model AI",
        model_options,
        index=0,
        help="Gemini: cần API key | Ollama: cần Ollama đang chạy (ollama serve)",
    )
    
    # Show Ollama hint if local model selected
    if ":" in model or model.startswith(("gemma", "llama", "mistral", "phi", "qwen")):
        st.info("💻 Đang dùng Ollama local. Đảm bảo: `ollama serve` đang chạy")
    
    st.divider()
    
    st.markdown("""
    ### 📚 Hướng dẫn
    1. Nhập thông tin dự án
    2. Bấm "Tạo báo cáo"
    3. Tải xuống file DOCX
    
    ### 📋 Tài liệu
    - [Luật BVMT 2020](https://thuvienphapluat.vn)
    - [Nghị định 08/2022](https://thuvienphapluat.vn)
    """)


# =============================================================================
# Main Content
# =============================================================================

st.title("🌍 EIA Generator")
st.markdown("### Hệ thống Tạo Báo cáo Đánh giá Tác động Môi trường")

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Nhập dự án", "📊 Kết quả", "📖 Hướng dẫn"])

# =============================================================================
# Tab 1: Project Input
# =============================================================================

with tab1:
    st.header("Thông tin dự án")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏭 Thông tin cơ bản")
        
        project_name = st.text_input(
            "Tên dự án *",
            placeholder="Nhà máy điện mặt trời ABC",
        )
        
        project_type = st.selectbox(
            "Loại dự án *",
            options=[
                ("Điện mặt trời", ProjectType.ENERGY_SOLAR),
                ("Điện gió", ProjectType.ENERGY_WIND),
                ("Sản xuất công nghiệp", ProjectType.INDUSTRIAL_MANUFACTURING),
                ("Đường giao thông", ProjectType.INFRA_ROAD),
                ("Khu đô thị", ProjectType.URBAN_RESIDENTIAL),
                ("Khu công nghiệp", ProjectType.URBAN_INDUSTRIAL_ZONE),
            ],
            format_func=lambda x: x[0],
        )
        
        description = st.text_area(
            "Mô tả dự án",
            placeholder="Mô tả ngắn gọn về dự án...",
            height=100,
        )
        
        st.subheader("📍 Vị trí")
        
        location = st.text_input(
            "Địa điểm *",
            placeholder="Xã X, Huyện Y, Tỉnh Z",
        )
        
        province = st.selectbox(
            "Tỉnh/Thành phố",
            ["Ninh Thuận", "Bình Thuận", "Bình Dương", "Đồng Nai", 
             "TP. Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Khác"],
        )
    
    with col2:
        st.subheader("📐 Quy mô")
        
        area = st.number_input(
            "Diện tích (ha) *",
            min_value=0.1,
            value=50.0,
            step=1.0,
        )
        
        capacity = st.text_input(
            "Công suất",
            placeholder="100 MW / 10,000 tấn/năm",
        )
        
        investment = st.number_input(
            "Vốn đầu tư (triệu USD)",
            min_value=0.0,
            value=10.0,
            step=1.0,
        )
        
        construction_months = st.slider(
            "Thời gian xây dựng (tháng)",
            min_value=6,
            max_value=60,
            value=18,
        )
        
        operation_years = st.slider(
            "Thời gian vận hành (năm)",
            min_value=5,
            max_value=50,
            value=20,
        )
        
        st.subheader("👤 Chủ đầu tư")
        
        investor_name = st.text_input(
            "Tên chủ đầu tư",
            placeholder="Công ty TNHH ABC",
        )
        
        investor_address = st.text_input(
            "Địa chỉ",
            placeholder="123 Đường XYZ, Quận 1, TP.HCM",
        )
    
    st.divider()
    
    # Generate button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        generate_clicked = st.button(
            "🚀 Tạo Báo cáo ĐTM",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.generating,
        )
    
    # Generate report
    if generate_clicked:
        if not project_name or not location:
            st.error("⚠️ Vui lòng nhập Tên dự án và Địa điểm")
        elif not api_key and model.startswith("gemini"):
            st.error("⚠️ Vui lòng nhập Google API Key để dùng Gemini")
        else:
            st.session_state.generating = True
            
            # Create project input
            project = ProjectInput(
                name=project_name,
                type=project_type[1],
                description=description,
                location=location,
                province=province,
                area_hectares=area,
                capacity=capacity,
                investment_usd=investment * 1_000_000,
                construction_months=construction_months,
                operation_years=operation_years,
                investor_name=investor_name,
                investor_address=investor_address,
            )
            
            # Create config
            config = EIAConfig(
                model=model,
                language="vi" if "Việt" in language else "en",
            )
            
            # Progress
            progress_bar = st.progress(0, text="Đang khởi tạo...")
            status_text = st.empty()
            
            try:
                # Generate
                status_text.info("🔍 Đang nghiên cứu quy định pháp luật...")
                progress_bar.progress(10, text="Nghiên cứu quy định...")
                
                orchestrator = EIAOrchestrator(config)
                
                status_text.info("📊 Đang phân tích môi trường nền...")
                progress_bar.progress(30, text="Phân tích môi trường nền...")
                
                # Run async
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                status_text.info("⚡ Đang đánh giá tác động...")
                progress_bar.progress(50, text="Đánh giá tác động...")
                
                report = loop.run_until_complete(orchestrator.generate(project))
                
                progress_bar.progress(90, text="Hoàn thiện báo cáo...")
                
                st.session_state.report = report
                progress_bar.progress(100, text="Hoàn thành!")
                status_text.success("✅ Tạo báo cáo thành công!")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
            finally:
                st.session_state.generating = False


# =============================================================================
# Tab 2: Results
# =============================================================================

with tab2:
    if st.session_state.report:
        report = st.session_state.report
        
        st.header("📊 Kết quả")
        
        # Score metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Điểm tổng",
                f"{report.compliance_score:.1f}/100",
                delta="Đạt" if report.compliance_score >= 70 else "Chưa đạt",
            )
        
        with col2:
            st.metric(
                "Hoàn thiện",
                f"{report.completeness_score:.1f}%",
            )
        
        with col3:
            st.metric(
                "Số chương",
                len(report.sections),
            )
        
        with col4:
            st.metric(
                "Ngày tạo",
                datetime.now().strftime("%d/%m/%Y"),
            )
        
        st.divider()
        
        # Executive Summary
        st.subheader("📋 Tóm tắt")
        with st.expander("Xem tóm tắt báo cáo", expanded=True):
            st.markdown(report.executive_summary)
        
        # Sections
        st.subheader("📑 Các chương")
        for section in report.sections:
            with st.expander(f"Chương {section.id}: {section.title}"):
                if section.content:
                    st.markdown(section.content[:2000] + "..." if len(section.content) > 2000 else section.content)
                else:
                    st.info("Nội dung đang được tạo...")
        
        # Validation notes
        if report.validation_notes:
            st.subheader("💡 Đề xuất cải thiện")
            for note in report.validation_notes:
                st.warning(note)
        
        st.divider()
        
        # Download
        st.subheader("📥 Tải xuống")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Tạo file DOCX", use_container_width=True):
                try:
                    output_path = f"outputs/eia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
                    generator = DocxGenerator()
                    generator.generate(report, output_path)
                    
                    with open(output_path, "rb") as f:
                        st.download_button(
                            "⬇️ Tải DOCX",
                            data=f,
                            file_name=f"EIA_{report.project.name}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True,
                        )
                    st.success("✅ File đã sẵn sàng!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        
        with col2:
            # JSON export
            import json
            json_data = json.dumps(report.to_dict(), ensure_ascii=False, indent=2, default=str)
            st.download_button(
                "📋 Tải JSON",
                data=json_data,
                file_name=f"EIA_{report.project.name}.json",
                mime="application/json",
                use_container_width=True,
            )
    
    else:
        st.info("👈 Nhập thông tin dự án và bấm 'Tạo Báo cáo ĐTM' để bắt đầu")


# =============================================================================
# Tab 3: Guide
# =============================================================================

with tab3:
    st.header("📖 Hướng dẫn sử dụng")
    
    st.markdown("""
    ## 1. Giới thiệu
    
    EIA Generator là hệ thống AI tự động tạo Báo cáo Đánh giá Tác động Môi trường (ĐTM) 
    theo quy định của Luật Bảo vệ Môi trường 2020 và Nghị định 08/2022/NĐ-CP.
    
    ## 2. Các bước sử dụng
    
    ### Bước 1: Chuẩn bị
    - Có Google API Key (Gemini)
    - Thu thập thông tin dự án
    
    ### Bước 2: Nhập thông tin
    - Điền đầy đủ thông tin trong tab "Nhập dự án"
    - Các trường có dấu (*) là bắt buộc
    
    ### Bước 3: Tạo báo cáo
    - Bấm nút "Tạo Báo cáo ĐTM"
    - Chờ hệ thống xử lý (5-10 phút)
    
    ### Bước 4: Xem và tải
    - Xem kết quả trong tab "Kết quả"
    - Tải file DOCX hoặc JSON
    
    ## 3. Cấu trúc báo cáo
    
    Báo cáo ĐTM bao gồm 6 chương:
    
    1. **Mô tả dự án** - Thông tin cơ bản về dự án
    2. **Điều kiện môi trường nền** - Hiện trạng môi trường khu vực
    3. **Đánh giá tác động** - Phân tích các tác động môi trường
    4. **Biện pháp giảm thiểu** - Các biện pháp bảo vệ môi trường
    5. **Chương trình giám sát** - Kế hoạch quan trắc môi trường
    6. **Tham vấn cộng đồng** - Kết quả tham vấn
    
    ## 4. Lưu ý quan trọng
    
    ⚠️ **Báo cáo được tạo tự động chỉ mang tính tham khảo.**
    
    Trước khi nộp cơ quan thẩm định, cần:
    - Bổ sung số liệu quan trắc thực tế
    - Kiểm tra và chỉnh sửa nội dung
    - Có ý kiến của chuyên gia môi trường
    - Hoàn thiện phụ lục theo quy định
    
    ## 5. Liên hệ hỗ trợ
    
    📧 Email: support@eia-generator.vn
    📞 Hotline: 1900-xxxx
    """)


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    🌍 EIA Generator v1.0 | Portfolio Project #5 | AI Engineer | 2024
</div>
""", unsafe_allow_html=True)
