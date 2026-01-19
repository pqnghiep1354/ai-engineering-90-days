"""
RAG Tool for EIA Knowledge Retrieval.

Retrieves relevant regulations, standards, and guidelines from knowledge base.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger

from ..config import get_settings


# =============================================================================
# Knowledge Base
# =============================================================================

class KnowledgeBase:
    """
    Knowledge base for EIA regulations and standards.
    
    Contains:
    - Vietnamese environmental laws (Luật BVMT 2020)
    - Environmental regulations (QCVN, TCVN)
    - International standards (IFC, World Bank)
    - EIA guidelines and templates
    """
    
    def __init__(self, data_dir: str = "./data/knowledge_base"):
        self.data_dir = Path(data_dir)
        self.documents: List[Document] = []
        
        # Initialize embeddings
        settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        
        self.vectorstore: Optional[Chroma] = None
    
    def load_regulations(self) -> None:
        """Load regulations from data directory."""
        regulations_dir = self.data_dir / "regulations"
        
        if not regulations_dir.exists():
            logger.warning(f"Regulations directory not found: {regulations_dir}")
            self._load_builtin_regulations()
            return
        
        for file_path in regulations_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                for item in data:
                    doc = Document(
                        page_content=item.get("content", ""),
                        metadata={
                            "source": str(file_path),
                            "title": item.get("title", ""),
                            "type": item.get("type", "regulation"),
                            "code": item.get("code", ""),
                        }
                    )
                    self.documents.append(doc)
                
                logger.info(f"Loaded {len(data)} items from {file_path.name}")
            
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    def _load_builtin_regulations(self) -> None:
        """Load built-in regulations."""
        builtin_regs = [
            {
                "title": "Luật Bảo vệ Môi trường 2020",
                "code": "72/2020/QH14",
                "type": "law",
                "content": """
                Luật Bảo vệ môi trường số 72/2020/QH14 quy định về hoạt động bảo vệ môi trường; 
                quyền, nghĩa vụ và trách nhiệm của cơ quan, tổ chức, cộng đồng dân cư, hộ gia đình 
                và cá nhân trong bảo vệ môi trường.
                
                Điều 30: Đối tượng phải thực hiện đánh giá tác động môi trường
                1. Dự án thuộc nhóm I có nguy cơ tác động xấu đến môi trường ở mức độ cao
                2. Dự án thuộc nhóm II có nguy cơ tác động xấu đến môi trường
                
                Điều 31: Nội dung báo cáo đánh giá tác động môi trường
                1. Xuất xứ của dự án, chủ dự án
                2. Sự phù hợp của dự án với quy hoạch
                3. Đánh giá tác động môi trường của dự án
                4. Biện pháp phòng ngừa, giảm thiểu tác động
                5. Chương trình quản lý và giám sát môi trường
                6. Kết quả tham vấn
                """,
            },
            {
                "title": "Nghị định 08/2022/NĐ-CP",
                "code": "08/2022/NĐ-CP",
                "type": "decree",
                "content": """
                Nghị định quy định chi tiết một số điều của Luật Bảo vệ môi trường về đánh giá 
                tác động môi trường, giấy phép môi trường.
                
                Phụ lục II: Danh mục dự án phải thực hiện ĐTM
                - Dự án điện mặt trời có công suất từ 50 MW trở lên
                - Dự án sản xuất công nghiệp có vốn đầu tư từ 50 tỷ đồng
                - Dự án khu công nghiệp, khu đô thị
                
                Thời gian thẩm định: 45 ngày làm việc
                """,
            },
            {
                "title": "QCVN 19:2009/BTNMT - Khí thải công nghiệp",
                "code": "QCVN 19:2009/BTNMT",
                "type": "standard",
                "content": """
                Quy chuẩn kỹ thuật quốc gia về khí thải công nghiệp đối với bụi và các chất vô cơ.
                
                Giá trị tối đa cho phép:
                - Bụi tổng: 200 mg/Nm³ (Cột B)
                - CO: 1000 mg/Nm³
                - SO2: 500 mg/Nm³
                - NOx: 850 mg/Nm³
                - Pb: 5 mg/Nm³
                
                Áp dụng cho các nguồn thải công nghiệp.
                """,
            },
            {
                "title": "QCVN 40:2011/BTNMT - Nước thải công nghiệp",
                "code": "QCVN 40:2011/BTNMT",
                "type": "standard",
                "content": """
                Quy chuẩn kỹ thuật quốc gia về nước thải công nghiệp.
                
                Giá trị tối đa cho phép (Cột B - Xả vào nguồn không dùng cho cấp nước):
                - pH: 5.5 - 9
                - BOD5: 50 mg/L
                - COD: 150 mg/L
                - TSS: 100 mg/L
                - Tổng N: 40 mg/L
                - Tổng P: 6 mg/L
                - Coliform: 5000 MPN/100mL
                """,
            },
            {
                "title": "QCVN 26:2010/BTNMT - Tiếng ồn",
                "code": "QCVN 26:2010/BTNMT",
                "type": "standard",
                "content": """
                Quy chuẩn kỹ thuật quốc gia về tiếng ồn.
                
                Giới hạn tối đa cho phép (dBA):
                
                Khu vực đặc biệt (bệnh viện, trường học):
                - Ban ngày (6h-21h): 55 dBA
                - Ban đêm (21h-6h): 45 dBA
                
                Khu dân cư:
                - Ban ngày: 70 dBA
                - Ban đêm: 55 dBA
                """,
            },
            {
                "title": "IFC EHS Guidelines",
                "code": "IFC-EHS-2007",
                "type": "international",
                "content": """
                IFC Environmental, Health, and Safety Guidelines.
                
                General Requirements:
                - Environmental and Social Management System (ESMS)
                - Pollution prevention and control
                - Community health and safety
                - Occupational health and safety
                
                Emission Standards (where no national standards):
                - PM: 50 mg/Nm³
                - SO2: 500 mg/Nm³
                - NOx: 320 mg/Nm³
                
                Wastewater Standards:
                - BOD: 30 mg/L
                - COD: 125 mg/L
                - TSS: 50 mg/L
                """,
            },
        ]
        
        for reg in builtin_regs:
            doc = Document(
                page_content=reg["content"],
                metadata={
                    "title": reg["title"],
                    "code": reg["code"],
                    "type": reg["type"],
                    "source": "builtin",
                }
            )
            self.documents.append(doc)
        
        logger.info(f"Loaded {len(builtin_regs)} built-in regulations")
    
    def build_index(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Build vector index from documents."""
        if not self.documents:
            self.load_regulations()
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = splitter.split_documents(self.documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="eia_knowledge",
        )
        
        logger.info(f"Built index with {len(splits)} chunks")
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_type: Optional[str] = None,
    ) -> List[Document]:
        """Search knowledge base."""
        if not self.vectorstore:
            self.build_index()
        
        filter_dict = None
        if filter_type:
            filter_dict = {"type": filter_type}
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter_dict,
        )
        
        return results


# =============================================================================
# RAG Tool
# =============================================================================

class RAGTool:
    """
    RAG tool for retrieving relevant EIA knowledge.
    
    Provides context for agents to generate accurate EIA content.
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        self.kb = knowledge_base or KnowledgeBase()
    
    def get_regulations(
        self,
        project_type: str,
        impact_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get relevant regulations for project type."""
        query = f"quy định pháp luật về đánh giá tác động môi trường cho dự án {project_type}"
        
        if impact_category:
            query += f" về {impact_category}"
        
        results = self.kb.search(query, k=5, filter_type="regulation")
        
        return [
            {
                "title": doc.metadata.get("title", ""),
                "code": doc.metadata.get("code", ""),
                "content": doc.page_content,
            }
            for doc in results
        ]
    
    def get_emission_standards(
        self,
        emission_type: str,  # air, water, noise
    ) -> List[Dict[str, Any]]:
        """Get emission standards by type."""
        query_map = {
            "air": "quy chuẩn khí thải công nghiệp QCVN bụi CO SO2",
            "water": "quy chuẩn nước thải công nghiệp QCVN BOD COD",
            "noise": "quy chuẩn tiếng ồn QCVN dBA",
        }
        
        query = query_map.get(emission_type, f"quy chuẩn {emission_type}")
        results = self.kb.search(query, k=3, filter_type="standard")
        
        return [
            {
                "title": doc.metadata.get("title", ""),
                "code": doc.metadata.get("code", ""),
                "content": doc.page_content,
            }
            for doc in results
        ]
    
    def get_mitigation_measures(
        self,
        impact_type: str,
        project_type: str,
    ) -> List[Dict[str, Any]]:
        """Get recommended mitigation measures."""
        query = f"biện pháp giảm thiểu tác động {impact_type} cho dự án {project_type}"
        
        results = self.kb.search(query, k=5)
        
        return [
            {
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content,
            }
            for doc in results
        ]
    
    def get_monitoring_requirements(
        self,
        project_type: str,
    ) -> List[Dict[str, Any]]:
        """Get monitoring requirements."""
        query = f"yêu cầu giám sát môi trường quan trắc {project_type}"
        
        results = self.kb.search(query, k=5)
        
        return [
            {
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content,
            }
            for doc in results
        ]
