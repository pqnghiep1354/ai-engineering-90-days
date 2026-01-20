"""
Validator Agent - Validates EIA report for compliance and completeness.
"""

from typing import Any, Dict, List, Tuple

from loguru import logger

from .base import BaseAgent, AgentState
from ..config import EIA_SECTIONS
from ..validators.vietnamese_validator import VietnameseTextValidator, ValidationResult


class ValidatorAgent(BaseAgent):
    """Agent responsible for validating EIA reports."""
    
    def __init__(self, model: str = None, temperature: float = 0.2):
        super().__init__(
            name="validator",
            description="Validate EIA report for regulatory compliance and completeness",
            model=model,
            temperature=temperature,
        )
        # Initialize Vietnamese text validator
        self.vn_validator = VietnameseTextValidator(use_phobert=False)
    
    def _build_system_prompt(self) -> str:
        return """Bạn là chuyên gia thẩm định báo cáo ĐTM (EIA Review Expert).

NHIỆM VỤ: Kiểm tra và đánh giá chất lượng báo cáo ĐTM.

TIÊU CHÍ ĐÁNH GIÁ:

1. TUÂN THỦ PHÁP LUẬT:
   - Đầy đủ các mục theo Nghị định 08/2022/NĐ-CP
   - Áp dụng đúng QCVN, TCVN
   - Trích dẫn văn bản pháp luật chính xác

2. TÍNH ĐẦY ĐỦ:
   - Đủ 6 chương theo quy định
   - Có bảng, biểu, sơ đồ minh họa
   - Có phụ lục đính kèm

3. TÍNH CHÍNH XÁC:
   - Số liệu có cơ sở
   - Tính toán đúng
   - Nguồn trích dẫn rõ ràng

4. TÍNH KHẢ THI:
   - Biện pháp giảm thiểu khả thi
   - Chi phí hợp lý
   - Có thể thực hiện được

5. CHẤT LƯỢNG VĂN BẢN:
   - Trình bày rõ ràng
   - Ngôn ngữ chuyên môn
   - Nhất quán format

OUTPUT: Điểm số từng tiêu chí và tổng điểm (thang 100)."""
    
    async def execute(self, state: AgentState) -> AgentState:
        """Validate the generated EIA report."""
        logger.info("Validator Agent: Validating EIA report")
        
        sections = state.get("sections", {})
        project = state["project"]
        
        # Check completeness
        completeness_result = self._check_completeness(sections)
        
        # Check compliance
        compliance_result = await self._check_compliance(sections, project)
        
        # Check quality
        quality_result = await self._check_quality(sections)
        
        # Calculate overall score
        overall_score = (
            completeness_result["score"] * 0.3 +
            compliance_result["score"] * 0.4 +
            quality_result["score"] * 0.3
        )
        
        # Compile validation results
        validation_results = {
            "completeness": completeness_result,
            "compliance": compliance_result,
            "quality": quality_result,
            "overall_score": overall_score,
            "passed": overall_score >= 70,
            "recommendations": self._generate_recommendations(
                completeness_result,
                compliance_result,
                quality_result,
            ),
        }
        
        # Update state
        state["validation_results"] = validation_results
        state["compliance_score"] = overall_score
        state["current_step"] = "validation_complete"
        
        logger.info(f"Validator Agent: Score = {overall_score:.1f}/100")
        return state
    
    def _check_completeness(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Check if all required sections are present with sufficient content."""
        # Required sections with minimum word count
        required_sections = {
            "regulations": {"min_words": 300, "name": "Cơ sở pháp lý"},
            "baseline_natural": {"min_words": 400, "name": "Điều kiện tự nhiên"},
            "baseline_socio": {"min_words": 300, "name": "Điều kiện KT-XH"},
            "baseline_env": {"min_words": 300, "name": "Hiện trạng môi trường"},
            "impact_construction": {"min_words": 400, "name": "Tác động giai đoạn xây dựng"},
            "impact_operation": {"min_words": 400, "name": "Tác động giai đoạn vận hành"},
            "mitigation_construction": {"min_words": 400, "name": "Biện pháp GĐ xây dựng"},
            "mitigation_operation": {"min_words": 400, "name": "Biện pháp GĐ vận hành"},
            "management": {"min_words": 300, "name": "Quản lý môi trường"},
            "monitoring": {"min_words": 300, "name": "Giám sát môi trường"},
        }
        
        present = []
        missing = []
        insufficient = []
        details = []
        
        for section_key, requirements in required_sections.items():
            min_words = requirements["min_words"]
            section_name = requirements["name"]
            
            if section_key in sections:
                content = sections[section_key]
                word_count = len(content.split())
                
                if word_count >= min_words:
                    present.append(section_key)
                    details.append(f"✓ {section_name}: {word_count} từ (đạt)")
                else:
                    insufficient.append(section_key)
                    details.append(f"⚠ {section_name}: {word_count}/{min_words} từ (thiếu)")
            else:
                missing.append(section_key)
                details.append(f"✗ {section_name}: Chưa có")
        
        # Score: full for present, half for insufficient, zero for missing
        total = len(required_sections)
        score = ((len(present) + len(insufficient) * 0.5) / total) * 100
        
        return {
            "score": score,
            "present": present,
            "missing": missing,
            "insufficient": insufficient,
            "total_required": total,
            "details": details,
            "summary": f"Đạt: {len(present)}, Thiếu nội dung: {len(insufficient)}, Chưa có: {len(missing)}",
        }
    
    async def _check_compliance(
        self,
        sections: Dict[str, str],
        project: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check regulatory compliance."""
        
        # Key compliance checks
        checks = [
            ("Trích dẫn Luật BVMT 2020", any("Luật" in s or "2020" in s for s in sections.values())),
            ("Áp dụng QCVN", any("QCVN" in s for s in sections.values())),
            ("Có ma trận tác động", "impact" in str(sections)),
            ("Có biện pháp giảm thiểu", "mitigation" in str(sections)),
            ("Có chương trình giám sát", "monitoring" in str(sections)),
        ]
        
        passed = sum(1 for _, result in checks if result)
        score = (passed / len(checks)) * 100
        
        details = []
        for check_name, result in checks:
            status = "✓" if result else "✗"
            details.append(f"{status} {check_name}")
        
        return {
            "score": score,
            "passed_checks": passed,
            "total_checks": len(checks),
            "details": details,
        }
    
    async def _check_quality(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Check content quality using Vietnamese text validator."""
        
        # Quality metrics
        total_length = sum(len(s) for s in sections.values())
        num_sections = len([s for s in sections.values() if len(s) > 50])
        
        # Use Vietnamese text validator for each section
        vn_validation_results = []
        section_scores = []
        all_issues = []
        all_suggestions = []
        
        for section_key, content in sections.items():
            if len(content) > 50:
                result = self.vn_validator.validate_section(
                    section_title=section_key,
                    section_content=content,
                )
                vn_validation_results.append({
                    "section": section_key,
                    "score": result.score,
                    "is_valid": result.is_valid,
                    "issues": result.issues,
                })
                section_scores.append(result.score)
                all_issues.extend(result.issues)
                all_suggestions.extend(result.suggestions)
        
        # Calculate average Vietnamese quality score
        vn_quality_score = sum(section_scores) / len(section_scores) if section_scores else 50
        
        # Estimate quality based on content
        quality_indicators = {
            "sufficient_length": total_length > 10000,
            "multiple_sections": num_sections >= 5,
            "has_numbers": any(any(c.isdigit() for c in s) for s in sections.values()),
            "has_structure": any("##" in s or "###" in s for s in sections.values()),
            "vietnamese_quality": vn_quality_score >= 70,
        }
        
        base_score = (sum(quality_indicators.values()) / len(quality_indicators)) * 100
        # Combine with Vietnamese quality score
        score = (base_score * 0.5) + (vn_quality_score * 0.5)
        
        return {
            "score": score,
            "total_length": total_length,
            "num_sections": num_sections,
            "indicators": quality_indicators,
            "vietnamese_validation": vn_validation_results,
            "vn_quality_score": vn_quality_score,
            "issues": all_issues[:10],  # Top 10 issues
            "suggestions": all_suggestions[:5],  # Top 5 suggestions
        }
    
    def _generate_recommendations(
        self,
        completeness: Dict[str, Any],
        compliance: Dict[str, Any],
        quality: Dict[str, Any],
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        # Completeness recommendations
        if completeness.get("missing"):
            for section in completeness["missing"][:3]:
                recommendations.append(f"Bổ sung mục: {section}")
        
        # Compliance recommendations  
        for detail in compliance.get("details", []):
            if detail.startswith("✗"):
                recommendations.append(f"Cần {detail[2:]}")
        
        # Quality recommendations
        if quality.get("total_length", 0) < 10000:
            recommendations.append("Bổ sung thêm nội dung chi tiết")
        
        return recommendations[:5]  # Top 5 recommendations
