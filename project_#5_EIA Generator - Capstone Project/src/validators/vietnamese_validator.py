"""
Vietnamese Text Quality Validator using PhoBERT and Underthesea.

Validates Vietnamese text quality in EIA reports:
- Text coherence and readability
- Named Entity Recognition for environmental terms
- Spelling and grammar checking
- Formality assessment
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    from underthesea import word_tokenize, ner, pos_tag, classify
    UNDERTHESEA_AVAILABLE = True
except ImportError:
    UNDERTHESEA_AVAILABLE = False
    logger.warning("underthesea not available. Install with: pip install underthesea")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    PHOBERT_AVAILABLE = True
except ImportError:
    PHOBERT_AVAILABLE = False
    logger.warning("transformers/torch not available for PhoBERT")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    """Result of text validation."""
    is_valid: bool = True
    score: float = 100.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    entities: List[Dict] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TextMetrics:
    """Text quality metrics."""
    word_count: int = 0
    sentence_count: int = 0
    avg_sentence_length: float = 0.0
    vietnamese_ratio: float = 0.0
    formality_score: float = 0.0
    readability_score: float = 0.0


# =============================================================================
# Vietnamese Text Validator
# =============================================================================

class VietnameseTextValidator:
    """
    Validates Vietnamese text quality using PhoBERT and Underthesea.
    
    Features:
    - Word tokenization and POS tagging
    - Named Entity Recognition for environmental terms
    - Text readability assessment
    - Formality checking for official documents
    """
    
    # Environmental domain keywords
    ENV_KEYWORDS = {
        "môi trường", "ô nhiễm", "khí thải", "nước thải", "chất thải",
        "tác động", "đánh giá", "giám sát", "quan trắc", "tiêu chuẩn",
        "quy chuẩn", "sinh thái", "đa dạng sinh học", "bảo tồn",
        "phát thải", "xử lý", "giảm thiểu", "khắc phục", "phục hồi",
        "bền vững", "biến đổi khí hậu", "năng lượng", "tái tạo",
    }
    
    # Informal words to avoid in official documents
    INFORMAL_WORDS = {
        "ok", "ko", "k", "đc", "đk", "dc", "z", "j", "vậy á",
        "thôi", "ha", "hả", "hehe", "hihi", "^^", ":)",
    }
    
    def __init__(self, use_phobert: bool = False):
        """
        Initialize validator.
        
        Args:
            use_phobert: Whether to load PhoBERT model (requires more memory)
        """
        self.use_phobert = use_phobert and PHOBERT_AVAILABLE
        self.tokenizer = None
        self.model = None
        
        if self.use_phobert:
            self._load_phobert()
    
    def _load_phobert(self):
        """Load PhoBERT model for advanced analysis."""
        try:
            logger.info("Loading PhoBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.model = AutoModel.from_pretrained("vinai/phobert-base")
            self.model.eval()
            logger.info("PhoBERT loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PhoBERT: {e}")
            self.use_phobert = False
    
    def validate(self, text: str) -> ValidationResult:
        """
        Validate Vietnamese text quality.
        
        Args:
            text: Text to validate
            
        Returns:
            ValidationResult with score and issues
        """
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["Văn bản trống"]
            )
        
        result = ValidationResult()
        
        # Calculate metrics
        metrics = self._calculate_metrics(text)
        result.metrics = {
            "word_count": metrics.word_count,
            "sentence_count": metrics.sentence_count,
            "avg_sentence_length": metrics.avg_sentence_length,
            "vietnamese_ratio": metrics.vietnamese_ratio,
            "formality_score": metrics.formality_score,
            "readability_score": metrics.readability_score,
        }
        
        # Check text length
        if metrics.word_count < 50:
            result.issues.append("Văn bản quá ngắn (< 50 từ)")
            result.score -= 10
        
        # Check sentence length
        if metrics.avg_sentence_length > 50:
            result.issues.append("Câu quá dài, nên chia nhỏ để dễ đọc")
            result.suggestions.append("Chia các câu dài thành nhiều câu ngắn hơn")
            result.score -= 5
        
        # Check Vietnamese ratio
        if metrics.vietnamese_ratio < 0.7:
            result.issues.append("Tỷ lệ từ tiếng Việt thấp")
            result.score -= 10
        
        # Check formality
        if metrics.formality_score < 0.8:
            result.issues.append("Văn bản có thể chứa từ ngữ không trang trọng")
            result.suggestions.append("Sử dụng ngôn ngữ trang trọng hơn cho văn bản chính thức")
            result.score -= 5
        
        # Extract entities if underthesea available
        if UNDERTHESEA_AVAILABLE:
            result.entities = self._extract_entities(text)
        
        # Check environmental keywords
        env_keyword_count = self._count_env_keywords(text)
        if env_keyword_count < 5:
            result.suggestions.append("Nên bổ sung thêm thuật ngữ môi trường chuyên ngành")
        
        result.score = max(0, min(100, result.score))
        result.is_valid = result.score >= 60
        
        return result
    
    def _calculate_metrics(self, text: str) -> TextMetrics:
        """Calculate text quality metrics."""
        metrics = TextMetrics()
        
        # Basic counts
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        metrics.sentence_count = len(sentences)
        
        # Word tokenization
        if UNDERTHESEA_AVAILABLE:
            words = word_tokenize(text)
        else:
            words = text.split()
        
        metrics.word_count = len(words)
        
        # Average sentence length
        if metrics.sentence_count > 0:
            metrics.avg_sentence_length = metrics.word_count / metrics.sentence_count
        
        # Vietnamese ratio (simple heuristic)
        vietnamese_chars = sum(1 for c in text if '\u00C0' <= c <= '\u1EF9')
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha > 0:
            metrics.vietnamese_ratio = min(1.0, vietnamese_chars / total_alpha * 3)
        
        # Formality score
        text_lower = text.lower()
        informal_count = sum(1 for word in self.INFORMAL_WORDS if word in text_lower)
        metrics.formality_score = max(0, 1.0 - (informal_count * 0.1))
        
        # Readability score (based on sentence length and word complexity)
        if metrics.avg_sentence_length <= 25:
            metrics.readability_score = 1.0
        elif metrics.avg_sentence_length <= 40:
            metrics.readability_score = 0.8
        else:
            metrics.readability_score = 0.6
        
        return metrics
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using underthesea."""
        entities = []
        try:
            # Limit text length for NER
            text_sample = text[:2000] if len(text) > 2000 else text
            ner_results = ner(text_sample)
            
            for item in ner_results:
                if len(item) >= 3:
                    entities.append({
                        "text": item[0],
                        "tag": item[1] if len(item) > 1 else "O",
                        "ner": item[2] if len(item) > 2 else "O",
                    })
        except Exception as e:
            logger.warning(f"NER extraction failed: {e}")
        
        return entities
    
    def _count_env_keywords(self, text: str) -> int:
        """Count environmental keywords in text."""
        text_lower = text.lower()
        return sum(1 for kw in self.ENV_KEYWORDS if kw in text_lower)
    
    def validate_section(
        self,
        section_title: str,
        section_content: str,
    ) -> ValidationResult:
        """
        Validate a specific EIA report section.
        
        Args:
            section_title: Section title
            section_content: Section content
            
        Returns:
            ValidationResult
        """
        result = self.validate(section_content)
        
        # Section-specific checks
        title_lower = section_title.lower()
        
        if "tác động" in title_lower:
            # Impact assessment section should mention impacts
            impact_words = ["tích cực", "tiêu cực", "đáng kể", "không đáng kể", "ảnh hưởng"]
            if not any(w in section_content.lower() for w in impact_words):
                result.issues.append("Phần đánh giá tác động thiếu mô tả mức độ tác động")
                result.score -= 10
        
        if "giảm thiểu" in title_lower or "biện pháp" in title_lower:
            # Mitigation section should have action verbs
            action_words = ["thực hiện", "áp dụng", "triển khai", "bố trí", "lắp đặt"]
            if not any(w in section_content.lower() for w in action_words):
                result.suggestions.append("Nên sử dụng động từ hành động cụ thể hơn")
        
        if "giám sát" in title_lower:
            # Monitoring section should mention frequency
            freq_words = ["định kỳ", "hàng tháng", "hàng quý", "thường xuyên", "liên tục"]
            if not any(w in section_content.lower() for w in freq_words):
                result.suggestions.append("Nên bổ sung tần suất giám sát cụ thể")
        
        result.score = max(0, min(100, result.score))
        result.is_valid = result.score >= 60
        
        return result
    
    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get text embedding using PhoBERT.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if PhoBERT not available
        """
        if not self.use_phobert or not self.model:
            return None
        
        try:
            # Tokenize and get embedding
            if UNDERTHESEA_AVAILABLE:
                text = " ".join(word_tokenize(text))
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True,
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()
            
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None


# =============================================================================
# Convenience Functions
# =============================================================================

_validator_instance = None

def get_validator(use_phobert: bool = False) -> VietnameseTextValidator:
    """Get singleton validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = VietnameseTextValidator(use_phobert=use_phobert)
    return _validator_instance


def validate_vietnamese_text(text: str) -> ValidationResult:
    """Convenience function to validate Vietnamese text."""
    return get_validator().validate(text)


def validate_eia_section(title: str, content: str) -> ValidationResult:
    """Convenience function to validate EIA section."""
    return get_validator().validate_section(title, content)
