"""
Prompt templates for Climate Q&A RAG System.

Contains both English and Vietnamese prompt templates for different use cases.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_EN = """You are an expert climate science and environmental assistant with deep knowledge from provided documents.

**Your expertise covers:**
- Climate change science (causes, mechanisms, effects, projections)
- International climate policies (Paris Agreement, NDCs, carbon markets)
- Carbon footprint and emissions (calculation, reduction strategies)
- ESG and sustainability (reporting, frameworks, best practices)
- Environmental issues specific to Vietnam and Southeast Asia

**CRITICAL INSTRUCTIONS:**
1. **Answer comprehensively**: Provide detailed, complete answers that fully address the question
2. **Use ALL relevant context**: Synthesize information from multiple source documents when available
3. **Be specific with data**: Include specific numbers, percentages, dates, and statistics from the documents
4. **Explain mechanisms**: Don't just state facts - explain WHY and HOW things work
5. **Structure your answer**: Use headers, bullet points, and clear organization
6. **Always cite sources**: Use [Source: filename] format after key claims

**Answer Format:**
1. **Direct Answer**: Start with a clear, direct answer to the question
2. **Details & Explanation**: Provide supporting details, data, and explanations
3. **Examples**: Include relevant examples when available
4. **Sources**: List the documents used at the end

**If context is insufficient:**
State clearly what information IS available and what is missing. Never make up facts."""

SYSTEM_PROMPT_VI = """Bạn là trợ lý chuyên gia về khí hậu và môi trường với kiến thức sâu rộng từ tài liệu được cung cấp.

**Chuyên môn của bạn:**
- Khoa học biến đổi khí hậu (nguyên nhân, cơ chế, tác động, dự báo)
- Chính sách khí hậu quốc tế (Hiệp định Paris, NDC, thị trường carbon)
- Dấu chân carbon và khí thải (tính toán, chiến lược giảm thiểu)
- ESG và phát triển bền vững (báo cáo, khung tiêu chuẩn, thực hành tốt)
- Các vấn đề môi trường tại Việt Nam và Đông Nam Á

**HƯỚNG DẪN QUAN TRỌNG:**
1. **Trả lời đầy đủ**: Cung cấp câu trả lời chi tiết, hoàn chỉnh cho câu hỏi
2. **Sử dụng TẤT CẢ ngữ cảnh liên quan**: Tổng hợp thông tin từ nhiều tài liệu nguồn
3. **Cụ thể với dữ liệu**: Bao gồm số liệu, tỷ lệ %, ngày tháng cụ thể từ tài liệu
4. **Giải thích cơ chế**: Không chỉ nêu sự thật - giải thích TẠI SAO và NHƯ THẾ NÀO
5. **Cấu trúc câu trả lời**: Sử dụng tiêu đề, gạch đầu dòng, tổ chức rõ ràng
6. **Luôn trích dẫn nguồn**: Dùng định dạng [Nguồn: tên_file] sau các thông tin chính

**Định dạng Câu trả lời:**
1. **Trả lời Trực tiếp**: Bắt đầu với câu trả lời rõ ràng cho câu hỏi
2. **Chi tiết & Giải thích**: Cung cấp chi tiết hỗ trợ, dữ liệu, giải thích
3. **Ví dụ**: Bao gồm các ví dụ liên quan khi có
4. **Nguồn**: Liệt kê các tài liệu đã sử dụng ở cuối

**Nếu ngữ cảnh không đủ:**
Nêu rõ thông tin NÀO có sẵn và thông tin NÀO còn thiếu. KHÔNG BAO GIỜ bịa đặt sự thật."""

# =============================================================================
# RAG Prompts
# =============================================================================

RAG_PROMPT_TEMPLATE_EN = """Use the following context to answer the question. 
If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context above. Include citations where appropriate."""

RAG_PROMPT_TEMPLATE_VI = """Sử dụng ngữ cảnh sau để trả lời câu hỏi.
Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy nói rõ điều đó.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Cung cấp câu trả lời đầy đủ dựa trên ngữ cảnh trên. Bao gồm trích dẫn khi thích hợp."""

# =============================================================================
# Conversational RAG Prompts (with memory)
# =============================================================================

CONVERSATIONAL_RAG_TEMPLATE_EN = """Given the following conversation history and a new question, 
answer based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Consider the conversation history for context
- Base your answer on the provided documents
- If the question refers to previous messages, use that context
- Cite sources when providing factual information"""

CONVERSATIONAL_RAG_TEMPLATE_VI = """Dựa trên lịch sử hội thoại sau và câu hỏi mới,
trả lời dựa trên ngữ cảnh được cung cấp.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Hướng dẫn:
- Xem xét lịch sử hội thoại để hiểu ngữ cảnh
- Dựa câu trả lời trên các tài liệu được cung cấp
- Nếu câu hỏi đề cập đến tin nhắn trước, sử dụng ngữ cảnh đó
- Trích dẫn nguồn khi cung cấp thông tin thực tế"""

# =============================================================================
# Query Expansion Prompts
# =============================================================================

QUERY_EXPANSION_PROMPT_EN = """Given the original question, generate 3 alternative versions 
that might help retrieve relevant documents. The alternatives should:
1. Use different terminology/synonyms
2. Be more specific or more general
3. Focus on different aspects of the question

Original question: {question}

Generate 3 alternative questions (one per line, no numbering):"""

QUERY_EXPANSION_PROMPT_VI = """Cho câu hỏi gốc, tạo 3 phiên bản thay thế 
có thể giúp truy xuất tài liệu liên quan. Các phiên bản thay thế nên:
1. Sử dụng thuật ngữ/từ đồng nghĩa khác nhau
2. Cụ thể hơn hoặc tổng quát hơn
3. Tập trung vào các khía cạnh khác nhau của câu hỏi

Câu hỏi gốc: {question}

Tạo 3 câu hỏi thay thế (mỗi dòng một câu, không đánh số):"""

# =============================================================================
# Standalone Question Prompt (for chat history contextualization)
# =============================================================================

STANDALONE_QUESTION_PROMPT_EN = """Given the following conversation history and a follow-up question, 
rephrase the follow-up question to be a standalone question that captures all necessary context.

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

STANDALONE_QUESTION_PROMPT_VI = """Cho lịch sử hội thoại sau và câu hỏi tiếp theo,
diễn đạt lại câu hỏi tiếp theo thành câu hỏi độc lập nắm bắt tất cả ngữ cảnh cần thiết.

Lịch sử chat:
{chat_history}

Câu hỏi tiếp theo: {question}

Câu hỏi độc lập:"""

# =============================================================================
# Evaluation Prompts
# =============================================================================

ANSWER_RELEVANCE_PROMPT = """Rate the relevance of the answer to the question on a scale of 1-5.

Question: {question}
Answer: {answer}

Criteria:
5 - Directly and completely answers the question
4 - Mostly answers the question with minor gaps
3 - Partially answers the question
2 - Tangentially related to the question
1 - Does not answer the question at all

Rating (just the number):"""

FAITHFULNESS_PROMPT = """Evaluate if the answer is faithful to the provided context.
The answer should only contain information that can be derived from the context.

Context: {context}
Answer: {answer}

Is the answer faithful to the context? (yes/no):
If no, what information is not supported by the context?"""

# =============================================================================
# Prompt Factory Functions
# =============================================================================

def get_rag_prompt(language: str = "en", with_history: bool = False) -> ChatPromptTemplate:
    """
    Get the appropriate RAG prompt template.
    
    Args:
        language: Language code ("en" or "vi")
        with_history: Whether to include conversation history
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    system_prompt = SYSTEM_PROMPT_VI if language == "vi" else SYSTEM_PROMPT_EN
    
    if with_history:
        human_template = (
            CONVERSATIONAL_RAG_TEMPLATE_VI if language == "vi" 
            else CONVERSATIONAL_RAG_TEMPLATE_EN
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", human_template),
        ])
    else:
        human_template = (
            RAG_PROMPT_TEMPLATE_VI if language == "vi" 
            else RAG_PROMPT_TEMPLATE_EN
        )
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_template),
        ])


def get_query_expansion_prompt(language: str = "en") -> ChatPromptTemplate:
    """
    Get query expansion prompt template.
    
    Args:
        language: Language code ("en" or "vi")
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    template = (
        QUERY_EXPANSION_PROMPT_VI if language == "vi" 
        else QUERY_EXPANSION_PROMPT_EN
    )
    return ChatPromptTemplate.from_template(template)


def get_standalone_question_prompt(language: str = "en") -> ChatPromptTemplate:
    """
    Get standalone question prompt for contextualizing follow-up questions.
    
    Args:
        language: Language code ("en" or "vi")
        
    Returns:
        ChatPromptTemplate: Configured prompt template
    """
    template = (
        STANDALONE_QUESTION_PROMPT_VI if language == "vi" 
        else STANDALONE_QUESTION_PROMPT_EN
    )
    return ChatPromptTemplate.from_template(template)


# =============================================================================
# Example Questions for Testing
# =============================================================================

SAMPLE_QUESTIONS_EN = [
    "What are the main causes of climate change?",
    "How does deforestation contribute to global warming?",
    "What is the Paris Agreement and what are its main goals?",
    "How can individuals reduce their carbon footprint?",
    "What are the effects of climate change on Vietnam?",
    "Explain the greenhouse effect and its role in global warming.",
    "What are ESG criteria and why are they important?",
    "How does air pollution affect human health?",
    "What are renewable energy sources and their benefits?",
    "What is carbon neutrality and how can it be achieved?",
]

SAMPLE_QUESTIONS_VI = [
    "Nguyên nhân chính gây ra biến đổi khí hậu là gì?",
    "Phá rừng góp phần vào sự nóng lên toàn cầu như thế nào?",
    "Hiệp định Paris là gì và mục tiêu chính của nó là gì?",
    "Cá nhân có thể giảm dấu chân carbon như thế nào?",
    "Tác động của biến đổi khí hậu đối với Việt Nam là gì?",
    "Giải thích hiệu ứng nhà kính và vai trò của nó trong sự nóng lên toàn cầu.",
    "Tiêu chí ESG là gì và tại sao chúng quan trọng?",
    "Ô nhiễm không khí ảnh hưởng đến sức khỏe con người như thế nào?",
    "Nguồn năng lượng tái tạo là gì và lợi ích của chúng?",
    "Trung hòa carbon là gì và làm thế nào để đạt được?",
]
