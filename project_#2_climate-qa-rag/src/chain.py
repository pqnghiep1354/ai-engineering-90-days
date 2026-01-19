"""
RAG chain composition for Climate Q&A RAG System.

Combines retrieval, reranking, and generation into unified chains.
"""

from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStore
from loguru import logger
import time

from .config import settings
from .embeddings import get_embedding_model
from .llm import get_llm, LLMWrapper
from .prompts import get_rag_prompt, get_standalone_question_prompt
from .reranker import rerank_documents
from .retriever import get_retriever
from .utils import Timer, detect_language, format_sources
from .vector_store import VectorStoreManager, load_existing_index


# =============================================================================
# Document Formatting
# =============================================================================

def format_documents(documents: List[Document]) -> str:
    """
    Format documents for context in prompt.
    
    Args:
        documents: List of documents
        
    Returns:
        Formatted string
    """
    formatted_docs = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "Unknown")
        # Extract just the filename from path
        source_name = source.split("/")[-1].split("\\")[-1]
        page = doc.metadata.get("page", "")
        content = doc.page_content.strip()
        
        # Use source filename as header for clarity
        header = f"[{source_name}]" if not page or page == "N/A" else f"[{source_name}, Page {page}]"
        formatted_docs.append(
            f"{header}\n"
            f"{content}\n"
        )
    
    return "\n---\n".join(formatted_docs)


# =============================================================================
# Basic RAG Chain
# =============================================================================

class RAGChain:
    """
    Basic RAG chain for question answering.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[BaseChatModel] = None,
        retriever_type: str = "basic",
        use_reranker: bool = None,
        language: str = "en",
    ):
        """
        Initialize RAG chain.
        
        Args:
            vector_store: Vector store instance
            llm: LLM instance
            retriever_type: Type of retriever to use
            use_reranker: Whether to use reranking
            language: Response language ("en" or "vi")
        """
        # Initialize components
        self.embeddings = get_embedding_model(provider="gemini")
        
        if vector_store is None:
            manager = load_existing_index(embeddings=self.embeddings)
            self.vector_store = manager.vector_store
        else:
            self.vector_store = vector_store
        
        self.llm = llm or get_llm()
        self.use_reranker = use_reranker if use_reranker is not None else settings.use_reranker
        self.language = language
        
        # Initialize retriever
        self.retriever = get_retriever(
            vector_store=self.vector_store,
            retriever_type=retriever_type,
            llm=self.llm if retriever_type in ["expansion", "compression"] else None,
        )
        
        # Get prompt template
        self.prompt = get_rag_prompt(language=language, with_history=False)
        
        logger.info(
            f"RAGChain initialized: retriever={retriever_type}, "
            f"reranker={self.use_reranker}, language={language}"
        )
    
    def _retrieve_and_rerank(self, query: str) -> List[Document]:
        """
        Retrieve and optionally rerank documents.
        
        Args:
            query: Query text
            
        Returns:
            List of documents
        """
        with Timer("Retrieval"):
            documents = self.retriever.invoke(query)
        
        if self.use_reranker and len(documents) > 0:
            with Timer("Reranking"):
                documents = rerank_documents(
                    query=query,
                    documents=documents,
                    top_k=settings.reranker_top_k,
                )
        
        return documents
    
    def invoke(
        self,
        query: str,
        return_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Invoke RAG chain.
        
        Args:
            query: User query
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Detect language if not set
        detected_lang = detect_language(query)
        if detected_lang != self.language:
            logger.debug(f"Detected language: {detected_lang}, using: {self.language}")
        
        # Retrieve documents
        documents = self._retrieve_and_rerank(query)
        
        if not documents:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
            }
        
        # Format context
        context = format_documents(documents)
        
        # Generate answer
        with Timer("Generation"):
            messages = self.prompt.format_messages(
                context=context,
                question=query,
            )
            response = self.llm.invoke(messages)
        
        result = {"answer": response.content}
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ]
        
        return result
    
    def stream(self, query: str) -> Iterator[str]:
        """
        Stream RAG response.
        
        Args:
            query: User query
            
        Yields:
            Response chunks
        """
        # Retrieve documents
        documents = self._retrieve_and_rerank(query)
        
        if not documents:
            yield "I couldn't find any relevant information to answer your question."
            return
        
        # Format context
        context = format_documents(documents)
        
        # Stream response
        messages = self.prompt.format_messages(
            context=context,
            question=query,
        )
        
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, "content"):
                yield chunk.content


# =============================================================================
# Conversational RAG Chain
# =============================================================================

class ConversationalRAGChain:
    """
    RAG chain with conversation memory.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[BaseChatModel] = None,
        language: str = "en",
        max_history: int = 10,
    ):
        """
        Initialize conversational RAG chain.
        
        Args:
            vector_store: Vector store instance
            llm: LLM instance
            language: Response language
            max_history: Maximum conversation history to keep
        """
        self.base_chain = RAGChain(
            vector_store=vector_store,
            llm=llm,
            language=language,
        )
        self.llm = self.base_chain.llm
        self.language = language
        self.max_history = max_history
        self.chat_history: List[BaseMessage] = []
        
        # Prompt for contextualizing questions
        self.contextualize_prompt = get_standalone_question_prompt(language)
    
    def _contextualize_question(self, question: str) -> str:
        """
        Rephrase question to be standalone using chat history.
        
        Args:
            question: Follow-up question
            
        Returns:
            Standalone question
        """
        if not self.chat_history:
            return question
        
        # Format chat history
        history_text = "\n".join(
            f"{'Human' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in self.chat_history[-self.max_history:]
        )
        
        prompt = self.contextualize_prompt.format(
            chat_history=history_text,
            question=question,
        )
        
        response = self.llm.invoke(prompt)
        standalone = response.content.strip()
        
        logger.debug(f"Contextualized question: {question} -> {standalone}")
        return standalone
    
    def invoke(
        self,
        query: str,
        return_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Invoke conversational RAG chain.
        
        Args:
            query: User query
            return_sources: Whether to return sources
            
        Returns:
            Dictionary with answer and optional sources
        """
        # Contextualize question if there's history
        if self.chat_history:
            standalone_query = self._contextualize_question(query)
        else:
            standalone_query = query
        
        # Get answer from base chain
        result = self.base_chain.invoke(standalone_query, return_sources=return_sources)
        
        # Update chat history
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=result["answer"]))
        
        # Trim history if needed
        if len(self.chat_history) > self.max_history * 2:
            self.chat_history = self.chat_history[-self.max_history * 2:]
        
        return result
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history = []
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history as list of dicts.
        
        Returns:
            List of message dictionaries
        """
        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content,
            }
            for m in self.chat_history
        ]


# =============================================================================
# Advanced RAG Chain with All Features
# =============================================================================

class AdvancedRAGChain:
    """
    Advanced RAG chain with all features enabled.
    """
    
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm: Optional[BaseChatModel] = None,
        enable_reranker: bool = True,
        enable_query_expansion: bool = False,
        enable_memory: bool = True,
        language: str = "auto",
    ):
        """
        Initialize advanced RAG chain.
        
        Args:
            vector_store: Vector store instance
            llm: LLM instance
            enable_reranker: Enable reranking
            enable_query_expansion: Enable query expansion
            enable_memory: Enable conversation memory
            language: Language ("en", "vi", or "auto")
        """
        self.embeddings = get_embedding_model(provider="gemini")
        
        if vector_store is None:
            manager = load_existing_index(embeddings=self.embeddings)
            self.vector_store = manager.vector_store
        else:
            self.vector_store = vector_store
        
        self.llm = llm or get_llm()
        self.enable_reranker = enable_reranker
        self.enable_query_expansion = enable_query_expansion
        self.enable_memory = enable_memory
        self.language = language
        
        # Conversation history
        self.chat_history: List[BaseMessage] = []
        
        # Initialize retriever
        retriever_type = "expansion" if enable_query_expansion else "basic"
        self.retriever = get_retriever(
            vector_store=self.vector_store,
            retriever_type=retriever_type,
            llm=self.llm,
        )
        
        logger.info(
            f"AdvancedRAGChain initialized: reranker={enable_reranker}, "
            f"expansion={enable_query_expansion}, memory={enable_memory}"
        )
    
    def _get_language(self, text: str) -> str:
        """Get language for response."""
        if self.language == "auto":
            return detect_language(text)
        return self.language
    
    def invoke(
        self,
        query: str,
        return_sources: bool = True,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """
        Invoke advanced RAG chain.
        
        Args:
            query: User query
            return_sources: Return source documents
            return_context: Return formatted context
            
        Returns:
            Result dictionary
        """
        start_time = __import__("time").time()
        
        # Detect language
        language = self._get_language(query)
        
        # Retrieve documents
        with Timer("Retrieval"):
            documents = self.retriever.invoke(query)
        
        # Rerank if enabled
        if self.enable_reranker and documents:
            with Timer("Reranking"):
                documents = rerank_documents(
                    query=query,
                    documents=documents,
                    top_k=settings.reranker_top_k,
                )
        
        if not documents:
            return {
                "answer": (
                    "Tôi không tìm thấy thông tin liên quan." if language == "vi"
                    else "I couldn't find any relevant information."
                ),
                "sources": [],
                "latency_ms": int((time.time() - start_time) * 1000),
            }
        
        # Format context
        context = format_documents(documents)
        
        # Get prompt - only use history mode if there's actual chat history
        has_history = self.enable_memory and len(self.chat_history) > 0
        prompt = get_rag_prompt(language=language, with_history=has_history)
        
        # Build messages
        if has_history:
            messages = prompt.format_messages(
                context=context,
                question=query,
                chat_history=self.chat_history,
            )
        else:
            messages = prompt.format_messages(
                context=context,
                question=query,
            )
        
        # Generate response
        with Timer("Generation"):
            response = self.llm.invoke(messages)
        
        # Update history
        if self.enable_memory:
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response.content))
        
        # Build result
        result = {
            "answer": response.content,
            "latency_ms": int((time.time() - start_time) * 1000),
            "language": language,
            "num_sources": len(documents),
        }
        
        if return_sources:
            result["sources"] = [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                }
                for doc in documents
            ]
        
        if return_context:
            result["context"] = context
        
        return result
    
    def stream(self, query: str) -> Iterator[Dict[str, Any]]:
        """
        Stream response with metadata.
        
        Args:
            query: User query
            
        Yields:
            Response chunks with metadata
        """
        language = self._get_language(query)
        
        # Retrieve
        documents = self.retriever.invoke(query)
        if self.enable_reranker and documents:
            documents = rerank_documents(query, documents)
        
        # Yield sources first
        yield {
            "type": "sources",
            "sources": [
                {"content": doc.page_content[:300], "metadata": doc.metadata}
                for doc in documents
            ],
        }
        
        if not documents:
            yield {"type": "answer", "content": "No relevant information found."}
            return
        
        # Format and generate
        context = format_documents(documents)
        prompt = get_rag_prompt(language=language)
        messages = prompt.format_messages(context=context, question=query)
        
        # Stream answer
        for chunk in self.llm.stream(messages):
            if hasattr(chunk, "content"):
                yield {"type": "answer", "content": chunk.content}
        
        yield {"type": "done"}
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.chat_history = []


# =============================================================================
# Chain Factory
# =============================================================================

def create_rag_chain(
    chain_type: str = "basic",
    **kwargs,
) -> Any:
    """
    Create RAG chain based on type.
    
    Args:
        chain_type: Type of chain ("basic", "conversational", "advanced")
        **kwargs: Additional arguments
        
    Returns:
        RAG chain instance
    """
    if chain_type == "basic":
        return RAGChain(**kwargs)
    elif chain_type == "conversational":
        return ConversationalRAGChain(**kwargs)
    elif chain_type == "advanced":
        return AdvancedRAGChain(**kwargs)
    else:
        raise ValueError(f"Unknown chain type: {chain_type}")


# =============================================================================
# Quick Query Function
# =============================================================================

def query_climate_docs(
    question: str,
    return_sources: bool = True,
    language: str = "auto",
) -> Dict[str, Any]:
    """
    Quick function to query climate documents.
    
    Args:
        question: User question
        return_sources: Whether to return sources
        language: Response language
        
    Returns:
        Answer with optional sources
    """
    chain = AdvancedRAGChain(language=language)
    return chain.invoke(question, return_sources=return_sources)
