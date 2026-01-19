"""
LLM interface for Climate Q&A RAG System.

Supports multiple LLM providers: OpenAI, Anthropic, Google.
"""

from typing import Any, Dict, Iterator, List, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from loguru import logger

from .config import settings
from .utils import Timer


# =============================================================================
# LLM Factory
# =============================================================================

def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: bool = False,
    **kwargs,
) -> BaseChatModel:
    """
    Get LLM instance based on model name.
    
    Args:
        model_name: Model name (default from settings)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        streaming: Whether to enable streaming
        **kwargs: Additional model arguments
        
    Returns:
        LLM instance
    """
    model_name = model_name or settings.llm_model
    temperature = temperature if temperature is not None else settings.llm_temperature
    max_tokens = max_tokens or settings.llm_max_tokens
    
    # Determine provider from model name
    if model_name.startswith("gpt"):
        return get_openai_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )
    elif model_name.startswith("claude"):
        return get_anthropic_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )
    elif model_name.startswith("gemini"):
        return get_google_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    else:
        # Default to OpenAI
        logger.warning(f"Unknown model prefix, defaulting to OpenAI: {model_name}")
        return get_openai_llm(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            **kwargs,
        )


def get_openai_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    streaming: bool = False,
    **kwargs,
) -> ChatOpenAI:
    """
    Get OpenAI LLM instance.
    
    Args:
        model_name: OpenAI model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens
        streaming: Enable streaming
        **kwargs: Additional arguments
        
    Returns:
        ChatOpenAI instance
    """
    if not settings.has_openai:
        raise ValueError("OpenAI API key not configured")
    
    logger.info(f"Initializing OpenAI LLM: {model_name}")
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        openai_api_key=settings.openai_api_key,
        **kwargs,
    )


def get_anthropic_llm(
    model_name: str = "claude-3-haiku-20240307",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    streaming: bool = False,
    **kwargs,
) -> BaseChatModel:
    """
    Get Anthropic LLM instance.
    
    Args:
        model_name: Claude model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens
        streaming: Enable streaming
        **kwargs: Additional arguments
        
    Returns:
        ChatAnthropic instance
    """
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic not available. "
            "Install with: pip install langchain-anthropic"
        )
    
    if not settings.has_anthropic:
        raise ValueError("Anthropic API key not configured")
    
    logger.info(f"Initializing Anthropic LLM: {model_name}")
    
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        anthropic_api_key=settings.anthropic_api_key,
        **kwargs,
    )


def get_google_llm(
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    **kwargs,
) -> BaseChatModel:
    """
    Get Google AI LLM instance.
    
    Args:
        model_name: Gemini model name
        temperature: Temperature for generation
        max_tokens: Maximum tokens
        **kwargs: Additional arguments
        
    Returns:
        ChatGoogleGenerativeAI instance
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai not available. "
            "Install with: pip install langchain-google-genai"
        )
    
    if not settings.google_api_key:
        raise ValueError("Google API key not configured")
    
    logger.info(f"Initializing Google AI LLM: {model_name}")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=settings.google_api_key,
        **kwargs,
    )


# =============================================================================
# LLM Wrapper with Utilities
# =============================================================================

class LLMWrapper:
    """
    Wrapper for LLM with additional utilities.
    """
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize LLM wrapper.
        
        Args:
            llm: Pre-configured LLM instance
            model_name: Model name for new instance
        """
        self.llm = llm or get_llm(model_name=model_name)
        self._token_count = 0
    
    def invoke(
        self,
        messages: Union[str, List[BaseMessage]],
        **kwargs,
    ) -> AIMessage:
        """
        Invoke LLM with messages.
        
        Args:
            messages: String or list of messages
            **kwargs: Additional arguments
            
        Returns:
            AI response message
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        
        with Timer("LLM invoke"):
            response = self.llm.invoke(messages, **kwargs)
        
        # Track token usage if available
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            self._token_count += usage.get("total_tokens", 0)
        
        return response
    
    def stream(
        self,
        messages: Union[str, List[BaseMessage]],
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream LLM response.
        
        Args:
            messages: String or list of messages
            **kwargs: Additional arguments
            
        Yields:
            Response chunks
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        
        for chunk in self.llm.stream(messages, **kwargs):
            if hasattr(chunk, "content"):
                yield chunk.content
    
    def generate_with_system(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs,
    ) -> str:
        """
        Generate response with system prompt.
        
        Args:
            system_prompt: System message
            user_message: User message
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.invoke(messages, **kwargs)
        return response.content
    
    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self._token_count
    
    def reset_token_count(self) -> None:
        """Reset token counter."""
        self._token_count = 0


# =============================================================================
# Model Information
# =============================================================================

LLM_MODELS = {
    "openai": {
        "gpt-4o": {
            "context_window": 128000,
            "input_cost_per_1k": 0.0025,
            "output_cost_per_1k": 0.01,
        },
        "gpt-4o-mini": {
            "context_window": 128000,
            "input_cost_per_1k": 0.00015,
            "output_cost_per_1k": 0.0006,
        },
        "gpt-4-turbo": {
            "context_window": 128000,
            "input_cost_per_1k": 0.01,
            "output_cost_per_1k": 0.03,
        },
    },
    "anthropic": {
        "claude-3-opus-20240229": {
            "context_window": 200000,
            "input_cost_per_1k": 0.015,
            "output_cost_per_1k": 0.075,
        },
        "claude-3-sonnet-20240229": {
            "context_window": 200000,
            "input_cost_per_1k": 0.003,
            "output_cost_per_1k": 0.015,
        },
        "claude-3-haiku-20240307": {
            "context_window": 200000,
            "input_cost_per_1k": 0.00025,
            "output_cost_per_1k": 0.00125,
        },
    },
    "google": {
        "gemini-1.5-pro": {
            "context_window": 2000000,
            "input_cost_per_1k": 0.00125,
            "output_cost_per_1k": 0.005,
        },
        "gemini-1.5-flash": {
            "context_window": 1000000,
            "input_cost_per_1k": 0.000075,
            "output_cost_per_1k": 0.0003,
        },
    },
}


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Get information about an LLM model.
    
    Args:
        model_name: Model name
        
    Returns:
        Model information dictionary
    """
    for provider, models in LLM_MODELS.items():
        if model_name in models:
            return {"provider": provider, **models[model_name]}
    return {}


def list_available_models() -> Dict[str, List[str]]:
    """
    List all available LLM models.
    
    Returns:
        Dictionary of providers and their models
    """
    return {
        provider: list(models.keys())
        for provider, models in LLM_MODELS.items()
    }


def estimate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """
    Estimate cost for LLM usage.
    
    Args:
        model_name: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Estimated cost in USD
    """
    info = get_model_info(model_name)
    if not info:
        return 0.0
    
    input_cost = (input_tokens / 1000) * info.get("input_cost_per_1k", 0)
    output_cost = (output_tokens / 1000) * info.get("output_cost_per_1k", 0)
    
    return input_cost + output_cost
