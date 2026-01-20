"""
LLM Factory for EIA Generator.

Supports multiple LLM providers: Google Gemini, Ollama (local), OpenAI.
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from loguru import logger

from .config import get_settings


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.4,
    **kwargs,
) -> BaseChatModel:
    """
    Get LLM instance based on model name.
    
    Args:
        model_name: Model name (auto-detects provider from prefix)
        temperature: Temperature for generation
        **kwargs: Additional model arguments
        
    Returns:
        LLM instance
        
    Supported models:
        - Gemini: gemini-2.0-flash, gemini-1.5-flash, gemini-1.5-pro
        - Ollama: gemma3:4b, llama3.2, mistral, phi, etc.
        - OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
    """
    settings = get_settings()
    model_name = model_name or settings.default_model
    
    # Determine provider from model name
    if model_name.startswith("gemini"):
        return get_gemini_llm(model_name, temperature, **kwargs)
    elif model_name.startswith("gpt"):
        return get_openai_llm(model_name, temperature, **kwargs)
    elif ":" in model_name or model_name in ["gemma", "llama", "mistral", "phi", "qwen"]:
        # Ollama local models (e.g., gemma3:4b, llama3.2, mistral)
        return get_ollama_llm(model_name, temperature, **kwargs)
    else:
        # Default to Ollama for unknown models (assume local)
        logger.warning(f"Unknown model prefix, trying Ollama: {model_name}")
        return get_ollama_llm(model_name, temperature, **kwargs)


def get_gemini_llm(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.4,
    **kwargs,
) -> BaseChatModel:
    """Get Google Gemini LLM instance."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai not available. "
            "Install with: pip install langchain-google-genai"
        )
    
    settings = get_settings()
    if not settings.google_api_key:
        raise ValueError("Google API key not configured. Set GOOGLE_API_KEY in .env")
    
    logger.info(f"Initializing Gemini LLM: {model_name}")
    
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=settings.google_api_key,
        **kwargs,
    )


def get_ollama_llm(
    model_name: str = "gemma3:4b",
    temperature: float = 0.4,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """
    Get Ollama local LLM instance.
    
    Args:
        model_name: Ollama model name (e.g., gemma3:4b, llama3.2, mistral, phi)
        temperature: Temperature for generation
        base_url: Ollama server URL (default from settings or http://localhost:11434)
        **kwargs: Additional arguments
        
    Returns:
        ChatOllama instance
        
    Note:
        Ensure Ollama is running: `ollama serve`
        Pull model first: `ollama pull gemma3:4b`
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama not available. "
            "Install with: pip install langchain-ollama"
        )
    
    settings = get_settings()
    base_url = base_url or settings.ollama_base_url
    
    logger.info(f"Initializing Ollama LLM: {model_name} at {base_url}")
    
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        **kwargs,
    )


def get_openai_llm(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.4,
    **kwargs,
) -> BaseChatModel:
    """Get OpenAI LLM instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not available. "
            "Install with: pip install langchain-openai"
        )
    
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")
    
    logger.info(f"Initializing OpenAI LLM: {model_name}")
    
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=settings.openai_api_key,
        **kwargs,
    )


# =============================================================================
# Model Information
# =============================================================================

LLM_MODELS = {
    "gemini": {
        "gemini-2.0-flash": {"context_window": 1000000, "cost": "free"},
        "gemini-1.5-flash": {"context_window": 1000000, "cost": "free"},
        "gemini-1.5-pro": {"context_window": 2000000, "cost": "paid"},
    },
    "ollama": {
        "gemma3:4b": {"context_window": 8192, "cost": "free", "vram": "3GB"},
        "gemma3:12b": {"context_window": 8192, "cost": "free", "vram": "8GB"},
        "llama3.2:3b": {"context_window": 128000, "cost": "free", "vram": "2GB"},
        "mistral:7b": {"context_window": 32768, "cost": "free", "vram": "5GB"},
        "phi3:mini": {"context_window": 4096, "cost": "free", "vram": "2GB"},
        "qwen2.5:7b": {"context_window": 128000, "cost": "free", "vram": "5GB"},
    },
    "openai": {
        "gpt-4o": {"context_window": 128000, "cost": "paid"},
        "gpt-4o-mini": {"context_window": 128000, "cost": "paid"},
    },
}


def list_available_models() -> dict:
    """List all available LLM models."""
    return LLM_MODELS
