import logging
from typing import Optional, AsyncIterator, Dict, Any
import asyncio

import litellm
from litellm import completion, acompletion
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import settings

logger = logging.getLogger(__name__)

# Configure litellm
litellm.set_verbose = settings.debug


class LLMClient:
    """Wrapper for LLM interactions using litellm."""
    
    def __init__(
        self,
        provider: str = settings.llm_provider,
        model: str = settings.llm_model,
        temperature: float = settings.llm_temperature,
        max_tokens: int = settings.llm_max_tokens,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set API keys based on provider
        self._set_api_keys()
        
        logger.info(f"Initialized LLM client: {provider}/{model}")
    
    def _set_api_keys(self) -> None:
        """Set API keys for litellm."""
        if self.provider == "openai" and settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key
        elif self.provider == "anthropic" and settings.anthropic_api_key:
            litellm.anthropic_key = settings.anthropic_api_key
        elif self.provider == "google" and settings.google_api_key:
            litellm.google_key = settings.google_api_key
    
    def _build_model_string(self) -> str:
        """Build the model string for litellm."""
        if self.provider in ["openai", "anthropic", "google"]:
            return f"{self.provider}/{self.model}"
        return self.model
    
    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate completion synchronously."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = completion(
                model=self._build_model_string(),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                timeout=settings.llm_timeout,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(settings.llm_max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate completion asynchronously."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await acompletion(
                model=self._build_model_string(),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                timeout=settings.llm_timeout,
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating async completion: {e}")
            raise
    
    async def agenerate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate completion with streaming."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await acompletion(
                model=self._build_model_string(),
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                timeout=settings.llm_timeout,
                stream=True,
            )
            
            async for chunk in response:
                if hasattr(chunk.choices[0].delta, "content"):
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        
        except Exception as e:
            logger.error(f"Error generating streaming completion: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4


# Global LLM client instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client