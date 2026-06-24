"""
app/services/openrouter_client.py
==================================
Fallback LLM provider. Uses OpenRouter to access Claude, GPT, etc.
"""

import os
import json
import time
from typing import Optional
import httpx


class OpenRouterClient:
    """
    OpenRouter unified API.
    Default model: anthropic/claude-3.5-haiku (fast, cheap, good for synthesis)
    """
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    # Model mapping: (model_id, max_tokens, cost_per_1k_input, cost_per_1k_output)
    MODELS = {
        "claude-haiku": ("anthropic/claude-haiku-4-5", 4096, 0.00025, 0.00125),
        "claude-sonnet": ("anthropic/claude-3.5-sonnet", 4096, 0.003, 0.015),
        "gpt-4o-mini": ("openai/gpt-4o-mini", 4096, 0.00015, 0.0006),
        "gemini-flash": ("google/gemini-2.5-flash", 8192, 0.000, 0.000),
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.http = httpx.AsyncClient(timeout=60.0)
        self.sync_http = httpx.Client(timeout=60.0)
    
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "PFE2026 Document AI",
        }
    
    def generate_sync(self, prompt: str, model: str = "claude-haiku",
                      temperature: float = 0.7, max_tokens: int = 2048,
                      system_prompt: Optional[str] = None) -> str:
        """Synchronous generation — used in Celery tasks."""
        
        model_id, model_max, _, _ = self.MODELS.get(model, self.MODELS["claude-haiku"])
        max_tokens = min(max_tokens, model_max)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = self.sync_http.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            
            return "[OpenRouter: empty response]"
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"OpenRouter rate limited: {e}")
            raise LLMError(f"OpenRouter HTTP error: {e}")
        except Exception as e:
            raise LLMError(f"OpenRouter error: {e}")
    
    async def generate(self, prompt: str, model: str = "claude-haiku",
                       temperature: float = 0.7, max_tokens: int = 2048,
                       system_prompt: Optional[str] = None) -> str:
        """Async generation — used in FastAPI endpoints."""
        model_id, model_max, _, _ = self.MODELS.get(model, self.MODELS["claude-haiku"])
        max_tokens = min(max_tokens, model_max)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            response = await self.http.post(
                f"{self.BASE_URL}/chat/completions",
                headers=self._headers(),
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            
            return "[OpenRouter: empty response]"
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(f"OpenRouter rate limited: {e}")
            raise LLMError(f"OpenRouter HTTP error: {e}")
        except Exception as e:
            raise LLMError(f"OpenRouter error: {e}")


class RateLimitError(Exception):
    pass


class LLMError(Exception):
    pass