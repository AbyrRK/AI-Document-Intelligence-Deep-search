"""
app/services/llm_client.py
===========================
Unified LLM client with intelligent caching + automatic fallback.

Priority chain:
  1. Gemini (fast, free tier) — primary
  2. OpenRouter/Claude (cheap, reliable) — fallback on rate limit
  3. Cached response — if available, skip API call entirely

Usage:
    from app.services.llm_client import llm
    answer = llm.generate_sync(prompt, temperature=0.7)
    embedding = llm.embed_sync("text to embed")
"""

import os
import time
import hashlib
from functools import lru_cache
from typing import Optional, List

from google import genai
from google.genai.errors import ServerError, ClientError
from google.genai.types import EmbedContentConfig

from app.services.openrouter_client import OpenRouterClient, RateLimitError, LLMError
from app.services.cache import get_embedding_cache, set_embedding_cache, get_llm_cache, set_llm_cache
from app.core.config import settings

# gemini-embedding-001 produces 3072-dimensional vectors
_EMBEDDING_DIM = 3072


class UnifiedLLMClient:
    """
    Single interface for all LLM operations.
    Handles caching, rate limiting, and provider fallback transparently.
    """

    def __init__(self):
        self.gemini = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.fallback = OpenRouterClient()
        self._gemini_failures = 0
        self._gemini_recovery = 0

    # ═══════════════════════════════════════════════════════════════════════
    # EMBEDDINGS (Gemini only — no good fallback for embeddings yet)
    # ═══════════════════════════════════════════════════════════════════════

    def embed_sync(self, text: str) -> List[float]:
        """
        Synchronous embedding with L2 (Redis) cache.
        Falls back to zero vector on API failure — same dimension as real
        embeddings so Qdrant never receives a mismatched vector.
        """
        cache_key = text.strip().lower()

        # L2: Redis
        cached = get_embedding_cache(cache_key)
        if cached:
            print(f"[LLM] Embedding cache HIT for '{text[:30]}...'")
            return cached

        # API call
        print(f"[LLM] Embedding API call for '{text[:30]}...'")
        try:
            result = self.gemini.models.embed_content(
                model="models/gemini-embedding-001",
                contents=[text],
                config=EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            embedding = result.embeddings[0].values

            # Store in L2 cache (24h TTL)
            set_embedding_cache(cache_key, embedding, ttl=86400)
            return embedding

        except Exception as e:
            print(f"[LLM] Embedding failed: {e}")
            # FIX: was 768 — gemini-embedding-001 produces 3072-dim vectors.
            # Wrong dimension causes Qdrant to reject the vector on insert
            # and return garbage on search.
            return [0.0] * _EMBEDDING_DIM

    def embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Batch embedding — much more efficient than individual calls.
        Gemini supports up to 100 texts per batch.
        """
        if not texts:
            return []

        results = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = text.strip().lower()
            cached = get_embedding_cache(cache_key)
            if cached:
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if not uncached_texts:
            print(f"[LLM] Batch embedding: ALL {len(texts)} cached")
            return [results[i] for i in range(len(texts))]

        print(f"[LLM] Batch embedding: {len(uncached_texts)}/{len(texts)} uncached")
        try:
            response = self.gemini.models.embed_content(
                model="models/gemini-embedding-001",
                contents=uncached_texts,
                config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )

            for idx_in_batch, embedding in enumerate(response.embeddings):
                original_index = uncached_indices[idx_in_batch]
                vector = embedding.values
                results[original_index] = vector

                cache_key = uncached_texts[idx_in_batch].strip().lower()
                set_embedding_cache(cache_key, vector, ttl=86400)

            return [results[i] for i in range(len(texts))]

        except Exception as e:
            print(f"[LLM] Batch embedding failed: {e}. Falling back to individual calls.")
            for idx_in_batch, text in enumerate(uncached_texts):
                original_index = uncached_indices[idx_in_batch]
                results[original_index] = self.embed_sync(text)

            return [results[i] for i in range(len(texts))]

    # ═══════════════════════════════════════════════════════════════════════
    # TEXT GENERATION (Gemini → OpenRouter fallback)
    # ═══════════════════════════════════════════════════════════════════════

    def generate_sync(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """
        Synchronous generation with caching + automatic fallback.

        Flow:
          1. Check LLM cache (2h TTL for identical prompts)
          2. Try Gemini
          3. On rate limit → fallback to OpenRouter/Claude
          4. Cache successful result
        """
        # 1. Check cache
        # FIX: include system_prompt in hash — different system prompts on the
        # same user prompt must not collide in cache
        if use_cache:
            prompt_hash = hashlib.sha256(
                f"{prompt}:{system_prompt or ''}:{temperature}:{max_tokens}".encode()
            ).hexdigest()[:16]
            cached = get_llm_cache(prompt_hash)
            if cached:
                print(f"[LLM] Response cache HIT")
                return cached

        # 2. Try Gemini
        if self._gemini_failures < 3 or time.time() > self._gemini_recovery:
            try:
                answer = self._call_gemini_sync(prompt, temperature, max_tokens, system_prompt)
                self._gemini_failures = 0

                if use_cache:
                    set_llm_cache(prompt_hash, answer, ttl=7200)

                return answer

            except (ServerError, ClientError) as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                    self._gemini_failures += 1
                    self._gemini_recovery = time.time() + 60
                    print(f"[LLM] Gemini rate limited (failure {self._gemini_failures}/3). Switching to fallback...")
                else:
                    print(f"[LLM] Gemini error: {e}. Trying fallback...")
            except Exception as e:
                print(f"[LLM] Gemini unexpected error: {e}. Trying fallback...")
        else:
            print(f"[LLM] Gemini in cooldown. Using fallback directly.")

        # 3. Fallback to OpenRouter
        try:
            answer = self.fallback.generate_sync(
                prompt=prompt,
                model="anthropic/claude-haiku-4-5",
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt
            )
            print(f"[LLM] Fallback (OpenRouter/Claude) succeeded")

            if use_cache:
                set_llm_cache(prompt_hash, answer, ttl=7200)

            return answer

        except RateLimitError:
            print(f"[LLM] CRITICAL: Both Gemini and OpenRouter rate limited")
            return "[System temporarily overloaded. Please try again in 1-2 minutes.]"
        except Exception as e:
            print(f"[LLM] Fallback failed: {e}")
            return "[System error. Please try again.]"

    def _call_gemini_sync(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str]
    ) -> str:
        """Raw Gemini call — no caching, no fallback."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        response = self.gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        return response.text

    # ═══════════════════════════════════════════════════════════════════════
    # ASYNC VARIANTS (for FastAPI endpoints)
    # ═══════════════════════════════════════════════════════════════════════

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> str:
        """Async wrapper — runs sync version in thread pool."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate_sync(prompt, temperature, max_tokens, system_prompt, use_cache)
        )

    async def embed(self, text: str) -> List[float]:
        """Async wrapper for embed_sync."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_sync, text)

    # ═══════════════════════════════════════════════════════════════════════
    # STATS & HEALTH
    # ═══════════════════════════════════════════════════════════════════════

    def health(self) -> dict:
        return {
            "gemini_ready":    bool(settings.GEMINI_API_KEY),
            "fallback_ready":  bool(self.fallback.api_key),
            "gemini_failures": self._gemini_failures,
            "gemini_cooldown": max(0, self._gemini_recovery - time.time()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL SINGLETON — this is what other files import
# ═══════════════════════════════════════════════════════════════════════════
llm = UnifiedLLMClient()