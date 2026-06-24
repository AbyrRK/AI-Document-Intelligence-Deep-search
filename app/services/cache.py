
"""
app/services/cache.py
=====================
Multi-layer cache: Local LRU (per-process) + Redis (shared).
Handles embeddings, query results, and LLM responses.
"""

import json
import hashlib
import time
from functools import lru_cache
from typing import Optional, Any
import redis
from app.core.config import settings


class CacheManager:
    """
    Two-tier cache:
      L1: in-memory LRU (fastest, per-process)
      L2: Redis (shared across workers, persists across restarts)
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        self._redis: Optional[redis.Redis] = None
        self._local_hits = 0
        self._redis_hits = 0
        self._misses = 0

    @property
    def redis_client(self) -> Optional[redis.Redis]:
        """Lazy connection — only connects when first needed."""
        if self._redis is None:
            try:
                self._redis = redis.Redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    health_check_interval=30
                )
                self._redis.ping()
            except Exception as e:
                print(f"[Cache] Redis unavailable: {e}. Using local-only cache.")
                self._redis = None
        return self._redis

    def _key(self, namespace: str, text: str) -> str:
        """Deterministic hash key."""
        h = hashlib.sha256(f"{namespace}:{text}".encode()).hexdigest()[:16]
        return f"pfe2026:{namespace}:{h}"

    def get(self, namespace: str, text: str) -> Optional[Any]:
        """
        L1 → L2 lookup. Returns parsed value or None.
        """
        key = self._key(namespace, text)

        # L1: Not implemented here — use @lru_cache on the caller
        # L2: Redis
        r = self.redis_client
        if r:
            try:
                cached = r.get(key)
                if cached:
                    self._redis_hits += 1
                    return json.loads(cached)
            except Exception:
                pass

        self._misses += 1
        return None

    def set(self, namespace: str, text: str, value: Any, ttl_seconds: int = 3600):
        """Store in L2 (Redis). L1 is handled by @lru_cache on callers."""
        key = self._key(namespace, text)
        r = self.redis_client
        if r:
            try:
                r.setex(key, ttl_seconds, json.dumps(value))
            except Exception:
                pass

    def invalidate(self, namespace: str, pattern: str = "*"):
        """Clear a namespace. Use with care."""
        r = self.redis_client
        if r:
            try:
                for key in r.scan_iter(match=f"pfe2026:{namespace}:{pattern}"):
                    r.delete(key)
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "redis_hits": self._redis_hits,
            "misses": self._misses,
            "redis_connected": self._redis is not None,
        }


# Global singleton
cache = CacheManager()


# ── Convenience wrappers for common namespaces ───────────────────────────────

def get_embedding_cache(text: str) -> Optional[list]:
    return cache.get("embedding", text)

def set_embedding_cache(text: str, embedding: list, ttl: int = 86400):
    cache.set("embedding", text, embedding, ttl)

def get_query_cache(doc_id: str, query: str) -> Optional[list]:
    return cache.get("query", f"{doc_id}:{query}")

def set_query_cache(doc_id: str, query: str, results: list, ttl: int = 3600):
    cache.set("query", f"{doc_id}:{query}", results, ttl)

def get_llm_cache(prompt_hash: str) -> Optional[str]:
    return cache.get("llm", prompt_hash)

def set_llm_cache(prompt_hash: str, response: str, ttl: int = 7200):
    cache.set("llm", prompt_hash, response, ttl)