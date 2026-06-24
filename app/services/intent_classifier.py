"""
app/services/intent_classifier.py
===================================
Classifies user prompts into one of three intents using Gemini Flash.
Zero extra cost — uses the same free Gemini API already in the project.

Intents:
  DOCUMENT_LEVEL  → "summarize", "what is this about", empty prompt
  RETRIEVAL       → "what does it say about X", specific questions
  HYBRID          → "analyze all sections about X", comprehensive tasks
"""

from enum import Enum
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings


class QueryIntent(Enum):
    DOCUMENT_LEVEL = "document_level"
    RETRIEVAL      = "retrieval"
    HYBRID         = "hybrid"


# Maps intent → best extraction mode for that intent
def intent_to_extraction_mode(intent: QueryIntent) -> str:
    return {
        QueryIntent.DOCUMENT_LEVEL: "brief",
        QueryIntent.RETRIEVAL:      "brief",   # RAG handles retrieval, not extraction
        QueryIntent.HYBRID:         "summary",
    }[intent]


# ── Keyword-based fast classifier (no API call needed) ──────────────────────

_DOCUMENT_LEVEL_KEYWORDS = {
    "summarize", "summary", "brief", "overview", "what is this",
    "what's this", "about", "describe", "explain this", "tell me about",
    "what does this document", "give me a summary", "résumé", "résumer",
    "ما هذا", "لخص", "اشرح",
}

_HYBRID_KEYWORDS = {
    "analyze all", "comprehensive", "every section", "all sections",
    "full analysis", "detailed analysis", "compare all", "throughout",
}


def _keyword_classify(prompt: str) -> QueryIntent | None:
    """Fast path — no API call. Returns None if uncertain."""
    lower = prompt.lower().strip()

    if not lower:
        return QueryIntent.DOCUMENT_LEVEL

    for kw in _HYBRID_KEYWORDS:
        if kw in lower:
            return QueryIntent.HYBRID

    for kw in _DOCUMENT_LEVEL_KEYWORDS:
        if kw in lower:
            return QueryIntent.DOCUMENT_LEVEL

    return None  # uncertain → fall through to Gemini


# ── Gemini-based classifier (used only when keywords are ambiguous) ──────────

class IntentClassifier:
    def __init__(self):
        self._llm = None  # lazy init — don't load at import time

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.0,   # deterministic classification
            )
        return self._llm

    async def classify(self, prompt: str) -> dict:
        """
        Classify prompt intent.
        Returns: {intent, confidence, reason, via}
        via = "keyword" (fast) or "llm" (Gemini call)
        """
        # Fast path — keyword matching
        keyword_result = _keyword_classify(prompt)
        if keyword_result is not None:
            return {
                "intent":     keyword_result,
                "confidence": 0.95,
                "reason":     "keyword match",
                "via":        "keyword",
            }

        # Slow path — ask Gemini (only for ambiguous prompts)
        try:
            llm    = self._get_llm()
            system = (
                "Classify this user prompt into exactly one category:\n"
                "- DOCUMENT_LEVEL: wants a summary, overview, or brief of the whole document\n"
                "- RETRIEVAL: asks a specific question about a part of the document\n"
                "- HYBRID: wants comprehensive analysis of specific sections\n\n"
                "Reply with ONLY the category name, nothing else."
            )
            response = llm.invoke(f"{system}\n\nPrompt: {prompt}")
            raw      = response.content.strip().upper()

            intent_map = {
                "DOCUMENT_LEVEL": QueryIntent.DOCUMENT_LEVEL,
                "RETRIEVAL":      QueryIntent.RETRIEVAL,
                "HYBRID":         QueryIntent.HYBRID,
            }
            intent = intent_map.get(raw, QueryIntent.RETRIEVAL)

            return {
                "intent":     intent,
                "confidence": 0.85,
                "reason":     f"LLM classified as {raw}",
                "via":        "llm",
            }

        except Exception as e:
            # Fallback — if Gemini fails, assume retrieval
            print(f"⚠️  Intent classification failed ({e}) — defaulting to RETRIEVAL")
            return {
                "intent":     QueryIntent.RETRIEVAL,
                "confidence": 0.5,
                "reason":     f"fallback after error: {e}",
                "via":        "fallback",
            }


classifier = IntentClassifier()