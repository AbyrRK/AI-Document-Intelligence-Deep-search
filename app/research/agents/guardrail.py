"""
app/research/agents/guardrail_agent.py
=======================================
Agent 1 — Domain Gatekeeper

Refuses off-topic questions immediately, before any expensive LLM/API call.
Replies in the SAME language as the user's question.
"""

import re
from functools import lru_cache

from app.core.language import detect_language, get_guardrail_refusal

# ---------------------------------------------------------------------------
# Business keyword lists (multilingual) for fast path check
# ---------------------------------------------------------------------------
BUSINESS_KEYWORDS = {
    # English
    "invoice", "contract", "report", "quote", "quotation", "offer",
    "payment", "balance", "tax", "vat", "revenue", "expense", "budget",
    "supplier", "vendor", "client", "customer", "order", "purchase",
    "agreement", "clause", "penalty", "deadline", "delivery", "amount",
    "financial", "statement", "accounting", "audit", "fiscal", "bill",
    "receipt", "document", "file", "company", "business", "firm",

    # French
    "facture", "contrat", "rapport", "devis", "offre", "paiement",
    "solde", "tva", "taxe", "revenu", "depense", "budget", "fournisseur",
    "client", "commande", "achat", "accord", "clause", "penalite",
    "livraison", "montant", "financier", "comptabilite", "audit",
    "fiscal", "recu", "document", "societe", "entreprise",

    # Arabic
    "فاتورة", "عقد", "تقرير", "عرض", "دفع", "رصيد", "ضريبة",
    "إيراد", "مصروف", "ميزانية", "مورد", "عميل", "طلب", "شراء",
    "اتفاقية", "بند", "غرامة", "تسليم", "مبلغ", "مالي", "محاسبة",
    "تدقيق", "ضريبي", "وصل", "وثيقة", "شركة", "مؤسسة",
}

# Pre-filter Arabic keywords for O(1) fast-path check
ARABIC_BUSINESS_KEYWORDS = {kw for kw in BUSINESS_KEYWORDS if "\u0600" <= kw[0] <= "\u06FF"}

# Hard off-topic signals — if any match, refuse immediately
OFF_TOPIC_SIGNALS = {
    # Nature / geography
    "flower", "plant", "garden", "animal", "weather", "climate",
    "mountain", "beach", "ocean", "river", "lake", "forest",
    "fleur", "plante", "jardin", "animal", "météo", "montagne",
    "زهرة", "نبتة", "حيوان", "طقس", "جبل", "شاطئ", "نهر",

    # Entertainment / pop culture
    "movie", "film", "song", "music", "game", "sport", "recipe",
    "film", "chanson", "musique", "jeu", "sport", "recette",
    "فيلم", "أغنية", "موسيقى", "لعبة", "رياضة", "وصفة",
    "batman", "superhero", "marvel", "dc comics", "comic", "anime",
    "cartoon", "netflix", "spotify", "celebrity", "actor", "singer",

    # Personal / medical
    "diet", "health", "doctor", "medicine", "love", "relationship",
    "santé", "médecin", "amour", "relation",
    "صحة", "طبيب", "دواء", "حب", "علاقة",

    # Coding / tech (unrelated to documents)
    "python tutorial", "javascript", "html css", "algorithm",
    "sort array", "machine learning tutorial",
}

# General-knowledge phrasing patterns — catch "who is X" / "how is X" / "tell
# me about X" style questions that won't hit a keyword list but are almost
# never about business documents.
GENERAL_KNOWLEDGE_PATTERNS = [
    re.compile(r"\bwho is\b"), re.compile(r"\bhow is\b"),
    re.compile(r"\btell me about\b"), re.compile(r"\bwhat is\b.*\?$"),
    re.compile(r"\bqui est\b"), re.compile(r"\bcomment va\b"),
    re.compile(r"\bparle[- ]moi de\b"),
    re.compile(r"\bمن هو\b"), re.compile(r"\bكيف حال\b"),
]


def _fast_keyword_check(question: str) -> str:
    """
    Fast path: keyword scan before any LLM call.
    Returns: "proceed" | "refuse" | "uncertain"
    """
    q_lower = question.lower()

    # Check hard off-topic signals first (substring match)
    for signal in OFF_TOPIC_SIGNALS:
        if signal in q_lower:
            return "refuse"

    # Check business keywords using word boundaries for Latin scripts
    # NOTE: \b\w+\b does NOT work reliably for Arabic, so we handle Arabic separately
    words = set(re.findall(r"\b[a-zA-Z0-9_]+\b", q_lower))
    if words & BUSINESS_KEYWORDS:
        return "proceed"

    # Check Arabic tokens directly (no word boundaries for Arabic)
    for kw in ARABIC_BUSINESS_KEYWORDS:
        if kw in question:
            return "proceed"

    # General-knowledge phrasing with no business keyword present is a strong
    # off-topic signal (e.g. "how is batman", "who is the CEO of Apple" would
    # still hit BUSINESS_KEYWORDS via "company"/"CEO" isn't listed — extend
    # BUSINESS_KEYWORDS rather than special-casing here if that case matters).
    for pattern in GENERAL_KNOWLEDGE_PATTERNS:
        if pattern.search(q_lower):
            return "refuse"

    return "uncertain"


def run_guardrail(state: dict) -> dict:
    """
    Guardrail agent node for LangGraph.

    Reads:  state["question"], state["language"]
    Writes: state["guardrail_decision"], state["final_answer"] (if refused)

    The refusal message is always in the SAME language as the question.
    """
    question: str = state.get("question", "")
    language: str = state.get("language", "en")

    # Always detect from the actual question text (more reliable than frontend hint)
    try:
        detected_lang = detect_language(question)
    except Exception:
        detected_lang = language  # fallback to state hint if detector fails

    effective_lang = detected_lang if detected_lang != "en" else language

    # --- Fast path ---
    fast = _fast_keyword_check(question)

    if fast == "proceed":
        return {
            **state,
            "guardrail_decision": "proceed",
            "guardrail_reason": "Business keyword matched (fast path)",
            "language": effective_lang,
        }

    if fast == "refuse":
        return {
            **state,
            "guardrail_decision": "refuse",
            "guardrail_reason": "Off-topic signal matched (fast path)",
            "final_answer": get_guardrail_refusal(effective_lang),
            "language": effective_lang,
        }

    # --- Slow path: LLM disambiguation ---
    try:
        llm_decision = _llm_guardrail_check(question, effective_lang)
        if llm_decision == "proceed":
            return {
                **state,
                "guardrail_decision": "proceed",
                "guardrail_reason": "LLM classified as business-related",
                "language": effective_lang,
            }
        else:
            return {
                **state,
                "guardrail_decision": "refuse",
                "guardrail_reason": "LLM classified as off-topic",
                "final_answer": get_guardrail_refusal(effective_lang),
                "language": effective_lang,
            }
    except Exception as e:
        # Fail CLOSED: if the LLM classifier is unavailable, refuse rather than
        # silently let an unclassified question bypass domain scoping. A flaky
        # API call should not be able to disable the guardrail.
        return {
            **state,
            "guardrail_decision": "refuse",
            "guardrail_reason": f"LLM guardrail failed ({type(e).__name__}) — defaulting to refuse",
            "final_answer": get_guardrail_refusal(effective_lang),
            "language": effective_lang,
        }


@lru_cache(maxsize=1)
def _get_gemini_client():
    """Lazy-load and cache the Gemini client."""
    from google import genai
    from app.core.config import settings
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _llm_guardrail_check(question: str, language: str) -> str:
    """
    Use Gemini to classify ambiguous questions.
    Returns "proceed" or "refuse".
    """
    # Sanitize to avoid prompt injection via quotes
    safe_question = question.replace('"', "'").replace("{", "").replace("}", "")

    prompt = f"""You are a business document assistant domain classifier.
The user's question is written in language code "{language}".

Classify this question as either:
- "proceed"  → the question is about business documents (invoices, contracts, reports, financial data, company info, suppliers, payments, tax, etc.)
- "refuse"   → the question is NOT about business documents (nature, entertainment, personal topics, unrelated tech, etc.)

Question: "{safe_question}"

Reply with ONLY one word: proceed  OR  refuse"""

    client = _get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    result = response.text.strip().lower()
    return "proceed" if "proceed" in result else "refuse"


guardrail_node = run_guardrail