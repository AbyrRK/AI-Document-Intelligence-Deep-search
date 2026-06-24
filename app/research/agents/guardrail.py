"""
app/research/agents/guardrail_agent.py
=======================================
Agent 1 — Domain Gatekeeper

Refuses off-topic questions immediately, before any expensive LLM/API call.
Replies in the SAME language as the user's question.
"""

import importlib
_re = importlib.import_module("re")

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

# Hard off-topic signals — if any match, refuse immediately
OFF_TOPIC_SIGNALS = {
    # Nature / geography
    "flower", "plant", "garden", "animal", "weather", "climate",
    "mountain", "beach", "ocean", "river", "lake", "forest",
    "fleur", "plante", "jardin", "animal", "météo", "montagne",
    "زهرة", "نبتة", "حيوان", "طقس", "جبل", "شاطئ", "نهر",

    # Entertainment
    "movie", "film", "song", "music", "game", "sport", "recipe",
    "film", "chanson", "musique", "jeu", "sport", "recette",
    "فيلم", "أغنية", "موسيقى", "لعبة", "رياضة", "وصفة",

    # Personal / medical
    "diet", "health", "doctor", "medicine", "love", "relationship",
    "santé", "médecin", "amour", "relation",
    "صحة", "طبيب", "دواء", "حب", "علاقة",

    # Coding / tech (unrelated to documents)
    "python tutorial", "javascript", "html css", "algorithm",
    "sort array", "machine learning tutorial",
}


def _fast_keyword_check(question: str) -> str:
    """
    Fast path: keyword scan before any LLM call.
    Returns: "proceed" | "refuse" | "uncertain"
    """
    q_lower = question.lower()
    words = set(_re.findall(r'\b\w+\b', q_lower))

    # Check hard off-topic signals first
    for signal in OFF_TOPIC_SIGNALS:
        if signal in q_lower:
            return "refuse"

    # Check business keywords
    if words & BUSINESS_KEYWORDS:
        return "proceed"

    # Also check Arabic tokens directly (no word boundaries for Arabic)
    for kw in BUSINESS_KEYWORDS:
        if "\u0600" <= kw[0] <= "\u06FF" and kw in question:
            return "proceed"

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
    detected_lang = detect_language(question)
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
    # Only reached for ambiguous questions (no clear keyword match)
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
    except Exception:
        # If LLM fails, default to proceed (don't block the user on a guardrail error)
        return {
            **state,
            "guardrail_decision": "proceed",
            "guardrail_reason": "LLM guardrail failed — defaulting to proceed",
            "language": effective_lang,
        }


def _llm_guardrail_check(question: str, language: str) -> str:
    """
    Use Gemini to classify ambiguous questions.
    Returns "proceed" or "refuse".
    """
    from google import genai
    from app.core.config import settings

    prompt = f"""You are a business document assistant domain classifier.

Classify this question as either:
- "proceed"  → the question is about business documents (invoices, contracts, reports, financial data, company info, suppliers, payments, tax, etc.)
- "refuse"   → the question is NOT about business documents (nature, entertainment, personal topics, unrelated tech, etc.)

Question: "{question}"

Reply with ONLY one word: proceed  OR  refuse"""

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    result = response.text.strip().lower()
    return "proceed" if "proceed" in result else "refuse"

guardrail_node = run_guardrail
