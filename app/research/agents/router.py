# app/research/agents/router.py
"""
Agent 3 — Router (Strategy Decider)

Reads:  question, reformulated_query, session_has_documents, language
Writes: retrieval_mode
        validated_answer  (only when mode == "none" — written here so the
                           graph can jump straight to END without going
                           through synthesis or citation_formatter)
"""

from google import genai
from app.core.config import settings
from app.research.state import ResearchState

# Multilingual "please upload a file" messages
_NO_DOCS_MSG = {
    "en": "No documents found in this session. Please upload a file first.",
    "fr": "Aucun document trouvé dans cette session. Veuillez d'abord télécharger un fichier.",
    "ar": "لم يتم العثور على وثائق في هذه الجلسة. يرجى تحميل ملف أولاً.",
    "es": "No se encontraron documentos en esta sesión. Por favor, suba un archivo primero.",
    "de": "Keine Dokumente in dieser Sitzung gefunden. Bitte laden Sie zuerst eine Datei hoch.",
    "it": "Nessun documento trovato in questa sessione. Si prega di caricare prima un file.",
}

_CLARIFY_MSG = {
    "en": "Your question is too vague. Could you clarify what you are looking for?",
    "fr": "Votre question est trop vague. Pourriez-vous préciser ce que vous cherchez ?",
    "ar": "سؤالك غامض جداً. هل يمكنك توضيح ما تبحث عنه؟",
    "es": "Su pregunta es demasiado vaga. ¿Podría aclarar qué está buscando?",
    "de": "Ihre Frage ist zu vage. Könnten Sie klarstellen, was Sie suchen?",
    "it": "La sua domanda è troppo vaga. Potrebbe chiarire cosa sta cercando?",
}


def router_node(state: ResearchState) -> dict:
    question           = state.get("question", "")
    reformulated_query = state.get("reformulated_query") or question
    session_has_docs   = state.get("session_has_documents", False)
    doc_id             = state.get("doc_id", "")
    doc_ids            = state.get("doc_ids", [])
    language           = state.get("language", "en")

    # FIX: single, clean computation of effective_has_docs — no duplicate calls
    has_explicit_docs  = bool(doc_id) or bool(doc_ids)
    effective_has_docs = has_explicit_docs or session_has_docs

    print(
        f"🔀 ROUTER: session_has_docs={session_has_docs} "
        f"explicit_docs={has_explicit_docs} "
        f"doc_id={doc_id[:8] if doc_id else 'NONE'} "
        f"doc_ids={len(doc_ids) if doc_ids else 0}"
    )

    # Single call to _decide_mode
    mode = _decide_mode(question, reformulated_query, effective_has_docs)

    if mode == "none:no_docs":
        return {
            **state,
            "retrieval_mode":   "none",
            "validated_answer": _NO_DOCS_MSG.get(language, _NO_DOCS_MSG["en"]),
        }

    if mode == "none:vague":
        return {
            **state,
            "retrieval_mode":   "none",
            "validated_answer": _CLARIFY_MSG.get(language, _CLARIFY_MSG["en"]),
        }

    return {
        **state,
        "retrieval_mode": mode,  # "rag" | "web" | "hybrid"
    }


def _decide_mode(question: str, reformulated_query: str, has_docs: bool) -> str:
    """
    Heuristic + LLM decision.
    Returns: "rag" | "web" | "hybrid" | "none:no_docs" | "none:vague"
    """
    print(f"🔀 ROUTER: has_docs={has_docs} q='{question[:60]}'")

    q = question.lower().strip()

    # Fast path — too vague
    if len(q.split()) < 2 and not any(c.isalpha() for c in q):
        return "none:vague"

    # Fast path — references uploaded docs but none exist
    doc_refs = {
        "my contract", "my invoice", "my file", "my document",
        "mon contrat", "ma facture", "mon fichier",
        "عقدي", "فاتورتي", "ملفي", "وثيقتي",
    }
    mentions_doc = any(ref in q for ref in doc_refs)

    if mentions_doc and not has_docs:
        return "none:no_docs"

    if mentions_doc and has_docs:
        return "hybrid" if _needs_current_info(q) else "rag"

    if not has_docs:
        return "web"

    # Document exists but no explicit doc reference
    if _needs_current_info(q):
        return "hybrid"

    # LLM fallback for truly ambiguous cases
    try:
        return _llm_route(question, has_docs)
    except Exception:
        return "hybrid"  # safe default when docs exist


def _needs_current_info(q: str) -> bool:
    signals = {
        "law", "regulation", "rate", "2024", "2025", "2026", "current",
        "latest", "new", "loi", "règlement", "taux", "actuel", "nouveau",
        "قانون", "نسبة", "حالي", "جديد",
        "tva", "vat", "tax", "taxe", "tunisie", "tunisia",
        "correct", "correctement", "vérifie", "vérifiez",
        "actuelle", "calcul", "conforme", "fiscal",
    }
    return any(s in q for s in signals)


def _llm_route(question: str, has_docs: bool) -> str:
    prompt = f"""You are a query router for a business document assistant.

Session has uploaded documents: {has_docs}

Question: "{question}"

Decide the retrieval strategy. Reply with ONLY one of:
- rag      → answer comes from the uploaded documents
- web      → answer requires current external information (laws, news, rates)
- hybrid   → answer needs both uploaded documents AND current external information
- none     → question is too vague to answer; ask for clarification

Reply with ONLY the single word."""

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    result = response.text.strip().lower()

    for mode in ("hybrid", "rag", "web", "none"):
        if mode in result:
            return mode if mode != "none" else "none:vague"

    return "rag" if has_docs else "web"