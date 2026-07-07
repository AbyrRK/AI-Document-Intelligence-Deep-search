"""
eval_scorer.py
─────────────────────────────────────────────────────────────────────────────
Reusable scoring module for prompt evaluation.

Usage:
    from eval_scorer import score_answer, is_refusal

    scores = score_answer(answer_text, test_case_dict)
"""

import re
from typing import Dict, Any


# ── Refusal detection (multilingual) ────────────────────────────────────────

REFUSAL_PHRASES = [
    # English
    "not found in document", "don't know", "doesn't contain", "not in the context",
    "cannot find", "not available in the context", "not present in the context",
    "not mentioned in the context", "not contained in", "no information",
    "not provided in", "not included in", "does not mention", "is not mentioned",
    "i don't have", "i do not have", "no relevant information",
    # French
    "je ne sais pas", "pas dans le document", "pas dans le contexte",
    "n'est pas dans", "ne contient pas", "pas disponible", "pas mentionné",
    "n'est pas présent", "introuvable", "aucune information", "pas présente",
    "pas fourni", "n'est pas mentionné", "je n'ai pas", "aucune information",
    # Arabic
    "لا أعرف", "غير متوفرة", "غير موجود", "لا يحتوي", "لا يوجد",
    "ليس في", "غير متاح", "لا تتوفر", "غير موجودة", "لا أملك",
    "ليست متوفرة", "غير مذكور", "لا توجد معلومات",
]


def is_refusal(answer_lower: str) -> bool:
    """Return True if the answer appears to be a refusal in any supported language."""
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)


# ── Scoring function ────────────────────────────────────────────────────────

def score_answer(answer: str, case: Dict[str, Any]) -> Dict[str, float]:
    """
    Score a single answer against a test case.

    Returns a dict with keys:
        contains, no_hallucination, citation, refusal, language, total
    All values are 0.0–1.0.
    """
    answer_lower = answer.lower()
    scores: Dict[str, float] = {}

    # 1. must_contain — partial credit per keyword found
    must = case.get("must_contain", [])
    if must:
        hits = sum(1 for w in must if w.lower() in answer_lower)
        scores["contains"] = round(hits / len(must), 2)
    else:
        scores["contains"] = 1.0  # no requirement → full marks

    # 2. must_not_contain — binary penalty
    must_not = case.get("must_not_contain", [])
    if must_not:
        bad_hits = sum(1 for w in must_not if w.lower() in answer_lower)
        scores["no_hallucination"] = 1.0 if bad_hits == 0 else 0.0
    else:
        scores["no_hallucination"] = 1.0

    # 3. citation
    if case.get("expect_citation"):
        # If the answer is a refusal, we don't penalise missing citation
        if case.get("expect_refusal") and is_refusal(answer_lower):
            scores["citation"] = 1.0
        else:
            has_cite = (
                "[page" in answer_lower
                or "[excerpt" in answer_lower
                or "page " in answer_lower
            )
            scores["citation"] = 1.0 if has_cite else 0.0
    else:
        scores["citation"] = 1.0

    # 4. refusal
    if case.get("expect_refusal"):
        scores["refusal"] = 1.0 if is_refusal(answer_lower) else 0.0
    else:
        scores["refusal"] = 1.0

    # 5. language match
    lang = case.get("expect_lang")
    if lang == "ar":
        arabic_chars = sum(1 for c in answer if "\u0600" <= c <= "\u06FF")
        scores["language"] = 1.0 if arabic_chars > 15 else 0.0
    elif lang == "fr":
        words = set(re.findall(r"\b\w+\b", answer_lower))
        french_markers = {
            "le", "la", "les", "des", "est", "sont", "pour", "dans",
            "je", "pas", "ne", "que", "qui", "ce", "cette", "un", "une",
            "et", "en", "au", "aux", "du", "de", "sur", "avec", "sont",
            "supportés", "fichiers", "système", "informations", "contexte",
            "document", "répondu", "disponible", "présent", "mentionné",
            "budget", "total", "projet", "euros", "base", "données",
        }
        hits = len(words & french_markers)
        scores["language"] = 1.0 if hits >= 2 else 0.5 if hits >= 1 else 0.0
    else:
        scores["language"] = 1.0

    # total = average of all dimensions
    scores["total"] = round(sum(scores.values()) / len(scores), 3)
    return scores