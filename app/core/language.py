"""
app/core/language.py
====================
Single source of truth for language detection.
"""

import importlib
_re = importlib.import_module("re")  # bulletproof: bypasses any local re.py shadow

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0

LANG_MAP = {
    "ar": "ar", "fr": "fr", "en": "en",
    "es": "es", "de": "de", "it": "it",
}

# French keyword fallback for short/ambiguous text
FRENCH_KEYWORDS = {
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "suis", "es", "est", "sommes", "etes", "sont", "etre",
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux",
    "et", "ou", "mais", "donc", "car", "ni",
    "que", "qui", "quoi", "dont", "comment", "pourquoi", "quand",
    "ce", "cette", "ces", "cet", "c",
    "mon", "ton", "son", "notre", "votre", "leur",
    "pour", "dans", "sur", "avec", "sans", "par", "vers", "sous",
    "donne", "donnes", "donnez", "montre", "montrez", "montre-moi",
    "societe", "entreprise", "localisation", "adresse", "siege",
    "facture", "contrat", "rapport", "devis", "tva",
    "bonjour", "merci", "svp", "s'il", "plait",
}

# Arabic keyword fallback for short business text
ARABIC_KEYWORDS = {
    "ما", "هل", "كيف", "متى", "أين", "من", "لماذا",
    "الفاتورة", "العقد", "التقرير", "الشركة", "المورد",
    "الدفع", "المبلغ", "السعر", "الضريبة", "القيمة",
    "أريد", "أحتاج", "أعطني", "اعطني", "اريد", "احتاج",
}


# ---------------------------------------------------------------------------
# Guardrail refusal messages — one per supported language
# ---------------------------------------------------------------------------
GUARDRAIL_REFUSAL = {
    "en": (
        "I am specialized in business documents only "
        "(invoices, contracts, reports, quotes, financial statements). "
        "I cannot help with this question."
    ),
    "fr": (
        "Je suis spécialisé uniquement dans les documents professionnels "
        "(factures, contrats, rapports, devis, états financiers). "
        "Je ne peux pas répondre à cette question."
    ),
    "ar": (
        "أنا متخصص في الوثائق التجارية فقط "
        "(الفواتير، العقود، التقارير، عروض الأسعار، القوائم المالية). "
        "لا أستطيع الإجابة على هذا السؤال."
    ),
    "es": (
        "Estoy especializado únicamente en documentos empresariales "
        "(facturas, contratos, informes, presupuestos, estados financieros). "
        "No puedo responder a esta pregunta."
    ),
    "de": (
        "Ich bin ausschließlich auf Geschäftsdokumente spezialisiert "
        "(Rechnungen, Verträge, Berichte, Angebote, Jahresabschlüsse). "
        "Ich kann diese Frage nicht beantworten."
    ),
    "it": (
        "Sono specializzato esclusivamente in documenti aziendali "
        "(fatture, contratti, report, preventivi, bilanci). "
        "Non posso rispondere a questa domanda."
    ),
}


def get_guardrail_refusal(language: str) -> str:
    """Return the guardrail refusal message in the given language."""
    return GUARDRAIL_REFUSAL.get(language, GUARDRAIL_REFUSAL["en"])


def detect_language(text: str) -> str:
    """
    Detect language with cascading fallback:
    1. Arabic script check (fastest, script-level)
    2. Arabic keyword heuristic (short business text)
    3. French keyword heuristic (reliable for short text)
    4. langdetect (general purpose)
    5. Default to English
    """
    if not text or not text.strip():
        return "en"

    text = text.strip()

    # 1. Arabic script detection (Unicode block U+0600–U+06FF)
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars > 0:
        return "ar"

    # 2. Arabic keyword heuristic (handles transliterated or mixed text)
    words = set(_re.findall(r'\b\w+\b', text.lower()))
    if len(words & ARABIC_KEYWORDS) >= 1:
        return "ar"

    # 3. French keyword heuristic
    if len(words & FRENCH_KEYWORDS) >= 2:
        return "fr"

    # 4. langdetect
    try:
        detected = detect(text)
        return LANG_MAP.get(detected, "en")
    except LangDetectException:
        return "en"


def resolve_language(question: str, frontend_hint: str = "auto") -> str:
    """
    Resolve final language from question + frontend hint.

    Priority:
    1. Arabic/French detection (highly reliable) → trust immediately
    2. langdetect returns non-English → trust it
    3. Frontend hint (if explicitly set and valid)
    4. Default to English
    """
    detected = detect_language(question)

    # If detection is confident (not English default), trust it
    if detected != "en":
        return detected

    # If frontend explicitly set a language, trust it
    if frontend_hint and frontend_hint != "auto":
        return frontend_hint if frontend_hint in LANG_MAP else "en"

    return "en"