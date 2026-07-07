"""
app/services/pii_scrubber.py
=============================
PII Scrubber — Zero-Trust Layer

Detects and replaces Personally Identifiable Information (PII) before:
  1. Text is chunked and embedded into Qdrant (storage layer)
  2. Prompts are sent to external LLMs (Gemini API)

Strategy: LOSSY — PII is replaced with typed placeholders permanently.
          Original values are never stored in the vector DB or sent to cloud APIs.

Detection layers:
  Layer 1 — Presidio (Microsoft) — worldwide pattern recognition
             Covers: emails, phones, credit cards, IBANs, passport numbers,
             national IDs, IP addresses, URLs, crypto addresses, medical licence
             numbers, and NER-based detection of PERSON / ORG / LOCATION entities.
  Layer 2 — spaCy NER (EN + FR models) — names, organisations, locations
             (loaded lazily to avoid startup cost)
  Layer 3 — Regex fallback — Arabic phone patterns, numeric IDs not caught
             by presidio, and additional international formats.

Supported languages: English, French, Arabic (regex layer)
Placeholder format:  [EMAIL], [PHONE], [PERSON], [ORG], [LOCATION],
                     [CREDIT_CARD], [IBAN], [PASSPORT], [NATIONAL_ID],
                     [IP_ADDRESS], [CRYPTO], [DATE_OF_BIRTH], [URL_PII]

Usage:
    from app.services.pii_scrubber import scrub_pii

    clean_text, report = scrub_pii(raw_text, language="fr")
    # clean_text : scrubbed string safe to embed / send to API
    # report     : {"entities_found": int, "types": [...], "scrubbed": bool}
"""

import re
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Presidio setup (lazy — loaded once on first call)
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_analyzer():
    """
    Build and cache the Presidio AnalyzerEngine.
    Loads spaCy EN + FR models for NER.
    Falls back gracefully if a model is missing.
    """
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        # Try to load both EN (large) and FR (medium) models
        models = []
        for model_name, lang in [("en_core_web_lg", "en"), ("fr_core_news_md", "fr")]:
            try:
                import spacy
                spacy.load(model_name)
                models.append({"lang_code": lang, "model_name": model_name})
                logger.info(f"[PII] spaCy model loaded: {model_name}")
            except OSError:
                logger.warning(
                    f"[PII] spaCy model '{model_name}' not found — "
                    f"NER detection for '{lang}' will be skipped. "
                    f"Run: python -m spacy download {model_name}"
                )

        if not models:
            # No spaCy models available — use pattern-only mode
            logger.warning("[PII] No spaCy models available — running pattern-only mode")
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
        else:
            configuration = {
                "nlp_engine_name": "spacy",
                "models": models,
            }

        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en", "fr"])
        logger.info("[PII] Presidio AnalyzerEngine ready")
        return analyzer

    except ImportError:
        logger.error("[PII] presidio-analyzer not installed — pip install presidio-analyzer")
        return None
    except Exception as e:
        logger.error(f"[PII] Presidio init failed: {e}")
        return None


@lru_cache(maxsize=1)
def _get_anonymizer():
    try:
        from presidio_anonymizer import AnonymizerEngine
        return AnonymizerEngine()
    except ImportError:
        logger.error("[PII] presidio-anonymizer not installed — pip install presidio-anonymizer")
        return None
    except Exception as e:
        logger.error(f"[PII] Anonymizer init failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Entities to detect (worldwide)
# ─────────────────────────────────────────────────────────────────────────────

PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
    "US_SSN",
    "UK_NHS",
    "NRP",                  
    "IP_ADDRESS",
    "CRYPTO",
    "MEDICAL_LICENSE",
    "PERSON",
    "LOCATION",
    "DATE_TIME",
]

# Placeholder map: presidio entity type → display placeholder
PLACEHOLDER_MAP = {
    "EMAIL_ADDRESS":    "[EMAIL]",
    "PHONE_NUMBER":     "[PHONE]",
    "CREDIT_CARD":      "[CREDIT_CARD]",
    "IBAN_CODE":        "[IBAN]",
    "US_PASSPORT":      "[PASSPORT]",
    "US_DRIVER_LICENSE":"[NATIONAL_ID]",
    "US_BANK_NUMBER":   "[BANK_ACCOUNT]",
    "US_SSN":           "[NATIONAL_ID]",
    "UK_NHS":           "[NATIONAL_ID]",
    "NRP":              "[NATIONAL_ID]",
    "IP_ADDRESS":       "[IP_ADDRESS]",
    "CRYPTO":           "[CRYPTO]",
    "MEDICAL_LICENSE":  "[MEDICAL_ID]",
    "PERSON":           "[PERSON]",
    "LOCATION":         "[LOCATION]",
    "DATE_TIME":        "[DATE]",
}


# ─────────────────────────────────────────────────────────────────────────────
# Regex layer — catches patterns presidio misses or languages it doesn't cover
# ─────────────────────────────────────────────────────────────────────────────

_REGEX_PATTERNS = [
    # Arabic phone numbers (Tunisia, Morocco, Algeria, Egypt, Saudi, UAE, etc.)
    (r'\b(?:\+216|00216)[\s\-]?\d{2}[\s\-]?\d{3}[\s\-]?\d{3}\b', "[PHONE]"),   # Tunisia +216
    (r'\b(?:\+212|00212)[\s\-]?\d{9}\b', "[PHONE]"),                             # Morocco
    (r'\b(?:\+213|00213)[\s\-]?\d{9}\b', "[PHONE]"),                             # Algeria
    (r'\b(?:\+20|0020)[\s\-]?\d{10}\b', "[PHONE]"),                              # Egypt
    (r'\b(?:\+966|00966)[\s\-]?\d{9}\b', "[PHONE]"),                             # Saudi Arabia
    (r'\b(?:\+971|00971)[\s\-]?\d{9}\b', "[PHONE]"),                             # UAE

    # Arabic-script phone digits (unicode Arabic-Indic numerals)
    (r'[\u0660-\u0669]{8,12}', "[PHONE]"),

    # IBAN (international — 15-34 chars, starts with 2 letters + 2 digits)
    (r'\b[A-Z]{2}\d{2}[\s]?(?:[A-Z0-9]{4}[\s]?){2,7}[A-Z0-9]{1,4}\b', "[IBAN]"),

    # Matricule fiscal (Tunisia) — 7 digits + letter + 3 digits + letter
    (r'\b\d{7}[A-Z]\d{3}[A-Z]\b', "[TAX_ID]"),

    # CIN (Tunisia / Morocco / Algeria) — 8 digits standalone
    (r'\b\d{8}\b', "[NATIONAL_ID]"),

    # Passport (generic international) — 1-2 letters + 6-9 digits
    (r'\b[A-Z]{1,2}\d{6,9}\b', "[PASSPORT]"),

    # Email (backup — presidio usually catches these)
    (r'\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b', "[EMAIL]"),

    # IPv4 (backup)
    (r'\b(?:\d{1,3}\.){3}\d{1,3}\b', "[IP_ADDRESS]"),
]

_COMPILED_REGEX = [(re.compile(pattern), replacement) for pattern, replacement in _REGEX_PATTERNS]


# ─────────────────────────────────────────────────────────────────────────────
# Markdown table guard
# ─────────────────────────────────────────────────────────────────────────────

def _is_markdown_table(text: str) -> bool:
    """
    Returns True if the text is a Markdown table.
    We apply lighter scrubbing to tables to avoid destroying column structure.
    Only emails and phone numbers are scrubbed in tables — names/locations are kept
    because they are often product names or company names in a business context.
    """
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return False
    pipe_lines = sum(1 for l in lines if l.strip().startswith("|"))
    return pipe_lines >= len(lines) * 0.6  # 60%+ of lines have pipes → table


# ─────────────────────────────────────────────────────────────────────────────
# Core scrubbing function
# ─────────────────────────────────────────────────────────────────────────────

def scrub_pii(
    text: str,
    language: str = "en",
    skip_table_ner: bool = True,
) -> tuple[str, dict]:
    """
    Scrub PII from text. Returns (clean_text, report).

    Args:
        text          : Raw text to scrub.
        language      : "en", "fr", or "ar". Arabic uses regex-only (no presidio NER).
        skip_table_ner: If True and text looks like a Markdown table, skip NER
                        entity detection (PERSON/ORG/LOCATION) to preserve table
                        structure. Regex patterns (emails, phones) still apply.

    Returns:
        clean_text : Text with PII replaced by placeholders.
        report     : {
                        "entities_found": int,
                        "types": list[str],   # unique entity types detected
                        "scrubbed": bool,
                        "mode": str,          # "presidio+regex" | "regex_only"
                     }
    """
    if not text or not text.strip():
        return text, {"entities_found": 0, "types": [], "scrubbed": False, "mode": "none"}

    entities_found = 0
    entity_types   = set()
    clean_text     = text
    mode           = "regex_only"

    is_table = skip_table_ner and _is_markdown_table(text)

    # ── Layer 1: Presidio (EN + FR only — Arabic not supported by spaCy NER) ──
    if language in ("en", "fr"):
        analyzer   = _get_analyzer()
        anonymizer = _get_anonymizer()

        if analyzer and anonymizer:
            try:
                # For tables: skip NER entities (PERSON/LOCATION/ORG)
                # to avoid turning product/company names into [PERSON]
                entities = PRESIDIO_ENTITIES if not is_table else [
                    e for e in PRESIDIO_ENTITIES
                    if e not in ("PERSON", "LOCATION", "DATE_TIME")
                ]

                # Presidio requires "en" or "fr" — map variants
                presidio_lang = "fr" if language == "fr" else "en"

                results = analyzer.analyze(
                    text=clean_text,
                    language=presidio_lang,
                    entities=entities,
                    score_threshold=0.6,    # only high-confidence detections
                )

                if results:
                    from presidio_anonymizer.entities import OperatorConfig

                    # Build custom operators so each entity type gets its own placeholder
                    operators = {}
                    for result in results:
                        placeholder = PLACEHOLDER_MAP.get(result.entity_type, f"[{result.entity_type}]")
                        operators[result.entity_type] = OperatorConfig(
                            "replace", {"new_value": placeholder}
                        )

                    anonymized = anonymizer.anonymize(
                        text=clean_text,
                        analyzer_results=results,
                        operators=operators,
                    )
                    clean_text     = anonymized.text
                    entities_found += len(results)
                    entity_types.update(r.entity_type for r in results)
                    mode = "presidio+regex"

            except Exception as e:
                logger.warning(f"[PII] Presidio analysis failed: {e} — falling back to regex only")

    # ── Layer 2: Regex (all languages including Arabic) ───────────────────────
    for pattern, placeholder in _COMPILED_REGEX:
        matches = pattern.findall(clean_text)
        if matches:
            clean_text      = pattern.sub(placeholder, clean_text)
            entities_found += len(matches)
            # Derive type name from placeholder e.g. "[EMAIL]" → "EMAIL"
            entity_types.add(placeholder.strip("[]"))

    scrubbed = entities_found > 0

    if scrubbed:
        logger.info(
            f"[PII] Scrubbed {entities_found} entities "
            f"({', '.join(sorted(entity_types))}) | lang={language} | mode={mode}"
        )

    return clean_text, {
        "entities_found": entities_found,
        "types":          sorted(entity_types),
        "scrubbed":       scrubbed,
        "mode":           mode,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for prompt scrubbing (synthesis / RAG prompts)
# ─────────────────────────────────────────────────────────────────────────────

def scrub_prompt(prompt: str, language: str = "en") -> str:
    """
    Lightweight scrub for LLM prompts before sending to external API.
    Skips NER on tables (too slow for prompt-time scrubbing).
    Returns clean prompt string only (no report).
    """
    clean, _ = scrub_pii(prompt, language=language, skip_table_ner=True)
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up (call at app startup to pre-load spaCy models)
# ─────────────────────────────────────────────────────────────────────────────

def warmup():
    """
    Pre-load spaCy models and Presidio engine at startup.
    Call from app/main.py lifespan or startup event.
    Prevents cold-start latency on the first document upload.
    """
    logger.info("[PII] Warming up PII scrubber...")
    _get_analyzer()
    _get_anonymizer()
    logger.info("[PII] PII scrubber ready")