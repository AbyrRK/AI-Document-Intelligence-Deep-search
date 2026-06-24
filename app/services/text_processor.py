# app/services/text_processor.py

import re
import unicodedata
from langdetect import detect, DetectorFactory, LangDetectException

# Make langdetect deterministic (same text = same result every time)
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {"fr", "en", "ar"}


def detect_language(text: str) -> str:
    """
    Detects the dominant language of the extracted text.
    Returns ISO 639-1 code: 'fr', 'en', 'ar', or 'unknown'.
    Falls back to 'unknown' if text is too short or detection fails.
    """
    sample = text.strip()[:2000]  # ✅ Sample only — faster, still accurate
    if len(sample) < 20:
        return "unknown"

    try:
        lang = detect(sample)
        return lang if lang in SUPPORTED_LANGUAGES else "unknown"  # ✅ explicit fallback
    except LangDetectException:
        return "unknown"


def clean_text(text: str, language: str = "unknown") -> str:
    """
    Cleans extracted text before chunking.
    Handles common OCR artifacts, encoding issues, and noisy characters.

    Steps:
        1. Unicode normalization
        2. Fix common OCR artifacts
        3. Remove control characters (keep newlines and tabs)
        4. Arabic-specific cleaning
        5. Remove noise-only lines
        6. Collapse excessive newlines
        7. Collapse multiple spaces
        8. Remove repeated punctuation
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode normalization — fixes garbled accented chars (é→é, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix common OCR artifacts
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    text = text.replace("\u00ad", "")   # soft hyphen
    text = text.replace("\u00a0", " ")  # non-breaking space

    # 3. Remove control characters — keep \n (newline) and \t (tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 4. Arabic-specific cleaning
    if language == "ar":
        text = text.replace("\u0640", "")                    # remove tatweel
        text = re.sub(r"[\u0622\u0623\u0625]", "\u0627", text)  # normalize Alef variants

    # 5. Remove lines that are pure noise (only symbols, no letters/digits)
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "" or re.search(r"[\w\u0600-\u06ff]", stripped):
            clean_lines.append(stripped)
    text = "\n".join(clean_lines)

    # 6. Collapse 3+ consecutive newlines into 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 7. Collapse multiple spaces into one (but not newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # 8. Remove repeated punctuation
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"-{3,}", "---", text)
    text = re.sub(r"_{3,}", "___", text)

    return text.strip()


def process_text(raw_text: str) -> dict:
    """
    Main entry point — runs detection then cleaning.
    Returns a dict ready to pass to the chunker.

    Usage:
        result = process_text(extracted_text)
        # result["clean_text"]  → cleaned text
        # result["language"]    → 'fr' | 'en' | 'ar' | 'unknown'
    """
    language = detect_language(raw_text)
    clean    = clean_text(raw_text, language=language)
    return {
        "clean_text":      clean,
        "language":        language,
        "original_length": len(raw_text),
        "clean_length":    len(clean),
        "word_count":      len(clean.split()),           
        "reduction_pct":   round(max(0.0, (1 - len(clean) / max(len(raw_text), 1)) * 100), 1),
    }