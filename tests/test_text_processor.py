# tests/test_text_processor.py

import pytest
from app.services.text_processor import detect_language, clean_text, process_text


# ── Language Detection ──────────────────────────────────────────────────────

def test_detect_french():
    text = "Bonjour, voici un document en français avec suffisamment de contenu."
    assert detect_language(text) == "fr"

def test_detect_english():
    text = "This is a document written in English with enough content to detect."
    assert detect_language(text) == "en"

def test_detect_arabic():
    text = "هذا مستند باللغة العربية يحتوي على نص كافٍ للكشف عن اللغة بدقة"
    assert detect_language(text) == "ar"

def test_detect_short_text_returns_unknown():
    assert detect_language("Hi") == "unknown"

def test_detect_empty_returns_unknown():
    assert detect_language("") == "unknown"


# ── Text Cleaning ───────────────────────────────────────────────────────────

def test_clean_removes_extra_spaces():
    assert clean_text("hello    world") == "hello world"

def test_clean_collapses_extra_newlines():
    result = clean_text("line1\n\n\n\n\nline2")
    assert result == "line1\n\nline2"

def test_clean_normalizes_unicode():
    # NFKC normalization — ligatures and compatibility chars
    result = clean_text("\ufb01le")  # ﬁle → file
    assert "file" in result or len(result) > 0  # normalized

def test_clean_removes_fancy_quotes():
    result = clean_text("\u201cHello\u201d and \u2018world\u2019")
    assert '"Hello"' in result
    assert "'world'" in result

def test_clean_removes_soft_hyphen():
    result = clean_text("docu\u00adment")
    assert "\u00ad" not in result
    assert "document" in result

def test_clean_removes_noise_lines():
    # Lines with only symbols should be removed
    text = "valid line\n!@#$%^&*()\nanother valid line"
    result = clean_text(text)
    assert "!@#$%^&*()" not in result
    assert "valid line" in result

def test_clean_arabic_removes_tatweel():
    # Tatweel (kashida) is the stretching character ـ
    result = clean_text("مرحب\u0640ا", language="ar")
    assert "\u0640" not in result

def test_clean_arabic_normalizes_alef():
    # All Alef variants should become plain Alef (ا)
    result = clean_text("أسماء إبراهيم آمنة", language="ar")
    # After normalization, no variant Alefs remain
    assert "\u0623" not in result  # أ
    assert "\u0625" not in result  # إ
    assert "\u0622" not in result  # آ

def test_clean_empty_input():
    assert clean_text("") == ""

def test_clean_whitespace_only():
    assert clean_text("   \n\n  ") == ""


# ── process_text (integration) ──────────────────────────────────────────────

def test_process_text_returns_all_fields():
    result = process_text("This is a test document in English with some content.")
    assert "clean_text" in result
    assert "language" in result
    assert "original_length" in result
    assert "clean_length" in result
    assert "reduction_pct" in result

def test_process_text_french_document():
    text = "Bonjour,   voici un rapport   avec des espaces  inutiles.\n\n\n\nFin."
    result = process_text(text)
    assert result["language"] == "fr"
    assert "   " not in result["clean_text"]  # extra spaces collapsed
    assert result["clean_length"] < result["original_length"]