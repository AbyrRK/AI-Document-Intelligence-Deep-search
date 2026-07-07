# tests/test_pipeline.py

import pytest
import os
from app.services.document_service import process_document

SAMPLES = {
    "native_pdf_en":   "tests/samples/AI_Deep_Research_Module_Architecture.pdf",
    "docx_fr":         "tests/samples/Devis 32.docx",
    "xlsx_fr":         "tests/samples/Business_Plan_Excel.xlsx",
    "scanned_pdf_fr":  "tests/samples/BP+CCTP LOT VRD.pdf",
    "handwritten_img": "tests/samples/handwritingpage.png",
    "arabic_image":    "tests/samples/facture_arabic.png",
}

AVAILABLE = {k: v for k, v in SAMPLES.items() if os.path.exists(v)}


# ═════════════════════════════════════════════════════════════════════════════
# FAST TESTS
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif("native_pdf_en" not in AVAILABLE, reason="No sample PDF")
def test_full_pipeline_indexes_correctly():
    result = process_document(file_path=SAMPLES["native_pdf_en"], mode="full", index=True)
    extraction = result["extraction"]
    assert result["status"] == "success"
    assert extraction["word_count"] > 0
    assert extraction["language"] in ("fr", "en", "ar", "unknown")
    assert extraction["content"] != ""
    assert extraction["indexed"] is True
    assert extraction["chunks_created"] > 0
    print(f"\n✅ Language: {extraction['language']} | Words: {extraction['word_count']} | Chunks: {extraction['chunks_created']} | Time: {result['processing_time_ms']}ms")


@pytest.mark.skipif("native_pdf_en" not in AVAILABLE, reason="No native PDF")
class TestNativePDF:
    def test_extraction_method_is_native(self):
        result = process_document(file_path=SAMPLES["native_pdf_en"], mode="full", index=True)
        assert result["extraction"]["extraction_method"] == "native"
        assert result["extraction"]["is_scanned"] is False

    def test_page_markers_present(self):
        result = process_document(file_path=SAMPLES["native_pdf_en"], mode="full", index=True)
        assert "[Page" in result["extraction"]["full_text"]

    def test_brief_vs_full(self):
        brief = process_document(file_path=SAMPLES["native_pdf_en"], mode="brief", index=True)
        full = process_document(file_path=SAMPLES["native_pdf_en"], mode="full", index=True)
        assert brief["extraction"]["pages_read"] <= full["extraction"]["pages_total"]


@pytest.mark.skipif("docx_fr" not in AVAILABLE, reason="No DOCX")
class TestDocx:
    def test_extraction_method_is_direct(self):
        result = process_document(file_path=SAMPLES["docx_fr"], mode="full", index=True)
        assert result["extraction"]["extraction_method"] == "direct"
        assert result["extraction"]["is_scanned"] is False

    def test_word_count_positive(self):
        result = process_document(file_path=SAMPLES["docx_fr"], mode="full", index=True)
        assert result["extraction"]["word_count"] > 0

    def test_produces_chunks(self):
        result = process_document(file_path=SAMPLES["docx_fr"], mode="full", index=True)
        assert result["extraction"]["chunks_created"] > 0


@pytest.mark.skipif("xlsx_fr" not in AVAILABLE, reason="No XLSX")
class TestXlsx:
    def test_sheets_read_positive(self):
        result = process_document(file_path=SAMPLES["xlsx_fr"], mode="full", index=True)
        assert result["extraction"]["pages_read"] >= 1

    def test_sheet_markers_in_text(self):
        result = process_document(file_path=SAMPLES["xlsx_fr"], mode="full", index=True)
        assert "[Sheet:" in result["extraction"]["full_text"]

    def test_produces_chunks(self):
        result = process_document(file_path=SAMPLES["xlsx_fr"], mode="full", index=True)
        assert result["extraction"]["chunks_created"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# SLOW TESTS — run with: pytest tests/test_pipeline.py -v -m slow
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif("scanned_pdf_fr" not in AVAILABLE, reason="No scanned PDF")
@pytest.mark.slow
class TestScannedPDFFR:
    def test_is_scanned_and_ocr(self):
        result = process_document(file_path=SAMPLES["scanned_pdf_fr"], mode="full", index=True)
        assert result["extraction"]["is_scanned"] is True
        assert result["extraction"]["extraction_method"] == "ocr"

    def test_word_count_positive(self):
        result = process_document(file_path=SAMPLES["scanned_pdf_fr"], mode="full", index=True)
        assert result["extraction"]["word_count"] > 0

    def test_produces_chunks(self):
        result = process_document(file_path=SAMPLES["scanned_pdf_fr"], mode="full", index=True)
        assert result["extraction"]["chunks_created"] > 0


@pytest.mark.skipif("handwritten_img" not in AVAILABLE, reason="No handwritten image")
@pytest.mark.slow
class TestHandwrittenImage:
    def test_is_scanned(self):
        result = process_document(file_path=SAMPLES["handwritten_img"], mode="full", index=True)
        assert result["extraction"]["is_scanned"] is True

    def test_produces_text_and_chunks(self):
        result = process_document(file_path=SAMPLES["handwritten_img"], mode="full", index=True)
        assert result["extraction"]["word_count"] > 0 or result["extraction"]["char_count"] > 0
        assert result["extraction"]["chunks_created"] > 0


@pytest.mark.skipif("arabic_image" not in AVAILABLE, reason="No Arabic image")
@pytest.mark.slow
class TestArabicImage:
    def test_is_scanned(self):
        result = process_document(file_path=SAMPLES["arabic_image"], mode="full", index=True)
        assert result["extraction"]["is_scanned"] is True

    def test_language_is_arabic(self):
        result = process_document(file_path=SAMPLES["arabic_image"], mode="full", index=True)
        assert result["extraction"]["language"] == "ar"

    def test_arabic_characters_present(self):
        result = process_document(file_path=SAMPLES["arabic_image"], mode="full", index=True)
        text = result["extraction"]["full_text"]
        arabic_chars = [c for c in text if '\u0600' <= c <= '\u06FF']
        assert len(arabic_chars) > 20


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-CUTTING
# ═════════════════════════════════════════════════════════════════════════════

FAST_FORMATS = {k: v for k, v in AVAILABLE.items() if k not in ("scanned_pdf_fr", "handwritten_img", "arabic_image")}

@pytest.mark.parametrize("doc_key", list(FAST_FORMATS.keys()))
def test_doc_id_preserved_for_fast_formats(doc_key):
    test_id = f"TEST-{doc_key.upper()}-123"
    result = process_document(file_path=SAMPLES[doc_key], mode="full", index=True, doc_id=test_id)
    assert result["doc_id"] == test_id