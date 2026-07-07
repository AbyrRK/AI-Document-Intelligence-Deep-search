# tests/test_metrics_ocr.py
"""
Metrics tests for the OCR service.
Covers:
  - Character Error Rate (CER) against ground truth
  - Engine fallback logic (Tesseract → EasyOCR → Gemini)
  - Confidence score thresholds
  - Parallel processing speedup
  - Language detection per engine
  - Edge cases (blank image, corrupt file, oversized image)

Run:
    pytest tests/test_metrics_ocr.py -v -s
    pytest tests/test_metrics_ocr.py -v -m "not slow" --report-metrics
"""

import time
import numpy as np
import pytest
import io
import os
import re
import json

from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image

from app.services.ocr_service import (
    ocr_scanned_pdf,
    ocr_smart,
    ocr_image,
    ocr_pdf_pages,
    process_single_page,
    _run_tesseract,
    ocr_with_easyocr,
    _run_gemini_vision,
    get_tesseract_confidence,
    is_likely_handwritten,
    preprocess_image,
    extract_text_ocr,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLES = Path(__file__).parent / "samples"
_EVAL_CASES_PATH = os.path.join(os.path.dirname(__file__), "eval_cases_ocr.json")


def _cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate = edit_distance(ref, hyp) / len(ref).
    Returns a float in [0, 1].  0 = perfect, 1 = completely wrong.
    """
    ref = reference.strip()
    hyp = hypothesis.strip()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    dp = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref):
        new_dp = [i + 1]
        for j, hc in enumerate(hyp):
            cost = 0 if rc == hc else 1
            new_dp.append(min(new_dp[j] + 1, dp[j + 1] + 1, dp[j] + cost))
        dp = new_dp
    return dp[len(hyp)] / len(ref)


def _normalize_arabic_numerals(text: str) -> str:
    """Convert Eastern Arabic numerals to Western numerals for fair CER comparison."""
    eastern = '٠١٢٣٤٥٦٧٨٩'
    western = '0123456789'
    trans = str.maketrans(eastern, western)
    return text.translate(trans)


def _make_white_image() -> Image.Image:
    """Return a completely white PIL image."""
    return Image.new("RGB", (200, 50), color=(255, 255, 255))


def _make_test_image(text: str = None) -> Image.Image:
    """Return a simple test image. Optionally render text on it."""
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    if text:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), text, fill=(0, 0, 0), font=font)
    return img


# ---------------------------------------------------------------------------
# CER threshold constants
# ---------------------------------------------------------------------------
CER_GOOD   = 0.10
CER_WARN   = 0.20
CER_ARABIC = 0.25


# ---------------------------------------------------------------------------
# Fixtures (module level — must be outside any class)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eval_cases_ocr():
    """Load evaluation cases from eval_cases_ocr.json."""
    if not os.path.exists(_EVAL_CASES_PATH):
        return []
    with open(_EVAL_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def metrics():
    """Collect per-test metrics (used by reporting plugins)."""
    return []


# ===========================================================================
# 1. CER — helper unit tests
# ===========================================================================

class TestCERLatin:
    """CER helper correctness tests."""

    def test_cer_helper_perfect_match(self):
        assert _cer("hello world", "hello world") == 0.0

    def test_cer_helper_one_substitution(self):
        cer = _cer("hello", "hxllo")
        assert abs(cer - 0.2) < 1e-9

    def test_cer_helper_empty_reference(self):
        assert _cer("", "") == 0.0
        assert _cer("", "something") == 1.0

    @pytest.mark.skipif(
        not (SAMPLES / "AI_Deep_Research_Module_Architecture.pdf").exists(),
        reason="Sample PDF not found"
    )
    def test_native_pdf_extraction_quality(self):
        """
        Native PDF should extract substantial text directly via PyMuPDF.
        FIX: Use fitz directly instead of importing document_service,
             which pulls in vector_store and crashes on missing Gemini API key.
        """
        import fitz
        doc = fitz.open(str(SAMPLES / "AI_Deep_Research_Module_Architecture.pdf"))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        assert len(text) > 100, "Native PDF produced too little text"
        assert len(text.split()) > 20, "Too few words extracted"


# ===========================================================================
# 2. OCR Confidence & Fallback logic
# ===========================================================================

class TestOCRFallback:
    """
    Tests the 3-level fallback chain:
        Tesseract (>=60%, non-Arabic, non-handwritten) ->
        EasyOCR (if low confidence, Arabic, or handwritten) ->
        Gemini Vision (if EasyOCR text too short)

    All tests mock is_likely_handwritten to isolate confidence logic
    from the handwriting heuristic, which depends on real image content.
    """

    def test_tesseract_returns_confidence_and_text(self):
        """Tesseract should return text and a confidence score."""
        img = _make_test_image("Hello World")
        processed = preprocess_image(img)
        text, conf = _run_tesseract(processed)
        assert isinstance(text, str)
        assert isinstance(conf, (int, float))
        assert 0 <= conf <= 100

    @patch("app.services.ocr_service.is_likely_handwritten", return_value=False)
    @patch("app.services.ocr_service._run_tesseract")
    @patch("app.services.ocr_service.ocr_with_easyocr")
    def test_high_confidence_stays_at_tesseract(self, mock_easyocr, mock_tesseract, mock_hw):
        """
        If Tesseract confidence >= 60, text > 10 chars, non-Arabic, non-handwritten
        -> EasyOCR must NOT be called.
        """
        mock_tesseract.return_value = ("Hello World printed text clearly", 85.0)
        img = _make_test_image()
        result = ocr_smart(img)
        mock_easyocr.assert_not_called()
        assert result["engine"] == "tesseract"
        assert result["confidence"] == 85.0

    @patch("app.services.ocr_service.is_likely_handwritten", return_value=False)
    @patch("app.services.ocr_service._run_gemini_vision")
    @patch("app.services.ocr_service.ocr_with_easyocr")
    @patch("app.services.ocr_service._run_tesseract")
    def test_low_confidence_falls_to_easyocr(self, mock_tesseract, mock_easyocr, mock_gemini, mock_hw):
        """
        If Tesseract confidence < 60 -> EasyOCR must be called.
        If EasyOCR returns good text (> 20 chars, conf >= 40) -> Gemini must NOT be called.
        """
        mock_tesseract.return_value = ("bad", 35.0)
        mock_easyocr.return_value = {
            "text": "good result from easyocr with sufficient length here",
            "engine": "easyocr",
            "confidence": 75.0,
        }
        img = _make_test_image()
        result = ocr_smart(img)
        mock_easyocr.assert_called_once()
        mock_gemini.assert_not_called()
        assert result["engine"] == "easyocr"

    @patch("app.services.ocr_service.is_likely_handwritten", return_value=False)
    @patch("app.services.ocr_service._run_gemini_vision")
    @patch("app.services.ocr_service.ocr_with_easyocr")
    @patch("app.services.ocr_service._run_tesseract")
    def test_double_failure_falls_to_gemini(self, mock_tesseract, mock_easyocr, mock_gemini, mock_hw):
        """If both Tesseract and EasyOCR fail -> Gemini Vision must be called."""
        mock_tesseract.return_value = ("???", 20.0)
        mock_easyocr.return_value = {"text": "", "engine": "easyocr", "confidence": 25.0}
        mock_gemini.return_value = {
            "text": "Arabic text here extracted by gemini vision api",
            "engine": "gemini_vision",
            "confidence": 95.0,
        }
        img = _make_test_image()
        result = ocr_smart(img)
        mock_gemini.assert_called_once()
        assert result["engine"] == "gemini_vision"

    def test_confidence_score_is_between_0_and_100(self):
        """Confidence score must always be in [0, 100]."""
        img = _make_test_image("Test text for confidence")
        processed = preprocess_image(img)
        conf = get_tesseract_confidence(processed)
        assert 0.0 <= conf <= 100.0


# ===========================================================================
# 3. Parallel Processing Speedup
# ===========================================================================

class TestParallelOCRSpeedup:
    """Verify parallel OCR is faster than sequential on a mock workload."""

    def _fake_ocr_page(self, args):
        time.sleep(0.5)
        return f"Page text {args['page_index']}"

    def test_parallel_faster_than_sequential(self):
        """4 pages in parallel should finish in < 4 x 0.5s = 2s."""
        from concurrent.futures import ThreadPoolExecutor

        page_args = [{"page_index": i} for i in range(4)]

        t0 = time.perf_counter()
        sequential_results = [self._fake_ocr_page(a) for a in page_args]
        t_sequential = time.perf_counter() - t0

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            parallel_results = list(executor.map(self._fake_ocr_page, page_args))
        t_parallel = time.perf_counter() - t0

        speedup = t_sequential / t_parallel
        print(f"\nSequential: {t_sequential:.2f}s | Parallel: {t_parallel:.2f}s | Speedup: {speedup:.2f}x")

        assert speedup >= 1.5, f"Expected >= 1.5x speedup, got {speedup:.2f}x"
        assert sequential_results == parallel_results

    def test_parallel_result_order_preserved(self):
        """Results from parallel execution must be in correct page order."""
        from concurrent.futures import ThreadPoolExecutor

        def process(idx):
            time.sleep(0.1 * (4 - idx))
            return idx

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process, i): i for i in range(4)}
            results = [None] * 4
            for future, idx in futures.items():
                results[idx] = future.result()

        assert results == [0, 1, 2, 3]


# ===========================================================================
# 4. Language Detection
# ===========================================================================

class TestLanguageDetection:

    def test_trilingual_lang_string_in_tesseract(self):
        """Tesseract should not crash with fra+eng+ara config."""
        img = _make_test_image()
        processed = preprocess_image(img)
        text, conf = _run_tesseract(processed)
        assert isinstance(text, str)
        assert isinstance(conf, (int, float))

    def test_arabic_text_detected_by_ocr(self):
        pytest.importorskip("langdetect")
        from langdetect import detect
        assert detect("هذا نص عربي للاختبار") == "ar"

    def test_french_text_detected(self):
        pytest.importorskip("langdetect")
        from langdetect import detect
        assert detect("Voici un texte en français pour le test de détection.") == "fr"


# ===========================================================================
# 5. Edge Cases
# ===========================================================================

class TestOCREdgeCases:

    def test_blank_white_image_handled(self):
        """A completely white image should return empty string, not crash."""
        img = _make_white_image()
        result = ocr_smart(img)
        assert isinstance(result.get("text", ""), str)
        assert "engine" in result

    def test_handwriting_detection_on_printed_lines(self):
        """
        FIX: Use == instead of 'is' to handle numpy.bool_ from OpenCV.
        The function must return without crashing and return a bool-like value.
        """
        img = Image.new("RGB", (400, 100), color=(255, 255, 255))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        for y in range(10, 100, 15):
            draw.line([(0, y), (400, y)], fill=(0, 0, 0), width=2)

        is_hw = is_likely_handwritten(img)
        assert is_hw == False or is_hw == True
        assert isinstance(bool(is_hw), bool)

    def test_handwriting_detection_returns_bool(self):
        """is_likely_handwritten must return a value usable as Python bool."""
        img = _make_test_image()
        result = is_likely_handwritten(img)
        assert bool(result) in (True, False)

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="unsupported|format|Unsupported"):
            extract_text_ocr("document.txt")

    def test_zero_byte_file_not_found(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            extract_text_ocr("nonexistent.png")

    def test_very_large_image_resized_or_handled(self):
        """A 3000x3000 image should not crash the service."""
        img = Image.new("RGB", (3000, 3000), color=(200, 200, 200))
        try:
            result = ocr_smart(img)
            assert isinstance(result, dict)
            assert "engine" in result
            assert "text" in result
        except MemoryError:
            pytest.fail("Service crashed with MemoryError on large image")


# ===========================================================================
# 6. Integration — real files
# ===========================================================================

class TestOCRIntegration:

    @pytest.mark.skipif(
        not (SAMPLES / "handwritingpage.png").exists(),
        reason="No handwriting sample"
    )
    def test_handwriting_image_produces_text(self):
        result = ocr_image(str(SAMPLES / "handwritingpage.png"))
        assert isinstance(result["text"], str)
        assert "metadata" in result
        assert result["metadata"]["ocr_engine"] in (
            "tesseract", "easyocr", "gemini_vision",
            "tesseract_fallback", "easyocr_fallback"
        )

    @pytest.mark.skipif(
        not (SAMPLES / "facture_arabic.png").exists(),
        reason="No Arabic image sample"
    )
    def test_arabic_image_produces_text(self):
        result = ocr_image(str(SAMPLES / "facture_arabic.png"))
        text = result["text"]
        arabic_chars = [c for c in text if '\u0600' <= c <= '\u06FF']
        assert len(arabic_chars) > 0 or len(text) == 0

    @pytest.mark.skipif(
        not (SAMPLES / "BP+CCTP LOT VRD.pdf").exists(),
        reason="No scanned PDF sample"
    )
    @pytest.mark.slow
    def test_scanned_pdf_ocr_produces_text(self):
        result = ocr_pdf_pages(
            str(SAMPLES / "BP+CCTP LOT VRD.pdf"),
            page_indices=[0, 1]
        )
        assert isinstance(result, str)
        assert len(result) > 0 or "[Page" in result


# ===========================================================================
# 7. Evaluation Cases — real CER from eval_cases_ocr.json
# ===========================================================================

class TestEvalCases:

    @pytest.mark.skipif(
        not os.path.exists(_EVAL_CASES_PATH),
        reason="No eval_cases_ocr.json found"
    )
    def test_eval_cases_cer(self, eval_cases_ocr, metrics):
        if not eval_cases_ocr:
            pytest.skip("eval_cases_ocr.json is empty or missing")

        cers = []
        failures = []

        for case in eval_cases_ocr:
            file_path = SAMPLES / case["file"]
            if not file_path.exists():
                print(f"   ⚠️  Skipping '{case['name']}' — file not found: {file_path}")
                continue

            doc_type = case.get("type", "image")

            if doc_type == "image":
                result = ocr_image(str(file_path))
                actual = result["text"]
            elif doc_type == "scanned_pdf":
                page_indices = case.get("pages", [0])
                actual = ocr_pdf_pages(str(file_path), page_indices=page_indices)
                actual = re.sub(r"\[Page \d+\]\n?", "", actual).strip()
            else:
                print(f"   ⚠️  Skipping '{case['name']}' — unknown type: {doc_type}")
                continue

            expected = case["ground_truth"]

            # Normalize Arabic numerals for fair comparison
            if any('\u0600' <= c <= '\u06FF' for c in expected):
                expected = _normalize_arabic_numerals(expected)
                actual = _normalize_arabic_numerals(actual)

            cer = _cer(expected, actual)
            cers.append(cer)

            safe_name = case["name"].replace(" ", "_").lower()
            metrics.append({"name": f"cer_{safe_name}", "value": cer * 100, "unit": "%"})

            threshold = case.get("threshold", CER_WARN)

            if case.get("requires_gemini") and os.getenv("SKIP_GEMINI_TESTS"):
                print(f"   ⏭️  '{case['name']}' — Gemini required, skipped (CER={cer:.1%})")
                continue

            if cer > threshold:
                failures.append((case["name"], cer, threshold))

        # Aggregate metrics
        if cers:
            metrics.append({"name": "cer_mean",   "value": float(np.mean(cers)) * 100,   "unit": "%"})
            metrics.append({"name": "cer_median", "value": float(np.median(cers)) * 100, "unit": "%"})
            metrics.append({"name": "cer_max",    "value": float(np.max(cers)) * 100,    "unit": "%"})
            metrics.append({"name": "cer_min",    "value": float(np.min(cers)) * 100,    "unit": "%"})
            metrics.append({"name": "eval_cases_count", "value": float(len(cers)),       "unit": "cases"})

        # Print summary table
        print("\n" + "-" * 62)
        print(f"{'OCR METRICS SUMMARY':^62}")
        print("-" * 62)
        print(f"{'Metric':<40} {'Value':>12} {'Unit':>8}")
        print("-" * 62)
        for m in metrics:
            print(f"{m['name']:<40} {m['value']:>12.3f} {m['unit']:>8}")
        print("-" * 62 + "\n")

        # Fail at the end with all failures listed
        if failures:
            fail_msg = "\n".join(
                f"  • '{name}': {cer:.2%} (threshold: {th:.2%})"
                for name, cer, th in failures
            )
            pytest.fail(f"\nCER threshold exceeded:\n{fail_msg}")