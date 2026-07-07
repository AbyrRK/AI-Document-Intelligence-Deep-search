"""
app/services/ocr_service.py
============================
OCR SERVICE — 3-Level Pipeline
--------------------------------------------
Level 0 : Handwriting detection (OpenCV heuristic)
Level 1 : Tesseract  — fast, free, best for printed text
Level 2 : EasyOCR    — local, free, handles handwriting + Arabic
Level 3 : Gemini Vision — cloud API, best quality, key rotation support,
          BUDGET-LIMITED per document (cost guard)
Fallback : Tesseract result if everything fails
"""

import io
import pytesseract
import numpy as np
import cv2
import easyocr
import fitz

from PIL import Image, ImageEnhance
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

from app.core.config import settings
if settings.TESSERACT_PATH and settings.TESSERACT_PATH != "tesseract":
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_PATH

TESSERACT_THRESHOLD = 60.0   # below this → try EasyOCR
EASYOCR_THRESHOLD   = 40.0   # below this → Tesseract fallback
HANDWRITING_LAP_VAR = 800    # Laplacian variance threshold
HANDWRITING_EDGE    = 0.05   # edge density threshold

# ── Cost guard ───────────────────────────────────────────────────────────────
# Maximum number of pages per document allowed to escalate to Gemini Vision
# (Level 3). Prevents a single low-quality scanned PDF from burning through
# the Gemini API quota/credits page by page. Tunable via settings if needed.
GEMINI_PAGE_BUDGET = getattr(settings, "GEMINI_PAGE_BUDGET", 5)

_key_index = 0

def _get_next_gemini_key() -> str:
    global _key_index
    keys = settings.gemini_keys
    if not keys:
        raise ValueError("No Gemini API keys configured")
    key = keys[_key_index % len(keys)]
    _key_index += 1
    print(f"   🔑 Using Gemini key {(_key_index - 1) % len(keys) + 1}/{len(keys)}")
    return key
# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR — lazy singleton
# ─────────────────────────────────────────────────────────────────────────────


_easyocr_reader_latin = None   # fr + en → for French/Latin handwriting
_easyocr_reader_arabic = None  # ar + en → for Arabic documents

def _get_easyocr_reader(arabic: bool = False) -> easyocr.Reader:
    global _easyocr_reader_latin, _easyocr_reader_arabic
    if arabic:
        if _easyocr_reader_arabic is None:
            print("   🔄 Loading EasyOCR Arabic model...")
            _easyocr_reader_arabic = easyocr.Reader(["ar", "en"], gpu=False)
            print("   ✅ EasyOCR Arabic ready")
        return _easyocr_reader_arabic
    else:
        if _easyocr_reader_latin is None:
            print("   🔄 Loading EasyOCR Latin model (fr+en)...")
            _easyocr_reader_latin = easyocr.Reader(["fr", "en"], gpu=False)
            print("   ✅ EasyOCR Latin ready")
        return _easyocr_reader_latin

# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(image: Image.Image) -> Image.Image:
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Noise removal
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Adaptive threshold (VERY important for handwriting)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Optional: deskew (advanced)
    return Image.fromarray(thresh)

def preprocess_arabic_image(image: Image.Image) -> Image.Image:
    """
    Arabic-specific preprocessing — better than standard adaptive threshold.
    CLAHE improves contrast on low-quality scans.
    Otsu binarization works better than adaptive for dense Arabic script.
    """
    img = np.array(image.convert("L"))

    # CLAHE — enhance local contrast (critical for faded Arabic ink)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Otsu binarization — better than adaptive for Arabic connected script
    _, binary = cv2.threshold(enhanced, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(binary)
# ─────────────────────────────────────────────────────────────────────────────
# Handwriting detection
# ─────────────────────────────────────────────────────────────────────────────

def is_likely_handwritten(image: Image.Image) -> bool:
    """
    Heuristic detector — decides if an image is handwritten or printed.

    Adjusted thresholds to handle:
        - Grid/lined paper backgrounds (common in FR school notebooks)
        - Pink/beige tinted paper
        - Mixed color handwriting (blue, green, red ink)
    """
    img_array     = np.array(image.convert("L"))
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    edges         = cv2.Canny(img_array, 50, 150)
    edge_density  = float(edges.sum()) / float(edges.size)

    # Grid paper adds ~0.03-0.06 edge density from the lines alone
    # Printed documents have very high Laplacian variance (>2000)
    # Handwriting on grid paper sits between 300-1500 Laplacian, 0.03-0.12 edge density
    # So we relax the edge threshold and rely more on Laplacian
    return laplacian_var < 2000 and edge_density < 0.15

# ─────────────────────────────────────────────────────────────────────────────
# Engine: Tesseract
# ─────────────────────────────────────────────────────────────────────────────

def get_tesseract_confidence(image: Image.Image) -> float:
    """
    Runs Tesseract and returns average word confidence (0-100).
    Score >= 60 → result is reliable.
    Score <  60 → fall through to EasyOCR.
    """
    data = pytesseract.image_to_data(
        image,
        lang="fra+eng+ara",
        config="--psm 3 --oem 3",
        output_type=pytesseract.Output.DICT,
    )
    confidences = [
        int(c) for c in data["conf"]
        if str(c) != "-1" and int(c) >= 0
    ]
    return sum(confidences) / len(confidences) if confidences else 0.0


def _run_tesseract(image: Image.Image) -> tuple[str, float]:
    """Returns (text, confidence)."""
    confidence = get_tesseract_confidence(image)
    text = pytesseract.image_to_string(
        image,
        lang="fra+eng+ara",
        config="--psm 3 --oem 3",
    ).strip()
    return text, confidence


# ─────────────────────────────────────────────────────────────────────────────
# Engine: EasyOCR
# ─────────────────────────────────────────────────────────────────────────────

def ocr_with_easyocr(image: Image.Image, has_arabic: bool = False) -> dict:
    reader = _get_easyocr_reader(arabic=has_arabic)
    # Arabic gets dedicated preprocessing before EasyOCR
    if has_arabic:
        image = preprocess_arabic_image(image)
    # Resize large images — EasyOCR doesn't need full resolution
    max_dim = 1600
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"   📐 Resized to {new_size} for EasyOCR")

    img_array = np.array(image.convert("RGB"))
    results   = reader.readtext(img_array, detail=1)
    if not results:
        return {"text": "", "engine": "easyocr", "confidence": 0.0}

    lines          = [text for (_, text, conf) in results if conf > 0.1]
    avg_confidence = sum(conf for (_, _, conf) in results) / len(results)

    return {
        "text":       "\n".join(lines),
        "engine":     "easyocr",
        "confidence": round(avg_confidence * 100, 1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Engine: Gemini Vision (Level 3)
# ─────────────────────────────────────────────────────────────────────────────

def _run_gemini_vision(image: Image.Image) -> dict:
    import base64
    import httpx
    import time

    if not settings.gemini_keys:
        raise ValueError("No Gemini API keys configured")

    # Resize large images to prevent timeouts
    max_dimension = 1024
    if max(image.size) > max_dimension:
        ratio = max_dimension / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"   📐 Resized image to {new_size} for Gemini")

    # Convert PIL image to base64 PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    print(f"   📦 Image payload: {len(buffer.getvalue()) / 1024:.1f} KB")

    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64,
                    }
                },
                {
                    "text": (
                        "Extract ALL text from this image exactly as written. "
                        "Preserve line breaks. Output only the extracted text, "
                        "no explanations. Support Arabic, French, and English."
                    )
                }
            ]
        }]
    }

    timeout_config = httpx.Timeout(connect=10, read=120, write=10, pool=10)
    max_retries = len(settings.gemini_keys) * 2  # try each key twice

    for attempt in range(max_retries):
        try:
            api_key = _get_next_gemini_key()  # ← rotates automatically
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.5-flash:generateContent?key={api_key}"
            )
            print(f"   🌐 Gemini Vision attempt {attempt + 1}/{max_retries}...")

            with httpx.Client(timeout=timeout_config) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

            print(f"   ✅ Gemini succeeded ({len(text)} chars)")
            return {
                "text":        text,
                "engine":      "gemini_vision",
                "confidence":  95.0,
                "handwritten": True,
            }

        except httpx.TimeoutException:
            print(f"   ⏱️  Gemini timeout (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(2)

        except Exception as e:
            if "429" in str(e):
                print(f"   ⚠️  Key exhausted (429) — rotating to next key...")
                # _get_next_gemini_key() already rotated, just retry
            else:
                print(f"   ❌ Gemini failed: {e}")
                raise

    raise RuntimeError("All Gemini keys exhausted or failed")
# ─────────────────────────────────────────────────────────────────────────────
# Smart OCR — main decision engine
# ─────────────────────────────────────────────────────────────────────────────

def ocr_smart(image: Image.Image, allow_gemini: bool = True) -> dict:
    """
    3-Level OCR Pipeline with Arabic/Language-aware routing.

    Tesseract can report high confidence (>90%) with ZERO text.
    We must check text length, not just confidence score.

    allow_gemini : cost guard flag. When False, Level 3 (Gemini Vision)
                   is skipped entirely regardless of quality, and the
                   pipeline falls back to the best local (Tesseract/EasyOCR)
                   result instead. Set by the caller based on a per-document
                   page budget (see GEMINI_PAGE_BUDGET).
    """
    processed   = preprocess_image(image)
    handwritten = is_likely_handwritten(processed)

    # ── Level 1 — Tesseract (always run for baseline) ───────────────────────
    tesseract_text, tesseract_conf = _run_tesseract(processed)
    tesseract_len = len(tesseract_text.strip()) if tesseract_text else 0

    print(f"   📊 Tesseract: confidence={tesseract_conf:.1f}%, text_length={tesseract_len} chars")

    # CRITICAL: High confidence with empty text = FAIL, not success
    # Tesseract reports 95% confidence when it sees a blank page!
    tesseract_has_content = tesseract_len > 10  # At least 10 chars to count as text

    # Detect if text contains Arabic characters
    arabic_ratio = sum(1 for c in tesseract_text if '\u0600' <= c <= '\u06FF') / max(tesseract_len, 1) if tesseract_has_content else 0
    has_arabic = arabic_ratio > 0.15

    print(f"   🔤 Arabic detected: {has_arabic} (ratio: {arabic_ratio:.1%}), has_content: {tesseract_has_content}")

    if not allow_gemini:
        print(f"   🚫 Gemini Vision disabled for this page (budget guard)")

    # ── Fast path: Non-Arabic, non-handwritten, high confidence, AND has content ──
    if (not handwritten and not has_arabic and
        tesseract_conf >= TESSERACT_THRESHOLD and tesseract_has_content):
        print(f"   ✅ Level 1 — Tesseract ({tesseract_conf:.1f}%, {tesseract_len} chars) [Non-Arabic printed]")
        return {
            "text":        tesseract_text,
            "engine":      "tesseract",
            "confidence":  tesseract_conf,
            "handwritten": False,
        }

    # ── Tesseract empty or Arabic/Handwritten → Force EasyOCR ────────────────
    if not tesseract_has_content:
        print(f"   ⚠️  Tesseract returned EMPTY text (confidence={tesseract_conf:.1f}%) — forcing EasyOCR")
    elif has_arabic:
        print(f"   🔤 Arabic text detected — forcing EasyOCR (Tesseract Arabic is unreliable)")
    elif handwritten:
        print("   ✍️  Handwriting detected — forcing EasyOCR")
    else:
        print(f"   ⚠️  Low confidence ({tesseract_conf:.1f}%) — trying EasyOCR...")

        # ── Level 2 — EasyOCR ────────────────────────────────────────────────────
    easyocr_text = ""
    easyocr_conf = 0.0

    try:
        easyocr_result = ocr_with_easyocr(image, has_arabic=has_arabic)
        easyocr_text   = easyocr_result["text"]
        easyocr_conf   = easyocr_result["confidence"]
        easyocr_len    = len(easyocr_text.strip()) if easyocr_text else 0

        print(f"   📊 EasyOCR: confidence={easyocr_conf:.1f}%, text_length={easyocr_len} chars")
        print(f"   📝 EasyOCR text preview: {easyocr_text[:100]}...")

        # If EasyOCR got text but confidence is low, try Gemini for better quality
        # (only if the budget guard allows it)
        if allow_gemini and easyocr_text.strip() and easyocr_len > 20 and easyocr_conf < 40:
            print(f"   ⚠️  EasyOCR confidence low ({easyocr_conf:.1f}%) — trying Gemini Vision...")
            try:
                gemini_result = _run_gemini_vision(image)
                gemini_text = gemini_result["text"]
                gemini_len = len(gemini_text.strip()) if gemini_text else 0

                # Use Gemini if it got reasonable text
                if gemini_len > easyocr_len * 0.5:  # At least 50% as much
                    print(f"   ✅ Level 3 — Gemini Vision (text length: {gemini_len})")
                    return {
                        "text": gemini_text,
                        "engine": "gemini_vision",
                        "confidence": 95.0,
                        "handwritten": handwritten,
                    }
            except Exception as e:
                print(f"   ⚠️  Gemini failed ({e}) — using EasyOCR anyway")

        # Use EasyOCR result
        if easyocr_text.strip() and easyocr_len > 20:
            print(f"   ✅ Level 2 — EasyOCR (text length: {easyocr_len} chars)")
            return {
                "text": easyocr_text,
                "engine": "easyocr",
                "confidence": easyocr_conf,
                "handwritten": handwritten,
            }

        print(f"   ⚠️  EasyOCR returned empty/short text...")

    except Exception as e:
        print(f"   ⚠️  EasyOCR failed ({e})...")

    # ── Level 3 — Gemini Vision (budget-gated) ───────────────────────────────
    if allow_gemini:
        try:
            gemini_result = _run_gemini_vision(image)
            gemini_text = gemini_result["text"]
            gemini_len = len(gemini_text.strip()) if gemini_text else 0

            print(f"   📊 Gemini: text_length={gemini_len} chars")
            print(f"   📝 Gemini text preview: {gemini_text[:100]}...")

            # Validate Gemini actually got text
            if gemini_len < 10:
                print(f"   ⚠️  Gemini returned empty/short text — falling back")
                raise ValueError("Gemini failed to extract meaningful text")

            print(f"   ✅ Level 3 — Gemini Vision (text length: {gemini_len})")
            return {
                "text":        gemini_text,
                "engine":      "gemini_vision",
                "confidence":  95.0,
                "handwritten": handwritten,
            }

        except Exception as e:
            print(f"   ⚠️  Gemini Vision unavailable ({e})")
            print(f"   🔁 Falling back to best local result...")
    else:
        print(f"   🔁 Skipping Gemini Vision (budget exhausted) — using best local result...")

    # ── Fallback: Use whatever has text ─────────────────────────────────────
    # Prefer EasyOCR if it got anything, else Tesseract (even if empty, return it)
    if easyocr_text and len(easyocr_text.strip()) > 0:
        print(f"   ✅ Using EasyOCR as fallback ({easyocr_conf:.1f}%)")
        return {
            "text":        easyocr_text,
            "engine":      "easyocr_fallback",
            "confidence":  easyocr_conf,
            "handwritten": handwritten,
        }
    else:
        print(f"   ✅ Using Tesseract as fallback (empty result)")
        return {
            "text":        tesseract_text or "",  # Ensure string, not None
            "engine":      "tesseract_fallback",
            "confidence":  tesseract_conf,
            "handwritten": handwritten,
        }
# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by document_service.py
# ─────────────────────────────────────────────────────────────────────────────

def ocr_image(path: str) -> dict:
    """
    OCR a single image file.
    Called by document_service._extract_image().
    Returns full result dict including metadata.

    Standalone images always get full Gemini budget (allow_gemini=True) —
    the cost-guard only matters for multi-page scanned PDFs where a single
    low-quality document could otherwise trigger dozens of Gemini calls.
    """
    image  = Image.open(path)
    result = ocr_smart(image, allow_gemini=True)

    return {
        "text": result["text"],
        "metadata": {
            "file_type":      "image",
            "file_name":      Path(path).name,
            "languages":      "fra+eng+ara",
            "ocr_engine":     result["engine"],
            "ocr_confidence": result["confidence"],
            "is_handwritten": result["handwritten"],
            "char_count":     len(result["text"]),
            "word_count":     len(result["text"].split()),
            "extracted_at":   datetime.now().isoformat(),
        },
    }


def ocr_image_file(image_path: str) -> str:
    """
    OCR a standalone image file.
    Returns plain text string (used by older callers).
    """
    return ocr_image(image_path)["text"]


def process_single_page(args: tuple) -> dict:
    """
    Processes one PDF page — converts to image then runs ocr_smart.
    Designed to run in parallel with ThreadPoolExecutor.

    args : (page_index, fitz_page, languages, allow_gemini)

    allow_gemini is computed upstream by ocr_scanned_pdf()/ocr_pdf_pages()
    based on the document-wide GEMINI_PAGE_BUDGET — only the first N pages
    (in processing order) are permitted to escalate to Level 3.

    Resolution Matrix(2,2) = ~144 DPI :
        Faster than Matrix(3,3) with minimal quality loss
        for standard printed/scanned documents.
    """
    page_num, page, languages, allow_gemini = args

    mat       = fitz.Matrix(2, 2)
    pix       = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    image     = Image.open(io.BytesIO(img_bytes))

    result = ocr_smart(image, allow_gemini=allow_gemini)

    return {
        "page_number":    page_num + 1,
        "text":           result["text"],
        "char_count":     len(result["text"]),
        "word_count":     len(result["text"].split()),
        "has_text":       len(result["text"]) > 50,
        "ocr_engine":     result["engine"],
        "ocr_confidence": result["confidence"],
        "is_handwritten": result["handwritten"],
        "gemini_allowed": allow_gemini,
    }


def ocr_pdf_pages(pdf_path: str, page_indices: list) -> str:
    """
    OCR only the specified pages (0-based indices) of a scanned PDF.
    Called by document_service for brief/summary/full modes.
    Returns concatenated text with [Page N] markers.
    Processes pages in parallel (4 workers).

    Gemini Vision (Level 3) is capped at GEMINI_PAGE_BUDGET pages for this
    call — since brief/summary modes already only request a handful of
    pages, this is mostly a safety net.
    """
    doc  = fitz.open(pdf_path)
    args = [
        (i, doc[i], "fra+eng+ara", idx < GEMINI_PAGE_BUDGET)
        for idx, i in enumerate(page_indices)
    ]
    results = [None] * len(page_indices)

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(process_single_page, arg): idx
            for idx, arg in enumerate(args)
        }
        for future in as_completed(future_to_index):
            idx          = future_to_index[future]
            results[idx] = future.result()

    doc.close()

    return "\n\n".join(
        f"[Page {r['page_number']}]\n{r['text']}"
        for r in results
        if r and r["has_text"]
    )


def ocr_scanned_pdf(path: str) -> dict:
    """
    OCR all pages of a scanned PDF using parallel processing.
    Used by extract_text_ocr() for full document extraction.

    Cost guard: only the first GEMINI_PAGE_BUDGET pages (in document order)
    are allowed to escalate to Gemini Vision (Level 3). Remaining pages
    that would otherwise need Gemini fall back to the best local OCR result
    (EasyOCR or Tesseract). This prevents a single large low-quality scanned
    PDF from generating dozens/hundreds of Gemini API calls.

    NOTE: this is a simple "first N pages" budget, not a "worst quality
    pages first" allocation. A smarter version would run Tesseract/EasyOCR
    across all pages first, rank which pages actually need Gemini, then
    allocate the budget to those — documented as a future improvement.
    """
    doc         = fitz.open(path)
    total_pages = len(doc)

    print(f"\n   ⚙️  Processing {total_pages} pages in parallel (4 workers)...")
    print(f"   💰 Gemini Vision budget: {GEMINI_PAGE_BUDGET} pages max for this document")

    args = [
        (i, doc[i], "fra+eng+ara", i < GEMINI_PAGE_BUDGET)
        for i in range(total_pages)
    ]
    pages = [None] * total_pages

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_index = {
            executor.submit(process_single_page, arg): idx
            for idx, arg in enumerate(args)
        }
        completed = 0
        for future in as_completed(future_to_index):
            idx        = future_to_index[future]
            pages[idx] = future.result()
            completed += 1
            if completed % 10 == 0 or completed == total_pages:
                print(f"   ✅ {completed}/{total_pages} pages processed...")

    doc.close()

    full_text = "\n\n".join(
        f"[Page {p['page_number']}]\n{p['text']}"
        for p in pages
        if p["has_text"]
    )

    gemini_calls_used = sum(
        1 for p in pages if p["ocr_engine"] == "gemini_vision"
    )

    print(f"   🎉 Done! {total_pages} pages processed. "
          f"Gemini Vision used on {gemini_calls_used}/{GEMINI_PAGE_BUDGET} budgeted pages.")

    return {
        "text": full_text,
        "metadata": {
            "file_type":          "scanned_pdf",
            "file_name":          Path(path).name,
            "page_count":         total_pages,
            "pages_with_text":    sum(1 for p in pages if p["has_text"]),
            "languages":          "fra+eng+ara",
            "pages":              pages,
            "char_count":         len(full_text),
            "word_count":         len(full_text.split()),
            "gemini_budget_limit": GEMINI_PAGE_BUDGET,
            "gemini_budget_used":  gemini_calls_used,
            "extracted_at":        datetime.now().isoformat(),
        },
    }


def extract_text_ocr(file_path: str) -> dict:
    """
    Main OCR entry point — routes by file type.
    """
    ext = Path(file_path).suffix.lower()

    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}:
        return ocr_image(file_path)
    elif ext == ".pdf":
        return ocr_scanned_pdf(file_path)
    else:
        raise ValueError(
            f"Unsupported format for OCR: '{ext}'. "
            f"Supported: jpg, jpeg, png, bmp, tiff, webp, gif, pdf"
        )


def is_scanned_pdf(path: str) -> bool:
    """
    Returns True if PDF needs OCR (scanned), False if native text.
    Checks first page: < 50 chars of text = scanned.
    """
    doc  = fitz.open(path)
    text = doc[0].get_text().strip()
    doc.close()
    return len(text) < 50