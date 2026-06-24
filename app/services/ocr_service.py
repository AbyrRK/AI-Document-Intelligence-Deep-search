"""
app/services/ocr_service.py
============================
OCR SERVICE — 3-Level Pipeline (No Gemini)
--------------------------------------------
Level 0 : Handwriting detection (OpenCV heuristic)
Level 1 : Tesseract  — fast, free, best for printed text
Level 2 : EasyOCR    — local, free, handles handwriting + Arabic
Fallback : Tesseract result if everything fails

Performance improvement (parallel processing) :
    Before : 57 pages in 6 minutes
    After  : 57 pages in ~2.5 minutes (4 workers, Matrix 2x2)
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


# ─────────────────────────────────────────────────────────────────────────────
# EasyOCR — lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

_easyocr_reader = None

def _get_easyocr_reader() -> easyocr.Reader:
    """
    Loads EasyOCR once and reuses it for every call.
    First load takes ~5-10 seconds (downloads models).
    Arabic is only compatible with English in EasyOCR,
    not French — this is an EasyOCR limitation.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        print("   🔄 Loading EasyOCR model (first time only)...")
        _easyocr_reader = easyocr.Reader(["ar", "en"], gpu=False)
        print("   ✅ EasyOCR ready")
    return _easyocr_reader


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

def ocr_with_easyocr(image: Image.Image) -> dict:
    reader = _get_easyocr_reader()
    
    # Resize large images — EasyOCR doesn't need full resolution
    max_dim = 1600
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"   📐 Resized to {new_size} for EasyOCR")

    img_array = np.array(image.convert("RGB"))
    results   = reader.readtext(img_array, detail=1)
    ...
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
    """
    Level 3 — Gemini Vision API.
    Called only when Tesseract < 60% AND EasyOCR < 40%.
    Best for: Arabic handwriting, degraded scans, mixed scripts.
    Requires GEMINI_API_KEY in .env.
    """
    import base64
    import httpx
    from app.core.config import settings

    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set — cannot use Gemini Vision")

    # Convert PIL image to base64 PNG
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={settings.GEMINI_API_KEY}"
    )

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

    response = httpx.post(url, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

    return {
        "text":        text,
        "engine":      "gemini_vision",
        "confidence":  95.0,
        "handwritten": True,
    }
# ─────────────────────────────────────────────────────────────────────────────
# Smart OCR — main decision engine
# ─────────────────────────────────────────────────────────────────────────────

def ocr_smart(image: Image.Image) -> dict:
    """
    3-Level OCR Pipeline with Arabic/Language-aware routing.

    Tesseract can report high confidence (>90%) with ZERO text.
    We must check text length, not just confidence score.
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
    has_arabic = tesseract_has_content and any('\u0600' <= c <= '\u06FF' for c in tesseract_text)
    arabic_ratio = sum(1 for c in tesseract_text if '\u0600' <= c <= '\u06FF') / max(tesseract_len, 1) if tesseract_has_content else 0

    print(f"   🔤 Arabic detected: {has_arabic} (ratio: {arabic_ratio:.1%}), has_content: {tesseract_has_content}")

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
        easyocr_result = ocr_with_easyocr(image)
        easyocr_text   = easyocr_result["text"]
        easyocr_conf   = easyocr_result["confidence"]
        easyocr_len    = len(easyocr_text.strip()) if easyocr_text else 0
        
        print(f"   📊 EasyOCR: confidence={easyocr_conf:.1f}%, text_length={easyocr_len} chars")
        print(f"   📝 EasyOCR text preview: {easyocr_text[:100]}...")

        # NEW: If EasyOCR got text but confidence is low, try Gemini for better quality
        if easyocr_text.strip() and easyocr_len > 20 and easyocr_conf < 40:
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
        
        print(f"   ⚠️  EasyOCR returned empty/short text — trying Gemini Vision...")

    except Exception as e:
        print(f"   ⚠️  EasyOCR failed ({e}) — trying Gemini Vision...")

    # ── Level 3 — Gemini Vision ──────────────────────────────────────────────
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
    """
    image  = Image.open(path)
    result = ocr_smart(image)

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

    args : (page_index, fitz_page, languages)

    Resolution Matrix(2,2) = ~144 DPI :
        Faster than Matrix(3,3) with minimal quality loss
        for standard printed/scanned documents.
    """
    page_num, page, languages = args

    mat       = fitz.Matrix(2, 2)
    pix       = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    image     = Image.open(io.BytesIO(img_bytes))

    result = ocr_smart(image)

    return {
        "page_number":    page_num + 1,
        "text":           result["text"],
        "char_count":     len(result["text"]),
        "word_count":     len(result["text"].split()),
        "has_text":       len(result["text"]) > 50,
        "ocr_engine":     result["engine"],
        "ocr_confidence": result["confidence"],
        "is_handwritten": result["handwritten"],
    }


def ocr_pdf_pages(pdf_path: str, page_indices: list) -> str:
    """
    OCR only the specified pages (0-based indices) of a scanned PDF.
    Called by document_service for brief/summary/full modes.
    Returns concatenated text with [Page N] markers.
    Processes pages in parallel (4 workers).
    """
    doc     = fitz.open(pdf_path)
    args    = [(i, doc[i], "fra+eng+ara") for i in page_indices]
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
    """
    doc         = fitz.open(path)
    total_pages = len(doc)

    print(f"\n   ⚙️  Processing {total_pages} pages in parallel (4 workers)...")

    args  = [(i, doc[i], "fra+eng+ara") for i in range(total_pages)]
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

    print(f"   🎉 Done! {total_pages} pages processed.")

    return {
        "text": full_text,
        "metadata": {
            "file_type":       "scanned_pdf",
            "file_name":       Path(path).name,
            "page_count":      total_pages,
            "pages_with_text": sum(1 for p in pages if p["has_text"]),
            "languages":       "fra+eng+ara",
            "pages":           pages,
            "char_count":      len(full_text),
            "word_count":      len(full_text.split()),
            "extracted_at":    datetime.now().isoformat(),
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