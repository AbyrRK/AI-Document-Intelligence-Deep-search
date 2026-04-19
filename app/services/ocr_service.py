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

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

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
# Smart OCR — main decision engine
# ─────────────────────────────────────────────────────────────────────────────

def ocr_smart(image: Image.Image) -> dict:
    processed = preprocess_image(image)
    handwritten = is_likely_handwritten(processed)

    # Always run Tesseract once and cache the result
    tesseract_text, confidence = _run_tesseract(processed)
    print(f"   📊 Tesseract confidence: {confidence:.1f}%")

    if not handwritten and confidence >= TESSERACT_THRESHOLD:
        print(f"   ✅ Using Tesseract ({confidence:.1f}%)")
        return {
            "text":        tesseract_text,
            "engine":      "tesseract",
            "confidence":  confidence,
            "handwritten": False,
        }

    if handwritten:
        print("   ✍️  Handwriting detected — trying EasyOCR")
    else:
        print(f"   ⚠️  Low confidence ({confidence:.1f}%) — trying EasyOCR...")

    # Level 2 — EasyOCR
    try:
        easyocr_result = ocr_with_easyocr(image)
        easyocr_conf   = easyocr_result["confidence"]
        print(f"   📊 EasyOCR confidence: {easyocr_conf:.1f}%")

        if easyocr_conf >= EASYOCR_THRESHOLD:
            print(f"   ✅ Using EasyOCR ({easyocr_conf:.1f}%)")
            return {
                "text":        easyocr_result["text"],
                "engine":      "easyocr",
                "confidence":  easyocr_conf,
                "handwritten": handwritten,
            }
        print(f"   ⚠️  EasyOCR too low — using cached Tesseract result")

    except Exception as e:
        print(f"   ⚠️  EasyOCR failed ({e}) — using cached Tesseract result")

    # Fallback — reuse the ALREADY computed Tesseract result, no second call
    return {
        "text":        tesseract_text,
        "engine":      "tesseract_fallback",
        "confidence":  confidence,
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