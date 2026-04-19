"""
app/services/table_service.py
==============================
Table extraction pipeline for invoices, contracts, and any tabular document.

Strategy (same philosophy as OCR service):
    Native PDF  → pdfplumber  (fast, accurate, no dependencies)
         ↓ no tables found
    Scanned PDF / Image → img2table + Tesseract
         ↓ still no tables
    Gemini Vision fallback (handles borderless / complex tables)

Output per table:
    - JSON  : structured rows/columns for RAG pipeline
    - Markdown : human-readable for display
"""

import io
import json
import tempfile
from pathlib import Path
from typing import Optional

import fitz                         # PyMuPDF — PDF → image rasterization
import pdfplumber
from PIL import Image

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _page_to_image(pdf_path: str, page_index: int, dpi: int = 200) -> Image.Image:
    """Render one PDF page to a PIL Image."""
    doc  = fitz.open(pdf_path)
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _table_to_markdown(table: list[list]) -> str:
    """Convert a list-of-rows table into a Markdown table string."""
    if not table:
        return ""

    # Clean None values
    cleaned = [
        [str(cell) if cell is not None else "" for cell in row]
        for row in table
    ]

    header    = cleaned[0]
    separator = ["---"] * len(header)
    rows      = cleaned[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in rows:
        # Pad row if it has fewer columns than header
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def _table_to_json(table: list[list]) -> list[dict]:
    """
    Convert a list-of-rows into a list of dicts using the first row as keys.
    [ {"col1": "val", "col2": "val"}, ... ]
    """
    if not table or len(table) < 2:
        return []

    headers = [str(h) if h is not None else f"col_{i}"
               for i, h in enumerate(table[0])]
    result  = []
    for row in table[1:]:
        row_dict = {}
        for i, header in enumerate(headers):
            cell = row[i] if i < len(row) else None
            row_dict[header] = str(cell) if cell is not None else ""
        result.append(row_dict)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — pdfplumber (native PDFs only)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_pdfplumber(pdf_path: str, page_indices: list[int]) -> list[dict]:
    """
    Extract tables from native PDF pages using pdfplumber.
    Returns list of table dicts with page info, JSON and Markdown.
    """
    tables_found = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx in page_indices:
            if idx >= len(pdf.pages):
                continue
            page        = pdf.pages[idx]
            page_tables = page.extract_tables()

            for t_idx, raw_table in enumerate(page_tables):
                if not raw_table:
                    continue
                tables_found.append({
                    "page":     idx + 1,
                    "table_index": t_idx + 1,
                    "source":   "pdfplumber",
                    "json":     _table_to_json(raw_table),
                    "markdown": _table_to_markdown(raw_table),
                    "row_count": len(raw_table) - 1,          # minus header
                    "col_count": len(raw_table[0]) if raw_table else 0,
                })

    return tables_found


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — img2table (scanned PDFs and images)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_img2table_image(image: Image.Image, page_num: int) -> list[dict]:
    """
    Extract tables from a PIL Image using img2table + Tesseract.
    """
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR

    # img2table needs the image saved to a temp file or passed as bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    ocr     = TesseractOCR(lang="fra+eng+ara")
    doc     = Img2TableImage(src=buf)
    # implicit_rows=True helps detect borderless tables
    extracted = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        borderless_tables=True,
        min_confidence=50,
    )

    tables_found = []
    for t_idx, table in enumerate(extracted):
        raw = table.df.values.tolist()
        # Prepend column names as first row
        cols     = list(table.df.columns)
        raw_full = [cols] + raw

        tables_found.append({
            "page":        page_num,
            "table_index": t_idx + 1,
            "source":      "img2table",
            "json":        _table_to_json(raw_full),
            "markdown":    _table_to_markdown(raw_full),
            "row_count":   len(raw),
            "col_count":   len(cols),
        })

    return tables_found


def _extract_tables_img2table_pdf(pdf_path: str, page_indices: list[int]) -> list[dict]:
    """
    Rasterize each selected page and run img2table on it.
    Used for scanned PDFs.
    """
    tables_found = []
    for idx in page_indices:
        image  = _page_to_image(pdf_path, idx, dpi=200)
        tables = _extract_tables_img2table_image(image, page_num=idx + 1)
        tables_found.extend(tables)
    return tables_found


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — Gemini Vision fallback
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_gemini(image: Image.Image, page_num: int) -> list[dict]:
    """
    Ask Gemini Vision to extract tables as JSON.
    Used when pdfplumber and img2table find nothing.
    """
    import base64
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    prompt = """
    Look at this document image and extract ALL tables you can see.
    For each table, return a JSON array of objects where keys are column headers.
    
    Return ONLY a valid JSON array like this:
    [
      {
        "table_index": 1,
        "data": [
          {"col1": "val", "col2": "val"},
          {"col1": "val", "col2": "val"}
        ]
      }
    ]
    
    If no tables are found, return an empty array: []
    Return ONLY the JSON, no explanation.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        ]
    )

    raw_text = response.text.strip()
    # Strip markdown fences if Gemini wraps in ```json
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    tables_found = []
    for item in parsed:
        data = item.get("data", [])
        if not data:
            continue

        # Convert back to row format for markdown
        headers  = list(data[0].keys())
        rows     = [[row.get(h, "") for h in headers] for row in data]
        raw_full = [headers] + rows

        tables_found.append({
            "page":        page_num,
            "table_index": item.get("table_index", len(tables_found) + 1),
            "source":      "gemini",
            "json":        data,
            "markdown":    _table_to_markdown(raw_full),
            "row_count":   len(rows),
            "col_count":   len(headers),
        })

    return tables_found


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction alongside tables
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_for_pages(pdf_path: str, page_indices: list[int], is_scanned: bool) -> str:
    """
    Extract surrounding text (non-table content) for context.
    Native → pdfplumber text. Scanned → reuse OCR service.
    """
    if is_scanned:
        from app.services.ocr_service import ocr_pdf_pages
        return ocr_pdf_pages(pdf_path, page_indices)

    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx in page_indices:
            if idx >= len(pdf.pages):
                continue
            text = pdf.pages[idx].extract_text() or ""
            if text.strip():
                parts.append(f"[Page {idx + 1}]\n{text.strip()}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — called by document_service.py
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables(
    file_path: str,
    page_indices: list[int],
    is_scanned: bool,
) -> dict:
    """
    Extract tables + surrounding text from a PDF or image.

    Parameters
    ----------
    file_path    : path to the file
    page_indices : 0-based list of pages to process (from _select_pages)
    is_scanned   : True → use img2table, False → try pdfplumber first

    Returns
    -------
    dict with:
        tables       : list of table dicts (page, json, markdown, row/col counts)
        text         : surrounding text content
        tables_found : total number of tables detected
        engines_used : which extraction engines were invoked
    """
    suffix       = Path(file_path).suffix.lower()
    is_image     = suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    tables_found = []
    engines_used = []

    # ── Image file ────────────────────────────────────────────────────────────
    if is_image:
        image  = Image.open(file_path).convert("RGB")

        # img2table first
        tables_found = _extract_tables_img2table_image(image, page_num=1)
        engines_used.append("img2table")

        # Gemini fallback if nothing found
        if not tables_found and settings.GEMINI_API_KEY:
            tables_found = _extract_tables_gemini(image, page_num=1)
            engines_used.append("gemini")

        # Text via OCR
        from app.services.ocr_service import ocr_image_file
        text = ocr_image_file(file_path)

    # ── Native PDF ────────────────────────────────────────────────────────────
    elif not is_scanned:
        # Level 1 — pdfplumber
        tables_found = _extract_tables_pdfplumber(file_path, page_indices)
        engines_used.append("pdfplumber")

        # Level 2 — img2table if pdfplumber found nothing
        if not tables_found:
            tables_found = _extract_tables_img2table_pdf(file_path, page_indices)
            engines_used.append("img2table")

        # Level 3 — Gemini fallback
        if not tables_found and settings.GEMINI_API_KEY:
            for idx in page_indices:
                image  = _page_to_image(file_path, idx, dpi=200)
                tables = _extract_tables_gemini(image, page_num=idx + 1)
                tables_found.extend(tables)
            if tables_found:
                engines_used.append("gemini")

        # Surrounding text
        text = _extract_text_for_pages(file_path, page_indices, is_scanned=False)

    # ── Scanned PDF ───────────────────────────────────────────────────────────
    else:
        # img2table directly (already works on images)
        tables_found = _extract_tables_img2table_pdf(file_path, page_indices)
        engines_used.append("img2table")

        # Gemini fallback
        if not tables_found and settings.GEMINI_API_KEY:
            for idx in page_indices:
                image  = _page_to_image(file_path, idx, dpi=200)
                tables = _extract_tables_gemini(image, page_num=idx + 1)
                tables_found.extend(tables)
            if tables_found:
                engines_used.append("gemini")

        # Surrounding text via OCR
        text = _extract_text_for_pages(file_path, page_indices, is_scanned=True)

    return {
        "tables":        tables_found,
        "text":          text,
        "tables_found":  len(tables_found),
        "engines_used":  engines_used,
    }