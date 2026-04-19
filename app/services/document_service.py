"""
document_service.py
===================
Document ingestion pipeline — Phase 2 & 4

Supports : PDF, DOCX, XLSX, HTML, Images
Modes    : brief | summary | full | table
"""

import time
from datetime import datetime
from pathlib import Path

import fitz
import openpyxl
from bs4 import BeautifulSoup
from docx import Document
from app.services.text_processor import process_text
from app.services.chunking_service import create_chunks
from app.services.vector_store import vector_store

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB  = 50

SUPPORTED_FORMATS = {
    ".pdf", ".docx", ".xlsx", ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}

BRIEF_PAGES           = 1
BRIEF_DOCX_MAX_PARAS  = 2
BRIEF_DOCX_MAX_WORDS  = 150

SUMMARY_FIRST    = 5
SUMMARY_LAST     = 3
BIG_TITLE_RATIO  = 1.5
BIG_TITLE_MIN_PT = 14


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_file(file_path: str) -> Path:
    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ValueError("File is empty (0 bytes)")

    size_mb = size_bytes / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {size_mb:.1f} MB (limit {MAX_FILE_SIZE_MB} MB)")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_FORMATS)}"
        )

    return path


# ─────────────────────────────────────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def _is_title_page(text: str) -> bool:
    return _word_count(text.strip()) < 50


def _is_big_title_page(page) -> bool:
    blocks = page.get_text("dict")["blocks"]
    sizes  = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sz = span.get("size", 0)
                if sz > 0:
                    sizes.append(sz)

    if not sizes:
        return False

    avg    = sum(sizes) / len(sizes)
    max_sz = max(sizes)
    return max_sz > (avg * BIG_TITLE_RATIO) and max_sz > BIG_TITLE_MIN_PT


def _is_scanned_pdf(doc) -> bool:
    pages_to_check = min(3, len(doc))
    for i in range(pages_to_check):
        page   = doc[i]
        text   = page.get_text("text").strip()
        images = page.get_images(full=True)
        if len(text) < 50 and len(images) > 0:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Page selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_pages(doc, mode: str, total_pages: int, is_scanned: bool) -> list[int]:
    if mode in ("full", "table"):
        return list(range(total_pages))

    if mode == "brief":
        # Find the first page with real content (skip cover pages)
        for i in range(min(BRIEF_PAGES + 1, total_pages)):
            text = doc[i].get_text("text").strip()
            if len(text) > 100:
                return [i]
        return [0]  # Fallback to first page

    # summary mode
    selected = set()
    for i in range(min(SUMMARY_FIRST, total_pages)):
        selected.add(i)

    if not is_scanned:
        for i in range(total_pages):
            if _is_big_title_page(doc[i]):
                selected.add(i)

    for i in range(max(0, total_pages - SUMMARY_LAST), total_pages):
        selected.add(i)

    return sorted(selected)


# ─────────────────────────────────────────────────────────────────────────────
# DOCX helpers
# ─────────────────────────────────────────────────────────────────────────────

def _docx_select_brief(paragraphs: list) -> list:
    selected   = []
    word_total = 0
    for para in paragraphs[:3]:
        text = para.text.strip()
        if not text:
            continue
        selected.append(para)
        word_total += len(text.split())
        if word_total >= BRIEF_DOCX_MAX_WORDS:
            break
    return selected


def _docx_select_summary(paragraphs: list) -> list:
    selected_indices = set()
    count = 0
    for i, para in enumerate(paragraphs):
        if para.text.strip():
            selected_indices.add(i)
            count += 1
            if count >= SUMMARY_FIRST:
                break

    for i, para in enumerate(paragraphs):
        style_name = para.style.name if para.style and hasattr(para.style, "name") else ""
        if style_name.startswith("Heading") and para.text.strip():
            selected_indices.add(i)
            if i + 1 < len(paragraphs):
                selected_indices.add(i + 1)

    for i in range(max(0, len(paragraphs) - SUMMARY_LAST), len(paragraphs)):
        if paragraphs[i].text.strip():
            selected_indices.add(i)

    return [paragraphs[i] for i in sorted(selected_indices)]


# ─────────────────────────────────────────────────────────────────────────────
# Format extractors
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pdf(path: Path, mode: str) -> dict:
    doc          = fitz.open(str(path))
    total        = len(doc)
    is_scanned   = _is_scanned_pdf(doc)
    page_indices = _select_pages(doc, mode, total, is_scanned)

    if mode == "table":
        doc.close()
        from app.services.table_service import extract_tables
        result = extract_tables(str(path), page_indices, is_scanned)

        parts = []
        if result["text"].strip():
            parts.append(result["text"])
        if result["tables"]:
            parts.append("\n\n--- EXTRACTED TABLES ---\n")
            for t in result["tables"]:
                parts.append(
                    f"\n[Page {t['page']} — Table {t['table_index']}]\n"
                    f"{t['markdown']}"
                )

        return {
            "text":              "\n\n".join(parts),
            "pages_total":       total,
            "pages_read":        len(page_indices),
            "pages_selected":    [i + 1 for i in page_indices],
            "extraction_method": "+".join(result["engines_used"]) or "none",
            "is_scanned":        is_scanned,
            "tables":            result["tables"],
            "tables_found":      result["tables_found"],
        }

    if is_scanned:
        doc.close()
        from app.services.ocr_service import ocr_pdf_pages
        text   = ocr_pdf_pages(str(path), page_indices)
        method = "ocr"
    else:
        parts = [
            f"[Page {idx + 1}]\n{doc[idx].get_text('text')}"
            for idx in page_indices
        ]
        text   = "\n\n".join(parts)
        method = "native"
        doc.close()

    return {
        "text":              text,
        "pages_total":       total,
        "pages_read":        len(page_indices),
        "pages_selected":    [i + 1 for i in page_indices],
        "extraction_method": method,
        "is_scanned":        is_scanned,
        "tables":            [],
        "tables_found":      0,
    }


def _extract_docx(path: Path, mode: str) -> dict:
    doc   = Document(str(path))
    paras = [p for p in doc.paragraphs if p.text.strip()]

    if mode == "brief":
        selected = _docx_select_brief(paras)
    elif mode == "summary":
        selected = _docx_select_summary(paras)
    else:
        selected = paras

    text = "\n\n".join(p.text for p in selected)

    tables       = []
    tables_found = 0
    if mode == "table":
        from app.services.table_service import _table_to_markdown, _table_to_json
        for t_idx, table in enumerate(doc.tables):
            raw = [[cell.text for cell in row.cells] for row in table.rows]
            if not raw:
                continue
            tables.append({
                "page":        1,
                "table_index": t_idx + 1,
                "source":      "python-docx",
                "json":        _table_to_json(raw),
                "markdown":    _table_to_markdown(raw),
                "row_count":   len(raw) - 1,
                "col_count":   len(raw[0]) if raw else 0,
            })
        tables_found = len(tables)

        if tables:
            text += "\n\n--- EXTRACTED TABLES ---\n"
            for t in tables:
                text += f"\n[Table {t['table_index']}]\n{t['markdown']}\n"

    return {
        "text":              text,
        "pages_total":       len(paras),
        "pages_read":        len(selected),
        "pages_selected":    [],
        "extraction_method": "direct",
        "is_scanned":        False,
        "tables":            tables,
        "tables_found":      tables_found,
    }


def _extract_xlsx(path: Path, mode: str) -> dict:
    wb               = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    all_sheet_names  = wb.sheetnames          # ✅ Save before closing
    sheets           = all_sheet_names if mode == "full" else all_sheet_names[:1]
    row_limit        = {"brief": 20, "summary": 50}.get(mode)
    parts            = []

    for name in sheets:
        ws        = wb[name]
        rows_text = []
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if row_limit is not None and i >= row_limit:
                break
            row_str = "\t".join("" if c is None else str(c) for c in row)
            if row_str.strip():
                rows_text.append(row_str)
        parts.append(f"[Sheet: {name}]\n" + "\n".join(rows_text))

    wb.close()
    text = "\n\n".join(parts)

    return {
        "text":              text,
        "pages_total":       len(all_sheet_names),   # ✅ uses saved value
        "pages_read":        len(sheets),
        "pages_selected":    [],
        "extraction_method": "direct",
        "is_scanned":        False,
        "tables":            [],
        "tables_found":      0,
    }


def _extract_html(path: Path, mode: str) -> dict:
    raw   = path.read_text(encoding="utf-8", errors="replace")
    soup  = BeautifulSoup(raw, "html.parser")
    lines = [
        l for l in soup.get_text(separator="\n", strip=True).splitlines()
        if l.strip()
    ]
    if mode == "brief":
        lines = lines[:50]
    elif mode == "summary":
        lines = lines[:150]

    return {
        "text":              "\n".join(lines),
        "pages_total":       1,
        "pages_read":        1,
        "pages_selected":    [1],
        "extraction_method": "direct",
        "is_scanned":        False,
        "tables":            [],
        "tables_found":      0,
    }


def _extract_image(path: Path, mode: str) -> dict:
    if mode == "table":
        from app.services.table_service import extract_tables
        result = extract_tables(str(path), page_indices=[0], is_scanned=True)

        parts = []
        if result["text"].strip():
            parts.append(result["text"])
        if result["tables"]:
            parts.append("\n\n--- EXTRACTED TABLES ---\n")
            for t in result["tables"]:
                parts.append(f"\n[Table {t['table_index']}]\n{t['markdown']}")

        return {
            "text":              "\n\n".join(parts),
            "pages_total":       1,
            "pages_read":        1,
            "pages_selected":    [1],
            "extraction_method": "+".join(result["engines_used"]) or "none",
            "is_scanned":        True,
            "is_handwritten":    False,
            "tables":            result["tables"],
            "tables_found":      result["tables_found"],
        }

    from app.services.ocr_service import ocr_image
    result = ocr_image(str(path))

    return {
        "text":              result["text"],
        "pages_total":       1,
        "pages_read":        1,
        "pages_selected":    [1],
        "extraction_method": result["metadata"]["ocr_engine"],
        "is_scanned":        True,
        "is_handwritten":    result["metadata"].get("is_handwritten", False),
        "tables":            [],
        "tables_found":      0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_document(file_path: str, mode: str = "brief", index: bool = False) -> dict:
    """
    Parameters
    ----------
    mode : "brief" | "summary" | "full" | "table"
    index : bool
        If True, chunks and indexes the document into Qdrant.
    """
    if mode not in ("brief", "summary", "full", "table"):
        raise ValueError(f"Invalid mode '{mode}'. Choose: brief | summary | full | table")

    t0     = time.perf_counter()
    path   = validate_file(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        result = _extract_pdf(path, mode)
    elif suffix == ".docx":
        result = _extract_docx(path, mode)
    elif suffix == ".xlsx":
        result = _extract_xlsx(path, mode)
    elif suffix in {".html", ".htm"}:
        result = _extract_html(path, mode)
    elif suffix in IMAGE_EXTENSIONS:
        result = _extract_image(path, mode)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    text = result["text"]

    # ── Language detection + text cleaning ───────────────────────────────────
    if mode != "table":
        processed = process_text(text)
        text      = processed["clean_text"]
        language  = processed["language"]
    else:
        language = "unknown"

    # ── Chunking and indexing ────────────────────────────────────────────────
    chunks_created = 0
    if index:                                    # ✅ respect caller's intent only
        chunks = create_chunks(
            text=text,
            metadata={
                "file_name":         path.name,
                "file_type":         suffix.lstrip("."),
                "language":          language,
                "extraction_method": result["extraction_method"],
                "mode":              mode,
            }
        )
        vector_store.ensure_collection()
        chunks_created = vector_store.add_documents(chunks)
        print(f"📦 {chunks_created} chunks indexed in Qdrant")

    # ── Final response ───────────────────────────────────────────────────────
    word_count      = _word_count(text)          # ✅ before truncation
    display_content = text if mode == "table" else text[:5000]
    elapsed_ms      = (time.perf_counter() - t0) * 1000

    return {
        "status":    "success",
        "file_name": path.name,
        "file_type": suffix.lstrip("."),
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "mode": mode,
        "extraction": {
            "pages_total":       result["pages_total"],
            "pages_read":        result["pages_read"],
            "pages_selected":    result["pages_selected"],
            "word_count":        word_count,         # ✅ full text count
            "char_count":        len(text),
            "extraction_method": result["extraction_method"],
            "is_scanned":        result["is_scanned"],
            "is_handwritten":    result.get("is_handwritten", False),
            "language":          language,
            "content":           display_content,    # ✅ truncated only for display
            "tables":            result.get("tables", []),
            "tables_found":      result.get("tables_found", 0),
            "indexed":           chunks_created > 0,
            "chunks_created":    chunks_created,
            "extracted_at":      datetime.utcnow().isoformat(),
        },
        "processing_time_ms": round(elapsed_ms, 1),
    }

def ingest_document(extracted_text: str, file_metadata: dict) -> dict:
    if not extracted_text or not extracted_text.strip():
        return {
            "chunks_indexed": 0,
            "language":       "unknown",
            "word_count":     0,
            "reduction_pct":  0.0,
            "clean_length":   0,
        }

    processed = process_text(extracted_text)

    metadata = {
        **file_metadata,
        "language":   processed["language"],
        "word_count": processed["word_count"],
    }

    chunks = create_chunks(processed["clean_text"], metadata)
    count  = vector_store.add_documents(chunks)

    return {
        "chunks_indexed": count,
        "language":       processed["language"],
        "word_count":     processed["word_count"],
        "reduction_pct":  processed["reduction_pct"],
        "clean_length":   processed["clean_length"],
    }