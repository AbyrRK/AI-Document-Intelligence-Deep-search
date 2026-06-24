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
from typing import Optional
from uuid import uuid4                          # ← NEW

import fitz
import openpyxl
from bs4 import BeautifulSoup
from docx import Document
from app.services.text_processor import process_text
from app.services.chunking_service import create_chunks
from app.services.vector_store import vector_store
from app.services.rag_service import rag_service

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
        for i in range(min(BRIEF_PAGES + 1, total_pages)):
            text = doc[i].get_text("text").strip()
            if len(text) > 100:
                return [i]
        return [0]

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

    # ── ALWAYS extract tables regardless of mode ──────────────────────────────
    table_text = ""
    for table in doc.tables:
        rows = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            # deduplicate merged cells (DOCX repeats merged cell content)
            seen_cells = []
            for cell in row_data:
                if not seen_cells or cell != seen_cells[-1]:
                    seen_cells.append(cell)
            row_str = " | ".join(c for c in seen_cells if c)
            if row_str.strip():
                rows.append(row_str)
        if rows:
            table_text += "\n" + "\n".join(rows)

    if table_text.strip():
        text = text + "\n\n" + table_text.strip()
        print(f"[DOCX] Table text appended: {len(table_text)} chars")  
    else:
        print(f"[DOCX] No table text found in {path.name}")  
    # ─────────────────────────────────────────────────────────────────────────

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
    all_sheet_names  = wb.sheetnames
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
        "pages_total":       len(all_sheet_names),
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
        if result.get("text", "").strip():
            parts.append(result["text"])
        if result.get("tables"):
            parts.append("\n\n--- EXTRACTED TABLES ---\n")
            for t in result["tables"]:
                parts.append(f"\n[Table {t['table_index']}]\n{t['markdown']}")

        return {
            "text":              "\n\n".join(parts),
            "pages_total":       1,
            "pages_read":        1,
            "pages_selected":    [1],
            "extraction_method": "+".join(result.get("engines_used", [])) or "none",
            "is_scanned":        True,
            "is_handwritten":    False,
            "ocr_confidence":    100,
            "tables":            result.get("tables", []),
            "tables_found":      result.get("tables_found", 0),
        }

    from app.services.ocr_service import ocr_image
    result = ocr_image(str(path))

    print(f"[DEBUG] Image OCR: engine={result['metadata'].get('ocr_engine', 'unknown')}, "
          f"confidence={result['metadata'].get('ocr_confidence', 0):.1f}%, "
          f"text_length={len(result.get('text', ''))} chars")

    return {
        "text":              result.get("text", ""),
        "pages_total":       1,
        "pages_read":        1,
        "pages_selected":    [1],
        "extraction_method": result.get("metadata", {}).get("ocr_engine", "unknown"),
        "is_scanned":        True,
        "is_handwritten":    result.get("metadata", {}).get("is_handwritten", False),
        "ocr_confidence":    result.get("metadata", {}).get("ocr_confidence", 0),
        "tables":            [],
        "tables_found":      0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_skeleton(file_path: str) -> dict:
    path   = Path(file_path)
    suffix = path.suffix.lower()
    titles = []

    if suffix == ".pdf":
        doc         = fitz.open(str(path))
        total_pages = len(doc)
        is_scanned  = _is_scanned_pdf(doc)

        if is_scanned:
            doc.close()
            return {
                "titles":      [],
                "total_pages": total_pages,
                "is_scanned":  True,
                "supported":   False,
                "note":        f"Scanned document ({total_pages} pages) — OCR running in background",
            }

        for page_num, page in enumerate(doc):
            blocks    = page.get_text("dict").get("blocks", [])
            all_sizes = [
                span["size"]
                for block in blocks if block.get("type") == 0
                for line in block.get("lines", [])
                for span in line.get("spans", [])
                if span.get("size", 0) > 0
            ]
            if not all_sizes:
                continue

            avg_size = sum(all_sizes) / len(all_sizes)
            seen_on_page = set()

            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        size = span.get("size", 0)
                        if (
                            size > avg_size * BIG_TITLE_RATIO
                            and size > BIG_TITLE_MIN_PT
                            and text
                            and text not in seen_on_page
                        ):
                            seen_on_page.add(text)
                            titles.append({
                                "page":      page_num + 1,
                                "title":     text,
                                "font_size": round(size, 1),
                            })
        doc.close()

        return {
            "titles":      titles,
            "total_pages": total_pages,
            "is_scanned":  False,
            "supported":   True,
            "note":        None,
        }

    elif suffix == ".docx":
        word_doc = Document(str(path))
        for para in word_doc.paragraphs:
            style_name = para.style.name if para.style and hasattr(para.style, "name") else ""
            text       = para.text.strip()
            if style_name.startswith("Heading") and text:
                level = style_name.replace("Heading", "").strip()
                titles.append({
                    "page":      1,
                    "title":     text,
                    "font_size": None,
                    "level":     level,
                })

        return {
            "titles":      titles,
            "total_pages": 1,
            "is_scanned":  False,
            "supported":   True,
            "note":        None,
        }

    else:
        return {
            "titles":      [],
            "total_pages": 1,
            "is_scanned":  suffix in IMAGE_EXTENSIONS,
            "supported":   False,
            "note":        "Structure detection not available for this format",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_document(
    file_path:  str,
    mode:       str  = "brief",
    index:      bool = False,
    session_id: str  = "",
    file_name:  str  = "",
    doc_id:     str  = "",          # ← NEW: pass in or leave blank to skip indexing with doc_id
) -> dict:
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

    if mode != "table":
        processed = process_text(text)
        text      = processed["clean_text"]
        language  = processed["language"]
    else:
        language = "unknown"

    ocr_confidence = result.get("ocr_confidence", 100)
    text_has_content = bool(text.strip()) and len(text.strip()) > 50

    if index and ocr_confidence < 40 and not text_has_content:
        print(f"⚠️  OCR confidence too low ({ocr_confidence}%) and text too short — skipping indexing")
        index = False
    elif index and ocr_confidence < 40 and text_has_content:
        print(f"⚠️  OCR confidence low ({ocr_confidence}%) but text has {len(text)} chars — indexing anyway")

    display_name = file_name or path.name

    chunks_created = 0
    if index:
        chunks = create_chunks(
            text=text,
            metadata={
                "file_name":         display_name,
                "doc_id":            doc_id,           # ← NEW
                "file_type":         suffix.lstrip("."),
                "language":          language,
                "extraction_method": result["extraction_method"],
                "session_id":        session_id,
                "mode":              mode,
                "created_at":        datetime.utcnow().isoformat(),
            }
        )
        vector_store.ensure_collection()
        chunks_created = vector_store.add_documents(chunks, session_id=session_id)
        print(f"📦 {chunks_created} chunks indexed in Qdrant (mode={mode}, doc_id={doc_id[:8]}...)")

    word_count      = _word_count(text)
    display_content = text if mode == "table" else text[:5000]
    elapsed_ms      = (time.perf_counter() - t0) * 1000

    return {
        "status":    "success",
        "file_name": display_name,
        "doc_id":    doc_id,          # ← NEW: returned so caller can store/forward it
        "file_type": suffix.lstrip("."),
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "mode": mode,
        "extraction": {
            "pages_total":       result["pages_total"],
            "pages_read":        result["pages_read"],
            "pages_selected":    result["pages_selected"],
            "word_count":        word_count,
            "char_count":        len(text),
            "extraction_method": result["extraction_method"],
            "is_scanned":        result["is_scanned"],
            "is_handwritten":    result.get("is_handwritten", False),
            "language":          language,
            "content":           display_content,
            "full_text":         text,
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
    metadata  = {
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


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-DOCUMENT & LAZY INDEXING
# ═════════════════════════════════════════════════════════════════════════════

from app.services.chat_memory import get_session_history, save_session, add_session_file, get_session_files

_deferred_chunks: dict[str, list] = {}


def process_and_respond(
    file_path:  str,
    prompt:     Optional[str] = None,
    session_id: str = "",
    mode:       str = "brief",
    index:      bool = True,
    lazy:       bool = True,
    file_name:  str = "",
    doc_id:     str = "",          # ← NEW: generated once by the route, passed all the way down
) -> dict:
    """
    Extract document and either answer immediately or return a brief.

    doc_id is generated ONCE in the upload route (uuid4), then passed here
    and forwarded into every chunk's metadata. This is the stable identifier
    for this document across all operations (search, delete, Phase 5 RAG agent).
    """
    # 1. Extract (never index here — we control it)
    result = process_document(
        file_path=file_path,
        mode=mode,
        index=False,
        session_id=session_id,
        file_name=file_name,
        doc_id=doc_id,             # ← NEW
    )

    text         = result.get("extraction", {}).get("full_text", "")
    scanned_note = None
    if result.get("extraction", {}).get("is_scanned"):
        scanned_note = (
            f"Scanned document ({result['extraction']['pages_total']} pages) — "
            "OCR running in background"
        )

    # 2. Track this file in session metadata
    display_name = file_name or result["file_name"]
    add_session_file(session_id, display_name)

    # 3. Build chunks — doc_id stamped on every chunk here
    chunks = create_chunks(
        text=text,
        metadata={
            "file_name":         display_name,
            "doc_id":            doc_id,        # ← NEW
            "file_type":         result["file_type"],
            "language":          result["extraction"]["language"],
            "extraction_method": result["extraction"]["extraction_method"],
            "session_id":        session_id,
            "mode":              mode,
            "created_at":        datetime.utcnow().isoformat(),
        }
    )

    # 4. Check if session already has indexed content
    already_indexed = False
    try:
        existing = vector_store.search_by_session("test", session_id=session_id, k=1)
        already_indexed = len(existing) > 0
    except Exception:
        pass

    # 5. Lazy path: defer indexing
    should_lazy = lazy and mode in ("brief", "summary") and not prompt and not already_indexed

    if should_lazy:
        if session_id in _deferred_chunks:
            _deferred_chunks[session_id].extend(chunks)
            print(f"⏳ Lazy mode: appended {len(chunks)} chunks (total: {len(_deferred_chunks[session_id])}) for session {session_id[:8]}")
        else:
            _deferred_chunks[session_id] = chunks
            print(f"⏳ Lazy mode: stored {len(chunks)} chunks for session {session_id[:8]}")

        all_chunks = _deferred_chunks[session_id]
        fallback = "\n\n".join(c["content"] for c in all_chunks[:8])
        brief = rag_service.generate_brief(session_id, fallback_text=fallback)

        return {
            "status":       "summarized",
            "summary":      brief,
            "message":      brief,
            "chunks_ready": len(all_chunks),
            "indexed":      False,
            "scanned_note": scanned_note,
            "file_name":    display_name,
            "doc_id":       doc_id,             # ← NEW
        }

    # 6. Eager path: index now
    chunks_created = 0
    if index:
        vector_store.ensure_collection()
        chunks_created = vector_store.add_documents(chunks, session_id=session_id)
        if session_id in _deferred_chunks:
            deferred = _deferred_chunks.pop(session_id)
            if deferred:
                deferred_count = vector_store.add_documents(deferred, session_id=session_id)
                chunks_created += deferred_count
                print(f"✅ Also indexed {deferred_count} deferred chunks for session {session_id[:8]}")
        print(f"✅ Indexed {chunks_created} chunks for session {session_id[:8]}, doc_id={doc_id[:8]}...")

    # 7. Prompt given → RAG answer
    if prompt and prompt.strip():
        rag_result = rag_service.ask(prompt, session_id=session_id)
        return {
            "status":         "answered",
            "session_id":     session_id,
            "doc_id":         doc_id,           # ← NEW
            "chunks_indexed": chunks_created,
            "answer":         rag_result["answer"],
            "sources":        rag_result["sources"],
            "scanned_note":   scanned_note,
            "file_name":      display_name,
        }

    # 8. No prompt → brief
    brief = rag_service.generate_brief(session_id)

    history = get_session_history(session_id)
    history.add_ai_message(brief)
    save_session(session_id)

    return {
        "status":         "summarized",
        "session_id":     session_id,
        "doc_id":         doc_id,               # ← NEW
        "chunks_indexed": chunks_created,
        "summary":        brief,
        "scanned_note":   scanned_note,
        "message":        brief,
        "file_name":      display_name,
    }


def ensure_indexed(session_id: str) -> bool:
    """Trigger indexing for a lazy session before the first question."""
    if session_id in _deferred_chunks:
        chunks = _deferred_chunks.pop(session_id)
        if not chunks:
            return False
        vector_store.ensure_collection()
        count = vector_store.add_documents(chunks, session_id=session_id)
        print(f"🚀 Lazy indexing triggered: {count} chunks for session {session_id[:8]}")
        return True

    try:
        existing = vector_store.search_by_session("test", session_id=session_id, k=1)
        return len(existing) > 0
    except Exception:
        return False