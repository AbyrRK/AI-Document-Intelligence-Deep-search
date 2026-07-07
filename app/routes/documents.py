"""
routes/documents.py
===================
POST /documents/upload  — upload & extract / summarize / answer
GET  /documents/session/{session_id}/files — list files in session
GET  /documents/history/{session_id} — get session messages + metadata
"""

import shutil
import tempfile
import traceback
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from app.services import vector_store
from app.services.document_service import (
    extract_skeleton,
    process_and_respond,
    process_document,
    ensure_indexed,
)
from app.services.chat_memory import (
    set_session_metadata,
    get_session_metadata,      # ← ADD THIS IMPORT
    list_sessions,
    get_session_messages,
    get_session_files,
    add_session_file,          # ← ADD THIS IMPORT (was missing!)
    delete_session,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".xlsx",
    ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}


# ─────────────────────────────────────────────────────────────────────────────
# Background task
# ─────────────────────────────────────────────────────────────────────────────

import time

def _background_full_index(
    tmp_path:      str,
    session_id:    str,
    doc_id:        str,
    original_name: str = "",
) -> None:
    try:
        # First: flush any deferred chunks from the lazy path
        from app.services.document_service import _deferred_chunks
        from app.services.vector_store import vector_store as vs

        if session_id in _deferred_chunks:
            deferred = _deferred_chunks.pop(session_id)
            if deferred:
                print(f"🚀 Flushing {len(deferred)} deferred chunks for session {session_id[:8]}...")
                # Retry loop for embedding
                for attempt in range(4):
                    try:
                        vs.ensure_collection()
                        count = vs.add_documents(deferred, session_id=session_id)
                        print(f"✅ Deferred indexing complete: {count} chunks for session {session_id[:8]}")
                        break
                    except Exception as e:
                        wait = 2 ** attempt
                        print(f"⚠️  Embedding attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                        time.sleep(wait)
                else:
                    print(f"❌ All embedding attempts failed for session {session_id[:8]}")
                    return

        # Then: index remaining pages not covered by brief mode
        # (only if file still exists)
        if Path(tmp_path).exists():
            process_document(
                file_path=tmp_path,
                mode="full",
                index=True,
                session_id=session_id,
                file_name=original_name,
                doc_id=doc_id,
            )
            print(f"✅ Full indexing complete — session: {session_id[:8]}, doc_id: {doc_id[:8]}")

    except Exception as e:
        print(f"❌ Background indexing failed — session: {session_id[:8]} — {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Upload route
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file:       UploadFile    = File(...),
    mode:       str           = Form("brief"),
    index:      str           = Form("true"),
    prompt:     str | None    = Form(None),
    session_id: str | None    = Form(None),
    source:     Optional[str] = Form(None),
):
    # ── 1. Sanitize session_id ────────────────────────────────────────────────
    if not session_id or session_id.strip().lower() in ("null", "undefined", "none", ""):
        session_id = str(uuid4())
        is_new_session = True
    else:
        is_new_session = False

    # ── 2. Validate mode and file type ────────────────────────────────────────
    if mode not in ("brief", "summary", "full", "table"):
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: '{suffix}'")

    should_index = index.lower() in ("true", "1", "yes")
    clean_prompt = prompt.strip() if prompt and prompt.strip() else None

    # ── 3. Decide lazy vs eager BEFORE anything runs ─────────────────────────
    force_eager = (source == "deep_research") or bool(clean_prompt)

    # ── 4. Save upload to disk ────────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    background_queued = False

    try:
        doc_id = str(uuid4())

        # ── 5. Skeleton ───────────────────────────────────────────────────────
        skeleton = extract_skeleton(tmp_path)

        # ── 6. Extract + respond (lazy=False when deep research or has prompt) ─
        result = process_and_respond(
            file_path=tmp_path,
            prompt=clean_prompt,
            session_id=session_id,
            mode="brief",
            index=should_index,
            lazy=not force_eager,   # ← THE FIX: eager for deep_research/prompt
            file_name=file.filename,
            doc_id=doc_id,
        )

        # ── 7. Full indexing in background (skip if already eagerly indexed) ──
        if not force_eager:
            background_tasks.add_task(
                _background_full_index,
                tmp_path,
                session_id,
                doc_id,
                file.filename,
            )
            background_queued = True
        else:
            # eager path already indexed — still run full index for completeness
            background_tasks.add_task(
                _background_full_index,
                tmp_path,
                session_id,
                doc_id,
                file.filename,
            )
            background_queued = True

        # ── 8. Session metadata ───────────────────────────────────────────────
        add_session_file(session_id, file.filename)
        existing_files = get_session_files(session_id)
        set_session_metadata(
            session_id=session_id,
            file_name=file.filename,
            title=file.filename if len(existing_files) <= 1
                  else f"{len(existing_files)} documents",
        )

        result["file_name"]  = file.filename
        result["session_id"] = session_id
        result["doc_id"]     = doc_id
        result["structure"]  = skeleton

        if skeleton.get("is_scanned"):
            result["scanned_note"] = skeleton["note"]

        return JSONResponse(content=result)

    except Exception as e:
        traceback.print_exc()
        status = 422 if isinstance(e, ValueError) else 500
        raise HTTPException(status_code=status, detail=str(e))

    finally:
        if not background_queued:
            Path(tmp_path).unlink(missing_ok=True)
# ── List files in a session ─────────────────────────────────────────────────

@router.get("/session/{session_id}/files")
def get_session_files_list(session_id: str):
    files = get_session_files(session_id)
    return {"session_id": session_id, "files": files, "count": len(files)}


# ── List documents with their doc_ids for a session ─────────────────────────

@router.get("/session/{session_id}/docs")
def get_session_docs(session_id: str):
    """Returns [{doc_id, file_name}] — useful for the frontend to show a document picker."""
    from app.services.vector_store import vector_store as vs
    docs = vs.list_docs_by_session(session_id)
    return {"session_id": session_id, "docs": docs, "count": len(docs)}


# ── Delete a single document by doc_id ──────────────────────────────────────

@router.delete("/doc/{doc_id}")
def delete_document(doc_id: str):
    """Delete all chunks for one document. Reliable — uses doc_id, not filename."""
    from app.services.vector_store import vector_store as vs
    deleted = vs.delete_by_doc(doc_id)
    return {"deleted_chunks": deleted, "doc_id": doc_id}


# ═════════════════════════════════════════════════════════════════════════════
# HISTORY ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/history")
def get_history():
    return {"sessions": list_sessions()}


# ── FIX: Return metadata along with messages ───────────────────────────────

@router.get("/history/{session_id}")
def get_session(session_id: str):
    msgs = get_session_messages(session_id)
    if not msgs:
        # Still return metadata even if no messages (e.g., just uploaded doc)
        pass  # Don't 404 — let metadata through
    
    meta = get_session_metadata(session_id)  # ← GET METADATA
    
    return {
        "session_id": session_id,
        "messages": msgs,
        "metadata": meta,                      # ← RETURN METADATA
        "session_type": meta.get("session_type", "document"),
    }


@router.delete("/history/{session_id}")
def delete_history_session(session_id: str):
    """Delete a chat session and its vectors from Qdrant."""
    deleted = delete_session(session_id)

    try:
        vector_store.client.delete(
            collection_name=vector_store.collection_name,
            points_selector=Filter(
                must=[FieldCondition(
                    key="metadata.session_id",
                    match=MatchValue(value=session_id)
                )]
            )
        )
    except Exception as e:
        print(f"Warning: Could not delete vectors for {session_id}: {e}")

    return {"deleted": deleted, "session_id": session_id}