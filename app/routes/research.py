"""
app/routes/research.py
======================
Deep Research endpoint — Celery async tasks.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.research.research_task import run_research_task
from app.celery_app import celery
from app.services.document_service import ensure_indexed

router = APIRouter()

# ── Sentinel values that mean "no session" ────────────────────────────────────
_BAD_SESSION_VALUES = {"null", "undefined", "none", ""}


def _sanitize_session_id(sid: Optional[str]) -> Optional[str]:
    """Return None if the client sent a non-value, otherwise the clean string."""
    if not sid or str(sid).strip().lower() in _BAD_SESSION_VALUES:
        return None
    return sid.strip()


class ResearchRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    language: str = "en"
    session_has_documents: bool = False
    doc_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None


class SaveSourcesPayload(BaseModel):
    sources: list
    answer: str


@router.post("/query")
def start_research(payload: ResearchRequest):
    """Start research — returns task_id immediately."""

    # Sanitize on the route level before anything touches the value
    clean_session_id = _sanitize_session_id(payload.session_id)
    clean_doc_id     = _sanitize_session_id(payload.doc_id) or ""
    clean_doc_ids    = payload.doc_ids or []

    # Flush any lazy-deferred chunks BEFORE the research agents run
    if clean_session_id and payload.session_has_documents:
        flushed = ensure_indexed(clean_session_id)
        print(f"[Research] ensure_indexed for session {clean_session_id[:8]}: flushed={flushed}")

    task = run_research_task.delay(
        question=payload.question,
        session_id=clean_session_id,
        language=payload.language,
        session_has_documents=payload.session_has_documents,
        doc_id=clean_doc_id,
        doc_ids=clean_doc_ids,
    )
    return {"task_id": task.id, "status": "started"}




@router.get("/status/{task_id}")
def get_research_status(task_id: str):
    result = celery.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"status": "pending",  "answer": None, "sources": [], "session_id": None}

    if result.state == "PROGRESS":
        return {
            "status":     "running",
            "message":    result.info.get("status", "Processing..."),
            "answer":     None,
            "sources":    [],
            "session_id": None,
        }

    if result.state == "SUCCESS":
        return {"status": "done", **result.result}

    return {
        "status":     "error",
        "answer":     "An unexpected error occurred.",
        "sources":    [],
        "session_id": None,
    }


@router.post("/session/{session_id}/save-sources")
async def save_session_sources(session_id: str, payload: SaveSourcesPayload):
    """Save sources and answer to session metadata for persistence."""
    from app.services.chat_memory import set_session_metadata

    set_session_metadata(
        session_id=session_id,
        extra_metadata={
            "last_research_sources": payload.sources,
            "last_research_answer":  payload.answer,
        }
    )
    return {"saved": True}