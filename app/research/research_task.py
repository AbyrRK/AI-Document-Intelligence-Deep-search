"""
app/research/research_task.py
=============================
Celery task for deep research — keeps same session ID.
"""

from pathlib import Path

from app.celery_app import celery
from app.research.graph import run_research_graph
from app.services.vector_store import vector_store
from app.services.chat_memory import (
    get_session_history,
    set_session_metadata,
    get_session_metadata,
    get_session_files,
)
from app.services.document_service import ensure_indexed
from app.core.language import resolve_language

import uuid
import os
from langchain_core.tracers.langchain import wait_for_all_tracers

# Force LangSmith init inside Celery worker
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
os.environ.setdefault("LANGCHAIN_PROJECT", "PFE2026")

def _clean_title(question: str, max_len: int = 45) -> str:
    """Create a clean sidebar title from user question."""
    clean = question.strip().rstrip('?.!').strip()
    if len(clean) > max_len:
        clean = clean[:max_len - 3].rsplit(' ', 1)[0] + '...'
    return clean


def _sanitize_session_id(session_id: str) -> tuple[str, bool]:
    """
    Sanitize session_id received from the client.
    Returns (clean_sid, is_new_session).
    """
    BAD_VALUES = {"null", "undefined", "none", ""}
    if not session_id or str(session_id).strip().lower() in BAD_VALUES:
        return f"research_{uuid.uuid4().hex[:12]}", True
    return session_id, False


@celery.task(bind=True)
def run_research_task(self, question: str, session_id: str,
                      language: str, session_has_documents: bool,
                      doc_id: str = "", doc_ids: list = None):
    """
    Run deep research. Keeps the same session ID if one exists.
    """

    effective_language = resolve_language(question, language)
    print(f"[Language] question='{question[:50]}...' detected='{effective_language}' (hint='{language}')")

    sid, is_new_session = _sanitize_session_id(session_id)
    print(f"[Session] received='{session_id}' → using='{sid}' new={is_new_session}")

    try:
        # ── 1. Setup session ──────────────────────────────────────────────────
        set_session_metadata(
            sid,
            title=_clean_title(question),
            session_type="research",
            research_question=question,
        )

        # ── 2. Save user question to history ─────────────────────────────────
        history = get_session_history(sid)
        history.add_user_message(question)

        # ── 3. Check documents server-side ───────────────────────────────────
        self.update_state(state="PROGRESS", meta={"status": "🔍 Checking documents..."})

        files = get_session_files(sid)
        print(f"[DEBUG] sid='{sid}'")
        print(f"[DEBUG] files={files}")

        session_has_documents = bool(files)
        print(f"[DEBUG] session_has_documents={session_has_documents}")

        if files:
            self.update_state(
                state="PROGRESS",
                meta={"status": f"📄 Indexing {len(files)} document(s)..."}
            )
            ensure_indexed(sid)

        # ── 4. Discover doc_id from session files ─────────────────────────────
        # get_session_files() may return either dicts or plain filename strings
        # depending on how the session was created — handle both.
        effective_doc_id = doc_id
        if not effective_doc_id and sid:
            for f in files:
                if isinstance(f, dict):
                    did = f.get("doc_id") or f.get("id") or f.get("document_id")
                    if did:
                        effective_doc_id = did
                        print(f"[ResearchTask] Discovered doc_id='{effective_doc_id}' from session files")
                        break
                # f is a plain filename string — no doc_id available, skip

        # ── 5. Run research graph ─────────────────────────────────────────────
        self.update_state(state="PROGRESS", meta={"status": "🔍 Analyzing question..."})

        result = run_research_graph(
            question=question,
            session_id=sid,
            language=effective_language,
            session_has_documents=session_has_documents,
            doc_id=effective_doc_id or "",
            doc_ids=doc_ids or ([effective_doc_id] if effective_doc_id else []),
        )

        # ── 6. Extract answer and sources ─────────────────────────────────────
        full_answer = result.get("validated_answer") or result.get("final_answer", "")
        sources     = result.get("sources") or result.get("cited_sources") or []

        # ── 7. Save assistant answer to history ───────────────────────────────
        history.add_ai_message(full_answer)

        # ── 8. Return ─────────────────────────────────────────────────────────
        return {
            "status":   "done",
            "answer":   full_answer,
            "sources":  sources,
            "session_id": sid,
            "language": effective_language,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        try:
            get_session_history(sid).add_ai_message(f"❌ Research failed: {str(e)}")
        except Exception:
            pass

        return {
            "status":   "error",
            "answer":   f"Research failed: {str(e)}",
            "sources":  [],
            "recommendations": result.get("recommendations", []) if 'result' in locals() else [],
            "session_id": sid,
            "language": effective_language,
        }