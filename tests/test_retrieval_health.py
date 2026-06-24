"""
test_retrieval_health.py
─────────────────────────────────────────────────────────────────────────────
Pre-flight check: before testing prompts, verify that Qdrant actually
returns chunks for the configured session.

Usage:
    pytest tests/test_retrieval_health.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.vector_store import vector_store


# ── CONFIG ──────────────────────────────────────────────────────────────────

SESSION_ID = os.environ.get("EVAL_SESSION_ID", "")
MIN_CHUNKS_REQUIRED = 1
TEST_QUERIES = [
    "What vector database is used?",
    "What are the processing time requirements?",
    "What OCR engines are used?",
]


# ── TESTS ───────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    SESSION_ID == "REPLACE_WITH_YOUR_SESSION_ID",
    reason="SESSION_ID not configured",
)
def test_session_id_configured():
    """Sanity: session_id has been set."""
    assert SESSION_ID != "REPLACE_WITH_YOUR_SESSION_ID"


@pytest.mark.skipif(
    SESSION_ID == "REPLACE_WITH_YOUR_SESSION_ID",
    reason="SESSION_ID not configured",
)
def test_retrieval_returns_chunks():
    """Core check: Qdrant returns at least one chunk for a known query."""
    all_ok = True
    for q in TEST_QUERIES:
        docs = vector_store.search_by_session(q, session_id=SESSION_ID, k=5)
        if len(docs) < MIN_CHUNKS_REQUIRED:
            pytest.fail(
                f"Query '{q}' returned {len(docs)} chunks (expected >= {MIN_CHUNKS_REQUIRED}). "
                f"Check that documents were uploaded and indexed for session {SESSION_ID}.")
        # Print first chunk preview for diagnostics
        print(f"\n  Query: {q}")
        print(f"  Retrieved {len(docs)} chunks")
        print(f"  Top chunk preview: {docs[0].page_content[:200]}...")