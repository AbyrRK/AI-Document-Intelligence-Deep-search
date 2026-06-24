from app.research.state import ResearchState
from app.services.vector_store import vector_store
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document

# Documents with this many chunks or fewer use high-k semantic search
# instead of raw scroll — ensures proper cosine scoring for all chunks
_SMALL_DOC_THRESHOLD = 10


def _count_chunks(doc_id: str = "", session_id: str = "") -> int:
    """Count total chunks for a doc or session — fast, no vectors fetched."""
    try:
        if doc_id:
            filt = Filter(must=[FieldCondition(
                key="metadata.doc_id", match=MatchValue(value=doc_id)
            )])
        elif session_id:
            filt = Filter(must=[FieldCondition(
                key="metadata.session_id", match=MatchValue(value=session_id)
            )])
        else:
            return 0

        points, _ = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=filt,
            limit=500,
            with_payload=False,
            with_vectors=False,
        )
        return len(points)
    except Exception as e:
        print(f"📄 RAG: chunk count failed: {e}")
        return 999  # assume large — fall back to normal path


def _get_tail_chunks(doc_id: str = "", session_id: str = "", n: int = 5) -> list:
    """
    Fetch the last N chunks by chunk_index — totals/signatures live at the bottom.
    Used for large docs only to catch summary rows that semantic search misses.
    Returns (Document, point_id) tuples.
    """
    try:
        if doc_id:
            filt = Filter(must=[FieldCondition(
                key="metadata.doc_id", match=MatchValue(value=doc_id)
            )])
        elif session_id:
            filt = Filter(must=[FieldCondition(
                key="metadata.session_id", match=MatchValue(value=session_id)
            )])
        else:
            return []

        all_points, _ = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=filt,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        sorted_points = sorted(
            all_points,
            key=lambda p: p.payload.get("metadata", {}).get("chunk_index", 0)
        )

        tail = sorted_points[-n:] if len(sorted_points) >= n else sorted_points

        return [
            (
                Document(
                    page_content=p.payload.get("page_content", ""),
                    metadata=p.payload.get("metadata", {})
                ),
                str(p.id)
            )
            for p in tail
        ]
    except Exception as e:
        print(f"📄 RAG: tail fetch failed: {e}")
        return []


def _wrap_search_results(docs: list) -> list:
    """
    Wrap Document objects from semantic search into (Document, dedup_key) tuples.
    Uses content[:120] as dedup key since semantic search doesn't return point IDs.
    """
    return [(doc, doc.page_content[:120].strip()) for doc in docs]


def rag_node(state: ResearchState) -> dict:
    session_id = state.get("session_id", "")
    doc_id     = state.get("doc_id", "")
    query      = state.get("reformulated_query") or state["question"]

    print(f"📄 RAG: query='{query[:50]}...' | session={session_id[:8] if session_id else 'NONE'} | doc_id={doc_id[:8] if doc_id else 'NONE'}")

    if not session_id and not doc_id:
        print("📄 RAG: ⚠️  No session_id or doc_id — skipping document search")
        return {"rag_results": [], "rag_confidence": 0.0}

    # ── Count chunks to choose retrieval strategy ─────────────────────────────
    total_chunks = _count_chunks(
        doc_id=doc_id,
        session_id=session_id if not doc_id else ""
    )

    if total_chunks <= _SMALL_DOC_THRESHOLD:
        # ── Small doc: semantic search with k=total_chunks ────────────────────
        # This gets ALL chunks back, properly scored by cosine similarity.
        # No raw scroll — every chunk gets a real relevance score.
        print(f"📄 RAG: small doc ({total_chunks} chunks) → high-k semantic search")
        if doc_id:
            raw = vector_store.search_by_doc(query, doc_id=doc_id, k=total_chunks)
            print(f"📄 RAG: used doc_id scope → {len(raw)} chunks")
            if not raw:
                raw = vector_store.search_by_session(query, session_id=session_id, k=total_chunks)
                print(f"📄 RAG: fallback session scope → {len(raw)} chunks")
        else:
            raw = vector_store.search_by_session(query, session_id=session_id, k=total_chunks)
            print(f"📄 RAG: used session scope → {len(raw)} chunks")

        paired      = _wrap_search_results(raw)
        tail_paired = []  # not needed — semantic search already covers all chunks

    else:
        # ── Large doc: semantic search + tail chunks ──────────────────────────
        if doc_id:
            raw = vector_store.search_by_doc(query, doc_id=doc_id, k=15)
            print(f"📄 RAG: used doc_id scope → {len(raw)} chunks")
            if not raw:
                raw = vector_store.search_by_session(query, session_id=session_id, k=15)
                print(f"📄 RAG: fallback session scope → {len(raw)} chunks")
        else:
            raw = vector_store.search_by_session(query, session_id=session_id, k=15)
            print(f"📄 RAG: used session scope → {len(raw)} chunks")

        paired      = _wrap_search_results(raw)
        tail_paired = _get_tail_chunks(doc_id=doc_id, session_id=session_id, n=5)
        print(f"📄 RAG: tail chunks fetched → {len(tail_paired)}")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    seen   = set()
    merged = []
    for doc, dedup_key in paired + tail_paired:
        if dedup_key and dedup_key not in seen:
            seen.add(dedup_key)
            merged.append(doc)

    print(f"📄 RAG: {len(paired)} semantic + {len(tail_paired)} tail → {len(merged)} after dedup")

    if not merged:
        print("📄 RAG: ⚠️  No chunks found")
        return {"rag_results": [], "rag_confidence": 0.0}

    # ── Format results ────────────────────────────────────────────────────────
    formatted = []
    for r in merged:
        meta   = r.metadata or {}
        nested = meta.get("metadata", {})

        file_name    = meta.get("file_name")   or nested.get("file_name")   or "Document"
        page         = meta.get("page_number") or meta.get("page") or nested.get("page_number") or nested.get("page") or "?"
        chunk_doc_id = meta.get("doc_id")      or nested.get("doc_id")      or doc_id
        chunk_index  = meta.get("chunk_index") or nested.get("chunk_index") or 0
        score        = meta.get("score", 0.7)

        formatted.append({
            "content":     r.page_content,
            "file_name":   file_name,
            "page":        page,
            "doc_id":      chunk_doc_id,
            "chunk_index": chunk_index,
            "session_id":  session_id,
            "source_type": "document",
            "score":       score,
        })

    confidence = min(len(formatted) / 10, 1.0)
    print(f"📄 RAG: {len(formatted)} chunks total, confidence={confidence:.2f}")
    return {"rag_results": formatted, "rag_confidence": confidence}