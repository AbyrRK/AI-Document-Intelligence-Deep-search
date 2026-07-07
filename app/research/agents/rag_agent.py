from app.research.state import ResearchState
from app.services.vector_store import vector_store
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document

_SMALL_DOC_THRESHOLD = 10

# ── Module-level constants (not inside any function) ──────────────────────────
SUMMARY_SIGNALS = {
    "ttc", "tva", "total", "net à payer", "montant total",
    "hors taxes", "toutes taxes",
    "مجموع", "الإجمالي", "إجمالي", "ضريبة",
}


def _has_summary_chunk(chunks: list) -> bool:
    for c in chunks:
        content_lower = (
            c.get("content", "").lower() if isinstance(c, dict)
            else c.page_content.lower()
        )
        if any(sig in content_lower for sig in SUMMARY_SIGNALS):
            return True
    return False


def _count_chunks(doc_id: str = "", session_id: str = "") -> int:
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
        return 999


def _get_tail_chunks(doc_id: str = "", session_id: str = "", n: int = 5) -> list:
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
    return [(doc, doc.page_content[:120].strip()) for doc in docs]


def rag_node(state: ResearchState) -> dict:
    session_id = state.get("session_id", "")
    doc_id     = state.get("doc_id", "")
    query      = state.get("reformulated_query") or state["question"]

    print(f"📄 RAG: query='{query[:50]}...' | session={session_id[:8] if session_id else 'NONE'} | doc_id={doc_id[:8] if doc_id else 'NONE'}")

    if not session_id and not doc_id:
        print("📄 RAG: ⚠️  No session_id or doc_id — skipping document search")
        return {"rag_results": [], "rag_confidence": 0.0}

    total_chunks = _count_chunks(
        doc_id=doc_id,
        session_id=session_id if not doc_id else ""
    )

    if total_chunks <= _SMALL_DOC_THRESHOLD:
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
        tail_paired = []

    else:
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

    # ── Totals safety net — runs BEFORE format loop ───────────────────────────
    if not _has_summary_chunk(merged):
        print("📄 RAG: ⚠️  No totals/TVA chunk found — force-fetching tail chunks")
        force_tail = _get_tail_chunks(
            doc_id=doc_id,
            session_id=session_id if not doc_id else "",
            n=2
        )
        for doc, dedup_key in force_tail:
            if dedup_key not in seen:
                seen.add(dedup_key)
                merged.append(doc)
                print(f"📄 RAG: ✅ force-added tail chunk ({len(doc.page_content)} chars)")
            else:
                print("📄 RAG: tail chunk already in merged — skipping")

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
            "content":      r.page_content,
            "file_name":    file_name,
            "page":         page,
            "doc_id":       chunk_doc_id,
            "chunk_index":  chunk_index,
            "session_id":   session_id,
            "source_type":  "document",
            "score":        score,
            "content_type": meta.get("content_type") or nested.get("content_type", "text"),
        })

    confidence = min(len(formatted) / 10, 1.0)
    print(f"📄 RAG: {len(formatted)} chunks total, confidence={confidence:.2f}")
    return {"rag_results": formatted, "rag_confidence": confidence}