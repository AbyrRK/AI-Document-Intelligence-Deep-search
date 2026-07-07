from app.research.state import ResearchState

TOP_K = 7

SOURCE_WEIGHT = {
    "document": 1.0,
    "web":      0.85,
    "academic": 0.95,
}

SIMILARITY_THRESHOLD = 0.6


def reranker_node(state: ResearchState) -> dict:
    rag = state.get("rag_results", [])
    web = state.get("web_results", [])

    print(f"\n🔀  RERANKER: {len(rag)} RAG + {len(web)} web results → top {TOP_K}")

    if not rag and not web:
        print("🔀  RERANKER: no results to rank")
        return {"ranked_context": []}

    # ── Step 1: Normalize scores ──────────────────────────────────────────────
    unified = []

    for r in rag:
        raw_score = r.get("score", 0.7)
        weight    = SOURCE_WEIGHT["document"]
        unified.append({
            "content":     r.get("content", ""),
            "file_name":   r.get("file_name", ""),
            "page":        r.get("page", ""),
            "url":         "",
            "title":       r.get("file_name", ""),
            "source_type": "document",
            "doc_id":      r.get("doc_id", ""),
            "chunk_index": r.get("chunk_index", 0),
            "score":       round(raw_score * weight, 4),
            "content_type": r.get("content_type", "text"),
        })

    for w in web:
        raw_score   = w.get("score", 0.6)
        source_type = w.get("source_type", "web")
        weight      = SOURCE_WEIGHT.get(source_type, 0.85)
        unified.append({
            "content":     w.get("content", ""),
            "file_name":   "",
            "page":        "",
            "url":         w.get("url", ""),
            "title":       w.get("title", ""),
            "source_type": source_type,
            "doc_id":      "",
            "chunk_index": None,
            "score":       round(raw_score * weight, 4),
        })

    # ── Step 2: Sort by score descending ─────────────────────────────────────
    unified.sort(key=lambda x: x["score"], reverse=True)

    # ── Step 3: Deduplicate ───────────────────────────────────────────────────
    # FIX: never deduplicate two document chunks against each other.
    # Document chunks from the same file contain different parts of the document
    # and may share vocabulary (company name, headers) without being duplicates.
    # Only deduplicate: web vs web, or web vs doc (web is redundant if doc says same thing).
    deduplicated = []
    for candidate in unified:
        if not _is_duplicate(candidate, deduplicated):
            deduplicated.append(candidate)

    # ── Step 4: Keep top K ───────────────────────────────────────────────────
    ranked = deduplicated[:TOP_K]

    print(f"🔀  RERANKER: {len(unified)} → {len(deduplicated)} after dedup → {len(ranked)} kept")
    for i, r in enumerate(ranked, 1):
        src = r["file_name"] or r["title"] or r["url"]
        print(f"   [{i}] score={r['score']:.3f} | {r['source_type']} | {src[:50]}")

    return {"ranked_context": ranked}


def _is_duplicate(candidate: dict, existing: list[dict]) -> bool:
    """
    Returns True if candidate is a duplicate of something already kept.

    Rules:
    - Two document chunks are NEVER duplicates of each other — they contain
      different parts of the document even if they share vocabulary.
    - A web result can be a duplicate of another web result (same article, different source).
    - A web result can be dropped if a document chunk already covers the same fact
      (doc is more authoritative).
    - A document chunk is never dropped in favor of a web result.
    """
    candidate_type = candidate["source_type"]

    # Document chunks are always kept — never deduplicated
    if candidate_type == "document":
        return False

    # Web result — check against existing web results only
    candidate_words = _word_set(candidate["content"])
    if not candidate_words:
        return False

    for kept in existing:
        # Only compare web vs web
        if kept["source_type"] != "web":
            continue

        kept_words = _word_set(kept["content"])
        if not kept_words:
            continue

        overlap = len(candidate_words & kept_words) / len(candidate_words | kept_words)

        if overlap >= SIMILARITY_THRESHOLD:
            winner = kept["title"] or kept["url"] or "kept"
            loser  = candidate["title"] or candidate["url"] or "duplicate"
            print(f"🔀  RERANKER: dedup — '{loser[:40]}' overlaps '{winner[:40]}' ({overlap:.0%})")
            return True

    return False


def _word_set(text: str) -> set[str]:
    stop = {"the", "a", "an", "is", "in", "of", "and", "or", "to", "for",
            "de", "le", "la", "les", "un", "une", "et", "du", "في", "من",
            "إلى", "على", "مع", "هو", "هي"}
    return {w for w in text.lower().split() if len(w) > 2 and w not in stop}