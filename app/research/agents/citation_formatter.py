import re
from urllib.parse import urlparse
from app.research.state import ResearchState

# Max chars of chunk content to expose as excerpt for frontend highlighting
_EXCERPT_LEN = 400


def citation_formatter_node(state: ResearchState) -> dict:
    answer     = state.get("final_answer", "") or ""
    source_map = state.get("cited_sources") or []

    # ── Short-circuit 1: guardrail refused ───────────────────────────────────
    if state.get("guardrail_decision") == "refuse":
        return {"validated_answer": answer, "sources": []}

    # ── Short-circuit 2: nothing to format ───────────────────────────────────
    if not answer or not source_map:
        return {"validated_answer": answer, "sources": []}

    # ── Validate which citation numbers appear in the answer ─────────────────
    used_ids = set(int(n) for n in re.findall(r'\[(\d+)\]', answer))

    sources = []
    for s in source_map:
        if s["id"] not in used_ids:
            continue

        if s["type"] == "document":
            sources.append({
                "id":          s["id"],
                "type":        "document",
                "title":       s["title"],
                "subtitle":    f"Page {s['page']}" if s.get("page") else "",
                "url":         None,
                "icon":        "📄",
                # ── NEW: fields for frontend excerpt panel ────────────────
                "doc_id":      s.get("doc_id", ""),
                "chunk_index": s.get("chunk_index", 0),
                "page":        s.get("page", ""),
                "excerpt":     s.get("content", "")[:_EXCERPT_LEN],
            })
        else:
            url    = s.get("url", "")
            domain = urlparse(url).netloc.replace("www.", "") if url else "Web"
            sources.append({
                "id":       s["id"],
                "type":     "web",
                "title": domain if domain != "Web" else (s["title"] or "Web")[:50],
                "subtitle": domain,
                "url":      url,
                "icon":     "🌐",
                # ── NEW: excerpt for web sources too ─────────────────────
                "excerpt":  s.get("content", "")[:_EXCERPT_LEN],
            })

    # Sort by first appearance in the answer
    order = {int(n): i for i, n in enumerate(re.findall(r'\[(\d+)\]', answer))}
    sources.sort(key=lambda s: order.get(s["id"], 999))

    if used_ids or sources:
        print(f"📎  CITATION FORMATTER: {len(used_ids)} citations used, {len(sources)} sources attached")

    return {
        "validated_answer": answer,
        "sources": sources
    }