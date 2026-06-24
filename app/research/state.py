from typing import TypedDict, List


class ResearchState(TypedDict, total=False):

    # ── Inputs (from client) ──────────────────────────────────────────────────
    question:              str          # Raw user question
    session_id:            str          # Session identifier
    language:              str          # "ar", "fr", or "en"
    doc_id:                str          # scope search to one document
    doc_ids: List[str]
    session_has_documents: bool 

    # ── Layer 1: Guardrail ────────────────────────────────────────────────────
    guardrail_decision:    str          # "proceed" | "refuse"
    guardrail_reason:      str

    # ── Layer 2: Reformulator → Router ───────────────────────────────────────
    reformulated_query:    str          # Precision-rewritten version of question

    # ── Layer 3: Router → Retrieval ──────────────────────────────────────────
    retrieval_mode:        str          # "rag" | "web" | "hybrid" | "none"
    recommendation_mode:   bool         # True if user wants a ranked recommendation

    # ── Layer 4: Retrieval → Re-Ranker ───────────────────────────────────────
    rag_results:           List[dict]   # Chunks from Qdrant
    rag_confidence:        float
    web_results:           List[dict]   # Results from Tavily
    web_confidence:        float

    # ── Layer 5: Re-Ranker → Synthesis ───────────────────────────────────────
    ranked_context:        List[dict]   # Merged + deduplicated results

    # ── Layer 6: Synthesis → Formatter ───────────────────────────────────────
    final_answer:          str          # Raw LLM output with inline [N] citations
    cited_sources:         List[dict]   # Rich source map built during synthesis

    # ── Layer 7: Formatter → Client ──────────────────────────────────────────
    validated_answer:      str          # Answer after citation validation
    sources:               List[dict]   # Clean source list for frontend rendering