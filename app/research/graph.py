# app/research/graph.py
"""
LangGraph research pipeline — 8-agent directed state machine.

Fixes applied vs original:
  Bug 1 — removed unconditional `add_edge("router", "web")` that fired for all modes.
           Hybrid fan-out now uses LangGraph's Send API (true parallel execution).
  Bug 2 — `session_has_documents` is now passed in `run_research_graph()` initial state.
  Bug 3 — `none` mode routes directly to END; router writes `validated_answer` so the
           clarification message reaches the client without going through synthesis or
           citation_formatter (which both expect a non-empty ranked_context).
  Bug 4 — END added explicitly to the router conditional edges map.
"""

from langgraph.graph import StateGraph, END
from langgraph.constants import Send

from app.research.state import ResearchState

from app.research.agents.guardrail          import guardrail_node
from app.research.agents.reformulator       import reformulator_node
from app.research.agents.router             import router_node
from app.research.agents.rag_agent          import rag_node
from app.research.agents.web_agent          import web_node
from app.research.agents.reranker           import reranker_node
from app.research.agents.synthesis          import synthesis_node
from app.research.agents.citation_formatter import citation_formatter_node

# ── Routing functions ─────────────────────────────────────────────────────────

def route_guardrail(state: ResearchState) -> str:
    """
    After guardrail runs:
      - "refuse"  → jump straight to citation_formatter which will pass
                    the refusal message already written into validated_answer.
      - "proceed" → continue to reformulator.
    """
    return "refuse" if state.get("guardrail_decision") == "refuse" else "proceed"


def route_router(state: ResearchState):
    """
    After router runs, decide which retrieval branch(es) to enter.

    Returns a string for single-branch modes, or a list of Send() objects
    for hybrid mode so LangGraph executes RAG and Web concurrently.

    Modes:
      "rag"    → document search only.
      "web"    → web search only.
      "hybrid" → both rag and web run in parallel (Send fan-out).
      "none"   → no retrieval possible; router already wrote validated_answer;
                 skip everything and go straight to END.
    """
    mode = state.get("retrieval_mode", "web")

    if mode == "hybrid":
        return [Send("rag", state), Send("web", state)]

    if mode == "none":
        return END

    # "rag" or "web"
    return mode


# ── Build graph ───────────────────────────────────────────────────────────────

builder = StateGraph(ResearchState)

# Register all nodes
builder.add_node("guardrail",          guardrail_node)
builder.add_node("reformulator",       reformulator_node)
builder.add_node("router",             router_node)
builder.add_node("rag",                rag_node)
builder.add_node("web",                web_node)
builder.add_node("reranker",           reranker_node)
builder.add_node("synthesis",          synthesis_node)
builder.add_node("citation_formatter", citation_formatter_node)

builder.set_entry_point("guardrail")

# ── Guardrail → refuse or proceed ────────────────────────────────────────────
builder.add_conditional_edges(
    "guardrail",
    route_guardrail,
    {
        "refuse":  "citation_formatter",
        "proceed": "reformulator",
    }
)

# ── Reformulator → Router (always) ───────────────────────────────────────────
builder.add_edge("reformulator", "router")

# ── Router → fan-out ─────────────────────────────────────────────────────────
# FIX: END added explicitly so LangGraph recognises the none-mode termination path.
builder.add_conditional_edges(
    "router",
    route_router,
    {
        "rag": "rag",
        "web": "web",
        END:   END,     # none mode — terminates immediately
    }
)

# ── Both retrieval nodes → reranker ──────────────────────────────────────────
builder.add_edge("rag", "reranker")
builder.add_edge("web", "reranker")

# ── Linear tail ──────────────────────────────────────────────────────────────
builder.add_edge("reranker",           "synthesis")
builder.add_edge("synthesis",          "citation_formatter")
builder.add_edge("citation_formatter", END)

research_graph = builder.compile()


# ── Public entry point ────────────────────────────────────────────────────────


def run_research_graph(
    question: str,
    session_id: str,
    language: str,
    session_has_documents: bool,
    doc_id: str = "",
    doc_ids: list = None,
) -> dict:
    initial_state = {
        "question":              question,
        "session_id":            session_id,
        "language":              language,
        "doc_id":                doc_id,
        "doc_ids":               doc_ids or [],
        "session_has_documents": session_has_documents,
        "recommendation_mode":   False,
        "retrieval_mode":        None,
        "reformulated_query":    None,
        "guardrail_decision":    None,
        "guardrail_reason":      None,
        "rag_results":           [],
        "web_results":           [],
        "rag_confidence":        0.0,
        "web_confidence":        0.0,
        "ranked_context":        [],
        "cited_sources":         [],
        "raw_citations":         [],
        "final_answer":          None,
        "validated_answer":      None,
        "citations":             [],
        "sources":               [],
    }

    
    result = research_graph.invoke(initial_state)


    return result