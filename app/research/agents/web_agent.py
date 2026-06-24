import os
import httpx
from tavily import TavilyClient
from app.research.state import ResearchState

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ─────────────────────────────────────────────
# Academic query detection
# ─────────────────────────────────────────────

ACADEMIC_KEYWORDS = [
    # English
    "paper", "article", "research", "published", "arxiv", "authors",
    "citation", "conference", "journal", "pdf", "where can i find",
    "source of", "who wrote", "transformer", "attention is all you need",
    # French
    "article", "papier", "recherche", "publié", "auteurs", "revue",
    # Arabic
    "بحث", "ورقة", "مقال", "نشر", "مؤلف", "مجلة",
]

ACADEMIC_DOMAINS = [
    "arxiv.org",
    "semanticscholar.org",
    "proceedings.neurips.cc",
    "aclanthology.org",
    "openreview.net",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "researchgate.net",
]

def _is_academic_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in ACADEMIC_KEYWORDS)


# ─────────────────────────────────────────────
# Semantic Scholar enrichment (free, no key)
# ─────────────────────────────────────────────

def _enrich_with_semantic_scholar(title: str) -> dict | None:
    """
    Given a paper title, fetch its canonical arXiv URL,
    PDF link, authors and year from Semantic Scholar API.
    Returns None if not found or on error.
    """
    try:
        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": title,
                "fields": "title,authors,year,externalIds,openAccessPdf",
                "limit": 1,
            },
            timeout=5.0,
        )
        data = resp.json().get("data", [])
        if not data:
            return None

        paper = data[0]
        arxiv_id = paper.get("externalIds", {}).get("ArXiv")
        pdf_info = paper.get("openAccessPdf") or {}

        return {
            "canonical_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            "pdf_url":       pdf_info.get("url"),
            "authors":       [a["name"] for a in paper.get("authors", [])],
            "year":          paper.get("year"),
            "ss_title":      paper.get("title"),
        }
    except Exception as e:
        print(f"🔬 SemanticScholar enrichment failed: {e}")
        return None


# ─────────────────────────────────────────────
# Tavily search helpers
# ─────────────────────────────────────────────

def _tavily_search(query: str, domains: list[str] | None = None) -> list[dict]:
    """
    Run a Tavily search. If `domains` is provided, restrict results to those.
    Returns a list of {title, url, content, source_type} dicts.
    """
    kwargs = dict(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=False,
    )
    if domains:
        kwargs["include_domains"] = domains

    resp = tavily.search(**kwargs)

    return [
        {
            "title":       r.get("title"),
            "url":         r.get("url"),
            "content":     r.get("content"),
            "source_type": "academic" if domains else "web",
        }
        for r in resp.get("results", [])
    ]


# ─────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────

def web_node(state: ResearchState) -> dict:
    # FIX: use "reformulated_query" (not "reformulated_question") to match state schema
    query    = state.get("reformulated_query") or state["question"]
    question = state["question"]

    print(f"🌐 WEB: searching for '{query[:60]}...'")

    try:
        academic = _is_academic_query(question)

        if academic:
            print("🔬 WEB: academic query detected → using domain-restricted search")
            results = _tavily_search(query, domains=ACADEMIC_DOMAINS)

            # Enrich each result with Semantic Scholar metadata
            for r in results:
                enrichment = _enrich_with_semantic_scholar(r["title"] or query)
                if enrichment:
                    # Override URL with canonical arXiv link if available
                    if enrichment["canonical_url"]:
                        r["url"] = enrichment["canonical_url"]
                    r["pdf_url"] = enrichment.get("pdf_url")
                    r["authors"] = enrichment.get("authors", [])
                    r["year"]    = enrichment.get("year")

            # If academic search returned nothing, fall back to general search
            if not results:
                print("🌐 WEB: academic search empty → fallback to general")
                results = _tavily_search(query)

        else:
            results = _tavily_search(query)

        print(f"🌐 WEB: found {len(results)} results "
              f"({'academic' if academic else 'general'})")
        return {
            "web_results": results,
            "web_confidence": min(len(results) / 5, 1.0)
        }

    except Exception as e:
        print(f"🌐 WEB: ⚠️ failed ({e}), returning empty")
        return {"web_results": []}