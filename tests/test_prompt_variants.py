"""
test_prompt_variants.py
─────────────────────────────────────────────────────────────────────────────
Professional prompt evaluation suite — PFE2026

Features:
  • Modular scoring via eval_scorer.py
  • Response caching (survives crashes, respects API quotas)
  • Graceful rate-limit handling with auto-retry
  • Configurable delay between calls
  • Markdown report generation
  • Multiple LLM backend support (Gemini, Ollama, etc.)
  • Pre-flight retrieval check

Usage:
    # 1. Check retrieval first (costs $0)
    pytest tests/test_retrieval_health.py -v

    # 2. Run full suite (uses cache, respects rate limits)
    pytest tests/test_prompt_variants.py -v -s

    # 3. Or run directly (no pytest needed)
    python tests/test_prompt_variants.py

    # 4. Generate report
    python tests/test_prompt_variants.py --report

Environment variables:
    EVAL_SESSION_ID      override the session_id
    EVAL_LLM             choose backend: gemini | ollama (default: gemini)
    EVAL_DELAY           seconds between API calls (default: 2.0)
    EVAL_MAX_RETRIES     retry count on 429 (default: 2)
─────────────────────────────────────────────────────────────────────────────
"""

import time
import sys
import os
import json
import hashlib
import argparse
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.services.vector_store import vector_store
from app.core.config import settings

# ── Import our reusable scorer ──────────────────────────────────────────────
from eval_scorer import score_answer


# ══════════════════════════════════════════════════════════════════════════════
# ① CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SESSION_ID = os.environ.get("EVAL_SESSION_ID", "")
K = 5

# API pacing (critical for free tiers)
DELAY_BETWEEN_CALLS = float(os.environ.get("EVAL_DELAY", "2.0"))
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "2"))

# Cache
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".eval_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "response_cache.json")

# Report output
REPORT_PATH = os.path.join(os.path.dirname(__file__), "PROMPT_EVALUATION_REPORT.md")


# ══════════════════════════════════════════════════════════════════════════════
# ② PROMPT VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

VARIANT_A = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful document assistant. Answer using ONLY the retrieved context below.
If the context doesn't contain the answer, say you don't know. Be concise.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

VARIANT_B = ChatPromptTemplate.from_messages([
    ("system", """You are a document assistant. Answer ONLY from the context below.
For every claim you make, cite the source using [Excerpt X].
If the answer is not found in the context, say exactly: "Not found in document."
Never use outside knowledge.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

VARIANT_C = ChatPromptTemplate.from_messages([
    ("system", """You are a document assistant. Follow these steps for every question:
1. Identify which excerpts are relevant to the question.
2. Extract the key facts from those excerpts only.
3. Write a concise answer with inline citations [Excerpt X].
If no excerpt is relevant, say: "Not found in document."
Never use outside knowledge.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

VARIANT_D = ChatPromptTemplate.from_messages([
    ("system", """You are a multilingual document assistant supporting Arabic, French, and English.
Rules:
- Detect the language of the user's question.
- Always respond in that same language.
- Answer ONLY from the context below. Cite as [Excerpt X].
- If the answer is not in the context, say so in the user's language.
- Never use outside knowledge.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

VARIANTS: Dict[str, ChatPromptTemplate] = {
    "A  baseline (current)  ": VARIANT_A,
    "B  grounded + cited    ": VARIANT_B,
    "C  chain-of-thought    ": VARIANT_C,
    "D  multilingual + cited": VARIANT_D,
}


# ══════════════════════════════════════════════════════════════════════════════
# ③ LLM BACKEND FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_llm(backend: str = "gemini"):
    """
    Return a LangChain chat model instance.

    Supported backends:
      gemini  → ChatGoogleGenerativeAI (your current free tier)
      ollama  → ChatOllama (local, unlimited, requires Ollama running)
    """
    backend = backend.lower().strip()

    if backend == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )

    if backend == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. Run: pip install langchain-ollama"
            )
        return ChatOllama(
            model="gemma2:2b",   # or gemma2:9b, mistral:7b, etc.
            temperature=0.3,
        )

    raise ValueError(f"Unknown LLM backend: {backend}. Choose 'gemini' or 'ollama'.")


# ══════════════════════════════════════════════════════════════════════════════
# ④ CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cache_key(llm_backend: str, variant_name: str, case_id: str, session_id: str) -> str:
    payload = f"{llm_backend}:{variant_name.strip()}:{case_id}:{session_id}"
    return hashlib.md5(payload.encode()).hexdigest()


def load_cache() -> Dict[str, Any]:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, Any]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

def build_context(question: str, session_id: str, k: int) -> Tuple[str, int]:
    """Return (context_string, num_chunks_retrieved)."""
    docs = vector_store.search_by_session(question, session_id=session_id, k=k)
    if not docs:
        return "", 0
    context = "\n\n".join(
        f"[Excerpt {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    return context, len(docs)


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_variant(
    llm_backend: str,
    v_name: str,
    prompt: ChatPromptTemplate,
    case: Dict[str, Any],
    session_id: str,
    k: int,
    cache: Dict[str, Any],
) -> Tuple[str, float, str, int]:
    """
    Execute one variant against one test case.

    Returns:
        answer_text, elapsed_seconds, status, num_chunks
    status: ok | cached | no_chunks | rate_limited | error
    """
    ckey = _cache_key(llm_backend, v_name, case["id"], session_id)

    if ckey in cache:
        entry = cache[ckey]
        return entry["answer"], entry["elapsed"], "cached", entry.get("chunks", 0)

    # Build LLM chain
    llm = build_llm(llm_backend)
    parser = StrOutputParser()
    chain = prompt | llm | parser

    # Retrieve context
    context, n_chunks = build_context(case["question"], session_id, k)
    if not context:
        return "[NO CHUNKS RETRIEVED — check session_id and Qdrant]", 0.0, "no_chunks", 0

    answer = ""
    elapsed = 0.0
    status = "error"

    for attempt in range(MAX_RETRIES):
        try:
            # Pace ourselves (critical for free tiers)
            time.sleep(DELAY_BETWEEN_CALLS)
            t0 = time.time()
            answer = chain.invoke({
                "context": context,
                "question": case["question"],
                "history": [],
            })
            elapsed = round(time.time() - t0, 2)
            status = "ok"
            break
        except Exception as exc:
            err_text = str(exc)
            if "429" in err_text or "RESOURCE_EXHAUSTED" in err_text or "rate limit" in err_text.lower():
                if attempt < MAX_RETRIES - 1:
                    wait = 60 + (attempt * 30)
                    print(f"\n    ⚠️  429 rate limit — waiting {wait}s before retry {attempt+1}/{MAX_RETRIES}...")
                    time.sleep(wait)
                    continue
                status = "rate_limited"
                answer = f"[RATE LIMITED: {err_text[:200]}]"
            else:
                status = "error"
                answer = f"[ERROR: {err_text[:200]}]"
            elapsed = 0.0

    # Persist to cache (even errors, so we don't hammer a broken combo)
    cache[ckey] = {
        "answer": answer,
        "elapsed": elapsed,
        "status": status,
        "chunks": n_chunks,
    }
    save_cache(cache)
    return answer, elapsed, status, n_chunks


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ ORCHESTRATOR + REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests(session_id: str, llm_backend: str = "gemini") -> Dict[str, Any]:
    """Run the full evaluation suite and return structured results."""
    cache = load_cache()

    # Load test cases
    cases_path = os.path.join(os.path.dirname(__file__), "eval_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print("\n")
    print("═" * 72)
    print("  PROMPT ENGINEERING TEST SUITE — PFE2026")
    print("═" * 72)
    print(f"  Session ID : {session_id}")
    print(f"  LLM Backend: {llm_backend}")
    print(f"  Variants   : {len(VARIANTS)}")
    print(f"  Test cases : {len(test_cases)}")
    print(f"  Cache      : {len(cache)} entries loaded")
    print(f"  Delay      : {DELAY_BETWEEN_CALLS}s between calls")
    print("═" * 72)

    results: Dict[str, Dict[str, Any]] = {v: {} for v in VARIANTS}

    for case in test_cases:
        print(f"\n\n{'━'*72}")
        print(f"  TEST CASE: {case['id']}")
        print(f"  {case['description']}")
        print(f"  Question : {case['question']}")
        print(f"{'━'*72}")

        for v_name, prompt in VARIANTS.items():
            print(f"\n  ▶ Variant {v_name.strip()} …", end=" ", flush=True)
            answer, elapsed, status, n_chunks = run_variant(
                llm_backend, v_name, prompt, case, session_id, K, cache
            )
            scores = score_answer(answer, case)
            results[v_name][case["id"]] = {
                "answer": answer,
                "scores": scores,
                "elapsed": elapsed,
                "status": status,
                "chunks": n_chunks,
            }

            flag = {
                "ok": "✅",
                "cached": "💾",
                "no_chunks": "⚠️",
                "rate_limited": "⛔",
                "error": "❌",
            }.get(status, "?")
            print(f"{flag} {status} ({elapsed}s, {n_chunks} chunks)  |  score: {scores['total']:.0%}")
            print(f"  {'─'*68}")

            short = answer.replace("\n", " ")
            if len(short) > 400:
                short = short[:400] + "…"
            print(f"  {short}")

            score_line = "  Scores → " + " | ".join(
                f"{k}: {v:.0%}" for k, v in scores.items() if k != "total"
            )
            print(score_line)

    # ── LEADERBOARD ──────────────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print("  LEADERBOARD")
    print(f"{'═'*72}")
    print(f"  {'Variant':<30} {'Avg Score':>10}  {'Per-case scores'}")
    print(f"  {'─'*30} {'─'*10}  {'─'*28}")

    variant_totals: Dict[str, float] = {}
    for v_name in VARIANTS:
        case_scores = [results[v_name][c["id"]]["scores"]["total"] for c in test_cases]
        avg = sum(case_scores) / len(case_scores)
        variant_totals[v_name] = avg
        case_str = "  ".join(f"{s:.0%}" for s in case_scores)
        print(f"  {v_name:<30} {avg:>9.0%}   {case_str}")

    winner_name = max(variant_totals, key=variant_totals.get)
    winner_score = variant_totals[winner_name]

    print(f"\n{'═'*72}")
    print(f"  🏆 WINNER: {winner_name.strip()}  ({winner_score:.0%})")
    print(f"{'═'*72}")

    # ── PER-CASE BREAKDOWN ───────────────────────────────────────────────────
    print("\n\n  PER-CASE BREAKDOWN")
    print(f"  {'─'*68}")
    for case in test_cases:
        print(f"\n  Case: {case['id']}")
        ranked = sorted(
            VARIANTS.keys(),
            key=lambda v: results[v][case["id"]]["scores"]["total"],
            reverse=True,
        )
        for rank, v_name in enumerate(ranked, 1):
            s = results[v_name][case["id"]]["scores"]["total"]
            elapsed = results[v_name][case["id"]]["elapsed"]
            status = results[v_name][case["id"]]["status"]
            print(f"    {rank}. {v_name.strip():<30} {s:.0%}  ({elapsed}s) [{status}]")

    # ── RECOMMENDATION ───────────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print("  RECOMMENDATION")
    print(f"{'═'*72}")
    print(f"  Best overall  → {winner_name.strip()}")
    print()

    def best_for(case_id: str, dim: str) -> str:
        return max(
            VARIANTS.keys(),
            key=lambda v: results[v].get(case_id, {}).get("scores", {}).get(dim, 0),
        )

    print(f"  Best for Arabic responses → {best_for('arabic_question', 'language').strip()}")
    print(f"  Best for citations        → {best_for('citation_test', 'citation').strip()}")
    print(f"  Best at refusing OOS      → {best_for('out_of_scope', 'refusal').strip()}")
    print()
    print("  → Copy the winning variant's system prompt into:")
    print("    app/services/rag_service.py  →  self.rag_prompt")
    print(f"{'═'*72}\n")

    return {
        "results": results,
        "variant_totals": variant_totals,
        "winner": {"name": winner_name.strip(), "score": winner_score},
        "test_cases": test_cases,
        "llm_backend": llm_backend,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
    }


def generate_markdown_report(run_data: Dict[str, Any], output_path: str = REPORT_PATH):
    """Write a human-readable Markdown report for company delivery."""
    results = run_data["results"]
    totals = run_data["variant_totals"]
    winner = run_data["winner"]
    cases = run_data["test_cases"]
    backend = run_data["llm_backend"]
    ts = run_data["timestamp"]

    lines = [
        "# Prompt Engineering Evaluation Report",
        f"**Project:** PFE2026  ",
        f"**Date:** {ts}  ",
        f"**LLM Backend:** {backend}  ",
        f"**Session ID:** {run_data['session_id']}  ",
        "",
        "## Executive Summary",
        f"",
        f"**Winner:** `{winner['name']}` with an average score of **{winner['score']:.0%}**.",
        f"",
        "## Leaderboard",
        "",
        "| Variant | Avg Score | Case Scores |",
        "|---------|-----------|-------------|",
    ]

    for v_name in sorted(totals.keys(), key=lambda v: totals[v], reverse=True):
        scores = [results[v_name][c["id"]]["scores"]["total"] for c in cases]
        score_str = " ".join(f"{s:.0%}" for s in scores)
        lines.append(f"| {v_name.strip()} | {totals[v_name]:.0%} | {score_str} |")

    lines += [
        "",
        "## Detailed Results",
        "",
    ]

    for case in cases:
        lines.append(f"### {case['id']}: {case['description']}")
        lines.append(f"**Question:** {case['question']}")
        lines.append("")
        lines.append("| Rank | Variant | Score | Status | Answer Preview |")
        lines.append("|------|---------|-------|--------|----------------|")

        ranked = sorted(
            VARIANTS.keys(),
            key=lambda v: results[v][case["id"]]["scores"]["total"],
            reverse=True,
        )
        for rank, v_name in enumerate(ranked, 1):
            entry = results[v_name][case["id"]]
            ans = entry["answer"].replace("\n", " ")[:120]
            if len(entry["answer"]) > 120:
                ans += "…"
            lines.append(
                f"| {rank} | {v_name.strip()} | {entry['scores']['total']:.0%} | {entry['status']} | {ans} |"
            )
        lines.append("")

    lines += [
        "## Regression Assertions",
        "The following assertions must pass for any prompt change to be accepted:",
        "",
    ]

    # Auto-generate regression thresholds from current best scores
    for case in cases:
        for dim in ["contains", "citation", "refusal", "language"]:
            best = max(results[v][case["id"]]["scores"][dim] for v in VARIANTS)
            if best > 0:
                lines.append(f"- `{case['id']}` → `{dim}` must be ≥ {best:.0%}")

    lines += [
        "",
        "## Methodology",
        "1. Retrieve top-5 chunks from Qdrant per question.",
        "2. Feed chunks + question into each prompt variant.",
        "3. Score answers on: content accuracy, hallucination avoidance, citation compliance, refusal correctness, language match.",
        "4. Cache all responses to survive rate-limit interruptions.",
        "5. Average per-case scores to determine the winner.",
        "",
        "---",
        "*Generated automatically by `test_prompt_variants.py`*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📄 Markdown report written to: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ⑧ PYTEST ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def test_prompt_variants():
    """pytest entry point."""
    assert SESSION_ID != "REPLACE_WITH_YOUR_SESSION_ID", (
        "Set SESSION_ID or export EVAL_SESSION_ID before running."
    )

    llm_backend = os.environ.get("EVAL_LLM", "gemini")
    run_data = run_all_tests(SESSION_ID, llm_backend)

    # Hard regression: winner must be above 50%
    assert run_data["winner"]["score"] > 0.5, (
        f"Winner scored {run_data['winner']['score']:.0%} — "
        "check your session_id and Qdrant index."
    )

    # Generate report if requested via env var
    if os.environ.get("EVAL_REPORT", "0") == "1":
        generate_markdown_report(run_data)

    return run_data


# ══════════════════════════════════════════════════════════════════════════════
# ⑨ DIRECT RUN (python tests/test_prompt_variants.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Evaluation Suite")
    parser.add_argument("--report", action="store_true", help="Generate Markdown report")
    parser.add_argument("--llm", default="gemini", help="LLM backend: gemini | ollama")
    parser.add_argument("--session", default=SESSION_ID, help="Qdrant session_id")
    args = parser.parse_args()

    if args.session == "REPLACE_WITH_YOUR_SESSION_ID":
        print("\n  ERROR: Set SESSION_ID or use --session.")
        sys.exit(1)

    os.environ["EVAL_LLM"] = args.llm
    run_data = run_all_tests(args.session, args.llm)

    if args.report:
        generate_markdown_report(run_data)