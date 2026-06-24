"""
compare_llms.py
─────────────────────────────────────────────────────────────────────────────
Compare multiple LLM backends under free-tier / limited-key constraints.

Purpose:
  Prove to the company which LLM is most reliable when API quotas are tight.

Backends tested:
  • Gemini 2.5 Flash (Google) — free tier, 20 req/day
  • Ollama (local) — unlimited, requires local GPU/CPU

Usage:
    # Compare all available backends on a subset of cases
    python tests/compare_llms.py --cases 3

    # Compare specific backends
    python tests/compare_llms.py --backends gemini ollama

    # Generate comparison report
    python tests/compare_llms.py --report
─────────────────────────────────────────────────────────────────────────────
"""

import time
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.services.vector_store import vector_store
from app.core.config import settings
from eval_scorer import score_answer


# ── CONFIG ──────────────────────────────────────────────────────────────────

SESSION_ID = os.environ.get("EVAL_SESSION_ID", "bf4def75-d317-47f3-bdeb-a4f06f419b6f")
K = 5
DELAY = float(os.environ.get("EVAL_DELAY", "2.0"))

# Default Ollama model — gemma2:2b is safe for 8GB RAM machines
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")

# ── PROMPT (use the winning variant or a fixed one for fair comparison) ─────

BENCHMARK_PROMPT = ChatPromptTemplate.from_messages([
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


# ── LLM FACTORY ─────────────────────────────────────────────────────────────

def build_llm(backend: str):
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
            raise ImportError("pip install langchain-ollama")
        return ChatOllama(
            model=DEFAULT_OLLAMA_MODEL,
            temperature=0.3,
        )

    raise ValueError(f"Unknown backend: {backend}")


# ── RETRIEVAL ─────────────────────────────────────────────────────────────────

def build_context(question: str, session_id: str, k: int) -> Tuple[str, int]:
    docs = vector_store.search_by_session(question, session_id=session_id, k=k)
    if not docs:
        return "", 0
    context = "\n\n".join(
        f"[Excerpt {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    return context, len(docs)


# ── RUNNER ────────────────────────────────────────────────────────────────────

def run_backend(backend: str, case: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Run a single backend against a single case. Return metrics."""
    llm = build_llm(backend)
    parser = StrOutputParser()
    chain = BENCHMARK_PROMPT | llm | parser

    context, n_chunks = build_context(case["question"], session_id, K)
    if not context:
        return {
            "answer": "[NO CHUNKS RETRIEVED]",
            "elapsed": 0.0,
            "status": "no_chunks",
            "chunks": 0,
            "scores": score_answer("[NO CHUNKS RETRIEVED]", case),
        }

    try:
        time.sleep(DELAY)
        t0 = time.time()
        answer = chain.invoke({
            "context": context,
            "question": case["question"],
            "history": [],
        })
        elapsed = round(time.time() - t0, 2)
        status = "ok"
    except Exception as exc:
        err = str(exc)
        if "429" in err or "rate limit" in err.lower() or "exhausted" in err.lower():
            status = "rate_limited"
        else:
            status = "error"
        answer = f"[{status.upper()}: {err[:180]}]"
        elapsed = 0.0

    return {
        "answer": answer,
        "elapsed": elapsed,
        "status": status,
        "chunks": n_chunks,
        "scores": score_answer(answer, case),
    }


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

def compare_backends(backends: List[str], max_cases: int = 12) -> Dict[str, Any]:
    cases_path = os.path.join(os.path.dirname(__file__), "eval_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    cases = all_cases[:max_cases]

    print("\n")
    print("═" * 72)
    print("  LLM BACKEND COMPARISON — PFE2026")
    print("═" * 72)
    print(f"  Backends   : {', '.join(backends)}")
    print(f"  Cases      : {len(cases)}")
    print(f"  Prompt     : Multilingual (Variant D)")
    print("═" * 72)

    results: Dict[str, Dict[str, Any]] = {b: {} for b in backends}

    for case in cases:
        print(f"\n\n{'━'*72}")
        print(f"  CASE: {case['id']} — {case['description']}")
        print(f"  Q: {case['question']}")
        print(f"{'━'*72}")

        for backend in backends:
            print(f"\n  ▶ {backend.upper()} …", end=" ", flush=True)
            res = run_backend(backend, case, SESSION_ID)
            results[backend][case["id"]] = res

            flag = {"ok": "✅", "no_chunks": "⚠️", "rate_limited": "⛔", "error": "❌"}.get(res["status"], "?")
            print(f"{flag} {res['status']} ({res['elapsed']}s, {res['chunks']} chunks) | score: {res['scores']['total']:.0%}")
            short = res["answer"].replace("\n", " ")[:300]
            print(f"  {short}{'…' if len(res['answer']) > 300 else ''}")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n\n{'═'*72}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*72}")
    print(f"  {'Backend':<15} {'Avg Score':>10} {'OK':>6} {'Rate-Lim':>10} {'Error':>8} {'Avg Time':>10}")
    print(f"  {'─'*15} {'─'*10} {'─'*6} {'─'*10} {'─'*8} {'─'*10}")

    summary: Dict[str, Dict[str, Any]] = {}
    for backend in backends:
        scores = [results[backend][c["id"]]["scores"]["total"] for c in cases]
        statuses = [results[backend][c["id"]]["status"] for c in cases]
        times = [results[backend][c["id"]]["elapsed"] for c in cases if results[backend][c["id"]]["elapsed"] > 0]

        avg_score = sum(scores) / len(scores)
        ok_count = statuses.count("ok")
        rl_count = statuses.count("rate_limited")
        err_count = statuses.count("error")
        avg_time = sum(times) / len(times) if times else 0.0

        summary[backend] = {
            "avg_score": avg_score,
            "ok": ok_count,
            "rate_limited": rl_count,
            "error": err_count,
            "avg_time": avg_time,
        }

        print(f"  {backend:<15} {avg_score:>9.0%} {ok_count:>6} {rl_count:>10} {err_count:>8} {avg_time:>9.1f}s")

    # Winner by score
    winner = max(summary.keys(), key=lambda b: summary[b]["avg_score"])
    print(f"\n  🏆 Best by score: {winner.upper()} ({summary[winner]['avg_score']:.0%})")

    # Winner by reliability (fewest failures)
    reliable = min(summary.keys(), key=lambda b: summary[b]["rate_limited"] + summary[b]["error"])
    if reliable != winner:
        print(f"  🛡️  Most reliable: {reliable.upper()} (fewest failures)")

    return {
        "results": results,
        "summary": summary,
        "winner": winner,
        "backends": backends,
        "cases": cases,
        "timestamp": datetime.now().isoformat(),
    }


def generate_comparison_report(data: Dict[str, Any], path: str = "LLM_COMPARISON_REPORT.md"):
    lines = [
        "# LLM Backend Comparison Report",
        f"**Date:** {data['timestamp']}  ",
        f"**Backends:** {', '.join(data['backends'])}  ",
        f"**Cases:** {len(data['cases'])}  ",
        "",
        "## Executive Summary",
        f"",
        f"**Best by accuracy:** `{data['winner'].upper()}`  ",
        "",
        "## Scoreboard",
        "",
        "| Backend | Avg Score | OK | Rate-Limited | Error | Avg Latency |",
        "|---------|-----------|----|--------------|-------|-------------|",
    ]

    for backend in sorted(data["summary"].keys(), key=lambda b: data["summary"][b]["avg_score"], reverse=True):
        s = data["summary"][backend]
        lines.append(
            f"| {backend} | {s['avg_score']:.0%} | {s['ok']} | {s['rate_limited']} | {s['error']} | {s['avg_time']:.1f}s |"
        )

    lines += ["", "## Per-Case Results", ""]

    for case in data["cases"]:
        lines.append(f"### {case['id']}: {case['description']}")
        lines.append(f"**Question:** {case['question']}")
        lines.append("")
        lines.append("| Backend | Score | Status | Time | Answer Preview |")
        lines.append("|---------|-------|--------|------|----------------|")

        for backend in data["backends"]:
            entry = data["results"][backend][case["id"]]
            ans = entry["answer"].replace("\n", " ")[:100]
            if len(entry["answer"]) > 100:
                ans += "…"
            lines.append(
                f"| {backend} | {entry['scores']['total']:.0%} | {entry['status']} | {entry['elapsed']}s | {ans} |"
            )
        lines.append("")

    lines += [
        "## Recommendation",
        "",
        f"For production deployment under API constraints, **{data['winner'].upper()}** is recommended based on this benchmark.",
        "",
        "---",
        "*Generated by compare_llms.py*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📄 Comparison report written to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["gemini", "ollama"],
                        help="Backends to compare: gemini ollama")
    parser.add_argument("--cases", type=int, default=12,
                        help="Number of test cases to run (default: all)")
    parser.add_argument("--report", action="store_true",
                        help="Generate Markdown report")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_OLLAMA_MODEL})")
    args = parser.parse_args()

    # Allow CLI override of Ollama model
    if args.ollama_model != DEFAULT_OLLAMA_MODEL:
        DEFAULT_OLLAMA_MODEL = args.ollama_model

    data = compare_backends(args.backends, max_cases=args.cases)

    if args.report:
        generate_comparison_report(data)