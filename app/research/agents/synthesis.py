import os
import hashlib
import re
import json
from urllib.parse import urlparse

from sympy import content

from app.research.state import ResearchState
from app.services.llm_client import llm

# How many chars of each chunk to show in the prompt
# 200 was too short — financial tables need ~500 to show all rows
_DOC_SNIPPET_LEN = 600
_WEB_SNIPPET_LEN = 300


def synthesis_node(state: ResearchState) -> dict:
    ranked            = state.get("ranked_context", [])
    rag               = [r for r in ranked if r.get("source_type") == "document"]
    web               = [r for r in ranked if r.get("source_type") == "web"]

    lang              = state["language"]
    question          = state["question"]
    is_recommendation = state.get("recommendation_mode", False)

    print(f"\n🧠  SYNTHESIS: {len(rag)} RAG + {len(web)} Web chunks | recommendation={is_recommendation}")

    # DEBUG — shows exactly what the LLM sees per chunk
    for i, r in enumerate(ranked, 1):
        print(f"  [{i}] {r.get('source_type')} | {r.get('file_name') or r.get('title','')} | score={r.get('score',0):.3f}")
        print(f"       {r.get('content','')[:200]}")

    if not rag and not web:
        return {
            "final_answer": _no_sources_message(lang),
            "cited_sources": []
        }

    # ── Build numbered source list ────────────────────────────────────────────
    source_map    = []
    source_blocks = []

    for i, r in enumerate(rag, 1):
        name    = r.get("file_name", f"Document-{i}")
        page    = r.get("page", "")
        content = r.get("content", "")
        snippet_len = 1000 if r.get("content_type") == "table" else _DOC_SNIPPET_LEN
        snippet = content[:snippet_len]
        source_blocks.append(f"[{i}] {name}{f', p.{page}' if page else ''}:\n{snippet}")
        source_map.append({
            "id":        i,
            "type":      "document",
            "name":      name,
            "title":     name,
            "file_name": name,
            "page":      page,
            "url":       "",
            "content":   content,
        })

    for i, w in enumerate(web, len(rag) + 1):
        url     = w.get("url", "")
        title   = w.get("title", "")
        domain  = urlparse(url).netloc.replace("www.", "") if url else f"Web-{i}"
        display = title[:40] if title else domain
        snippet = w.get("content", "")[:_WEB_SNIPPET_LEN]
        source_blocks.append(f"[{i}] {display} ({domain}):\n{snippet}")
        source_map.append({
            "id":        i,
            "type":      "web",
            "name":      display,
            "title":     title or domain,
            "file_name": "",
            "page":      "",
            "url":       url,
            "content":   w.get("content", ""),
        })

    sources_text = "\n\n".join(source_blocks)

    lang_instruction = {
        "ar": "Answer entirely in Arabic (العربية). Do not use any other language.",
        "fr": "Answer entirely in French. Do not use any other language.",
        "en": "Answer entirely in English. Do not use any other language.",
    }.get(lang, f"Answer in the same language as the question ({lang}).")
    
    from app.services.pii_scrubber import scrub_prompt
    sources_text = scrub_prompt(sources_text, language=lang)

    if is_recommendation:
        llm_prompt = _recommendation_prompt(question, sources_text, lang_instruction)
    else:
        llm_prompt = _analysis_prompt(question, sources_text, lang_instruction)

    try:
        answer = llm.generate_sync(
            prompt=llm_prompt,
            temperature=0.7,
            max_tokens=3000,      
            use_cache=True,
        )

        if answer.startswith("[System"):
            print(f"🧠  SYNTHESIS: ⚠️  {answer}")
        else:
            print(f"🧠  SYNTHESIS: ✅ {len(answer)} chars generated")

        recommendations, clean_answer = _extract_recommendations(answer)

        return {
            "final_answer":    clean_answer,
            "cited_sources":   source_map,
            "recommendations": recommendations,
        }

    except Exception as e:
        print(f"🧠  SYNTHESIS: ❌ Unexpected error — {e}")
        return {
            "final_answer":    _no_sources_message(lang),
            "cited_sources":   source_map,
            "recommendations": [],
        }


def _analysis_prompt(question: str, sources_text: str, lang_instruction: str) -> str:
    return f"""You are a senior business analyst producing a concise, structured intelligence report.

LANGUAGE: {lang_instruction}

SOURCES (numbered — you must cite every factual claim):
{sources_text}

QUESTION: {question}

INSTRUCTIONS:
1. Open with a one-sentence verdict or direct answer to the question. No preamble.
2. Organize the body into 2-4 sections with ## headings. Each section covers ONE distinct finding.
   - State the finding clearly and concisely.
   - Cite every claim inline: [1], [2], or [2][4].
   - Only add context where it genuinely adds insight.
3. If sources contradict each other, state both positions and note the conflict explicitly.
4. Never write a raw URL in the answer body.
5. If no source supports a claim, do not make it.
6. End with a ## Conclusion (2-3 sentences max) that gives the key takeaway and any recommended action.
7. Do not repeat information already stated in an earlier section.
8. After ## Conclusion, if there are genuine actionable recommendations based strictly on
   the sources, append this block:
   ```json
   {{
     "recommendations": [
       {{"text": "Recommendation text here", "source": "[1]"}},
       {{"text": "Another recommendation", "source": "[2][3]"}}
     ]
   }}
   ```
   Rules for recommendations:
   - Maximum 3 items
   - Each must cite at least one source using [N] notation
   - Only include if genuinely actionable — omit entirely if nothing concrete to recommend
   - Same language as the rest of the answer

OUTPUT FORMAT:
⚠️ / ✅ / ℹ️  [One-sentence verdict]

## [Section Title]
[Finding with inline citations like [1] or [2][3]]

## Conclusion
[Concise takeaway + recommended action if relevant, with citations]
"""


def _recommendation_prompt(question: str, sources_text: str, lang_instruction: str) -> str:
    return f"""You are a senior business analyst. The user wants a clear recommendation, not just a summary.

LANGUAGE: {lang_instruction}

SOURCES (numbered — cite every factual claim):
{sources_text}

QUESTION: {question}

INSTRUCTIONS:
1. Start with a clear verdict: "🏆 Recommended: [Option Name]" — one sentence, no hedging.
2. Write a ## Compared Options section with a mini scorecard for each option:
   - Name the option
   - List its 2-3 strongest points (cite each with [N])
   - List its 1-2 weakest points (cite each with [N])
3. Write a ## Why [Winner] section explaining exactly why this option wins over the others.
   - Be specific — reference actual data from sources, not generic praise.
   - Cite every claim.
4. Write a ## When to Consider [Runner-Up] section — under what conditions would the
   second-best option be the right choice instead? (This shows nuance, not indecision.)
5. Never write a raw URL in the answer body.
6. If no source supports a claim, do not make it.
7. End with a ## Bottom Line (2 sentences max): repeat the winner and the single most
   important reason to choose it.

OUTPUT FORMAT:
🏆 Recommended: [Option Name] — [one-sentence reason]

## Compared Options
### [Option A]
✅ [Strength 1] [1]
✅ [Strength 2] [2]
❌ [Weakness 1] [3]

### [Option B]
...

## Why [Winner]
[Specific reasoning with citations]

## When to Consider [Runner-Up]
[Conditions where runner-up wins, with citations]

## Bottom Line
[2-sentence summary of the recommendation]
"""


def _no_sources_message(lang: str) -> str:
    return {
        "ar": "لم أجد معلومات كافية للإجابة على هذا السؤال. يرجى تحميل وثيقة أو إعادة صياغة السؤال.",
        "fr": "Je n'ai pas trouvé suffisamment d'informations pour répondre. Veuillez charger un document ou reformuler votre question.",
        "en": "I could not find enough information to answer this question. Please upload a document or rephrase your query.",
    }.get(lang, "No relevant information found.")


def _rate_limit_message(lang: str) -> str:
    return {
        "ar": "⚠️ الخدمة مشغولة حالياً. يرجى الانتظار دقيقة أو دقيقتين والمحاولة مجدداً.",
        "fr": "⚠️ Le service est temporairement surchargé. Veuillez patienter 1-2 minutes et réessayer.",
        "en": "⚠️ The service is temporarily busy. Please wait 1-2 minutes and try again.",
    }.get(lang, "⚠️ Service temporarily unavailable. Please try again shortly.")


def _extract_recommendations(answer: str) -> tuple[list, str]:
    # Match either a complete ```json block or a truncated one
    pattern = r'```json\s*(\{.*?"recommendations".*?\})\s*```'
    match = re.search(pattern, answer, re.DOTALL)

    if match:
        try:
            data = json.loads(match.group(1))
            recommendations = data.get("recommendations", [])
            clean_answer = answer[:match.start()].rstrip()
            return recommendations, clean_answer
        except Exception:
            # JSON malformed — strip the block anyway
            clean_answer = answer[:match.start()].rstrip()
            return [], clean_answer

    # Also strip truncated/unclosed ```json blocks that never got a closing ```
    partial_pattern = r'```json\s*\{.*?"recommendations".*'
    partial_match = re.search(partial_pattern, answer, re.DOTALL)
    if partial_match:
        clean_answer = answer[:partial_match.start()].rstrip()
        return [], clean_answer

    return [], answer