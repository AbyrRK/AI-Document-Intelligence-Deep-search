# AI Document Intelligence & Deep Research
**PFE 2026 — End of Study Project**

---

## What This Project Does

Companies deal with hundreds of business documents — invoices, contracts, quotes, reports. Finding specific information inside them is slow and manual.

This project solves that by turning any business document into a conversational interface. Upload a document, ask a question in Arabic, French, or English, and get a precise answer with the exact source cited. No prompt engineering required from the user. No hallucinations.

For questions that go beyond the uploaded documents — current tax rates, new regulations, market data — the system automatically searches the web and combines both sources into a single cited answer.

---

## What Was Built

### 1. Document Ingestion Pipeline
Handles any business document format: PDF (native or scanned), Word, Excel, and images.

- Detects whether a PDF is native text or a scanned image and routes accordingly
- 3-level OCR fallback: Tesseract → EasyOCR → Gemini Vision, so even handwritten Arabic documents are handled
- Smart page selection (brief / summary / full) so users get a response in under 5 seconds regardless of document size
- Language detection works across Arabic, French, and English without configuration
- All text is chunked, embedded, and stored in Qdrant with full metadata (session, document ID, language, page number)

### 2. RAG Service
Retrieves the most relevant chunks from the vector database and generates a grounded answer using Gemini 2.5 Flash. Answers are streamed token by token. Conversation history is maintained per session so follow-up questions work naturally.

### 3. Multi-Agent Deep Research (LangGraph)
The most complex part of the system. A directed graph of 8 specialized agents that each do one thing:

| Agent | Role |
|---|---|
| **Guardrail** | Rejects off-topic questions before any API call is made |
| **Reformulator** | Rewrites vague user questions into precise retrieval queries |
| **Router** | Decides whether to use documents, web, or both |
| **RAG Agent** | Searches Qdrant for relevant document chunks |
| **Web Agent** | Searches the web via Tavily, with academic source support |
| **Re-Ranker** | Merges and scores results from both sources |
| **Synthesis** | Generates a structured cited answer using Gemini |
| **Citation Formatter** | Validates every citation, removes any hallucinated sources |

RAG and Web agents run **in parallel** for hybrid queries, cutting response time roughly in half.

### 4. Production Infrastructure
- **Celery + Redis** — deep research tasks run asynchronously, the frontend polls for results without blocking
- **Two-tier cache** — Redis shared across workers + local LRU per process, reducing redundant Gemini API calls
- **Automatic LLM fallback** — when Gemini hits rate limits, the system switches to Claude Haiku via OpenRouter transparently
- **Docker Compose** — the full stack (API, Celery worker, Redis, Qdrant) starts with one command
- **Domain guardrail** — the system refuses non-business questions immediately, keeping the assistant focused

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                       │
│                                                             │
│   POST /documents/upload   POST /chat/ask   POST /research/query  │
└──────────────┬──────────────────┬──────────────────┬────────┘
               │                  │                  │
               ▼                  ▼                  ▼
   ┌───────────────────┐  ┌──────────────┐  ┌──────────────────────┐
   │ Document Pipeline │  │ RAG Service  │  │ Deep Research Agent  │
   │                   │  │              │  │ (LangGraph)          │
   │ Extract           │  │ Embed query  │  │                      │
   │ Detect language   │  │ Search Qdrant│  │ Guardrail            │
   │ Chunk             │  │ Generate     │  │ → Reformulator       │
   │ Embed             │  │ Stream       │  │ → Router             │
   │ → Qdrant          │  │              │  │ → RAG ──┐            │
   └───────────────────┘  └──────────────┘  │ → Web ──┴→ Re-Rank  │
                                            │ → Synthesize         │
                                            │ → Cite → Stream      │
                                            └──────────────────────┘
               │                  │                  │
               └──────────────────┴──────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │         Qdrant            │  Vector storage
                    │         Redis             │  Cache + Celery
                    │         Gemini API        │  LLM + Embeddings
                    │         Tavily API        │  Web search
                    └───────────────────────────┘
```

### OCR — 3-Level Fallback
```
Image or scanned page
         │
         ▼
    Tesseract v5          Fast, free — handles standard printed text (FR/EN/AR)
         │  confidence < 60%
         ▼
    EasyOCR               Local, free — better Arabic and handwriting
         │  confidence < 40%
         ▼
    Gemini Vision API     Best quality — handles degraded scans and handwriting
```

---

## Tools & Models

| Category | Choice |
|---|---|
| API Framework | FastAPI |
| Agent Orchestration | LangGraph |
| RAG Framework | LangChain |
| LLM | Gemini 2.5 Flash |
| Embeddings | gemini-embedding-001 (3072 dimensions) |
| Vector Database | Qdrant |
| Web Search | Tavily API |
| OCR Level 1 | Tesseract v5 — FR + EN + AR |
| OCR Level 2 | EasyOCR |
| OCR Level 3 | Gemini Vision API |
| Background Tasks | Celery + Redis |
| LLM Fallback | OpenRouter → Claude Haiku |
| PDF Extraction | PyMuPDF |
| Word Extraction | python-docx |
| Excel Extraction | openpyxl |
| Language Detection | langdetect + Arabic script heuristic |

---

## How to Run

### Docker — recommended

```bash
git clone https://github.com/AbyrRK/AI-Document-Intelligence-Deep-search.git
cd AI-Document-Intelligence-Deep-search
cp .env.example .env        # fill in API keys (see below)
docker compose up --build
```

| | URL |
|---|---|
| API | http://localhost:8000 |
| Interactive docs | http://localhost:8000/docs |

### Local Development

**Requirements:** Python 3.11, Docker Desktop, [Tesseract v5](https://github.com/UB-Mannheim/tesseract/wiki) with FR + AR + EN packs

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

docker compose up redis qdrant -d       # start infrastructure only
uvicorn app.main:app --reload           # terminal 1
celery -A app.celery_app worker         # terminal 2
python frontend/server.py               # terminal 3 → http://localhost:3000
```

### API Keys needed

| Variable | Get it at |
|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/app/apikey |
| `TAVILY_API_KEY` | https://app.tavily.com |
| `OPENROUTER_API_KEY` | https://openrouter.ai *(optional — LLM fallback)* |

---

## Tests

```bash
python -m pytest tests/ -v
```

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| 1 | Environment setup | ✅ |
| 2 | Document ingestion pipeline | ✅ |
| 3 | OCR + text processing + vector storage | ✅ |
| 4 | RAG service + chat memory | ✅ |
| 4.5 | Prompt engineering evaluation | ✅ |
| 5 | Multi-agent deep research | ✅ |
| 6 | Domain guardrail | ✅ |
| 7 | Docker + microservice packaging |  |