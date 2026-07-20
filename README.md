<div align="center">

# 🤖 AI Document Intelligence & Deep Research

**PFE 2026 — End of Study Project**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?logo=langchain)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-E25822?logo=qdrant)](https://qdrant.tech/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

<p align="center">
  <b>Turn any business document into a conversational interface.</b><br>
  Upload invoices, contracts, or reports — ask in Arabic, French, or English — get precise, cited answers.
</p>

[Quick Start](#quick-start) · [Architecture](#architecture) · [API](#api-contract) · [Evaluation](#evaluation)

</div>

---

## Overview

Companies handle hundreds of business documents daily. Finding specific information is slow, manual, and error-prone.

This system solves that by making documents **queryable through natural language**:

- **Fast chat** — ask about anything inside your uploaded documents; get a cited answer in 1–2 seconds.
- **Deep research** — for questions beyond the document (tax rates, regulations, market data), an 8-agent pipeline searches the web and blends both sources into a single verified answer.
- **Zero prompt engineering** — users just ask. The system handles language detection, routing, and citation automatically.

Designed as two independent microservices behind an API Gateway, ready to drop into an existing company chatbot.

---

## Features

| Capability | Detail |
|---|---|
| 📄 **Multi-format ingestion** | PDF (native & scanned), DOCX, XLSX, images — auto-detects scanned vs. native PDF |
| 🔍 **3-level OCR fallback** | Tesseract → EasyOCR → Gemini Vision. Handles printed text, handwriting, and degraded scans in Arabic, French, and English |
| 💬 **Streaming RAG** | Token-by-token answers with inline source citations and per-session conversation memory |
| 🧠 **8-agent deep research** | LangGraph pipeline: Guardrail → Reformulator → Router → RAG + Web (parallel) → Re-Ranker → Synthesis → Citation Formatter |
| 🌍 **Trilingual** | Automatic language detection across Arabic, French, and English — no configuration needed |
| ⚡ **Production-ready** | Celery + Redis async tasks, two-tier caching, automatic LLM fallback (Gemini → Claude via OpenRouter), domain guardrail |
| 🐳 **One-command deploy** | `docker compose up --build` spins up the full stack with data persistence |

---

## Architecture

```
                    Company Chatbot
                           │
                           ▼
              ┌────────────────────────┐
              │    API Gateway (:8000) │
              │  auth · routing · rate │
              └───────────┬────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────────┐     ┌──────────────────────┐
│  Document Service    │     │     Q&A Service      │
│      (:8001)         │     │      (:8002)         │
│  upload · OCR · chunk│     │  fast chat · deep    │
│  · embed · index     │     │  research · Celery   │
└──────────┬───────────┘     └──────────┬───────────┘
           │                            │
           └────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐
        │     Shared Infrastructure     │
        │  Qdrant  ·  Redis  ·  Tavily  │
        └───────────────────────────────┘
```

**Two services, deliberately.** Separate containers, separate scaling, separate failure domains. A full 8-service production breakdown is scoped as a [roadmap](#known-limitations), not built here — keeping the system fully correct and tested within project scope.

### OCR Pipeline

```
Image or scanned page
         │
         ▼
    ┌─────────────┐   Tesseract v5 — fast, free, standard printed text (FR/EN/AR)
    │  Level 1    │   confidence < 60%
    └──────┬──────┘
           ▼
    ┌─────────────┐   EasyOCR — local, better Arabic & handwriting
    │  Level 2    │   confidence < 40%
    └──────┬──────┘
           ▼
    ┌─────────────┐   Gemini Vision API — best quality, degraded scans & handwriting
    │  Level 3    │
    └─────────────┘
```

### Deep Research Agent Graph (LangGraph)

```
User Query
    │
    ▼
┌──────────┐
│ Guardrail│──→ reject off-topic (zero API cost)
└────┬─────┘
     ▼
┌────────────┐
│Reformulator│──→ rewrite vague → precise retrieval query
└────┬───────┘
     ▼
┌────────┐
│ Router │──→ document only / web only / hybrid
└───┬────┘
    │
    ├──→ RAG Agent ──┐
    │   (Qdrant)     │   parallel execution
    │                ├→ Re-Ranker → Synthesis → Citation Formatter → Streaming Answer
    ├──→ Web Agent ──┘
        (Tavily)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Gateway & Services** | FastAPI |
| **Agent Orchestration** | LangGraph |
| **RAG Framework** | LangChain |
| **LLM** | Gemini 2.5 Flash |
| **Embeddings** | gemini-embedding-001 (3072-dim) |
| **Vector Store** | Qdrant |
| **Web Search** | Tavily API |
| **OCR** | Tesseract v5 → EasyOCR → Gemini Vision |
| **Async Tasks** | Celery + Redis |
| **LLM Fallback** | OpenRouter → Claude Haiku |
| **PDF** | PyMuPDF |
| **Word / Excel** | python-docx / openpyxl |
| **Language Detection** | langdetect + Arabic script heuristic |
| **Containerization** | Docker Compose |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- API keys: [Gemini](https://aistudio.google.com/app/apikey) · [Tavily](https://app.tavily.com)

### 1. Clone & Configure

```bash
git clone https://github.com/YOUR-USERNAME/pfe2026-microservices.git
cd pfe2026-microservices
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Launch

```bash
docker compose up --build
```

The full stack (6 containers: Gateway, Document Service, Q&A Service, Celery Worker, Redis, Qdrant) starts with one command. Existing Qdrant/Redis data is reused with zero loss.

### 3. Verify

| Endpoint | URL |
|---|---|
| Gateway | http://localhost:8000 |
| Health check | http://localhost:8000/health |
| Document Service docs | http://localhost:8001/docs |
| Q&A Service docs | http://localhost:8002/docs |

### Local Development (without Docker for services)

Requires Python 3.11 and [Tesseract v5](https://github.com/UB-Mannheim/tesseract/wiki) with FR + AR + EN language packs.

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Terminal 1 — infrastructure
docker compose up redis qdrant -d

# Terminal 2 — Gateway
cd gateway && uvicorn main:app --reload --port 8000

# Terminal 3 — Document Service
cd services/document && uvicorn main:app --reload --port 8001

# Terminal 4 — Q&A Service
cd services/qa && uvicorn main:app --reload --port 8002

# Terminal 5 — Celery Worker
celery -A services.qa.celery_app worker --loglevel=info
```

---

## API Contract

All endpoints are prefixed with `/api/v1`.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload and index a document (PDF, DOCX, XLSX, image) |
| `POST` | `/chat/ask` | Fast streaming chat with document-grounded answers |
| `POST` | `/research/query` | Submit a deep research question (async, returns `task_id`) |
| `GET` | `/research/status/{task_id}` | Poll for research result |

### Example: Upload a Document

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@invoice.pdf" \
  -F "session_id=session-001"
```

### Example: Fast Chat

```bash
curl -X POST "http://localhost:8000/api/v1/chat/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-001",
    "question": "What is the total amount on the invoice?"
  }'
```

### Example: Deep Research

```bash
# Submit
curl -X POST "http://localhost:8000/api/v1/research/query" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session-001",
    "question": "What are the current VAT rates in Morocco and how do they compare to the invoice?"
  }'
# → {"task_id": "abc-123"}

# Poll
curl "http://localhost:8000/api/v1/research/status/abc-123"
```

---

## Evaluation

The system was evaluated end-to-end using **RAGAS** (Retrieval-Augmented Generation Assessment):

- **Faithfulness** — answers are grounded in retrieved context, not hallucinated.
- **Answer Relevance** — responses directly address the question asked.
- **Context Precision** — retrieved chunks are relevant and minimal.

Tested document formats: DOCX, native PDF, scanned PDF (OCR), XLSX.

---

## Verified Working

Tested end-to-end through both the API directly and the live frontend:

- ✅ Full stack (6 containers) starts cleanly with one command, reusing existing Qdrant/Redis data with zero loss
- ✅ Upload → immediate question → correct answer, proving lazy indexing is correctly flushed before retrieval
- ✅ Deep research correctly blends document data with live web sources, verified figures, per-source citations
- ✅ Fast chat streaming works end-to-end in the UI, with source citations rendered correctly
- ✅ Multiple document formats tested (DOCX, native PDF, scanned PDF via OCR)

**Bugs found and fixed during testing:**
1. Duplicated URL prefix causing failed uploads — resolved
2. Missing streaming route in Gateway causing chat to hang silently — resolved
3. Truncated document summaries caused by Gemini's internal "thinking" tokens exhausting the output budget — resolved

---

## Known Limitations

| Limitation | Detail |
|---|---|
| **PII scrubbing** | Uses a lightweight fallback unless `presidio-analyzer` / `presidio-anonymizer` are installed |
| **Service granularity** | A full 8-service production breakdown (separate RAG, Guardrail, Web Search services) is scoped and documented as a roadmap, not built in this delivery |

---

## Project Status

| Phase | Status |
|---|---|
| Document ingestion, OCR, vector storage | ✅ |
| RAG service + chat memory | ✅ |
| Multi-agent deep research (LangGraph) | ✅ |
| Domain guardrail | ✅ |
| Microservices packaging (Docker Compose) | ✅ |
| RAGAS evaluation | ✅ |
| Final testing & delivery | ✅ |

---



---

<div align="center">

Built with ❤️ for PFE 2026

</div>
