# Agentic RAG E-Government Chatbot (FastAPI backend)

A production-oriented **Agentic Retrieval-Augmented Generation** backend for a
Malaysian public-sector chatbot (LHDN taxation, KWSP/EPF provident fund). It acts
as a **semantic translation layer**: it ingests bureaucratic policy documents and
answers citizens' questions in plain, conversational language.

All user interaction happens over **WhatsApp**, bridged by an **n8n** workflow
that forwards messages to this backend via secure JSON webhooks and delivers the
agent's reply back to the user.

All AI models are accessed via **remote APIs** — no local inference, no
`torch`/`transformers`, no GPU.

---

## Table of contents
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Database migration](#database-migration)
- [Running the service](#running-the-service)
- [Data ingestion toolkit](#data-ingestion-toolkit)
- [Embedding convention](#embedding-convention-critical)
- [Agent design](#agent-design)
- [Configuration](#configuration)

---

## Architecture

```
WhatsApp user ⇄ n8n ──webhook──▶ FastAPI
                                   │  X-Webhook-Token auth
                                   │  payload-size guard
                                   │  Redis SETNX dedupe (before dispatch)
                                   │  → HTTP 202 Accepted
                                   ▼
                             Redis (broker + dedupe store)
                                   │
                             Celery worker
              download media (service-role key) → route by type
                 audio → Qwen ASR    image → Qwen3-VL OCR    text → passthrough
              → hydrate conversation memory from DB
              → LangGraph agent
              → persist answer + retrieved_chunk_ids
              → outbound callback to n8n (retry/backoff + dead-letter log)
```

LangGraph flow:

```
guard1 ──harmful/out-of-scope──▶ blocked ─▶ END
   │ ok
   ▼
retrieve ──zero docs──▶ fallback ─▶ END        (Guardrail 2: anti-hallucination)
   │ chunks
   ▼
generate ─▶ validate ──grounded──▶ finalize ─▶ END
                │ not grounded
                ├─ retry_count >= 1 ─▶ fallback   (bounded: never loops forever)
                └─ else ─▶ retrieve               (exactly one retry)
```

---

## Tech stack

| Layer | Choice |
|---|---|
| API | FastAPI + Uvicorn |
| Async workers | Celery + Redis (broker) |
| Database | Supabase (PostgreSQL + `pgvector`) |
| Orchestration | LangGraph (stateful, conditional routing, bounded retry) |
| Reasoning / agent | Google Gemini (`langchain-google-genai`) |
| Translation + Vision/OCR | Qwen3-VL & Qwen ASR via Alibaba DashScope (HTTP) |
| Embeddings | `intfloat/multilingual-e5-large-instruct`, **1024-dim**, Alibaba API primary + HuggingFace fallback |
| Ingestion | Playwright + BeautifulSoup, `RecursiveCharacterTextSplitter`, pypdf + PyMuPDF |
| Observability | Langfuse tracing (DeepEval/UpTrain offline — see `requirements-eval.txt`) |

---

## Project structure

```
main.py                     FastAPI app (/health, /webhook/whatsapp)
api/                        routes (webhook auth + idempotency + 202 dispatch)
core/                       settings, Redis client + idempotency, logging
services/                   embed() helper, Qwen ASR, Qwen3-VL, Gemini,
                            Supabase data access, private-bucket download,
                            n8n outbound callback (retry + DLQ)
agent/                      LangGraph state, hybrid_search tool, nodes, graph
worker/                     Celery app + task pipeline
migrations/001_init.sql     schema: extensions → tables → simple tsvector + GIN
                            → HNSW → hybrid_search RRF RPC
scripts/                    data ingestion toolkit (see below)
knowledge/                  hand-curated *.md docs for non-scrapable content
requirements.txt            API/worker runtime deps (no torch/transformers)
requirements-eval.txt       offline eval libs (deepeval) — heavy, optional
scripts/requirements-ingest.txt   ingestion-only deps (Playwright, pypdf, ...)
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate

# API / worker runtime
pip install -r requirements.txt

# Ingestion toolkit (only if you will populate the DB)
pip install -r scripts/requirements-ingest.txt
playwright install chromium

cp .env.example .env        # then fill in keys
```

---

## Database migration

Apply [`migrations/001_init.sql`](migrations/001_init.sql) in the Supabase SQL
editor (or via `psql`) **before** running the app or ingesting data. Ordering is
load-bearing: extensions → tables (with `document_id`/`chunk_index`) → generated
`simple` `tsvector` + GIN index → HNSW index → parameterized `hybrid_search` RRF
function.

Notes:
- `fts` uses the **`simple`** config (not `english`) so the stemmer doesn't
  butcher Bahasa Melayu terms (e.g. "kewangan") and ruin LHDN/KWSP keyword search.
- `hybrid_search` fuses vector + FTS results with **Reciprocal Rank Fusion in the
  database**; RRF weights and the `k` constant are function parameters, so search
  balance can be retuned without a migration.

---

## Running the service

```bash
# API (separate shell)
uvicorn main:app --reload

# Celery worker (separate shell)
celery -A worker.celery_app.celery worker --loglevel=info
```

`GET /health` returns liveness + Redis reachability.
`POST /webhook/whatsapp` requires the `X-Webhook-Token` header.

---

## Data ingestion toolkit

Four scripts populate the same `embeddings` table. All are **idempotent per URL**
(delete-by-`file_url` before insert), reuse the **same `embed_passage()` helper**
(so ingest/query embeddings never drift), and support `--preview` (no DB writes).
Run them from the project root.

| Script | Source | Use for |
|---|---|---|
| `scripts/list_links.py` | a landing/card page | **discover** destination URLs to scrape |
| `scripts/ingest_scrape.py` | live web pages | KWSP/LHDN FAQ & content pages |
| `scripts/ingest_pdf.py` | PDFs | **user manuals / guides / explainers** |
| `scripts/ingest_docs.py` | `knowledge/*.md` | content not in page text (rules in JS, transcribed policy) |

### What to ingest vs. what to just link

- **Web page** → scrape it (`ingest_scrape`). It carries both the explanations
  *and* any form download links (links are preserved inline during scraping).
- **User manual / guide PDF** → embed it (`ingest_pdf`). It holds knowledge the
  agent must reason over.
- **Fillable form PDF** (e.g. KWSP 9KM) → **do not embed**. It's just field
  labels. The agent surfaces its **download link** from the referring web page.
- **Rules buried in JavaScript / transcribed policy** → curate a `knowledge/*.md`
  (verify it), then `ingest_docs`.

### Typical workflow

```bash
# 1. discover card URLs on a landing page (deduped, /en/-canonical)
python -m scripts.list_links --url "https://www.kwsp.gov.my/en/member/life-stages"

# 2. paste the good URLs into TARGET_URLS in scripts/ingest_scrape.py, then preview
python -m scripts.ingest_scrape --preview

# 3. upload
python -m scripts.ingest_scrape

# PDFs: paste into PDF_URLS in scripts/ingest_pdf.py
python -m scripts.ingest_pdf --preview
python -m scripts.ingest_pdf
```

### Scraping notes (KWSP/LHDN behind a WAF)

- **Fresh browser context per URL.** The WAF flags a session after its first
  request, so each page gets a clean cookie jar to look like a first-time visitor.
- **Cloudflare JS challenge / forced PDF downloads** are handled by driving a real
  browser (`page.goto` / download event), not plain HTTP clients (which get 403).
- **Cleaning before chunking:** strips nav/UI noise + zero-width chars and
  de-duplicates repeated blocks (KWSP renders the same FAQ accordion several times).
- **OCR fallback:** image-only PDF pages are rendered with PyMuPDF and OCR'd via
  Qwen3-VL (needs `DASHSCOPE_API_KEY`; skip with `--no-ocr`).

---

## Embedding convention (CRITICAL)

`multilingual-e5-large-instruct` requires **asymmetric prefixing**. ALL embedding
calls go through [`services/embeddings.py`](services/embeddings.py) so the
ingestion path and query path can never drift apart:

- **Query:** `Instruct: {task_description}\nQuery: {text}`
- **Passage / stored document:** raw text, **no prefix**.
- Output is enforced to exactly **1024 dimensions** to match `vector(1024)`.
- Primary backend: Alibaba embedding API; fallback: HuggingFace Inference API
  (HF has cold-start/503 issues for this model).

---

## Agent design

- **Conversation memory** is hydrated from the `message`/`conversation` tables at
  the start of each task (each WhatsApp message is a separate webhook → separate
  task → fresh state, so memory is loaded explicitly from the DB).
- **Guardrail 1 (pre-filter):** blocks harmful / out-of-scope content *before*
  retrieval (Gemini structured-output classifier; fails safe).
- **Guardrail 2 (empty retrieval):** if `hybrid_search` returns zero docs, a
  strict fallback states the bot lacks that specific policy (anti-hallucination) —
  kept distinct from Guardrail 1.
- **Bounded retry:** the validate→retrieve edge checks `retry_count`; at `>= 1` it
  forces an exit. Never an unbounded loop.
- **Inline validation:** a lightweight Gemini groundedness check gates the single
  retry. Heavier DeepEval/UpTrain context-precision/recall run **offline** over the
  stored `retrieved_chunk_ids` + `useful` feedback.
- **Reply style:** mirrors the user's language (BM↔EN), simplifies jargon, formats
  for a WhatsApp mobile screen, and includes official **form/document download
  links** that appear in the retrieved context (never invented).

---

## Configuration

All settings load from `.env` via `core/config.py` (pydantic-settings). See
[`.env.example`](.env.example) for the full list: Gemini, DashScope, embedding API
+ HF fallback, Supabase URL + service-role key, Redis URL, `X-Webhook-Token`, n8n
callback URL, Langfuse keys.

> The Supabase **service-role key** is server-side only — never expose it to clients.

---
