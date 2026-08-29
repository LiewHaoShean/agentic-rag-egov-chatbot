# TODO — Outstanding Work

Agentic RAG e-government assistant (MESRA). Status as of **20 August 2026**.
Grouped by what actually blocks the FYP deliverable, not by subsystem.

Search the codebase for `TODO` / `# >>> PENDING` to jump to exact lines.

---

## 🔴 Report-critical — these gate Chapter 5 and the demo

### 1. Offline evaluation harness (DeepEval)
Nothing built yet. This is the single highest-priority item: Chapter 5 requires
quantitative results, and `message.retrieved_chunk_ids` has been accumulating
since day one precisely so this can run retrospectively.

- [ ] Batch job reading `message.retrieved_chunk_ids` + `message.useful`
- [ ] DeepEval: context precision / recall, faithfulness, answer relevancy
- [ ] Build a fixed question set (spread across epf / tax / socso / identity,
      and across EN / BM / ZH) so results are reproducible in the report
- [ ] Report sink — CSV or Markdown table that can be pasted into Chapter 5
- [ ] Decision: DeepEval only. UpTrain stays commented in `requirements.txt`.

### 2. Deployment — GCP VM via Docker Compose
Currently a 4-terminal local stack behind a cloudflared tunnel, which is fragile
for a live demo and cannot be screenshotted as a deployed system.

- [ ] Dockerfile + compose for api / worker / redis
- [ ] Deploy to GCP VM (trial credit), persistent public webhook URL
- [ ] Point the n8n webhook at the deployed host, retire the tunnel
- [ ] Verify the full multimodal loop end-to-end after deployment

### 3. Vertex AI quota
429 "resource exhausted" appears under normal load (up to 4 LLM calls/message).
Not yet mitigated — a demo-day rate limit would be highly visible.

- [ ] Request a quota increase, or move `VERTEX_LOCATION` off `us-central1`

### 4. Tests
No `tests/` directory exists. Chapter 5's unit-testing table currently has no
automated backing.

- [ ] Unit tests for the pure logic: `_reply_language()`, `_needs_rewriting()`,
      `_format_input()` (e5 asymmetry), `route_after_validate()` bounds
- [ ] Integration test: webhook → Celery → agent → callback with mocked externals

---

## 🟡 Verify / clean up before submission

- [ ] **HF embedding fallback is near-dead code.** `_embed_hf()` is now only
      reachable on a 5xx, and has never been verified against
      `multilingual-e5-large-instruct`. Either verify it or delete it — an
      unverified fallback documented in Chapter 4 is worse than none.
  - File: [services/embeddings.py](services/embeddings.py)
- [ ] **Langfuse has never emitted a trace.** The handler exists but `.env` has
      no keys, so `_langfuse_handler()` returns `None` on every call. Decide:
      configure it and add task-level grouping (one trace per `message_id`), or
      drop it from the report's architecture description.
  - File: [services/gemini.py](services/gemini.py)
- [ ] Re-capture blocked/fallback screenshots — the messages now name four
      agencies, older screenshots show the KWSP/LHDN-only text
- [ ] Verify Malay query-rewrite preserves language (blocked on Vertex 429s)

---

## 🟢 Future work — Chapter 6.3 Recommendations, not build items

Genuine improvements, but out of scope for the remaining timeline. Written up as
recommendations rather than left as unfinished work.

- [ ] Retry currently re-runs **identical** retrieval. Query rewriting runs once
      before the first retrieval; the retry edge re-uses the stored
      `search_query` by design. Widening `match_count` or reformulating on retry
      is the natural next step.
  - File: [agent/nodes.py](agent/nodes.py) (`route_after_validate`)
- [ ] Tune `hybrid_search` RRF weights for `'simple'` FTS strictness — currently
      1.0/1.0. Weights are already RPC params, so no migration is needed.
  - File: [migrations/001_init.sql](migrations/001_init.sql)
- [ ] Category routing: `filter_category` is plumbed through `retrieve()` but is
      always `None`. Letting the guardrail scope retrieval to one agency would
      cut cross-agency bleed.
  - File: [agent/tools.py](agent/tools.py)
- [ ] Adjacent-context fetch via `document_id` / `chunk_index` — the RPC returns
      matched chunks only, so a fact split across a chunk boundary can be halved
- [ ] Per-chunk `summary` is left NULL throughout the corpus
- [ ] Conversation memory window (`load_recent_messages` limit = 20) — untuned
- [ ] DLQ replayer for `callback_dlq.log` — writes happen, nothing re-delivers
  - File: [services/n8n_callback.py](services/n8n_callback.py)
- [ ] Rate limiting / abuse protection on the webhook (token auth + a payload
      size cap exist; neither limits request rate)

---

## ✅ Completed

Kept for the report's implementation narrative.

**Knowledge base** — `scripts/ingest_scrape.py`: Playwright scrape + cleaning,
`RecursiveCharacterTextSplitter` (1000/100), deterministic uuid5 `document_id`,
`embed_passage()` with no instruct prefix, idempotent delete-by-`file_url`.
**1,557 chunks from 245 pages** across four agencies (epf 555, tax 449,
socso 224, identity 329). MyGov nav-noise stripping rules derived via `--preview`.

**Retrieval** — `hybrid_search` RPC: pgvector HNSW + GIN `tsvector` (`'simple'`),
fused by RRF (k=60, pool 50, top 8) entirely in-DB.

**Agent** — LangGraph: guard1 → rewrite → retrieve → generate → validate →
finalize, plus blocked / fallback / busy / meta terminals. Bounded single retry,
`recursion_limit=12`.

**Multimodal** — text, voice (Qwen3-ASR + domain-vocab biasing), image
(Qwen3-VL OCR), PDF (`services/pdf_text.py`, per-page OCR fallback, 10-page /
8k-char caps), graceful rejection of other types.

**Multilingual** — EN / BM / ZH / TA, four-stage `_reply_language()` cascade.

**Fixed defects** — LangGraph edge-mutation retry loop; broken conversation
memory (`get_or_create_conversation` inserted unconditionally); Malay-greeting
misdetection; n8n boolean status-callback filter. All four are Chapter 5 material.

**Infra** — Redis SETNX idempotency, HTTP 202 async dispatch, Celery retry →
busy message, Meta System User permanent token, n8n typing indicator + read
receipts, repo on GitHub (private).

**Verified** — DashScope response envelopes for ASR and VL (the paths originally
guessed in this file were wrong; corrected in code). Alibaba embeddings return
exactly 1024 dims, enforced by `_enforce_dims()`.
