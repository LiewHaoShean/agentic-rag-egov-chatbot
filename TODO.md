# TODO — Outstanding Work

Tracking items not yet finalized in the Agentic RAG e-government backend.
Grouped by priority. Search the codebase for `TODO` / `# >>> PENDING` to jump
to the exact lines.

---

## 🔴 Blocking — verify before production traffic

### 1. Qwen DashScope response envelopes
The JSON paths used to pull text out of the DashScope responses are best-guess
and must be confirmed against live responses for the chosen models.

- [ ] **Qwen ASR** — confirm `data["output"]["text"]` for `qwen3-asr-flash`
  - File: [services/qwen_asr.py](services/qwen_asr.py) (`transcribe`)
- [ ] **Qwen3-VL** — confirm `data["output"]["choices"][0]["message"]["content"][0]["text"]` for `qwen3-vl-plus`
  - File: [services/qwen_vision.py](services/qwen_vision.py) (`extract_text`)
- [ ] Confirm the correct DashScope endpoint paths + base URL (intl vs cn region)
  - `DASHSCOPE_BASE_URL` in [.env.example](.env.example)

### 2. Embedding API response shape
- [ ] Confirm Alibaba `text-embedding-v3` accepts the `dimensions: 1024` param and
      returns `data["data"][0]["embedding"]` at exactly 1024 dims
  - File: [services/embeddings.py](services/embeddings.py) (`_embed_alibaba`)
- [ ] Confirm HF Inference API fallback returns a flat 1024-dim vector for
      `multilingual-e5-large-instruct`
  - File: [services/embeddings.py](services/embeddings.py) (`_embed_hf`)

---

## 🟡 Ingestion pipeline — DONE (verify against live sites)

### 3. Document ingestion script — ✅ `scripts/ingest_scrape.py`
Web scraper rewritten for the new `embeddings` schema (ported from the old
`rag-scapper/targeted_scraper.py`). Run with `python -m scripts.ingest_scrape`
(add `--preview` for a dry run). Deps in `scripts/requirements-ingest.txt`.

- [x] Load source policy docs (LHDN/KWSP) via Playwright + cleaning
- [x] `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=100)
- [x] Per page: deterministic `document_id` (uuid5), sequential `chunk_index`
- [x] Embed each chunk via `embed_passage()` (NO instruct prefix — passage path)
- [x] Insert into `embeddings` with `category`, `public`, `original_text`,
      `translate_text`, `current_language`, `file_url`
- [x] Idempotent re-runs (delete-by-`file_url` before insert)
- [ ] **Verify** chunk quality on live pages (`--preview`) before bulk upload
- [ ] `summary` left NULL — add per-chunk/per-doc summarization if needed
- [ ] Expand `TARGET_URLS` (LHDN tax pages currently commented out)

---

## 🟢 Eval & observability follow-ups

### 4. Offline evaluation harness (DeepEval / UpTrain)
Inline validation is a lightweight Gemini groundedness check. The heavier eval
runs offline over stored data.

- [ ] Batch job reading `message.retrieved_chunk_ids` + `message.useful`
- [ ] DeepEval context precision / recall + faithfulness + answer relevancy
- [ ] Schedule (cron) + report sink
- [ ] Decide UpTrain vs DeepEval (currently DeepEval pinned; UpTrain commented in
      [requirements.txt](requirements.txt))

### 5. Langfuse tracing verification
- [ ] Confirm traces appear end-to-end across guardrail → generate → validate
- [ ] Add task-level trace grouping (one trace per WhatsApp message_id)

---

## ⚙️ Hardening / nice-to-have

- [ ] `validate_node` retry currently only re-runs identical retrieval — consider
      query reformulation or widening `match_count` on the retry
  - File: [agent/nodes.py](agent/nodes.py) (`route_after_validate`)
- [ ] Tune `hybrid_search` RRF weights for `simple` FTS strictness (weights are
      already RPC params — no migration needed)
  - File: [migrations/001_init.sql](migrations/001_init.sql)
- [ ] Adjacent-context fetch via `document_id`/`chunk_index` (spec mentions
      fetching neighbouring chunks; RPC currently returns matched chunks only)
- [ ] Conversation memory window size (`load_recent_messages` limit = 20) — tune
- [ ] DLQ replayer for `callback_dlq.log` (re-deliver failed n8n callbacks)
  - File: [services/n8n_callback.py](services/n8n_callback.py)
- [ ] Rate limiting / abuse protection on the webhook
- [ ] Unit + integration tests (none yet)
- [ ] `category` routing — let guardrail/agent scope retrieval to `tax` vs `epf`
