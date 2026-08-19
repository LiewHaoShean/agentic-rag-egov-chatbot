-- ============================================================================
--  Agentic RAG E-Government Chatbot — initial schema
--  Target: Supabase (PostgreSQL + pgvector)
--
--  ORDERING IS LOAD-BEARING:
--    1. extensions  -> 2. tables (with document_id/chunk_index)
--    -> 3. generated tsvector (simple) + GIN -> 4. HNSW -> 5. hybrid_search RPC
-- ============================================================================

-- 1) ---------------------------------------------------------------- EXTENSIONS
--    Must come BEFORE any vector(1024) column or tsvector usage.
CREATE EXTENSION IF NOT EXISTS vector;     -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;    -- trigram fuzzy matching (optional FTS aid)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 2) -------------------------------------------------------------------- TABLES

-- ---- USER ----
CREATE TABLE IF NOT EXISTS "user" (
    user_id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name            TEXT,
    email           TEXT UNIQUE,
    phone_number    TEXT UNIQUE,            -- WhatsApp identity
    hashed_password TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---- CONVERSATION ----
CREATE TABLE IF NOT EXISTS conversation (
    conversation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL REFERENCES "user"(user_id) ON DELETE CASCADE,
    title           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversation_user ON conversation(user_id);

-- ---- MESSAGE ----
CREATE TABLE IF NOT EXISTS message (
    message_id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id     UUID NOT NULL REFERENCES conversation(conversation_id) ON DELETE CASCADE,
    role                TEXT NOT NULL,                 -- 'user' | 'assistant' | 'system'
    message             TEXT,
    translate_message   TEXT,                          -- normalized text
    language            TEXT,
    -- Soft link: ties an answer to the exact Embeddings chunks used.
    -- Postgres cannot FK a UUID[] — accepted as an audit-trail soft link
    -- powering DeepEval/UpTrain context precision/recall.
    retrieved_chunk_ids UUID[] DEFAULT '{}',
    useful              BOOLEAN,                        -- Good(true)/Bad(false) eval feedback
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_message_conversation ON message(conversation_id);
CREATE INDEX IF NOT EXISTS idx_message_created ON message(created_at);

-- ---- EMBEDDINGS ----
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id          UUID REFERENCES "user"(user_id) ON DELETE SET NULL,
    message_id       UUID REFERENCES message(message_id) ON DELETE SET NULL,  -- nullable
    -- Chunk grouping: all chunks of one parent doc share document_id; prevents
    -- chunk isolation and lets the agent fetch adjacent context via chunk_index.
    document_id      UUID NOT NULL,
    chunk_index      INTEGER NOT NULL DEFAULT 0,
    embedding_vector vector(1024) NOT NULL,            -- e5-large-instruct, 1024 dims
    translate_text   TEXT,                             -- normalized
    original_text    TEXT,
    summary          TEXT,
    current_language TEXT,
    file_url         TEXT,
    category         TEXT,                             -- e.g. 'tax', 'epf', 'immigration'
    public           BOOLEAN NOT NULL DEFAULT TRUE,    -- public/private visibility
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- 3) GENERATED tsvector — 'simple' config (NOT 'english').
    -- The English stemmer butchers Bahasa Melayu terms ("kewangan") and ruins
    -- LHDN/KWSP keyword search, so we use 'simple' to keep tokens intact.
    fts tsvector GENERATED ALWAYS AS (
        to_tsvector(
            'simple',
            coalesce(translate_text, '') || ' ' || coalesce(original_text, '')
        )
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_embeddings_document ON embeddings(document_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_embeddings_category ON embeddings(category);

-- 3b) GIN index on the generated FTS column.
CREATE INDEX IF NOT EXISTS idx_embeddings_fts ON embeddings USING GIN (fts);

-- 4) HNSW index on the vector column (no IVFFlat lists-tuning headache).
--    Cosine distance operator class.
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 5) ----------------------------------------------------- HYBRID SEARCH RPC
-- Reciprocal Rank Fusion fully in-DB (NOT fused in Python).
-- RRF weights and the k constant are FUNCTION PARAMETERS so search balance can
-- be retuned without a migration — important because 'simple' makes FTS stricter,
-- so the vector channel weight should stay tunable.
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text       TEXT,
    query_embedding  vector(1024),
    match_count      INTEGER DEFAULT 8,
    rrf_k            INTEGER DEFAULT 60,     -- RRF smoothing constant
    weight_vector    FLOAT   DEFAULT 1.0,    -- vector channel weight
    weight_fts       FLOAT   DEFAULT 1.0,    -- keyword channel weight
    filter_category  TEXT    DEFAULT NULL,   -- optional category scope
    only_public      BOOLEAN DEFAULT TRUE,   -- restrict to public docs
    candidate_pool   INTEGER DEFAULT 50      -- per-channel candidate depth
)
RETURNS TABLE (
    embedding_id   UUID,
    document_id    UUID,
    chunk_index    INTEGER,
    translate_text TEXT,
    original_text  TEXT,
    summary        TEXT,
    category       TEXT,
    file_url       TEXT,
    rrf_score      FLOAT
)
LANGUAGE sql
STABLE
AS $$
WITH
-- Vector channel: rank by cosine distance (smaller = closer).
vector_ranked AS (
    SELECT
        e.embedding_id,
        ROW_NUMBER() OVER (
            ORDER BY e.embedding_vector <=> query_embedding
        ) AS rank
    FROM embeddings e
    WHERE (filter_category IS NULL OR e.category = filter_category)
      AND (NOT only_public OR e.public IS TRUE)
    ORDER BY e.embedding_vector <=> query_embedding
    LIMIT candidate_pool
),
-- Keyword channel: rank by ts_rank over the GIN-indexed 'simple' fts.
fts_ranked AS (
    SELECT
        e.embedding_id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank(e.fts, websearch_to_tsquery('simple', query_text)) DESC
        ) AS rank
    FROM embeddings e
    WHERE e.fts @@ websearch_to_tsquery('simple', query_text)
      AND (filter_category IS NULL OR e.category = filter_category)
      AND (NOT only_public OR e.public IS TRUE)
    ORDER BY ts_rank(e.fts, websearch_to_tsquery('simple', query_text)) DESC
    LIMIT candidate_pool
),
-- Reciprocal Rank Fusion: sum weighted 1/(k+rank) across channels.
fused AS (
    SELECT
        coalesce(v.embedding_id, f.embedding_id) AS embedding_id,
        coalesce(weight_vector / (rrf_k + v.rank), 0.0)
          + coalesce(weight_fts / (rrf_k + f.rank), 0.0) AS rrf_score
    FROM vector_ranked v
    FULL OUTER JOIN fts_ranked f USING (embedding_id)
)
SELECT
    e.embedding_id,
    e.document_id,
    e.chunk_index,
    e.translate_text,
    e.original_text,
    e.summary,
    e.category,
    e.file_url,
    fused.rrf_score
FROM fused
JOIN embeddings e ON e.embedding_id = fused.embedding_id
ORDER BY fused.rrf_score DESC
LIMIT match_count;
$$;
