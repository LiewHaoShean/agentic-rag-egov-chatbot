-- 002: OR-semantics keyword channel (hybrid_search_v2)
--
-- WHY
-- hybrid_search builds its lexical query with websearch_to_tsquery('simple',...),
-- which joins EVERY term with AND. A chunk therefore matches only if it contains
-- every word of the question, stopwords included. Measured against the 30-question
-- evaluation set, only 2 of 30 questions produced ANY lexical match, so retrieval
-- was effectively dense-only and the RRF fusion was summing a channel that was
-- almost always zero.
--
-- Switching to OR alone is not enough. The fts column is generated with the
-- 'simple' configuration, which removes no stopwords, so an OR query would match
-- nearly every chunk on 'the' or 'yang'. The stopwords must be dropped first.
--
-- 'simple' is retained deliberately. The original schema comment records that the
-- English stemmer damages Malay terms such as "kewangan", and that reasoning still
-- holds, so stemming stays off and only stopword removal is added.
--
-- SAFETY
-- Nothing here modifies hybrid_search, the embeddings table, or any index. This
-- migration only ADDS two functions. To roll back, see 002_rollback.sql, which
-- simply drops them.

-- ---------------------------------------------------------------- stopwords
-- Held in SQL rather than a Postgres stopword file, because hosted Supabase gives
-- no filesystem access to install one, and because the corpus is multilingual —
-- an 'english' dictionary would not remove "yang", "untuk" or "adalah".
CREATE OR REPLACE FUNCTION fts_stopwords()
RETURNS text[] LANGUAGE sql IMMUTABLE AS $$
    SELECT ARRAY[
        -- English
        'a','an','and','are','as','at','be','by','can','do','does','did','for',
        'from','how','i','if','in','is','it','its','me','my','of','on','or','so',
        'that','the','their','them','then','there','these','this','those','to',
        'was','were','what','when','where','which','who','why','will','with',
        'you','your','about','any','have','has','had','not','but','we','us',
        -- Bahasa Melayu
        'ada','adakah','adalah','akan','apa','apakah','atau','bagaimana','bagi',
        'boleh','bolehkah','dan','dari','daripada','dengan','di','ia','ini','itu',
        'juga','kepada','ke','lain','mana','pada','perlu','saya','satu','sila',
        'supaya','tidak','untuk','yang','anda','kami','kita','oleh','akaun'
    ]::text[]
$$;

-- ------------------------------------------------------------- OR tsquery
-- Lexemes are taken from to_tsvector('simple', ...) so tokenisation matches the
-- stored column exactly, then stopwords are filtered and the remainder OR-joined.
-- quote_literal keeps punctuation and non-Latin tokens from breaking the cast.
CREATE OR REPLACE FUNCTION fts_or_query(q TEXT)
RETURNS tsquery LANGUAGE sql IMMUTABLE AS $$
    SELECT COALESCE(
        (
            SELECT string_agg(quote_literal(t.lexeme), ' | ')
            FROM unnest(to_tsvector('simple', coalesce(q, '')))
                 AS t(lexeme, positions, weights)
            WHERE t.lexeme <> ALL (fts_stopwords())
              AND length(t.lexeme) > 1
        )::tsquery,
        ''::tsquery
    )
$$;

-- --------------------------------------------------------- hybrid_search_v2
-- Identical to hybrid_search except for the three tsquery expressions in
-- fts_ranked. Kept as a separate function so both can be evaluated on the same
-- corpus without a destructive change.
CREATE OR REPLACE FUNCTION hybrid_search_v2(
    query_text       TEXT,
    query_embedding  vector(1024),
    match_count      INTEGER DEFAULT 8,
    rrf_k            INTEGER DEFAULT 60,
    weight_vector    FLOAT   DEFAULT 1.0,
    weight_fts       FLOAT   DEFAULT 1.0,
    filter_category  TEXT    DEFAULT NULL,
    only_public      BOOLEAN DEFAULT TRUE,
    candidate_pool   INTEGER DEFAULT 50
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
fts_ranked AS (
    SELECT
        e.embedding_id,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank(e.fts, fts_or_query(query_text)) DESC
        ) AS rank
    FROM embeddings e
    WHERE e.fts @@ fts_or_query(query_text)
      AND (filter_category IS NULL OR e.category = filter_category)
      AND (NOT only_public OR e.public IS TRUE)
    ORDER BY ts_rank(e.fts, fts_or_query(query_text)) DESC
    LIMIT candidate_pool
),
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
JOIN embeddings e USING (embedding_id)
ORDER BY fused.rrf_score DESC
LIMIT match_count;
$$;
