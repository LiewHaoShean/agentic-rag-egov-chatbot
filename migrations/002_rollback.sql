-- Rollback for 002_fts_or_semantics.sql
--
-- 002 only ADDED functions, so undoing it is three drops. hybrid_search, the
-- embeddings table, the generated fts column and the GIN index were never
-- touched, so there is nothing to restore and no data to migrate back.
--
-- Run this only after retrieve() in agent/tools.py has been pointed back at
-- "hybrid_search", otherwise the app will call a function that no longer exists.

DROP FUNCTION IF EXISTS hybrid_search_v2(
    TEXT, vector(1024), INTEGER, INTEGER, FLOAT, FLOAT, TEXT, BOOLEAN, INTEGER
);
DROP FUNCTION IF EXISTS fts_or_query(TEXT);
DROP FUNCTION IF EXISTS fts_stopwords();
