"""Retrieval tool — the LangGraph node simply calls the in-DB hybrid_search RPC.

Fusion (RRF) happens entirely in Postgres; this is a thin client. The query is
embedded via the centralized embed() helper using the e5 QUERY prefix.
"""
from __future__ import annotations

from typing import Any

from core.config import settings
from services.embeddings import embed_query
from services.supabase_client import hybrid_search


def retrieve(
    query_text: str,
    *,
    match_count: int = 8,
    rrf_k: int = 60,
    weight_vector: float = 1.0,
    weight_fts: float = 1.0,
    filter_category: str | None = None,
) -> list[dict[str, Any]]:
    """Embed the query (e5 query prefix) and run hybrid_search RRF in-DB."""
    query_embedding = embed_query(query_text)
    return hybrid_search(
        query_text=query_text,
        query_embedding=query_embedding,
        match_count=match_count,
        rrf_k=rrf_k,
        weight_vector=weight_vector,
        weight_fts=weight_fts,
        filter_category=filter_category,
        only_public=True,
    )
