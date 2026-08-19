"""CRITICAL: single source of truth for ALL embedding calls.

e5-large-instruct requires ASYMMETRIC prefixing. Centralizing here guarantees
the ingestion path and the query path can never drift apart:

  * QUERY  -> "Instruct: {task_description}\nQuery: {text}"
  * PASSAGE/DOCUMENT -> no instruct prefix (raw text)

The helper enforces exactly EMBEDDING_DIMENSIONS (1024) so it always matches
vector(1024) in Supabase. Primary backend is the Alibaba embedding API; if it
fails we fall back to the HuggingFace Inference API (HF has cold-start/503 on
this specific model, hence it is the fallback, not the primary).

This convention is independent of deployment mode — it lives in how the string
is formatted before being sent to the API.
"""
from __future__ import annotations

from enum import Enum

import httpx

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

# Default task description for the e5 query instruct prefix.
# Canonical e5 phrasing (generic retrieval) — portable if scope expands.
# Applied to QUERIES ONLY; stored passages never receive this prefix.
DEFAULT_TASK_DESCRIPTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class EmbedType(str, Enum):
    QUERY = "query"
    PASSAGE = "passage"


def _format_input(text: str, kind: EmbedType, task_description: str) -> str:
    """Apply the asymmetric e5 prefix convention."""
    if kind is EmbedType.QUERY:
        return f"Instruct: {task_description}\nQuery: {text}"
    # Stored passages/documents receive NO instruct prefix.
    return text


def _enforce_dims(vector: list[float]) -> list[float]:
    n = settings.embedding_dimensions
    if len(vector) != n:
        raise ValueError(
            f"Embedding dim mismatch: got {len(vector)}, expected {n}. "
            "Refusing to store/query a vector that won't match vector(1024)."
        )
    return vector


def _embed_alibaba(formatted: str) -> list[float]:
    """Primary backend — Alibaba (DashScope compatible-mode) embeddings API."""
    resp = httpx.post(
        settings.alibaba_embedding_url,
        headers={"Authorization": f"Bearer {settings.alibaba_embedding_api_key}"},
        json={
            "model": settings.alibaba_embedding_model,
            "input": formatted,
            "dimensions": settings.embedding_dimensions,
            "encoding_format": "float",
        },
        timeout=settings.embedding_timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]


def _embed_hf(formatted: str) -> list[float]:
    """Fallback backend — HuggingFace Inference API (cold-start/503 prone)."""
    resp = httpx.post(
        settings.hf_embedding_url,
        headers={"Authorization": f"Bearer {settings.hf_api_key}"},
        json={"inputs": formatted, "options": {"wait_for_model": True}},
        timeout=settings.embedding_timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    # HF returns either a flat vector or a nested list depending on the model.
    if data and isinstance(data[0], list):
        return data[0]
    return data


def embed(
    text: str,
    kind: EmbedType = EmbedType.QUERY,
    task_description: str = DEFAULT_TASK_DESCRIPTION,
) -> list[float]:
    """Return a 1024-dim embedding for `text`.

    Args:
        text: raw text (no prefix — prefixing is applied here).
        kind: EmbedType.QUERY for user questions, EmbedType.PASSAGE for stored docs.
        task_description: e5 instruct task description (queries only).
    """
    if not text.strip():
        # The API 400s on empty input; fail fast with a clear message instead.
        raise ValueError("Cannot embed empty text")
    formatted = _format_input(text, kind, task_description)
    try:
        return _enforce_dims(_embed_alibaba(formatted))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code < 500:
            # Client error (bad input/auth) — HF would reject it too; surface it.
            raise
        log.warning("Alibaba embedding failed (%s) — falling back to HF", exc)
        return _enforce_dims(_embed_hf(formatted))
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("Alibaba embedding failed (%s) — falling back to HF", exc)
        return _enforce_dims(_embed_hf(formatted))


def embed_query(text: str, task_description: str = DEFAULT_TASK_DESCRIPTION) -> list[float]:
    return embed(text, EmbedType.QUERY, task_description)


def embed_passage(text: str) -> list[float]:
    return embed(text, EmbedType.PASSAGE)
