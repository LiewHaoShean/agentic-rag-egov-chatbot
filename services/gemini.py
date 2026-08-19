"""Gemini client (agent brain) — remote API only.

Two interchangeable backends (set GEMINI_BACKEND):
  * "google-api": AI Studio API key via langchain-google-genai. Free tier is
    ~20 requests/day per model — fine for smoke tests, not for real traffic.
  * "vertex":     Vertex AI via langchain-google-vertexai. Bills the GCP
    project (trial credit applies). Auth = Application Default Credentials
    (`gcloud auth application-default login` locally; service account in prod).

Same models, same LangChain interface — the agent nodes never know the difference.
Centralizes LLM construction, the per-call timeout, and optional Langfuse
tracing so every node shares one configured client.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)


@lru_cache
def get_llm(temperature: float = 0.2):
    if settings.gemini_backend == "vertex":
        from langchain_google_vertexai import ChatVertexAI

        return ChatVertexAI(
            model=settings.gemini_model,
            project=settings.google_cloud_project,
            location=settings.vertex_location,
            temperature=temperature,
            max_retries=1,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
        timeout=settings.gemini_timeout_seconds,
        max_retries=1,
    )


@lru_cache
def _langfuse_handler():
    """Optional Langfuse LangChain callback. Returns None if not configured."""
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    try:
        from langfuse.callback import CallbackHandler

        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # noqa: BLE001 — tracing must never break the request
        log.warning("Langfuse handler unavailable: %s", exc)
        return None


def trace_config(trace_name: str, **metadata: Any) -> dict[str, Any]:
    """Build a LangChain `config` dict carrying the Langfuse callback (if any)."""
    handler = _langfuse_handler()
    cfg: dict[str, Any] = {"run_name": trace_name, "metadata": metadata}
    if handler is not None:
        cfg["callbacks"] = [handler]
    return cfg
