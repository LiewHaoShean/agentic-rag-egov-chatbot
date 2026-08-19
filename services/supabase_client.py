"""Supabase client + data-access helpers (service-role, server-side only)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)


@lru_cache
def get_client() -> Client:
    """Service-role client. NEVER expose this key to clients."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


# ----------------------------------------------------------------- Retrieval
def hybrid_search(
    query_text: str,
    query_embedding: list[float],
    match_count: int = 8,
    rrf_k: int = 60,
    weight_vector: float = 1.0,
    weight_fts: float = 1.0,
    filter_category: str | None = None,
    only_public: bool = True,
) -> list[dict[str, Any]]:
    """Call the in-DB hybrid_search RPC (pgvector + FTS, RRF fused in Postgres)."""
    res = get_client().rpc(
        "hybrid_search",
        {
            "query_text": query_text,
            "query_embedding": query_embedding,
            "match_count": match_count,
            "rrf_k": rrf_k,
            "weight_vector": weight_vector,
            "weight_fts": weight_fts,
            "filter_category": filter_category,
            "only_public": only_public,
        },
    ).execute()
    return res.data or []


# ----------------------------------------------------------- Conversation memory
def get_or_create_conversation(user_id: str, conversation_id: str | None) -> str:
    client = get_client()
    if conversation_id:
        return conversation_id
    res = (
        client.table("conversation")
        .insert({"user_id": user_id})
        .execute()
    )
    return res.data[0]["conversation_id"]


def load_recent_messages(conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Hydrate prior turns from the DB (each webhook is a fresh task/state)."""
    res = (
        get_client()
        .table("message")
        .select("role, message, translate_message, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return list(reversed(res.data or []))


def get_user_by_phone(phone_number: str) -> dict[str, Any] | None:
    res = (
        get_client()
        .table("user")
        .select("*")
        .eq("phone_number", phone_number)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def upsert_user_by_phone(phone_number: str) -> str:
    existing = get_user_by_phone(phone_number)
    if existing:
        return existing["user_id"]
    res = get_client().table("user").insert({"phone_number": phone_number}).execute()
    return res.data[0]["user_id"]


def save_message(
    conversation_id: str,
    role: str,
    message: str,
    translate_message: str | None = None,
    language: str | None = None,
    retrieved_chunk_ids: list[str] | None = None,
) -> str:
    res = (
        get_client()
        .table("message")
        .insert(
            {
                "conversation_id": conversation_id,
                "role": role,
                "message": message,
                "translate_message": translate_message,
                "language": language,
                "retrieved_chunk_ids": retrieved_chunk_ids or [],
            }
        )
        .execute()
    )
    return res.data[0]["message_id"]
