"""Supabase client + data-access helpers (service-role, server-side only)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)


def _force_http1(session) -> None:
    """Rebuild an httpx session with HTTP/2 disabled, preserving its config.

    Supabase's client negotiates HTTP/2. On some networks an intermediary
    terminates the h2 connection after the first exchange (GOAWAY with
    error_code 0), so every SECOND request on a reused connection dies with
    RemoteProtocolError while the first succeeds. Measured behaviour on the
    development network was 200 / fail / 200 / fail over HTTP/2 and 6 of 6
    successes over HTTP/1.1 against the same endpoint.

    supabase 2.11.0 exposes no option for this, so the session is rebuilt.
    HTTP/1.1 costs a little multiplexing efficiency and buys reliability.
    """
    import httpx

    session._transport = httpx.HTTPTransport(http2=False)


@lru_cache
def get_client() -> Client:
    """Service-role client. NEVER expose this key to clients."""
    client = create_client(settings.supabase_url, settings.supabase_service_role_key)
    _force_http1(client.postgrest.session)
    storage_session = getattr(client.storage, "_client", None)
    if storage_session is not None:
        _force_http1(storage_session)
    return client


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
    rpc_name: str = "hybrid_search",
) -> list[dict[str, Any]]:
    """Call the in-DB hybrid RPC (pgvector + FTS, RRF fused in Postgres).

    rpc_name selects the keyword-channel implementation. "hybrid_search" is the
    original AND-semantics version; "hybrid_search_v2" (migration 002) drops
    stopwords and ORs the remaining terms. Both exist simultaneously so the two
    can be evaluated against the same corpus without a destructive change.
    """
    res = get_client().rpc(
        rpc_name,
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
# A WhatsApp chat has no explicit session boundary, so consecutive messages are
# treated as one conversation until this much time passes with no activity.
CONVERSATION_IDLE_HOURS = 12


def get_or_create_conversation(user_id: str, conversation_id: str | None) -> str:
    """Resume the user's current conversation, or start a new one.

    Without this lookup every inbound message created a fresh conversation, so
    load_recent_messages() always returned an empty history and the agent had
    no multi-turn memory at all (follow-ups like "explain that in English" or
    "what about after 55?" lost all context).
    """
    client = get_client()
    if conversation_id:
        return conversation_id

    # Most recent conversation for this user.
    recent = (
        client.table("conversation")
        .select("conversation_id")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if recent:
        cid = recent[0]["conversation_id"]
        # Reuse it only if it is still "live" — i.e. its last message is recent.
        last = (
            client.table("message")
            .select("created_at")
            .eq("conversation_id", cid)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
            .data
        )
        if last:
            from datetime import datetime, timedelta, timezone

            ts = datetime.fromisoformat(last[0]["created_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - ts < timedelta(hours=CONVERSATION_IDLE_HOURS):
                return cid
        else:
            # Empty conversation (previous task failed before saving) — reuse it
            # rather than leaving orphans behind on every Celery retry.
            return cid

    res = client.table("conversation").insert({"user_id": user_id}).execute()
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


# PostgreSQL text columns cannot hold a NUL byte, and PostgREST surfaces the
# attempt as error 22P05 ("\\u0000 cannot be converted to text"). PDF text
# extraction and OCR both emit NULs on some inputs, so an ordinary user
# attachment could crash the whole task: the insert failed, Celery retried
# three times, and the reply was never delivered. Stripping here covers every
# caller rather than trusting each extractor to clean up after itself.
def _pg_safe(text: str | None) -> str | None:
    """Remove characters PostgreSQL rejects in a text column."""
    if text is None:
        return None
    return text.replace("\x00", "")


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
                "message": _pg_safe(message),
                "translate_message": _pg_safe(translate_message),
                "language": language,
                "retrieved_chunk_ids": retrieved_chunk_ids or [],
            }
        )
        .execute()
    )
    return res.data[0]["message_id"]
