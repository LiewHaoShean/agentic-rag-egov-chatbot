"""Shared Redis client + idempotency helper.

Redis serves two roles:
  1. Celery broker (configured in worker/celery_app.py).
  2. Idempotency dedupe store for inbound WhatsApp messages.
"""
import redis

from core.config import settings

# Decoded (str) client for app-level use (dedupe, health). Celery uses its own.
_redis: redis.Redis = redis.from_url(settings.redis_url, decode_responses=True)

_DEDUPE_PREFIX = "wa:dedupe:"


def get_redis() -> redis.Redis:
    return _redis


def claim_message(message_id: str) -> bool:
    """Atomically claim a WhatsApp message_id for processing.

    Uses SETNX + TTL so redelivery storms (n8n/WhatsApp retries) collapse to a
    single Celery dispatch. Returns True if this caller won the claim (i.e. the
    message is new and should be processed), False if it was already seen.

    MUST be called BEFORE dispatching the Celery task.
    """
    key = f"{_DEDUPE_PREFIX}{message_id}"
    # nx=True -> only set if absent; ex=TTL -> auto-expire the dedupe record.
    won = _redis.set(key, "1", nx=True, ex=settings.idempotency_ttl_seconds)
    return bool(won)


def ping() -> bool:
    try:
        return bool(_redis.ping())
    except redis.RedisError:
        return False
