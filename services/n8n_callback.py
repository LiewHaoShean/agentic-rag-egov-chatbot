"""Outbound callback to n8n with retry + exponential backoff + dead-letter log.

The user's final answer MUST NOT be silently dropped if n8n is down or there is
a network blip. We retry with exponential backoff; on exhaustion we append the
payload to a dead-letter log so delivery is recoverable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)


@retry(
    retry=retry_if_exception_type(httpx.HTTPError),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    stop=stop_after_attempt(settings.n8n_callback_max_retries),
    reraise=True,
)
def _post(payload: dict) -> None:
    resp = httpx.post(
        settings.n8n_callback_url,
        json=payload,
        timeout=settings.n8n_callback_timeout_seconds,
    )
    resp.raise_for_status()


def _dead_letter(payload: dict, error: str) -> None:
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "payload": payload,
    }
    with open(settings.callback_dlq_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.error("Callback DLQ write for message_id=%s", payload.get("message_id"))


def send_reply(message_id: str, to_number: str, reply_text: str) -> bool:
    """Push the final agent reply back to n8n. Returns True on success."""
    payload = {
        "message_id": message_id,
        "to": to_number,
        "reply": reply_text,
    }
    try:
        _post(payload)
        log.info("Delivered reply to n8n for message_id=%s", message_id)
        return True
    except (httpx.HTTPError, RetryError) as exc:
        _dead_letter(payload, str(exc))
        return False
