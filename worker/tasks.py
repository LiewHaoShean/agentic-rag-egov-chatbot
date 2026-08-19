"""Celery task: end-to-end processing of one WhatsApp message.

Pipeline (mirrors the architecture diagram, worker half):
  download media (service-role) -> route by type
    audio -> Qwen ASR
    image -> Qwen3-VL OCR
    text  -> passthrough
  -> hydrate memory from DB -> run LangGraph agent
  -> persist answer + retrieved_chunk_ids -> outbound callback to n8n (retry/DLQ)

Each external API call (Gemini/Qwen/embeddings) carries its own timeout in its
service module; the task itself sets retry boundaries.
"""
from __future__ import annotations

from celery import shared_task
from celery.exceptions import MaxRetriesExceededError

from core.logging import get_logger
from models.schemas import ATTACHMENT_MARKER, WhatsAppWebhookPayload
from services import pdf_text, qwen_asr, qwen_vision, storage, supabase_client, n8n_callback
from worker.celery_app import celery  # noqa: F401  (ensures app is configured)

log = get_logger(__name__)

# Sent when a user attaches a file type we cannot read. Static + trilingual
# because we have no user text to detect a language from.
UNSUPPORTED_FILE_MESSAGE = (
    "Sorry, I can only read PDF files and images. / "
    "Maaf, saya hanya boleh membaca fail PDF dan imej. / "
    "抱歉，我只能读取 PDF 文件和图片。"
)

# Last-resort message when the whole task keeps failing (all retries spent).
# Static + trilingual: the failure may predate language detection.
SYSTEM_BUSY_MESSAGE = (
    "Maaf, sistem sibuk buat masa ini. Sila cuba sebentar lagi. / "
    "Sorry, the system is busy right now. Please try again in a moment. / "
    "抱歉，系统暂时繁忙，请稍后再试。"
)


class UnsupportedMediaError(Exception):
    """User sent a file type we cannot extract text from (e.g. .docx, .zip)."""


def _with_caption(payload: WhatsAppWebhookPayload, extracted: str) -> str:
    """Caption first: it is the user's actual question; the extracted text is
    the attachment. _reply_language keys off the caption side."""
    caption = (payload.caption or "").strip()
    if caption:
        return f"{caption}\n\n{ATTACHMENT_MARKER}\n{extracted}"
    return extracted


def _resolve_input_text(payload: WhatsAppWebhookPayload) -> str:
    """Turn any input modality into plain text for the agent."""
    if payload.type == "text":
        return payload.text or ""

    if not payload.media_path:
        raise ValueError(f"media_path required for type={payload.type}")

    media = storage.download_media(payload.media_path)
    mime = payload.media_mime_type or ""

    if payload.type == "audio":
        return qwen_asr.transcribe(media, mime or "audio/wav")
    if payload.type == "image":
        return _with_caption(payload, qwen_vision.extract_text(media, mime or "image/jpeg"))
    if payload.type == "document":
        # Route by actual content, not just declared mime (magic bytes win).
        if mime == "application/pdf" or media[:4] == b"%PDF":
            return _with_caption(payload, pdf_text.extract_text(media))
        if mime.startswith("image/"):
            # e.g. a PNG sent "as document" to avoid WhatsApp compression.
            return _with_caption(payload, qwen_vision.extract_text(media, mime))
        raise UnsupportedMediaError(mime or "unknown")

    raise ValueError(f"Unsupported type={payload.type}")


@shared_task(
    bind=True,
    name="worker.tasks.process_whatsapp_message",
    max_retries=2,
    default_retry_delay=10,
    acks_late=True,
)
def process_whatsapp_message(self, raw_payload: dict):
    payload = WhatsAppWebhookPayload.model_validate(raw_payload)
    log.info("Processing message_id=%s type=%s", payload.message_id, payload.type)

    try:
        # 1) Normalize input to text (ASR / OCR / PDF extraction / passthrough).
        try:
            user_text = _resolve_input_text(payload)
        except UnsupportedMediaError as exc:
            # Not transient — tell the user instead of retrying into silence.
            log.info("Unsupported media (%s) for message_id=%s", exc, payload.message_id)
            n8n_callback.send_reply(
                message_id=payload.message_id,
                to_number=payload.from_number,
                reply_text=UNSUPPORTED_FILE_MESSAGE,
            )
            return {"message_id": payload.message_id, "delivered": True}

        # 2) Resolve user + conversation; hydrate memory from DB.
        user_id = supabase_client.upsert_user_by_phone(payload.from_number)
        conversation_id = supabase_client.get_or_create_conversation(
            user_id, payload.conversation_id
        )
        history = supabase_client.load_recent_messages(conversation_id)

        # Persist the inbound user turn.
        supabase_client.save_message(
            conversation_id=conversation_id,
            role="user",
            message=user_text,
        )

        # 3) Run the LangGraph agent (imported lazily to keep import graph light).
        from agent.graph import run_agent

        result = run_agent(
            user_text=user_text,
            history=history,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        reply_text: str = result["reply"]
        retrieved_chunk_ids: list[str] = result.get("retrieved_chunk_ids", [])

        # 4) Persist the assistant turn with the chunks it used (eval audit trail).
        supabase_client.save_message(
            conversation_id=conversation_id,
            role="assistant",
            message=reply_text,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )

    except Exception as exc:  # noqa: BLE001 — bounded retry, then busy message
        log.exception("Task failed for message_id=%s", payload.message_id)
        try:
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            # All retries spent — never leave the user in silence.
            log.error("Retries exhausted for message_id=%s — sending busy message",
                      payload.message_id)
            n8n_callback.send_reply(
                message_id=payload.message_id,
                to_number=payload.from_number,
                reply_text=SYSTEM_BUSY_MESSAGE,
            )
            return {"message_id": payload.message_id, "delivered": True, "degraded": True}

    # 5) Outbound callback to n8n (own retry/backoff + DLQ inside).
    n8n_callback.send_reply(
        message_id=payload.message_id,
        to_number=payload.from_number,
        reply_text=reply_text,
    )
    return {"message_id": payload.message_id, "delivered": True}
