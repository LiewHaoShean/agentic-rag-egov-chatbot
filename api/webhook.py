"""POST /webhook/whatsapp — secure, idempotent, non-blocking ingress.

Flow (mirrors the architecture diagram):
  X-Webhook-Token valid?  -> no: 401
  message_id seen? (Redis SETNX) -> duplicate: ack + drop, no reprocessing
  new: dispatch Celery task, return HTTP 202 Accepted immediately.

The route NEVER runs ASR/vision/LangGraph inline — those are slow and would
time out the n8n webhook.
"""
from fastapi import APIRouter, Header, HTTPException, Request, status

from core.config import settings
from core.logging import get_logger
from core.redis_client import claim_message
from models.schemas import WebhookAck, WhatsAppWebhookPayload
from worker.tasks import process_whatsapp_message

router = APIRouter(prefix="/webhook", tags=["webhook"])
log = get_logger(__name__)


@router.post(
    "/whatsapp",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=WebhookAck,
)
async def whatsapp_webhook(
    request: Request,
    x_webhook_token: str | None = Header(default=None, alias="X-Webhook-Token"),
):
    # --- Auth: constant-ish token check against configured secret ---
    if not settings.webhook_token or x_webhook_token != settings.webhook_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook token"
        )

    # --- Payload-size guard (protect Redis/broker before we parse) ---
    raw = await request.body()
    if len(raw) > settings.max_payload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Payload too large",
        )

    # --- Parse + validate ---
    try:
        payload = WhatsAppWebhookPayload.model_validate_json(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )

    # --- Idempotency: claim BEFORE dispatch (absorbs redelivery storms) ---
    if not claim_message(payload.message_id):
        log.info("Duplicate message_id=%s — ack + drop", payload.message_id)
        return WebhookAck(status="duplicate", message_id=payload.message_id)

    # --- Async dispatch; return 202 immediately ---
    process_whatsapp_message.delay(payload.model_dump())
    log.info("Dispatched message_id=%s type=%s", payload.message_id, payload.type)
    return WebhookAck(status="accepted", message_id=payload.message_id)
