"""Pydantic request/response models for the WhatsApp webhook bridge."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

MessageType = Literal["text", "audio", "image", "document"]

# Separates a media caption (the user's own words) from extracted media text in
# the combined user_text. Language detection must key off the caption side.
ATTACHMENT_MARKER = "[ATTACHED FILE CONTENT]"


class WhatsAppWebhookPayload(BaseModel):
    """JSON forwarded by the n8n workflow.

    n8n uploads any media to a PRIVATE Supabase Storage bucket and passes only
    the private storage path (e.g. 'whatsapp-media/audio/abc.wav') — never raw
    Base64 (would bloat Redis/crash the broker) and never a public/signed URL.
    """

    message_id: str = Field(..., description="WhatsApp message id — dedupe key")
    from_number: str = Field(..., description="Sender WhatsApp phone number")
    type: MessageType

    # Present for type == 'text'
    text: Optional[str] = None

    # Present for type in {'audio', 'image'} — private bucket path only.
    media_path: Optional[str] = None
    media_mime_type: Optional[str] = None

    # WhatsApp image/document caption — the user's own words with the media.
    caption: Optional[str] = None

    # Optional conversation hint from n8n; otherwise resolved/created server-side.
    conversation_id: Optional[str] = None


class WebhookAck(BaseModel):
    """Immediate 202 body returned to n8n after dispatch."""

    status: Literal["accepted", "duplicate"]
    message_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    redis: bool
