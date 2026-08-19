"""Qwen3-VL (image OCR + extraction/translation) via DashScope — thin HTTP client.

Remote API only; no local inference. Has its own timeout.
"""
from __future__ import annotations

import base64

import httpx

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

_OCR_PROMPT = (
    "Extract ALL text from this image of a Malaysian government document or form. "
    "Preserve numbers, dates, and form-field labels exactly. If the text is in "
    "Bahasa Melayu, also provide a faithful English translation. Return only the "
    "extracted/translated text, no commentary."
)


def extract_text(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """OCR + extract text from an image (e.g. a tax form photo)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    url = f"{settings.dashscope_base_url}/services/aigc/multimodal-generation/generation"

    resp = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.qwen_vl_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:{mime_type};base64,{b64}"},
                            {"text": _OCR_PROMPT},
                        ],
                    }
                ]
            },
        },
        timeout=settings.qwen_timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    # Envelope verified against the live API (qwen3-vl-plus, 2026-07-05).
    return data["output"]["choices"][0]["message"]["content"][0]["text"]
