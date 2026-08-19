"""Private Supabase Storage download.

The Celery worker downloads media directly from the PRIVATE bucket using the
service-role key AT THE MOMENT OF PROCESSING. This avoids 403 URL-expiry errors
even if the queue is backed up (no signed-URL TTL races), and keeps raw bytes
out of Redis/the broker.
"""
from __future__ import annotations

from core.config import settings
from core.logging import get_logger
from services.supabase_client import get_client

log = get_logger(__name__)


def download_media(media_path: str) -> bytes:
    """Download bytes from the private bucket.

    `media_path` is the path WITHIN the bucket (n8n passes e.g.
    'audio/abc.wav' or 'bucket/audio/abc.wav'). We strip a leading bucket
    segment if present so callers can pass either form.
    """
    bucket = settings.supabase_storage_bucket
    path = media_path
    if path.startswith(f"{bucket}/"):
        path = path[len(bucket) + 1 :]

    log.info("Downloading media from bucket=%s path=%s", bucket, path)
    return get_client().storage.from_(bucket).download(path)
