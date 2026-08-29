"""User-uploaded PDF -> text, for the message pipeline (worker side).

pypdf extracts each page's embedded text; sparse/image-only pages fall back to
Qwen3-VL OCR via a PyMuPDF render. Same approach as scripts/ingest_pdf.py, but
with page/char caps: a user can attach a 100-page PDF and the agent prompt
must stay bounded (and OCR is a paid call per page).
"""
from __future__ import annotations

import io

import fitz  # PyMuPDF
from pypdf import PdfReader

from core.logging import get_logger

log = get_logger(__name__)

# Pages with fewer extractable chars than this are treated as image-only.
_MIN_PAGE_CHARS = 80
# Caps: users mostly send single forms/letters; anything longer is truncated.
MAX_PAGES = 10
MAX_CHARS = 8000


def _ocr_page(pdf_bytes: bytes, page_index: int) -> str:
    from services.qwen_vision import extract_text as ocr

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    png = doc[page_index].get_pixmap(dpi=150).tobytes("png")
    return ocr(png, "image/png")


def extract_text(pdf_bytes: bytes) -> str:
    """Combined text of the first MAX_PAGES pages, OCR'ing sparse ones."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total = len(reader.pages)
    out: list[str] = []
    for i, page in enumerate(reader.pages[:MAX_PAGES]):
        text = (page.extract_text() or "").strip()
        if len(text) < _MIN_PAGE_CHARS:
            log.info("PDF page %d/%d sparse (%d chars) — OCR", i + 1, total, len(text))
            try:
                text = _ocr_page(pdf_bytes, i).strip()
            except Exception as exc:  # noqa: BLE001 — keep whatever pypdf found
                log.warning("PDF page %d OCR failed: %s", i + 1, exc)
        out.append(text)

    # Truncate FIRST, then append the notice — appending before the slice meant
    # the warning was cut off in exactly the case it matters most (a long
    # document trips both caps, since 10 pages of policy text far exceeds
    # MAX_CHARS). The model must know it is reading a fragment.
    combined = "\n".join(out).strip()
    was_truncated = len(combined) > MAX_CHARS
    combined = combined[:MAX_CHARS]

    notes: list[str] = []
    if total > MAX_PAGES:
        notes.append(f"{total - MAX_PAGES} more pages not shown")
    if was_truncated:
        notes.append("text truncated")
    if notes:
        combined += f"\n\n[... document continues: {'; '.join(notes)}]"
    return combined
