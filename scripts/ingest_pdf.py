"""Ingest KWSP/LHDN PDFs into Supabase embeddings — one command per PDF.

Pipeline:
  1. Download the PDF through Playwright (handles Cloudflare's JS challenge and
     the forced-download response that plain HTTP clients get 403 on).
  2. Extract each page's embedded text with pypdf.
  3. For image-only / sparse pages, fall back to Qwen3-VL OCR (render the page to
     PNG with PyMuPDF, send to DashScope). Reuses the project's vision service.
  4. Normalise (de-dup repeated slide captions), chunk, translate, embed, insert.
     Same chunk -> embed -> insert pipeline as the web scraper, so query/ingest
     embeddings stay consistent. Idempotent per URL (delete-by-file_url).

Run from the project root:
  python -m scripts.ingest_pdf --url "<pdf-url>" --preview        # text only, no API/DB
  python -m scripts.ingest_pdf --url "<pdf-url>"                  # upload (OCR auto)
  python -m scripts.ingest_pdf --url "<pdf-url>" --category epf --no-ocr
"""
from __future__ import annotations

import argparse
import io
import sys
import time

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

from core.logging import configure_logging, get_logger
from services.embeddings import embed_passage
from services.supabase_client import get_client
from scripts.ingest_scrape import (
    category_for,
    document_id_for,
    normalize_text,
    translate_to_english,
)

configure_logging()
log = get_logger("ingest_pdf")

# ---------------------------------------------------------------- Target PDFs
# Paste PDF URLs here and run `python -m scripts.ingest_pdf` with no args to
# loop through all of them. `--url` on the command line overrides this list.
PDF_URLS = [
    "https://www.kwsp.gov.my/documents/d/guest/panduan-mudah-pengeluaran-akaun-fleksibel_eng",
    "https://www.kwsp.gov.my/documents/d/guest/easy-guide_-recuring-transfer-pdf",
    "https://www.kwsp.gov.my/documents/d/guest/easy-guide_one-time-transfer-pdf"
    # add more KWSP / LHDN PDF URLs here ...
]

# Pages with fewer than this many extractable chars are treated as image-only
# and routed to OCR (if enabled).
_MIN_PAGE_CHARS = 80
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")


# ----------------------------------------------------------------- download
def download_pdf(url: str) -> bytes:
    """Fetch a PDF through a real browser (passes Cloudflare + forced download)."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=_UA, accept_downloads=True, locale="en-US")
        page = ctx.new_page()
        try:
            with page.expect_download(timeout=60000) as dl:
                try:
                    page.goto(url, timeout=60000)
                except Exception:  # noqa: BLE001 — goto aborts when it becomes a download
                    pass
            data = dl.value.path().read_bytes()
        finally:
            ctx.close()
            browser.close()
    if data[:4] != b"%PDF":
        raise RuntimeError("Downloaded content is not a PDF (likely blocked).")
    return data


# ----------------------------------------------------------------- extraction
def _ocr_page(pdf_bytes: bytes, page_index: int) -> str:
    """Render one page to PNG and OCR it via Qwen3-VL (DashScope)."""
    from services.qwen_vision import extract_text

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[page_index].get_pixmap(dpi=150)
    png = pix.tobytes("png")
    return extract_text(png, "image/png")


def extract_pages(pdf_bytes: bytes, *, use_ocr: bool, preview: bool) -> str:
    """Return combined text for all pages, OCR'ing sparse pages when enabled."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    out: list[str] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if len(text) >= _MIN_PAGE_CHARS:
            out.append(text)
            continue
        # sparse / image-only page
        if preview:
            log.info("  page %d: sparse (%d chars) — would OCR", i + 1, len(text))
            out.append(text)
        elif use_ocr:
            log.info("  page %d: sparse — OCR via Qwen3-VL", i + 1)
            try:
                out.append(_ocr_page(pdf_bytes, i))
            except Exception as exc:  # noqa: BLE001
                log.warning("  page %d OCR failed: %s", i + 1, exc)
                out.append(text)
        else:
            log.info("  page %d: sparse, OCR disabled — keeping %d chars", i + 1, len(text))
            out.append(text)
    return normalize_text("\n".join(out))


# ----------------------------------------------------------------- pipeline
def ingest_pdf(url: str, splitter, *, category: str | None, use_ocr: bool, preview: bool) -> None:
    log.info("PDF %s", url)
    pdf_bytes = download_pdf(url)
    log.info("  downloaded %d bytes", len(pdf_bytes))

    text = extract_pages(pdf_bytes, use_ocr=use_ocr, preview=preview)
    cat = category or category_for(url)
    document_id = document_id_for(url)

    chunks = [c.strip() for c in splitter.split_text(text)]
    chunks = [c for c in chunks if len(c) >= 15]
    log.info("  %d chunks (category=%s)", len(chunks), cat)

    if preview:
        for i, c in enumerate(chunks):
            print(f"\n--- CHUNK {i} ---\n{c}")
        return

    rows = []
    for index, original_text in enumerate(chunks):
        translate_text, lang = translate_to_english(original_text)
        rows.append(
            {
                "document_id": document_id,
                "chunk_index": index,
                "embedding_vector": embed_passage(translate_text),
                "original_text": original_text,
                "translate_text": translate_text,
                "current_language": lang,
                "file_url": url,
                "category": cat,
                "public": True,
            }
        )
    if not rows:
        log.warning("  nothing to store")
        return
    client = get_client()
    client.table("embeddings").delete().eq("file_url", url).execute()
    client.table("embeddings").insert(rows).execute()
    log.info("  stored %d chunks", len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest PDF(s) into Supabase embeddings.")
    parser.add_argument("--url", action="append",
                        help="PDF URL; repeatable. Defaults to the PDF_URLS list.")
    parser.add_argument("--category", help="Override category (else inferred from host).")
    parser.add_argument("--preview", action="store_true", help="Text only; no OCR/embed/DB.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable Qwen3-VL OCR fallback.")
    parser.add_argument("--delay", type=float, default=6.0,
                        help="Seconds between PDFs (be polite to the server).")
    args = parser.parse_args()

    urls = args.url or PDF_URLS
    if not urls:
        log.warning("No PDF URLs — add some to PDF_URLS or pass --url.")
        return 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    log.info("Mode=%s | %d PDF(s)", "PREVIEW" if args.preview else "UPLOAD", len(urls))
    for i, url in enumerate(urls):
        try:
            ingest_pdf(url, splitter, category=args.category,
                       use_ocr=not args.no_ocr, preview=args.preview)
        except Exception as exc:  # noqa: BLE001 — one bad PDF shouldn't stop the batch
            log.error("Skip %s: %s", url, exc)
        if i < len(urls) - 1:
            time.sleep(args.delay)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
