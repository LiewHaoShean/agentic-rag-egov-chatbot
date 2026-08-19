"""Ingest hand-curated knowledge docs (knowledge/*.md) into Supabase embeddings.

For authoritative content that is NOT scrapable as page text — e.g. rules that
live in a calculator's JavaScript, PDFs, or policy we transcribe by hand. Uses
the same chunk -> embed -> insert pipeline as the web scraper, so query/ingest
embeddings stay consistent.

Each doc carries front-matter as HTML comments at the top:
    <!-- category: epf -->
    <!-- source_url: https://www.kwsp.gov.my/... -->

Run from the project root:
    python -m scripts.ingest_docs --preview     # dry run (chunk counts only)
    python -m scripts.ingest_docs               # upload knowledge/*.md
    python -m scripts.ingest_docs --file knowledge/kwsp-late-payment-charge.md
"""
from __future__ import annotations

import argparse
import glob
import re
import sys

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logging import configure_logging, get_logger
from services.embeddings import embed_passage
from services.supabase_client import get_client
from scripts.ingest_scrape import document_id_for, translate_to_english

configure_logging()
log = get_logger("ingest_docs")

_FM = re.compile(r"<!--\s*(\w+)\s*:\s*(.*?)\s*-->")


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Pull `<!-- key: value -->` headers; return (meta, body without them)."""
    meta = {k.lower(): v for k, v in _FM.findall(text)}
    body = _FM.sub("", text).strip()
    return meta, body


def ingest_file(path: str, splitter, *, preview: bool) -> None:
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()

    meta, body = parse_front_matter(raw)
    category = meta.get("category", "general")
    source_url = meta.get("source_url") or f"file://{path}"
    document_id = document_id_for(source_url)

    chunks = [c.strip() for c in splitter.split_text(body)]
    chunks = [c for c in chunks if len(c) >= 15]
    log.info("%s -> %d chunks (category=%s)", path, len(chunks), category)

    if preview:
        for i, c in enumerate(chunks):
            print(f"\n--- {path} CHUNK {i} ---\n{c}")
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
                "file_url": source_url,
                "category": category,
                "public": True,
            }
        )

    if not rows:
        return
    client = get_client()
    client.table("embeddings").delete().eq("file_url", source_url).execute()
    client.table("embeddings").insert(rows).execute()
    log.info("  stored %d chunks", len(rows))


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest curated knowledge docs.")
    parser.add_argument("--preview", action="store_true", help="Dry run: print chunks.")
    parser.add_argument("--file", action="append", help="Specific file(s); repeatable.")
    args = parser.parse_args()

    files = args.file or sorted(glob.glob("knowledge/*.md"))
    if not files:
        log.warning("No knowledge/*.md files found.")
        return 1

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    log.info("Mode=%s | %d file(s)", "PREVIEW" if args.preview else "UPLOAD", len(files))
    for path in files:
        ingest_file(path, splitter, preview=args.preview)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
