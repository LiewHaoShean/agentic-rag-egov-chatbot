"""Repair chunks whose translate_text is a Google error page.

Google serves its throttling pages with HTTP 200, so a rate-limited run stores
the error page as the English text and then embeds *that*. The affected chunks
retrieve for nothing, and because they all share the same wording their vectors
cluster together, so a query that does reach one tends to pull in the rest.

This finds those rows, re-translates from the stored original_text, re-embeds,
and updates in place. It never re-scrapes, so it costs no requests to the source
site and cannot change chunk boundaries.

Run from the project root:
    python -m scripts.repair_translations --dry-run
    python -m scripts.repair_translations
"""
from __future__ import annotations

import argparse
import socket
import sys
import time

from core.logging import configure_logging, get_logger
from services.embeddings import embed_passage
from services.supabase_client import get_client
from scripts.ingest_scrape import _looks_like_error_page, translate_to_english

configure_logging()
log = get_logger("repair_translations")

# deep_translator issues its HTTP calls without a timeout, so if the connection
# dies under it (changing network, dropped Wi-Fi) the run blocks forever on a
# socket that will never answer rather than failing and retrying. A default
# timeout turns that hang into an exception the retry loop already handles.
socket.setdefaulttimeout(45)


def _fetch_all(client):
    rows, page, size = [], 0, 1000
    while True:
        got = (
            client.table("embeddings")
            .select("embedding_id,file_url,original_text,translate_text")
            .range(page * size, (page + 1) * size - 1)
            .execute()
        )
        if not got.data:
            break
        rows += got.data
        if len(got.data) < size:
            break
        page += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-translate and re-embed damaged chunks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing.")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Seconds between chunks, to stay under the translator's limit.")
    args = parser.parse_args()

    client = get_client()
    rows = _fetch_all(client)
    damaged = [r for r in rows if _looks_like_error_page(r.get("translate_text") or "")]
    log.info("Scanned %d chunks | %d damaged", len(rows), len(damaged))

    if args.dry_run:
        for r in damaged:
            log.info("  would repair %s [chunk %s]", r["file_url"], r["embedding_id"])
        return 0

    repaired = failed = 0
    for i, row in enumerate(damaged, 1):
        original = row.get("original_text") or ""
        if not original.strip():
            log.warning("[%d/%d] no original_text, skipping %s", i, len(damaged), row["file_url"])
            failed += 1
            continue

        english, lang = translate_to_english(original)
        # translate_to_english now falls back to the source text rather than
        # returning an error page, so this should not trigger; it is here so a
        # future change to that contract cannot quietly reintroduce the bug.
        if _looks_like_error_page(english):
            log.error("[%d/%d] still an error page, leaving alone: %s",
                      i, len(damaged), row["file_url"])
            failed += 1
            continue

        try:
            vector = embed_passage(english)
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not kill the run
            log.warning("[%d/%d] embed failed (%s): %s", i, len(damaged), exc, row["file_url"])
            failed += 1
            continue

        client.table("embeddings").update(
            {
                "translate_text": english,
                "current_language": lang,
                "embedding_vector": vector,
            }
        ).eq("embedding_id", row["embedding_id"]).execute()
        repaired += 1
        log.info("[%d/%d] repaired (%s -> en, %d chars) %s",
                 i, len(damaged), lang, len(english), row["file_url"])
        time.sleep(args.delay)

    log.info("Done. repaired=%d failed=%d", repaired, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
