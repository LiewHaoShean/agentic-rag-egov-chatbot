"""Discover JPN pages that are not yet in TARGET_URLS.

JPN runs WordPress, so the whole page inventory is published at
/wp-sitemap.xml. This walks that index, keeps the entries that look like
citizen-facing content, and diffs them against the scraper's TARGET_URLS.

It does NOT scrape content and does NOT touch the DB — it only lists URLs, so
the output is meant to be reviewed by hand before anything is pasted into
TARGET_URLS.

The fetching deliberately mirrors ingest_scrape: Playwright with a fresh
browser context per request and a randomised delay between them. JPN's edge
refuses connections outright after a handful of rapid automated hits, and a
plain requests/curl loop gets the IP banned within about four requests.

Run from the project root:
    python -m scripts.discover_jpn
    python -m scripts.discover_jpn --include-all   # skip the relevance filter
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

from core.logging import configure_logging, get_logger
from scripts.ingest_scrape import TARGET_URLS

configure_logging()
log = get_logger("discover_jpn")

SITEMAP_INDEX = "https://www.jpn.gov.my/wp-sitemap.xml"

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

# Sitemaps whose contents are not citizen guidance. Photo galleries, videos,
# press releases, tender notices and dated announcements age out and would add
# retrieval noise without answering any question the chatbot is asked.
_SKIP_SITEMAPS = (
    "jpn_galeri_foto",
    "jpn_video",
    "jpn_pusat_media",
    "jpn_sebut_harga",
    "jpn_pengumuman",
    "taxonomies",
)

# Path fragments that mark a page as organisational rather than procedural.
_SKIP_PATH_HINTS = (
    "/galeri", "/video", "/pengumuman", "/sebut-harga", "/tender",
    "/berita", "/media", "/siaran-akhbar", "/aduan", "/hubungi",
    "/dasar-privasi", "/penafian", "/peta-laman", "/wp-content",
    "/piagam-pelanggan", "/carta-organisasi", "/pengurusan-tertinggi",
    "/pekeliling", "/laporan-tahunan", "/muat-turun-borang-permohonan-jawatan",
)

_ASSET_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip",
              ".doc", ".docx", ".xls", ".xlsx")


def _normalise(url: str) -> str:
    """Strip fragment, query and trailing slash so the diff compares like for like."""
    p = urlparse(url)
    return p._replace(fragment="", query="", path=p.path.rstrip("/")).geturl()


def _fetch_xml(page, url: str, *, max_attempts: int = 4) -> str:
    """Load an XML document, backing off when the edge throttles us."""
    backoff = 10.0
    last = "No Response"
    for attempt in range(1, max_attempts + 1):
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
            if response and response.status == 200:
                return page.content()
            last = response.status if response else "No Response"
        except Exception as exc:  # noqa: BLE001 — connection refused counts as a throttle
            last = type(exc).__name__
        if attempt < max_attempts:
            wait = backoff * attempt + random.uniform(0, 5)
            log.warning("  %s on attempt %d/%d — backing off %.1fs",
                        last, attempt, max_attempts, wait)
            time.sleep(wait)
    raise RuntimeError(f"Could not fetch {url}. Last status: {last}")


def _locs(xml: str) -> list[str]:
    return re.findall(r"<loc>\s*(.*?)\s*</loc>", xml, flags=re.S)


def _is_relevant(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(_ASSET_EXT):
        return False
    return not any(hint in path for hint in _SKIP_PATH_HINTS)


def main() -> int:
    parser = argparse.ArgumentParser(description="List JPN pages missing from TARGET_URLS.")
    parser.add_argument("--include-all", action="store_true",
                        help="Do not filter out news, galleries and corporate pages.")
    parser.add_argument("--delay", type=float, default=8.0,
                        help="Base seconds between sitemap fetches.")
    parser.add_argument("--out", default="jpn_missing.txt",
                        help="Where to write the missing-URL list.")
    args = parser.parse_args()

    known = {_normalise(u) for u in TARGET_URLS if "jpn.gov.my" in u}
    log.info("TARGET_URLS already covers %d JPN pages", len(known))

    found: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        def get(url: str) -> str:
            # Fresh context per request: the WAF flags a session after its first
            # hit, so a clean cookie jar makes each look like a first-time visitor.
            ctx = browser.new_context(user_agent=_UA, locale="en-US")
            try:
                return _fetch_xml(ctx.new_page(), url)
            finally:
                ctx.close()

        index = get(SITEMAP_INDEX)
        children = [u for u in _locs(index)
                    if not any(s in u for s in _SKIP_SITEMAPS)]
        log.info("Sitemap index lists %d child sitemaps, %d worth reading",
                 len(_locs(index)), len(children))

        for i, child in enumerate(children):
            log.info("[%d/%d] %s", i + 1, len(children), child)
            urls = _locs(get(child))
            log.info("  %d URLs", len(urls))
            found.extend(urls)
            if i < len(children) - 1:
                wait = max(args.delay + random.uniform(-2, 4), 1.0)
                log.info("  waiting %.1fs", wait)
                time.sleep(wait)

        browser.close()

    everything = {_normalise(u) for u in found if "jpn.gov.my" in u}
    missing = sorted(u for u in everything - known
                     if args.include_all or _is_relevant(u))

    log.info("Sitemap total %d | already covered %d | missing %d",
             len(everything), len(everything & known), len(missing))

    # Group by section so the review pass reads in a sensible order.
    by_section: dict[str, list[str]] = {}
    for u in missing:
        segs = urlparse(u).path.strip("/").split("/")
        by_section.setdefault(segs[0] if segs and segs[0] else "(root)", []).append(u)

    lines: list[str] = []
    for section in sorted(by_section, key=lambda s: (-len(by_section[s]), s)):
        lines.append(f"\n# ---- /{section}/  ({len(by_section[section])}) ----")
        lines.extend(f'    "{u}",' for u in by_section[section])

    text = "\n".join(lines).lstrip("\n")
    print(text)
    with open(args.out, "w") as fh:
        fh.write(text + "\n")
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
