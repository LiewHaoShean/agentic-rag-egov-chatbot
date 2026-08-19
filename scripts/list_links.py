"""Link-discovery helper: list the content links on a landing/card page.

Point it at a page full of cards; it prints each destination link (text -> URL)
so you can copy the good ones into TARGET_URLS. It does NOT scrape content and
does NOT touch the DB — it just discovers URLs.

  * Reuses the scraper's Cloudflare handling (fresh context + retry) and its
    junk-selector stripping, so header/footer/nav links are removed and you see
    mostly real card/content links.
  * De-duplicates URLs (normalised: no #fragment, no trailing slash) — each URL
    is listed once.

Run from the project root:
  python -m scripts.list_links --url "https://www.kwsp.gov.my/en/member/life-stages"
  python -m scripts.list_links --url "<landing>" --children   # only links under this path
"""
from __future__ import annotations

import argparse
import sys
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

from core.logging import configure_logging, get_logger
from scripts.ingest_scrape import _JUNK_SELECTORS, _goto_with_retry

configure_logging()
log = get_logger("list_links")

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
# Extensions that are downloads/assets, not content pages.
_ASSET_EXT = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip",
              ".doc", ".docx", ".xls", ".xlsx", ".css", ".js")


_LANG_CODES = {"en", "ms", "bm"}


def _apply_lang(url: str, base_lang: str | None) -> str:
    """Inject the landing page's language prefix if a link omitted it.

    KWSP cards sometimes link to '/member/...' (no /en/) while the actual pages
    live at '/en/member/...'. Canonicalising avoids scraping the same content
    twice under two URLs.
    """
    if not base_lang:
        return url
    p = urlparse(url)
    segs = p.path.lstrip("/").split("/", 1)
    first = segs[0].lower() if segs and segs[0] else ""
    if first in _LANG_CODES:
        return url  # already has a language prefix
    new_path = f"/{base_lang}/{p.path.lstrip('/')}"
    return p._replace(path=new_path).geturl()


def _normalise(url: str) -> str:
    """Canonical key for de-dup: drop #fragment and trailing slash, lowercase host."""
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return f"{p.scheme}://{p.netloc.lower()}{path}" + (f"?{p.query}" if p.query else "")


def discover(url: str, *, children_only: bool) -> list[tuple[str, str]]:
    base = urlparse(url)
    base_path = base.path.rstrip("/")
    # Language prefix of the landing page (e.g. 'en'), used to canonicalise links.
    _segs = base.path.lstrip("/").split("/", 1)
    base_lang = _segs[0].lower() if _segs and _segs[0].lower() in _LANG_CODES else None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=_UA, locale="en-US")
        page = ctx.new_page()
        try:
            _goto_with_retry(page, url)
            page.wait_for_timeout(3000)
            html = page.content()
        finally:
            ctx.close()
            browser.close()

    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.select(", ".join(_JUNK_SELECTORS)):
        junk.decompose()

    self_key = _normalise(url)
    seen: set[str] = set()
    results: list[tuple[str, str]] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        full = urljoin(url, href)
        pu = urlparse(full)
        if pu.scheme not in ("http", "https"):
            continue
        if pu.netloc.lower() != base.netloc.lower():   # same domain only
            continue
        if pu.path.lower().endswith(_ASSET_EXT):        # skip assets/downloads
            continue
        if "/-/" in pu.path:                            # skip Liferay facet/filter links
            continue
        text = " ".join(a.get_text(strip=True).split())
        if not text:                                    # skip empty/icon-only links
            continue
        full = _apply_lang(full, base_lang)             # canonicalise language prefix
        key = _normalise(full)
        if key == self_key or key in seen:             # de-dup + skip self
            continue
        if children_only and not pu.path.rstrip("/").startswith(base_path + "/"):
            continue
        seen.add(key)
        results.append((text, key))

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="List content links on a page.")
    parser.add_argument("--url", required=True, help="Landing/card page URL.")
    parser.add_argument("--children", action="store_true",
                        help="Only links whose path is under the landing page's path.")
    args = parser.parse_args()

    links = discover(args.url, children_only=args.children)
    log.info("Found %d unique link(s) on %s", len(links), args.url)
    print()
    for text, url in links:
        print(f"  {text[:60]:<60}  {url}")
    print()
    # Bare URL list — easy to copy straight into TARGET_URLS.
    print("# --- copy/paste URLs ---")
    for _, url in links:
        print(f'    "{url}",')
    return 0


if __name__ == "__main__":
    sys.exit(main())
