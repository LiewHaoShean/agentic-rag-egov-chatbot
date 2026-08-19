"""Scrape Malaysian gov FAQ pages -> chunk -> translate -> embed -> Supabase.

Rewrite of the old `rag-scapper/targeted_scraper.py` for the NEW `embeddings`
schema. The robust HTML cleaning (Playwright, Cloudflare email decode, link
enrichment, table->markdown, junk-selector stripping) is ported verbatim; the
storage half is reworked:

  OLD (doc_embeddings)            NEW (embeddings)
  -----------------------------   --------------------------------------------
  thenlper/gte-large (local)      embed_passage() -> e5-large-instruct, 1024d API
  file_name / chunk_order         file_url / chunk_index / document_id
  English-only original_text      original_text (source lang) + translate_text (EN)
  blind insert (dupes on re-run)  deterministic document_id + delete-by-url (idempotent)
  --                              category (tax/epf) + public

Run from the PROJECT ROOT so the package imports resolve:
    pip install -r scripts/requirements-ingest.txt && playwright install chromium
    python -m scripts.ingest_scrape              # upload
    python -m scripts.ingest_scrape --preview    # dry-run to previews/
"""
from __future__ import annotations

import argparse
import re
import random
import sys
import time
import uuid
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import LangDetectException, detect
from deep_translator import GoogleTranslator
from playwright.sync_api import sync_playwright

from core.logging import configure_logging, get_logger
from services.embeddings import embed_passage
from services.supabase_client import get_client

configure_logging()
log = get_logger("ingest_scrape")

# Deterministic namespace so the same URL always yields the same document_id
# (lets us delete-then-insert for idempotent re-runs).
_DOC_NS = uuid.uuid5(uuid.NAMESPACE_URL, "egov-rag-ingest")

# ---------------------------------------------------------------- Target pages
TARGET_URLS = [
    # ---- KWSP / EPF ----
    # Intro
    "https://www.kwsp.gov.my/en/member/overview",
    "https://www.kwsp.gov.my/en/member/overview/contribution",
    "https://www.kwsp.gov.my/en/member/overview/registration",
    "https://www.kwsp.gov.my/en/member/overview/benefits",
    # savings
    "https://www.kwsp.gov.my/en/member/savings",
    "https://www.kwsp.gov.my/en/member/savings/mandatory-contribution",
    "https://www.kwsp.gov.my/en/member/savings/self-contribution",
    "https://www.kwsp.gov.my/en/member/savings/i-suri",
    "https://www.kwsp.gov.my/en/member/savings/i-saraan",
    "https://www.kwsp.gov.my/en/member/savings/i-saraan-plus",
    "https://www.kwsp.gov.my/en/member/savings/akaun-persaraan-top-up",
    "https://www.kwsp.gov.my/en/member/savings/voluntary-excess",
    "https://www.kwsp.gov.my/en/member/savings/i-invest",
    "https://www.kwsp.gov.my/en/member/savings/i-sayang",
    # Manage EPF account
    "https://www.kwsp.gov.my/en/member/account-centre",
    "https://www.kwsp.gov.my/en/member/account-centre/nomination",
    "https://www.kwsp.gov.my/en/member/account-centre/simpanan-shariah",
    "https://www.kwsp.gov.my/en/member/account-centre/pensionable-employees",
    "https://www.kwsp.gov.my/en/others/resource-centre/tools",
    "https://www.kwsp.gov.my/en/member/account-centre/account-restructuring",
    "https://www.kwsp.gov.my/en/member/account-centre/unclaimed-contribution",
    "https://www.kwsp.gov.my/en/member/account-centre/transfer-savings",
    "https://www.kwsp.gov.my/en/member/e-kyc",
    "https://www.kwsp.gov.my/en/member/account-centre/death",
    "https://www.kwsp.gov.my/en/member/account-centre/leaving-country",
    # Home Ownership
    "https://www.kwsp.gov.my/en/member/house-withdrawal",
    "https://www.kwsp.gov.my/en/member/house-withdrawal/buy-house",
    "https://www.kwsp.gov.my/en/member/house-withdrawal/build-house",
    "https://www.kwsp.gov.my/en/member/house-withdrawal/loan-instalment",
    "https://www.kwsp.gov.my/en/member/house-withdrawal/reduce-housing-loan",
    "https://www.kwsp.gov.my/en/member/house-withdrawal/flexible-housing",
    "https://www.kwsp.gov.my/en/member/house-withdrawal/pr1ma-housing",
    # Financial Life Stages
    "https://www.kwsp.gov.my/en/member/life-stages",
    "https://www.kwsp.gov.my/en/member/life-stages/akaun-fleksibel-withdrawal",
    "https://www.kwsp.gov.my/en/member/life-stages/education-withdrawal",
    "https://www.kwsp.gov.my/en/member/life-stages/age-50-withdrawal",
    "https://www.kwsp.gov.my/en/member/life-stages/age-55-60-withdrawal",
    "https://www.kwsp.gov.my/en/member/life-stages/hajj-withdrawal",
    "https://www.kwsp.gov.my/en/member/life-stages/million-savings",
    "https://www.kwsp.gov.my/en/member/life-stages/i-legasi",
    # Health Protection
    "https://www.kwsp.gov.my/en/member/healthcare",
    "https://www.kwsp.gov.my/en/member/healthcare/i-lindung",
    "https://www.kwsp.gov.my/en/member/healthcare/critical-illness",
    "https://www.kwsp.gov.my/en/member/healthcare/fertility",
    "https://www.kwsp.gov.my/en/member/healthcare/incapacitation",
    # KWSP Iaccount App
    "https://www.kwsp.gov.my/en/member/kwsp-i-akaun",
    "https://www.kwsp.gov.my/en/member/kwsp-i-akaun/secure",
    "https://www.kwsp.gov.my/en/member/e-kyc",
    # ---- LHDN / tax (site restructured 2026: citizen pages live under /en/individu/) ----
    # Individual
    "https://www.hasil.gov.my/en/individu/pengenalan-cukai-pendapatan-individu",
    "https://www.hasil.gov.my/en/individu/pendaftaran",
    "https://www.hasil.gov.my/en/individu/lapor-pendapatan",
    "https://www.hasil.gov.my/en/individu/pelepasan-cukai",
    "https://www.hasil.gov.my/en/individu/kadar-cukai",
    "https://www.hasil.gov.my/en/individu/bayaran",
    "https://www.hasil.gov.my/en/individu/bayaran/cukai-terlebih-bayar",
    "https://www.hasil.gov.my/en/individu/semakan-dan-kemaskini",
    "https://www.hasil.gov.my/en/individu/sekatan-perjalanan",
    # Company
    "https://www.hasil.gov.my/en/syarikat/anggaran-cukai",
    "https://www.hasil.gov.my/en/syarikat/insentif",
    "https://www.hasil.gov.my/en/syarikat/kadar-cukai-syarikat",
    "https://www.hasil.gov.my/en/syarikat/kemaskini-maklumat-syarikat",
    "https://www.hasil.gov.my/en/syarikat/pembayaran-cukai-syarikat",
    "https://www.hasil.gov.my/en/syarikat/pendaftaran-fail-cukai",
    "https://www.hasil.gov.my/en/syarikat/pindaan-selepas-penghantaran-borang",
    "https://www.hasil.gov.my/en/syarikat/sme",
    # Employers / PCB (MTD)
    "https://www.hasil.gov.my/en/majikan/garis-panduan",
    "https://www.hasil.gov.my/en/majikan/jadual-pcb-dan-spesifikasi-data",
    "https://www.hasil.gov.my/en/majikan/kaedah",
    "https://www.hasil.gov.my/en/majikan/pembayaran-pcb",
    "https://www.hasil.gov.my/en/majikan/pemberitahuan-pekerja-baharu",
    "https://www.hasil.gov.my/en/majikan/pemberitahuan-pemberhentian-kerja",
    "https://www.hasil.gov.my/en/majikan/tanggungjawab-majikan",
    # International
    "https://www.hasil.gov.my/en/antarabangsa/automatic-exchange-of-information-aeoi",
    "https://www.hasil.gov.my/en/antarabangsa/country-by-country-reporting-cbcr",
    "https://www.hasil.gov.my/en/antarabangsa/exchange-of-information",
    "https://www.hasil.gov.my/en/antarabangsa/global-minimum-tax-gmt",
    "https://www.hasil.gov.my/en/antarabangsa/hal-ehwal-antarabangsa",
    "https://www.hasil.gov.my/en/antarabangsa/harga-pindahan",
    "https://www.hasil.gov.my/en/antarabangsa/instrumen-multilateral-mli",
    "https://www.hasil.gov.my/en/antarabangsa/perjanjian-pengelakan-pencukaian-dua-kali-pppdk",
    "https://www.hasil.gov.my/en/antarabangsa/perkiraan-harga-awal-apa",
    "https://www.hasil.gov.my/en/antarabangsa/sijil-taraf-mastautin-e-residence",
    "https://www.hasil.gov.my/en/antarabangsa/tatacara-persetujuan-bersama-map",
    # Legislation
    "https://www.hasil.gov.my/en/perundangan/akta",
    "https://www.hasil.gov.my/en/perundangan/cukai-pegangan",
    "https://www.hasil.gov.my/en/perundangan/garis-panduan",
    "https://www.hasil.gov.my/en/perundangan/kesalahan-denda-dan-penalti",
    "https://www.hasil.gov.my/en/perundangan/ketetapan-umum",
    "https://www.hasil.gov.my/en/perundangan/nota-amalan",
    # Forms & filing programmes
    "https://www.hasil.gov.my/en/borang/format-baucar-dividen",
    "https://www.hasil.gov.my/en/borang/kriteria-bncp-tidak-lengkap-yang-tidak-boleh-diterima",
    "https://www.hasil.gov.my/en/borang/program-memfail-borang-nyata",
    "https://www.hasil.gov.my/en/borang/program-memfail-borang-nyata-ckht",
    "https://www.hasil.gov.my/en/borang/program-memfail-borang-nyata-ckm",
    # Stamp duty / RPGT
    "https://www.hasil.gov.my/en/duti-setem",
    "https://www.hasil.gov.my/en/duti-setem/e-duti-setem",
    "https://www.hasil.gov.my/en/ckht",
    "https://www.hasil.gov.my/en/ckht/tanggungjawab-pelupus-dan-pemeroleh",
    # e-Invoice / e-Services / misc
    "https://www.hasil.gov.my/en/e-invois",
    "https://www.hasil.gov.my/en/e-invois/pelaksanaan-e-invois-di-malaysia/mengenai-e-invois-manfaatnya",
    "https://www.hasil.gov.my/en/e-perkhidmatan",
    "https://www.hasil.gov.my/en/ejen-cukai",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/semakan-kelulusan-derma",
    "https://www.hasil.gov.my/en/tadbir-urus-percukaian-korporat",
    "https://www.hasil.gov.my/en/mengenai-hasil/profil-korporat",
    "https://www.hasil.gov.my/en/polisi-portal/peringatan-penipuan",
    "https://www.hasil.gov.my/en/hubungi-kami/meja-bantuan-hasil",
]

# Map source host -> category column value.
_CATEGORY_BY_HOST = {
    "kwsp.gov.my": "epf",
    "hasil.gov.my": "tax",
}


def category_for(url: str) -> str:
    host = urlparse(url).netloc.lower().lstrip("www.")
    for key, cat in _CATEGORY_BY_HOST.items():
        if key in host:
            return cat
    return "general"


def document_id_for(url: str) -> str:
    return str(uuid.uuid5(_DOC_NS, url))


# ========================================================= HTML cleaning helpers
# (ported from the original targeted_scraper.py — proven against these sites)
def translate_to_english(text: str) -> tuple[str, str]:
    """Detect language; translate to English if needed. Returns (english, lang)."""
    try:
        lang = detect(text)
    except LangDetectException:
        return text, "unknown"
    if lang == "en":
        return text, lang
    try:
        max_chars = 4500  # Google Translate per-call cap
        parts = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
        translated = [
            GoogleTranslator(source="auto", target="en").translate(p) for p in parts
        ]
        return " ".join(t for t in translated if t), lang
    except Exception as exc:  # noqa: BLE001 — keep original on translation failure
        log.warning("Translation failed (%s) — keeping original text", exc)
        return text, lang


def decode_cloudflare_emails(soup: BeautifulSoup) -> None:
    for el in soup.select("[data-cfemail]"):
        encoded = el.get("data-cfemail", "")
        if not encoded:
            continue
        try:
            key = int(encoded[:2], 16)
            decoded = "".join(
                chr(int(encoded[i : i + 2], 16) ^ key)
                for i in range(2, len(encoded), 2)
            )
            el.replace_with(decoded)
        except Exception:  # noqa: BLE001
            pass


def enrich_links(soup: BeautifulSoup, base_url: str) -> None:
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        text = a.get_text(strip=True)
        if not href or href.startswith("#") or href.startswith("javascript"):
            continue
        if "/cdn-cgi/l/email-protection" in href:
            a.replace_with(text)
        elif href.startswith("mailto:"):
            a.replace_with(href.replace("mailto:", ""))
        else:
            full_url = urljoin(base_url, href)
            a.replace_with(f"{text} ({full_url})" if text and text != full_url else full_url)


def convert_tables_to_markdown(soup: BeautifulSoup) -> None:
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True) for td in tr.find_all(["th", "td"])]
            if any(cells):
                rows.append(f"| {' | '.join(cells)} |")
        if not rows:
            table.decompose()
            continue
        header_count = rows[0].count("|") - 1
        separator = f"| {' | '.join(['---'] * header_count)} |"
        rows.insert(1, separator) if len(rows) > 1 else rows.append(separator)
        table.replace_with("\n" + "\n".join(rows) + "\n")


_JUNK_SELECTORS = [
    "header", "footer", "nav", "aside", ".uk-nav", ".uk-navbar",
    "#sp-header", "#sp-bottom", ".sppb-col-sm-3", ".breadcrumb",
    ".mod-languages", ".uk-dropdown", "#sp-footer", ".uk-card-secondary",
    ".tm-header", ".tm-header-mobile", ".uk-offcanvas", ".tm-toolbar",
    "#breadcrumb", "#chooseCat", ".content-nav-wrapper", ".uk-section-secondary",
    ".usn_pod_searchlinks", ".elementor-location-header", ".elementor-location-footer",
    ".elementor-nav-menu", ".elementor-menu-toggle", ".sub-menu", ".elementor-hidden-mobile",
    ".elementor-widget-nav-menu", 'a[href="#content"]', ".skip-link",
    ".elementor-social-icons-wrapper", ".elementor-widget-eael-breadcrumbs", ".fbc-page",
    "#scrollToTopBtn", ".whatsapp-btn", ".help-btn", ".login-link-tag",
    ".language-switcher", ".navigation",
    ".lfr-layout-structure-item-carousel-main-fragment",
]


def _goto_with_retry(page, url: str, *, max_attempts: int = 4):
    """Navigate with exponential backoff on WAF throttling (e.g. 403/429/503).

    Gov sites (KWSP behind a WAF) instant-block rapid sequential automated
    requests. The first page usually passes, then follow-ups get an edge 403 in
    ~50ms. Backing off and retrying lets the rate-limit window reset.
    """
    backoff = 8.0
    last_status = "No Response"
    for attempt in range(1, max_attempts + 1):
        response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        if response and response.status == 200:
            return response
        last_status = response.status if response else "No Response"
        if attempt < max_attempts:
            wait = backoff * attempt + random.uniform(0, 4)
            log.warning(
                "  status=%s on attempt %d/%d — backing off %.1fs",
                last_status, attempt, max_attempts, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to load page after {max_attempts} attempts. "
                       f"Last status: {last_status}")


def extract_clean_text(page, url: str) -> tuple[str, str]:
    """Load with Playwright, strip chrome, return (clean_text, page_title)."""
    _goto_with_retry(page, url)
    page.wait_for_timeout(3000)  # let dynamic content settle

    soup = BeautifulSoup(page.content(), "html.parser")
    for junk in soup.select(", ".join(_JUNK_SELECTORS)):
        junk.decompose()
    decode_cloudflare_emails(soup)
    enrich_links(soup, url)
    convert_tables_to_markdown(soup)
    for p in soup.find_all("p"):
        if "Choose by categories:" in p.get_text():
            p.decompose()

    page_title = soup.title.string.strip() if soup.title else "Unknown Title"

    main_block = None
    for selector in [
        "article", "main", "#main-content", "#main", "#content",
        ".portlet-layout", "#sp-main-body", ".elementor-section:has(.accordions-head)",
    ]:
        found = soup.select_one(selector)
        if found:
            main_block = found
            break
    main_block = main_block or soup

    clean_text = main_block.get_text(separator="\n", strip=True)
    return clean_text, page_title


# ======================================================== Post-extraction cleaning
# Exact nav/UI labels to drop (case-insensitive).
_NAV_NOISE = {
    "breadcrumb", "view", "view more", "read more", "show more", "less",
    "frequently asked question", "frequently asked questions", "faq",
    "asset publisher",  # Liferay widget label
}

# Short exact-match CMS placeholder lines (checked lowercased).
_CMS_PLACEHOLDER_LINES = {
    "title 1", "title 2", "title 3",
    "text 1", "text 2", "text 1.2", "text 2.2",
}

# Inline placeholder substrings stripped from the full text before line
# splitting — they appear inside table cells so line-level filtering misses them.
_INLINE_PLACEHOLDERS = [
    # Liferay paragraph field placeholder (inside table cells on KWSP pages).
    (
        "A paragraph is a self-contained unit of a discourse in writing dealing "
        "with a particular point or idea. Paragraphs are usually an expected part "
        "of formal writing, used to organize longer prose."
    ),
    # Liferay asset-link component renders the link text as "Go Somewhere" followed
    # by a date string (e.g. "Go Somewhere 02/03/11 00:00 AM").
    "Go Somewhere",
]

# Regex-based placeholder patterns: list of (compiled_pattern, replacement).
# Used for patterns that appear mid-line (e.g. inside table cells) where simple
# string replacement would be too blunt.
_INLINE_PLACEHOLDER_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Liferay date-time remnant left after "Go Somewhere" removal
    # (e.g. " 02/03/11 00:00 AM ✓" → " ✓").
    (re.compile(r"\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}\s+[AP]M"), ""),
    # Liferay table skeleton rows where every cell is the placeholder word "content".
    (re.compile(r"^\|\s*content(\s*\|\s*content\s*)+\s*\|$", re.MULTILINE | re.IGNORECASE), ""),
    # "heading" as the sole content of a table cell (last-column header placeholder).
    (re.compile(r"\|\s*heading\s*\|", re.IGNORECASE), "|"),
]

# Everything from this sentinel onward is editorial/promotional noise.
_TRAILING_NOISE_SENTINEL = "related articles"

# Pages with fewer than this many chars after cleaning are pure navigation or
# card landing pages (e.g. a tools index with only 4 calculator links).
_MIN_PAGE_CHARS = 200
# Lines >= this length are de-duplicated GLOBALLY (kills repeated FAQ blocks).
# Shorter lines only have consecutive duplicates collapsed (keeps real headers).
_MIN_DEDUP_LEN = 12

# Invisible chars KWSP sprinkles everywhere — strip so they don't (a) pollute
# embeddings or (b) defeat the de-dup key ("RM99,688." vs "RM99,688.​").
_INVISIBLES = str.maketrans({"​": "", "﻿": "", " ": " ", "⁠": ""})


def _dedup_key(line: str) -> str:
    # whitespace-insensitive, punctuation-trimmed, lowercased
    return " ".join(line.lower().split()).strip(" .:;,-•●*")


def _is_noise(line: str) -> bool:
    low = line.lower().strip()
    if low in _NAV_NOISE:
        return True
    if low in _CMS_PLACEHOLDER_LINES:
        return True
    # Widget labels like "... Accordion)" / "Age 50 Withdrawal Accordion" /
    # "i-Simpan FAQ EN" / "Voluntary Contribution FAQ Accordion".
    if low.endswith("accordion") or low.endswith("accordion)"):
        return True
    if low.endswith("faq en") or low.endswith("faq accordion"):
        return True
    return False


def normalize_text(text: str) -> str:
    """Strip nav noise and de-duplicate repeated content before chunking.

    KWSP pages render the same FAQ accordion several times, tripling the chunk
    count with near-identical text — bad for embedding cost and retrieval (RRF
    top-K fills with duplicates). We:
      1. strip inline CMS placeholder substrings (appear inside table cells),
      2. truncate at the "Related Articles" sentinel (editorial promo section),
      3. drop nav/UI label lines and short CMS placeholder lines,
      4. collapse consecutive duplicate lines,
      5. globally de-duplicate substantial lines (>= _MIN_DEDUP_LEN chars).
    """
    # Step 1: remove inline placeholder substrings before line splitting.
    for placeholder in _INLINE_PLACEHOLDERS:
        text = text.replace(placeholder, "")

    # Step 1b: regex-based placeholder patterns (mid-line / table-cell artifacts).
    for pattern, replacement in _INLINE_PLACEHOLDER_PATTERNS:
        text = pattern.sub(replacement, text)

    # Step 2: truncate at the "Related Articles" promotional section.
    sentinel_pos = text.lower().find(_TRAILING_NOISE_SENTINEL)
    if sentinel_pos != -1:
        text = text[:sentinel_pos]

    seen: set[str] = set()
    out: list[str] = []
    prev_key: str | None = None

    for raw in text.split("\n"):
        line = raw.translate(_INVISIBLES).strip()
        if not line or _is_noise(line):
            continue
        key = _dedup_key(line)
        if not key:
            continue
        if key == prev_key:  # collapse consecutive duplicates
            continue
        if len(line) >= _MIN_DEDUP_LEN:
            if key in seen:  # global block de-dup
                continue
            seen.add(key)
        out.append(line)
        prev_key = key

    return "\n".join(out)


# ===================================================================== Pipeline
def process_url(page, url: str, splitter, *, preview: bool) -> None:
    log.info("Scraping %s", url)
    try:
        clean_text, page_title = extract_clean_text(page, url)
    except Exception as exc:  # noqa: BLE001
        log.error("Skip %s: %s", url, exc)
        return

    if not clean_text or len(clean_text) < 10:
        log.warning("No usable text at %s", url)
        return

    # Strip nav noise + de-duplicate repeated blocks BEFORE chunking.
    raw_len = len(clean_text)
    clean_text = normalize_text(clean_text)
    log.info("  cleaned %d -> %d chars", raw_len, len(clean_text))

    if len(clean_text) < _MIN_PAGE_CHARS:
        log.warning("  skipping — only %d chars after cleaning (nav-only page)", len(clean_text))
        return

    # Chunk the ORIGINAL-language text so original_text/translate_text stay aligned.
    docs = [Document(page_content=clean_text, metadata={"title": page_title})]
    chunks = [c.page_content.strip() for c in splitter.split_documents(docs)]
    chunks = [c for c in chunks if len(c) >= 15]
    log.info("  %d chunks", len(chunks))

    category = category_for(url)
    document_id = document_id_for(url)

    # ---- Preview mode: write to previews/ and stop (no API/DB calls) ----
    if preview:
        import os

        os.makedirs("previews", exist_ok=True)
        slug = url.strip("/").split("/")[-1] or "index"
        with open(f"previews/{slug}.txt", "w", encoding="utf-8") as fh:
            fh.write(f"URL: {url}\nTITLE: {page_title}\nCATEGORY: {category}\n")
            fh.write(f"DOCUMENT_ID: {document_id}\n" + "=" * 50 + "\n\n")
            for i, c in enumerate(chunks):
                fh.write(f"--- CHUNK {i} ---\n{c}\n\n")
        log.info("  preview -> previews/%s.txt", slug)
        return

    # ---- Upload mode ----
    rows = []
    for index, original_text in enumerate(chunks):
        translate_text, lang = translate_to_english(original_text)
        if not translate_text.strip():
            # Translator can return empty for link-only/symbol chunks; the
            # embeddings API 400s on empty input. Fall back to the original.
            translate_text = original_text
        try:
            vector = embed_passage(translate_text)  # 1024-dim, NO instruct prefix
        except Exception as exc:  # noqa: BLE001 — one bad chunk must not kill the run
            log.warning("  chunk %d embed failed (%s) — skipping chunk", index, exc)
            continue
        rows.append(
            {
                "document_id": document_id,
                "chunk_index": index,
                "embedding_vector": vector,
                "original_text": original_text,
                "translate_text": translate_text,
                "current_language": lang,
                "file_url": url,
                "category": category,
                "public": True,
                # summary / user_id / message_id left NULL (nullable);
                # fts is a GENERATED column — never inserted.
            }
        )

    if not rows:
        return

    client = get_client()
    # Idempotent: clear any prior chunks for this page before re-inserting.
    client.table("embeddings").delete().eq("file_url", url).execute()
    client.table("embeddings").insert(rows).execute()
    log.info("  stored %d chunks (category=%s)", len(rows), category)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape gov FAQs into Supabase embeddings.")
    parser.add_argument(
        "--preview", action="store_true", help="Dry run: write chunks to previews/ only."
    )
    parser.add_argument(
        "--url", action="append", help="Override target URL(s); repeatable."
    )
    parser.add_argument(
        "--delay", type=float, default=12.0,
        help="Base seconds between pages (randomized +/- to dodge WAF throttling).",
    )
    args = parser.parse_args()

    urls = args.url or TARGET_URLS
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    log.info("Mode=%s | %d URL(s)", "PREVIEW" if args.preview else "UPLOAD", len(urls))

    user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for i, url in enumerate(urls):
            # FRESH context per URL: the WAF flags a session after its first
            # request, so a clean cookie jar per page makes each look like a
            # first-time visitor (the only request type it reliably allows).
            context = browser.new_context(
                user_agent=user_agent,
                locale="en-US",
                viewport={"width": 1366, "height": 900},
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9,ms;q=0.8"},
            )
            page = context.new_page()
            try:
                process_url(page, url, splitter, preview=args.preview)
            finally:
                context.close()
            # Randomized polite delay between pages (skip after the last one).
            if i < len(urls) - 1:
                wait = args.delay + random.uniform(-3, 5)
                log.info("  waiting %.1fs before next page", max(wait, 1.0))
                time.sleep(max(wait, 1.0))
        browser.close()

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
