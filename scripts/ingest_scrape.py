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
    # ---- KWSP: gaps found by walking the Liferay sitemap at /en/sitemap.xml ----
    # curl gets a 403 on every sitemap path because Cloudflare refuses non-browser
    # clients, so this needed a real browser to reach. The member journeys turned
    # out to be well covered already; what was missing was the legal reference and
    # the employer side. Employer pages are included deliberately even though the
    # assistant answers members: "my employer has not paid my EPF" is a member's
    # question, and the obligation it turns on is only documented here. The ~85
    # /w/ items (news, announcements, scam alerts) are excluded as dated content.
    "https://www.kwsp.gov.my/en/others/resource-centre",
    "https://www.kwsp.gov.my/en/others/resource-centre/references",
    "https://www.kwsp.gov.my/en/others/resource-centre/references/epf-act-1991",
    "https://www.kwsp.gov.my/en/employer/introduction",
    "https://www.kwsp.gov.my/en/employer/responsibilities",
    "https://www.kwsp.gov.my/en/employer/responsibilities/compliance",
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
    # ---- PERKESO / SOCSO (social security, employment injury, invalidity) ----
    "https://www.perkeso.gov.my/ejen-keselamatan-sosial-pekerjaan-sendiri.html",
    "https://www.perkeso.gov.my/geran-padanan-caruman-sksps.html",
    "https://www.perkeso.gov.my/hubungi-kami/saluran-hubungan/soalan-lazim.html",
    "https://www.perkeso.gov.my/mengenai-kami/maklumat-korporat/prinsip-perlindungan-keselamatan-sosial.html",
    "https://www.perkeso.gov.my/mengenai-kami/maklumat-korporat/profil.html",
    "https://www.perkeso.gov.my/mengenai-kami/rujukan/akta-peraturan.html",
    "https://www.perkeso.gov.my/mengenai-kami/rujukan/borang-borang.html",
    "https://www.perkeso.gov.my/mengenai-kami/rujukan/tarikh-bayaran-faedah.html",
    "https://www.perkeso.gov.my/pembayaran-caruman.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/majikan-pekerja/caruman.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/majikan-pekerja/kadar-caruman.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/majikan-pekerja/pembayaran.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/majikan-pekerja/pendaftaran-majikan.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/majikan-pekerja/penguatkuasaan.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/insurans-pekerjaan.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/lindung-24-jam.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/pekerja-asing.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/pekerja-bermajikan.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/pekerja-domestik.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/pekerjaan-sendiri.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/permohonan-faedah.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/skim-bencana-pekerjaan.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/skim-keilatan.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perlindungan/suri-rumah.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perubatan/klinik-panel.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perubatan/perkhidmatan-pemulihan.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perubatan/program-pemandu-sihat-selamat.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perubatan/program-pengurusan-hilang-upaya-perkeso.html",
    "https://www.perkeso.gov.my/perkhidmatan-kami/perubatan/pusat-dialisis.html",
    "https://www.perkeso.gov.my/perkhidmatan-pekerjaan.html",
    # ---- JPN / National Registration: FAQs ----
    "https://www.jpn.gov.my/soalan-lazim/slanakangkat",
    "https://www.jpn.gov.my/soalan-lazim/slkadpengenalan",
    "https://www.jpn.gov.my/soalan-lazim/slkelahiran",
    "https://www.jpn.gov.my/soalan-lazim/slkematian",
    "https://www.jpn.gov.my/soalan-lazim/slperkahwinan",
    "https://www.jpn.gov.my/soalan-lazim/slwarganegara",
    "https://www.perkeso.gov.my/hubungi-kami/saluran-hubungan/soalan-lazim.html",
    # ---- JPN / National Registration: service pages (MyKad, birth, death,
    #      marriage, divorce, adoption, citizenship) ----
    "https://www.jpn.gov.my/dokumen-pengenalan-diri",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-cabutan-daftar-kad-pengenalan-dan-cabutan-daftar-alamat",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-daftar-lewat-kad-pengenalan-mykad-atau-mypr",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-gantian-hilang-kad-pengenalan-mykad-atau-mypr",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-gantian-kad-pengenalan-mykad-atau-mypr-18-tahun-warganegara-atau-bukan-warganegara",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-gantian-kad-pengenalan-mypr-bukan-warganegara",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-gantian-rosak-kad-pengenalan",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-gantian-tukar-alamat-kad-pengenalan",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-kad-pengenalan-bagi-orang-baru-tiba-obt",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-kad-pengenalan-mypr-kanak-kanak-12-tahun-atau-kali-pertama-bukan-warganegara",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-kad-pengenalan-pesara-polis-atau-tentera",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-mykad-bagi-kanak-kanak-12-tahun",
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/permohonan-pindaan-butiran-kad-pengenalan",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/cabutan-sijil-kelahiran-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/cabutan-sijil-kelahiran-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/carian-cabutan-daftar-kelahiran-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/carian-cabutan-sijil-kelahiran-di-hedjaz-semenajung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/carian-cabutan-sijil-kelahiran-luar-negara-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-biasa-kelahiran-mati-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-kelahiran-biasa-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-kelahiran-biasa-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-kelahiran-biasa-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-kelahiran-lambat-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-kelahiran-lambat-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-kelahiran-mati-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-kelahiran-semasa-menunaikan-haji-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-lambat-kelahiran-mati-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-lewat-kelahiran-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-lewat-kelahiran-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/pendaftaran-lewat-kelahiran-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-carian-dalam-daftar-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-daftar-semula-kelahiran-hilang-atau-musnah-seksyen-4a-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-daftar-semula-orang-yang-disahtaraf-seksyen-17-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-12-dalam-sijil-kelahiran-bagi-kemasukan-nama-kanak-kanak-kurang-daripada-satu-1-tahun-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-13-dalam-sijil-kelahiran-bagi-pengubahan-nama-kanak-kanak-di-bawah-satu-1-tahun-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-15-2-b-dalam-sijil-kelahiran-bagi-kemasukan-nama-kanak-kanak-berumur-21-tahun-dan-lebih-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-151-dalam-sijil-kelahiran-bagi-pengubahan-nama-kanak-kanak-berumur-di-bawah-1-tahun-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-152a-dalam-sijil-kelahiran-bagi-kemasukan-nama-kanak-kanak-berumur-di-bawah-21-tahun-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-252-dalam-sijil-kelahiran-bagi-kesilapan-perkeranian-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-253-dalam-sijil-kelahiran-bagi-pembetulan-kesilapan-fakta-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-30-dalam-sijil-kelahiran-bagi-kesilapan-fakta-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-30-dalam-sijil-kelahiran-bagi-kesilapan-perkeranian-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kematian/cabutan-sijil-kematian-di-hedjaz-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/cabutan-sijil-kematian-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kematian/cabutan-sijil-kematian-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kematian/carian-cabutan-daftar-kematian-luar-negara-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/carian-cabutan-daftar-kematian-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/laporan-anggapan-kematian-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kematian/laporan-anggapan-kematian-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kematian/laporan-anggapan-kematian-semanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/laporan-kematian-luar-negara-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kematian/laporan-kematian-luar-negara-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-biasa-kematian-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-biasa-kematian-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-biasa-kematian-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-kematian-semasa-menunaikan-haji-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-lambat-kematian-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-lambat-kematian-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-lewat-kematian-sabah",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-lewat-kematian-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kematian/pendaftaran-lewat-kematian-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/permohonan-carian-dalam-daftar-kematian-seksyen-31-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/permohonan-pembetulan-maklumat-seksyen-25-2-dalam-sijil-kematian-bagi-pembetulan-kesilapan-perkeranian-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kematian/permohonan-pembetulan-maklumat-seksyen-25-3-dalam-sijil-kematian-bagi-pembetulan-kesilapan-fakta-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/kematian/permohonan-pembetulan-maklumat-seksyen-272-dalam-sijil-kematian-bagi-kesilapan-perkeranian-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/permohonan-pembetulan-maklumat-seksyen-273-dalam-sijil-kematian-bagi-pembetulan-kesilapan-fakta-semenanjung",
    "https://www.jpn.gov.my/perkhidmatan/kematian/permohonan-pembetulan-maklumat-seksyen-31-2-dalam-sijil-kematian-bagi-kesilapan-perkeranian-sabah",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/permohonan-cabutan-sijil-kelahiran-anak-angkat-melalui-perintah-mahkamah",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/permohonan-carian-dan-cabutan-daftar-pengangkatan-de-facto",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/permohonan-pendaftaran-pengangkatande-facto-di-semenanjung-malaysia",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/permohonan-pengangkatan-di-negeri-sabah",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/permohonan-pengangkatan-di-negeri-sarawak",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/permohonan-pengangkatan-melalui-perintah-mahkamah-di-semenanjung-malaysia",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/permohonan-pindaan-maklumat-dalam-daftar-pengangkatan-de-facto",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/pendaftaran-perkahwinan-di-malaysia-bagi-pemohon-bukan-beragama-islam-dengan-permohonan-tanpa-lesen-di-bawah-akta-membaharui-undang-undang-perkahwinan-dan-perceraian-1976-akta-164",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/pendaftaran-perkahwinan-di-malaysia-bagi-pemohon-bukan-beragama-islam-melalui-permohonan-lesen-atau-persetujuan-perkahwinan-di-bawah-akta-membaharui-undang-undang-perkahwinan-dan-perceraian-1976-ak",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/pendaftaran-perkahwinan-di-penolong-pendaftaran-perkahwinan-kuil-gereja-persatuan-agama-di-bawah-aktar-membaharui-undang-undang-perkahwinan-dan-perceraian-1976-akta-164",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/pendaftaran-semula-perkahwinan-bagi-pasangan-bukan-islam-yang-telah-berkahwin-mengikut-sesuatu-undang-undang-agama-adat-atau-kelaziman-sebelum-1-3-1982-di-bawah-akta-membaharui-undang-undang-perkah",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/pendaftaran-semula-perkahwinan-luar-negara-di-bawah-akta-membaharui-undang-undang-perkahwinan-dan-perceraian-1976-akta-164-bagi-pasangan-bukan-islam-yang-telah-didaftarkan-sebelum-atau-selepas-1-3",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/perceraian",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/perceraian/pengemaskinian-rekod-perkahwinan-perceraian-pembatalan",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/permohonan-carian-atau-cabutan-daftar-perkahwinan",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/permohonan-kaveat-semasa-borang-permohonan-perkahwinan-dipamerkan",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/permohonan-pembetulan-kesilapan-dalam-daftar-perkahwinan",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/permohonan-permohonan-prosiding-tribunal-perkahwinan-di-bawah-akta-membaharui-undang-undang-perkahwinan-dan-perceraian-1976-akta-164",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/permohonan-sebagai-penolong-pendaftar-perkahwinan-di-gereja-kuil-persatuan-agama-di-bawah-akta-membaharui-undang-undang-perkahwinan-dan-perceraian-1976-akta-164",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/permohonan-surat-pengesahan-taraf-perkahwinan",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/persetujuan-perkawinan-bagi-seseorang-yang-telah-genap-18-tahun-tetapi-belum-genap-21-tahun-jpn-kc01b",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-cabutan-atau-salinan-diperakui-sah-dalam-daftar-kewarganegaraan",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-carian-dalam-daftar-kewarganegaraan",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-gantian-sijil-kewarganegaraan",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-pelepasan-taraf-kewarganegaraan-di-bawah-perkara-23-perlembagaan-persekutuan",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-pindaan-butiran-dalam-dokumen-kewarganegaraan",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-taraf-kewarganegaraan-di-bawah-perkara-14-perlembagaan-persekutuan-kelahiran-luar-negara",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-taraf-kewarganegaraan-di-bawah-perkara-14-perlembagaan-persekutuan-kewarganegaraan-kelahiran-dalam-negara",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-taraf-kewarganegaraan-di-bawah-perkara-151-perlembagaan-persekutuan-isteri-kepada-warganegara",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-taraf-kewarganegaraan-di-bawah-perkara-152-perlembagaan-persekutuan-anak-kepada-warganegara-yang-berumur-kurang-daripada-21-tahun",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-taraf-kewarganegaraan-di-bawah-perkara-15a-perlembagaan-persekutuan-pendaftaran-kewarganegaraan-dalam-keadaan-khas",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-taraf-kewarganegaraan-di-bawah-perkara-16-perlembagaan-persekutuan-seorang-yang-lahir-di-persekutuan-sebelum-hari-merdeka-31-ogos-1957",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/permohonan-taraf-kewarganegaraan-di-bawah-perkara-19-perlembagaan-persekutuan-berumur-21-tahun-atau-lebih",

    # ---- JPN: MyKad information (added after evaluation revealed a gap) ----
    # The corpus covered how to APPLY for and REPLACE a MyKad but contained no
    # page explaining what one IS, because JPN's service pages assume the
    # reader knows. Definitional queries ("what is IC", "what is MyKad")
    # therefore retrieved KWSP and LHDN pages, which mention MyKad constantly
    # as a required document and so matched the term more densely than any JPN
    # page did. These are the content-bearing children of the MyKad hub at
    # /informasi/mykad/; its technical children (command set, card readers,
    # postal address) are deliberately excluded as citizen-irrelevant.
    "https://www.jpn.gov.my/informasi/pengenalan-kepada-mykad/",
    "https://www.jpn.gov.my/pelbagai-kegunaan-mykad/",
    "https://www.jpn.gov.my/mykad-kelebihan-di-tangan-anda/",
    "https://www.jpn.gov.my/aplikasi-utama/",
    "https://www.jpn.gov.my/aplikasi-tambahan/",
    "https://www.jpn.gov.my/struktur-baru-mykad/",

    # ---- JPN: coverage gaps found by diffing TARGET_URLS against the site's
    #      own wp-sitemap.xml (see scripts/discover_jpn.py) ----
    # JPN publishes 457 Bahasa Melayu pages; the list above reached 112 of them.
    # Most of the remainder is corporate material with no citizen question behind
    # it (galleries, tenders, org charts, policy notices) and the ~220 individual
    # branch-office pages are deliberately left out, since they are near-identical
    # address blocks that would roughly double the JPN chunk count while matching
    # none of the procedural queries the chatbot actually receives. What follows
    # is the part of the gap that citizens do ask about.
    #
    # The other three identity documents. The corpus described MyKad in detail
    # but held nothing at all on the cards issued to children, temporary
    # residents and permanent residents, so "what is MyKid" had no source to
    # ground against. Note these are the /informasi/ routes; /mykid/, /mykas/
    # and /mypr/ are aliases of the same pages and are not listed twice.
    "https://www.jpn.gov.my/informasi/mykid/",
    "https://www.jpn.gov.my/informasi/mykas/",
    "https://www.jpn.gov.my/informasi/mypr/",
    # Practical service information rather than procedure. Counter hours and the
    # client charter answer "when are you open" and "how long will this take",
    # neither of which any procedural page states; the state and country codes
    # explain the digits of the IC number itself.
    "https://www.jpn.gov.my/hubungi-kami/waktu-operasi-kaunter/",
    "https://www.jpn.gov.my/hubungi-kami/senarai-pejabat-jpn-pusat-cetakan-teragih/",
    "https://www.jpn.gov.my/informasi/piagam-pelanggan/",
    "https://www.jpn.gov.my/informasi/kod-negeri/",
    "https://www.jpn.gov.my/informasi/kod-negara/",
    # Online and outreach channels.
    "https://www.jpn.gov.my/e-services/",
    "https://www.jpn.gov.my/e-services/myphone-in/",
    "https://www.jpn.gov.my/perkhidmatan/mekar/",
    # Birth certificate corrections for Peninsular Malaysia. The list above had
    # the Sarawak (s.25) and Sabah (s.30) equivalents but skipped the Peninsular
    # sections, so the most populous region was the one not covered.
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-272-dalam-sijil-kelahiran-bagi-kesilapan-perkeranian-semenanjung/",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/permohonan-pembetulan-maklumat-seksyen-273dalam-sijil-kelahiran-bagi-pembetulan-kesilapan-fakta-semenanjung/",
    # Category hub pages. Each lists every service under it, which gives a
    # broad question ("what can JPN do about marriage") something to match
    # before it has to pick one specific procedure.
    "https://www.jpn.gov.my/perkhidmatan/kad-pengenalan/",
    "https://www.jpn.gov.my/perkhidmatan/kelahiran/",
    "https://www.jpn.gov.my/perkhidmatan/kematian/",
    "https://www.jpn.gov.my/perkhidmatan/perkahwinan/",
    "https://www.jpn.gov.my/perkhidmatan/pengangkatan/",
    "https://www.jpn.gov.my/perkhidmatan/warganegara/",
    # The one FAQ topic the six already listed did not include.
    "https://www.jpn.gov.my/soalan-lazim/sladuan/",
    # NOTE: https://www.jpn.gov.my/informasi/mykad/ is deliberately NOT listed.
    # That URL is claimed by knowledge/jpn-what-is-mykad.md, whose front-matter
    # sets it as source_url, and ingestion is delete-by-file_url before insert.
    # Scraping it therefore overwrites the curated trilingual definition with the
    # hub page's navigation text, which is exactly the definitional content the
    # curated document was written to supply. Its
    # /informasi/mykad/struktur-baru-mykad/ child is also absent: the card
    # diagram is an image and the page yields 19 characters of text.

    # ---- LHDN: coverage gaps found by diffing TARGET_URLS against
    #      hasil.gov.my/sitemap_index.xml (618 content pages against the 60
    #      originally listed). The e-Invoice programme is deliberately left out:
    #      it is 259 pages aimed at businesses issuing invoices, not at the
    #      individual taxpayer this assistant answers for. Corporate, careers,
    #      international treaty and office-location pages are excluded too.
    #      These use the /en/ routes, matching the LHDN entries above and
    #      giving English source text directly, so no machine translation runs.
    # Individual taxpayer pages the original list did not reach. Appeals,
    # offences and penalties, payment methods, residency status, rebates and
    # the individual FAQ are all core to a citizen-facing assistant.
    "https://www.hasil.gov.my/en/individu/",
    "https://www.hasil.gov.my/en/individu/bayaran/baki-cukai-kena-bayar/",
    "https://www.hasil.gov.my/en/individu/bayaran/bayaran-pendahuluan-anggaran-cukai/",
    "https://www.hasil.gov.my/en/individu/bayaran/kaedah-pembayaran/",
    "https://www.hasil.gov.my/en/individu/bayaran/kenaikan-cukai-lewat-bayar/",
    "https://www.hasil.gov.my/en/individu/derma-hadiah/",
    "https://www.hasil.gov.my/en/individu/kesalahan/",
    "https://www.hasil.gov.my/en/individu/lapor-pendapatan/dalam-talian/",
    "https://www.hasil.gov.my/en/individu/lapor-pendapatan/individu-tidak-bermastautin/",
    "https://www.hasil.gov.my/en/individu/lapor-pendapatan/jenis-taksiran/",
    "https://www.hasil.gov.my/en/individu/lapor-pendapatan/kod-perniagaan/",
    "https://www.hasil.gov.my/en/individu/lapor-pendapatan/manual/",
    "https://www.hasil.gov.my/en/individu/lapor-pendapatan/tempoh-melapor/",
    "https://www.hasil.gov.my/en/individu/penamatan-perkhidmatan/",
    "https://www.hasil.gov.my/en/individu/pindaan-taksiran/",
    "https://www.hasil.gov.my/en/individu/rayuan/",
    "https://www.hasil.gov.my/en/individu/rayuan/permohonan-relif-dibawah-acp/",
    "https://www.hasil.gov.my/en/individu/rayuan/prosiding-resolusi-pertikaian-prp/",
    "https://www.hasil.gov.my/en/individu/rebat/",
    "https://www.hasil.gov.my/en/individu/soalan-lazim-individu/",
    "https://www.hasil.gov.my/en/individu/taraf-mastautin/",
    # Real Property Gains Tax. Nothing in the corpus covered it, so any
    # question about selling a house had no source to ground against.
    "https://www.hasil.gov.my/en/ckht/harga-pelupusan-dan-harga-pemerolehan/",
    "https://www.hasil.gov.my/en/ckht/harga-pelupusan-dianggap-bersamaan-dengan-harga-pemerolehan/",
    "https://www.hasil.gov.my/en/ckht/jenis-borang-nyata-ckht/",
    "https://www.hasil.gov.my/en/ckht/kadar-cukai-keuntungan-harta-tanah/",
    "https://www.hasil.gov.my/en/ckht/pegangan-dan-remitan-wang-oleh-pemeroleh/",
    "https://www.hasil.gov.my/en/ckht/penentuan-keuntungan-yang-boleh-dikenakan-cukai-atau-kerugian-dibenarkan/",
    "https://www.hasil.gov.my/en/ckht/pengecualian/",
    "https://www.hasil.gov.my/en/ckht/pengenaan-penalti-dan-kenaikan-atas-taksiran-cukai/",
    "https://www.hasil.gov.my/en/ckht/pengenalan-sistem-taksir-sendiri-ckht-sts-ckht/",
    "https://www.hasil.gov.my/en/ckht/pindaan-borang-nyata-ckht/",
    "https://www.hasil.gov.my/en/ckht/pindah-milik-harta-tanah-yang-dipusakai/",
    "https://www.hasil.gov.my/en/ckht/prosedur-bayaran-ckht/",
    "https://www.hasil.gov.my/en/ckht/prosedur-pengemukaan-borang-nyata-cukai-keuntungan-harta-tanah/",
    "https://www.hasil.gov.my/en/ckht/saham-dalam-syarikat-harta-tanah-sht/",
    "https://www.hasil.gov.my/en/ckht/taksiran-cukai-keuntungan-harta-tanah/",
    "https://www.hasil.gov.my/en/ckht/tarikh-pelupusan-dan-tarikh-pemerolehan-2/",
    # Stamp duty, likewise absent. It applies to buying property and to
    # ordinary agreements, so it comes up more often than company tax does.
    "https://www.hasil.gov.my/en/duti-setem/adjudikasi-surat-cara/",
    "https://www.hasil.gov.my/en/duti-setem/kaedah-penyeteman/",
    "https://www.hasil.gov.my/en/duti-setem/penalti-duti-setem/",
    "https://www.hasil.gov.my/en/duti-setem/pengecualian-dan-relief/",
    "https://www.hasil.gov.my/en/duti-setem/pengenalan-duti-setem/",
    "https://www.hasil.gov.my/en/duti-setem/perintah-duti-setem/",
    "https://www.hasil.gov.my/en/duti-setem/perintah-penerangan-duti-setem/",
    "https://www.hasil.gov.my/en/duti-setem/sistem-taksir-sendiri-duti-setem-stsds/",
    "https://www.hasil.gov.my/en/duti-setem/soalan-lazim/",
    "https://www.hasil.gov.my/en/duti-setem/tanggungjawab-membayar-duti/",
    "https://www.hasil.gov.my/en/duti-setem/tanggungjawab-pelanggan-utama-peguam-setiausaha-syarikat-pendaftar-institusi-perbankan-dan-kewangan/",
    # Form download pages and the form index.
    "https://www.hasil.gov.my/en/muat-turun-borang/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-ckht/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-cukai-keuntungan-modal/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-cukai-pegangan/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-duti-setem/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-entiti-labuan/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-individu/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-lain-lain-borang/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-majikan/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-pendaftaran/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-petroleum/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-selain-syarikat-selain-individu/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-surat-penyelesaian-cukai/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-syarikat/",
    "https://www.hasil.gov.my/en/muat-turun-borang/muat-turun-borang-tuntutan-galakan/",
    "https://www.hasil.gov.my/en/borang/",
    "https://www.hasil.gov.my/en/borang/program-memfail-dokumen-yang-ditentukan-di-bawah-seksyen-22eb-acapl-1990-melalui-mitrs/",
    "https://www.hasil.gov.my/en/borang/program-memfail-dokumen-yang-ditentukan-di-bawah-seksyen-82b-acp-1967-melalui-mitrs/",
    "https://www.hasil.gov.my/en/borang/program-memfail-dokumen-yang-ditentukan-di-bawah-seksyen-82b-acp-1967-melalui-mitrs/tahun-taksiran-2025/",
    "https://www.hasil.gov.my/en/borang/program-memfail-dokumen-yang-ditentukan-di-bawah-seksyen-82b-acp-1967-melalui-mitrs/tahun-taksiran-2026/",
    "https://www.hasil.gov.my/en/borang/program-memfail-penyata-keuntungan-oleh-suatu-entiti-labuan-bagi-tahun-taksiran-2025-di-bawah-sistem-taksir-sendiri/",
    # Company tax. Outside a strictly citizen-facing reading, but included
    # deliberately since small business owners ask through the same channel.
    "https://www.hasil.gov.my/en/syarikat/",
    "https://www.hasil.gov.my/en/syarikat/cukai-koperasi/",
    "https://www.hasil.gov.my/en/syarikat/cukai-korporat/",
    "https://www.hasil.gov.my/en/syarikat/double-deduction-for-promotion-of-exports/",
    "https://www.hasil.gov.my/en/syarikat/insentif/industrial-adjustment-allowance-iaa/",
    "https://www.hasil.gov.my/en/syarikat/insentif/infrasructure-allowance/",
    "https://www.hasil.gov.my/en/syarikat/insentif/investment-tax-allowance/",
    "https://www.hasil.gov.my/en/syarikat/insentif/pioneer-status/",
    "https://www.hasil.gov.my/en/syarikat/jadual-kod-bayaran-di-bank-ejen-lhdnm-dan-pos-malaysia/",
    "https://www.hasil.gov.my/en/syarikat/lain-lain-situasi/",
    "https://www.hasil.gov.my/en/syarikat/mitrs/",
    "https://www.hasil.gov.my/en/syarikat/pembayaran-cukai-syarikat/kaedah-pembayaran/",
    "https://www.hasil.gov.my/en/syarikat/pembayaran-cukai-syarikat/memohon-bayaran-baki-cukai-secara-ansuran/",
    "https://www.hasil.gov.my/en/syarikat/pembayaran-cukai-syarikat/pembayaran-cukai/",
    "https://www.hasil.gov.my/en/syarikat/pembayaran-cukai-syarikat/semak-kedudukan-cukai/",
    "https://www.hasil.gov.my/en/syarikat/pengecualian-cukai/",
    "https://www.hasil.gov.my/en/syarikat/perniagaan-digital/",
    "https://www.hasil.gov.my/en/syarikat/perniagaan-digital/e-daftar/",
    "https://www.hasil.gov.my/en/syarikat/pertukaran-tarikh-penutupan-akaun-syarikat/",
    "https://www.hasil.gov.my/en/syarikat/rayuan/",
    "https://www.hasil.gov.my/en/syarikat/rayuan/permohonan-relif-dibawah-acp/",
    "https://www.hasil.gov.my/en/syarikat/rayuan/prosiding-resolusi-pertikaian-prp/",
    "https://www.hasil.gov.my/en/syarikat/soalan-lazim-syarikat/",
    "https://www.hasil.gov.my/en/syarikat/syarikat-tidak-bermastautin/",
    "https://www.hasil.gov.my/en/syarikat/tanggungjawab-pembayar-cukai/",
    "https://www.hasil.gov.my/en/syarikat/taraf-mastautin-syarikat/",
    "https://www.hasil.gov.my/en/syarikat/tempoh-asas-syarikat/",
    # Employer obligations.
    "https://www.hasil.gov.my/en/majikan/",
    "https://www.hasil.gov.my/en/majikan/senarai-pembekal-perisian-atau-majikan/",
    # Institutions, organisations and funds, mainly tax-exempt body rules.
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/garis-panduan-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/garis-panduan-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/garis-panduan-di-bawah-kelulusan-ss4411d-acp-1967/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/garis-panduan-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/garis-panduan-di-bawah-kelulusan-ss446-acp-1967/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/garis-panduan-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/garis-panduan-di-bawah-perintah-cukai-pendapatan-pengecualian-2020-p-u-a-139-2020/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/garis-panduan-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/garis-panduan-lain-lain/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-permohonan/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-permohonan/penjelasan-aktiviti-yang-layak/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-permohonan/penjelasan-berkaitan-kandungan-perlembagaan/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-permohonan/penjelasan-berkaitan-penolakan-permohonan/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-permohonan/persediaan-untuk-memohon-kelulusan-kphdn/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-umum/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-umum/pengenalan-dan-sepintas-lalu-kelulusan-kphdn-subseksyen-446/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-umum/penjelasan-umum-konsesi-kelulusan-kphdn/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/info-umum/peranan-dan-tanggungjawab-kphdn/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/manual-pengguna-sistem-e-derma/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/perkhidmatan-atas-talian/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/resit-derma-sumbangan/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/semakan-kelulusan-derma/p-u-a-139-2020/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/semakan-kelulusan-derma/subseksyen-4411d-akta-cukai-pendapatan-1967/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/semakan-kelulusan-derma/subseksyen-446-akta-cukai-pendapatan-1967/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/soalan-lazim-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/soalan-lazim-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/berkaitan-perkara-umum-subseksyen-446/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/soalan-lazim-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/berkaitan-rumah-ibadat-subsekyen-446-pu-a-139-2020-pua-52-2017/",
    "https://www.hasil.gov.my/en/institusi-organisasi-tabung/soalan-lazim-institusi-organisasi-tabung-bukan-berasaskan-keuntungan/berkaitan-wakaf-dan-endowment-4411d/",
]

# Map source host -> category column value.
_CATEGORY_BY_HOST = {
    "kwsp.gov.my": "epf",
    "hasil.gov.my": "tax",
    "perkeso.gov.my": "socso",
    "jpn.gov.my": "identity",
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
# Google serves its throttling and outage pages with HTTP 200, so the translator
# returns the error page *as the translation* and raises nothing. Left unchecked
# that text is what gets embedded, which silently makes the chunk unretrievable
# and parks its vector next to every other failed chunk.
_TRANSLATE_ERROR_MARKERS = (
    "that’s an error",
    "that's an error",
    "that’s all we know",
    "that's all we know",
    "server error",
    "error 500",
    "error 502",
    "error 503",
    "our systems have detected unusual traffic",
)

_TRANSLATE_ATTEMPTS = 4
_TRANSLATE_BACKOFF = 6.0


def _looks_like_error_page(text: str) -> bool:
    head = text[:400].lower()
    return any(marker in head for marker in _TRANSLATE_ERROR_MARKERS)


def translate_to_english(text: str) -> tuple[str, str]:
    """Detect language; translate to English if needed. Returns (english, lang)."""
    try:
        lang = detect(text)
    except LangDetectException:
        return text, "unknown"
    if lang == "en":
        return text, lang

    max_chars = 4500  # Google Translate per-call cap
    parts = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    for attempt in range(1, _TRANSLATE_ATTEMPTS + 1):
        try:
            translated = [
                GoogleTranslator(source="auto", target="en").translate(p) for p in parts
            ]
            joined = " ".join(t for t in translated if t)
        except Exception as exc:  # noqa: BLE001
            joined = ""
            log.warning("  translation attempt %d/%d raised (%s)",
                        attempt, _TRANSLATE_ATTEMPTS, exc)
        else:
            if joined.strip() and not _looks_like_error_page(joined):
                return joined, lang
            log.warning("  translation attempt %d/%d returned an error page",
                        attempt, _TRANSLATE_ATTEMPTS)
        if attempt < _TRANSLATE_ATTEMPTS:
            time.sleep(_TRANSLATE_BACKOFF * attempt + random.uniform(0, 3))

    # Falling back to the source text keeps the chunk retrievable: the embedding
    # model is multilingual, so a Malay passage still matches a Malay question
    # and, less strongly, an English one. An error page matches nothing.
    log.warning("  translation failed after %d attempts — keeping original text",
                _TRANSLATE_ATTEMPTS)
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
        # JPN's WordPress theme wraps the real content in .jpn-page-content and
        # emits no <article> on ordinary pages, so without this the whole
        # document was the fallback and the MyGov portal banner rode along in
        # every chunk. Its counter-hours page does use <article>, but one per
        # card, so select_one there would keep a single card and drop both the
        # other three and the headings saying which states each applies to.
        ".jpn-page-content",
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
    # MyGov standard portal banner + accessibility toolbar (JPN and other
    # federal portals render these on EVERY page, above the real content).
    "portal rasmi kerajaan malaysia", "portal rasmi", "kenal pasti begini",
    "portal yang selamat menggunakan https", "bahasa", "bahasa melayu",
    "english", "a-", "a+", "a", "|", "v", "🏛", "›",
    "laman utama", "perolehan", "soalan lazim", "informasi", "perkhidmatan",
    "+", "-",  # accordion expand/collapse markers on JPN service pages
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
    # MyGov portal-authenticity banner (JPN and other federal portals) — the
    # sentences run inline so line-level filtering alone misses them.
    "Pautan portal rasmi berakhir dengan .gov.my",
    "Sekiranya pautan tidak berakhir dengan .gov.my, segera tutup halaman itu "
    "walaupun ia kelihatan serupa.",
    "Periksa ikon mangga atau https:// pada pautan. Jika tiada, jangan "
    "kongsikan sebarang maklumat sensitif.",
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

# Everything from the EARLIEST of these sentinels onward is editorial/
# promotional noise or footer navigation.
_TRAILING_NOISE_SENTINELS = (
    "related articles",           # KWSP promotional block
    "senarai sebutharga/tender",  # JPN footer procurement nav
)

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

    # Step 2: truncate at the earliest trailing-noise section (promotional
    # block or footer nav), whichever appears first.
    low = text.lower()
    positions = [p for p in (low.find(s) for s in _TRAILING_NOISE_SENTINELS) if p != -1]
    if positions:
        text = text[: min(positions)]

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

    #preview mode: write to previews/ and stop (no API/DB calls)
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
            #translator can return empty for link-only/symbol chunks; the
            #embeddings API 400s on empty input. Fall back to the original.
            translate_text = original_text
        try:
            vector = embed_passage(translate_text)  #1024-dim, NO instruct prefix
        except Exception as exc:  #noqa: BLE001 — one bad chunk must not kill the run
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
                #summary / user_id / message_id left NULL (nullable);
                #fts is a GENERATED column — never inserted.
            }
        )

    if not rows:
        return

    client = get_client()
    #idempotent: clear any prior chunks for this page before re-inserting.
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
    parser.add_argument(
        "--proxy",
        help="Route the browser through a proxy, e.g. socks5://127.0.0.1:1080. "
             "Useful when the local address has been rate-limited by a site's "
             "edge: an SSH tunnel to another host egresses from that host's IP "
             "without needing the scraper installed there.",
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
                **({"proxy": {"server": args.proxy}} if args.proxy else {}),
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
