"""Qwen ASR (audio -> text) via the Alibaba DashScope API — thin HTTP client.

Remote API only; no local inference. Has its own timeout.
"""
from __future__ import annotations

import base64

import httpx

from core.config import settings
from core.logging import get_logger

log = get_logger(__name__)

# qwen3-asr-flash accepts a system message as biasing context: terms listed
# here are far more likely to be transcribed correctly. Users mix Malay
# domain jargon into English/Chinese speech, which the model otherwise
# mangles (e.g. "akaun persaraan" -> "account bazaar").
# The list must cover every agency in the corpus. It originally named only
# KWSP and LHDN terms, so PERKESO was transcribed as ordinary English words
# and the retrieved chunks came back from the wrong agency.
_ASR_CONTEXT = (
    "Malaysian government services vocabulary: KWSP, EPF, "
    "Kumpulan Wang Simpanan Pekerja, 公积金, Akaun Persaraan, "
    "Akaun Sejahtera, Akaun Fleksibel, Akaun 55, Akaun Emas, i-Emas, "
    "i-Akaun, i-Lindung, i-Saraan, caruman, pengeluaran, dividen, "
    "penama, LHDN, Lembaga Hasil Dalam Negeri, cukai, e-Filing, "
    "PCB, EA form, borang, Borang BE, nombor cukai pendapatan, "
    "CKHT, cukai keuntungan harta tanah, duti setem, "
    "PERKESO, SOCSO, Pertubuhan Keselamatan Sosial, "
    "Skim Bencana Pekerjaan, Skim Keilatan, Sistem Insurans Pekerjaan, "
    "SIP, EIS, faedah hilang upaya, klinik panel, MyFutureJobs, "
    "JPN, Jabatan Pendaftaran Negara, MyKad, MyKid, MyKAS, MyPR, "
    "kad pengenalan, sijil kelahiran, sijil kematian, sijil perkahwinan, "
    "pendaftaran kelahiran, pendaftaran kematian, perkahwinan, "
    "perceraian, warganegara, kewarganegaraan, pengangkatan."
)


def transcribe(audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    """Transcribe a voice note to text.

    Sends the audio inline (base64) to DashScope. The bytes come from the
    private bucket download in the worker — they are never put on the broker.
    """
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    # qwen3-asr-flash is an LLM-style ASR model — it uses the multimodal
    # generation endpoint (audio message content), NOT a dedicated ASR route.
    url = f"{settings.dashscope_base_url}/services/aigc/multimodal-generation/generation"

    resp = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.qwen_asr_model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": [{"text": _ASR_CONTEXT}],
                    },
                    {
                        "role": "user",
                        "content": [{"audio": f"data:{mime_type};base64,{b64}"}],
                    }
                ]
            },
        },
        timeout=settings.qwen_timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    # Envelope verified against the live API (qwen3-asr-flash, 2026-07-05).
    return data["output"]["choices"][0]["message"]["content"][0]["text"]
