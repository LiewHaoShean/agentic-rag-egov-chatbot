"""Centralized application settings.

All secrets and tunables are loaded from the environment via pydantic-settings.
Nothing in the codebase should read os.environ directly — import `settings` from here.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- App
    app_env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    # Reject webhook payloads larger than this (bytes). Guards Redis/broker.
    max_payload_bytes: int = Field(default=256_000, description="Webhook body size guard")

    # ----------------------------------------------------- Gemini (agent brain)
    # Backend: "google-api" = AI Studio API key (free tier: ~20 req/day/model —
    # too small for the 3-calls-per-message agent). "vertex" = Vertex AI via
    # Cloud billing (uses GCP credit; auth via Application Default Credentials).
    gemini_backend: Literal["google-api", "vertex"] = "google-api"
    gemini_api_key: str = ""  # google-api backend only
    gemini_model: str = "gemini-2.0-flash"
    gemini_timeout_seconds: float = 30.0
    # vertex backend only:
    google_cloud_project: str = ""
    vertex_location: str = "us-central1"

    # --------------------------------------- DashScope (Qwen3-VL + Qwen ASR)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope-intl.aliyuncs.com/api/v1"
    qwen_vl_model: str = "qwen3-vl-plus"
    qwen_asr_model: str = "qwen3-asr-flash"
    qwen_timeout_seconds: float = 60.0

    # ------------------------------------------------- Embeddings (e5-large-instruct)
    # Primary: Alibaba embedding API. Fallback: HuggingFace Inference API.
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_dimensions: int = 1024  # MUST match vector(1024) in Supabase
    embedding_timeout_seconds: float = 30.0
    # Alibaba (primary)
    alibaba_embedding_api_key: str = ""
    alibaba_embedding_url: str = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/embeddings"
    )
    alibaba_embedding_model: str = "text-embedding-v3"
    # HuggingFace Inference API (fallback — known cold-start/503 on this model)
    hf_api_key: str = ""
    hf_embedding_url: str = (
        "https://api-inference.huggingface.co/models/"
        "intfloat/multilingual-e5-large-instruct"
    )

    # ---------------------------------------------------------- Supabase
    supabase_url: str = ""
    supabase_service_role_key: str = ""  # service-role: server-side ONLY, never client
    supabase_storage_bucket: str = "whatsapp-media"

    # ------------------------------------------------------------- Redis
    redis_url: str = "redis://localhost:6379/0"
    # Idempotency dedupe key TTL (seconds) — absorbs redelivery storms.
    idempotency_ttl_seconds: int = 60 * 60 * 6  # 6h

    # -------------------------------------------------------- Webhook auth
    webhook_token: str = ""  # validated against X-Webhook-Token header

    # ----------------------------------------------------- n8n outbound callback
    n8n_callback_url: str = ""
    n8n_callback_timeout_seconds: float = 15.0
    n8n_callback_max_retries: int = 5
    # Dead-letter log file for callbacks that exhaust all retries.
    callback_dlq_path: str = "callback_dlq.log"

    # --------------------------------------------------------- Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
