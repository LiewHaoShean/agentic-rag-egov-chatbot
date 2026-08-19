"""FastAPI application entrypoint.

Run:  uvicorn main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI

from api import health, webhook
from core.logging import configure_logging

configure_logging()

app = FastAPI(
    title="Agentic RAG E-Government Chatbot",
    description=(
        "Semantic translation layer over Malaysian public-sector policy "
        "(LHDN tax, KWSP/EPF). WhatsApp-only, bridged by n8n via secure webhooks."
    ),
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(webhook.router)


@app.get("/", include_in_schema=False)
async def root():
    return {"service": "agentic-rag-egov", "status": "ok"}
