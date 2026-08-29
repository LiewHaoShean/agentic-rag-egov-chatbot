"""LangGraph state object.

Each WhatsApp message is a separate webhook -> separate Celery task -> fresh
state. Conversation memory is therefore HYDRATED explicitly from the DB into
`messages` at graph start; it is never assumed to persist in-process.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Conversation messages (hydrated from DB + the current user turn).
    messages: Annotated[list, add_messages]

    # Raw user query text (post ASR/OCR normalization).
    user_text: str

    # Identity / routing context.
    user_id: str
    conversation_id: str

    # Standalone search query: the user's message with pronouns/ellipsis
    # resolved from conversation context ("how much does it cost?" ->
    # "EPF withdrawal fee at age 55"). Falls back to user_text verbatim.
    search_query: Optional[str]

    # Retrieval outputs.
    retrieved_chunks: list[dict[str, Any]]
    retrieved_chunk_ids: list[str]

    # Bounded retry counter — conditional edge forces exit at >= 1.
    retry_count: int

    # Guardrail / routing decisions.
    #   "meta"  = conversational/about-the-assistant; answered without RAG.
    #   "error" = the guard LLM call itself failed (outage/quota) -> busy msg.
    guard_decision: Optional[Literal["ok", "harmful", "meta", "error"]]

    # Generated answer + validation verdict.
    draft_answer: Optional[str]
    answer_valid: Optional[bool]

    # Final reply returned to the worker.
    reply: Optional[str]
