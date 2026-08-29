"""LangGraph workflow assembly + entrypoint called by the Celery task.

Topology (matches the architecture diagram):

  guard1 ──harmful──▶ blocked ─▶ END
    │ │ meta (greeting / about-the-assistant / repeat-in-other-language)
    │ ├────────▶ meta ─▶ END        (answered directly; no RAG, no grounding)
    │ │ error (guard LLM itself failed)
    │ └────────▶ busy ─▶ END
    │ ok
    ▼
  rewrite   (resolve pronouns/ellipsis from history -> standalone query)
    │
    ▼
  retrieve ──zero docs──▶ fallback ─▶ END
    │ chunks
    ▼
  generate ─▶ validate ──valid──▶ finalize ─▶ END
                  │ invalid
                  ├─ retry_count >= 1 ─▶ fallback
                  └─ else ─▶ retrieve   (bounded: exactly one retry)
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent import nodes
from agent.state import AgentState
from core.logging import get_logger

log = get_logger(__name__)


@lru_cache
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guard1", nodes.guardrail_prefilter)
    g.add_node("blocked", nodes.blocked_node)
    g.add_node("busy", nodes.busy_node)
    g.add_node("meta", nodes.meta_reply_node)
    g.add_node("rewrite", nodes.rewrite_query_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("fallback", nodes.fallback_node)
    g.add_node("generate", nodes.generate_node)
    g.add_node("validate", nodes.validate_node)
    g.add_node("finalize", nodes.finalize_node)

    g.add_edge(START, "guard1")

    g.add_conditional_edges(
        "guard1",
        nodes.route_after_guard1,
        {"blocked": "blocked", "retrieve": "rewrite", "busy": "busy",
         "meta": "meta"},
    )
    # Rewriting happens once, before the first retrieval; the validate->retrieve
    # retry edge re-uses the stored search_query rather than paying for it again.
    g.add_edge("rewrite", "retrieve")
    g.add_edge("blocked", END)
    g.add_edge("busy", END)
    g.add_edge("meta", END)

    g.add_conditional_edges(
        "retrieve",
        nodes.route_after_retrieval,
        {"generate": "generate", "fallback": "fallback"},
    )
    g.add_edge("fallback", END)

    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        nodes.route_after_validate,
        {"finalize": "finalize", "retrieve": "retrieve", "fallback": "fallback"},
    )
    g.add_edge("finalize", END)

    return g.compile()


def _hydrate_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Convert DB rows into LangGraph message dicts (conversation memory)."""
    msgs: list[dict[str, str]] = []
    for row in history:
        role = "assistant" if row.get("role") == "assistant" else "user"
        content = row.get("translate_message") or row.get("message") or ""
        if content:
            msgs.append({"role": role, "content": content})
    return msgs


def run_agent(
    user_text: str,
    history: list[dict[str, Any]],
    user_id: str,
    conversation_id: str,
) -> dict[str, Any]:
    """Entrypoint invoked by the Celery worker. Returns reply + chunk ids."""
    graph = build_graph()
    initial: AgentState = {
        "user_text": user_text,
        "messages": _hydrate_messages(history) + [{"role": "user", "content": user_text}],
        "user_id": user_id,
        "conversation_id": conversation_id,
        "retrieved_chunks": [],
        "retrieved_chunk_ids": [],
        "retry_count": 0,
    }
    # Defence in depth: the longest legitimate path is
    # guard -> retrieve -> generate -> validate -> retrieve -> generate ->
    # validate -> fallback (8 steps). A low ceiling turns any future routing
    # bug into one fast failure instead of 25 billed LLM calls.
    final_state = graph.invoke(initial, config={"recursion_limit": 12})
    return {
        "reply": final_state.get("reply") or nodes.FALLBACK_MESSAGE,
        "retrieved_chunk_ids": final_state.get("retrieved_chunk_ids", []),
    }
