"""
FastAPI route: POST /api/chat — SSE streaming chat endpoint.

Invokes the LangGraph graph and emits Server-Sent Events (SSE) for:
- node_start  : when a reasoning node begins
- node_end    : when a reasoning node completes
- token       : streaming response tokens from response_formatter
- done        : when the full graph run is complete
- error       : on any error

Frontend uses EventSource to receive these events and render:
- Reasoning progress tracker (node_start/node_end)
- Streaming response (token events)
"""
from __future__ import annotations

import json
import logging
import traceback

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from agent.graph import ALL_NODE_NAMES, create_initial_state
from app.models.request import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])

# Human-readable node labels for the frontend progress tracker
NODE_LABELS: dict[str, str] = {
    "intent_classifier":   "🧠 Understanding your question",
    "context_gatherer":    "📊 Gathering market data",
    "risk_analyzer":       "⚠️ Analysing portfolio risk",
    "news_reasoner":       "📰 Building causal chain",
    "market_analyzer":     "📈 Synthesising market context",
    "advisor_synthesizer": "💡 Generating advice",
    "response_formatter":  "✍️ Formatting response",
}


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """
    Stream the LangGraph agent response via SSE.
    Every reasoning node emits progress events before the final token stream.
    """
    graph    = request.app.state.graph
    registry = request.app.state.registry

    # Build initial state dynamically — no hardcoded values
    initial_state = create_initial_state(
        user_message = req.message,
        portfolio_id = req.portfolio_id,
        session_id   = req.session_id,
        registry     = registry,
    )

    # LangGraph config — thread_id enables MemorySaver persistence
    config = {
        "configurable": {"thread_id": req.session_id},
        "recursion_limit": request.app.state.settings.graph_recursion_limit,
    }

    async def event_generator():
        try:
            async for event in graph.astream_events(
                initial_state,
                config=config,
                version="v2",
            ):
                event_name = event.get("event", "")
                node_name  = event.get("name", "")
                tags       = event.get("tags", [])

                # ── Node start events → show progress tracker ─────────────────
                if event_name == "on_chain_start" and node_name in ALL_NODE_NAMES:
                    payload = json.dumps({
                        "type":  "node_start",
                        "node":  node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                    })
                    yield f"data: {payload}\n\n"

                # ── Node end events → tick off progress tracker ────────────────
                elif event_name == "on_chain_end" and node_name in ALL_NODE_NAMES:
                    payload = json.dumps({
                        "type":  "node_end",
                        "node":  node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                    })
                    yield f"data: {payload}\n\n"

                # ── Streaming tokens from response_formatter ───────────────────
                elif (
                    event_name == "on_chat_model_stream"
                    and "streaming_response" in tags
                ):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        payload = json.dumps({
                            "type":    "token",
                            "content": chunk.content,
                        })
                        yield f"data: {payload}\n\n"

            # ── Done sentinel ─────────────────────────────────────────────────
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            logger.error("Chat stream error: %s\n%s", exc, traceback.format_exc())
            error_payload = json.dumps({
                "type":    "error",
                "message": str(exc),
            })
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
