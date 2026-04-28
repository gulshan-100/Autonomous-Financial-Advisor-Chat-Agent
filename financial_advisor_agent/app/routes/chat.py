"""
FastAPI route: POST /api/chat — SSE streaming chat endpoint.

Emits Server-Sent Events for the ReAct agent's autonomous reasoning loop:
  node_start  : agent node begins (financial_advisor | tool_executor)
  node_end    : agent node completes
  tool_call   : agent decided to invoke a specific tool (shows WHICH tool)
  tool_done   : tool execution finished
  token       : streaming response token from the LLM's final answer
  done        : full graph run complete
  error       : any error

Token streaming works because:
  - All LLM calls are tagged "streaming_response" via .with_config()
  - Tool-calling LLM chunks have empty content (only tool_calls payload)
  - Final response chunks have non-empty content → these are the streamed tokens
  - The SSE handler filters on chunk.content being non-empty
"""
from __future__ import annotations

import asyncio
import json
import logging
import traceback

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from agent.graph import ALL_NODE_NAMES, NODE_JUDGE, create_initial_state
from agent.tracing import get_langfuse_handler, flush_langfuse
from app.models.request import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Chat"])

# Node labels shown in the frontend reasoning tracker
NODE_LABELS: dict[str, str] = {
    "financial_advisor": "🧠 Reasoning",
    "tool_executor":     "🔧 Executing tools",
    "response_judge":    "⚖️ Evaluating response",
}

# Human-readable labels for each tool the agent may call
TOOL_LABELS: dict[str, str] = {
    "think":                    "💭 Planning next steps",
    "list_portfolios":          "📋 Listing available portfolios",
    "get_portfolio_analysis":   "💼 Analysing portfolio",
    "get_portfolio_risk":       "⚠️ Assessing portfolio risk",
    "get_market_overview":      "📊 Fetching market overview",
    "get_stock_details":        "📈 Looking up stock data",
    "get_sector_analysis":      "🏭 Analysing sector",
    "search_news":              "📰 Searching relevant news",
    "get_top_movers":           "🔝 Finding top movers",
    "get_mutual_fund_details":  "💰 Fetching mutual fund data",
    "build_causal_chain":       "🔗 Building causal chain",
}


@router.post("/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """
    Stream the ReAct agent response via SSE.
    The frontend receives tool-call progress events and streaming final tokens.
    """
    graph    = request.app.state.graph
    registry = request.app.state.registry

    initial_state = create_initial_state(
        user_message = req.message,
        portfolio_id = req.portfolio_id,
        session_id   = req.session_id,
        registry     = registry,
    )

    langfuse_handler = get_langfuse_handler()
    config = {
        "configurable": {"thread_id": req.session_id},
        "recursion_limit": request.app.state.settings.graph_recursion_limit,
        "callbacks": [langfuse_handler] if langfuse_handler else [],
        "metadata": {
            "langfuse_session_id": req.session_id,
            "langfuse_tags":       ["financial-advisor"],
        } if langfuse_handler else {},
    }

    async def event_generator():
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        _DONE = None  # sentinel

        async def _run_graph() -> None:
            try:
                async for event in graph.astream_events(
                    initial_state,
                    config=config,
                    version="v2",
                ):
                    event_name = event.get("event", "")
                    node_name  = event.get("name", "")
                    tags       = event.get("tags", [])

                    # ── Agent / tool-executor / judge node lifecycle ──────────
                    if event_name == "on_chain_start" and node_name in ALL_NODE_NAMES:
                        await queue.put(f"data: {json.dumps({'type': 'node_start', 'node': node_name, 'label': NODE_LABELS.get(node_name, node_name)})}\n\n")

                    elif event_name == "on_chain_end" and node_name in ALL_NODE_NAMES:
                        if node_name == NODE_JUDGE:
                            output = event.get("data", {}).get("output", {})
                            jr = output.get("judge_result")
                            if jr and "error" not in jr:
                                await queue.put(f"data: {json.dumps({'type': 'judge_result', 'result': jr})}\n\n")
                        await queue.put(f"data: {json.dumps({'type': 'node_end', 'node': node_name, 'label': NODE_LABELS.get(node_name, node_name)})}\n\n")

                    elif event_name == "on_tool_start":
                        tool_name = node_name
                        label     = TOOL_LABELS.get(tool_name, f"🔧 {tool_name}")
                        await queue.put(f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name, 'label': label})}\n\n")

                    elif event_name == "on_tool_end":
                        await queue.put(f"data: {json.dumps({'type': 'tool_done', 'tool': node_name})}\n\n")

                    elif (
                        event_name == "on_chat_model_stream"
                        and "streaming_response" in tags
                    ):
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            await queue.put(f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n")

                await queue.put(f"data: {json.dumps({'type': 'done'})}\n\n")
                flush_langfuse()

            except Exception as exc:
                logger.error("Chat stream error: %s\n%s", exc, traceback.format_exc())
                await queue.put(f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n")
            finally:
                await queue.put(_DONE)

        task = asyncio.create_task(_run_graph())

        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=3.0)
                    if item is _DONE:
                        break
                    yield item
                except asyncio.TimeoutError:
                    yield ": ping\n\n"  # keep-alive for Render / proxies
        finally:
            await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Connection":                  "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
