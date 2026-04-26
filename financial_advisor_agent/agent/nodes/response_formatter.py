"""
LangGraph node: Response Formatter

Converts structured advice from advisor_synthesizer into beautiful streaming
markdown using GPT-4o with streaming enabled.
The streaming tokens are captured by FastAPI's astream_events for SSE delivery.
"""
from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.prompts.templates import RESPONSE_FORMAT_PROMPT
from agent.utils import sanitize_for_llm, safe_json_dumps

logger = logging.getLogger(__name__)


def make_response_formatter(llm: ChatOpenAI):
    # Create a streaming variant tagged for SSE capture
    streaming_llm = llm.with_config(tags=["streaming_response"])

    def response_formatter(state: AgentState) -> dict:
        """
        Node: Convert structured advice → streaming markdown response.
        Uses the streaming LLM — FastAPI's astream_events captures tokens
        from the 'streaming_response' tagged LLM call and forwards them via SSE.
        """
        user_question    = state.get("user_query", "")
        query_complexity = state.get("query_complexity", "MODERATE")
        advice           = state.get("advice_structured", {})
        portfolio_ctx    = state.get("portfolio_context")

        if not advice:
            return {"final_response": "I'm sorry, I wasn't able to generate a response. Please try again."}

        # Build quick portfolio summary for context
        portfolio_quick = "No portfolio selected."
        if portfolio_ctx:
            val     = portfolio_ctx.get("current_value", 0)
            day_pct = portfolio_ctx.get("day_change_pct", 0)
            portfolio_quick = (
                f"{portfolio_ctx.get('user_name')}'s portfolio: "
                f"Rs.{val:,.0f} | Day: {day_pct:+.2f}%"
            )

        prompt = sanitize_for_llm(RESPONSE_FORMAT_PROMPT.format(
            user_question           = sanitize_for_llm(user_question or "Financial advice request"),
            query_complexity        = query_complexity,
            advice_json             = safe_json_dumps(advice, indent=2),
            portfolio_quick_summary = sanitize_for_llm(portfolio_quick),
        ))

        logger.info("response_formatter: generating streaming markdown response")

        full_response = ""
        try:
            # streaming=True means tokens arrive chunk by chunk
            # astream_events in FastAPI captures these via "on_chat_model_stream" events
            for chunk in streaming_llm.stream(prompt):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
        except Exception as exc:
            logger.error("response_formatter streaming failed: %s", exc)
            # Fallback: build response from structured advice without LLM
            full_response = _format_fallback(advice, portfolio_ctx)

        return {"final_response": full_response}

    return response_formatter


def _format_fallback(advice: dict, portfolio_ctx: dict | None) -> str:
    """Template-based fallback when LLM is unavailable."""
    lines = []
    summary = advice.get("summary", "")
    if summary:
        lines.append(f"{summary}\n")

    risk_alerts = advice.get("risk_alerts", [])
    if risk_alerts:
        lines.append("\n## 🚨 Risk Alerts\n")
        for alert in risk_alerts:
            lines.append(f"- {alert}")

    recs = advice.get("recommendations", [])
    if recs:
        lines.append("\n## 💡 Recommendations\n")
        for rec in recs:
            lines.append(f"- {rec}")

    conflicts = advice.get("conflicting_signals", [])
    if conflicts:
        lines.append("\n## ⚡ Conflicting Signals\n")
        for c in conflicts:
            lines.append(f"- {c}")

    sources = advice.get("sources", [])
    if sources:
        lines.append(f"\n---\n📰 *Based on {len(sources)} news articles.*")

    return "\n".join(lines)
