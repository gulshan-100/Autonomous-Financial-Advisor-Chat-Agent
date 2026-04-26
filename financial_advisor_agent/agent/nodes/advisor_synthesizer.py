"""
LangGraph node: Advisor Synthesizer

Merges all gathered context into structured financial advice using GPT-4o.
The entire prompt is built from live AgentState — nothing hardcoded.
"""
from __future__ import annotations

import json
import logging
from typing import List

from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.prompts.templates import ADVISOR_SYNTHESIS_PROMPT
from agent.utils import sanitize_for_llm, safe_json_dumps

logger = logging.getLogger(__name__)


def make_advisor_synthesizer(llm: ChatOpenAI):

    def advisor_synthesizer(state: AgentState) -> dict:
        """
        Node: Generate structured financial advice using GPT-4o.
        All input data is injected dynamically from AgentState.
        Returns structured JSON that response_formatter converts to markdown.
        """
        user_question    = state.get("user_query", "")
        query_complexity = state.get("query_complexity", "MODERATE")
        portfolio_ctx    = state.get("portfolio_context")
        market_ctx       = state.get("market_context", {})
        sector_ctx       = state.get("sector_context", {})
        causal_chain     = state.get("causal_chain", "Not available.")
        risk_flags       = state.get("risk_flags", [])
        news_ctx         = state.get("news_context", [])

        # Build concise portfolio summary for prompt
        portfolio_text = "No portfolio selected."
        if portfolio_ctx:
            portfolio_text = safe_json_dumps(
                {
                    "user_name":          portfolio_ctx.get("user_name"),
                    "portfolio_type":     portfolio_ctx.get("portfolio_type"),
                    "risk_profile":       portfolio_ctx.get("risk_profile"),
                    "current_value":      portfolio_ctx.get("current_value"),
                    "day_change_pct":     portfolio_ctx.get("day_change_pct"),
                    "day_change_abs":     portfolio_ctx.get("day_change_absolute"),
                    "overall_gain_pct":   portfolio_ctx.get("overall_gain_loss_pct"),
                    "sector_allocation":  portfolio_ctx.get("sector_allocation"),
                    "risk_metrics":       portfolio_ctx.get("risk_metrics"),
                    "top_mover":          portfolio_ctx.get("top_mover"),
                    "has_concentration":  portfolio_ctx.get("has_concentration_risk"),
                },
                indent=2,
            )

        # Compact market summary for prompt
        market_text = safe_json_dumps(
            {
                "overall_sentiment": market_ctx.get("market_sentiment"),
                "key_indices": {
                    k: {
                        "change_pct": v.get("change_percent"),
                        "sentiment":  v.get("sentiment"),
                    }
                    for k, v in market_ctx.get("indices", {}).items()
                },
                "top_gainers": [m.get("symbol") for m in market_ctx.get("top_movers", {}).get("top_gainers", [])[:3]],
                "top_losers":  [m.get("symbol") for m in market_ctx.get("top_movers", {}).get("top_losers", [])[:3]],
                "fii_stance":  market_ctx.get("fii_dii_analysis", {}).get("fii_stance"),
                "fii_net_cr":  market_ctx.get("fii_dii_analysis", {}).get("fii_net_cr"),
                "breadth":     market_ctx.get("breadth_analysis", {}).get("sentiment_indicator"),
            },
            indent=2,
        )

        # Dynamic sector context
        sector_text = safe_json_dumps(
            {
                sector: {
                    "change_pct": data.get("performance", {}).get("change_percent"),
                    "sentiment":  data.get("performance", {}).get("sentiment"),
                    "key_drivers": data.get("performance", {}).get("key_drivers", []),
                }
                for sector, data in sector_ctx.items()
            },
            indent=2,
        ) if sector_ctx else "No sector-specific data."

        # News context (top 6 — dynamic)
        news_text = safe_json_dumps(
            [
                {
                    "id":         n.get("id"),
                    "headline":   n.get("headline"),
                    "sentiment":  n.get("sentiment"),
                    "impact":     n.get("impact_level"),
                    "stocks":     n.get("entities", {}).get("stocks", []),
                    "sectors":    n.get("entities", {}).get("sectors", []),
                    "conflict":   n.get("conflict_flag", False),
                }
                for n in news_ctx[:6]
            ],
            indent=2,
        )

        prompt = sanitize_for_llm(ADVISOR_SYNTHESIS_PROMPT.format(
            user_question     = sanitize_for_llm(user_question or "Please provide financial advice."),
            query_complexity  = query_complexity,
            portfolio_context = portfolio_text,
            market_context    = market_text,
            sector_context    = sector_text,
            causal_chain      = sanitize_for_llm(causal_chain),
            risk_flags        = safe_json_dumps(risk_flags, indent=2),
            news_context      = news_text,
        ))

        logger.info("advisor_synthesizer: synthesizing final advice")

        default_advice = {
            "summary":            "Unable to generate advice. Please try again.",
            "risk_alerts":        risk_flags,
            "causal_explanation": causal_chain,
            "recommendations":    [],
            "conflicting_signals": [],
            "watchlist":          [],
            "confidence":         "LOW",
            "confidence_reason":  "LLM error",
            "sources":            [],
        }

        try:
            response = llm.invoke(
                prompt,
                response_format={"type": "json_object"},
            )
            advice = json.loads(response.content)
        except (json.JSONDecodeError, Exception) as exc:
            logger.error("advisor_synthesizer LLM call failed: %s", exc)
            advice = default_advice

        return {
            "advice_structured":  advice,
            "recommendations":    advice.get("recommendations", []),
            "sources_used":       advice.get("sources", []),
            "confidence_level":   advice.get("confidence", "MEDIUM"),
        }

    return advisor_synthesizer
