"""
LangGraph node: Market Analyzer

Uses GPT-4o to synthesize market context into a coherent narrative.
All data is injected dynamically from state — no hardcoded values.
"""
from __future__ import annotations

import json
import logging

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.prompts.templates import MARKET_ANALYSIS_PROMPT
from agent.utils import sanitize_for_llm, safe_json_dumps

logger = logging.getLogger(__name__)


def make_market_analyzer(llm: ChatOpenAI):

    def market_analyzer(state: AgentState, config: RunnableConfig = None) -> dict:
        """
        Node: Synthesize market data into a narrative.
        Injects dynamic market data from state into GPT-4o prompt.
        """
        market_ctx = state.get("market_context", {})
        if not market_ctx:
            return {"market_summary": "Market data unavailable."}

        # Prepare all data sections dynamically from state
        indices_text = safe_json_dumps(
            {
                k: {
                    "value":       v.get("current_value"),
                    "change_pct":  v.get("change_percent"),
                    "sentiment":   v.get("sentiment"),
                }
                for k, v in market_ctx.get("indices", {}).items()
            },
            indent=2,
        )

        sector_ranking_text = safe_json_dumps(
            [
                {
                    "sector":     s.get("sector"),
                    "change_pct": s.get("change_percent"),
                    "sentiment":  s.get("sentiment"),
                    "key_drivers": s.get("key_drivers", []),
                }
                for s in market_ctx.get("sector_ranking", [])
            ],
            indent=2,
        )

        top_movers_text = safe_json_dumps(market_ctx.get("top_movers", {}), indent=2)

        fii_dii_text = safe_json_dumps(market_ctx.get("fii_dii_analysis", {}), indent=2)

        breadth_text = safe_json_dumps(market_ctx.get("breadth_analysis", {}), indent=2)

        index_trends_text = safe_json_dumps(market_ctx.get("index_trends", []), indent=2)

        macro_themes_text = safe_json_dumps(market_ctx.get("macro_context", []), indent=2)

        prompt = sanitize_for_llm(MARKET_ANALYSIS_PROMPT.format(
            indices          = indices_text,
            sector_ranking   = sector_ranking_text,
            top_movers       = top_movers_text,
            fii_dii_analysis = fii_dii_text,
            breadth_analysis = breadth_text,
            index_trends     = index_trends_text,
            macro_themes     = macro_themes_text,
        ))

        logger.info("market_analyzer: running market synthesis")

        try:
            response = llm.invoke(prompt, config=config)
            market_summary = response.content
        except Exception as exc:
            logger.error("market_analyzer LLM call failed: %s", exc)
            market_summary = "Market analysis unavailable due to LLM error."

        return {"market_summary": market_summary}

    return market_analyzer
