"""
LangGraph node: News Reasoner

Uses GPT-4o chain-of-thought to build causal chain narratives from
dynamically gathered news and portfolio context.
"""
from __future__ import annotations

import json
import logging
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from agent.state import AgentState
from agent.prompts.templates import CAUSAL_REASONING_PROMPT
from agent.utils import sanitize_for_llm, safe_json_dumps
from data_layer.registry import DataRegistry
from data_layer.news_processor import NewsProcessor
from config import Settings

logger = logging.getLogger(__name__)


def make_news_reasoner(llm: ChatOpenAI, registry: DataRegistry, settings: Settings):

    news_processor = NewsProcessor(registry, settings)

    def news_reasoner(state: AgentState, config: RunnableConfig = None) -> dict:
        """
        Node: Deep causal chain reasoning.
        Builds narrative: macro event → sector → stock → portfolio P&L.
        Uses GPT-4o with chain-of-thought — all data injected from state.
        """
        news_articles    = state.get("news_context", [])
        portfolio_ctx    = state.get("portfolio_context")
        market_ctx       = state.get("market_context", {})

        if not news_articles:
            return {
                "causal_chain":      "No relevant news articles found for context.",
                "causal_chain_data": {},
            }

        # Format portfolio holdings for the prompt
        portfolio_holdings_text = "No portfolio selected."
        causal_data             = {}

        if portfolio_ctx:
            portfolio_id = portfolio_ctx.get("portfolio_id")
            stock_pnl    = portfolio_ctx.get("stock_analysis", [])
            portfolio_holdings_text = safe_json_dumps(
                [
                    {
                        "symbol":          h.get("symbol"),
                        "sector":          h.get("sector"),
                        "weight_pct":      h.get("weight_in_portfolio"),
                        "day_change_pct":  h.get("day_change_percent"),
                        "day_change_inr":  h.get("day_change"),
                        "current_value":   h.get("current_value"),
                    }
                    for h in stock_pnl[:8]
                ],
                indent=2,
            )
            # Also run structured causal chain from news_processor
            try:
                causal_data = news_processor.build_portfolio_causal_chain(portfolio_id)
            except Exception as exc:
                logger.warning("Structured causal chain failed: %s", exc)

        # Format news articles for prompt (top 8 by impact)
        sorted_news = sorted(
            news_articles,
            key=lambda n: abs(n.get("sentiment_score", 0)),
            reverse=True,
        )[:8]
        news_text = safe_json_dumps(
            [
                {
                    "id":             n.get("id"),
                    "headline":       n.get("headline"),
                    "sentiment":      n.get("sentiment"),
                    "sentiment_score": n.get("sentiment_score"),
                    "impact_level":   n.get("impact_level"),
                    "scope":          n.get("scope"),
                    "affected_sectors": n.get("entities", {}).get("sectors", []),
                    "affected_stocks":  n.get("entities", {}).get("stocks", []),
                    "causal_factors":   n.get("causal_factors", []),
                    "conflict_flag":    n.get("conflict_flag", False),
                }
                for n in sorted_news
            ],
            indent=2,
        )

        sector_perf_text = safe_json_dumps(
            {
                k: {"change_pct": v.get("change_percent"), "sentiment": v.get("sentiment")}
                for k, v in market_ctx.get("sector_performance", {}).items()
            },
            indent=2,
        )

        breadth_text = safe_json_dumps(
            {
                "market_breadth": market_ctx.get("market_breadth", {}),
                "fii_dii":        market_ctx.get("fii_dii", {}),
            },
            indent=2,
        )

        prompt = sanitize_for_llm(CAUSAL_REASONING_PROMPT.format(
            news_articles      = news_text,
            portfolio_holdings = portfolio_holdings_text,
            sector_performance = sector_perf_text,
            market_breadth     = breadth_text,
        ))

        logger.info("news_reasoner: running causal chain analysis on %d articles", len(sorted_news))

        try:
            response = llm.invoke(prompt, config=config)
            causal_chain_text = response.content
        except Exception as exc:
            logger.error("news_reasoner LLM call failed: %s", exc)
            causal_chain_text = causal_data.get("narrative", "Unable to build causal chain.")

        return {
            "causal_chain":      causal_chain_text,
            "causal_chain_data": causal_data,
        }

    return news_reasoner
