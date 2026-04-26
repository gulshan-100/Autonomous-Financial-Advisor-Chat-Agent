"""
LangGraph node: Risk Analyzer

Computes portfolio risk flags dynamically using PortfolioAnalyzer.
All thresholds come from Settings — not a single magic number in this file.
"""
from __future__ import annotations

import logging
from typing import List

from agent.state import AgentState
from data_layer.registry import DataRegistry
from data_layer.news_processor import NewsProcessor
from config import Settings

logger = logging.getLogger(__name__)


def make_risk_analyzer(registry: DataRegistry, settings: Settings):

    news_processor = NewsProcessor(registry, settings)

    def risk_analyzer(state: AgentState) -> dict:
        """
        Node: Deep portfolio risk analysis.
        Works on portfolio_context already gathered by context_gatherer.
        Pure Python — no LLM needed, all logic is deterministic.
        """
        portfolio_ctx = state.get("portfolio_context")

        if not portfolio_ctx:
            # No portfolio context — still check for market-wide risks
            return {
                "risk_flags":         [],
                "concentration_risk": False,
                "portfolio_beta":     None,
                "conflict_signals":   news_processor.identify_conflict_signals()[:3],
            }

        # Risk flags are already computed by PortfolioAnalyzer
        risk_flags: List[str] = portfolio_ctx.get("risk_flags", [])

        # Additional derived risk flags from cross-referencing market context
        market_ctx = state.get("market_context", {})
        risk_flags.extend(
            _check_market_risks(portfolio_ctx, market_ctx, settings)
        )
        risk_flags.extend(
            _check_news_risks(portfolio_ctx, state.get("news_context", []), settings)
        )

        portfolio_id = portfolio_ctx.get("portfolio_id")
        conflict_signals = news_processor.identify_conflict_signals()

        portfolio_beta = (
            portfolio_ctx.get("risk_metrics", {}).get("beta")
        )

        logger.info(
            "risk_analyzer: %d risk flags, concentration=%s, beta=%s",
            len(risk_flags),
            portfolio_ctx.get("has_concentration_risk"),
            portfolio_beta,
        )

        return {
            "risk_flags":         risk_flags,
            "concentration_risk": len(risk_flags) > 0,
            "portfolio_beta":     portfolio_beta,
            "conflict_signals":   conflict_signals,
        }

    return risk_analyzer


def _check_market_risks(
    portfolio_ctx: dict,
    market_ctx: dict,
    settings: Settings,
) -> List[str]:
    """Cross-reference portfolio sector allocation vs. market sentiment."""
    flags: List[str] = []
    sector_alloc      = portfolio_ctx.get("sector_allocation", {})
    sector_performance = market_ctx.get("sector_performance", {})

    for sector, pct in sector_alloc.items():
        if sector in sector_performance and pct > settings.sector_concentration_warning_pct:
            sector_perf = sector_performance[sector]
            change      = sector_perf.get("change_percent", 0)
            sentiment   = sector_perf.get("sentiment", "NEUTRAL")
            if sentiment == "BEARISH" and change < -1.5:
                flags.append(
                    f"📉 MARKET RISK: {sector} (your {pct:.1f}% allocation) "
                    f"is currently BEARISH at {change:+.2f}% today"
                )
    return flags


def _check_news_risks(
    portfolio_ctx: dict,
    news_articles: List[dict],
    settings: Settings,
) -> List[str]:
    """Flag high-impact negative news affecting portfolio holdings."""
    flags: List[str] = []
    holding_symbols   = {s["symbol"] for s in portfolio_ctx.get("stock_analysis", [])}

    for article in news_articles:
        if (
            article.get("impact_level") == "HIGH"
            and article.get("sentiment_score", 0) <= settings.negative_sentiment_threshold
        ):
            affected = [
                s for s in article.get("entities", {}).get("stocks", [])
                if s in holding_symbols
            ]
            if affected:
                flags.append(
                    f"📰 HIGH-IMPACT NEWS on {', '.join(affected)}: "
                    f"{article['headline'][:80]}..."
                )
    return flags[:3]  # cap to avoid prompt bloat
