"""
LangGraph node: Context Gatherer

Calls data layer tools dynamically based on the intent and extracted entities
from the AgentState. No hardcoded tool calls — everything driven by state.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from agent.state import AgentState
from data_layer.registry import DataRegistry
from data_layer.portfolio_analyzer import PortfolioAnalyzer
from data_layer.market_analyzer import MarketAnalyzer
from data_layer.news_processor import NewsProcessor
from config import Settings

logger = logging.getLogger(__name__)


def make_context_gatherer(
    registry: DataRegistry,
    settings: Settings,
):
    """Factory returns the context_gatherer node with injected dependencies."""

    portfolio_analyzer = PortfolioAnalyzer(registry, settings)
    market_analyzer    = MarketAnalyzer(registry)
    news_processor     = NewsProcessor(registry, settings)

    def context_gatherer(state: AgentState) -> dict:
        """
        Node: Gather all relevant context by calling data layer analyzers.
        What to fetch is determined entirely by the state (intent + extracted entities).
        No hardcoded data queries.
        """
        intent         = state.get("intent", "general_query")
        portfolio_id   = state.get("portfolio_id")
        symbols        = state.get("symbols_mentioned", [])
        sectors        = state.get("sectors_mentioned", [])
        updates: dict  = {}

        logger.info(
            "context_gatherer: intent=%s, portfolio=%s, symbols=%s, sectors=%s",
            intent, portfolio_id, symbols, sectors,
        )

        # ── Portfolio context ──────────────────────────────────────────────────
        if portfolio_id and portfolio_id in registry.portfolio_ids:
            try:
                updates["portfolio_context"] = portfolio_analyzer.analyze(portfolio_id)
                logger.info("Gathered portfolio context for %s", portfolio_id)
            except Exception as exc:
                logger.error("Portfolio analysis failed: %s", exc)

        # ── Market context (always useful) ────────────────────────────────────
        try:
            updates["market_context"] = market_analyzer.get_full_snapshot()
        except Exception as exc:
            logger.error("Market snapshot failed: %s", exc)

        # ── Stock context — dynamically iterate mentioned symbols ──────────────
        if symbols:
            stock_ctx: Dict[str, dict] = {}
            for symbol in symbols:
                if symbol in registry.stock_symbols:
                    try:
                        stock_ctx[symbol] = market_analyzer.get_stock_detail(symbol)
                    except Exception as exc:
                        logger.warning("Stock detail failed for %s: %s", symbol, exc)
            if stock_ctx:
                updates["stock_context"] = stock_ctx

        # ── Sector context — dynamically iterate mentioned sectors ─────────────
        if sectors:
            sector_ctx: Dict[str, dict] = {}
            for sector in sectors:
                if sector in registry.sector_names:
                    try:
                        sector_ctx[sector] = market_analyzer.get_sector_detail(sector)
                    except Exception as exc:
                        logger.warning("Sector detail failed for %s: %s", sector, exc)
            if sector_ctx:
                updates["sector_context"] = sector_ctx

        # ── Auto-add sectors implied by portfolio holdings ─────────────────────
        if portfolio_id and not sectors:
            portfolio_data = updates.get("portfolio_context", {})
            sector_alloc   = portfolio_data.get("sector_allocation", {})
            top_sectors    = sorted(
                sector_alloc, key=lambda s: sector_alloc[s], reverse=True
            )[:3]
            sector_ctx = {}
            for sector in top_sectors:
                clean_sector = sector.replace("DIVERSIFIED_MF", "").replace("DEBT_FUNDS", "").strip()
                if clean_sector and clean_sector in registry.sector_names:
                    try:
                        sector_ctx[clean_sector] = market_analyzer.get_sector_detail(clean_sector)
                    except Exception:
                        pass
            if sector_ctx:
                updates["sector_context"] = sector_ctx

        # ── Auto-add symbols from portfolio if none mentioned ──────────────────
        if portfolio_id and not symbols:
            holding_symbols = registry.get_portfolio_holdings_symbols(portfolio_id)[:5]
            stock_ctx = updates.get("stock_context", {})
            for sym in holding_symbols:
                if sym in registry.stock_symbols and sym not in stock_ctx:
                    try:
                        stock_ctx[sym] = market_analyzer.get_stock_detail(sym)
                    except Exception:
                        pass
            if stock_ctx:
                updates["stock_context"] = stock_ctx

        # ── News context — gathered for portfolio or mentioned entities ─────────
        news_articles: List[dict] = []
        if portfolio_id:
            news_articles = registry.get_news_for_portfolio(portfolio_id)
        elif symbols:
            seen_ids: set = set()
            for sym in symbols:
                for article in registry.get_news_for_stock(sym):
                    if article["id"] not in seen_ids:
                        news_articles.append(article)
                        seen_ids.add(article["id"])
        elif sectors:
            seen_ids = set()
            for sector in sectors:
                for article in registry.get_news_for_sector(sector):
                    if article["id"] not in seen_ids:
                        news_articles.append(article)
                        seen_ids.add(article["id"])
        else:
            news_articles = news_processor.get_top_impacting_news(n=8)

        updates["news_context"] = news_articles

        # ── Mutual fund context for MF-related queries ─────────────────────────
        if portfolio_id:
            mf_codes = registry.get_portfolio_mf_codes(portfolio_id)
            mf_ctx: Dict[str, dict] = {}
            for code in mf_codes:
                try:
                    mf_ctx[code] = registry.get_mutual_fund(code)
                except ValueError:
                    pass
            if mf_ctx:
                updates["mf_context"] = mf_ctx

        logger.info(
            "context_gatherer done: portfolio=%s, stocks=%d, sectors=%d, news=%d",
            bool(updates.get("portfolio_context")),
            len(updates.get("stock_context", {})),
            len(updates.get("sector_context", {})),
            len(updates.get("news_context", [])),
        )
        return updates

    return context_gatherer
