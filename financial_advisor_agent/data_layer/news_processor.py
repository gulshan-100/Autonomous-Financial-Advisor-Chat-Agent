"""
News processor — dynamic causal chain builder.

Iterates all news articles from the registry to build causal chains,
produce impact summaries, and identify conflicting signals.
No news IDs, stock symbols, or sector names are hardcoded here.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from config import Settings
from .registry import DataRegistry

logger = logging.getLogger(__name__)

IMPACT_WEIGHT = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
SENTIMENT_DIRECTION = {"POSITIVE": 1, "NEGATIVE": -1, "NEUTRAL": 0, "MIXED": -0.5}


class NewsProcessor:
    """Dynamic news analysis and causal chain construction."""

    def __init__(self, registry: DataRegistry, settings: Settings) -> None:
        self._registry = registry
        self._settings = settings

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_market_wide_news(self) -> List[dict]:
        return sorted(
            self._registry.get_news_by_scope("MARKET_WIDE"),
            key=lambda n: IMPACT_WEIGHT.get(n.get("impact_level", "LOW"), 0),
            reverse=True,
        )

    def get_top_impacting_news(self, n: int = 5) -> List[dict]:
        """Return top-N news articles ranked by impact × |sentiment_score|."""
        all_news = self._registry.get_all_news()
        return sorted(
            all_news,
            key=lambda x: (
                IMPACT_WEIGHT.get(x.get("impact_level", "LOW"), 0)
                * abs(x.get("sentiment_score", 0))
            ),
            reverse=True,
        )[:n]

    def build_portfolio_causal_chain(self, portfolio_id: str) -> dict:
        """
        Trace: Root news event → affected sector → affected portfolio stocks → P&L impact.
        All done dynamically from registry data — no hardcoded IDs or names.
        """
        portfolio       = self._registry.get_portfolio(portfolio_id)
        holding_symbols = set(self._registry.get_portfolio_holdings_symbols(portfolio_id))
        holding_sectors = {
            h["sector"]
            for h in portfolio.get("holdings", {}).get("stocks", [])
        }
        relevant_news  = self._registry.get_news_for_portfolio(portfolio_id)
        causal_chains  = []
        conflict_flags = []

        for article in relevant_news[:6]:  # build chains for top 6 news articles
            affected_portfolio_stocks = [
                s for s in article.get("entities", {}).get("stocks", [])
                if s in holding_symbols
            ]
            affected_portfolio_sectors = [
                s for s in article.get("entities", {}).get("sectors", [])
                if s in holding_sectors
            ]

            if not (affected_portfolio_stocks or affected_portfolio_sectors):
                continue

            # Compute portfolio-level impact estimate
            pnl_contribution = self._estimate_pnl_contribution(
                portfolio, affected_portfolio_stocks, article
            )

            chain = {
                "news_id":          article["id"],
                "headline":         article["headline"],
                "sentiment":        article["sentiment"],
                "sentiment_score":  article.get("sentiment_score", 0),
                "impact_level":     article["impact_level"],
                "scope":            article["scope"],
                "affected_stocks":  affected_portfolio_stocks,
                "affected_sectors": affected_portfolio_sectors,
                "causal_factors":   article.get("causal_factors", []),
                "pnl_contribution_estimate": pnl_contribution,
            }
            causal_chains.append(chain)

            # Check for conflict flags
            if article.get("conflict_flag"):
                conflict_flags.append({
                    "news_id":     article["id"],
                    "headline":    article["headline"],
                    "explanation": article.get("conflict_explanation", ""),
                })

        # Build narrative string
        narrative = self._build_narrative(causal_chains, portfolio_id)

        return {
            "portfolio_id":   portfolio_id,
            "causal_chains":  causal_chains,
            "conflict_flags": conflict_flags,
            "narrative":      narrative,
            "news_count":     len(relevant_news),
            "high_impact_count": sum(
                1 for n in relevant_news if n.get("impact_level") == "HIGH"
            ),
        }

    def get_stock_news_analysis(self, symbol: str) -> dict:
        """Full news analysis for a specific stock."""
        news          = self._registry.get_news_for_stock(symbol)
        sentiment_agg = self._aggregate_sentiment(news)
        return {
            "symbol":              symbol,
            "news_articles":       news,
            "aggregate_sentiment": sentiment_agg,
            "conflict_present":    any(n.get("conflict_flag") for n in news),
            "top_headline":        news[0]["headline"] if news else None,
        }

    def get_sector_news_analysis(self, sector: str) -> dict:
        """Full news analysis for a specific sector."""
        news = self._registry.get_news_for_sector(sector)
        return {
            "sector":              sector,
            "news_articles":       news,
            "aggregate_sentiment": self._aggregate_sentiment(news),
        }

    def identify_conflict_signals(self) -> List[dict]:
        """
        Find all news articles where company-level news conflicts with sector sentiment.
        These are 'edge cases' where the agent must reason carefully.
        """
        conflicts = []
        for article in self._registry.get_all_news():
            if article.get("conflict_flag"):
                # Find what the sector sentiment is for this article's sector
                affected_sectors = article.get("entities", {}).get("sectors", [])
                sector_sentiments = []
                for sector in affected_sectors:
                    if sector in self._registry.sector_names:
                        s_data = self._registry.get_sector(sector)
                        sector_sentiments.append({
                            "sector":    sector,
                            "sentiment": s_data.get("sentiment"),
                            "change_pct": s_data.get("change_percent"),
                        })
                conflicts.append({
                    "news_id":         article["id"],
                    "headline":        article["headline"],
                    "article_sentiment": article["sentiment"],
                    "conflict_explanation": article.get("conflict_explanation", ""),
                    "sector_context":  sector_sentiments,
                })
        return conflicts

    # ── Private helpers ────────────────────────────────────────────────────────

    def _estimate_pnl_contribution(
        self,
        portfolio: dict,
        affected_symbols: List[str],
        article: dict,
    ) -> Optional[float]:
        """
        Very rough estimate: sum of (stock_day_change * weight / 100) for affected stocks.
        """
        if not affected_symbols:
            return None
        total: float = 0.0
        stocks = portfolio.get("holdings", {}).get("stocks", [])
        for holding in stocks:
            if holding["symbol"] in affected_symbols:
                day_change = holding.get("day_change", 0)
                total += day_change
        return round(total, 2)

    def _aggregate_sentiment(self, news: List[dict]) -> dict:
        if not news:
            return {"overall": "NEUTRAL", "score": 0.0, "distribution": {}}
        scores = [n.get("sentiment_score", 0) for n in news]
        avg_score = sum(scores) / len(scores)
        distribution: Dict[str, int] = {}
        for n in news:
            sentiment = n.get("sentiment", "NEUTRAL")
            distribution[sentiment] = distribution.get(sentiment, 0) + 1
        if avg_score <= self._settings.negative_sentiment_threshold:
            overall = "NEGATIVE"
        elif avg_score >= abs(self._settings.negative_sentiment_threshold):
            overall = "POSITIVE"
        else:
            overall = "MIXED/NEUTRAL"
        return {
            "overall":      overall,
            "score":        round(avg_score, 3),
            "distribution": distribution,
        }

    def _build_narrative(self, chains: List[dict], portfolio_id: str) -> str:
        if not chains:
            return "No significant news events directly linked to this portfolio's holdings."
        user_name = self._registry.user_map.get(portfolio_id, portfolio_id)
        parts = [f"Market events impacting {user_name}'s portfolio today:\n"]
        for i, chain in enumerate(chains[:3], 1):
            sentiment_icon = "📈" if chain["sentiment"] == "POSITIVE" else (
                "📉" if chain["sentiment"] == "NEGATIVE" else "⚡"
            )
            parts.append(
                f"{i}. {sentiment_icon} [{chain['impact_level']}] {chain['headline']}\n"
                f"   → Affected holdings: {', '.join(chain['affected_stocks']) or 'Sector-level impact'}\n"
                f"   → Sectors: {', '.join(chain['affected_sectors'])}\n"
            )
        return "".join(parts)
