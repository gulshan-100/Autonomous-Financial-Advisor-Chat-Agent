"""
Market intelligence analyzer.

Computes market summaries, sector rankings, trend analysis, and FII/DII
context dynamically from the DataRegistry. No data values are hardcoded.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .registry import DataRegistry

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Dynamic market intelligence — all numbers come from the registry."""

    def __init__(self, registry: DataRegistry) -> None:
        self._registry = registry

    def get_full_snapshot(self) -> dict:
        """Comprehensive market snapshot used by the market_analyzer LangGraph node."""
        snapshot = self._registry.get_market_snapshot()
        return {
            **snapshot,
            "sector_ranking":    self._rank_sectors(),
            "market_sentiment":  self._compute_overall_sentiment(),
            "top_movers":        self._registry.get_top_movers(5),
            "fii_dii_analysis":  self._analyze_fii_dii(),
            "breadth_analysis":  self._analyze_breadth(),
            "index_trends":      self._analyze_index_trends(),
            "macro_context":     self._build_macro_context(),
        }

    def get_stock_detail(self, symbol: str) -> dict:
        stock = self._registry.get_stock(symbol)
        history = self._registry.get_stock_history(symbol)
        sector = stock.get("sector")
        sector_data = {}
        if sector and sector in self._registry.sector_names:
            sector_data = self._registry.get_sector(sector)

        news = self._registry.get_news_for_stock(symbol)
        return {
            "symbol":       symbol,
            "stock_data":   stock,
            "history":      history,
            "sector":       sector,
            "sector_data":  sector_data,
            "related_news": news,
            "portfolios_holding": self._registry.stock_to_portfolios.get(symbol, []),
        }

    def get_sector_detail(self, sector: str) -> dict:
        perf       = self._registry.get_sector(sector)
        definition = self._registry.get_sector_definition(sector)
        weekly     = self._registry.get_sector_weekly(sector)
        stocks     = self._registry.get_stocks_in_sector(sector)
        news       = self._registry.get_news_for_sector(sector)
        macro_corr = self._compute_macro_correlation(sector)

        return {
            "sector":           sector,
            "performance":      perf,
            "definition":       definition,
            "weekly_trend":     weekly,
            "constituent_stocks": stocks,
            "related_news":     news,
            "macro_correlation": macro_corr,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _rank_sectors(self) -> List[dict]:
        """Returns all sectors sorted by day change % (best → worst)."""
        return sorted(
            [{"sector": k, **v} for k, v in self._registry._loader.all_sectors.items()],
            key=lambda x: x.get("change_percent", 0),
            reverse=True,
        )

    def _compute_overall_sentiment(self) -> str:
        """Derived dynamically from indices sentiment field."""
        sentiments = [
            idx.get("sentiment", "NEUTRAL")
            for idx in self._registry._loader.all_indices.values()
        ]
        bearish_count = sentiments.count("BEARISH")
        bullish_count = sentiments.count("BULLISH")
        if bearish_count > bullish_count:
            return "BEARISH"
        if bullish_count > bearish_count:
            return "BULLISH"
        return "NEUTRAL"

    def _analyze_fii_dii(self) -> dict:
        fii_dii = self._registry._loader.fii_dii_data
        if not fii_dii:
            return {}
        fii = fii_dii.get("fii", {})
        dii = fii_dii.get("dii", {})
        net_fii = fii.get("net_value_cr", 0)
        net_dii = dii.get("net_value_cr", 0)
        return {
            "fii_net_cr":      net_fii,
            "dii_net_cr":      net_dii,
            "fii_mtd_cr":      fii.get("mtd_net_cr"),
            "fii_ytd_cr":      fii.get("ytd_net_cr"),
            "fii_stance":      "SELLING" if net_fii < 0 else "BUYING",
            "dii_stance":      "BUYING" if net_dii > 0 else "SELLING",
            "net_institutional": net_fii + net_dii,
            "observation":     fii_dii.get("observation", ""),
        }

    def _analyze_breadth(self) -> dict:
        breadth = self._registry._loader.market_breadth
        if not breadth:
            return {}
        nifty50 = breadth.get("nifty50", {})
        return {
            "advances":              nifty50.get("advances", 0),
            "declines":              nifty50.get("declines", 0),
            "advance_decline_ratio": nifty50.get("advance_decline_ratio", 0),
            "new_52w_highs":         breadth.get("new_52_week_highs", 0),
            "new_52w_lows":          breadth.get("new_52_week_lows", 0),
            "sentiment_indicator":   breadth.get("sentiment_indicator", "NEUTRAL"),
            "interpretation":        self._interpret_breadth(nifty50),
        }

    def _interpret_breadth(self, nifty50: dict) -> str:
        ratio = nifty50.get("advance_decline_ratio", 1.0)
        if ratio < 0.4:
            return "VERY WEAK - Broad-based selling across market"
        if ratio < 0.7:
            return "WEAK - More stocks declining than advancing"
        if ratio < 1.0:
            return "SLIGHTLY WEAK - Mild selling pressure"
        if ratio < 1.5:
            return "NEUTRAL - Balanced market"
        return "STRONG - Broad-based buying"

    def _analyze_index_trends(self) -> List[dict]:
        index_history = self._registry._loader.index_history
        result = []
        for index_name, hist_data in index_history.items():
            data_points = hist_data.get("data", [])
            if len(data_points) >= 2:
                first_close = data_points[0]["close"]
                last_close  = data_points[-1]["close"]
                period_change = ((last_close - first_close) / first_close) * 100
            else:
                period_change = 0
            result.append({
                "index":           index_name,
                "trend":           hist_data.get("trend"),
                "trend_duration":  hist_data.get("trend_duration_days"),
                "cumulative_change_pct": hist_data.get("cumulative_change_percent"),
                "support":         hist_data.get("support_level"),
                "resistance":      hist_data.get("resistance_level"),
                "period_change_pct": round(period_change, 2),
            })
        return result

    def _build_macro_context(self) -> List[str]:
        """Identify active macro themes from news + sector performance."""
        macro_rules   = self._registry._loader.macro_correlations
        all_news      = self._registry.get_all_news()
        active_themes: List[str] = []

        # Check FII/DII for FII_OUTFLOW theme
        fii_data = self._registry._loader.fii_dii_data
        if fii_data.get("fii", {}).get("net_value_cr", 0) < 0:
            active_themes.append("FII_OUTFLOW")

        # Check sector performance for rate-related themes
        banking_perf = self._registry._loader.all_sectors.get("BANKING", {})
        if banking_perf.get("change_percent", 0) < -1.5:
            active_themes.append("INTEREST_RATE_UP")

        # Check IT performance for US tech theme
        it_perf = self._registry._loader.all_sectors.get("INFORMATION_TECHNOLOGY", {})
        if it_perf.get("change_percent", 0) > 0.5:
            active_themes.append("US_TECH_SPENDING_UP")

        # Check metals for China theme
        metals_perf = self._registry._loader.all_sectors.get("METALS", {})
        if metals_perf.get("change_percent", 0) < -1.0:
            active_themes.append("CHINA_SLOWDOWN")

        return active_themes

    def _compute_macro_correlation(self, sector: str) -> dict:
        macro_rules = self._registry._loader.macro_correlations
        impacts = {"positive": [], "negative": [], "neutral": []}
        for theme, rule in macro_rules.items():
            if sector in rule.get("positive_impact", []):
                impacts["positive"].append(theme)
            elif sector in rule.get("negative_impact", []):
                impacts["negative"].append(theme)
            else:
                impacts["neutral"].append(theme)
        return impacts
