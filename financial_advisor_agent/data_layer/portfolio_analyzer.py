"""
Portfolio analytics engine.

Computes risk metrics, P&L attribution, concentration flags, and news impact
dynamically from live portfolio data and the DataRegistry.
All thresholds are pulled from Settings — never hardcoded.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from config import Settings
from .registry import DataRegistry

logger = logging.getLogger(__name__)


class PortfolioAnalyzer:
    """Runs dynamic portfolio analysis given a portfolio_id."""

    def __init__(self, registry: DataRegistry, settings: Settings) -> None:
        self._registry = registry
        self._settings = settings

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(self, portfolio_id: str) -> dict:
        """
        Full portfolio analysis. Returns a rich dict consumed by:
          - LangGraph risk_analyzer node
          - /api/portfolio/{id} endpoint
          - Frontend portfolio panel
        """
        portfolio = self._registry.get_portfolio(portfolio_id)
        holdings  = portfolio.get("holdings", {})
        analytics = portfolio.get("analytics", {})

        stock_analysis  = self._analyze_stocks(holdings.get("stocks", []))
        mf_analysis     = self._analyze_mutual_funds(holdings.get("mutual_funds", []))
        risk_flags      = self._compute_risk_flags(portfolio_id, portfolio, analytics)
        news_impact     = self._registry.get_news_for_portfolio(portfolio_id)
        causal_summary  = self._build_causal_summary(risk_flags, news_impact)
        top_mover       = self._get_top_mover(holdings.get("stocks", []))

        return {
            "portfolio_id":           portfolio_id,
            "user_name":              portfolio.get("user_name"),
            "portfolio_type":         portfolio.get("portfolio_type"),
            "risk_profile":           portfolio.get("risk_profile"),
            "investment_horizon":     portfolio.get("investment_horizon"),
            "description":            portfolio.get("description"),
            "total_investment":       portfolio.get("total_investment"),
            "current_value":          portfolio.get("current_value"),
            "overall_gain_loss":      portfolio.get("overall_gain_loss"),
            "overall_gain_loss_pct":  portfolio.get("overall_gain_loss_percent"),
            "day_change_absolute":    analytics.get("day_summary", {}).get("day_change_absolute"),
            "day_change_pct":         analytics.get("day_summary", {}).get("day_change_percent"),
            "sector_allocation":      analytics.get("sector_allocation", {}),
            "asset_type_allocation":  analytics.get("asset_type_allocation", {}),
            "risk_metrics":           analytics.get("risk_metrics", {}),
            "stock_analysis":         stock_analysis,
            "mf_analysis":            mf_analysis,
            "risk_flags":             risk_flags,
            "has_concentration_risk": any("CRITICAL" in f or "WARNING" in f for f in risk_flags),
            "news_impact":            news_impact,
            "causal_summary":         causal_summary,
            "top_mover":              top_mover,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _analyze_stocks(self, stocks: List[dict]) -> List[dict]:
        """Enrich each stock holding with live market data from the registry."""
        enriched = []
        for holding in stocks:
            symbol = holding["symbol"]
            try:
                market_data = self._registry.get_stock(symbol)
                history     = self._registry.get_stock_history(symbol)
                news        = self._registry.get_news_for_stock(symbol)
            except ValueError:
                # Stock in portfolio not found in market data — handle gracefully
                market_data, history, news = {}, None, []

            enriched.append({
                **holding,
                "market_price":       market_data.get("current_price"),
                "beta":               market_data.get("beta"),
                "volume":             market_data.get("volume"),
                "avg_volume_20d":     market_data.get("avg_volume_20d"),
                "pe_ratio":           market_data.get("pe_ratio"),
                "week_52_high":       market_data.get("52_week_high"),
                "week_52_low":        market_data.get("52_week_low"),
                "trend":              history.get("trend") if history else None,
                "volatility":         history.get("volatility") if history else None,
                "related_news_count": len(news),
                "related_news":       news[:3],   # top 3 relevant news
                "risk_level":         self._classify_stock_risk(holding, market_data),
            })
        return enriched

    def _analyze_mutual_funds(self, mf_holdings: List[dict]) -> List[dict]:
        """Enrich each MF holding with detailed fund data from the registry."""
        enriched = []
        for holding in mf_holdings:
            code = holding.get("scheme_code", "")
            try:
                fund_data = self._registry.get_mutual_fund(code)
            except ValueError:
                fund_data = {}
            enriched.append({
                **holding,
                "risk_rating":        fund_data.get("risk_rating"),
                "expense_ratio":      fund_data.get("expense_ratio"),
                "aum_cr":             fund_data.get("aum_cr"),
                "fund_manager":       fund_data.get("fund_manager"),
                "benchmark":          fund_data.get("benchmark"),
                "returns":            fund_data.get("returns", {}),
                "sector_allocation":  fund_data.get("sector_allocation", {}),
                "top_holdings":       fund_data.get("top_holdings", [])
                                      or fund_data.get("top_equity_holdings", []),
            })
        return enriched

    def _compute_risk_flags(
        self,
        portfolio_id: str,
        portfolio: dict,
        analytics: dict,
    ) -> List[str]:
        """
        Dynamic risk flag computation.
        All thresholds from self._settings — no magic numbers in code.
        """
        flags: List[str] = []
        sector_alloc  = analytics.get("sector_allocation", {})
        risk_metrics  = analytics.get("risk_metrics", {})

        # 1. Sector concentration risk (iterates dynamically over all sectors)
        for sector, pct in sector_alloc.items():
            if pct >= self._settings.sector_concentration_critical_pct:
                flags.append(
                    f"🚨 CRITICAL: {sector} allocation = {pct:.1f}% "
                    f"(threshold: {self._settings.sector_concentration_critical_pct}%)"
                )
            elif pct >= self._settings.sector_concentration_warning_pct:
                flags.append(
                    f"⚠️ WARNING: {sector} allocation = {pct:.1f}% "
                    f"(threshold: {self._settings.sector_concentration_warning_pct}%)"
                )

        # 2. Single-stock concentration
        max_weight = risk_metrics.get("single_stock_max_weight", 0)
        if max_weight >= self._settings.single_stock_warning_pct:
            # Find which stock is the heaviest (dynamic)
            stocks = portfolio.get("holdings", {}).get("stocks", [])
            if stocks:
                heaviest = max(stocks, key=lambda h: h.get("weight_in_portfolio", 0))
                flags.append(
                    f"⚠️ HIGH SINGLE-STOCK: {heaviest['symbol']} = "
                    f"{heaviest['weight_in_portfolio']:.1f}% of portfolio"
                )

        # 3. Beta / market sensitivity
        beta = risk_metrics.get("beta", 0)
        if beta >= self._settings.high_beta_threshold:
            flags.append(
                f"⚠️ HIGH BETA: Portfolio β = {beta:.2f} — "
                f"portfolio is highly sensitive to market moves"
            )

        # 4. Explicit concentration_risk flag from data
        if risk_metrics.get("concentration_risk"):
            warning_text = risk_metrics.get("concentration_warning", "")
            if warning_text and not any(warning_text in f for f in flags):
                flags.append(f"🚨 DATA FLAG: {warning_text}")

        return flags

    def _classify_stock_risk(self, holding: dict, market_data: dict) -> str:
        beta = market_data.get("beta", 1.0)
        weight = holding.get("weight_in_portfolio", 0)
        if beta >= self._settings.high_beta_threshold and weight >= self._settings.single_stock_warning_pct:
            return "VERY_HIGH"
        if beta >= self._settings.high_beta_threshold or weight >= self._settings.single_stock_warning_pct:
            return "HIGH"
        if beta >= 1.0:
            return "MODERATE"
        return "LOW"

    def _build_causal_summary(
        self, risk_flags: List[str], news_impact: List[dict]
    ) -> str:
        """
        Build a short causal summary string from risk flags + top news.
        Used as a quick context hint for the LangGraph news_reasoner node.
        """
        if not risk_flags and not news_impact:
            return "No significant risk events identified."
        lines = []
        if risk_flags:
            lines.append("Risk flags: " + "; ".join(risk_flags[:2]))
        high_impact_news = [n for n in news_impact if n.get("impact_level") == "HIGH"]
        if high_impact_news:
            headline = high_impact_news[0].get("headline", "")
            lines.append(f"Top news: {headline}")
        return " | ".join(lines)

    def _get_top_mover(self, stocks: List[dict]) -> Optional[dict]:
        if not stocks:
            return None
        # Dynamic — finds max abs change across all holdings
        return max(stocks, key=lambda h: abs(h.get("day_change_percent", 0)))
