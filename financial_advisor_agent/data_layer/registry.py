"""
DataRegistry — the single source of truth for all discovered entities.

Built once at server startup by scanning the raw JSON files through DataLoader.
Every downstream component (tools, LangGraph nodes, API routes, frontend)
queries the registry instead of hardcoding IDs, names, or keys.

Zero hardcoding policy:
  ✓ portfolio_ids      → from portfolios.json keys
  ✓ stock_symbols      → from market_data.stocks keys
  ✓ sector_names       → from market_data.sector_performance keys
  ✓ index_names        → from market_data.indices keys
  ✓ news_ids           → from news_data.news[].id
  ✓ mutual_fund_codes  → from mutual_funds.mutual_funds keys
  ✓ historical symbols → from historical_data.stock_history keys
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .loader import DataLoader

logger = logging.getLogger(__name__)


class DataRegistry:
    """
    Auto-discovers and indexes all financial entities from the loaded JSON data.
    Exposes typed, validated query methods used by every layer of the system.
    """

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader
        self._build()
        logger.info(
            "DataRegistry built: %d portfolios, %d stocks, %d sectors, "
            "%d news articles, %d mutual funds.",
            len(self.portfolio_ids),
            len(self.stock_symbols),
            len(self.sector_names),
            len(self.news_ids),
            len(self.mutual_fund_codes),
        )

    # ── Build — scans all keys dynamically ────────────────────────────────────

    def _build(self) -> None:
        # Portfolios
        raw_portfolios = self._loader.all_portfolios
        self.portfolio_ids: List[str] = list(raw_portfolios.keys())
        self.user_map: Dict[str, str] = {
            pid: data["user_name"] for pid, data in raw_portfolios.items()
        }

        # Stocks (keys from market_data.stocks)
        self.stock_symbols: List[str] = list(self._loader.all_stocks.keys())

        # Sectors (keys from market_data.sector_performance)
        self.sector_names: List[str] = list(self._loader.all_sectors.keys())

        # Indices (keys from market_data.indices)
        self.index_names: List[str] = list(self._loader.all_indices.keys())

        # News (ids from news_data.news array)
        self.news_ids: List[str] = [
            article["id"] for article in self._loader.all_news
        ]

        # Mutual funds (keys from mutual_funds.mutual_funds)
        self.mutual_fund_codes: List[str] = list(self._loader.all_mutual_funds.keys())

        # Historical symbols (union of stock_history + index_history keys)
        self.historical_symbols: List[str] = list(
            set(self._loader.stock_history.keys())
            | set(self._loader.index_history.keys())
        )

        # Reverse maps — computed dynamically from data
        self.sector_to_stocks: Dict[str, List[str]] = self._build_sector_to_stocks()
        self.stock_to_portfolios: Dict[str, List[str]] = self._build_stock_to_portfolios()
        self.mf_to_portfolios: Dict[str, List[str]] = self._build_mf_to_portfolios()

    def _build_sector_to_stocks(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for symbol, stock_data in self._loader.all_stocks.items():
            sector = stock_data.get("sector", "UNKNOWN")
            mapping.setdefault(sector, []).append(symbol)
        return mapping

    def _build_stock_to_portfolios(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for pid, portfolio in self._loader.all_portfolios.items():
            for holding in portfolio.get("holdings", {}).get("stocks", []):
                sym = holding["symbol"]
                mapping.setdefault(sym, []).append(pid)
        return mapping

    def _build_mf_to_portfolios(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for pid, portfolio in self._loader.all_portfolios.items():
            for mf in portfolio.get("holdings", {}).get("mutual_funds", []):
                code = mf["scheme_code"]
                mapping.setdefault(code, []).append(pid)
        return mapping

    # ── Portfolio queries ──────────────────────────────────────────────────────

    def get_portfolio(self, portfolio_id: str) -> dict:
        portfolios = self._loader.all_portfolios
        if portfolio_id not in portfolios:
            raise ValueError(
                f"Unknown portfolio '{portfolio_id}'. Valid IDs: {self.portfolio_ids}"
            )
        return portfolios[portfolio_id]

    def get_all_portfolios_summary(self) -> List[dict]:
        return [
            {
                "id": pid,
                "user_name": self.user_map[pid],
                "portfolio_type": self.get_portfolio(pid).get("portfolio_type"),
                "risk_profile": self.get_portfolio(pid).get("risk_profile"),
                "investment_horizon": self.get_portfolio(pid).get("investment_horizon"),
                "current_value": self.get_portfolio(pid).get("current_value"),
                "overall_gain_loss_percent": self.get_portfolio(pid).get(
                    "overall_gain_loss_percent"
                ),
            }
            for pid in self.portfolio_ids
        ]

    def get_portfolio_holdings_symbols(self, portfolio_id: str) -> List[str]:
        """Return all stock symbols held in a portfolio."""
        portfolio = self.get_portfolio(portfolio_id)
        return [h["symbol"] for h in portfolio.get("holdings", {}).get("stocks", [])]

    def get_portfolio_mf_codes(self, portfolio_id: str) -> List[str]:
        """Return all mutual fund scheme codes held in a portfolio."""
        portfolio = self.get_portfolio(portfolio_id)
        return [h["scheme_code"] for h in portfolio.get("holdings", {}).get("mutual_funds", [])]

    # ── Stock queries ──────────────────────────────────────────────────────────

    def get_stock(self, symbol: str) -> dict:
        stocks = self._loader.all_stocks
        if symbol not in stocks:
            raise ValueError(
                f"Unknown symbol '{symbol}'. Valid symbols: {self.stock_symbols}"
            )
        return stocks[symbol]

    def get_stocks_in_sector(self, sector: str) -> List[dict]:
        """Return all stock dicts belonging to a given sector (from market data)."""
        return [
            {"symbol": sym, **self.get_stock(sym)}
            for sym in self.sector_to_stocks.get(sector, [])
        ]

    def get_stock_history(self, symbol: str) -> Optional[dict]:
        return self._loader.stock_history.get(symbol)

    def get_index_history(self, index: str) -> Optional[dict]:
        return self._loader.index_history.get(index)

    # ── Sector queries ─────────────────────────────────────────────────────────

    def get_sector(self, sector: str) -> dict:
        sectors = self._loader.all_sectors
        if sector not in sectors:
            raise ValueError(
                f"Unknown sector '{sector}'. Valid sectors: {self.sector_names}"
            )
        return sectors[sector]

    def get_sector_definition(self, sector: str) -> dict:
        """Extended sector definition from sector_mapping.json (metrics, stocks list)."""
        return self._loader.all_sector_definitions.get(sector, {})

    def get_sector_weekly(self, sector: str) -> Optional[dict]:
        return self._loader.sector_weekly_performance.get(sector)

    def get_all_sectors_performance(self) -> List[dict]:
        return [
            {"sector": s, **self.get_sector(s)} for s in self.sector_names
        ]

    # ── News queries ───────────────────────────────────────────────────────────

    def get_all_news(self) -> List[dict]:
        return self._loader.all_news

    def get_news_by_id(self, news_id: str) -> Optional[dict]:
        for article in self._loader.all_news:
            if article["id"] == news_id:
                return article
        return None

    def get_news_for_stock(self, symbol: str) -> List[dict]:
        return [
            n for n in self._loader.all_news
            if symbol in n.get("entities", {}).get("stocks", [])
        ]

    def get_news_for_sector(self, sector: str) -> List[dict]:
        return [
            n for n in self._loader.all_news
            if sector in n.get("entities", {}).get("sectors", [])
        ]

    def get_news_by_scope(self, scope: str) -> List[dict]:
        return [n for n in self._loader.all_news if n.get("scope") == scope]

    def get_news_by_sentiment(self, sentiment: str) -> List[dict]:
        return [n for n in self._loader.all_news if n.get("sentiment") == sentiment]

    def get_news_for_portfolio(self, portfolio_id: str) -> List[dict]:
        """Return all news articles that affect any holding in the given portfolio."""
        holding_symbols = set(self.get_portfolio_holdings_symbols(portfolio_id))
        portfolio = self.get_portfolio(portfolio_id)
        holding_sectors = {
            h["sector"]
            for h in portfolio.get("holdings", {}).get("stocks", [])
        }
        seen_ids: set = set()
        relevant: List[dict] = []
        for article in self._loader.all_news:
            entities = article.get("entities", {})
            article_stocks  = set(entities.get("stocks", []))
            article_sectors = set(entities.get("sectors", []))
            if (article_stocks & holding_symbols) or (article_sectors & holding_sectors):
                if article["id"] not in seen_ids:
                    seen_ids.add(article["id"])
                    relevant.append(article)
        return sorted(
            relevant,
            key=lambda x: (
                {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x.get("impact_level", "LOW"), 0),
                abs(x.get("sentiment_score", 0)),
            ),
            reverse=True,
        )

    # ── Mutual fund queries ────────────────────────────────────────────────────

    def get_mutual_fund(self, scheme_code: str) -> dict:
        funds = self._loader.all_mutual_funds
        if scheme_code not in funds:
            raise ValueError(
                f"Unknown mutual fund '{scheme_code}'. Valid codes: {self.mutual_fund_codes}"
            )
        return funds[scheme_code]

    # ── Market snapshot ────────────────────────────────────────────────────────

    def get_market_snapshot(self) -> dict:
        return {
            "metadata":           self._loader.market_data.get("metadata", {}),
            "indices":            self._loader.all_indices,
            "sector_performance": self._loader.all_sectors,
            "market_breadth":     self._loader.market_breadth,
            "fii_dii":            self._loader.fii_dii_data,
        }

    def get_top_movers(self, n: int = 5) -> dict:
        """Dynamically compute top gainers and losers from all stocks in market_data."""
        stocks = self._loader.all_stocks
        ranked = sorted(
            stocks.items(),
            key=lambda item: item[1].get("change_percent", 0),
        )
        losers  = [{"symbol": s, **d} for s, d in ranked[:n]]
        gainers = [{"symbol": s, **d} for s, d in reversed(ranked[-n:])]
        return {"top_gainers": gainers, "top_losers": losers}

    def get_worst_sectors(self, n: int = 3) -> List[dict]:
        """Dynamically find worst performing sectors."""
        return sorted(
            [{"sector": k, **v} for k, v in self._loader.all_sectors.items()],
            key=lambda x: x.get("change_percent", 0),
        )[:n]

    def get_best_sectors(self, n: int = 3) -> List[dict]:
        return sorted(
            [{"sector": k, **v} for k, v in self._loader.all_sectors.items()],
            key=lambda x: x.get("change_percent", 0),
            reverse=True,
        )[:n]

    # ── Registry summary (used by /api/registry endpoint) ─────────────────────

    def get_registry_summary(self) -> dict:
        """
        Full entity summary returned to the frontend on load.
        The frontend builds ALL UI elements (dropdowns, chips, dashboard) from this.
        Nothing is hardcoded in HTML.
        """
        return {
            "portfolios":         self.get_all_portfolios_summary(),
            "stocks":             self.stock_symbols,
            "sectors":            self.sector_names,
            "indices":            self.index_names,
            "news_count":         len(self.news_ids),
            "mutual_fund_codes":  self.mutual_fund_codes,
            "market_date":        self._loader.market_data.get("metadata", {}).get("date", ""),
            "market_status":      self._loader.market_data.get("metadata", {}).get("market_status", ""),
            "currency":           self._loader.market_data.get("metadata", {}).get("currency", "INR"),
            "top_movers":         self.get_top_movers(3),
            "worst_sectors":      self.get_worst_sectors(3),
            "best_sectors":       self.get_best_sectors(3),
            "defensive_sectors":  self._loader.sector_mapping.get("defensive_sectors", []),
            "cyclical_sectors":   self._loader.sector_mapping.get("cyclical_sectors", []),
            "rate_sensitive_sectors": self._loader.sector_mapping.get("rate_sensitive_sectors", []),
        }
