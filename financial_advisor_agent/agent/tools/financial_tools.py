"""
LangChain Tool definitions for the Financial Advisor Agent.

These are the ONLY way the agent accesses financial data. The LLM autonomously
decides which tools to call, when, and in what order — purely based on what
the user's query requires. No data is pre-fetched, pre-routed, or stuffed
into prompts before the LLM sees the question.

Every number, price, portfolio value, news headline, or risk flag in the final
response MUST originate from a tool call result.
"""
from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from data_layer.registry import DataRegistry
from data_layer.portfolio_analyzer import PortfolioAnalyzer
from data_layer.market_analyzer import MarketAnalyzer
from data_layer.news_processor import NewsProcessor
from config import Settings

logger = logging.getLogger(__name__)


def build_financial_tools(registry: DataRegistry, settings: Settings) -> list:
    """
    Factory: creates all 10 financial tools with registry/settings injected via closure.
    Call llm.bind_tools(build_financial_tools(registry, settings)) to give the LLM
    autonomous access to all financial data.
    """
    portfolio_analyzer = PortfolioAnalyzer(registry, settings)
    market_analyzer    = MarketAnalyzer(registry)
    news_processor     = NewsProcessor(registry, settings)

    # ── Tool 0: Think (chain-of-thought scratchpad) ───────────────────────────

    @tool
    def think(thought: str) -> str:
        """
        Use this tool to reason and plan BEFORE calling data tools.
        Write your step-by-step thinking here: what the user needs, which tools
        will help, what to look for in the results, and how to synthesize them.
        This helps you avoid unnecessary tool calls and produce better answers.
        Call this FIRST for any complex or multi-part question.
        Args:
            thought: Your detailed reasoning and action plan as a free-form string.
        """
        logger.debug("Agent think: %s", thought[:120])
        return f"Reasoning recorded. Proceed with your plan."

    # ── Tool 1: List Portfolios ────────────────────────────────────────────────

    @tool
    def list_portfolios() -> str:
        """
        List all available user portfolios with their IDs, user names, types,
        risk profiles, current values, and overall gain/loss percentages.
        Call this first when the user mentions 'my portfolio' but no portfolio
        is selected, or when you need to discover available portfolio IDs.
        """
        try:
            summaries = registry.get_all_portfolios_summary()
            return json.dumps(summaries, indent=2)
        except Exception as exc:
            logger.error("list_portfolios failed: %s", exc)
            return f"Error listing portfolios: {exc}"

    # ── Tool 2: Portfolio Analysis ─────────────────────────────────────────────

    @tool
    def get_portfolio_analysis(portfolio_id: str) -> str:
        """
        Get comprehensive portfolio analysis including all holdings with day P&L,
        sector allocation breakdown, risk metrics (beta, Sharpe ratio, volatility,
        max drawdown), risk flags, overall gain/loss since inception, and the
        top moving stock today.
        Use this when the user asks about portfolio performance, holdings, P&L,
        investments, or wants to know how their portfolio is doing.
        Args:
            portfolio_id: Portfolio ID string e.g. 'P001', 'P002'.
                          Call list_portfolios() first if you are unsure of the ID.
        """
        try:
            data = portfolio_analyzer.analyze(portfolio_id)
            stocks = data.get("stock_analysis", [])
            mfs    = data.get("mf_analysis", [])
            result = {
                "portfolio_id":           data.get("portfolio_id"),
                "user_name":              data.get("user_name"),
                "portfolio_type":         data.get("portfolio_type"),
                "risk_profile":           data.get("risk_profile"),
                "investment_horizon":     data.get("investment_horizon"),
                "current_value":          data.get("current_value"),
                "invested_value":         data.get("total_investment"),
                "overall_gain_pct":       data.get("overall_gain_loss_pct"),
                "overall_gain_abs":       data.get("overall_gain_loss"),
                "day_change_pct":         data.get("day_change_pct"),
                "day_change_abs":         data.get("day_change_absolute"),
                "top_mover":              data.get("top_mover"),
                "holdings_count":         len(stocks) + len(mfs),
                "sector_allocation":      data.get("sector_allocation"),
                "asset_type_allocation":  data.get("asset_type_allocation"),
                "risk_metrics":           data.get("risk_metrics"),
                "risk_flags":             data.get("risk_flags"),
                "has_concentration_risk": data.get("has_concentration_risk"),
                "causal_summary":         data.get("causal_summary"),
                "stock_analysis":         stocks[:12],
                "mf_analysis":            mfs[:6],
            }
            return json.dumps(result, indent=2)
        except ValueError:
            return (
                f"Portfolio '{portfolio_id}' not found. "
                "Call list_portfolios() to see all valid IDs."
            )
        except Exception as exc:
            logger.error("get_portfolio_analysis(%s) failed: %s", portfolio_id, exc)
            return f"Error analyzing portfolio: {exc}"

    # ── Tool 3: Portfolio Risk ─────────────────────────────────────────────────

    @tool
    def get_portfolio_risk(portfolio_id: str) -> str:
        """
        Get deep risk analysis for a portfolio: concentration risk flags, sector
        risk cross-referenced against current market conditions (bearish sectors
        where the portfolio has heavy allocation), news-driven risk alerts for
        holdings, conflict signals, beta, Sharpe ratio, and max drawdown.
        Use this when assessing risk, before making investment recommendations,
        or when the user asks about portfolio safety or concerns.
        Args:
            portfolio_id: Portfolio ID string e.g. 'P001'.
        """
        try:
            data      = portfolio_analyzer.analyze(portfolio_id)
            market    = market_analyzer.get_full_snapshot()
            conflicts = news_processor.identify_conflict_signals()

            sector_alloc = data.get("sector_allocation", {})
            sector_perf  = market.get("sector_performance", {})
            market_risk_warnings = []
            for sector, pct in sector_alloc.items():
                if sector in sector_perf and pct > settings.sector_concentration_warning_pct:
                    change    = sector_perf[sector].get("change_percent", 0)
                    sentiment = sector_perf[sector].get("sentiment", "NEUTRAL")
                    if sentiment == "BEARISH" and change < -1.5:
                        market_risk_warnings.append(
                            f"{sector}: {pct:.1f}% allocation — currently {change:+.2f}% BEARISH today"
                        )

            return json.dumps({
                "portfolio_id":           portfolio_id,
                "user_name":              data.get("user_name"),
                "risk_flags":             data.get("risk_flags", []),
                "market_risk_warnings":   market_risk_warnings,
                "has_concentration_risk": data.get("has_concentration_risk"),
                "risk_metrics": {
                    "beta":         data.get("risk_metrics", {}).get("beta"),
                    "sharpe_ratio": data.get("risk_metrics", {}).get("sharpe_ratio"),
                    "volatility":   data.get("risk_metrics", {}).get("volatility"),
                    "max_drawdown": data.get("risk_metrics", {}).get("max_drawdown"),
                },
                "conflict_signals": [
                    {
                        "headline":    c.get("headline"),
                        "explanation": c.get("conflict_explanation"),
                        "sectors":     [s["sector"] for s in c.get("sector_context", [])],
                    }
                    for c in conflicts[:4]
                ],
            }, indent=2)
        except ValueError:
            return f"Portfolio '{portfolio_id}' not found."
        except Exception as exc:
            logger.error("get_portfolio_risk(%s) failed: %s", portfolio_id, exc)
            return f"Error: {exc}"

    # ── Tool 4: Market Overview ────────────────────────────────────────────────

    @tool
    def get_market_overview() -> str:
        """
        Get the current full market overview: all major Indian indices (NIFTY 50,
        SENSEX, NIFTY BANK, NIFTY IT, etc.) with values and % changes, all sectors
        ranked from best to worst performer today, top 5 gainers and losers,
        FII/DII institutional flow data, market breadth (advance/decline ratio),
        index trend analysis, and active macro themes driving the market.
        Call this for any broad market question, market summary request, or
        when you need context about what sectors or indices are doing.
        """
        try:
            data = market_analyzer.get_full_snapshot()
            return json.dumps(data, indent=2)
        except Exception as exc:
            logger.error("get_market_overview failed: %s", exc)
            return f"Error fetching market data: {exc}"

    # ── Tool 5: Stock Details ──────────────────────────────────────────────────

    @tool
    def get_stock_details(symbol: str) -> str:
        """
        Get detailed data for a specific NSE-listed stock: current price, % change
        today, 52-week high/low, volume, the sector it belongs to, its sector's
        current performance, recent news affecting this stock, and which portfolios
        currently hold it.
        Args:
            symbol: NSE ticker in CAPS e.g. 'INFY', 'RELIANCE', 'HDFCBANK', 'TCS',
                    'WIPRO', 'HDFC', 'ICICIBANK'. Use exact NSE symbol.
        """
        try:
            data = market_analyzer.get_stock_detail(symbol)
            return json.dumps(data, indent=2)
        except Exception as exc:
            logger.error("get_stock_details(%s) failed: %s", symbol, exc)
            return (
                f"Stock '{symbol}' not found or error: {exc}. "
                "Verify the exact NSE ticker symbol."
            )

    # ── Tool 6: Sector Analysis ────────────────────────────────────────────────

    @tool
    def get_sector_analysis(sector: str) -> str:
        """
        Get comprehensive analysis for a specific market sector: today's % change
        and sentiment, all constituent stocks with individual performance, weekly
        performance trend, macro correlation factors (which macroeconomic events
        positively or negatively impact this sector), and related news.
        Args:
            sector: Sector name in CAPS e.g. 'BANKING', 'INFORMATION_TECHNOLOGY',
                    'PHARMA', 'AUTO', 'FMCG', 'METALS', 'ENERGY', 'REALTY',
                    'CONSUMER_DURABLES'. Use the exact sector name from the system.
        """
        try:
            data = market_analyzer.get_sector_detail(sector)
            return json.dumps(data, indent=2)
        except Exception as exc:
            logger.error("get_sector_analysis(%s) failed: %s", sector, exc)
            return f"Sector '{sector}' not found or error: {exc}."

    # ── Tool 7: Search News ────────────────────────────────────────────────────

    @tool
    def search_news(symbol: str = "", sector: str = "", top_n: int = 8) -> str:
        """
        Search and retrieve financial news articles. Filter by stock symbol or
        sector. Each article includes: headline, sentiment
        (POSITIVE/NEGATIVE/NEUTRAL/MIXED), impact level (HIGH/MEDIUM/LOW),
        scope (MARKET_WIDE/SECTOR/STOCK), affected stocks and sectors, causal
        factors explaining why the news matters, and a conflict_flag if the news
        contradicts the broader sector sentiment (useful for spotting anomalies).
        Args:
            symbol: Optional NSE ticker to filter news e.g. 'INFY'. Leave '' for all.
            sector: Optional sector name to filter e.g. 'BANKING'. Leave '' for all.
            top_n: Number of articles to return. Default 8.
        """
        try:
            if symbol and symbol in registry.stock_symbols:
                articles = registry.get_news_for_stock(symbol)
            elif sector and sector in registry.sector_names:
                articles = registry.get_news_for_sector(sector)
            else:
                articles = news_processor.get_top_impacting_news(n=top_n)

            return json.dumps([
                {
                    "id":                   a.get("id"),
                    "headline":             a.get("headline"),
                    "sentiment":            a.get("sentiment"),
                    "sentiment_score":      a.get("sentiment_score"),
                    "impact_level":         a.get("impact_level"),
                    "scope":                a.get("scope"),
                    "stocks":               a.get("entities", {}).get("stocks", []),
                    "sectors":              a.get("entities", {}).get("sectors", []),
                    "causal_factors":       a.get("causal_factors", []),
                    "conflict_flag":        a.get("conflict_flag", False),
                    "conflict_explanation": a.get("conflict_explanation", ""),
                }
                for a in articles[:top_n]
            ], indent=2)
        except Exception as exc:
            logger.error("search_news failed: %s", exc)
            return f"Error fetching news: {exc}"

    # ── Tool 8: Top Movers ─────────────────────────────────────────────────────

    @tool
    def get_top_movers(n: int = 5) -> str:
        """
        Get today's top N gaining and top N losing stocks across the entire market.
        Returns symbol, current price, % change, and sector for each stock.
        Args:
            n: Number of stocks per category (gainers and losers). Default 5.
        """
        try:
            data = registry.get_top_movers(n=n)
            return json.dumps(data, indent=2)
        except Exception as exc:
            logger.error("get_top_movers failed: %s", exc)
            return f"Error: {exc}"

    # ── Tool 9: Mutual Fund Details ────────────────────────────────────────────

    @tool
    def get_mutual_fund_details(scheme_code: str) -> str:
        """
        Get mutual fund details: fund name, category (Large Cap/Mid Cap/Debt/etc.),
        current NAV, 1-year/3-year/5-year returns, AUM, fund manager, expense ratio,
        and top portfolio holdings of the fund.
        Args:
            scheme_code: Fund scheme code e.g. 'MF001', 'MF002'.
                         Call get_portfolio_analysis() to see which scheme codes
                         a user's portfolio holds.
        """
        try:
            data = registry.get_mutual_fund(scheme_code)
            return json.dumps(data, indent=2)
        except ValueError as exc:
            avail = registry.mutual_fund_codes
            return f"Fund '{scheme_code}' not found. Available codes: {avail}"
        except Exception as exc:
            logger.error("get_mutual_fund_details(%s) failed: %s", scheme_code, exc)
            return f"Error: {exc}"

    # ── Tool 10: Build Causal Chain ────────────────────────────────────────────

    @tool
    def build_causal_chain(portfolio_id: str = "", symbol: str = "") -> str:
        """
        Build a causal chain analysis tracing how macro/global events flow into
        sector impacts, then into specific stock movements, and finally into
        portfolio P&L. Identifies which news events are the root causes of
        today's market moves and how they affect specific holdings.
        Use this for deep 'why' questions: 'Why is my portfolio down?',
        'What caused this sector to fall?', 'What is driving INFY today?'
        Args:
            portfolio_id: Optional. Portfolio ID for a portfolio-specific chain.
            symbol: Optional. NSE ticker for a stock-specific chain.
            Leave both empty for a market-wide causal chain analysis.
        """
        try:
            if portfolio_id and portfolio_id in registry.portfolio_ids:
                data = news_processor.build_portfolio_causal_chain(portfolio_id)
                return json.dumps(data, indent=2)

            if symbol and symbol in registry.stock_symbols:
                stock_news = registry.get_news_for_stock(symbol)
                stock_data = market_analyzer.get_stock_detail(symbol)
                return json.dumps({
                    "type":              "stock_causal_chain",
                    "symbol":            symbol,
                    "current_data":      stock_data.get("stock_data", {}),
                    "driving_news":      stock_news[:5],
                    "macro_correlation": stock_data.get("sector_data", {}).get(
                        "macro_correlation", {}
                    ),
                }, indent=2)

            # Market-wide causal chain
            market   = market_analyzer.get_full_snapshot()
            top_news = news_processor.get_top_impacting_news(n=6)
            return json.dumps({
                "type":               "market_causal_chain",
                "market_sentiment":   market.get("market_sentiment"),
                "active_macro_themes": market.get("macro_context", []),
                "top_moving_sectors": market.get("sector_ranking", [])[:3],
                "top_impacting_news": top_news,
                "fii_activity":       market.get("fii_dii_analysis", {}),
            }, indent=2)

        except Exception as exc:
            logger.error("build_causal_chain failed: %s", exc)
            return f"Error building causal chain: {exc}"

    # ── Return all tools ───────────────────────────────────────────────────────

    return [
        think,
        list_portfolios,
        get_portfolio_analysis,
        get_portfolio_risk,
        get_market_overview,
        get_stock_details,
        get_sector_analysis,
        search_news,
        get_top_movers,
        get_mutual_fund_details,
        build_causal_chain,
    ]
