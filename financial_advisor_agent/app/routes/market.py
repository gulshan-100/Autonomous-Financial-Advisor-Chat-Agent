"""
FastAPI routes: /api/market — market data endpoints.
All responses are computed dynamically from the DataRegistry.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from data_layer.market_analyzer import MarketAnalyzer

router = APIRouter(prefix="/api/market", tags=["Market"])


@router.get("/snapshot")
async def market_snapshot(request: Request) -> dict:
    """Full market snapshot: indices, sector performance, breadth, FII/DII."""
    registry = request.app.state.registry
    analyzer = MarketAnalyzer(registry)
    return analyzer.get_full_snapshot()


@router.get("/movers")
async def market_movers(
    request: Request,
    n: int = Query(5, ge=1, le=20),
) -> dict:
    """Top N gainers and losers — dynamically computed from all stocks."""
    registry = request.app.state.registry
    return registry.get_top_movers(n)


@router.get("/sectors")
async def all_sectors(request: Request) -> list:
    """All sector performance data sorted by day change %."""
    registry = request.app.state.registry
    analyzer = MarketAnalyzer(registry)
    return analyzer._rank_sectors()


@router.get("/sector/{sector}")
async def sector_detail(sector: str, request: Request) -> dict:
    """Detailed sector analysis including constituents, news, macro correlation."""
    registry = request.app.state.registry
    if sector not in registry.sector_names:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Unknown sector '{sector}'", "available": registry.sector_names},
        )
    analyzer = MarketAnalyzer(registry)
    return analyzer.get_sector_detail(sector)


@router.get("/stock/{symbol}")
async def stock_detail(symbol: str, request: Request) -> dict:
    """Stock detail: price, history, sector, related news."""
    registry = request.app.state.registry
    if symbol not in registry.stock_symbols:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Unknown symbol '{symbol}'", "available": registry.stock_symbols},
        )
    analyzer = MarketAnalyzer(registry)
    return analyzer.get_stock_detail(symbol)


@router.get("/news")
async def market_news(
    request: Request,
    scope: str | None = Query(None),
    sentiment: str | None = Query(None),
    limit: int = Query(10, ge=1, le=25),
) -> list:
    """
    Filtered news feed. scope: MARKET_WIDE | SECTOR_SPECIFIC | STOCK_SPECIFIC
    All articles come from news_data.json — dynamically filtered.
    """
    registry = request.app.state.registry
    news = registry.get_all_news()
    if scope:
        news = [n for n in news if n.get("scope") == scope]
    if sentiment:
        news = [n for n in news if n.get("sentiment") == sentiment.upper()]
    return news[:limit]
