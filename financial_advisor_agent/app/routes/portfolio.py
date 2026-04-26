"""
FastAPI routes: /api/portfolio — portfolio data endpoints.
Portfolio IDs are validated dynamically against the registry.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from data_layer.portfolio_analyzer import PortfolioAnalyzer
from config import settings

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.get("/{portfolio_id}")
async def get_portfolio_analysis(portfolio_id: str, request: Request) -> dict:
    """
    Full portfolio analysis for a given portfolio_id.
    ID is validated against the live registry — no hardcoded IDs.
    """
    registry = request.app.state.registry
    if portfolio_id not in registry.portfolio_ids:
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"Portfolio '{portfolio_id}' not found.",
                "available": registry.portfolio_ids,
            },
        )
    analyzer = PortfolioAnalyzer(registry, settings)
    return analyzer.analyze(portfolio_id)


@router.get("/{portfolio_id}/risk")
async def get_portfolio_risk(portfolio_id: str, request: Request) -> dict:
    """Risk flags and concentration analysis only."""
    registry = request.app.state.registry
    if portfolio_id not in registry.portfolio_ids:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")
    analyzer = PortfolioAnalyzer(registry, settings)
    analysis = analyzer.analyze(portfolio_id)
    return {
        "portfolio_id":       portfolio_id,
        "risk_flags":         analysis["risk_flags"],
        "concentration_risk": analysis["has_concentration_risk"],
        "risk_metrics":       analysis["risk_metrics"],
    }


@router.get("/{portfolio_id}/news")
async def get_portfolio_news(portfolio_id: str, request: Request) -> dict:
    """News articles relevant to this portfolio's holdings."""
    registry = request.app.state.registry
    if portfolio_id not in registry.portfolio_ids:
        raise HTTPException(status_code=404, detail=f"Portfolio not found: {portfolio_id}")
    news = registry.get_news_for_portfolio(portfolio_id)
    return {"portfolio_id": portfolio_id, "news_count": len(news), "news": news}
