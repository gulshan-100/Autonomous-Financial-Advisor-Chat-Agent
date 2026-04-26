"""
FastAPI routes: /api/registry — serves entity discovery to the frontend.
The frontend fetches this on load to build all UI elements dynamically.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["Registry"])


@router.get("/registry")
async def get_registry(request: Request) -> dict:
    """
    Returns the complete entity registry built from all 6 JSON files at startup.
    The frontend uses this to dynamically populate:
    - Portfolio dropdown (no hardcoded options)
    - Stock symbols list for intent hints
    - Sector names for quick chips
    - Market date, status, currency
    - Top movers and worst sectors for dynamic quick-action chips
    """
    registry = request.app.state.registry
    return registry.get_registry_summary()


@router.get("/registry/portfolios")
async def get_portfolio_list(request: Request) -> list:
    """List of all discovered portfolios with basic metadata."""
    registry = request.app.state.registry
    return registry.get_all_portfolios_summary()


@router.get("/registry/stocks")
async def get_stock_list(request: Request) -> list:
    """All stock symbols discovered from market_data.json."""
    registry = request.app.state.registry
    return registry.stock_symbols


@router.get("/registry/sectors")
async def get_sector_list(request: Request) -> list:
    """All sector names discovered from market_data.json."""
    registry = request.app.state.registry
    return registry.sector_names
