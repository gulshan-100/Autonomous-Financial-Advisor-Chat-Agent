"""Pydantic request/response models for FastAPI routes."""
from __future__ import annotations

import uuid
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message:      str            = Field(..., description="User's chat message")
    portfolio_id: Optional[str]  = Field(None, description="Selected portfolio ID (from registry)")
    session_id:   str            = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation memory",
    )


class PortfolioRequest(BaseModel):
    portfolio_id: str


class StockRequest(BaseModel):
    symbol: str


class SectorRequest(BaseModel):
    sector: str


class MarketMoversRequest(BaseModel):
    n: int = Field(5, ge=1, le=20, description="Number of top/bottom movers to return")
