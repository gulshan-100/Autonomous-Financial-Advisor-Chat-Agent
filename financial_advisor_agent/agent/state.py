"""
LangGraph AgentState — the shared state passed between all reasoning nodes.

Every field is populated dynamically at runtime. No field has a hardcoded
default value tied to a specific portfolio, stock, or sector.
"""
from __future__ import annotations

import operator
from typing import Annotated, Dict, List, Optional, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # ── Conversation ──────────────────────────────────────────────────────────
    messages:    Annotated[List[BaseMessage], operator.add]  # append-only message history
    session_id:  str
    user_query:  str                  # raw user query text

    # ── Intent classification (populated by intent_classifier node) ───────────
    intent:             str           # portfolio_query | market_query | stock_query |
                                      # news_query | advice_request | general_query
    portfolio_id:       Optional[str] # None until extracted from query or preset by UI
    symbols_mentioned:  List[str]     # stock tickers extracted from query (dynamic)
    sectors_mentioned:  List[str]     # sector names extracted from query (dynamic)
    urgency:            str           # HIGH | MEDIUM | LOW
    query_complexity:   str           # SIMPLE | MODERATE | DEEP

    # ── Gathered context (populated by context_gatherer node) ─────────────────
    portfolio_context:  Optional[dict]       # full PortfolioAnalyzer.analyze() result
    market_context:     Optional[dict]       # full MarketAnalyzer.get_full_snapshot()
    stock_context:      Optional[Dict[str, dict]]  # symbol → stock detail dict
    sector_context:     Optional[Dict[str, dict]]  # sector → sector detail dict
    news_context:       Optional[List[dict]] # filtered, ranked news articles
    mf_context:         Optional[Dict[str, dict]]  # scheme_code → MF detail

    # ── Risk analysis (populated by risk_analyzer node) ───────────────────────
    risk_flags:          List[str]    # list of risk alert strings
    concentration_risk:  bool         # True if any CRITICAL/WARNING flags
    portfolio_beta:      Optional[float]
    conflict_signals:    List[dict]   # news with conflict_flag=True

    # ── Causal chain (populated by news_reasoner node) ────────────────────────
    causal_chain:       Optional[str] # human-readable causal narrative
    causal_chain_data:  Optional[dict] # structured causal chain dict

    # ── Market summary (populated by market_analyzer node) ───────────────────
    market_summary:     Optional[str] # market narrative text

    # ── Final advice (populated by advisor_synthesizer node) ─────────────────
    advice_structured:  Optional[dict]  # structured JSON advice
    recommendations:    List[str]
    sources_used:       List[str]       # news IDs cited
    confidence_level:   str            # HIGH | MEDIUM | LOW

    # ── Response (populated by response_formatter node) ───────────────────────
    final_response:     Optional[str]  # final markdown response text

    # ── Registry snapshot (injected at graph entry) ───────────────────────────
    # Allows all nodes to know what entities exist without importing registry
    available_portfolios: List[str]
    available_stocks:     List[str]
    available_sectors:    List[str]
    registry_summary:     Optional[dict]
