"""
LangGraph node: Intent Classifier

Uses GPT-4o with structured JSON output to classify the user's query intent
and extract mentioned entities (portfolios, stocks, sectors).
All valid entities are injected dynamically from the DataRegistry — not hardcoded.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI

from agent.state import AgentState
from agent.prompts.templates import INTENT_PROMPT
from agent.utils import sanitize_for_llm, safe_json_dumps
from data_layer.registry import DataRegistry

logger = logging.getLogger(__name__)


def make_intent_classifier(llm: ChatOpenAI, registry: DataRegistry):
    """
    Factory function returns a LangGraph-compatible node function
    with registry and llm captured via closure (dependency injection).
    """

    def intent_classifier(state: AgentState) -> dict:
        """
        Node: Classify user intent and extract entities.
        Injects live registry entity lists into the GPT-4o prompt dynamically.
        """
        raw_message = state.get("user_query", "")
        if not raw_message and state.get("messages"):
            # Fall back to last human message
            from langchain_core.messages import HumanMessage
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    raw_message = msg.content
                    break

        # Sanitize: strips surrogate chars (from emoji in browser input)
        user_message = sanitize_for_llm(raw_message)

        # Build entity lists dynamically from registry — no hardcoded lists
        available_portfolios = registry.portfolio_ids
        available_stocks     = registry.stock_symbols
        available_sectors    = registry.sector_names
        user_portfolio_map   = registry.user_map

        prompt = INTENT_PROMPT.format(
            available_portfolios = safe_json_dumps(available_portfolios),
            user_portfolio_map   = safe_json_dumps(user_portfolio_map),
            available_stocks     = safe_json_dumps(available_stocks),
            available_sectors    = safe_json_dumps(available_sectors),
            user_message         = user_message,
        )
        prompt = sanitize_for_llm(prompt)

        logger.info("intent_classifier: classifying query '%s'", user_message[:80])

        try:
            response = llm.invoke(
                prompt,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.content)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Intent classification failed: %s — defaulting to general_query", exc)
            result = {
                "intent": "general_query",
                "portfolio_id": state.get("portfolio_id"),
                "symbols_mentioned": [],
                "sectors_mentioned": [],
                "urgency": "MEDIUM",
            }

        # Validate extracted entities against registry (never trust LLM blindly)
        validated_symbols = [
            s for s in result.get("symbols_mentioned", [])
            if s in registry.stock_symbols
        ]
        validated_sectors = [
            s for s in result.get("sectors_mentioned", [])
            if s in registry.sector_names
        ]
        validated_portfolio = (
            result.get("portfolio_id")
            if result.get("portfolio_id") in registry.portfolio_ids
            else state.get("portfolio_id")  # keep existing if set by UI
        )

        logger.info(
            "intent_classifier → intent=%s, portfolio=%s, stocks=%s, sectors=%s",
            result.get("intent"),
            validated_portfolio,
            validated_symbols,
            validated_sectors,
        )

        return {
            "intent":             result.get("intent", "general_query"),
            "portfolio_id":       validated_portfolio,
            "symbols_mentioned":  validated_symbols,
            "sectors_mentioned":  validated_sectors,
            "urgency":            result.get("urgency", "MEDIUM"),
            "query_complexity":   result.get("query_complexity", "MODERATE"),
            "user_query":         user_message,   # already sanitized
            # Inject registry entity lists into state for downstream nodes
            "available_portfolios": available_portfolios,
            "available_stocks":     available_stocks,
            "available_sectors":    available_sectors,
        }

    return intent_classifier
