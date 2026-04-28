"""
AgentState — shared state for the ReAct financial advisor agent.

Minimal state: conversation messages and session context only.
All financial data (portfolio, market, news, risk) is fetched on-demand
via tool calls during the agent's reasoning loop — never pre-stored in state.
"""
from __future__ import annotations

import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    # Append-only conversation history (system + human + AI + tool messages)
    messages:     Annotated[List[BaseMessage], operator.add]
    session_id:   str
    # Portfolio selected by the user in the UI (optional hint for the agent)
    portfolio_id: Optional[str]
    # LLM-as-a-Judge evaluation of the final response (populated by judge node)
    judge_result: Optional[dict]
