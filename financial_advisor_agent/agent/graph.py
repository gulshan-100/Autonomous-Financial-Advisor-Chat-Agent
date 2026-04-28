"""
LangGraph ReAct Agent — Autonomous Financial Advisor.

Architecture: true tool-calling ReAct loop.
  START
    → financial_advisor  (LLM reasons, plans, calls tools)
    → tool_executor      (executes the tools the LLM chose)  [repeated if needed]
    → financial_advisor  (LLM sees results, reasons further or gives final answer)
    → END

The LLM drives ALL decisions:
  - Which tools to call (not hardcoded)
  - In what order (not hardcoded)
  - When it has enough data (not hardcoded)
  - What to include in the final answer (not hardcoded)

No fixed pipelines. No pre-fetching. No data stuffed into prompts before the LLM
decides it needs it. Every data point in the final response comes from a tool call.
"""
from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from agent.tools.financial_tools import build_financial_tools
from agent.nodes.judge import make_judge
from agent.prompts.templates import SYSTEM_PROMPT
from data_layer.registry import DataRegistry
from config import Settings

logger = logging.getLogger(__name__)

# ── Node name constants ────────────────────────────────────────────────────────
NODE_AGENT = "financial_advisor"
NODE_TOOLS = "tool_executor"
NODE_JUDGE = "response_judge"

ALL_NODE_NAMES = [NODE_AGENT, NODE_TOOLS, NODE_JUDGE]


def build_graph(
    registry: DataRegistry,
    settings: Settings,
    llm: ChatOpenAI,
) -> object:
    """
    Build the ReAct financial advisor agent graph.

    The LLM is bound with 10 financial data tools. For every user query:
      1. LLM reads the question and thinks about what data it needs.
      2. LLM calls one or more tools (its own decision — no hardcoded routing).
      3. Tools execute against the data layer and return real data.
      4. LLM reads the tool results and decides: call more tools, or answer?
      5. When the LLM has enough data, it produces the final response.
    """
    logger.info("Building ReAct financial advisor agent...")

    # Build all tools — each wraps a data-layer call
    tools = build_financial_tools(registry, settings)
    logger.info("Registered tools: %s", [t.name for t in tools])

    # Bind tools to LLM + tag all LLM calls for SSE streaming capture
    llm_with_tools = llm.bind_tools(tools).with_config(tags=["streaming_response"])

    # ── Agent node: the reasoning + planning core ──────────────────────────────

    def agent_node(state: AgentState) -> dict:
        """
        Core ReAct node: the LLM reasons about what the user needs, decides
        which tool(s) to call, and produces either tool_calls or a final answer.

        On each invocation the LLM sees:
          - A dynamic system message (tool list + portfolio hint)
          - The full conversation history (human messages + tool results so far)
        """
        portfolio_id = state.get("portfolio_id")

        system_content = SYSTEM_PROMPT.format(
            portfolio_hint=(
                f"\nThe user has pre-selected portfolio ID: '{portfolio_id}'. "
                f"When they refer to 'my portfolio' or 'my investments', "
                f"use get_portfolio_analysis('{portfolio_id}') automatically."
                if portfolio_id else
                "\nNo portfolio is currently selected by the user. "
                "If they ask about 'my portfolio', call list_portfolios() first."
            ),
            available_stocks=", ".join(registry.stock_symbols[:20]) + " ...(and more)",
            available_sectors=", ".join(registry.sector_names),
            portfolio_ids=", ".join(registry.portfolio_ids),
        )

        messages = [SystemMessage(content=system_content)] + list(state.get("messages", []))
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # ── Routing: tools or end? ─────────────────────────────────────────────────

    def should_continue(state: AgentState) -> Literal["tools", "judge"]:
        """
        If the LLM's last message contains tool_calls → execute those tools.
        If it produced plain text (no tool_calls) → final answer → send to judge.
        """
        last_msg = state["messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            logger.info(
                "Agent calling tools: %s",
                [tc["name"] for tc in last_msg.tool_calls],
            )
            return "tools"
        logger.info("Agent produced final answer — routing to judge.")
        return "judge"

    # ── ToolNode: executes whichever tools the LLM chose ──────────────────────
    tool_node = ToolNode(tools)

    # ── Judge node: LLM-as-a-Judge scores the final response ──────────────────
    judge_node = make_judge(llm)

    # ── Assemble graph ─────────────────────────────────────────────────────────
    graph = StateGraph(AgentState)
    graph.add_node(NODE_AGENT, agent_node)
    graph.add_node(NODE_TOOLS, tool_node)
    graph.add_node(NODE_JUDGE, judge_node)
    graph.set_entry_point(NODE_AGENT)

    graph.add_conditional_edges(
        NODE_AGENT,
        should_continue,
        {"tools": NODE_TOOLS, "judge": NODE_JUDGE},
    )
    graph.add_edge(NODE_TOOLS, NODE_AGENT)
    graph.add_edge(NODE_JUDGE, END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    logger.info("ReAct agent compiled with %d tools.", len(tools))
    return compiled


def create_initial_state(
    user_message: str,
    portfolio_id: str | None,
    session_id: str,
    registry: DataRegistry,
) -> AgentState:
    """Create the initial AgentState for a new chat turn."""
    return {
        "messages":    [HumanMessage(content=user_message)],
        "session_id":  session_id,
        "portfolio_id": portfolio_id,
    }
