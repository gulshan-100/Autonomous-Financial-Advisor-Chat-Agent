"""
LLM-as-a-Judge node.

After the ReAct agent produces its final answer, this node runs a separate
LLM evaluation pass to score the response across six quality dimensions:
  - Factual grounding  (are real data values cited?)
  - Causal reasoning   (does it explain WHY, not just WHAT?)
  - Completeness       (did it answer everything asked?)
  - Actionability      (are recommendations concrete?)
  - Risk awareness     (were critical risks surfaced?)
  - Conciseness        (appropriate length for the question?)

The judge result is stored in AgentState["judge_result"] and forwarded to
the frontend via a "judge_result" SSE event.
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agent.prompts.templates import JUDGE_PROMPT
from agent.state import AgentState

logger = logging.getLogger(__name__)


def make_judge(llm: ChatOpenAI):
    """
    Factory: returns the judge node function.
    Uses a dedicated LLM call (no tool-binding, no streaming tag)
    so its output never leaks into the frontend token stream.
    """
    # Low temperature for deterministic scoring; no tools bound
    judge_llm = llm.with_config({"tags": ["judge"], "temperature": 0.1})

    def judge_node(state: AgentState) -> dict:
        """
        Score the agent's final response using a rubric-based LLM evaluation.
        Reads the original human query and the last AIMessage with content.
        """
        messages = list(state.get("messages", []))

        # Extract original user query (first HumanMessage)
        user_query = next(
            (m.content for m in messages if isinstance(m, HumanMessage)),
            "",
        )

        # Extract the final AI answer (last AIMessage that has content, not tool calls)
        ai_response = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                ai_response = m.content
                break

        if not ai_response or not user_query:
            logger.warning("Judge: missing user_query or ai_response — skipping.")
            return {"judge_result": None}

        prompt = JUDGE_PROMPT.format(
            user_query=user_query,
            ai_response=ai_response,
        )

        try:
            response = judge_llm.invoke([SystemMessage(content=prompt)])
            raw = response.content.strip()

            # Extract JSON robustly (LLM may wrap it in markdown fences)
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in judge output: {raw[:200]}")

            result = json.loads(match.group())

            # Validate expected keys are present
            scores = result.get("scores", {})
            required = {"factual_grounding", "causal_reasoning", "completeness",
                        "actionability", "risk_awareness", "conciseness"}
            if not required.issubset(scores.keys()):
                raise ValueError(f"Judge scores incomplete: {list(scores.keys())}")

            logger.info(
                "Judge verdict: %s | overall: %s/10",
                result.get("verdict"),
                result.get("overall"),
            )
            return {"judge_result": result}

        except Exception as exc:
            logger.error("Judge node failed: %s", exc)
            return {"judge_result": {"error": str(exc), "verdict": "ERROR"}}

    return judge_node
