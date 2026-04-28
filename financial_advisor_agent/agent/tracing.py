"""
Langfuse tracing integration for the Financial Advisor Agent.

Follows best practices from https://github.com/langfuse/skills:
  - Uses the official LangChain integration: langfuse.langchain.CallbackHandler
  - Single shared handler instance (session/user context goes in config metadata)
  - Flush via get_client().flush() — the singleton pattern recommended by Langfuse

Usage in a route handler:
    handler = get_langfuse_handler()
    if handler:
        config["callbacks"] = [handler]
        config["metadata"]  = {
            "langfuse_session_id": session_id,
            "langfuse_user_id":    user_id,
            "langfuse_tags":       ["financial-advisor"],
        }
    # after streaming completes:
    flush_langfuse()
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import guard — langfuse is an optional dependency
try:
    # Correct import per latest Langfuse docs (langfuse.langchain, NOT langfuse.callback)
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
    from langfuse import get_client as _get_langfuse_client
    _LANGFUSE_AVAILABLE = True
except ImportError:
    _LANGFUSE_AVAILABLE = False
    logger.warning(
        "langfuse package not installed — tracing disabled. Run: pip install langfuse"
    )

# Module-level singleton — one handler shared across all requests.
# Session/user context is passed per-request via config["metadata"], not here.
_handler: Optional[object] = None
_handler_initialised = False


def get_langfuse_handler() -> Optional[object]:
    """
    Return the shared Langfuse CallbackHandler, or None if not configured.

    The handler is initialised once and reused. Trace attributes
    (session_id, user_id, tags) must be set per-request via LangGraph config:

        config["metadata"] = {
            "langfuse_session_id": req.session_id,
            "langfuse_tags":       ["financial-advisor"],
        }

    Captures automatically (via LangChain integration):
      - All LLM prompts and completions
      - Token usage (prompt / completion / total) → enables cost calculation
      - Node-level spans with latency
      - Model name → enables model comparison
    """
    global _handler, _handler_initialised

    if not _LANGFUSE_AVAILABLE:
        return None

    if _handler_initialised:
        return _handler

    from config import settings  # import after env is loaded to avoid credential miss

    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.debug("Langfuse keys not configured — tracing disabled.")
        _handler_initialised = True
        return None

    try:
        # pydantic-settings loads .env into the Settings object but does NOT
        # set os.environ. The new langfuse.langchain.CallbackHandler() reads
        # credentials exclusively from os.environ, so we must inject them.
        os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
        os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
        os.environ["LANGFUSE_HOST"]       = settings.langfuse_base_url

        _handler = LangfuseCallbackHandler()
        logger.info("Langfuse CallbackHandler initialised → %s", settings.langfuse_base_url)
    except Exception as exc:
        logger.warning("Failed to initialise Langfuse handler: %s — tracing skipped.", exc)
        _handler = None

    _handler_initialised = True
    return _handler


def flush_langfuse() -> None:
    """
    Flush all pending Langfuse trace events via the singleton client.

    Must be called after every graph run in async/streaming contexts to ensure
    batched events are sent before the SSE connection closes.
    Ref: https://langfuse.com/docs/integrations/langchain#queuing-and-flushing
    """
    if not _LANGFUSE_AVAILABLE or _handler is None:
        return  # skip flush if handler never initialised — avoids credentialless client
    try:
        _get_langfuse_client().flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed (non-critical): %s", exc)
