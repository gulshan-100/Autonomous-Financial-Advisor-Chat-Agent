"""
FastAPI main application entry point.

Startup sequence:
  1. Load all 6 JSON files via DataLoader
  2. Build DataRegistry (auto-discovers all entities)
  3. Initialize LangChain ChatOpenAI LLM
  4. Compile LangGraph StateGraph
  5. Store everything in app.state for route access

All values come from Settings/.env — no hardcoding anywhere.
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from data_layer.loader   import DataLoader
from data_layer.registry import DataRegistry
from agent.graph         import build_graph
from app.routes          import chat, market, portfolio, registry

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Lifespan context manager ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server startup: initialise data layer, LLM, and LangGraph.
    Server shutdown: nothing to clean up (in-memory only).
    """
    # Resolve data_dir relative to THIS file's location
    base_dir = Path(__file__).parent.parent  # financial_advisor_agent/
    data_dir = (base_dir / settings.data_dir).resolve()

    logger.info("=" * 60)
    logger.info("Starting Autonomous Financial Advisor Agent")
    logger.info("Data directory: %s", data_dir)
    logger.info("LLM model: %s", settings.openai_model)
    logger.info("=" * 60)

    # 1. Load raw JSON data
    loader = DataLoader(str(data_dir))

    # 2. Build dynamic entity registry
    reg = DataRegistry(loader)

    # 3. Initialise OpenAI LLM
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens,
            api_key=settings.openai_api_key,
            streaming=True,
        )
        logger.info("OpenAI LLM initialised: %s", settings.openai_model)
    except Exception as exc:
        logger.error("Failed to initialise OpenAI LLM: %s", exc)
        raise

    # 4. Compile LangGraph
    graph = build_graph(registry=reg, settings=settings, llm=llm)

    # 5. Store everything in app.state — used by all route handlers
    app.state.loader   = loader
    app.state.registry = reg
    app.state.llm      = llm
    app.state.graph    = graph
    app.state.settings = settings

    if settings.langfuse_secret_key and settings.langfuse_public_key:
        logger.info("✅ Langfuse tracing ENABLED → %s", settings.langfuse_base_url)
    else:
        logger.warning("⚠️  Langfuse tracing DISABLED — LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY not set in .env")

    logger.info("✅ Server ready. Open http://localhost:%d", settings.port)
    logger.info("   Portfolio IDs discovered: %s", reg.portfolio_ids)
    logger.info("   Stocks: %d | Sectors: %d | News: %d | MFs: %d",
                len(reg.stock_symbols), len(reg.sector_names),
                len(reg.news_ids), len(reg.mutual_fund_codes))

    yield  # ← server is running

    logger.info("Server shutting down.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomous Financial Advisor Agent",
    description="LangGraph-powered AI financial advisor with dynamic data discovery",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ───────────────────────────────────────────────────────────────

app.include_router(chat.router)
app.include_router(registry.router)
app.include_router(portfolio.router)
app.include_router(market.router)

# ── Static frontend ───────────────────────────────────────────────────────────

_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="static")
else:
    logger.warning("Frontend directory not found: %s", _frontend_dir)


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=[str(Path(__file__).parent.parent)],
    )
