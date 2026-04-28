# Autonomous Financial Advisor Chat Agent

An AI-powered financial advisor that **reasons through market data** — not just reports it.  
Built with LangGraph, GPT-4o, and FastAPI. The agent autonomously decides which data tools to call, builds causal chains linking macro events to portfolio impact, and evaluates its own response quality.

## Assignment Coverage

| Phase | Requirement | Implementation |
|-------|-------------|----------------|
| **1: Market Intelligence** | Trend analysis (NIFTY/SENSEX) | `MarketAnalyzer._analyze_index_trends()` + `_compute_overall_sentiment()` |
| | Sector extraction | `DataRegistry.sector_to_stocks` + `_rank_sectors()` |
| | News classification | `NewsProcessor` — sentiment, scope (MARKET_WIDE/SECTOR/STOCK), impact level |
| **2: Portfolio Analytics** | Daily P&L | `PortfolioAnalyzer.analyze()` → `day_change_pct`, `day_change_absolute` |
| | Asset allocation | `sector_allocation` + `asset_type_allocation` breakdown |
| | Risk detection | Configurable thresholds: sector >30% warning, >60% critical, single-stock >15%, high-beta >1.2 |
| **3: Autonomous Reasoning** | Causal linking | `build_causal_chain` tool: macro → sector → stock → portfolio P&L |
| | Conflict resolution | `conflict_flag` in news + `identify_conflict_signals()` |
| | Prioritization | News ranked by `impact_level × |sentiment_score|`; agent only surfaces high-impact signals |
| | Agent autonomy | True ReAct loop: LLM reasons → calls tools → reasons → answers. 11 tools, `think` for planning |
| **4: Observability** | Tracing | Langfuse integration (`agent/tracing.py`) — prompts, responses, token usage |
| | Self-evaluation | LLM-as-a-Judge (`agent/nodes/judge.py`) — 6-dimension quality scoring |
| | Confidence score | Every response ends with `**Confidence: HIGH/MEDIUM/LOW**` + justification |
| **Architecture** | Modularity | 3 layers: `data_layer/` → `agent/` → `app/` |
| | Latency | Simple queries: 1 tool call. Deep analysis: parallel multi-tool calls |
| | Type hints | Full type annotations across all modules |
| | Missing data | Graceful `try/except` in every tool with informative error messages |

## Architecture

```
financial_advisor_agent/
├── config.py                    # Pydantic Settings — all config from .env
├── run.py                       # Startup script
├── requirements.txt
├── .env.example                 # Template — copy to .env
│
├── data_layer/                  # Phase 1 + 2: Market Intelligence + Portfolio Analytics
│   ├── loader.py                # Loads 6 JSON data files at startup
│   ├── registry.py              # DataRegistry: auto-discovers all entities (zero hardcoding)
│   ├── portfolio_analyzer.py    # P&L, sector allocation, risk flags, causal summaries
│   ├── market_analyzer.py       # Index trends, sector ranking, FII/DII, macro themes
│   └── news_processor.py        # Sentiment aggregation, causal chain builder, conflict detection
│
├── agent/                       # Phase 3 + 4: Autonomous Reasoning + Evaluation
│   ├── state.py                 # Minimal AgentState: messages + session_id + portfolio_id
│   ├── graph.py                 # LangGraph ReAct agent (financial_advisor ↔ tool_executor → judge)
│   ├── prompts/templates.py     # SYSTEM_PROMPT (planning protocol) + JUDGE_PROMPT (scoring rubric)
│   ├── tracing.py               # Langfuse integration (optional — auto-disabled if keys missing)
│   ├── tools/
│   │   └── financial_tools.py   # 11 @tool functions: think, portfolios, market, stocks, news, etc.
│   └── nodes/
│       └── judge.py             # LLM-as-a-Judge: 6-dimension quality scoring
│
├── app/                         # FastAPI server + SSE streaming
│   ├── main.py                  # Lifespan startup: data → registry → LLM → graph
│   ├── models/request.py        # Pydantic request schemas
│   └── routes/
│       ├── chat.py              # POST /api/chat — SSE stream with tool_call + judge events
│       ├── registry.py          # GET /api/registry — entity discovery for frontend
│       ├── portfolio.py         # GET /api/portfolio/{id} — REST portfolio analysis
│       └── market.py            # GET /api/market/* — REST market data
│
└── frontend/                    # Single-page chat UI
    ├── index.html               # Shell only — all content built by JS from API data
    ├── style.css                # Dark glassmorphic theme + judge panel styles
    └── script.js                # Dynamic UI: reasoning tracker, streaming chat, judge score panel
```

## ReAct Agent Flow

```
User Query
    ↓
┌─────────────────────────────────────────────┐
│  financial_advisor (LLM with 11 tools)      │
│                                             │
│  1. think("User wants to know why portfolio │
│     is down. I need portfolio data, then    │
│     causal chain, then news.")              │
│                                             │
│  2. get_portfolio_analysis("P001")          │
│  3. build_causal_chain(portfolio_id="P001") │
│  4. search_news(sector="BANKING")           │
│                                             │
│  5. Final answer with causal chain +        │
│     confidence score                        │
└───────────────┬─────────────────────────────┘
                ↓
┌─────────────────────────────────────────────┐
│  response_judge (LLM-as-a-Judge)            │
│  Scores: Factual Grounding, Causal          │
│  Reasoning, Completeness, Actionability,    │
│  Risk Awareness, Conciseness → verdict      │
└───────────────┬─────────────────────────────┘
                ↓
              END → SSE stream to frontend
```

The LLM decides **which tools** to call, **how many**, and **in what order** — purely based on the question. Simple questions get 1 tool call; deep analysis questions get 4–6 tool calls with multi-hop reasoning.

## Tools (11)

| Tool | Purpose |
|------|---------|
| `think` | Chain-of-thought scratchpad — plan before acting |
| `list_portfolios` | Discover available portfolio IDs and users |
| `get_portfolio_analysis` | Full holdings, P&L, sector allocation, risk metrics |
| `get_portfolio_risk` | Deep risk: beta, concentration, bearish sector alerts |
| `get_market_overview` | All indices, sector rankings, FII/DII, breadth |
| `get_stock_details` | Price, change, sector, news for a specific stock |
| `get_sector_analysis` | Sector performance, constituents, macro drivers |
| `search_news` | Filter by stock/sector; sentiment, impact, causal factors |
| `get_top_movers` | Top N gainers and losers |
| `get_mutual_fund_details` | NAV, returns, AUM, expense ratio |
| `build_causal_chain` | Macro → sector → stock → portfolio P&L causal chain |

## SSE Events

The frontend receives these Server-Sent Events during streaming:

| Event | When |
|-------|------|
| `node_start` | Agent starts reasoning |
| `tool_call` | Agent decided to invoke a specific tool (e.g. "📊 Fetching market overview") |
| `tool_done` | Tool execution completed |
| `token` | Streaming token from the final LLM response |
| `judge_result` | Quality score from the LLM-as-a-Judge (6 dimensions + verdict) |
| `done` | Full run complete |

## Setup

1. **Install dependencies:**
   ```powershell
   cd "Agent Assignment\financial_advisor_agent"
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```powershell
   copy .env.example .env
   # Edit .env and set OPENAI_API_KEY=sk-your-key-here
   ```

3. **Run the server:**
   ```powershell
   python run.py
   ```

4. **Open in browser:** http://localhost:8000

## Key Design Principles

- **Zero hardcoding**: All portfolio IDs, stock symbols, sector names, news IDs discovered at runtime from JSON
- **True autonomy**: LLM decides which tools to call — no hardcoded pipeline or routing logic
- **Causal reasoning**: `build_causal_chain` traces macro events through sectors to portfolio P&L
- **Self-evaluation**: LLM-as-a-Judge scores every response on 6 quality dimensions
- **Confidence scoring**: Every response includes a confidence assessment with justification
- **Streaming**: SSE delivers tool-call progress + token-by-token response to the frontend
- **Configurable thresholds**: All risk limits, model params, tracing keys in `.env`
- **Per-session memory**: LangGraph `MemorySaver` maintains conversation history per `session_id`
- **Langfuse tracing**: Optional but full — tracks every LLM call, token usage, latency per node
