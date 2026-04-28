# Autonomous Financial Advisor Chat Agent

> **An AI agent that doesn't just report financial data — it reasons through it.**
> Built on GPT-4o + LangGraph with a true ReAct loop, 11 financial tools, a causal chain engine, LLM-as-a-Judge self-evaluation, and a real-time streaming chat UI.

---

## What This Does

Ask it anything finance-related and it will autonomously decide which data to fetch, reason through causal links between macro events and your portfolio, flag risks you didn't ask about, and score the quality of its own answer. Example output:

> *"Your portfolio fell **-2.73%** (₹-57,390) today primarily because **BANKING** — where you have a **critical 91.58% concentration** — sold off after the RBI rate hike news. Your largest holding **HDFC Bank (22.62% weight)** dropped **-3.1%**, contributing roughly **₹-35,000** to your loss. Note: there's a conflict signal — Kotak Mahindra's strong Q4 results are positive for banking but the sector is still down due to macro pressure overriding stock-specific news. **Confidence: HIGH** — derived from portfolio data, causal chain analysis, and 4 corroborating news sources."*

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | GPT-4o (via `langchain-openai`) |
| Agent Framework | LangGraph `StateGraph` — ReAct loop |
| Backend | FastAPI + Server-Sent Events (SSE) |
| Tracing | Langfuse (optional) |
| Config | Pydantic `BaseSettings` + `.env` |
| Frontend | Vanilla JS + CSS — fully dynamic, no hardcoded data |

---

## Project Structure

```
financial_advisor_agent/
│
├── config.py                      # All settings from .env — zero hardcoding
├── run.py                         # python run.py → starts server
├── requirements.txt
├── .env.example                   # Copy to .env, add OPENAI_API_KEY
│
├── data_layer/                    ← Market Intelligence + Portfolio Analytics
│   ├── loader.py                  # Loads 6 JSON files once at startup
│   ├── registry.py                # DataRegistry: auto-discovers all entities
│   ├── portfolio_analyzer.py      # P&L, allocation, risk flags, causal summary
│   ├── market_analyzer.py         # Index trends, sector ranking, FII/DII, breadth
│   └── news_processor.py          # Sentiment, causal chains, conflict detection
│
├── agent/                         ← Autonomous Reasoning + Self-Evaluation
│   ├── state.py                   # AgentState TypedDict (messages, session_id, portfolio_id, judge_result)
│   ├── graph.py                   # LangGraph: financial_advisor ↔ tool_executor → response_judge → END
│   ├── tracing.py                 # Langfuse callback handler (auto-disabled if keys missing)
│   ├── prompts/
│   │   └── templates.py           # SYSTEM_PROMPT (planning protocol) + JUDGE_PROMPT (6-dim rubric)
│   ├── tools/
│   │   └── financial_tools.py     # 11 @tool functions — the only data access path for the LLM
│   └── nodes/
│       └── judge.py               # LLM-as-a-Judge: isolated scoring call, temperature=0.1
│
├── app/                           ← FastAPI Server
│   ├── main.py                    # Lifespan: loader → registry → LLM → graph → app.state
│   ├── models/request.py          # ChatRequest Pydantic model
│   └── routes/
│       ├── chat.py                # POST /api/chat — SSE streaming
│       ├── registry.py            # GET /api/registry — entity discovery
│       ├── portfolio.py           # GET /api/portfolio/{id}
│       └── market.py              # GET /api/market/*
│
└── frontend/
    ├── index.html                 # Pure HTML shell — zero data in markup
    ├── style.css                  # Dark glassmorphic UI + judge score panel styles
    └── script.js                  # Fetches /api/registry on load, builds all UI dynamically
```

---

## Agent Architecture — The ReAct Loop

```
                        User Query
                            │
                            ▼
              ┌─────────────────────────────┐
              │    financial_advisor        │  ← GPT-4o with 11 tools bound
              │                             │
              │  Per turn the LLM sees:     │
              │  • Dynamic SYSTEM_PROMPT    │
              │  • Full conversation history│
              │  • All prior tool results   │
              └──────────┬──────────────────┘
                         │
              has tool_calls?
             /             \
           YES              NO  ──────────────────────────────┐
            │                                                  │
            ▼                                                  ▼
  ┌──────────────────┐                          ┌─────────────────────────┐
  │  tool_executor   │                          │    response_judge       │
  │  (ToolNode)      │                          │                         │
  │  Runs tools the  │                          │  Separate LLM call      │
  │  LLM chose. Returns│                        │  temperature=0.1        │
  │  results back to │                          │  No tool binding        │
  │  financial_adv.  │                          │  Scores 6 dimensions    │
  └──────────┬───────┘                          └───────────┬─────────────┘
             │                                              │
             └──────────── loops back ─────────►           ▼
                                                          END
                                             (judge_result → SSE → UI)
```

**Key principle:** The LLM drives every decision. No hardcoded routing, no pre-fetched data stuffed into prompts. The system prompt instructs the agent *what tools exist* — the agent decides *which ones to use*.

---

## The 11 Tools

All tools are `@tool`-decorated functions in `agent/tools/financial_tools.py`. They are created via a factory `build_financial_tools(registry, settings)` that injects the data layer via closure — the LLM never touches the data layer directly.

| # | Tool | What It Returns |
|---|------|-----------------|
| 0 | `think(thought)` | Chain-of-thought scratchpad. LLM writes its plan here before calling data tools. Returns `"Reasoning recorded."` |
| 1 | `list_portfolios()` | All portfolio IDs, user names, types, risk profiles, current values, gain/loss % |
| 2 | `get_portfolio_analysis(portfolio_id)` | Full holdings (stocks + MFs), daily P&L (abs + %), sector allocation, asset-type allocation, risk metrics (beta, Sharpe, volatility, max drawdown), risk flags, causal summary, top mover |
| 3 | `get_portfolio_risk(portfolio_id)` | Risk flags cross-referenced against live market — e.g. *"BANKING: 91.6% allocation — currently -2.33% BEARISH today"*. Includes beta, Sharpe, conflict signals |
| 4 | `get_market_overview()` | All NSE indices with values/changes, 10 sectors ranked by performance, top 5 gainers/losers, FII/DII flows, advance/decline breadth, macro themes |
| 5 | `get_stock_details(symbol)` | Price, % change, 52-week high/low, volume, sector membership + sector performance, related news, portfolios holding this stock |
| 6 | `get_sector_analysis(sector)` | Sector day change + sentiment, all constituent stocks, weekly trend, macro correlation factors, related news |
| 7 | `search_news(symbol, sector, top_n)` | News with: headline, sentiment, sentiment_score, impact level, scope, affected stocks/sectors, causal factors, `conflict_flag` + conflict explanation |
| 8 | `get_top_movers(n)` | Top N gainers and losers with symbol, price, % change, sector |
| 9 | `get_mutual_fund_details(scheme_code)` | Fund name, category, NAV, 1Y/3Y/5Y returns, AUM, expense ratio, fund manager, top holdings |
| 10 | `build_causal_chain(portfolio_id, symbol)` | Traces: root news event → affected sector → affected portfolio stocks → estimated P&L contribution. Returns `causal_chains[]`, `conflict_flags[]`, narrative string |

### How the Agent Plans (SYSTEM_PROMPT Protocol)

The system prompt enforces a **4-step planning protocol**:

```
1. THINK   → Call think() to write the plan before touching data
2. GATHER  → Call the identified data tools
3. REFLECT → If data reveals surprise (conflict signal, hidden risk) → think() again
4. ANSWER  → Write response using ONLY numbers from tool results
```

With built-in multi-hop examples:
```
"Why is my portfolio down?"
  → think → get_portfolio_analysis → build_causal_chain → search_news → answer

"Should I buy more INFY?"
  → think → get_stock_details(INFY) → get_sector_analysis(IT) → search_news(INFY) → get_market_overview → answer

"Which sector is best today?"
  → think → get_market_overview → answer   ← single hop is sufficient
```

---

## Mock Dataset (6 JSON Files)

### Market Conditions (as of April 21, 2026)
- **NIFTY 50**: bearish, **-1.02%**
- **SENSEX**: bearish, **-0.98%**
- **NIFTY BANK**: **-2.33%** — worst performer
- **NIFTY IT**: **+1.22%** — only index in green (sector divergence test case)
- **10 Sectors**: BANKING (bearish), IT (bullish), ENERGY, PHARMA, FMCG, INFRASTRUCTURE, AUTO, METALS, REALTY, CONSUMER_DURABLES
- **FII**: net sellers | **DII**: net buyers
- **Advance/Decline**: 32% advancing — broad-based weakness

### News Corpus (edge cases built in)
| Scope | Examples |
|---|---|
| `MARKET_WIDE` | RBI interest rate hike — HIGH impact, NEGATIVE sentiment |
| `SECTOR_SPECIFIC` | US-India trade deal discussions — IT sector, POSITIVE |
| `STOCK_SPECIFIC` | USFDA approval for Sun Pharma — POSITIVE with `conflict_flag=true` (stock still falling due to sector drag) |
| `STOCK_SPECIFIC` | Infosys $1.5B cloud deal — POSITIVE, HIGH impact |

The `conflict_flag` articles are specifically designed to test the agent's conflict resolution logic.

### Portfolio Profiles

**Portfolio 1 — Rahul Sharma** (`PORTFOLIO_001`) — Diversified Growth
- Current value: ₹28,62,785 | Day P&L: **-0.44%** (-₹12,785)
- 38% stocks / 62% mutual funds | Max stock weight: 7.17% (TCS)
- No concentration risk | Sectors: BANKING, IT, ENERGY, PHARMA, FMCG, INFRA, AUTO

**Portfolio 2 — Priya Patel** (`PORTFOLIO_002`) — Sector-Concentrated ⚠️
- Current value: ₹20,41,610 | Day P&L: **-2.73%** (-₹57,390)
- 91% stocks / 9% mutual funds | Max stock weight: 22.62% (HDFC Bank)
- **CRITICAL concentration: 91.58% in BANKING + Financial Services**
- Built to demonstrate extreme risk detection and causal chain: RBI news → Banking sector → HDFC Bank → portfolio loss

**Portfolio 3 — Arun Krishnamurthy** (`PORTFOLIO_003`) — Conservative Defensive
- Current value: ₹43,95,242 | Day P&L: **-0.04%** (-₹1,758)
- 21% stocks / 79% mutual funds (34% in debt funds)
- No concentration risk | Max stock weight: 5.19% (ITC)

---

## Phase 1 — Market Intelligence Layer

### Index Trend Analysis
`MarketAnalyzer._analyze_index_trends()` reads every index in `market_data.json` and computes trend direction. `_compute_overall_sentiment()` aggregates across all indices → outputs `BULLISH / BEARISH / NEUTRAL`.

### Sector Extraction
`DataRegistry._build_sector_to_stocks()` scans every stock's `"sector"` field at startup — no sector names are hardcoded. `_rank_sectors()` returns all sectors sorted by `change_percent`. Adding a new sector to the JSON file makes it automatically discoverable.

### News Classification
`NewsProcessor` classifies articles by:
- **Sentiment**: `POSITIVE / NEGATIVE / NEUTRAL / MIXED` with score `-1.0` to `+1.0`
- **Scope**: `MARKET_WIDE | SECTOR_SPECIFIC | STOCK_SPECIFIC`
- **Impact**: `HIGH / MEDIUM / LOW` (weighted: HIGH=3, MEDIUM=2, LOW=1)
- **`conflict_flag`**: `true` when price action contradicts news sentiment — triggers conflict resolution in the agent

---

## Phase 2 — Portfolio Analytics Engine

`PortfolioAnalyzer.analyze(portfolio_id)` enriches raw holdings with live market data:

```python
# For each stock holding:
{
  "symbol":         "HDFCBANK",
  "market_price":   1673.45,         # from registry
  "day_change_pct": -3.1,            # live
  "beta":           1.15,            # from market_data
  "week_52_high":   1794.0,
  "trend":          "DOWNTREND",     # from historical_data
  "risk_level":     "HIGH",          # computed: beta × weight
  "related_news":   [...]            # top 3 news for this stock
}
```

**Risk flags** are computed against configurable thresholds (all in `.env`):

| Flag | Default Threshold | Configurable via |
|---|---|---|
| Sector WARNING | >30% allocation | `SECTOR_CONCENTRATION_WARNING_PCT` |
| Sector CRITICAL | >60% allocation | `SECTOR_CONCENTRATION_CRITICAL_PCT` |
| Single-stock WARNING | >15% weight | `SINGLE_STOCK_WARNING_PCT` |
| High beta | β > 1.2 | `HIGH_BETA_THRESHOLD` |

---

## Phase 3 — Causal Chain Engine

`build_causal_chain` is the core of the assignment's *"Macro News → Sector → Stock → Portfolio"* requirement:

```python
# For each HIGH/MEDIUM impact news article affecting the portfolio:
{
  "news_id":          "NEWS_003",
  "headline":         "RBI raises repo rate by 50bps",
  "sentiment":        "NEGATIVE",
  "sentiment_score":  -0.75,
  "impact_level":     "HIGH",
  "affected_stocks":  ["HDFCBANK", "ICICIBANK", "SBIN"],  # intersection with holdings
  "affected_sectors": ["BANKING"],
  "causal_factors":   ["Higher rates compress NIM", "Loan growth slows"],
  "pnl_contribution_estimate": -34200   # ₹ estimated P&L impact
}
```

### Conflict Resolution
Articles with `conflict_flag=true` (e.g. positive USFDA news for Sun Pharma while the stock is down) are surfaced in a separate `conflict_flags[]` array. The agent's system prompt explicitly requires: *"Flag conflict_flag=true news — it signals anomalies that need explicit mention."* The LLM then explains the ambiguity: macro sell-off overriding stock-specific positive catalyst.

---

## Phase 4 — Observability & Self-Evaluation

### Langfuse Tracing
`agent/tracing.py` wraps every graph run with a `CallbackHandler`. Tracked per run:
- Every LLM prompt (system + messages)
- Every completion (response content + tool calls)
- Token usage (prompt tokens, completion tokens, cost estimate)
- Latency per node
- Session ID and user context

Enable by setting `LANGFUSE_SECRET_KEY` + `LANGFUSE_PUBLIC_KEY` in `.env`. Gracefully no-ops if keys are missing.

### LLM-as-a-Judge (`agent/nodes/judge.py`)

After every final response, a **second isolated LLM call** evaluates quality:

```python
judge_llm = llm.with_config({"tags": ["judge"], "temperature": 0.1})
# Reads: original user_query + final ai_response
# Outputs: structured JSON scores
```

**Scoring Rubric (JUDGE_PROMPT):**

| Dimension | What It Measures |
|---|---|
| `factual_grounding` | Are real ₹ values, exact %, named stocks cited? (0 = vague, 10 = data-rich) |
| `causal_reasoning` | Does it explain WHY (chain) not just WHAT (list)? |
| `completeness` | Was every part of the question answered? |
| `actionability` | Are recommendations concrete and implementable? |
| `risk_awareness` | Were risks proactively identified even if not asked? |
| `conciseness` | Is length appropriate for the question complexity? |

**Output JSON:**
```json
{
  "scores": {
    "factual_grounding": 9,
    "causal_reasoning": 8,
    "completeness": 9,
    "actionability": 7,
    "risk_awareness": 10,
    "conciseness": 8
  },
  "overall": 8.5,
  "verdict": "EXCELLENT",
  "strengths": ["Rich ₹ data with exact attribution", "Proactive CRITICAL concentration flag"],
  "improvements": ["Could suggest specific rebalancing percentages"]
}
```

This renders as a **live score panel** below every AI response in the frontend.

### Confidence Score
Every agent response ends with a mandatory confidence line (enforced by SYSTEM_PROMPT):
```
**Confidence: HIGH** — derived from portfolio data, causal chain, and 3 corroborating news sources.
```
`HIGH` = multiple sources agree | `MEDIUM` = some signals conflict | `LOW` = data missing or contradictory

---

## Streaming Architecture (SSE Events)

`POST /api/chat` streams these events to the frontend in real time:

```
data: {"type": "node_start",   "node": "financial_advisor",  "label": "🧠 Reasoning"}
data: {"type": "tool_call",    "tool": "think",              "label": "💭 Planning next steps"}
data: {"type": "tool_call",    "tool": "get_portfolio_analysis", "label": "💼 Analysing portfolio"}
data: {"type": "tool_done",    "tool": "get_portfolio_analysis"}
data: {"type": "tool_call",    "tool": "build_causal_chain", "label": "🔗 Building causal chain"}
data: {"type": "tool_done",    "tool": "build_causal_chain"}
data: {"type": "token",        "content": "Your portfolio "}
data: {"type": "token",        "content": "fell **-2.73%**..."}
...
data: {"type": "judge_result", "result": {"scores": {...}, "verdict": "EXCELLENT", "overall": 8.5}}
data: {"type": "node_end",     "node": "response_judge"}
data: {"type": "done"}
```

The frontend **reasoning tracker** builds dynamically from these events — users see exactly which tools the agent is calling in real time before the response starts streaming.

---

## Data Flow (Startup to Response)

```
python run.py
    ↓
DataLoader          → reads 6 JSON files into memory
DataRegistry        → auto-discovers all entities (portfolios, stocks, sectors, news, MFs)
ChatOpenAI(gpt-4o)  → initialised with streaming=True
build_graph()       → compiles StateGraph: 3 nodes + conditional edges + MemorySaver
app.state           → stores loader, registry, llm, graph, settings

POST /api/chat
    ↓
create_initial_state(user_message, portfolio_id, session_id, registry)
    ↓ AgentState {messages: [HumanMessage], session_id, portfolio_id}
graph.astream_events(state, config, version="v2")
    ↓
  [financial_advisor node]
    SYSTEM_PROMPT.format(portfolio_hint, available_stocks, sectors, portfolio_ids)
    llm_with_tools.invoke([SystemMessage] + messages)
    → if tool_calls: route to tool_executor
    → if content:    route to response_judge
  [tool_executor node]
    ToolNode.execute(tool_calls) → returns ToolMessages into state
    → loops back to financial_advisor
  [response_judge node]
    judge_llm.invoke(JUDGE_PROMPT.format(user_query, ai_response))
    → {"judge_result": {scores, verdict, strengths, improvements}}
    ↓
SSE stream → frontend
```

---

## Setup

```powershell
# 1. Install dependencies
cd "Agent Assignment\financial_advisor_agent"
pip install -r requirements.txt

# 2. Configure
copy .env.example .env
# Edit .env — set OPENAI_API_KEY=sk-...

# 3. Run
python run.py
```

Open **http://localhost:8000**

API docs at **http://localhost:8000/docs**

---

## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | SSE stream — agent chat |
| `GET` | `/api/registry` | All discovered entities (used by frontend on load) |
| `GET` | `/api/registry/portfolios` | Portfolio list |
| `GET` | `/api/portfolio/{id}` | Full portfolio analysis |
| `GET` | `/api/portfolio/{id}/risk` | Risk flags only |
| `GET` | `/api/portfolio/{id}/news` | Portfolio-relevant news |
| `GET` | `/api/market/snapshot` | Full market snapshot |
| `GET` | `/api/market/movers?n=5` | Top N gainers + losers |
| `GET` | `/api/market/sectors` | All sectors ranked |
| `GET` | `/api/market/sector/{name}` | Sector detail |
| `GET` | `/api/market/stock/{symbol}` | Stock detail |
| `GET` | `/api/market/news` | News feed (filterable by scope/sentiment) |

---

## Key Design Decisions

**Why ReAct over a fixed pipeline?**
A fixed 7-node pipeline wastes LLM calls — every query would pass through all nodes. ReAct scales naturally: 1 tool call for "Is market bullish?", 5–6 for "Why is my portfolio down and what should I do?".

**Why a separate judge call (not inline)?**
The judge needs the *complete* response to score it. Using `temperature=0.1` with no tools bound ensures deterministic, objective scoring isolated from the response-generation reasoning.

**Why `DataRegistry` with zero hardcoding?**
Adding a new portfolio, stock, or sector to the JSON files makes it automatically discoverable across the entire stack — agent, API, and frontend — at next startup. No code changes required.

**Why `think` as a tool (not built into the prompt)?**
Making planning a *tool call* creates an explicit, logged trace of the agent's reasoning in Langfuse. It also forces the LLM to commit its plan before acting, reducing unnecessary tool calls on complex queries.
