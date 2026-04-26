# Autonomous Financial Advisor Agent

A fully dynamic, LangGraph-powered AI financial advisor that reasons over live market data.

## Architecture

```
financial_advisor_agent/
├── config.py                    # All settings from .env (no hardcoding)
├── run.py                       # Server startup script
├── requirements.txt
├── .env.example                 # Copy to .env and fill API key
│
├── data_layer/
│   ├── loader.py                # Loads all 6 JSON files once at startup
│   ├── registry.py              # DataRegistry: auto-discovers all entities
│   ├── portfolio_analyzer.py    # Dynamic portfolio analytics
│   ├── market_analyzer.py       # Market intelligence
│   └── news_processor.py        # Causal chain builder
│
├── agent/
│   ├── state.py                 # LangGraph AgentState TypedDict
│   ├── graph.py                 # StateGraph compilation
│   ├── prompts/templates.py     # All LLM prompt templates
│   └── nodes/
│       ├── intent_classifier.py  # Node 1: GPT-4o intent extraction
│       ├── context_gatherer.py   # Node 2: Dynamic data gathering
│       ├── risk_analyzer.py      # Node 3: Pure Python risk computation
│       ├── news_reasoner.py      # Node 4: GPT-4o causal chain reasoning
│       ├── market_analyzer.py    # Node 5: GPT-4o market synthesis
│       ├── advisor_synthesizer.py # Node 6: GPT-4o structured advice
│       └── response_formatter.py  # Node 7: Streaming markdown output
│
├── app/
│   ├── main.py                  # FastAPI app + lifespan startup
│   ├── models/request.py        # Pydantic request models
│   └── routes/
│       ├── chat.py              # POST /api/chat (SSE streaming)
│       ├── registry.py          # GET /api/registry
│       ├── portfolio.py         # GET /api/portfolio/{id}
│       └── market.py            # GET /api/market/*
│
└── frontend/
    ├── index.html               # Shell only — no data in HTML
    ├── style.css                # Dark glassmorphic UI
    └── script.js                # Dynamic — builds all UI from API
```

## LangGraph Reasoning Flow

```
User Query
    ↓
[1] intent_classifier    — GPT-4o: extract intent, portfolio_id, symbols, sectors
    ↓
[2] context_gatherer     — Python: dynamically call analyzers based on intent
    ↓
[3] risk_analyzer        — Python: compute risk flags from thresholds in Settings
    ↓        ↘
[4] news_reasoner        [market_only path: no portfolio]
    ↓
[5] market_analyzer      — GPT-4o: synthesize market narrative
    ↓
[6] advisor_synthesizer  — GPT-4o: structured JSON advice
    ↓
[7] response_formatter   — GPT-4o streaming: markdown response → SSE tokens
    ↓
   END
```

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

- **Zero hardcoding**: All portfolio IDs, stock symbols, sector names, news IDs are discovered at runtime from JSON files
- **Fully dynamic frontend**: HTML is a shell; JS fetches `/api/registry` and builds all UI elements
- **Configurable thresholds**: Risk limits, model name, temperature — all in `.env`
- **Streaming**: SSE delivers reasoning node progress + token-by-token response
- **Per-session memory**: LangGraph `MemorySaver` maintains conversation history per `session_id`
