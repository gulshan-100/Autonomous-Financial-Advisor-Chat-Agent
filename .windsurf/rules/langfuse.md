# Langfuse Skill
Source: https://github.com/langfuse/skills

This skill helps use Langfuse effectively across all common workflows: instrumenting applications, migrating prompts, debugging traces, and accessing data programmatically.

## Core Principles

1. **Documentation First**: NEVER implement based on memory. Always fetch current docs before writing code (Langfuse updates frequently).
2. **CLI for Data Access**: Use `langfuse-cli` when querying/modifying Langfuse data.
3. **Best Practices by Use Case**: Check the relevant reference below before implementing.
4. **Use latest Langfuse versions**: Unless specified otherwise, always use the latest SDK.

## Framework Integration (LangChain / LangGraph)

```python
from langfuse.langchain import CallbackHandler
from langfuse import get_client

# Single handler instance (reads credentials from env vars automatically)
langfuse_handler = CallbackHandler()

# Pass to LangGraph / LangChain via config
config = {
    "callbacks": [langfuse_handler],
    "metadata": {
        "langfuse_session_id": session_id,   # groups conversation turns
        "langfuse_user_id":    user_id,       # enables per-user analytics
        "langfuse_tags":       ["tag-1"],     # enables filtering
    }
}
graph.invoke(input, config=config)

# Flush before the process/connection closes (critical in async/streaming)
get_client().flush()
```

## Env vars required
```
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

## Baseline Requirements (every trace must have)
| Requirement | Check |
|---|---|
| Model name | Captured automatically by LangChain integration |
| Token usage | Captured automatically by LangChain integration |
| Good trace names | Use descriptive names: `financial-advisor-chat` |
| Session ID | Groups multi-turn conversations |
| Sensitive data masked | Never log PII or API keys |

## Common Mistakes
| Mistake | Fix |
|---|---|
| `from langfuse.callback import CallbackHandler` | Use `from langfuse.langchain import CallbackHandler` |
| `handler.flush()` | Use `get_client().flush()` |
| Session ID in constructor | Pass via `config["metadata"]["langfuse_session_id"]` |
| Langfuse import before env vars loaded | Import after `load_dotenv()` or Settings init |
| No flush in async/streaming context | Always call `get_client().flush()` after stream ends |
