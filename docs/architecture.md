# Architecture

## One request, step by step

1. **API layer** (`api/main.py`, `api/security.py`). The middleware gives the request an ID (or keeps the
   caller's `X-Request-ID`), times it and records metrics. The API key is checked in constant time and
   mapped to a client ID. A token bucket limits each client to `RIA_RATE_LIMIT_PER_MINUTE`.
2. **Input guardrails** (`guardrails/checks.py`). Text is normalised (look-alike letters folded, invisible
   characters removed), length-checked and scanned for prompt injection. Injection is blocked before any
   model call. Personal data is replaced with placeholders such as `[EMAIL]`.
3. **Memory** (`memory/manager.py`). The session's buyer profile and rolling summary go into the system
   prompt; recent turns go in as messages.
4. **Agent loop** (`agent/runner.py`). The model either answers or asks for tools. Tool results go back as
   tool messages. Hard limits: `RIA_MAX_AGENT_STEPS` model calls and `RIA_MAX_TOOL_CALLS` tool calls.
5. **Tools** (`tools/registry.py`). Typed inputs, JSON outputs, errors returned as data. Review text is
   sanitised before the model sees it, because the public wrote it.
6. **Output guardrails**. Every `[R...]` citation must be a review retrieved in this turn; an answer that used
   reviews must cite them; the system-prompt canary must not appear; PII is redacted.
7. **Repair or fallback**. A failed answer gets one retry with the exact problem stated. If it fails again,
   the user gets a safe fallback instead of an ungrounded answer.
8. **Save**. The turn goes to memory, the history is summarised if it is over budget, and an `agent_runs`
   row records status, steps, tools, flags, citations, tokens, cost and latency.

## Why it is built this way

| Decision | Reason |
|---|---|
| Tool calling, not one big RAG prompt | The model chooses between search, stats and comparison, and numbers come from the database, not the model |
| Guardrails enforced in code | Prompts are requests; code checks are guarantees |
| Review text treated as untrusted | Reviews are user-generated content, the classic route for indirect prompt injection |
| Database-backed sessions | Any API replica can serve any conversation |
| One `Repository` for all SQL | Tools, memory and API stay free of database code and are easy to test |
| Provider factory | NVIDIA Nemotron by default; switch to OpenAI, Anthropic, Gemini or OpenRouter with one setting |
| Offline stand-in model | The full pipeline, tests and evals run in CI without keys or cost |

## Data model

`products`, `reviews` (unique content hash blocks duplicates), `chat_sessions` (owner client, summary,
profile), `messages` (with a `summarized` flag), `agent_runs` (one row per request), `feedback`.
