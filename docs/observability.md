# Observability

Three signals, linked by IDs: a metric shows something is wrong, the log line gives the request ID, and the
trace shows which step caused it.

## Logs (`observability/logging.py`)

One JSON object per line with `request_id`, `session_id` and `trace_id` stamped on every record. The
`agent_run` line has status, steps, tools, guardrail flags, latency, tokens and cost. User text is never
logged, only flags and sizes, because messages can contain personal data.

## Traces (`observability/tracing.py`)

```text
agent.run                 request_id, session_id, agent.status, agent.steps, agent.cost_usd
├── guardrails.input      guardrail.flags
├── llm.call              gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens
├── tool.search_reviews   gen_ai.tool.name, tool.args, tool.outcome
├── llm.call
├── guardrails.output     guardrail.ok, guardrail.flags
└── memory.summarize      (only when history passes the budget)
```

Attribute names follow the OpenTelemetry GenAI conventions, so Jaeger, Grafana Tempo, Honeycomb or Datadog
read them as-is. `RIA_OTEL_EXPORTER=console` prints spans locally; `otlp` sends them to a collector.
LangSmith can run alongside (`LANGSMITH_TRACING=true`) for prompt-level debugging.

## Metrics (`observability/metrics.py`, served at `/metrics`)

| Metric | Question it answers |
|---|---|
| `ria_http_requests_total`, `ria_http_request_seconds` | Is the API up and fast? |
| `ria_agent_runs_total{status}` | How many answers, blocks, fallbacks, errors? |
| `ria_agent_run_seconds`, `ria_agent_steps` | How slow, and how many model calls per answer? |
| `ria_llm_tokens_total`, `ria_llm_cost_usd_total` | What does it cost? |
| `ria_llm_errors_total` | Is the provider failing? |
| `ria_tool_calls_total{tool,outcome}`, `ria_tool_seconds` | Which tools fail or slow down? |
| `ria_guardrail_events_total{flag}` | Which guardrails fire, and how often? |
| `ria_answers_total{grounded}` | Live grounding rate (online eval) |
| `ria_feedback_total{score}` | Are users happy? |

Alert rules for these are in `deploy/alerts.yml`.

## Cost tracking (`observability/cost.py`)

Each provider returns token usage per call. The runner adds it up per request and multiplies by
`RIA_LLM_INPUT_COST_PER_MTOK` and `RIA_LLM_OUTPUT_COST_PER_MTOK`. The cost is returned in the response,
stored in `agent_runs`, logged and counted in `ria_llm_cost_usd_total`. On NVIDIA's free tier both prices
are 0, so cost reads 0 while tokens are still counted; set the prices when moving to a paid model.
