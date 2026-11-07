# Memory and tools

## Memory: three layers (`memory/manager.py`)

| Layer | What it holds | How it is used |
|---|---|---|
| Working memory | Recent turns, word for word | Sent as chat messages, within `RIA_HISTORY_TOKEN_BUDGET` |
| Summary memory | A rolling summary of older turns | When history passes the budget, the oldest turns are summarised by the LLM and marked as summarised; the latest turn always stays word for word |
| Long-term profile | Buyer facts: company size, budget, industry, must-have feature, current tool | Saved by the `save_buyer_preference` tool; injected into the system prompt every turn |

All of it lives in the database, so any replica can serve any session, and `DELETE /v1/sessions/{id}`
erases it. Tokens are estimated at about 4 characters each, which is enough for budgeting.

**Safety**: summaries and saved preferences go back into the prompt, so they are sanitised first. A
preference containing an injection attempt ("ignore previous instructions and ...") is refused, values are
capped at 200 characters, and memory is scoped to the caller's own session, so nobody can plant text in
someone else's prompt. Blocked messages are never stored. If summarising fails, the request still succeeds
and the summary is retried next turn.

## Tools (`tools/registry.py`)

| Tool | Input | Returns |
|---|---|---|
| `list_products` | optional category | catalogue |
| `search_reviews` | query, optional product, min/max rating, limit (max 8) | review IDs, ratings, sanitised excerpts |
| `get_product_stats` | product ID | review count, average rating, distribution, latest review date |
| `compare_products` | 2-4 product IDs | stats side by side |
| `save_buyer_preference` | one of five keys, value | confirmation |

Design rules:

- **Typed inputs**: each tool has a Pydantic schema; the model sees it as JSON schema. Bad arguments come
  back to the model as a clear error instead of crashing the request.
- **Errors as data**: unknown products, unknown tools and the tool budget all return `{"error": ...}`, so
  the model can correct itself.
- **Numbers from the database**: ratings and counts are computed, never generated.
- **Evidence registry**: every review a tool returns is recorded for the turn; the output check uses it to
  verify citations.
- **Per-request context**: tools are built fresh for each request with that request's session, so one
  user's tool calls can never touch another user's data.
