# Interview guide

Short answers, one idea each. Open the file named in each answer to see the code.

**What is it, in 30 seconds?**
A FastAPI service where an LLM agent answers software buyers' questions from user reviews. It calls tools
to search reviews and pull ratings, cites the review behind every claim, and code checks those citations
before the answer goes out. It has layered guardrails, three-layer memory, eval suites that gate CI, and
logs, traces and metrics for production.

**Which model, and why?**
NVIDIA Nemotron 3 Super through NVIDIA's hosted API. It is an open-weight model built for agents, it
supports tool calling, and the API has a free tier. I turn its thinking mode off for fast tool calls,
and the provider is one setting, so I can compare it with OpenAI or Claude on the same evals.
(`agent/llm.py`)

**Why tool calling instead of plain RAG?**
Questions need different actions: search for opinions, compute an average, compare products. Tools let
the model pick the action, and numbers come from the database, so the model never guesses a rating.
(`tools/registry.py`)

**How do you stop hallucinations?**
Every claim must cite a review ID, and code checks that each cited ID was actually retrieved in this turn.
A failing answer gets one repair attempt with the exact problem; if it still fails, the user gets a safe
fallback. (`guardrails/checks.py::check_output`, `agent/runner.py`)

**How do you handle prompt injection?**
Four layers. The input check blocks known attack patterns, after normalising look-alike characters. Review
text is untrusted, so sentences with instructions are removed before the model reads them. Tools are
read-only except one scoped memory write. The output check catches invented citations and system-prompt
leaks through a canary token. (`guardrails/`)

**What does the pattern check miss?**
Paraphrased attacks with no trigger words. My eval set keeps two of them, so the 90% block rate is honest.
The next step is a small classifier model in front of the pattern check.

**How does memory work?**
Recent turns go in word for word within a token budget. Older turns are folded into a rolling summary by
the LLM. Lasting buyer facts, like team size, are saved by a tool into a profile that goes into the system
prompt. All of it is in the database, and it is sanitised before it re-enters the prompt.
(`memory/manager.py`)

**What are your evals?**
Three suites. Retrieval: recall@5 and MRR on 36 labelled questions. Guardrails: block rate on 20 attacks and
pass rate on 20 look-alike benign questions, plus PII and poisoned reviews. Agent: 18 end-to-end cases
checking tool choice, citations, refusals, safety and memory. Thresholds in `evals/thresholds.json` fail CI.

**Offline vs online evals?**
Offline runs the datasets in CI before release. Online checks every live answer for citation validity,
records it in `agent_runs`, and collects thumbs up/down. Flagged runs become new eval cases.

**How do you know the LLM judge is right?**
I don't assume it is. I label a sample by hand, compare with the judge, and fix the rubric where they disagree.

**What do you monitor in production?**
Latency, error rate, steps per answer, tokens and cost, tool failures, guardrail events, the live grounding
rate and user feedback, as Prometheus metrics with alert rules. Each request has one trace showing every
model and tool call, and JSON logs share the same request and trace IDs. (`observability/`, `deploy/alerts.yml`)

**How do you control cost and latency?**
Hard limits on model calls and tool calls per request, BM25 search with no embedding cost, short excerpts,
history summarised past a budget, and cost tracked per request so a spend alert fires early.

**How would you scale it?**
The API keeps no session state in memory, so it scales by adding replicas behind a load balancer with
PostgreSQL. I would move rate limiting to Redis and add a vector index once the review set is large.

**What are the numbers?**
60 tests at 95% coverage. Offline evals: retrieval recall@5 0.99, 18 of 20 attacks blocked with no benign
question blocked, all PII and poisoned reviews cleaned. These run on 50 synthetic reviews, so they prove the
pipeline works, not search quality at scale.

**How did you build it?**
With an AI coding agent (Claude Code). I set the requirements, reviewed the design and the code, and
verified the result with tests, evals and a live run.

Before an interview, run the project once, read `agent/runner.py` top to bottom, and be ready to walk
through one request from the API to the saved run.
