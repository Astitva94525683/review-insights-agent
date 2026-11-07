"""Prometheus metrics served at GET /metrics. These answer the on-call questions: is it up, is it
slow, what does it cost, are guardrails firing, are answers grounded, are users happy.
"""
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        self.http_requests = Counter("ria_http_requests_total", "HTTP requests",
                                     ["method", "route", "status"], registry=r)
        self.http_latency = Histogram("ria_http_request_seconds", "HTTP latency", ["route"], registry=r,
                                      buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30))
        self.agent_runs = Counter("ria_agent_runs_total", "Agent runs by outcome", ["status"], registry=r)
        self.agent_latency = Histogram("ria_agent_run_seconds", "End-to-end agent latency", registry=r,
                                       buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60))
        self.agent_steps = Histogram("ria_agent_steps", "LLM calls per request", registry=r,
                                     buckets=(1, 2, 3, 4, 6, 8, 12))
        self.llm_tokens = Counter("ria_llm_tokens_total", "LLM tokens", ["type"], registry=r)
        self.llm_cost = Counter("ria_llm_cost_usd_total", "Estimated LLM spend in USD", registry=r)
        self.llm_errors = Counter("ria_llm_errors_total", "LLM call failures", ["error"], registry=r)
        self.tool_calls = Counter("ria_tool_calls_total", "Tool calls", ["tool", "outcome"], registry=r)
        self.tool_latency = Histogram("ria_tool_seconds", "Tool latency", ["tool"], registry=r,
                                      buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1))
        self.guardrail_events = Counter("ria_guardrail_events_total", "Guardrail flags raised", ["flag"],
                                        registry=r)
        self.answers = Counter("ria_answers_total", "Answers by online grounding check", ["grounded"],
                               registry=r)
        self.feedback = Counter("ria_feedback_total", "User feedback", ["score"], registry=r)
        self.index_size = Gauge("ria_index_reviews", "Reviews in the search index", registry=r)
