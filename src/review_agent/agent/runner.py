"""The agent loop: input guardrails -> memory -> model/tool loop -> output guardrails -> save.

Hard budgets (steps, tool calls, characters) make every request finite. If the answer fails the
output checks, the model gets one automatic repair attempt with the exact problem; if it fails
again, the user gets a safe fallback instead of an ungrounded answer.
"""
import json
import re
import secrets
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from ..config import Settings
from ..guardrails import check_input, check_output
from ..memory import ConversationMemory
from ..observability import Metrics, Usage, get_logger, span
from ..retrieval.bm25 import BM25Index
from ..storage.repository import Repository
from ..tools.registry import ToolContext, build_tools
from .prompts import build_system_prompt, repair_instruction

log = get_logger(__name__)

FALLBACK_REPLY = ("I couldn't back that answer with the reviews I found, so I won't guess. "
                  "Try asking about a specific product or feature.")
STEP_LIMIT_REPLY = "That question needed more steps than I allow per request. Please ask something narrower."


class AgentUnavailable(Exception):
    """The model provider failed after retries. The API turns this into a 503."""


@dataclass
class AgentResult:
    request_id: str
    session_id: str
    status: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class _Budget:
    max_steps: int
    max_tool_calls: int
    steps: int = 0
    tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)


_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)


def _text(message: AIMessage) -> str:
    """Answer text only. Reasoning models may emit <think>...</think>; it never reaches the user."""
    if isinstance(message.content, str):
        text = message.content
    else:
        text = "".join(b.get("text", "") for b in message.content if isinstance(b, dict) and b.get("type") == "text")
    return _THINK.sub("", text).strip()


class AgentRunner:
    def __init__(self, settings: Settings, repo: Repository, index: BM25Index, model: BaseChatModel,
                 metrics: Metrics, canary: str | None = None):
        self.s, self.repo, self.index, self.model, self.metrics = settings, repo, index, model, metrics
        self.canary = canary or f"ria-{secrets.token_hex(8)}"
        self.memory = ConversationMemory(repo, model, settings.history_token_budget, settings.summary_max_chars)

    # ------------------------------------------------------------------------------
    def run(self, *, client_id: str, session_id: str, message: str, request_id: str) -> AgentResult:
        started = time.perf_counter()
        budget = _Budget(self.s.max_agent_steps, self.s.max_tool_calls)
        with span("agent.run", request_id=request_id, session_id=session_id) as root:
            with span("guardrails.input") as g:
                check = check_input(message, self.s.max_input_chars)
                g.set_attribute("guardrail.flags", check.flags)
            session = self.repo.get_or_create_session(session_id, client_id)
            if not check.allowed:  # blocked turns never reach the model or the memory
                return self._finish(request_id, session_id, "blocked_input", check.reply or "", [], check.flags,
                                    budget, started, root)

            ctx = ToolContext(self.repo, self.index, session_id, dict(session.profile or {}),
                              self.s.search_default_limit)
            tools = {t.name: t for t in build_tools(ctx)}
            llm = self.model.bind_tools(list(tools.values()))
            messages: list[BaseMessage] = [
                SystemMessage(build_system_prompt(self.repo.list_products(), ctx.profile, session.summary,
                                                  self.canary)),
                *self.memory.history(session_id),
                HumanMessage(check.text),
            ]
            try:
                answer, status = self._loop(llm, messages, tools, budget)
                flags = check.flags + ctx.flags
                out = None
                if status == "answered":
                    out = self._check_output(answer, ctx)
                    if not out.ok and "system_prompt_leak" not in out.flags and budget.steps < budget.max_steps:
                        self.metrics.guardrail_events.labels("repair_attempt").inc()
                        messages.append(HumanMessage(repair_instruction(out.flags, out.invalid_ids,
                                                                        list(ctx.retrieved))))
                        answer, status = self._loop(llm, messages, tools, budget)
                        flags.append("repaired")
                        out = self._check_output(answer, ctx) if status == "answered" else out
                    flags += [f for f in out.flags if f not in flags]
                    if not out.ok:
                        status, answer = "fallback", FALLBACK_REPLY
                    else:
                        answer = out.answer
                else:
                    answer = STEP_LIMIT_REPLY
            except AgentUnavailable:
                self._finish(request_id, session_id, "error", "", [], check.flags + ctx.flags, budget,
                             started, root)
                raise

            self.memory.save_turn(session_id, request_id, check.text, answer)
            self.memory.maybe_summarize(session_id, session.summary)
            cited = [ctx.retrieved[i] for i in (out.cited_ids if out and out.ok else []) if i in ctx.retrieved]
            return self._finish(request_id, session_id, status, answer, cited, flags, budget, started, root)

    # ------------------------------------------------------------------------------
    def _loop(self, llm, messages: list[BaseMessage], tools: dict, budget: _Budget) -> tuple[str, str]:
        while budget.steps < budget.max_steps:
            budget.steps += 1
            with span("llm.call", **{"gen_ai.request.model": self.s.llm_model,
                                     "gen_ai.system": self.s.llm_provider, "agent.step": budget.steps}) as s:
                try:
                    ai = llm.invoke(messages)
                except Exception as exc:  # provider already retried with backoff
                    self.metrics.llm_errors.labels(type(exc).__name__).inc()
                    log.error("llm_call_failed", extra={"error": type(exc).__name__})
                    raise AgentUnavailable(str(exc)) from exc
                budget.usage.add(ai.usage_metadata)
                s.set_attribute("gen_ai.usage.input_tokens", (ai.usage_metadata or {}).get("input_tokens", 0))
                s.set_attribute("gen_ai.usage.output_tokens", (ai.usage_metadata or {}).get("output_tokens", 0))
                s.set_attribute("agent.tool_calls", len(ai.tool_calls))
            messages.append(ai)
            if not ai.tool_calls:
                return _text(ai), "answered"
            for call in ai.tool_calls:
                messages.append(ToolMessage(content=self._call_tool(call, tools, budget),
                                            tool_call_id=call["id"], name=call["name"]))
        return "", "step_limit"

    def _call_tool(self, call: dict, tools: dict, budget: _Budget) -> str:
        name = call["name"]
        if name not in tools:
            self.metrics.tool_calls.labels("unknown", "rejected").inc()
            return json.dumps({"error": f"Unknown tool '{name}'. Available: {', '.join(tools)}."})
        if budget.tool_calls >= budget.max_tool_calls:
            self.metrics.tool_calls.labels(name, "budget_exceeded").inc()
            return json.dumps({"error": "Tool budget for this request is used up. Answer with what you have."})
        budget.tool_calls += 1
        budget.tools_used.append(name)
        started = time.perf_counter()
        with span(f"tool.{name}", **{"gen_ai.tool.name": name, "tool.args": json.dumps(call["args"])[:500]}) as s:
            try:
                result = tools[name].invoke(call["args"])
                outcome = "error" if result.startswith('{"error"') else "ok"
            except ValidationError as exc:  # bad arguments from the model: tell it what was wrong
                result = json.dumps({"error": "Invalid arguments: " + "; ".join(
                    f"{'.'.join(map(str, e['loc']))} {e['msg']}" for e in exc.errors())})
                outcome = "invalid_args"
            except Exception as exc:
                log.exception("tool_failed", extra={"tool": name})
                result, outcome = json.dumps({"error": f"{name} failed. Try a different approach."}), "exception"
                s.record_exception(exc)
            s.set_attribute("tool.outcome", outcome)
        self.metrics.tool_calls.labels(name, outcome).inc()
        self.metrics.tool_latency.labels(name).observe(time.perf_counter() - started)
        return result

    def _check_output(self, answer: str, ctx: ToolContext):
        with span("guardrails.output") as g:
            out = check_output(answer, set(ctx.retrieved), require_citations=bool(ctx.retrieved),
                               canary=self.canary, max_chars=self.s.max_output_chars)
            g.set_attribute("guardrail.flags", out.flags)
            g.set_attribute("guardrail.ok", out.ok)
        return out

    def _finish(self, request_id, session_id, status, answer, citations, flags, budget: _Budget, started,
                root) -> AgentResult:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        cost = budget.usage.cost_usd(self.s.llm_input_cost_per_mtok, self.s.llm_output_cost_per_mtok)
        grounded = status == "answered"
        result = AgentResult(request_id, session_id, status, answer,
                             [{"review_id": c["review_id"], "product": c["product"], "rating": c["rating"],
                               "title": c["title"], "excerpt": c["excerpt"][:240]} for c in citations],
                             flags, budget.tools_used, budget.steps, budget.usage.input_tokens,
                             budget.usage.output_tokens, cost, latency_ms)
        self.repo.save_run({"request_id": request_id, "session_id": session_id, "status": status,
                            "model": f"{self.s.llm_provider}:{self.s.llm_model}", "latency_ms": latency_ms,
                            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                            "cost_usd": cost, "steps": budget.steps, "tools_used": budget.tools_used,
                            "guardrail_flags": flags, "citations": [c["review_id"] for c in citations],
                            "citation_valid": status != "fallback"})
        m = self.metrics
        m.agent_runs.labels(status).inc()
        m.agent_latency.observe(latency_ms / 1000)
        m.agent_steps.observe(budget.steps)
        m.llm_tokens.labels("input").inc(result.input_tokens)
        m.llm_tokens.labels("output").inc(result.output_tokens)
        m.llm_cost.inc(cost)
        for f in flags:
            m.guardrail_events.labels(f).inc()
        if status in ("answered", "fallback"):
            m.answers.labels("yes" if grounded else "no").inc()
        root.set_attribute("agent.status", status)
        root.set_attribute("agent.steps", budget.steps)
        root.set_attribute("agent.cost_usd", cost)
        log.info("agent_run", extra={"status": status, "steps": budget.steps, "tools": budget.tools_used,
                                     "flags": flags, "latency_ms": latency_ms, "cost_usd": cost,
                                     "input_tokens": result.input_tokens, "output_tokens": result.output_tokens})
        return result
