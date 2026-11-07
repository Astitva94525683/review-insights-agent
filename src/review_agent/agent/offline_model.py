"""A deterministic, rule-based stand-in for an LLM, used for local runs without an API key and for CI.

It speaks the same interface as a real chat model (tool calls in, tool calls out), so the whole
pipeline runs end to end: guardrails, tools, memory, citation checks, tracing and evals. It is not
a language model and is refused in production (see config.py).
"""
import json
import re
import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_SOFTWARE_WORDS = re.compile(
    r"\b(software|tool|tools|crm|help ?desk|ticket|project|accounting|invoice|gst|chatbot|bot|app|apps|"
    r"integration|integrate|pricing|price|support|review|reviews|vendor|plan|team|feature|features|"
    r"product|products|onboarding|setup|sso|import|migration|dashboard|report|reports|slack|whatsapp)\b", re.I)
_TEAM = re.compile(r"\b(\d{1,5})\s*(?:-\s*)?(?:person|people|member|employee|agent|seat|user)s?\b|"
                   r"\bteam of (\d{1,5})\b", re.I)
_CATALOGUE_LINE = re.compile(r"^- ([a-z0-9_-]+): (.+?) \((.+)\)$", re.M)


def _catalogue(messages: list[BaseMessage]) -> list[tuple[str, str]]:
    system = next((m.content for m in messages if isinstance(m, SystemMessage)), "")
    return [(pid, name) for pid, name, _ in _CATALOGUE_LINE.findall(system)]


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]


class OfflineModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "offline-heuristic"

    def bind_tools(self, tools, **kwargs):  # tools are addressed by name
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self._respond(messages)
        prompt_chars = sum(len(str(m.content)) for m in messages)
        reply.usage_metadata = {"input_tokens": prompt_chars // 4 + 1,
                                "output_tokens": len(str(reply.content)) // 4 + 1,
                                "total_tokens": (prompt_chars + len(str(reply.content))) // 4 + 2}
        return ChatResult(generations=[ChatGeneration(message=reply)])

    # ------------------------------------------------------------------------------
    def _respond(self, messages: list[BaseMessage]) -> AIMessage:
        if isinstance(messages[0], SystemMessage) and messages[0].content.startswith("You maintain the memory"):
            turns = messages[-1].content.split("New turns:\n", 1)[-1]
            buyer = [ln[len("Buyer: "):][:160] for ln in turns.splitlines() if ln.startswith("Buyer: ")]
            return AIMessage("Buyer asked about: " + " | ".join(buyer))

        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        tool_msgs = [m for m in messages[last_human:] if isinstance(m, ToolMessage)]
        question = messages[last_human].content
        if question.startswith("[Automated quality check]"):
            earlier = max(i for i, m in enumerate(messages[:last_human]) if isinstance(m, HumanMessage))
            tool_msgs = [m for m in messages[earlier:] if isinstance(m, ToolMessage)]
            return AIMessage(self._compose(tool_msgs))
        if tool_msgs:
            return AIMessage(self._compose(tool_msgs))
        return self._plan(question, _catalogue(messages))

    def _plan(self, text: str, catalogue: list[tuple[str, str]]) -> AIMessage:
        low = text.lower()
        products = [pid for pid, name in catalogue
                    if pid in low or name.lower() in low or name.split()[0].lower() in low]
        calls = []
        team = _TEAM.search(text)
        if team:
            calls.append(("save_buyer_preference",
                          {"key": "company_size", "value": f"{team.group(1) or team.group(2)} people"}))
        if len(products) >= 2 and re.search(r"\b(compare|vs\.?|versus|better)\b", low):
            calls.append(("compare_products", {"product_ids": products[:4]}))
        elif len(products) == 1 and re.search(r"\b(average|rating|rated|score|how many reviews)\b", low):
            calls.append(("get_product_stats", {"product_id": products[0]}))
        elif re.search(r"\b(which|what|list)\b.{0,20}\bproducts\b|\bcatalog(ue)?\b", low) and not products:
            calls.append(("list_products", {}))
        elif products or _SOFTWARE_WORDS.search(low):
            args = {"query": text[:300]}
            if len(products) == 1:
                args["product_id"] = products[0]
            calls.append(("search_reviews", args))
        if not calls:
            return AIMessage("I can only help with choosing and comparing software using user reviews. "
                             "Which kind of tool are you evaluating?")
        return AIMessage("", tool_calls=[{"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:12]}",
                                          "type": "tool_call"} for n, a in calls])

    def _compose(self, tool_msgs: list[ToolMessage]) -> str:
        parts = []
        for m in tool_msgs:
            try:
                data = json.loads(m.content)
            except (TypeError, ValueError):
                continue
            if "error" in data and not data.get("products"):
                parts.append(f"I couldn't complete that lookup: {data['error']}")
            if "saved" in data:
                key, value = next(iter(data["saved"].items()))
                parts.append(f"Noted your {key.replace('_', ' ')}: {value}.")
            if "results" in data:
                if not data["results"]:
                    parts.append("I couldn't find reviews about that. Try naming a product or a feature.")
                else:
                    lines = [f"- {r['product']} ({r['rating']}/5, {r['reviewer_role'] or 'reviewer'}): "
                             f"{_first_sentence(r['excerpt'])} [{r['review_id']}]" for r in data["results"][:3]]
                    parts.append("Here is what reviewers say:\n" + "\n".join(lines))
            if "review_count" in data:
                parts.append(f"{data['name']} has {data['review_count']} reviews with an average rating of "
                             f"{data['average_rating']}/5.")
            if "products" in data and data["products"] and "review_count" in data["products"][0]:
                parts.append("\n".join(f"- {p['name']}: {p['average_rating']}/5 from {p['review_count']} reviews"
                                       for p in data["products"]))
            elif "products" in data and data["products"]:
                parts.append("Products: " + ", ".join(p["name"] for p in data["products"]) + ".")
        return "\n\n".join(parts) or "I couldn't find anything useful for that question."
