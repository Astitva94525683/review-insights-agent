"""Conversation memory in three layers:

1. Working memory: the most recent turns, kept word-for-word, within a token budget.
2. Summary memory: when the history grows past the budget, the oldest turns are folded into a
   short rolling summary by the LLM, so long chats keep context without growing the prompt.
3. Long-term profile: structured buyer facts (team size, budget, ...) saved by a tool and
   injected into the system prompt on every turn.

Everything is stored in the database, so any API worker can serve any session.
Anything written back into the prompt is sanitised first, so a user cannot plant instructions
in memory that later act like system rules (memory poisoning).
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..guardrails.patterns import neutralize_injection, redact_pii
from ..observability import get_logger, span
from ..storage.repository import Repository

log = get_logger(__name__)

SUMMARY_PROMPT = (
    "You maintain the memory of a software-buying assistant. Update the running summary with the new "
    "conversation turns. Keep only what matters for later answers: products discussed, the buyer's needs, "
    "conclusions reached and open questions. Do not include personal data. Write at most 120 words of "
    "plain sentences. Output only the summary."
)


def estimate_tokens(text: str) -> int:
    """About 4 characters per token for English. Good enough for budgeting, no tokenizer needed."""
    return len(text) // 4 + 1


class ConversationMemory:
    def __init__(self, repo: Repository, model: BaseChatModel, token_budget: int, summary_max_chars: int):
        self.repo, self.model = repo, model
        self.budget, self.summary_max_chars = token_budget, summary_max_chars

    def history(self, session_id: str) -> list[BaseMessage]:
        """Unsummarised turns, newest kept first if they ever exceed the budget."""
        rows = self.repo.messages(session_id, unsummarized_only=True)
        kept, used = [], 0
        for m in reversed(rows):
            used += estimate_tokens(m.content)
            if used > self.budget:
                break
            kept.append(HumanMessage(m.content) if m.role == "human" else AIMessage(m.content))
        kept.reverse()
        while kept and isinstance(kept[0], AIMessage):  # never start the history mid-turn
            kept.pop(0)
        return kept

    def save_turn(self, session_id: str, request_id: str, human: str, ai: str) -> None:
        self.repo.add_messages(session_id, request_id, [("human", human), ("ai", ai)])

    def maybe_summarize(self, session_id: str, current_summary: str) -> str:
        """Fold the oldest turns into the summary once the history passes the token budget."""
        rows = self.repo.messages(session_id, unsummarized_only=True)
        total = sum(estimate_tokens(m.content) for m in rows)
        if total <= self.budget:
            return current_summary
        fold, remaining = [], total
        for i, m in enumerate(rows):
            if i >= len(rows) - 2:
                break  # always keep the latest turn word-for-word
            if remaining <= self.budget // 2 and m.role == "human":
                break  # stop on a turn boundary
            fold.append(m)
            remaining -= estimate_tokens(m.content)
        if not fold:
            return current_summary
        transcript = "\n".join(f"{'Buyer' if m.role == 'human' else 'Assistant'}: {m.content}" for m in fold)
        try:
            with span("memory.summarize", messages=len(fold)):
                reply = self.model.invoke([
                    SystemMessage(SUMMARY_PROMPT),
                    HumanMessage(f"Current summary:\n{current_summary or '(none)'}\n\nNew turns:\n{transcript}"),
                ])
        except Exception:  # memory must never break a request; try again next turn
            log.warning("summary_failed", exc_info=True)
            return current_summary
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        summary = sanitize_memory(text, self.summary_max_chars)
        self.repo.update_session(session_id, summary=summary)
        self.repo.mark_summarized([m.id for m in fold])
        log.info("history_summarized", extra={"folded_messages": len(fold), "summary_chars": len(summary)})
        return summary


def sanitize_memory(text: str, max_chars: int) -> str:
    text, _ = neutralize_injection(text)
    text, _ = redact_pii(text)
    return " ".join(text.split())[:max_chars]
