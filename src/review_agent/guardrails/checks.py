"""Input, retrieved-content and output guardrails.

Input:   runs before the model sees anything. Blocks injection, enforces length, redacts PII.
Content: runs on every review a tool returns. Review text is written by the public, so it is
         untrusted data: injection phrases are removed and PII is redacted before the model reads it.
Output:  runs on the final answer. Every [R...] citation must point to a review retrieved in this
         turn, answers built on reviews must cite them, the system prompt must not leak, PII is redacted.
"""
import re
import unicodedata
from dataclasses import dataclass, field

from .patterns import detect_injection, neutralize_injection, redact_pii

CITATION = re.compile(r"\[(R[0-9A-Za-z_-]{1,63})\]")

BLOCKED_REPLY = ("I can only help with questions about software products and their user reviews. "
                 "Please rephrase your question.")


@dataclass
class InputCheck:
    allowed: bool
    text: str
    flags: list[str] = field(default_factory=list)
    reply: str | None = None  # shown to the user when the input is blocked


@dataclass
class OutputCheck:
    ok: bool
    answer: str
    flags: list[str] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    invalid_ids: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # folds look-alike characters used to dodge filters
    return "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()


def check_input(text: str, max_chars: int) -> InputCheck:
    text = _clean(text or "")
    if not text:
        return InputCheck(False, "", ["empty_input"], "Please type a question.")
    if len(text) > max_chars:
        return InputCheck(False, "", ["input_too_long"],
                          f"Please keep your question under {max_chars} characters.")
    hits = detect_injection(text)
    if hits:
        return InputCheck(False, "", [f"prompt_injection:{h}" for h in hits], BLOCKED_REPLY)
    redacted, kinds = redact_pii(text)
    return InputCheck(True, redacted, [f"pii_redacted:{k}" for k in kinds])


def sanitize_untrusted(text: str, max_chars: int = 600) -> tuple[str, list[str]]:
    flags = []
    text, injected = neutralize_injection(_clean(text))
    if injected:
        flags.append("review_injection_removed")
    text, kinds = redact_pii(text)
    flags += [f"review_pii_redacted:{k}" for k in kinds]
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text, flags


def check_output(answer: str, retrieved_ids: set[str], *, require_citations: bool,
                 canary: str, max_chars: int) -> OutputCheck:
    flags: list[str] = []
    answer = (answer or "").strip()
    if canary and canary in answer:
        return OutputCheck(False, "", ["system_prompt_leak"])
    cited = list(dict.fromkeys(CITATION.findall(answer)))
    invalid = [c for c in cited if c not in retrieved_ids]
    if invalid:
        flags.append("invalid_citation")
    if require_citations and not cited:
        flags.append("missing_citation")
    if not answer:
        flags.append("empty_answer")
    answer, kinds = redact_pii(answer)
    flags += [f"pii_redacted_output:{k}" for k in kinds]
    if len(answer) > max_chars:
        answer = answer[:max_chars].rsplit(" ", 1)[0] + " ..."
        flags.append("output_truncated")
    ok = not ({"invalid_citation", "missing_citation", "empty_answer"} & set(flags))
    return OutputCheck(ok, answer, flags, cited, invalid)
