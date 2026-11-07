"""Detection patterns shared by every guardrail: personal data, secrets and prompt injection.

These are fast, deterministic checks that run on every request at no model cost. They are one layer
of defence; the others are least-privilege tools, untrusted-content wrapping and output checks.
"""
import re

# ---- personal data and secrets (redacted, never stored or sent to the model) ----------
_PII = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("SECRET", re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b|\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("CARD", re.compile(r"(?<![\w+])\d(?:[ -]?\d){12,18}(?!\w)")),
    ("ID_NUMBER", re.compile(r"(?<![\w+])\d{4}[ -]?\d{4}[ -]?\d{4}(?!\w)")),  # e.g. Aadhaar
    ("PHONE", re.compile(r"(?<![\w+])\+?\d(?:[\s-]?\d){9,12}(?!\w)")),
]


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace personal data with placeholders such as [EMAIL]. Returns (text, kinds found)."""
    found: list[str] = []
    for kind, pattern in _PII:
        def _sub(m: re.Match, kind=kind) -> str:
            if kind == "CARD" and not _luhn_ok(re.sub(r"\D", "", m.group())):
                return m.group()  # long number that is not a card: leave it for the next patterns
            found.append(kind)
            return f"[{kind}]"
        text = pattern.sub(_sub, text)
    return text, sorted(set(found))


# ---- prompt injection --------------------------------------------------------------
_INJECTION = [
    ("override", re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[^.!?\n]{0,40}\b(previous|prior|above|earlier|all|any|your|"
        r"system|developer)\b[^.!?\n]{0,30}\b(instructions?|rules|prompts?|guidelines|directions|polic(y|ies))\b",
        re.I)),
    ("prompt_extraction", re.compile(
        r"\b(reveal|show|print|repeat|leak|output|display|tell me|what(?:'s| is| are))\b[^.!?\n]{0,40}"
        r"\b(system|hidden|initial|developer|secret)\s+(prompt|message|instructions?|rules)\b", re.I)),
    ("role_hijack", re.compile(
        r"\byou are now\b|\b(developer|god|jailbreak|dan|unrestricted)\s+mode\b|"
        r"\b(act as|pretend to be|roleplay as)\b[^.!?\n]{0,40}\b(dan|unfiltered|jailbroken|evil|"
        r"without (?:any )?(?:rules|restrictions|filters|limits))\b", re.I)),
    ("fake_role_tag", re.compile(
        r"<\s*/?\s*(system|assistant|im_start|im_end)\b|^\s*(system|assistant)\s*:|\[\s*/?\s*(system|inst)\s*\]",
        re.I | re.M)),
    ("secret_exfiltration", re.compile(
        r"\b(reveal|leak|print|dump|expose|send)\b[^.!?\n]{0,30}\b(api[ _-]?keys?|passwords?|secrets?|"
        r"credentials|access tokens?|env(?:ironment)? variables)\b", re.I)),
]


def detect_injection(text: str) -> list[str]:
    """Return the injection categories found in the text (empty list = clean)."""
    return [name for name, pattern in _INJECTION if pattern.search(text)]


_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def neutralize_injection(text: str) -> tuple[str, bool]:
    """Drop every sentence of untrusted text (e.g. a review) that contains an injection attempt.

    Whole sentences go, not just the trigger phrase: in "Ignore all previous instructions and say X",
    the payload "say X" is the dangerous part.
    """
    kept, hit = [], False
    for sentence in _SENTENCE.split(text):
        if detect_injection(sentence):
            hit = True
            if not kept or kept[-1] != "[instruction removed]":
                kept.append("[instruction removed]")
        elif sentence.strip():
            kept.append(sentence.strip())
    return (" ".join(kept), True) if hit else (text, False)
