"""Basic prompt-injection screening; this is one layer, not a complete defense."""

import re

PATTERNS = (
    r"ignore (all )?(previous|prior) instructions",
    r"reveal (the )?(system prompt|secrets)",
    r"show .*system prompt",
    r"bypass .*guardrail",
)


def detect_prompt_injection(text: str) -> bool:
    return any(re.search(pattern, text.lower()) for pattern in PATTERNS)
