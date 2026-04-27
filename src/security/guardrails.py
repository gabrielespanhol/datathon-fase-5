# src/security/guardrails.py

import re

BLOCK_PATTERNS = [
    r"ignore previous instructions",
    r"you are now",
    r"system:",
]


def validate_input(text: str):
    for pattern in BLOCK_PATTERNS:
        if re.search(pattern, text.lower()):
            return False
    return True
