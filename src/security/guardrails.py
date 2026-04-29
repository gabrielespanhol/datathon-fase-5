# src/security/guardrails.py

import re
import html

BLOCK_PATTERNS = [
    r"ignore previous instructions",
    r"you are now",
    r"system:",
    r"administrator",
    r"root access",
    r"delete.*data",
    r"show.*password",
    r"reveal.*secret",
    r"bypass.*security",
]

DANGEROUS_OUTPUT_PATTERNS = [
    r"<script",
    r"javascript:",
    r"eval\(",
    r"exec\(",
    r"system\(",
    r"import os",
    r"subprocess",
    r"rm -rf",
    r"format.*drive",
]


def validate_input(text: str) -> bool:
    """Valida input do usuário contra padrões de prompt injection."""
    text_lower = text.lower()
    for pattern in BLOCK_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False
    return True


def sanitize_output(text: str) -> str:
    """Sanitiza output do LLM para prevenir XSS e execução de código."""
    # Remove tags HTML
    text = html.escape(text)

    # Remove padrões perigosos
    for pattern in DANGEROUS_OUTPUT_PATTERNS:
        text = re.sub(pattern, "[BLOCKED]", text, flags=re.IGNORECASE)

    # Limita comprimento
    if len(text) > 1000:
        text = text[:1000] + "..."

    return text


def validate_output(text: str) -> bool:
    """Valida se output é seguro para exibição."""
    text_lower = text.lower()
    for pattern in DANGEROUS_OUTPUT_PATTERNS:
        if re.search(pattern, text_lower):
            return False
    return True
