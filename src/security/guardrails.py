import html
import re
import unicodedata
from typing import Iterable

MAX_OUTPUT_LENGTH = 4000

BLOCK_PATTERNS = [
    r"\bignore\s+(all\s+)?previous\s+instructions\b",
    r"\bdisregard\s+(all\s+)?previous\s+instructions\b",
    r"\bforget\s+(all\s+)?previous\s+instructions\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\b",
    r"\bsystem\s*:",
    r"\bdeveloper\s*:",
    r"\badministrator\b",
    r"\broot\s+access\b",
    r"\bdelete\b.*\b(data|database|files?)\b",
    r"\bshow\b.*\b(password|token|api\s*key|secret)\b",
    r"\breveal\b.*\b(secret|system\s+prompt|instructions|token|api\s*key)\b",
    r"\bbypass\b.*\b(security|guardrails|filters?)\b",
    r"\bdisable\b.*\b(security|guardrails|filters?)\b",
    r"\bjailbreak\b",
    r"\bprompt\s+injection\b",
]

DANGEROUS_OUTPUT_PATTERNS = [
    r"<\s*script\b",
    r"</\s*script\s*>",
    r"\bjavascript\s*:",
    r"\bdata\s*:\s*text/html",
    r"\bon\w+\s*=",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bsystem\s*\(",
    r"\bimport\s+os\b",
    r"\bsubprocess\b",
    r"\bos\.system\b",
    r"\brm\s+-rf\b",
    r"\bformat\b.*\bdrive\b",
    r"\bcurl\b.*\|\s*(sh|bash)",
    r"\bwget\b.*\|\s*(sh|bash)",
]


def _normalize_text(text: str) -> str:
    """Normaliza texto para reduzir bypass com unicode, espaços e casing."""
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\x00", "")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def validate_input(text: str) -> bool:
    """Valida input do usuário contra padrões comuns de prompt injection."""
    normalized = _normalize_text(text)

    if not normalized:
        return False

    if len(normalized) > 12000:
        return False

    return not _matches_any(normalized, BLOCK_PATTERNS)


def sanitize_output(text: str) -> str:
    """Sanitiza output do LLM para reduzir risco de XSS e comandos perigosos."""
    normalized = _normalize_text(text)

    for pattern in DANGEROUS_OUTPUT_PATTERNS:
        normalized = re.sub(
            pattern,
            "[BLOCKED]",
            normalized,
            flags=re.IGNORECASE,
        )

    escaped = html.escape(normalized, quote=True)

    if len(escaped) > MAX_OUTPUT_LENGTH:
        escaped = escaped[:MAX_OUTPUT_LENGTH].rstrip() + "..."

    return escaped


def validate_output(text: str) -> bool:
    """Valida se output parece seguro para exibição."""
    normalized = _normalize_text(text)

    if not normalized:
        return True

    if len(normalized) > MAX_OUTPUT_LENGTH * 2:
        return False

    return not _matches_any(normalized, DANGEROUS_OUTPUT_PATTERNS)
