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


PII_PATTERNS = [
    # E-mail
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    # CPF: 000.000.000-00 ou 00000000000
    r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
    # CNPJ: 00.000.000/0000-00 ou 00000000000000
    r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
    # Telefone BR simples
    r"\b(?:\+55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\s?)?\d{4}-?\d{4}\b",
    # Cartão de crédito básico
    r"\b(?:\d[ -]*?){13,19}\b",
    # Chaves comuns
    r"\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{16,}\b",
]


def _has_pii(text: str) -> bool:
    """Detecta possível vazamento de PII ou credenciais."""
    normalized = _normalize_text(text)

    for pattern in PII_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return True

    return False


def _looks_like_credit_card(text: str) -> bool:
    """Valida possível cartão usando Luhn para reduzir falso positivo."""
    digits = re.sub(r"\D", "", text)

    if not 13 <= len(digits) <= 19:
        return False

    total = 0
    reverse_digits = digits[::-1]

    for i, digit in enumerate(reverse_digits):
        n = int(digit)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n

    return total % 10 == 0


def _has_credit_card(text: str) -> bool:
    candidates = re.findall(r"\b(?:\d[ -]*?){13,19}\b", text)
    return any(_looks_like_credit_card(candidate) for candidate in candidates)


def validate_output(text: str) -> bool:
    """Valida se output é seguro para exibição e não vaza PII."""
    normalized = _normalize_text(text)

    if not normalized:
        return True

    if len(normalized) > MAX_OUTPUT_LENGTH * 2:
        return False

    if _matches_any(normalized, DANGEROUS_OUTPUT_PATTERNS):
        return False

    if _has_pii(normalized):
        return False

    if _has_credit_card(normalized):
        return False

    return True
