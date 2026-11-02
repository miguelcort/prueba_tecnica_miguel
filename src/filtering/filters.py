"""Document filters. A short page is not a useless page: brevity alone never rejects."""
from __future__ import annotations

import re
from typing import List, Optional

from ..common import technical_tokens


def substantive_chars(lines: List[str]) -> int:
    return sum(len(re.sub(r"\W", "", line)) for line in lines if line.strip() and not line.lstrip().startswith("#"))


def has_technical_signal(lines: List[str]) -> bool:
    body = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
    text = "\n".join(body)
    return "```" in text or text.lstrip().startswith("|") or "\n|" in text or len(technical_tokens(text)) >= 3


def is_repetitive(lines: List[str], min_ratio: float) -> bool:
    body = [line.strip() for line in lines if line.strip()]
    return len(body) > 20 and len(set(body)) / len(body) < min_ratio


def rejection_reason(role: str, status: str, excluded: Optional[str], lines: List[str], diagrams: List[dict], settings: dict) -> Optional[str]:
    """The first applicable reason, or None if the document is kept."""
    if status != "success":
        return status
    if role in ("navigation", "manifest"):
        return role
    if role == "other":
        return "unsupported_format"
    if excluded:
        return "excluded_by_config"
    if role == "spec":
        return None
    if not any(line.strip() for line in lines) and not diagrams:
        return "empty_after_cleaning"
    if is_repetitive(lines, settings["min_unique_line_ratio"]):
        return "repetitive"
    if substantive_chars(lines) < settings["min_substantive_chars"] and not diagrams and not has_technical_signal(lines):
        return "stub_no_technical_content"
    return None


def incomplete_markers(raw_lines: List[str]) -> int:
    return sum(bool(re.search(r"\bTO\s?DO\b", line, re.IGNORECASE)) for line in raw_lines)
