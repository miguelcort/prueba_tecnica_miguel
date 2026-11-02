"""Grounding: every answer, whoever wrote it, is checked against its source lines.

Two independent signals: technical tokens (identifiers, paths, numbers) may not be invented, and
most content words must come from the source. It catches unsupported claims and made-up
identifiers; it cannot prove that a sentence's meaning is right.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..common import normalize_text, technical_tokens

ALLOWED_TOKENS = {"http", "api", "json", "yes", "no"}
STOPWORDS = set(
    """the and for with that this from are was were has have had not but its can use uses used using when which what where into each their there than then them they
    will would should must does doing done about more most other some such only also very through documents documentation guide according service these those your you
    returned returns caller receives required action common causes solution symptom type types column columns table tables status codes code required optional
    para con que los las del por una uno unas unos como esta este estos estas tiene tienen sobre desde entre cuando donde cada debe deben puede pueden también según
    documentación servicio usa usar hace mediante tipo tabla tablas columna columnas estos obligatorio opcional código códigos estado parámetro acepta valores respuestas documentadas documentados expone sirve recibe compone documented responses exposes defines relies signals""" .split()
)
HTTP_PREFIX = re.compile(r"^(?:get|post|put|delete|patch) ")
WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)
VALIDATED, PENDING = "validated", "pending_review"


def fact_source(documents: Dict[str, dict], fact: dict, window: int = 5) -> str:
    """A fact may legitimately rest on its heading and, for specs, on the owning operation."""
    if fact["extra"].get("sources"):
        parts = []
        for source in fact["extra"]["sources"]:
            parts.append(f"{source['section'] or ''}\n{source_text(documents[source['document_id']], source['line_start'], source['line_end'], window)}")
        return "\n".join(parts)
    document = documents[fact["document_id"]]
    text = source_text(document, fact["line_start"], fact["line_end"], window)
    anchor = fact["extra"].get("anchor_lines")
    if anchor:
        text += "\n" + source_text(document, anchor[0], anchor[1], 0)
    return f"{fact['section'] or ''}\n{text}"


def source_text(document: dict, line_start: int, line_end: int, window: int = 5) -> str:
    """The cited lines with a small window, plus any diagram lines they fall into."""
    lines = document["lines"]
    start, end = max(line_start - 1 - window, 0), min(line_end + window, len(lines))
    parts = lines[start:end]
    for diagram in document.get("diagrams", []):
        if diagram["line_start"] <= line_end and line_start <= diagram["line_end"]:
            parts = parts + diagram["lines"]
    return "\n".join(parts)


def _supported(token: str, source_norm: str) -> bool:
    """`GET /x` and `table.column` are written apart in sources, so each part is checked."""
    bare = HTTP_PREFIX.sub("", token)
    return bare in source_norm or ("." in bare and "/" not in bare and all(part in source_norm for part in bare.split(".")))


def check_grounding(answer: str, source: str, key_facts: List[str], project: str, min_overlap: float, allowed: Tuple[str, ...] = ()) -> Dict[str, object]:
    reasons: List[str] = []
    source_norm = normalize_text(source)
    clean_answer = normalize_text(answer).replace(normalize_text(project), " ")
    missing_keys = [key for key in key_facts if normalize_text(key) not in source_norm]
    if missing_keys:
        reasons.append("key_fact_not_in_source: " + "; ".join(sorted(missing_keys)))
    known = ALLOWED_TOKENS | {normalize_text(item) for item in allowed}
    unsupported = [t for t in technical_tokens(answer) if not any(t in item for item in known) and not _supported(t, source_norm) and t not in normalize_text(project) and t in clean_answer]
    if unsupported:
        reasons.append("unsupported_token: " + "; ".join(unsupported))
    words = {w for w in WORD.findall(clean_answer) if w not in STOPWORDS}
    overlap = 1.0
    if words:
        overlap = round(sum(w in source_norm for w in words) / len(words), 3)
        if overlap < min_overlap:
            reasons.append(f"low_content_overlap: {overlap}")
    return {"status": PENDING if reasons else VALIDATED, "reasons": reasons, "overlap": overlap}
