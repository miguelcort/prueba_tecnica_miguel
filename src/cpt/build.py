"""Stage 3, CPT: continuous domain text. No questions, nothing invented, every row traceable."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterator, Optional

from ..common import Config, iter_jsonl, normalize_text, require, write_json, write_jsonl

MIN_TEXT_CHARS = 40
JUNK = re.compile(r"```mermaid|^\s*(?:graph|flowchart)\s+(?:TD|LR)\b|sequenceDiagram|erDiagram|style \w+ fill", re.MULTILINE)


def rejection(chunk: dict) -> Optional[str]:
    text = chunk["text"]
    if not text.strip():
        return "empty"
    if len(text) < MIN_TEXT_CHARS:
        return "too_short"
    if text.count("```") % 2:
        return "malformed_fence"
    if JUNK.search(text):
        return "diagram_junk"
    if not chunk["source_file"] or not chunk["source_hash"]:
        return "missing_provenance"
    return None


def to_row(chunk: dict, number: int) -> dict:
    """The context line keeps a chunk meaningful once it is separated from its page."""
    section = f" / {chunk['section'].replace('erDiagram > ', 'table ')}" if chunk["section"] else ""
    return {
        "id": f"cpt_{number:06d}",
        "text": f"{chunk['project']}{section}\n\n{chunk['text']}",
        "metadata": {"project": chunk["project"], "version": chunk["version"], "derived_from": chunk["derived_from"]},
        "provenance": {key: chunk[key] for key in ("source_file", "source_hash", "document_id", "section", "chunk_id", "line_start", "line_end")},
    }


def run(config: Config) -> dict:
    chunks_path = config.processed("chunks.jsonl")
    require(chunks_path)
    rejected: Counter = Counter()
    seen = set()

    def rows() -> Iterator[dict]:
        for chunk in iter_jsonl(chunks_path):
            if chunk["usage"] != "cpt":
                continue
            reason = rejection(chunk)
            key = normalize_text(chunk["text"])
            if not reason and key in seen:
                reason = "duplicate_text"
            if reason:
                rejected[reason] += 1
                continue
            seen.add(key)
            yield to_row(chunk, len(seen))

    total = write_jsonl(config.output("cpt"), rows())
    stats = {"rows": total, "rejected": dict(sorted(rejected.items()))}
    write_json(config.processed("cpt_stats.json"), stats)
    return stats
