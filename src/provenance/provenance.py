"""Provenance is a claim that can be checked: every field is verified against the raw files."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..common import normalize_text, sha256_file, stable_id
from ..preprocessing.normalize import normalize_lines

REQUIRED = ("source_file", "source_hash", "document_id", "section")


def _location(item: dict, duplicates: Optional[List[str]] = None) -> dict:
    row = {key: item[key] for key in ("source_file", "source_hash", "document_id", "section", "line_start", "line_end")}
    row["chunk_id"] = item.get("chunk_id")
    if duplicates:
        row["duplicate_source_files"] = duplicates
    return row


def from_fact(fact: dict, duplicates: Dict[str, List[str]]) -> dict:
    row = _location(fact, duplicates.get(fact["document_id"]))
    sources = fact["extra"].get("sources")
    if sources:
        row["sources"] = [_location(source) for source in sources]
    return row


def evidence_of(provenance: dict) -> List[dict]:
    """The passages behind a row: `sources` when there are several, else the row's own."""
    return provenance.get("sources") or [{key: value for key, value in provenance.items() if key not in ("sources", "duplicate_source_files")}]


class Verifier:
    """Caches raw files once. Every call recomputes nothing but the comparison."""

    def __init__(self, root: Path, raw_name: str):
        self.root, self.raw_name = root, raw_name
        self.hashes: Dict[str, str] = {}
        self.texts: Dict[str, str] = {}
        self.lines: Dict[str, int] = {}

    def _load(self, source_file: str) -> bool:
        if source_file in self.hashes:
            return True
        path = self.root / source_file
        if not source_file.startswith(f"{self.raw_name}/") or not path.is_file():
            return False
        self.hashes[source_file] = sha256_file(path)
        lines = normalize_lines(path.read_text("utf-8"))
        self.lines[source_file] = len(lines)
        self.texts[source_file] = normalize_text("\n".join(lines))
        return True

    def errors(self, provenance: dict) -> List[str]:
        problems: List[str] = []
        if any(key not in provenance for key in REQUIRED):
            return ["missing provenance field"]
        for item in provenance.get("sources", []) + [provenance]:
            problems.extend(self._check_one(item))
        for duplicate in provenance.get("duplicate_source_files", []):
            if not self._load(duplicate):
                problems.append(f"duplicate file not found: {duplicate}")
        return problems

    def _check_one(self, item: dict) -> List[str]:
        source = item["source_file"]
        if not self._load(source):
            return [f"raw file not found: {source}"]
        problems = []
        if self.hashes[source] != item["source_hash"]:
            problems.append(f"hash mismatch: {source}")
        if stable_id("doc", source.split("/", 1)[1]) != item["document_id"]:
            problems.append(f"document_id does not match path: {source}")
        start, end = item.get("line_start"), item.get("line_end")
        if start is None or end is None or not 1 <= start <= end <= self.lines[source]:
            problems.append(f"line range outside file: {source}:{start}-{end}")
        section = item.get("section")
        if section and not all(token in self.texts[source] for token in normalize_text(section.replace(" > ", " ")).split()):
            problems.append(f"section not in file: {source}: {section}")
        return problems

    def text_contains(self, provenance: dict, key: str) -> bool:
        """A key fact must exist in the raw text of one of the row's own sources."""
        needle = normalize_text(key)
        return any(self._load(item["source_file"]) and needle in self.texts[item["source_file"]] for item in provenance.get("sources", []) + [provenance])
