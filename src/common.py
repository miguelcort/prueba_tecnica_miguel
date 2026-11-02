"""Helpers shared by every stage: configuration, hashing, streaming JSONL, logging."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import yaml

ROOT_MARKER = "configs"


class PipelineError(RuntimeError):
    """A stage cannot run or a validation failed. The CLI turns this into a non-zero exit."""


class Config(dict):
    """The parsed YAML plus absolute paths, resolved from the project root."""

    def path(self, name: str) -> Path:
        return self["root"] / self["paths"][name]

    def processed(self, name: str) -> Path:
        return self.path("processed_dir") / name

    def output(self, name: str) -> Path:
        return self.path("outputs_dir") / f"{name}.jsonl"


def load_config(config_path: str) -> Config:
    path = Path(config_path).resolve()
    if not path.exists():
        raise PipelineError(f"config file not found: {config_path}")
    with path.open(encoding="utf-8") as handle:
        config = Config(yaml.safe_load(handle))
    config["root"] = path.parent.parent
    return config


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    """Identifiers derive from content, so they survive reruns and incremental batches."""
    joined = "|".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]}"


def rank(seed: int, *parts: object) -> str:
    """A seeded, order-independent sort key used for every sampling decision."""
    return stable_id("r", seed, *parts)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def require(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise PipelineError("missing inputs, run the earlier stages first: " + ", ".join(missing))


def normalize_text(text: str) -> str:
    """Comparison form: ignores case, Markdown emphasis and whitespace."""
    return " ".join(text.replace("`", "").replace("*", "").replace("\\", "").lower().split())


def normalize_question(text: str) -> str:
    return "".join(ch for ch in normalize_text(text) if ch.isalnum() or ch == " ")


class JsonLogger:
    """Structured logs. Timestamps live here only, never in datasets or reports."""

    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "pipeline.jsonl"

    def log(self, event: str, **fields: object) -> None:
        record: Dict[str, object] = {"ts": round(time.time(), 3), "event": event}
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line)


def token_jaccard(left: str, right: str) -> float:
    a, b = set(normalize_question(left).split()), set(normalize_question(right).split())
    return len(a & b) / len(a | b) if a | b else 0.0


def same_question(left: str, right: str, threshold: float) -> bool:
    """Templates share words on purpose, so high overlap only counts with the same identifiers."""
    return normalize_question(left) == normalize_question(right) or (token_jaccard(left, right) >= threshold and technical_tokens(left) == technical_tokens(right))


TECH_TOKEN = re.compile(
    r"`[^`\n]+`"
    r"|\b(?:GET|POST|PUT|DELETE|PATCH)\s+/\S*"
    r"|(?<![\w.])/[A-Za-z0-9_{}\-./]+"
    r"|\b[a-z]+_[a-z0-9_]+\b"
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z]{2,}[A-Z0-9_]*\b"
    r"|\b\d[\d.,%]*\w*"
)


def technical_tokens(text: str) -> List[str]:
    """Identifiers, paths and numbers: what must never change silently or be invented."""
    return sorted({normalize_text(match.group(0)).strip(".,:;") for match in TECH_TOKEN.finditer(text)} - {""})
