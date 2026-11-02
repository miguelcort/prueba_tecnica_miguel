"""Stage 1, ingestion: inventory the corpus without judging it.

Every file gets a row, including the ones that cannot be read, so no input is ever lost
silently. Decisions about quality belong to curation.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, Optional

import yaml

from ..common import Config, PipelineError, sha256_bytes, sha256_file, stable_id, write_json, write_jsonl

NAV_FILES = {"_sidebar.md", "_coverpage.md", "coverpage.md"}
MANIFEST_FILE = "documentationTypes.json"
H1 = re.compile(r"^#\s*(.+?)\s*#*\s*$")


def ensure_raw_documents(config: Config) -> Path:
    """The originals are read, never written."""
    raw_dir = config.path("raw_dir")
    if not raw_dir.exists():
        raise PipelineError(f"no raw documents at {raw_dir}")
    return raw_dir


def tree_digest(raw_dir: Path) -> str:
    """One hash over every original, used to prove that nothing changed them."""
    digest = hashlib.sha256()
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        digest.update(path.relative_to(raw_dir).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def classify_role(relative_path: str) -> str:
    name = relative_path.rsplit("/", 1)[-1]
    if name == MANIFEST_FILE:
        return "manifest"
    if name in NAV_FILES:
        return "navigation"
    if "/specs/" in relative_path and name.endswith((".yaml", ".yml")):
        return "spec"
    if name.endswith(".md"):
        return "markdown"
    return "other"


def manifest_version(manifest_text: str, project: str) -> Optional[str]:
    """The folder `latest` hides a real version; the manifest URLs reveal it."""
    for url in sorted(json.loads(manifest_text).values()):
        match = re.search(re.escape(project) + r"/([^/]+)/(?:guide|specs)/", url)
        if match:
            return match.group(1)
    return None


def resolve_canonical_folder(project: str, folders: list) -> str:
    """Trust the publisher's `latest` pointer; never infer recency from names or mtimes."""
    if "latest" in folders:
        return "latest"
    if len(folders) == 1:
        return folders[0]
    raise PipelineError(f"{project}: no `latest` and several versions: {sorted(folders)}")


def extract_title(role: str, text: str) -> Optional[str]:
    if role == "spec":
        try:
            info = (yaml.safe_load(text) or {}).get("info") or {}
        except yaml.YAMLError:
            return None
        return info.get("title")
    if role == "markdown":
        for line in text.split("\n"):
            match = H1.match(line)
            if match and not line.startswith("##"):
                return html.unescape(match.group(1)).replace("**", "").strip() or None
    return None


def iter_documents(raw_dir: Path) -> Iterator[dict]:
    paths = sorted(p for p in raw_dir.rglob("*") if p.is_file() and not p.name.startswith("."))
    folders: Dict[str, set] = defaultdict(set)
    versions: Dict[tuple, Optional[str]] = {}
    for path in paths:
        parts = path.relative_to(raw_dir).parts
        folders[parts[0]].add(parts[1])
        if path.name == MANIFEST_FILE:
            versions[parts[:2]] = manifest_version(path.read_text("utf-8"), parts[0])
    canonical = {project: resolve_canonical_folder(project, sorted(found)) for project, found in folders.items()}

    prefix = raw_dir.name
    for path in paths:
        relative = path.relative_to(raw_dir).as_posix()
        project, folder = relative.split("/")[:2]
        data = path.read_bytes()
        role = classify_role(relative)
        status, content = "success", None
        try:
            content = data.decode("utf-8")
            if not content.strip():
                status = "empty"
        except UnicodeDecodeError:
            status = "unreadable"
        yield {
            "document_id": stable_id("doc", relative),
            "source_file": f"{prefix}/{relative}",
            "relative_path": relative,
            "source_hash": sha256_bytes(data),
            "project": project,
            "version_folder": folder,
            "version": versions.get((project, folder)),
            "is_canonical": folder == canonical[project],
            "role": role,
            "title": extract_title(role, content) if content else None,
            "content": content,
            "file_size": len(data),
            "ingestion_status": status,
        }


def run(config: Config) -> dict:
    raw_dir = ensure_raw_documents(config)
    stats: Counter = Counter()
    by_project: Counter = Counter()

    def tracked() -> Iterator[dict]:
        for row in iter_documents(raw_dir):
            stats[f"status_{row['ingestion_status']}"] += 1
            stats[f"role_{row['role']}"] += 1
            by_project[row["project"]] += 1
            yield row

    total = write_jsonl(config.processed("ingested_documents.jsonl"), tracked())
    if not total:
        raise PipelineError(f"no files under {raw_dir}")
    summary = {"files": total, "docs_raw_tree_sha256": tree_digest(raw_dir), **dict(sorted(stats.items())), "files_by_project": dict(sorted(by_project.items()))}
    write_json(config.processed("ingestion_stats.json"), summary)
    return summary
