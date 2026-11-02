"""Which documents stay out of training. Decided once, before any dataset exists."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from ..common import rank


def subpath(relative_path: str) -> str:
    """The path inside a version folder, used to find other versions of the same page."""
    return "/".join(relative_path.split("/")[2:])


def select_holdout(documents: List[dict], facts: List[dict], settings: dict, seed: int) -> dict:
    """Per project, hold out one document, never its richest one, so training keeps coverage.

    Projects with too few documents keep everything: removing their only substantial page would
    teach nothing about them and make those facts unanswerable for every closed-book arm.
    """
    fact_counts = Counter(fact["document_id"] for fact in facts)
    per_project: Dict[str, List[dict]] = defaultdict(list)
    for doc in documents:
        if doc["is_canonical"] and fact_counts[doc["document_id"]] >= settings["min_document_facts"]:
            per_project[doc["project"]].append(doc)
    selected, skipped = [], {}
    for project in sorted({doc["project"] for doc in documents}):
        candidates = per_project.get(project, [])
        if len(candidates) < settings["min_project_documents"]:
            skipped[project] = f"{len(candidates)} documents with at least {settings['min_document_facts']} facts"
            continue
        richest = max(candidates, key=lambda doc: (fact_counts[doc["document_id"]], doc["relative_path"]))
        options = sorted((doc for doc in candidates if doc is not richest), key=lambda doc: rank(seed, "holdout", doc["document_id"]))
        selected.append(options[0])
    held_paths = {(doc["project"], subpath(doc["relative_path"])) for doc in selected}
    variants = [doc["document_id"] for doc in documents if not doc["is_canonical"] and (doc["project"], subpath(doc["relative_path"])) in held_paths]
    return {
        "document_ids": sorted(doc["document_id"] for doc in selected),
        "source_files": sorted(doc["source_file"] for doc in selected),
        "variant_document_ids": sorted(variants),
        "projects_without_holdout": skipped,
    }
