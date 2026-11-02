"""Stage 2, curation: from the raw inventory to clean documents, facts, holdout and chunks."""
from __future__ import annotations

import hashlib
from collections import Counter
from functools import partial
from multiprocessing import Pool
from typing import Dict, Iterable, Iterator, List

from ..chunking.chunker import chunk_document
from ..common import Config, iter_jsonl, load_yaml, normalize_text, read_json, require, technical_tokens, write_json, write_jsonl
from ..deduplication.dedup import lsh_candidates, minhash, similarity, technical_difference, threshold_sweep
from ..evaluation.holdout import select_holdout
from ..filtering.filters import incomplete_markers, rejection_reason
from ..sft.facts import extract_all
from .extract import structure_summary
from .normalize import apply_quarantine, normalize_lines, redact, strip_blocks

KEPT, VARIANT = "kept", "version_variant"


def curate_one(row: dict, rules: dict, settings: dict, dedup: dict, seed: int) -> dict:
    """Pure per-document work. It takes no shared state, so it can run in any worker."""
    counts: Counter = Counter()
    result = {k: row[k] for k in ("document_id", "source_file", "relative_path", "source_hash", "project", "version_folder", "version", "is_canonical", "role", "file_size")}
    result["version"] = row["version"] or row["version_folder"]
    normalized = normalize_lines(row["content"]) if row["content"] else []
    result["content_hash"] = hashlib.sha256("\n".join(normalized).strip().encode("utf-8")).hexdigest()
    excluded = {entry["path"]: entry["reason"] for entry in rules.get("excluded_files", [])}.get(row["relative_path"])
    cleaning_applies = row["ingestion_status"] == "success" and row["role"] in ("markdown", "spec") and not excluded
    lines: List[str] = normalized
    diagrams: List[dict] = []
    quarantined: List[dict] = []
    if cleaning_applies and row["role"] == "spec":
        lines = [redact(line, rules, counts) for line in normalized]
    elif cleaning_applies:
        kept, quarantined = apply_quarantine(row["relative_path"], normalized, rules)
        stripped, diagrams = strip_blocks(kept, normalized, settings["max_fence_lines"], counts)
        lines = [redact(line, rules, counts) for line in stripped]
        for diagram in diagrams:
            diagram["lines"] = [redact(line, rules, counts) for line in diagram["lines"]]
    reason = rejection_reason(row["role"], row["ingestion_status"], excluded, lines, diagrams, settings)
    result.update(
        {
            "title": redact(row["title"], rules, counts) if row["title"] else None,
            "lines": None if reason in ("excluded_by_config",) else lines,
            "diagrams": diagrams,
            "rejection_reason": reason,
            "incomplete_markers": incomplete_markers(normalized),
            "quarantined": quarantined,
            "redactions": dict(counts),
            "signature": None if reason else minhash("\n".join(lines) + "\n".join(l for d in diagrams for l in d["lines"]), dedup["shingle_words"], dedup["num_perm"], seed),
        }
    )
    result.update(structure_summary(lines) if not reason else {"section_count": 0, "code_block_count": 0, "table_count": 0, "endpoints": [], "parameters": []})
    return result


def map_documents(rows: Iterable[dict], function, workers: int, batch_size: int) -> Iterator[dict]:
    if workers <= 1:
        yield from map(function, rows)
        return
    with Pool(workers) as pool:
        yield from pool.imap(function, rows, chunksize=batch_size)


def resolve_duplicates(meta: List[dict], settings: dict) -> dict:
    """Mutates status in `meta` and returns the deduplication report."""
    alive = [m for m in meta if not m["rejection_reason"]]
    exact_groups: Dict[tuple, List[dict]] = {}
    for doc in alive:
        exact_groups.setdefault(doc["content_hash"], []).append(doc)
    exact_report = []
    for content_hash in sorted(exact_groups):
        copies = sorted(exact_groups[content_hash], key=lambda d: (not d["is_canonical"], d["relative_path"]))
        if len(copies) == 1:
            continue
        keeper, rest = copies[0], copies[1:]
        same_project = [c for c in rest if c["project"] == keeper["project"]]
        for other in same_project:
            other["rejection_reason"] = "exact_duplicate"
            other["duplicate_of"] = keeper["document_id"]
            keeper["duplicate_files"].append(other["source_file"])
        exact_report.append({"content_hash": content_hash, "kept": keeper["source_file"], "removed": [c["source_file"] for c in same_project], "cross_project": [c["source_file"] for c in rest if c not in same_project]})

    alive = {m["document_id"]: m for m in meta if not m["rejection_reason"]}
    signatures = {key: doc["signature"] for key, doc in alive.items()}
    threshold = settings["near_duplicate_threshold"]
    pairs = []
    for left, right in lsh_candidates(signatures, settings["lsh_bands"]):
        score = similarity(signatures[left], signatures[right])
        if score < threshold:
            continue
        a, b = sorted((alive[left], alive[right]), key=lambda d: (not d["is_canonical"], d["relative_path"]))
        difference = technical_difference("\n".join(a["lines_text"]), "\n".join(b["lines_text"]))
        different = bool(difference["only_in_first"] or difference["only_in_second"])
        same_project = a["project"] == b["project"]
        action = "report_only"
        if same_project and not b["is_canonical"]:
            action = "flagged_version_difference" if different else "removed_near_duplicate"
            if not different:
                b["rejection_reason"] = "near_duplicate"
                b["duplicate_of"] = a["document_id"]
                a["duplicate_files"].append(b["source_file"])
        pairs.append({"first": a["source_file"], "second": b["source_file"], "estimated_jaccard": score, "action": action, "technical_difference": difference})
    projects = {key: doc["project"] for key, doc in alive.items()}
    return {
        "method": "sha256 for exact; MinHash with LSH banding for near duplicates; technical-token diff before any removal",
        "near_duplicate_threshold": threshold,
        "threshold_sweep_same_project_pairs": threshold_sweep(signatures, projects, settings["sweep_thresholds"]),
        "exact_duplicate_groups": exact_report,
        "near_duplicate_pairs": sorted(pairs, key=lambda p: (p["first"], p["second"])),
    }


def _chunk_terms(rules: dict) -> List[str]:
    return list(rules.get("spec_parameter_exclusions", []))


def dedupe_chunks(chunks: List[dict], settings: dict, seed: int) -> Counter:
    """Boilerplate repeated inside a project, and anything echoing a held-out document."""
    counts: Counter = Counter()
    seen: Dict[str, str] = {}
    signature = {c["chunk_id"]: minhash(c["text"], 3, settings["num_perm"], seed) for c in chunks}
    held = [c for c in chunks if c["usage"] == "document_holdout"]
    for chunk in chunks:
        if chunk["usage"] != "cpt":
            continue
        key = normalize_text(chunk["text"])
        if key in seen:
            chunk["usage"] = "duplicate_chunk"
            counts["exact_duplicate_chunk"] += 1
            continue
        seen[key] = chunk["chunk_id"]
        own = signature[chunk["chunk_id"]]
        if any(similarity(own, signature[other["chunk_id"]]) >= settings["near_duplicate_threshold"] for other in held):
            chunk["usage"] = "holdout_near_duplicate"
            counts["near_duplicate_of_holdout"] += 1
    return counts


def dedupe_near_chunks(chunks: List[dict], settings: dict, seed: int) -> List[dict]:
    """Near-identical training chunks leave only if they carry the same technical tokens."""
    training = {c["chunk_id"]: c for c in chunks if c["usage"] == "cpt"}
    signatures = {key: minhash(c["text"], 3, settings["num_perm"], seed) for key, c in training.items()}
    pairs = []
    for left, right in lsh_candidates(signatures, settings["lsh_bands"]):
        score = similarity(signatures[left], signatures[right])
        if score < settings["chunk_near_duplicate_threshold"]:
            continue
        first, second = sorted((training[left], training[right]), key=lambda c: (c["source_file"], c["line_start"], c["chunk_id"]))
        difference = technical_difference(first["text"], second["text"])
        protected = bool(difference["only_in_first"] or difference["only_in_second"])
        if not protected and second["usage"] == "cpt":
            second["usage"] = "duplicate_chunk"
        pairs.append({"first": first["chunk_id"], "second": second["chunk_id"], "estimated_jaccard": score, "action": "kept_technical_difference" if protected else "removed_second", "technical_difference": difference})
    return pairs


def repeated_fragments(chunks: List[dict], min_chars: int) -> List[dict]:
    """Paragraphs that repeat across training chunks. Reported; a chunk still needs its context."""
    seen: Dict[str, dict] = {}
    for chunk in chunks:
        if chunk["usage"] != "cpt":
            continue
        for paragraph in chunk["text"].split("\n\n"):
            key = normalize_text(paragraph)
            if len(key) >= min_chars:
                entry = seen.setdefault(key, {"preview": key[:80], "chunks": set(), "source_files": set()})
                entry["chunks"].add(chunk["chunk_id"])
                entry["source_files"].add(chunk["source_file"])
    repeated = [e for e in seen.values() if len(e["chunks"]) > 1]
    return sorted(({"preview": e["preview"], "occurrences": len(e["chunks"]), "source_files": sorted(e["source_files"]), "action": "reported_kept"} for e in repeated), key=lambda e: e["preview"])


def link_facts(facts: List[dict], chunks: List[dict]) -> int:
    """Attach each fact to the most specific chunk covering its lines."""
    by_document: Dict[str, List[dict]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk["document_id"], []).append(chunk)
    missing = 0
    for fact in facts:
        for source in fact["extra"].get("sources", []):
            around = [c for c in by_document.get(source["document_id"], []) if c["line_start"] <= source["line_end"] and source["line_start"] <= c["line_end"]]
            around.sort(key=lambda c: (c["usage"] != "cpt", c["char_count"], c["chunk_id"]))
            source["chunk_id"] = around[0]["chunk_id"] if around else None
        start, end = fact["extra"].get("anchor_lines") or (fact["line_start"], fact["line_end"])
        own = by_document.get(fact["document_id"], [])
        overlapping = [c for c in own if c["line_start"] <= end and start <= c["line_end"]]
        if not overlapping:  # a shared spec component belongs to whichever operation states it
            overlapping = [c for c in own if _keys_inside(fact, c) == len(fact["key_facts"])]
        overlapping.sort(key=lambda c: (-_keys_inside(fact, c), c["char_count"], c["chunk_id"]))
        fact["chunk_id"] = overlapping[0]["chunk_id"] if overlapping else None
        missing += fact["chunk_id"] is None
    return missing


def _keys_inside(fact: dict, chunk: dict) -> int:
    text = normalize_text(chunk["text"])
    return sum(normalize_text(key) in text for key in fact["key_facts"])


def run(config: Config) -> dict:
    ingested = config.processed("ingested_documents.jsonl")
    require(ingested)
    rules = load_yaml(config.path("quarantine"))
    curated = load_yaml(config.path("curated_facts"))
    seed, runtime = config["seed"], config["runtime"]
    work = partial(curate_one, rules=rules, settings=config["curation"], dedup=config["deduplication"], seed=seed)

    # Signatures and metadata stay in memory; a bigger corpus would move them to a key-value store.
    meta: List[dict] = []
    for result in map_documents(iter_jsonl(ingested), work, runtime["workers"], runtime["batch_size"]):
        result["lines_text"] = result["lines"] or []
        result["duplicate_files"] = []
        result["duplicate_of"] = None
        meta.append(result)

    dedup_report = resolve_duplicates(meta, config["deduplication"])
    kept = [m for m in meta if not m["rejection_reason"]]
    for doc in kept:
        doc["status"] = KEPT if doc["is_canonical"] else VARIANT

    public_fields = ("document_id", "source_file", "relative_path", "source_hash", "content_hash", "project", "version", "version_folder", "is_canonical", "role", "title", "status", "duplicate_files", "section_count", "code_block_count", "table_count", "endpoints", "parameters", "incomplete_markers", "file_size", "diagrams", "lines")
    write_jsonl(config.processed("documents_cleaned.jsonl"), ({**{k: doc[k] for k in public_fields}, "content": "\n".join(doc["lines"])} for doc in kept))
    write_jsonl(
        config.processed("documents_rejected.jsonl"),
        ({k: m[k] for k in ("document_id", "source_file", "source_hash", "project", "version", "role", "rejection_reason", "duplicate_of", "file_size", "incomplete_markers")} for m in meta if m["rejection_reason"]),
    )

    facts, conflicts, rejected = extract_all(kept, rules, curated, config["language"]["spanish_projects"])
    holdout = select_holdout(kept, facts, config["holdout"], seed)
    held = set(holdout["document_ids"])
    held_variants = set(holdout["variant_document_ids"])

    chunks: List[dict] = []
    for doc in kept:
        usage = "document_holdout" if doc["document_id"] in held else "holdout_variant" if doc["document_id"] in held_variants else "cpt" if doc["is_canonical"] else "version_variant"
        document = {**doc, "raw_lines": doc["lines"]}
        chunks.extend(chunk_document(document, config["chunking"], usage, _chunk_terms(rules)))
    excluded_patterns = [normalize_text(p) for p in rules.get("chunk_exclusions", [])]
    usable = []
    removed = Counter()
    for chunk in chunks:
        if any(p in normalize_text(chunk["text"]) for p in excluded_patterns):
            removed["excluded_by_config"] += 1
            continue
        if chunk["section"] and chunk["section"].split(" > ")[-1].lower() in config["chunking"]["boilerplate_sections"]:
            removed["boilerplate_section"] += 1
            continue
        if chunk["char_count"] < config["chunking"]["min_context_chars"] and len(technical_tokens(chunk["text"])) < 2:
            removed["insufficient_context"] += 1
            continue
        usable.append(chunk)
    chunks = sorted(usable, key=lambda c: (c["source_file"], c["line_start"], c["chunk_id"]))
    removed.update(dedupe_chunks(chunks, config["deduplication"], seed))
    dedup_report["near_duplicate_chunks"] = dedupe_near_chunks(chunks, config["deduplication"], seed)
    removed["near_duplicate_chunk"] = sum(pair["action"] == "removed_second" for pair in dedup_report["near_duplicate_chunks"])
    dedup_report["repeated_fragments"] = repeated_fragments(chunks, config["deduplication"]["fragment_min_chars"])
    unlinked = link_facts(facts, chunks)

    write_jsonl(config.processed("chunks.jsonl"), chunks)
    write_jsonl(config.processed("facts.jsonl"), facts)
    write_json(config.processed("holdout.json"), holdout)
    write_json(config.path("reports_dir") / "deduplication_report.json", dedup_report)

    redactions: Counter = Counter()
    for m in meta:
        redactions.update(m["redactions"])
    stats = {
        "documents_input": len(meta),
        "documents_kept": sum(m.get("status") == KEPT for m in meta),
        "documents_version_variant": sum(m.get("status") == VARIANT for m in meta),
        "documents_rejected": dict(sorted(Counter(m["rejection_reason"] for m in meta if m["rejection_reason"]).items())),
        "documents_with_todo_markers": sum(m["incomplete_markers"] > 0 for m in kept),
        "redactions": dict(sorted((k, v) for k, v in redactions.items() if v)),
        "quarantined_lines": sorted((q for m in meta for q in m["quarantined"]), key=lambda q: (q["path"], q["line"])),
        "fact_conflicts": conflicts,
        "facts": len(facts),
        "facts_without_chunk": unlinked,
        "fact_rejections": sorted(rejected, key=lambda r: (r["reason"], r["id"])),
        "chunks": len(chunks),
        "chunks_by_usage": dict(sorted(Counter(c["usage"] for c in chunks).items())),
        "chunks_removed": dict(sorted(removed.items())),
    }
    write_json(config.processed("curation_stats.json"), stats)
    return {k: stats[k] for k in ("documents_input", "documents_kept", "documents_version_variant", "documents_rejected", "facts", "chunks", "chunks_by_usage")}
