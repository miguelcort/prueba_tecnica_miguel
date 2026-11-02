"""Stage 4, SFT: instruction-response pairs, kept only if they survive verification."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, List

from ..common import Config, iter_jsonl, load_yaml, normalize_question, normalize_text, rank, read_json, require, stable_id, write_json, write_jsonl
from ..preprocessing.extract import section_paths
from ..preprocessing.normalize import normalize_lines
from ..provenance.provenance import from_fact
from .abstentions import impostor_abstentions, stub_abstentions
from .grounding import PENDING, VALIDATED, check_grounding, fact_source
from .llm import generate_candidates, provider_from_env
from .templates import difficulty_of, render

CATEGORIES = ("definition", "services", "api_functionality", "parameters", "configuration", "dependencies", "architecture", "processing_flow", "error_handling", "limitations", "comparison")
AGGREGATE_KINDS = ("endpoint_list", "table_columns")
DANGLING_START = re.compile(r"^\W*(it|this|that|esto|eso|ese|esta)\b", re.IGNORECASE)


def is_ambiguous(question: str, project: str) -> bool:
    return project.lower() not in question.lower() or bool(DANGLING_START.match(question))


def _chunk_ids(fact: dict) -> List[str]:
    sources = fact["extra"].get("sources")
    return [s["chunk_id"] for s in sources] if sources else [fact.get("chunk_id")]


def _supported_by_cpt(fact: dict, cpt_text: Dict[str, str]) -> bool:
    text = " ".join(cpt_text.get(chunk or "", "") for chunk in _chunk_ids(fact))
    return bool(text.strip()) and all(normalize_text(key) in text for key in fact["key_facts"])


def _documents_of(fact: dict) -> List[str]:
    sources = fact["extra"].get("sources")
    return [s["document_id"] for s in sources] if sources else [fact["document_id"]]


def assign_splits(facts: List[dict], cpt_text: Dict[str, str], holdout: dict, settings: dict, seed: int) -> Dict[str, dict]:
    """Stratified by project and category, whole facts at a time, by seeded hash."""
    held_docs, variants = set(holdout["document_ids"]), set(holdout["variant_document_ids"])
    split = {}
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    reasoning: Dict[str, List[dict]] = defaultdict(list)
    for fact in facts:
        documents = _documents_of(fact)
        if set(documents) & variants:
            split[fact["fact_id"]] = {"split": "excluded_holdout_variant", "recall_probe": False}
        elif set(documents) & held_docs:
            split[fact["fact_id"]] = {"split": "document_holdout", "recall_probe": False}
        elif fact["kind"] == "multi_source":
            reasoning[fact["project"]].append(fact)
        else:
            groups[(fact["project"], fact["category"])].append(fact)
    for project in sorted(reasoning):
        ordered = sorted(reasoning[project], key=lambda f: rank(seed, "reasoning", f["fact_id"]))
        for index, fact in enumerate(ordered):
            evaluated = (index + 1) % settings["multi_source_eval_share"] == 0 and _supported_by_cpt(fact, cpt_text) and fact["version_scope"] == "canonical"
            split[fact["fact_id"]] = {"split": "fact_holdout" if evaluated else "sft", "recall_probe": False}
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda f: rank(seed, "split", f["fact_id"]))
        eligible = [f for f in ordered if _supported_by_cpt(f, cpt_text) and f["version_scope"] == "canonical" and f["kind"] not in AGGREGATE_KINDS]
        held_ids = {f["fact_id"] for f in eligible[: int(len(ordered) * settings["fact_holdout_fraction"] + 0.5)]}
        training = [f for f in ordered if f["fact_id"] not in held_ids]
        probes = int(len(training) * settings["recall_probe_fraction"] + 0.5)
        probe_ids = {f["fact_id"] for f in training[:probes]} | {f["fact_id"] for f in training if f["version_scope"] != "canonical"}
        for fact in ordered:
            identifier = fact["fact_id"]
            split[identifier] = {"split": "fact_holdout" if identifier in held_ids else "sft", "recall_probe": identifier in probe_ids}
    return split


def load_all_versions(config: Config) -> List[dict]:
    """Normalized text of every ingested document, needed to verify absence and stub quotes."""
    rows = []
    for row in iter_jsonl(config.processed("ingested_documents.jsonl")):
        if row["content"] and row["role"] in ("markdown", "spec"):
            lines = normalize_lines(row["content"])
            rows.append({**{k: row[k] for k in ("document_id", "project", "source_file", "source_hash", "relative_path", "role")}, "normalized_lines": lines, "sections": section_paths(lines) if row["role"] == "markdown" else None})
    return rows


def abstention_provenance(row: dict) -> dict:
    return {**{k: row[k] for k in ("source_file", "source_hash", "document_id", "section", "line_start", "line_end")}, "chunk_id": None, "relation": row["reason"]}


def abstention_difficulty(row: dict) -> str:
    """A real term of another service is harder to refuse than an empty section."""
    return "hard" if row["reason"] == "absent_from_target_service" else "medium"


def run(config: Config) -> dict:
    facts_path, chunks_path, holdout_path = config.processed("facts.jsonl"), config.processed("chunks.jsonl"), config.processed("holdout.json")
    require(facts_path, chunks_path, holdout_path, config.processed("documents_cleaned.jsonl"), config.output("cpt"))
    settings, seed = config["sft"], config["seed"]
    facts = list(iter_jsonl(facts_path))
    holdout = read_json(holdout_path)
    documents = {doc["document_id"]: doc for doc in iter_jsonl(config.processed("documents_cleaned.jsonl"))}
    duplicates = {key: doc["duplicate_files"] for key, doc in documents.items()}
    cpt_text = {row["provenance"]["chunk_id"]: normalize_text(row["text"]) for row in iter_jsonl(config.output("cpt"))}
    split = assign_splits(facts, cpt_text, holdout, settings, seed)

    # Abstentions: half for teaching the behaviour, half unseen, chosen by seeded hash.
    all_versions = load_all_versions(config)
    by_path = {doc["relative_path"]: doc for doc in all_versions}
    spanish = set(config["language"]["spanish_projects"])
    language = {doc["project"]: "es" if doc["project"] in spanish else "en" for doc in all_versions}
    curated = load_yaml(config.path("curated_facts"))
    abstentions = stub_abstentions(curated["abstentions"], by_path, language)
    abstentions += impostor_abstentions(facts, all_versions, language, set(holdout["document_ids"]), seed, settings["impostors_per_project"])
    per_project: Dict[str, List[dict]] = defaultdict(list)
    for row in abstentions:
        per_project[row["project"]].append(row)
    for project in sorted(per_project):
        ordered = sorted(per_project[project], key=lambda r: rank(seed, "abstain", r["abstention_id"]))
        for index, row in enumerate(ordered):
            row["split"] = "sft" if index % 2 == 0 else "evaluation"
    abstentions.sort(key=lambda r: r["abstention_id"])

    candidates: List[dict] = []
    stats: Counter = Counter()
    for fact in facts:
        if split[fact["fact_id"]]["split"] != "sft":
            continue
        train, _, answer = render(fact, seed)
        method = "curated_ai_assisted" if fact["extractor"].startswith("curated") else "rule_based"
        allowed = (str(fact["version"]), *fact["extra"].get("derived_tokens", []))
        result = check_grounding(answer, fact_source(documents, fact), fact["key_facts"], fact["project"], settings["grounding_min_overlap"], allowed)
        status = result["status"]
        if fact["chunk_id"] is None:
            status, result = PENDING, {**result, "reasons": result["reasons"] + ["no_chunk_for_fact"]}
        for question in train[: settings["questions_per_fact"]]:
            ambiguous = is_ambiguous(question, fact["project"])
            candidates.append(
                {
                    "key": stable_id("sft", fact["fact_id"], question),
                    "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                    "metadata": {"category": fact["category"], "difficulty": difficulty_of(fact), "project": fact["project"], "version": fact["version"], "version_scope": fact["version_scope"], "language": fact["language"], "fact_id": fact["fact_id"], "generation_method": method, "validation_status": PENDING if ambiguous else status},
                    "provenance": from_fact(fact, duplicates),
                    "reasons": result["reasons"] + (["ambiguous_question"] if ambiguous else []),
                }
            )
    for row in abstentions:
        if row["split"] != "sft":
            continue
        for question in row["train_questions"][: settings["questions_per_fact"]]:
            ambiguous = is_ambiguous(question, row["project"])
            candidates.append(
                {
                    "key": stable_id("sft", row["abstention_id"], question),
                    "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": row["answer"]}],
                    "metadata": {"category": "abstention", "difficulty": abstention_difficulty(row), "project": row["project"], "version": None, "version_scope": "canonical", "language": row["language"], "fact_id": row["abstention_id"], "generation_method": "curated_ai_assisted" if row["reason"] == "stub_or_placeholder" else "rule_based", "validation_status": PENDING if ambiguous else VALIDATED},
                    "provenance": abstention_provenance(row),
                    "reasons": ["ambiguous_question"] if ambiguous else [],
                }
            )

    if settings["llm"]["enabled"]:
        provider = provider_from_env()
        pool = [row for row in iter_jsonl(chunks_path) if row["usage"] == "cpt"]
        stats["llm_candidates"] = len(generate_candidates(pool, provider, settings["llm"], settings["grounding_min_overlap"]))

    # Same question: identical answers collapse, different answers contradict and leave.
    by_question: Dict[str, List[dict]] = defaultdict(list)
    for row in candidates:
        by_question[normalize_question(row["messages"][0]["content"])].append(row)
    validated, pending = [], []
    for key in sorted(by_question):
        group = sorted(by_question[key], key=lambda r: r["key"])
        if len({normalize_text(r["messages"][1]["content"]) for r in group}) > 1:
            stats["contradictory_questions_removed"] += len(group)
            continue
        stats["duplicate_questions_removed"] += len(group) - 1
        (validated if group[0]["metadata"]["validation_status"] == VALIDATED else pending).append(group[0])

    validated.sort(key=lambda r: r["key"])
    write_jsonl(config.output("sft"), ({"id": f"sft_{number:06d}", "messages": r["messages"], "metadata": r["metadata"], "provenance": r["provenance"]} for number, r in enumerate(validated, 1)))
    write_jsonl(config.processed("sft_pending_review.jsonl"), sorted(pending, key=lambda r: r["key"]))
    write_json(config.processed("fact_split.json"), split)
    write_jsonl(config.processed("abstentions.jsonl"), abstentions)
    coverage = Counter(r["metadata"]["category"] for r in validated)
    result = {
        "candidates": len(candidates),
        "validated": len(validated),
        "pending_review": len(pending),
        "validated_percent": round(100 * len(validated) / max(len(validated) + len(pending), 1), 1),
        "pending_reasons": dict(sorted(Counter(reason.split(":")[0] for r in pending for reason in r["reasons"]).items())),
        "category_coverage": {c: coverage.get(c, 0) for c in CATEGORIES + ("abstention",)},
        "difficulty_coverage": dict(sorted(Counter(r["metadata"]["difficulty"] for r in validated).items())),
        "fact_splits": dict(sorted(Counter(v["split"] for v in split.values()).items())),
        **dict(sorted(stats.items())),
    }
    write_json(config.processed("sft_stats.json"), result)
    return result
