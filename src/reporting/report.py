"""Stage 6, validation and reporting: every number comes from the files just written."""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List

from ..common import Config, PipelineError, iter_jsonl, read_json, require, write_json
from ..validation.validate import run as validate


def _average(values: Iterable[int]) -> float:
    items = list(values)
    return round(mean(items), 1) if items else 0.0


def _table(rows: List[tuple], header: tuple) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]


def _count(rows: List[dict], getter) -> Dict[str, int]:
    return dict(sorted(Counter(str(getter(row)) for row in rows).items()))


def project_split_viability(facts: List[dict], holdout: dict) -> dict:
    """Numbers behind choosing a document split over a project split."""
    per_project = Counter(fact["project"] for fact in facts)
    total = sum(per_project.values())
    return {
        "projects": len(per_project),
        "facts_per_project": dict(sorted(per_project.items())),
        "fact_share_percent_per_project": {project: round(100 * count / total, 1) for project, count in sorted(per_project.items())},
        "projects_without_enough_documents_for_document_holdout": holdout["projects_without_holdout"],
        "decision": "document split with fact holdout",
        "reason": "the facts are specific to each service, so a held-out project leaves closed-book arms with nothing to learn about it and only measures retrieval",
    }


def run(config: Config) -> dict:
    processed = config.processed
    require(config.output("cpt"), config.output("sft"), config.output("eval"), processed("curation_stats.json"))
    reports = config.path("reports_dir")
    cpt, sft, evaluation = (list(iter_jsonl(config.output(name))) for name in ("cpt", "sft", "eval"))
    probe = list(iter_jsonl(reports / "sft_recall_probe.jsonl"))
    facts = list(iter_jsonl(processed("facts.jsonl")))
    ingestion, curation = read_json(processed("ingestion_stats.json")), read_json(processed("curation_stats.json"))
    sft_stats, eval_stats = read_json(processed("sft_stats.json")), read_json(processed("evaluation_stats.json"))
    cpt_stats, holdout = read_json(processed("cpt_stats.json")), read_json(processed("holdout.json"))
    dedup = read_json(reports / "deduplication_report.json")
    validation = validate(config)

    def meta(key: str):
        return lambda row: row["metadata"][key]

    statistics = {
        "cpt": {"rows": len(cpt), "avg_chars": _average(len(r["text"]) for r in cpt), "by_project": _count(cpt, meta("project")), "by_derivation": _count(cpt, meta("derived_from"))},
        "sft": {"rows": len(sft), "avg_question_chars": _average(len(r["messages"][0]["content"]) for r in sft), "avg_answer_chars": _average(len(r["messages"][1]["content"]) for r in sft), "by_project": _count(sft, meta("project")), "by_category": _count(sft, meta("category")), "by_difficulty": _count(sft, meta("difficulty")), "by_generation_method": _count(sft, meta("generation_method")), "by_language": _count(sft, meta("language"))},
        "eval": {"rows": len(evaluation), "by_tier": _count(evaluation, lambda r: r["tier"]), "by_project": _count(evaluation, lambda r: r["project"]), "by_category": _count(evaluation, lambda r: r["category"]), "by_difficulty": _count(evaluation, lambda r: r["difficulty"]), "by_behavior": _count(evaluation, lambda r: r["expected_behavior"]), "by_language": _count(evaluation, lambda r: r["language"])},
        "sft_recall_probe": {"rows": len(probe), "note": "diagnostic only, not part of the leakage-free evaluation"},
        "chunks": {"total": curation["chunks"], "by_usage": curation["chunks_by_usage"]},
    }
    write_json(reports / "dataset_statistics.json", statistics)

    rejected_total = sum(curation["documents_rejected"].values())
    quality = {
        "documents_original": ingestion["files"],
        "documents_processed": curation["documents_kept"] + curation["documents_version_variant"],
        "documents_rejected": rejected_total,
        "documents_rejected_by_reason": curation["documents_rejected"],
        "documents_version_variants": curation["documents_version_variant"],
        "documents_with_todo_markers": curation["documents_with_todo_markers"],
        "redactions": curation["redactions"],
        "quarantined_lines": curation["quarantined_lines"],
        "fact_conflicts": curation["fact_conflicts"],
        "fact_rejections": dict(sorted(Counter(r["reason"] for r in curation["fact_rejections"]).items())),
        "curated_rejections": [r for r in curation["fact_rejections"] if r["reason"].startswith("curated")],
        "cpt_rejected": cpt_stats["rejected"],
        "chunks_removed": curation["chunks_removed"],
        "sft": {k: sft_stats[k] for k in ("candidates", "validated", "pending_review", "validated_percent", "pending_reasons", "category_coverage", "difficulty_coverage", "fact_splits")},
        "sft_duplicates_removed": sft_stats.get("duplicate_questions_removed", 0),
        "sft_contradictions_removed": sft_stats.get("contradictory_questions_removed", 0),
        "evaluation_dropped": eval_stats["dropped"],
        "validation": validation,
    }
    write_json(reports / "data_quality.json", quality)
    write_json(
        reports / "leakage_report.json",
        {
            "passed": not validation["errors"].get("leakage"),
            "counts": validation["leakage"],
            "reading_guide": "fact_holdout answers are in CPT by design: that tier measures transfer from CPT. Partial key overlap means one key fact, not the whole answer, also appears in an SFT answer. High Jaccard with other identifiers means the same template about another subject.",
            "held_out_documents": holdout["source_files"],
            "project_split_viability": project_split_viability(facts, holdout),
        },
    )

    near = dedup["near_duplicate_pairs"]
    actions = Counter(pair["action"] for pair in near)
    lines = ["# Data quality summary", "", "Generated by the pipeline; no number here is written by hand.", "", f"**Validation: {'PASSED' if validation['passed'] else 'FAILED'}**", ""]
    lines += _table(
        [
            ("Original files", ingestion["files"]),
            ("Documents kept (canonical)", curation["documents_kept"]),
            ("Version variants (not in CPT)", curation["documents_version_variant"]),
            ("Documents rejected", rejected_total),
            ("Near-duplicate pairs flagged as version differences", actions["flagged_version_difference"]),
            ("Repeated fragments reported", len(dedup["repeated_fragments"])),
            ("Facts", curation["facts"]),
            ("Chunks", curation["chunks"]),
            ("CPT rows", len(cpt)),
            ("SFT rows (validated)", len(sft)),
            ("SFT rows pending review (not in training)", sft_stats["pending_review"]),
            ("SFT validated percent", sft_stats["validated_percent"]),
            ("Evaluation questions", len(evaluation)),
            ("Recall probe rows (diagnostic)", len(probe)),
            ("Provenance valid percent", validation["provenance_valid_percent"]),
        ],
        ("Metric", "Value"),
    )
    cpt_p, sft_p, eval_p = statistics["cpt"]["by_project"], statistics["sft"]["by_project"], statistics["eval"]["by_project"]
    lines += ["", "## By project", ""] + _table([(p, ingestion["files_by_project"].get(p, 0), cpt_p.get(p, 0), sft_p.get(p, 0), eval_p.get(p, 0)) for p in sorted(ingestion["files_by_project"])], ("Project", "Files", "CPT", "SFT", "Evaluation"))
    lines += ["", "## SFT by category", ""] + _table(sorted(sft_stats["category_coverage"].items()), ("Category", "Rows"))
    lines += ["", "## SFT by difficulty", ""] + _table(sorted(sft_stats["difficulty_coverage"].items()), ("Difficulty", "Rows"))
    lines += ["", "## Evaluation by tier", ""] + _table(sorted(statistics["eval"]["by_tier"].items()), ("Tier", "Questions"))
    lines += ["", "## Evaluation by behavior", ""] + _table(sorted(statistics["eval"]["by_behavior"].items()), ("Behavior", "Questions"))
    lines += ["", "## Evaluation by difficulty", ""] + _table(sorted(statistics["eval"]["by_difficulty"].items()), ("Difficulty", "Questions"))
    lines += ["", "## Leakage counts", ""] + _table(sorted(validation["leakage"].items()), ("Check", "Count"))
    lines += ["", "## Rejected documents", ""] + _table(sorted(curation["documents_rejected"].items()), ("Reason", "Documents"))
    lines += ["", "## Held-out documents", ""] + [f"- `{path}`" for path in holdout["source_files"]]
    lines += ["", "## Near-duplicate threshold sweep", ""] + _table([(item["threshold"], item["pairs_at_or_above"]) for item in dedup["threshold_sweep_same_project_pairs"]], ("Threshold", "Pairs"))
    (reports / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not validation["passed"]:
        raise PipelineError("validation failed: " + str(validation["errors"]) + " " + str(validation["examples"]))
    return {"validation": "passed", "rows": validation["rows_checked"], "provenance_valid_percent": validation["provenance_valid_percent"], "evaluation_by_tier": validation["evaluation_by_tier"]}
