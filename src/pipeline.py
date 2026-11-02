"""Pipeline entry point: run everything, or one stage, with reruns that skip unchanged work."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .common import Config, JsonLogger, PipelineError, load_config, read_json, sha256_file, write_json
from .cpt import build as cpt
from .evaluation import build as evaluation
from .ingestion import ingest
from .preprocessing import curate
from .reporting import report
from .sft import build as sft

STAGES: List[Tuple[str, str, Callable[[Config], dict]]] = [
    ("ingestion", "Inventory every file, hash it, and record what could not be read.", ingest.run),
    ("curation", "Normalize, redact, deduplicate, filter, extract facts, pick holdouts, and chunk.", curate.run),
    ("cpt", "Build continuous domain text from the training chunks.", cpt.run),
    ("sft", "Render facts as questions and answers; keep what passes grounding.", sft.run),
    ("evaluation", "Build questions whose answers are not in SFT, per tier, including abstention.", evaluation.run),
    ("validation", "Verify provenance, grounding and leakage across datasets; write the reports.", report.run),
]


def stage_outputs(config: Config) -> Dict[str, List[Path]]:
    processed = config.processed
    reports = config.path("reports_dir")
    return {
        "ingestion": [processed("ingested_documents.jsonl"), processed("ingestion_stats.json")],
        "curation": [processed(name) for name in ("documents_cleaned.jsonl", "documents_rejected.jsonl", "chunks.jsonl", "facts.jsonl", "holdout.json", "curation_stats.json")] + [reports / "deduplication_report.json"],
        "cpt": [config.output("cpt"), processed("cpt_stats.json")],
        "sft": [config.output("sft"), processed("sft_pending_review.jsonl"), processed("fact_split.json"), processed("abstentions.jsonl"), processed("sft_stats.json")],
        "evaluation": [config.output("eval"), reports / "sft_recall_probe.jsonl", reports / "neutral_eval.jsonl", reports / "training_validation_split.json", processed("evaluation_stats.json")],
        "validation": [reports / name for name in ("data_quality.json", "dataset_statistics.json", "leakage_report.json", "constant_baseline.json", "summary.md")],
    }


def fingerprint(config: Config, name: str, outputs: Dict[str, List[Path]]) -> str:
    """Content hash of the code, the settings, and this stage's inputs. No timestamps involved."""
    digest = hashlib.sha256()
    root = config["root"]
    for path in sorted((root / "src").rglob("*.py")) + sorted((root / "configs").glob("*.yaml")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(sha256_file(path).encode())
    names = [stage[0] for stage in STAGES]
    if name == "ingestion":
        raw = config.path("raw_dir")
        for path in sorted(p for p in raw.rglob("*") if p.is_file()) if raw.exists() else []:
            digest.update(path.relative_to(raw).as_posix().encode())
            digest.update(sha256_file(path).encode())
    for earlier in names[: names.index(name)]:
        for path in outputs[earlier]:
            if path.exists():
                digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def run_stage(config: Config, logger: JsonLogger, name: str, description: str, function: Callable[[Config], dict], force: bool) -> None:
    outputs = stage_outputs(config)
    marker = config.path("checkpoint_dir") / f"{name}.json"
    current = fingerprint(config, name, outputs)
    done = marker.exists() and read_json(marker)["fingerprint"] == current and all(path.exists() for path in outputs[name])
    if done and not force:
        logger.log("stage_skipped", stage=name, reason="inputs, code and config unchanged")
        return
    logger.log("stage_started", stage=name, description=description)
    result = function(config)
    write_json(marker, {"fingerprint": fingerprint(config, name, outputs)})
    logger.log("stage_finished", stage=name, **result)


def main(argv: List[str] = None) -> int:
    names = [stage[0] for stage in STAGES]
    parser = argparse.ArgumentParser(description="LLM data preparation pipeline: raw docs to CPT, SFT and evaluation datasets.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"))
    parser.add_argument("--stage", default="all", choices=names + ["all"])
    parser.add_argument("--force", action="store_true", help="rerun even if nothing changed")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        logger = JsonLogger(config.path("logs_dir"))
        for name, description, function in STAGES:
            if args.stage in ("all", name):
                run_stage(config, logger, name, description, function, args.force)
    except PipelineError as error:
        print(f"PIPELINE ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
