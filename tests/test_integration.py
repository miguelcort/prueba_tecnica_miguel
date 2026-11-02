from __future__ import annotations

import json
import shutil
from pathlib import Path

from conftest import ROOT, make_project
from src.common import iter_jsonl, load_config, normalize_question, sha256_file
from src.ingestion.ingest import tree_digest
from src.pipeline import STAGES, main
from src.validation.validate import run as validate

OUTPUT_FOLDERS = ("outputs", "reports")
PROVENANCE_KEYS = {"source_file", "source_hash", "document_id", "section"}


def rows(project: Path, relative: str) -> list:
    return list(iter_jsonl(project / relative))


def snapshot(project: Path) -> dict:
    files = {}
    for folder in OUTPUT_FOLDERS:
        for path in sorted((project / folder).rglob("*")):
            if path.is_file():
                files[path.relative_to(project).as_posix()] = sha256_file(path)
    return files


def config_of(project: Path) -> str:
    return str(project / "configs" / "pipeline.yaml")


def _copy_built(built_project: Path, target: Path) -> Path:
    shutil.copytree(built_project, target)
    return target


def _rewrite(path: Path, change) -> None:
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    change(lines)
    path.write_text("".join(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n" for line in lines), encoding="utf-8")


def test_two_runs_are_byte_identical_and_workers_do_not_change_results(built_project, tmp_path):
    second = make_project(tmp_path / "second")
    assert main(["--config", config_of(second)]) == 0
    assert snapshot(second) == snapshot(built_project)
    parallel = make_project(tmp_path / "parallel")
    settings = parallel / "configs" / "pipeline.yaml"
    settings.write_text(settings.read_text().replace("workers: 1 ", "workers: 2 "))
    assert main(["--config", config_of(parallel)]) == 0
    assert snapshot(parallel) == snapshot(built_project)


def test_committed_outputs_match_a_regeneration_from_scratch(built_project):
    for name in ("cpt", "sft", "eval"):
        assert sha256_file(ROOT / "outputs" / f"{name}.jsonl") == sha256_file(built_project / "outputs" / f"{name}.jsonl")


def test_originals_are_never_modified(built_project):
    assert tree_digest(built_project / "docs_raw") == tree_digest(ROOT / "docs_raw")
    assert sum(1 for path in (built_project / "docs_raw").rglob("*") if path.is_file()) == 107


def test_validation_detects_a_changed_original(built_project, tmp_path):
    project = _copy_built(built_project, tmp_path / "changed")
    target = next((project / "docs_raw").rglob("README.md"))
    target.write_text(target.read_text(encoding="utf-8") + "\nedited\n", encoding="utf-8")
    report = validate(load_config(config_of(project)))
    assert "originals_changed" in report["errors"] and "provenance" in report["errors"]


def test_exactly_three_main_datasets(built_project):
    assert sorted(path.name for path in (built_project / "outputs").iterdir()) == ["cpt.jsonl", "eval.jsonl", "sft.jsonl"]


def test_every_stage_runs_alone_in_order_and_fails_clearly_out_of_order(tmp_path, capsys):
    project = make_project(tmp_path / "stages")
    assert main(["--config", config_of(project), "--stage", "sft"]) == 1
    assert "run the earlier stages first" in capsys.readouterr().err
    assert [stage[0] for stage in STAGES] == ["ingestion", "curation", "cpt", "sft", "evaluation", "validation"]
    for name, _, _ in STAGES:
        assert main(["--config", config_of(project), "--stage", name]) == 0


def test_rerun_skips_unchanged_stages(built_project, capsys):
    capsys.readouterr()
    assert main(["--config", config_of(built_project)]) == 0
    assert capsys.readouterr().out.count("stage_skipped") == len(STAGES)


def test_all_rows_have_required_provenance_and_schema(built_project):
    cpt = rows(built_project, "outputs/cpt.jsonl")
    sft = rows(built_project, "outputs/sft.jsonl")
    evaluation = rows(built_project, "outputs/eval.jsonl")
    assert cpt and sft and evaluation
    for row in cpt + sft + evaluation:
        assert PROVENANCE_KEYS <= set(row["provenance"]) and row["provenance"]["source_file"].startswith("docs_raw/")
    for row in cpt:
        assert {"id", "text"} <= set(row) and row["provenance"]["chunk_id"]
    for row in sft:
        assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    for item in evaluation:
        assert item["expected_behavior"] in ("answer", "answer_multi_source", "abstain")
        assert (item["expected_answer"] is None) == (item["expected_behavior"] == "abstain")


def test_all_provenance_is_valid_in_the_report(built_project):
    report = json.loads((built_project / "reports/data_quality.json").read_text())["validation"]
    assert report["passed"] and set(report["provenance_valid_percent"].values()) == {100.0}


def test_hard_rows_really_use_several_passages(built_project):
    hard_sft = [r for r in rows(built_project, "outputs/sft.jsonl") if r["metadata"]["difficulty"] == "hard" and r["metadata"]["category"] != "abstention"]
    assert hard_sft and all(len(r["provenance"]["sources"]) >= 2 for r in hard_sft)
    multi = [r for r in rows(built_project, "outputs/eval.jsonl") if r["expected_behavior"] == "answer_multi_source"]
    assert multi and all(len(r["evidence"]) >= 2 and r["difficulty"] == "hard" for r in multi)


def test_held_out_documents_are_absent_from_training(built_project):
    holdout = json.loads((built_project / "work/processed/holdout.json").read_text())
    assert holdout["document_ids"]
    held = set(holdout["document_ids"]) | set(holdout["variant_document_ids"])
    assert not held & {r["provenance"]["document_id"] for r in rows(built_project, "outputs/cpt.jsonl")}
    assert not held & {r["provenance"]["document_id"] for r in rows(built_project, "outputs/sft.jsonl")}


def test_eval_facts_never_appear_in_sft(built_project):
    sft = rows(built_project, "outputs/sft.jsonl")
    evaluation = rows(built_project, "outputs/eval.jsonl")
    assert not {r["fact_id"] for r in evaluation} & {r["metadata"]["fact_id"] for r in sft}
    assert not {normalize_question(r["question"]) for r in evaluation} & {normalize_question(r["messages"][0]["content"]) for r in sft}
    assert {r["tier"] for r in evaluation} == {"document_holdout", "fact_holdout", "unanswerable"}


def test_recall_probe_is_labeled_and_kept_out_of_eval(built_project):
    probe = rows(built_project, "reports/sft_recall_probe.jsonl")
    assert probe and all(r["diagnostic_only"] for r in probe)
    assert not {r["fact_id"] for r in probe} & {r["fact_id"] for r in rows(built_project, "outputs/eval.jsonl")}


def test_generation_method_is_honest_and_nothing_pending_in_training(built_project):
    sft = rows(built_project, "outputs/sft.jsonl")
    assert {row["metadata"]["generation_method"] for row in sft} == {"rule_based", "curated_ai_assisted"}
    assert {row["metadata"]["validation_status"] for row in sft} == {"validated"}


def test_validation_catches_a_planted_question_leak(built_project, tmp_path):
    project = _copy_built(built_project, tmp_path / "leak")
    target = rows(project, "outputs/eval.jsonl")[0]

    def plant(lines):
        lines[0]["messages"][0]["content"] = target["question"]

    _rewrite(project / "outputs/sft.jsonl", plant)
    report = validate(load_config(config_of(project)))
    assert not report["passed"] and "leakage" in report["errors"]


def test_validation_catches_a_planted_answer_leak(built_project, tmp_path):
    project = _copy_built(built_project, tmp_path / "answer")
    target = next(r for r in rows(project, "outputs/eval.jsonl") if r["tier"] == "fact_holdout" and r["expected_behavior"] == "answer")

    def plant(lines):
        row = next(line for line in lines if line["metadata"]["category"] == "abstention")
        row["messages"][1]["content"] = target["expected_answer"]

    _rewrite(project / "outputs/sft.jsonl", plant)
    assert "leakage" in validate(load_config(config_of(project)))["errors"]


def test_validation_catches_tampered_provenance(built_project, tmp_path):
    for field, value in (("source_hash", "0" * 64), ("document_id", "doc_invented"), ("source_file", "docs_raw/invented.md"), ("line_end", 999999), ("section", "Invented Section")):
        project = _copy_built(built_project, tmp_path / f"tamper_{field}")

        def corrupt(lines):
            lines[0]["provenance"][field] = value

        _rewrite(project / "outputs/cpt.jsonl", corrupt)
        report = validate(load_config(config_of(project)))
        assert "provenance" in report["errors"] and report["provenance_valid_percent"]["cpt"] < 100, field


def test_validation_catches_an_invented_answer(built_project, tmp_path):
    project = _copy_built(built_project, tmp_path / "invented")

    def corrupt(lines):
        row = next(line for line in lines if line["metadata"]["category"] != "abstention")
        row["messages"][1]["content"] = "`DELETE /invented/path` wipes 999 Zebras"

    _rewrite(project / "outputs/sft.jsonl", corrupt)
    assert "grounding" in validate(load_config(config_of(project)))["errors"]


def test_validation_catches_a_false_absence(built_project, tmp_path):
    project = _copy_built(built_project, tmp_path / "absence")

    def corrupt(lines):
        row = next(line for line in lines if line.get("unanswerable_reason") == "absent_from_target_service")
        row["absent_terms"] = [row["project"]]

    _rewrite(project / "outputs/eval.jsonl", corrupt)
    assert "tier_rule" in validate(load_config(config_of(project)))["errors"]


def test_validation_catches_a_secret_and_fails_the_cli(built_project, tmp_path):
    project = _copy_built(built_project, tmp_path / "secret")

    def plant(lines):
        lines[0]["text"] += " contact person@company.com"

    _rewrite(project / "outputs/cpt.jsonl", plant)
    assert "sensitive_data" in validate(load_config(config_of(project)))["errors"]
    assert main(["--config", config_of(project), "--stage", "validation", "--force"]) == 1
