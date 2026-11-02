"""Cross-dataset checks. Any violation fails the run, because a dataset is a claim to defend."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from ..common import Config, iter_jsonl, normalize_question, normalize_text, read_json, require, same_question, token_jaccard
from ..cpt.build import JUNK
from ..deduplication.dedup import minhash, similarity
from ..evaluation.build import answer_present, key_in_text, specific_keys
from ..ingestion.ingest import tree_digest
from ..preprocessing.normalize import EMAIL, PUBLIC_HOSTS, SLACK_CHANNEL, normalize_lines
from ..provenance.provenance import Verifier
from ..sft.grounding import VALIDATED, check_grounding, fact_source

SECRET = re.compile(r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|\bBearer\s+[A-Za-z0-9._-]{20,}|\bpassword\s*[:=]\s*(?!\*|<|your)[^\s,]{6,}", re.IGNORECASE)
HOST = re.compile(r"(?<![\w@-])(?:[A-Za-z0-9-]+\.)+(?:com|io|net|org)(?![\w-])")
TIERS = {"fact_holdout", "document_holdout", "unanswerable"}
BEHAVIORS = {"answer", "answer_multi_source", "abstain"}
DIFFICULTIES = {"easy", "medium", "hard"}
METHODS = {"rule_based", "curated_ai_assisted", "llm"}
SFT_META_KEYS = {"category", "difficulty", "project", "language", "fact_id", "generation_method", "validation_status"}
EVAL_KEYS = {"id", "question", "expected_answer", "expected_behavior", "tier", "category", "difficulty", "evidence", "provenance", "key_facts", "fact_id"}


class Checker:
    def __init__(self) -> None:
        self.errors: Counter = Counter()
        self.examples: Dict[str, List[str]] = {}

    def fail(self, check: str, detail: str) -> None:
        self.errors[check] += 1
        self.examples.setdefault(check, [])
        if len(self.examples[check]) < 5:
            self.examples[check].append(detail[:300])


def scan_sensitive(text: str, where: str, checker: Checker) -> None:
    for pattern, label in ((EMAIL, "email"), (SECRET, "secret"), (SLACK_CHANNEL, "slack_channel")):
        if pattern.search(text):
            checker.fail("sensitive_data", f"{label} in {where}")
    for match in HOST.finditer(text):
        if match.group(0) not in PUBLIC_HOSTS:
            checker.fail("sensitive_data", f"host {match.group(0)} in {where}")


class ProvenanceTally:
    def __init__(self, verifier: Verifier, checker: Checker):
        self.verifier, self.checker = verifier, checker
        self.total: Counter = Counter()
        self.valid: Counter = Counter()

    def check(self, dataset: str, row: dict, where: str) -> None:
        self.total[dataset] += 1
        provenance = row.get("provenance")
        problems = self.verifier.errors(provenance) if isinstance(provenance, dict) else ["no provenance"]
        for problem in problems:
            self.checker.fail("provenance", f"{where}: {problem}")
        self.valid[dataset] += not problems

    def percent(self) -> Dict[str, float]:
        return {dataset: round(100 * self.valid[dataset] / self.total[dataset], 2) for dataset in sorted(self.total)}


def absent_from_project(terms: List[str], project: str, raw_texts: Dict[str, str]) -> bool:
    return all(normalize_text(term) not in raw_texts.get(project, "") for term in terms)


def run(config: Config) -> dict:
    require(config.output("cpt"), config.output("sft"), config.output("eval"), config.processed("chunks.jsonl"), config.processed("holdout.json"))
    settings = config["evaluation"]
    checker = Checker()
    raw_dir = config.path("raw_dir")
    verifier = Verifier(config["root"], raw_dir.name)
    tally = ProvenanceTally(verifier, checker)
    documents = {doc["document_id"]: doc for doc in iter_jsonl(config.processed("documents_cleaned.jsonl"))}
    chunks = {chunk["chunk_id"]: chunk for chunk in iter_jsonl(config.processed("chunks.jsonl"))}
    holdout = read_json(config.processed("holdout.json"))
    split = read_json(config.processed("fact_split.json"))
    facts = {fact["fact_id"]: fact for fact in iter_jsonl(config.processed("facts.jsonl"))}
    ingestion = read_json(config.processed("ingestion_stats.json"))
    held_docs = set(holdout["document_ids"]) | set(holdout["variant_document_ids"])

    current_tree = tree_digest(raw_dir)
    if current_tree != ingestion["docs_raw_tree_sha256"]:
        checker.fail("originals_changed", "docs_raw differs from what ingestion read")

    raw_texts: Dict[str, List[str]] = {}
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix in (".md", ".yaml")):
        raw_texts.setdefault(path.relative_to(raw_dir).parts[0], []).extend(normalize_lines(path.read_text("utf-8")))
    project_text = {project: normalize_text("\n".join(lines)) for project, lines in raw_texts.items()}

    for chunk in chunks.values():
        scan_sensitive(chunk["text"], f"chunk {chunk['chunk_id']}", checker)

    # --- CPT
    cpt_rows = list(iter_jsonl(config.output("cpt")))
    seen_text = set()
    min_chars = config["validation"]["min_cpt_chars"]
    for row in cpt_rows:
        text, where = row.get("text", ""), f"cpt {row.get('id')}"
        if not text.strip() or len(text) < min_chars or "id" not in row or not isinstance(row.get("provenance"), dict):
            checker.fail("schema", where)
            continue
        tally.check("cpt", row, where)
        provenance = row["provenance"]
        scan_sensitive(text, where, checker)
        if provenance["document_id"] in held_docs:
            checker.fail("leakage", f"{where}: held-out document in CPT")
        if JUNK.search(text):
            checker.fail("cpt_junk", where)
        key = normalize_text(text)
        if key in seen_text:
            checker.fail("duplicates", where)
        seen_text.add(key)
        if provenance.get("chunk_id") not in chunks:
            checker.fail("provenance", f"{where}: unknown chunk")
        elif normalize_text(chunks[provenance["chunk_id"]]["text"]) not in key:
            checker.fail("provenance", f"{where}: text differs from its chunk")

    # --- near-duplicate chunks across the document boundary
    dedup = config["deduplication"]
    held_signatures = [minhash(c["text"], 3, dedup["num_perm"], config["seed"]) for c in chunks.values() if c["usage"] == "document_holdout"]
    for row in cpt_rows:
        chunk = chunks.get(row.get("provenance", {}).get("chunk_id"))
        if chunk is None:
            continue
        own = minhash(chunk["text"], 3, dedup["num_perm"], config["seed"])
        if any(similarity(own, other) >= dedup["near_duplicate_threshold"] for other in held_signatures):
            checker.fail("leakage", f"cpt {row['id']} is a near duplicate of a held-out chunk")

    # --- SFT
    sft_rows = list(iter_jsonl(config.output("sft")))
    sft_questions, sft_answers, sft_facts = {}, [], set()
    answers_by_question: Dict[str, str] = {}
    min_overlap = config["sft"]["grounding_min_overlap"]
    for row in sft_rows:
        meta, where = row.get("metadata", {}), f"sft {row.get('id')}"
        messages = row.get("messages", [])
        if [m.get("role") for m in messages] != ["user", "assistant"] or not SFT_META_KEYS <= set(meta) or not all(m.get("content") for m in messages) or not isinstance(row.get("provenance"), dict):
            checker.fail("schema", where)
            continue
        question, answer = messages[0]["content"], messages[1]["content"]
        tally.check("sft", row, where)
        scan_sensitive(question + " " + answer, where, checker)
        if meta["validation_status"] != VALIDATED:
            checker.fail("validation_status", where)
        if meta["generation_method"] not in METHODS or meta["difficulty"] not in DIFFICULTIES:
            checker.fail("schema", f"{where}: enum field")
        evidence_docs = [s["document_id"] for s in row["provenance"].get("sources", [])] or [row["provenance"].get("document_id")]
        if set(evidence_docs) & held_docs:
            checker.fail("leakage", f"{where}: held-out document in SFT")
        if meta["category"] != "abstention":
            fact = facts.get(meta["fact_id"])
            if fact is None:
                checker.fail("provenance", f"{where}: unknown fact")
                continue
            allowed = (str(fact["version"]), *fact["extra"].get("derived_tokens", []))
            result = check_grounding(answer, fact_source(documents, fact), fact["key_facts"], fact["project"], min_overlap, allowed)
            if result["status"] != VALIDATED:
                checker.fail("grounding", f"{where}: {result['reasons']}")
            for key in fact["key_facts"]:
                if not verifier.text_contains(row["provenance"], key):
                    checker.fail("grounding", f"{where}: key fact not in raw source: {key}")
            if meta["difficulty"] == "hard" and len(row["provenance"].get("sources", [])) < 2:
                checker.fail("schema", f"{where}: hard row needs two sources")
        key = normalize_question(question)
        if key in answers_by_question:
            checker.fail("duplicates", where)
            if normalize_text(answers_by_question[key]) != normalize_text(answer):
                checker.fail("contradiction", where)
        answers_by_question[key] = answer
        sft_questions[key] = question
        sft_answers.append(answer)
        sft_facts.add(meta["fact_id"])
    sft_by_project: Dict[str, List[str]] = {}
    for row in sft_rows:
        sft_by_project.setdefault(row["metadata"]["project"], []).append(normalize_text(row["messages"][1]["content"]))

    # --- evaluation
    tiers: Counter = Counter()
    behaviors: Counter = Counter()
    cpt_text = {row["provenance"]["chunk_id"]: normalize_text(row["text"]) for row in cpt_rows if isinstance(row.get("provenance"), dict)}
    cpt_by_project: Dict[str, List[str]] = {}
    for row in cpt_rows:
        if isinstance(row.get("provenance"), dict):
            cpt_by_project.setdefault(row["metadata"]["project"], []).append(cpt_text[row["provenance"]["chunk_id"]])
    seen_questions = set()
    leakage = {
        "question_exact_or_semantic_overlap_with_sft": 0,
        "question_high_jaccard_with_other_identifiers": 0,
        "answer_overlap_with_sft": 0,
        "answer_partial_key_overlap_with_sft": 0,
        "fact_holdout_answers_in_cpt_by_design": 0,
        "document_holdout_answers_in_cpt": 0,
        "document_holdout_keys_only_in_other_project_cpt": 0,
        "document_holdout_evidence_partly_trained": 0,
        "fact_holdout_reassigned_from_document_holdout": 0,
        "sft_facts_in_eval": 0,
    }
    question_threshold, answer_threshold = settings["question_jaccard_threshold"], settings["answer_jaccard_threshold"]
    generic = settings["generic_keys"]
    for item in iter_jsonl(config.output("eval")):
        where = f"eval {item.get('id')}"
        if not EVAL_KEYS <= set(item) or item["tier"] not in TIERS or item["expected_behavior"] not in BEHAVIORS or item["difficulty"] not in DIFFICULTIES or not item["evidence"] or not isinstance(item.get("provenance"), dict):
            checker.fail("schema", where)
            continue
        tier, question, answer, behavior = item["tier"], item["question"], item["expected_answer"], item["expected_behavior"]
        tiers[tier] += 1
        behaviors[behavior] += 1
        tally.check("eval", item, where)
        scan_sensitive(question + " " + (answer or ""), where, checker)
        if (answer is None) != (behavior == "abstain"):
            checker.fail("schema", f"{where}: expected_answer does not match behavior")
        if behavior == "answer_multi_source" and len(item["evidence"]) < 2:
            checker.fail("schema", f"{where}: multi-source item with one evidence")
        key = normalize_question(question)
        if key in seen_questions:
            checker.fail("duplicates", where)
        seen_questions.add(key)
        for other in sft_questions.values():
            if same_question(question, other, question_threshold):
                leakage["question_exact_or_semantic_overlap_with_sft"] += 1
                checker.fail("leakage", f"{where}: question duplicates SFT: {other}")
            elif token_jaccard(question, other) >= question_threshold:
                leakage["question_high_jaccard_with_other_identifiers"] += 1
        if item["fact_id"] in sft_facts:
            leakage["sft_facts_in_eval"] += 1
            checker.fail("leakage", f"{where}: fact also in SFT")
        if tier == "unanswerable":
            if item["unanswerable_reason"] == "absent_from_target_service" and not absent_from_project(item["absent_terms"], item["project"], project_text):
                checker.fail("tier_rule", f"{where}: claimed absent, found in the project")
            if item["fact_id"] in split:
                checker.fail("tier_rule", where)
            continue
        keys = item["key_facts"]
        for key_fact in keys:
            if not verifier.text_contains(item["provenance"], key_fact):
                checker.fail("grounding", f"{where}: key fact not in raw source: {key_fact}")
        project_sft = sft_by_project.get(item["project"], [])
        project_cpt = cpt_by_project.get(item["project"], [])
        if any(answer_present(keys, [text], generic) for text in project_sft) or any(token_jaccard(answer, other) >= answer_threshold for other in sft_answers):
            leakage["answer_overlap_with_sft"] += 1
            checker.fail("leakage", f"{where}: answer present in SFT")
        elif any(key_in_text(k, text) for k in specific_keys(keys, generic) for text in project_sft):
            leakage["answer_partial_key_overlap_with_sft"] += 1
        if tier == "fact_holdout":
            cited = " ".join(cpt_text.get(chunk or "", "") for chunk in [evidence["chunk_id"] for evidence in item["evidence"]] + item.get("cpt_support", []))
            if not all(normalize_text(k) in cited for k in keys):
                checker.fail("tier_rule", f"{where}: fact_holdout not supported by CPT")
            leakage["fact_holdout_answers_in_cpt_by_design"] += 1
            leakage["fact_holdout_reassigned_from_document_holdout"] += "reassigned_from" in item
        if tier == "document_holdout":
            evidence_docs = {evidence["document_id"] for evidence in item["evidence"]}
            if not evidence_docs & set(holdout["document_ids"]):
                checker.fail("tier_rule", f"{where}: not from a held-out document")
            if evidence_docs - set(holdout["document_ids"]):
                leakage["document_holdout_evidence_partly_trained"] += 1
                if item["closed_book_expected_behavior"] != "not_scored":
                    checker.fail("tier_rule", f"{where}: closed-book behavior scored on partly trained evidence")
            if answer_present(keys, project_cpt, generic):
                leakage["document_holdout_answers_in_cpt"] += 1
                checker.fail("leakage", f"{where}: answer found in CPT of its project")
            elif answer_present(keys, cpt_text.values(), generic):
                leakage["document_holdout_keys_only_in_other_project_cpt"] += 1

    missing = sorted(TIERS - set(tiers))
    if missing:
        checker.fail("tier_rule", f"empty tiers: {missing}")
    return {
        "passed": not checker.errors,
        "errors": dict(sorted(checker.errors.items())),
        "examples": {k: checker.examples[k] for k in sorted(checker.examples)},
        "evaluation_by_tier": dict(sorted(tiers.items())),
        "evaluation_by_behavior": dict(sorted(behaviors.items())),
        "provenance_valid_percent": tally.percent(),
        "rows_checked": dict(sorted(tally.total.items())),
        "leakage": leakage,
        "docs_raw_tree_sha256": current_tree,
    }
