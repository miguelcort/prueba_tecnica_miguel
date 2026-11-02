"""Stage 5, evaluation: questions whose answers never appear in SFT, by explicit tier.

  fact_holdout      the answer is absent from SFT and present in CPT prose: closed-book transfer.
  document_holdout  the fact comes from a page out of CPT and SFT, and its project's training text
                    does not contain it: only retrieval can answer; closed-book arms should abstain.
                    A fact that CPT of the same project does contain moves to fact_holdout. When only
                    part of the evidence is out of training, closed-book behavior is not scored.
  unanswerable      the documentation does not contain the answer.

Facts that stayed in SFT are out of this set. Their unseen wordings go to two separate files:
reports/sft_recall_probe.jsonl and reports/neutral_eval.jsonl.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from ..common import Config, iter_jsonl, normalize_question, normalize_text, rank, read_json, require, same_question, stable_id, token_jaccard, write_json, write_jsonl
from ..provenance.provenance import evidence_of, from_fact
from ..sft.build import abstention_difficulty, abstention_provenance
from ..sft.templates import difficulty_of, render
from .auxiliary import build_neutral, training_validation_split


def key_in_text(key: str, text: str) -> bool:
    """Whole-token match on normalized text, so `int` is not found in `print`."""
    return re.search(rf"(?<!\w){re.escape(normalize_text(key))}(?!\w)", text) is not None


def specific_keys(keys: List[str], generic: Iterable[str]) -> List[str]:
    ignored = {normalize_text(key) for key in generic}
    return [key for key in keys if normalize_text(key) not in ignored]


def answer_present(keys: List[str], texts: Iterable[str], generic: Iterable[str]) -> bool:
    """Every specific key somewhere in these texts. Without a specific key nothing can be proven."""
    specific = specific_keys(keys, generic)
    corpus = list(texts)
    return bool(specific) and all(any(key_in_text(key, text) for text in corpus) for key in specific)


def supporting_chunks(keys: List[str], chunks: Dict[str, str], generic: Iterable[str]) -> List[str]:
    """First chunk, in id order, that contains each specific key."""
    return sorted({next(chunk for chunk in sorted(chunks) if key_in_text(key, chunks[chunk])) for key in specific_keys(keys, generic)})


def _from_fact(fact: dict, tier: str, closed_book: str, siblings: List[dict], duplicates: Dict[str, List[str]], seed: int) -> dict:
    _, question, answer = render(fact, seed)
    own = {normalize_text(key) for key in fact["key_facts"]}
    # A value the correct answer itself contains cannot mark an answer as wrong.
    gold = normalize_text(answer)
    wrong = sorted({k for s in siblings if s["fact_id"] != fact["fact_id"] for k in s["key_facts"] if normalize_text(k) not in own and not any(normalize_text(k) in o for o in own) and normalize_text(k) not in gold})
    provenance = from_fact(fact, duplicates)
    return {
        "question": question,
        "expected_answer": answer,
        "expected_behavior": "answer_multi_source" if fact["kind"] == "multi_source" else "answer",
        "closed_book_expected_behavior": closed_book,
        "tier": tier,
        "category": fact["category"],
        "difficulty": difficulty_of(fact),
        "project": fact["project"],
        "language": fact["language"],
        "key_facts": fact["key_facts"],
        "must_not_contain": wrong,
        "fact_id": fact["fact_id"],
        "evidence": evidence_of(provenance),
        "provenance": provenance,
    }


def run(config: Config) -> dict:
    require(config.output("sft"), config.output("cpt"), config.processed("fact_split.json"), config.processed("abstentions.jsonl"))
    settings, seed = config["evaluation"], config["seed"]
    generic, threshold = settings["generic_keys"], settings["answer_jaccard_threshold"]
    facts = list(iter_jsonl(config.processed("facts.jsonl")))
    split = read_json(config.processed("fact_split.json"))
    documents = {doc["document_id"]: doc["duplicate_files"] for doc in iter_jsonl(config.processed("documents_cleaned.jsonl"))}
    sft_rows = list(iter_jsonl(config.output("sft")))
    sft_questions = {normalize_question(row["messages"][0]["content"]) for row in sft_rows}
    sft_question_texts = [row["messages"][0]["content"] for row in sft_rows]
    sft_answers = [row["messages"][1]["content"] for row in sft_rows]
    sft_by_project: Dict[str, List[str]] = defaultdict(list)
    for row in sft_rows:
        sft_by_project[row["metadata"]["project"]].append(normalize_text(row["messages"][1]["content"]))
    cpt_rows = list(iter_jsonl(config.output("cpt")))
    cpt_by_project: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in cpt_rows:
        cpt_by_project[row["metadata"]["project"]][row["provenance"]["chunk_id"]] = normalize_text(row["text"])
    held_documents = set(read_json(config.processed("holdout.json"))["document_ids"])
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for fact in facts:
        groups[(fact["project"], fact["subject"], fact["attribute"])].append(fact)

    items: List[dict] = []
    probes: List[dict] = []
    dropped: Counter = Counter()
    reassigned: Dict[str, str] = {}
    per_document: Counter = Counter()
    for fact in sorted(facts, key=lambda f: rank(seed, "eval", f["fact_id"])):
        info = split[fact["fact_id"]]
        siblings = groups[(fact["project"], fact["subject"], fact["attribute"])]
        if info["split"] in ("document_holdout", "fact_holdout"):
            candidate = _from_fact(fact, info["split"], "abstain" if info["split"] == "document_holdout" else "answer", siblings, documents, seed)
            project_chunks = cpt_by_project[fact["project"]]
            # One SFT answer that holds the whole fact teaches it. CPT counts as a whole, because a
            # document_holdout row claims that nothing in the project's training text can answer it.
            in_sft = any(answer_present(fact["key_facts"], [text], generic) for text in sft_by_project[fact["project"]]) or any(token_jaccard(candidate["expected_answer"], answer) >= threshold for answer in sft_answers)
            if in_sft:
                dropped[f"{info['split']}_answer_also_in_sft"] += 1
            elif info["split"] == "document_holdout" and answer_present(fact["key_facts"], project_chunks.values(), generic):
                # A fact repeated elsewhere is not really held out: CPT teaches it, so it tests transfer.
                reassigned[fact["fact_id"]] = candidate["question"]
                items.append({**candidate, "tier": "fact_holdout", "closed_book_expected_behavior": "answer", "reassigned_from": "document_holdout", "cpt_support": supporting_chunks(fact["key_facts"], project_chunks, generic)})
            elif info["split"] == "document_holdout" and per_document[fact["document_id"]] >= settings["max_items_per_holdout_document"]:
                dropped["document_holdout_over_cap"] += 1
            else:
                per_document[fact["document_id"]] += info["split"] == "document_holdout"
                if info["split"] == "document_holdout" and {e["document_id"] for e in candidate["evidence"]} - held_documents:
                    # Part of the answer was trained, so neither abstaining nor answering is the right behavior.
                    candidate = {**candidate, "closed_book_expected_behavior": "not_scored", "evidence_scope": "partly_trained"}
                items.append(candidate)
        elif info["recall_probe"]:
            probes.append({**_from_fact(fact, "sft_recall_probe", "answer", siblings, documents, seed), "diagnostic_only": True, "note": "Unseen wording of a fact trained in SFT. Not part of the leakage-free evaluation."})

    for row in iter_jsonl(config.processed("abstentions.jsonl")):
        if row["split"] != "evaluation":
            continue
        provenance = abstention_provenance(row)
        items.append(
            {
                "question": row["eval_question"],
                "expected_answer": None,
                "expected_behavior": "abstain",
                "closed_book_expected_behavior": "abstain",
                "tier": "unanswerable",
                "unanswerable_reason": row["reason"],
                "absent_terms": row["absent_terms"],
                "category": "abstention",
                "difficulty": abstention_difficulty(row),
                "project": row["project"],
                "language": row["language"],
                "key_facts": [],
                "must_not_contain": row["must_not_contain"],
                "fact_id": row["abstention_id"],
                "evidence": [{k: v for k, v in provenance.items() if k != "relation"}],
                "provenance": provenance,
            }
        )

    kept, seen = [], set()
    for item in sorted(items, key=lambda i: stable_id("eval", i["tier"], i["question"])):
        key = normalize_question(item["question"])
        if key in seen or any(same_question(item["question"], other, settings["question_jaccard_threshold"]) for other in sft_question_texts):
            dropped["question_also_in_sft_or_duplicate"] += 1
            continue
        seen.add(key)
        kept.append(item)
    total = write_jsonl(config.output("eval"), ({"id": f"eval_{number:06d}", **item} for number, item in enumerate(kept, 1)))
    probe_rows = [p for p in sorted(probes, key=lambda p: p["fact_id"]) if normalize_question(p["question"]) not in sft_questions]
    write_jsonl(config.path("reports_dir") / "sft_recall_probe.jsonl", ({"id": f"probe_{number:06d}", **probe} for number, probe in enumerate(probe_rows, 1)))
    protected_chunks = {e["chunk_id"] for i in kept + probe_rows for e in i["evidence"] if e["chunk_id"]} | {c for i in kept for c in i.get("cpt_support", [])}
    validation = training_validation_split(sft_rows, cpt_rows, {p["fact_id"] for p in probe_rows}, protected_chunks, config["training"]["validation_fraction"], seed)
    write_json(config.path("reports_dir") / "training_validation_split.json", validation)
    eligible = [f for f in sorted(facts, key=lambda f: f["fact_id"]) if split[f["fact_id"]]["split"] == "sft" and not split[f["fact_id"]]["recall_probe"]]
    rendered = [{**_from_fact(f, "neutral", "answer", groups[(f["project"], f["subject"], f["attribute"])], documents, seed), "diagnostic_only": True} for f in eligible]
    neutral = build_neutral(rendered, eligible, sft_rows, cpt_by_project, validation, settings)
    write_jsonl(config.path("reports_dir") / "neutral_eval.jsonl", ({"id": f"neutral_{number:06d}", **row} for number, row in enumerate(neutral, 1)))
    stats = {"items": total, "recall_probe_rows": len(probe_rows), "neutral_rows": len(neutral), "by_tier": dict(sorted(Counter(i["tier"] for i in kept).items())), "dropped": dict(sorted(dropped.items())), "reassigned_to_fact_holdout": sorted(reassigned)}
    write_json(config.processed("evaluation_stats.json"), stats)
    return stats
