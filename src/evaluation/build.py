"""Stage 5, evaluation: questions whose answers never appear in SFT, by explicit tier.

  fact_holdout      the answer is absent from SFT and present in CPT prose: closed-book transfer.
  document_holdout  the whole page is out of CPT and SFT: only retrieval can answer; closed-book
                    arms should abstain.
  unanswerable      the documentation does not contain the answer.

Facts that stayed in SFT are out of this set. Their unseen wordings go to a separate diagnostic.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

from ..common import Config, iter_jsonl, normalize_question, normalize_text, rank, read_json, require, same_question, stable_id, token_jaccard, write_json, write_jsonl
from ..provenance.provenance import evidence_of, from_fact
from ..sft.build import abstention_difficulty, abstention_provenance
from ..sft.templates import difficulty_of, render


def answer_present(keys: List[str], text: str, min_chars: int) -> bool:
    """All key facts in one text, when together they are specific enough to prove something."""
    return bool(keys) and len(" ".join(keys)) >= min_chars and all(normalize_text(key) in text for key in keys)


def _from_fact(fact: dict, tier: str, closed_book: str, siblings: List[dict], duplicates: Dict[str, List[str]], seed: int) -> dict:
    _, question, answer = render(fact, seed)
    own = {normalize_text(key) for key in fact["key_facts"]}
    wrong = sorted({k for s in siblings if s["fact_id"] != fact["fact_id"] for k in s["key_facts"] if normalize_text(k) not in own and not any(normalize_text(k) in o for o in own)})
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
    min_chars, threshold = settings["min_specific_key_chars"], settings["answer_jaccard_threshold"]
    facts = list(iter_jsonl(config.processed("facts.jsonl")))
    split = read_json(config.processed("fact_split.json"))
    documents = {doc["document_id"]: doc["duplicate_files"] for doc in iter_jsonl(config.processed("documents_cleaned.jsonl"))}
    sft_rows = list(iter_jsonl(config.output("sft")))
    sft_questions = {normalize_question(row["messages"][0]["content"]) for row in sft_rows}
    sft_question_texts = [row["messages"][0]["content"] for row in sft_rows]
    sft_answers = [row["messages"][1]["content"] for row in sft_rows]
    sft_norm = [normalize_text(answer) for answer in sft_answers]
    cpt_chunks = [normalize_text(row["text"]) for row in iter_jsonl(config.output("cpt"))]
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for fact in facts:
        groups[(fact["project"], fact["subject"], fact["attribute"])].append(fact)

    items: List[dict] = []
    probes: List[dict] = []
    dropped: Counter = Counter()
    per_document: Counter = Counter()
    for fact in sorted(facts, key=lambda f: rank(seed, "eval", f["fact_id"])):
        info = split[fact["fact_id"]]
        siblings = groups[(fact["project"], fact["subject"], fact["attribute"])]
        if info["split"] in ("document_holdout", "fact_holdout"):
            candidate = _from_fact(fact, info["split"], "abstain" if info["split"] == "document_holdout" else "answer", siblings, documents, seed)
            in_sft = any(answer_present(fact["key_facts"], text, min_chars) for text in sft_norm) or any(token_jaccard(candidate["expected_answer"], answer) >= threshold for answer in sft_answers)
            if in_sft:
                dropped[f"{info['split']}_answer_also_in_sft"] += 1
            elif info["split"] == "document_holdout" and any(answer_present(fact["key_facts"], text, min_chars) for text in cpt_chunks):
                # A fact repeated elsewhere is not really held out, so it cannot test retrieval.
                dropped["document_holdout_fact_also_in_cpt"] += 1
            elif info["split"] == "document_holdout" and per_document[fact["document_id"]] >= settings["max_items_per_holdout_document"]:
                dropped["document_holdout_over_cap"] += 1
            else:
                per_document[fact["document_id"]] += info["split"] == "document_holdout"
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
    stats = {"items": total, "recall_probe_rows": len(probe_rows), "by_tier": dict(sorted(Counter(i["tier"] for i in kept).items())), "dropped": dict(sorted(dropped.items()))}
    write_json(config.processed("evaluation_stats.json"), stats)
    return stats
