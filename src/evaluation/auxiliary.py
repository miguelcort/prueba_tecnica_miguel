"""What the experiment needs beyond eval.jsonl: a training validation split, a neutral tier, and a
floor showing what a fixed answer scores on key_facts. None of this changes key_facts."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from ..common import normalize_question, normalize_text, rank, same_question


def passes_constant(item: dict, answer: str) -> bool:
    """The factual accuracy rule of EXPERIMENT.md applied to a fixed answer."""
    text = normalize_text(answer)
    keys = item["key_facts"]
    return bool(keys) and all(normalize_text(key) in text for key in keys) and not any(normalize_text(word) in text for word in item["must_not_contain"])


def constant_answers(item: dict, generic: List[str]) -> Dict[str, str]:
    """generic_only never looks at the row. The other two repeat the question, as a weak model does."""
    fixed = " ".join(generic)
    return {"generic_only": fixed, "question_echo": item["question"], "question_echo_plus_generic": f"{item['question']} {fixed}"}


def _sample(groups: Dict[str, List[str]], fraction: float, seed: int, salt: str) -> List[str]:
    """Per project, the first groups by seeded hash. A project keeps everything if it rounds to zero."""
    chosen: List[str] = []
    for project in sorted(groups):
        ordered = sorted(groups[project], key=lambda group: rank(seed, salt, group))
        chosen += ordered[: int(len(ordered) * fraction + 0.5)]
    return chosen


def training_validation_split(sft_rows: List[dict], cpt_rows: List[dict], probe_facts: set, protected_chunks: set, fraction: float, seed: int) -> dict:
    """Rows to keep out of training, to choose the epoch and detect overfitting.

    SFT goes by fact, because the questions of a fact share one answer. CPT goes by section, because
    split sections overlap. Whatever an evaluation or probe row needs stays in training, or that row
    would test something the arm never saw.
    """
    sft_groups: Dict[str, set] = defaultdict(set)
    for row in sft_rows:
        if row["metadata"]["fact_id"] not in probe_facts:
            sft_groups[row["metadata"]["project"]].add(row["metadata"]["fact_id"])
    sft_facts = set(_sample({p: sorted(g) for p, g in sft_groups.items()}, fraction, seed, "validation_sft"))

    sections: Dict[str, List[dict]] = defaultdict(list)
    for row in cpt_rows:
        sections["|".join((row["provenance"]["document_id"], row["provenance"]["section"]))].append(row)
    cpt_groups: Dict[str, List[str]] = defaultdict(list)
    for key, members in sections.items():
        if not any(row["provenance"]["chunk_id"] in protected_chunks for row in members):
            cpt_groups[members[0]["metadata"]["project"]].append(key)
    cpt_sections = set(_sample(cpt_groups, fraction, seed, "validation_cpt"))
    return {
        "note": "Keep these rows out of training. Selected by seeded hash; nothing here has been used to train.",
        "fraction": fraction,
        "sft_validation_ids": sorted(row["id"] for row in sft_rows if row["metadata"]["fact_id"] in sft_facts),
        "sft_validation_fact_ids": sorted(sft_facts),
        "cpt_validation_ids": sorted(row["id"] for key in cpt_sections for row in sections[key]),
        "cpt_validation_chunk_ids": sorted(row["provenance"]["chunk_id"] for key in cpt_sections for row in sections[key]),
    }


def build_neutral(rendered: List[dict], facts: List[dict], sft_rows: List[dict], cpt_by_project: Dict[str, Dict[str, str]], validation: dict, settings: dict) -> List[dict]:
    """Facts that both CPT and SFT teach, under a wording SFT never used, so no adapted arm starts
    without the answer. Rows a fixed answer already passes are out, since they measure nothing."""
    trained_facts = {row["metadata"]["fact_id"]: row for row in sft_rows if row["id"] not in validation["sft_validation_ids"]}
    questions = [row["messages"][0]["content"] for row in sft_rows]
    keys_seen = {normalize_question(question) for question in questions}
    held_chunks = set(validation["cpt_validation_chunk_ids"])
    kept = []
    for item, fact in zip(rendered, facts):
        if fact["fact_id"] not in trained_facts or fact["version_scope"] != "canonical" or fact["kind"] in ("endpoint_list", "table_columns"):
            continue
        chunks = cpt_by_project[fact["project"]]
        cited = " ".join(chunks.get(e["chunk_id"] or "", "") for e in item["evidence"] if e["chunk_id"] not in held_chunks)
        if not cited.strip() or not all(normalize_text(key) in cited for key in item["key_facts"]):
            continue
        if normalize_question(item["question"]) in keys_seen or any(same_question(item["question"], other, settings["question_jaccard_threshold"]) for other in questions):
            continue
        if any(passes_constant(item, answer) for answer in constant_answers(item, settings["generic_keys"]).values()):
            continue
        kept.append(item)
    return kept
