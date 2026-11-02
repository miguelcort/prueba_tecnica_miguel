"""Questions whose honest answer is "the documentation does not say".

Two sources: stubs a human reviewed, and impostors, where service B is asked about a real term of
service A after verifying that B's documentation never mentions it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from ..common import normalize_text, rank, stable_id
from .facts import locate_evidence


def _section_at(doc: dict, line: int) -> Optional[str]:
    sections = doc.get("sections")
    return " > ".join(sections[line - 1]) or None if sections else None


def _project_text(documents: List[dict]) -> Dict[str, str]:
    parts: Dict[str, List[str]] = defaultdict(list)
    for doc in documents:
        parts[doc["project"]].extend(doc["normalized_lines"])
    return {project: normalize_text("\n".join(lines)) for project, lines in parts.items()}


def stub_abstentions(entries: List[dict], by_path: Dict[str, dict], language: Dict[str, str]) -> List[dict]:
    rows = []
    for entry in entries:
        doc = by_path.get(entry["source_path"])
        quotes = entry["evidence"] if isinstance(entry["evidence"], list) else [entry["evidence"]]
        location = locate_evidence(doc["normalized_lines"], quotes) if doc else None
        if location is None:
            continue
        rows.append(
            {
                "abstention_id": stable_id("abs", entry["id"]),
                "reason": "stub_or_placeholder",
                "project": doc["project"],
                "language": language[doc["project"]],
                "answer": entry["answer"],
                "train_questions": entry["train_questions"],
                "eval_question": entry["eval_question"],
                "must_not_contain": entry.get("must_not_contain", []),
                "absent_terms": [],
                "wrong_claim": None,
                "document_id": doc["document_id"],
                "source_file": doc["source_file"],
                "source_hash": doc["source_hash"],
                "section": _section_at(doc, location[0]),
                "line_start": location[0],
                "line_end": location[1],
            }
        )
    return rows


QUESTION_POOL = {
    ("endpoint", "es"): ["¿Qué hace `{term}` en {target}?", "Explica el endpoint `{term}` de {target}.", "¿Para qué sirve llamar a `{term}` en {target}?", "¿Cuándo debo usar `{term}` de {target}?"],
    ("endpoint", "en"): ["What does `{term}` do in {target}?", "Explain the `{term}` endpoint of {target}.", "What is a call to `{term}` for in {target}?", "When should I use `{term}` of {target}?"],
    ("table", "es"): ["¿Qué columnas tiene la tabla `{term}` en {target}?", "Enumera las columnas de la tabla `{term}` de {target}.", "¿Qué campos componen la tabla `{term}` en {target}?"],
    ("table", "en"): ["Which columns does table `{term}` have in {target}?", "List the columns of table `{term}` in {target}.", "What fields make up the `{term}` table in {target}?"],
}


def _questions(kind: str, language: str, term: str, target: str, seed: int) -> tuple:
    """The last wording of the seeded order stays out of training."""
    pool = QUESTION_POOL[("endpoint" if kind == "endpoint" else "table", language)]
    ordered = sorted(pool, key=lambda wording: rank(seed, "abstain_wording", term, target, wording))
    questions = [wording.format(term=term, target=target) for wording in ordered]
    return questions[:-1], questions[-1]


def impostor_abstentions(facts: List[dict], documents: List[dict], language: Dict[str, str], held: set, seed: int, per_project: int) -> List[dict]:
    """Verified absence uses every version of the target project, not only the canonical one."""
    project_text = _project_text(documents)
    candidates = sorted(
        (f for f in facts if f["kind"] in ("endpoint", "table_columns") and f["version_scope"] == "canonical" and f["document_id"] not in held),
        key=lambda f: rank(seed, "impostor", f["fact_id"]),
    )
    rows, used = [], set()
    for target in sorted(project_text):
        taken = 0
        lang = language[target]
        for fact in candidates:
            if taken == per_project:
                break
            extra = fact["extra"]
            term = f"{extra['method']} {extra['path']}" if fact["kind"] == "endpoint" else extra["table"]
            needle = extra["path"] if fact["kind"] == "endpoint" else extra["table"]
            if fact["project"] == target or term in used or normalize_text(needle) in project_text[target]:
                continue
            used.add(term)
            taken += 1
            train, held_out = _questions(fact["kind"], lang, term, target, seed)
            if lang == "es":
                answer = f"La documentación de {target} no menciona `{term}`, así que no puedo confirmar que exista en ese servicio."
                claim = f"En {target}, `{term}` se usa para: {extra.get('purpose')}"
            else:
                answer = f"The {target} documentation does not mention `{term}`, so I cannot confirm that it exists in that service."
                claim = f"In {target}, `{term}` is used for: {extra.get('purpose')}."
            rows.append(
                {
                    "abstention_id": stable_id("abs", target, term),
                    "reason": "absent_from_target_service",
                    "project": target,
                    "language": lang,
                    "answer": answer,
                    "train_questions": train,
                    "eval_question": held_out,
                    "must_not_contain": [extra["purpose"]] if fact["kind"] == "endpoint" else [],
                    "absent_terms": [needle],
                    "wrong_claim": claim if fact["kind"] == "endpoint" else None,
                    "document_id": fact["document_id"],
                    "source_file": fact["source_file"],
                    "source_hash": fact["source_hash"],
                    "section": fact["section"],
                    "line_start": fact["line_start"],
                    "line_end": fact["line_end"],
                }
            )
    return rows
