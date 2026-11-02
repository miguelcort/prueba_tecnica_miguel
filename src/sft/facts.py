"""Atomic facts: parsed from trusted structures, or curated and mechanically verified."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ..chunking.chunker import load_spec, plain, resolve_ref
from ..common import normalize_text, stable_id
from ..preprocessing.extract import HTTP_METHODS, is_table_start, section_paths, split_row

CANONICAL, VERSION_SCOPED = "canonical", "version_scoped"
DESCRIBED_ENDPOINT = re.compile(r"^#{2,6}\s+\d+\.\s+(GET|POST|PUT|DELETE|PATCH)\s+(\S+)\s*$")
DESCRIPTION_LINE = re.compile(r"^\*\*Description\*\*:\s*(.+)$")
ER_ENTITY = re.compile(r"^\s*(\w+)\s*\{\s*$")
ER_COLUMN = re.compile(r"^\s*(\w+)\s+(\w+)(?:\s+(PK|FK))?\s*$")

CATEGORY_BY_KIND = {
    "endpoint": "api_functionality",
    "endpoint_list": "api_functionality",
    "responses": "api_functionality",
    "parameter": "parameters",
    "column": "parameters",
    "table_columns": "parameters",
    "error_code": "error_handling",
    "error_cause": "error_handling",
    "http_action": "error_handling",
    "troubleshooting": "error_handling",
    "alert": "error_handling",
    "exception": "error_handling",
    "queue": "dependencies",
    "database": "architecture",
    "stack": "architecture",
    "cron": "processing_flow",
}

RowResult = Tuple[str, str, str, str, List[str], dict]


def _endpoint(method: str, path: str, purpose: str) -> RowResult:
    return ("endpoint", f"{method} {path}", "purpose", purpose, [method, path], {"method": method, "path": path, "purpose": purpose})


def _column_row(c: List[str], sections: List[str]) -> Optional[RowResult]:
    name, type_, constraints, description, _ = c
    if name == "id":
        return None
    table = sections[0].split(" ")[0] if sections else "unknown"
    value = f"{type_}, {constraints.lower()}" + (f", {description}" if description else "")
    return ("column", f"{table}.{name}", "definition", value, [type_], {"table": table, "column": name, "type": type_, "constraints": constraints.lower(), "description": description})


def _param_row(c: List[str], sections: List[str]) -> RowResult:
    name, type_, required, description = c
    operation = sections[-2] if len(sections) >= 2 else sections[-1]
    extra = {"operation": operation, "name": name, "type": type_, "required": required.lower() == "yes", "description": description}
    return ("parameter", f"{operation} parameter {name}", "details", f"{type_}, required={required}, {description}", [type_], extra)


def _env(sections: List[str]) -> str:
    return sections[-1] if sections else "default"


# Header signatures are exact: an unknown table is ignored rather than guessed at.
TABLE_RULES: Dict[tuple, Callable[[List[str], List[str]], Optional[RowResult]]] = {
    ("method", "endpoint", "description"): lambda c, s: _endpoint(c[0], c[1], c[2]),
    ("method", "operation", "path"): lambda c, s: _endpoint(c[0], c[2], c[1]),
    ("field", "type", "constraints", "description", "extra"): _column_row,
    ("parameter", "type", "required", "description"): _param_row,
    ("code", "error", "description"): lambda c, s: ("error_code", c[1], "meaning", f"HTTP {c[0]}: {c[2]}", [c[0]], {"error": c[1], "code": c[0], "description": c[2]}),
    ("error code", "description", "common causes"): lambda c, s: ("error_cause", c[0], "causes", c[2], [c[2]], {"error": c[0], "causes": c[2]}),
    ("code", "error type", "description", "action required"): lambda c, s: ("http_action", f"HTTP {c[0]}", "action", c[3], [c[3]], {"code": c[0], "error_type": c[1], "description": c[2], "action": c[3]}),
    ("issue", "symptoms", "solution"): lambda c, s: ("troubleshooting", c[0], "solution", c[2], [c[2]], {"issue": c[0], "symptoms": c[1], "solution": c[2]}),
    ("alert", "description", "url"): lambda c, s: ("alert", c[0], "trigger", c[1], [c[1]], {"alert": c[0], "description": c[1]}),
    ("bigq scope", "topic", "consumer", "path"): lambda c, s: ("queue", f"{_env(s)} scope {c[0]}", "routing", f"{c[1]} -> {c[3]}", [c[1], c[3]], {"env": _env(s), "scope": c[0], "topic": c[1], "consumer": c[2], "path": c[3]}),
    ("type", "schema", "cluster"): lambda c, s: ("database", f"{_env(s)} database", "location", f"{c[1]} on {c[2]}", [c[1], c[2]], {"env": _env(s), "type": c[0], "schema": c[1], "cluster": c[2]}),
    ("name", "timerule", "endpoint"): lambda c, s: ("cron", c[0], "schedule", f"{c[1]} -> {c[2]}", [c[1], c[2]], {"name": c[0], "rule": c[1], "endpoint": c[2]}),
    ("layer", "technology", "purpose"): lambda c, s: ("stack", c[0], "technology", c[1], [c[1]], {"layer": c[0], "technology": c[1], "purpose": c[2]}),
}


class FactFactory:
    def __init__(self, spanish_projects: Iterable[str]):
        self.spanish = set(spanish_projects)

    def language(self, project: str) -> str:
        return "es" if project in self.spanish else "en"

    def make(self, doc: dict, kind: str, subject: str, attribute: str, value: str, keys: List[str], extra: dict, start: int, end: int, section: List[str], extractor: str, category: Optional[str] = None) -> dict:
        return {
            "fact_id": stable_id("fact", doc["project"], doc["version"], subject, attribute, value),
            "project": doc["project"],
            "version": doc["version"],
            "version_scope": CANONICAL if doc["is_canonical"] else VERSION_SCOPED,
            "subject": subject,
            "attribute": attribute,
            "value": value,
            "language": self.language(doc["project"]),
            "kind": kind,
            "category": category or CATEGORY_BY_KIND[kind],
            "document_id": doc["document_id"],
            "source_file": doc["source_file"],
            "relative_path": doc["relative_path"],
            "source_hash": doc["source_hash"],
            "line_start": start,
            "line_end": end,
            "section": " > ".join(section) or None,
            "extractor": extractor,
            "key_facts": keys,
            "extra": extra,
        }


def extract_tables(doc: dict, factory: FactFactory) -> List[dict]:
    facts: List[dict] = []
    lines = doc["lines"]
    sections = section_paths(lines)
    index = 0
    while index < len(lines):
        if not is_table_start(lines, index):
            index += 1
            continue
        header = tuple(cell.lower() for cell in split_row(lines[index]))
        rows, cursor = [], index + 2
        while cursor < len(lines) and lines[cursor].lstrip().startswith("|"):
            cells = split_row(lines[cursor])
            if len(cells) == len(header) and any(cells):
                rows.append((cursor + 1, cells))
            cursor += 1
        rule = TABLE_RULES.get(header)
        section = sections[index]
        if rule:
            for number, cells in rows:
                result = rule(cells, section)
                if result:
                    facts.append(factory.make(doc, *result, number, number, section, "markdown_table"))
            if header[0] == "field":
                names = [cells[0] for _, cells in rows]
                entity = section[0].split(" ")[0] if section else "unknown"
                facts.append(factory.make(doc, "table_columns", f"table {entity}", "columns", ", ".join(names), names, {"table": entity, "columns": names}, index + 1, cursor, section, "markdown_table"))
            if header == ("method", "operation", "path"):
                paths = [cells[2] for _, cells in rows]
                facts.append(factory.make(doc, "endpoint_list", "endpoints", "paths", ", ".join(paths), paths, {"paths": paths}, index + 1, cursor, section, "markdown_table"))
        index = cursor
    return facts


def extract_described_endpoints(doc: dict, factory: FactFactory) -> List[dict]:
    facts = []
    lines = doc["lines"]
    sections = section_paths(lines)
    for index, line in enumerate(lines[:-1]):
        heading = DESCRIBED_ENDPOINT.match(line)
        description = DESCRIPTION_LINE.match(lines[index + 1]) if heading else None
        if heading and description:
            facts.append(factory.make(doc, *_endpoint(heading.group(1), heading.group(2), description.group(1)), index + 1, index + 2, sections[index], "described_endpoint"))
    return facts


def extract_er_diagrams(doc: dict, factory: FactFactory) -> List[dict]:
    facts = []
    for diagram in doc["diagrams"]:
        if diagram["kind"] != "erDiagram":
            continue
        entity, columns, first = None, [], 0
        for offset, line in enumerate(diagram["lines"]):
            number = diagram["line_start"] + 1 + offset
            start, column = ER_ENTITY.match(line), ER_COLUMN.match(line)
            if start:
                entity, columns, first = start.group(1), [], number
            elif line.strip() == "}" and entity:
                facts.append(factory.make(doc, "table_columns", f"table {entity}", "columns", ", ".join(columns), columns, {"table": entity, "columns": columns}, first, number, ["erDiagram", entity], "er_diagram"))
                entity = None
            elif entity and column:
                columns.append(column.group(2))
    return facts


def extract_exception_flows(doc: dict, factory: FactFactory) -> List[dict]:
    facts = []
    for diagram in doc["diagrams"]:
        if diagram["kind"] != "sequenceDiagram":
            continue
        title = next((doc["lines"][i].strip() for i in range(diagram["line_start"] - 2, -1, -1) if doc["lines"][i].strip()), "")
        if not re.match(r"^(GET|POST|PUT|DELETE)\s+/", title):
            continue
        for offset, line in enumerate(diagram["lines"][:-1]):
            exception = re.search(r":\s*(\w+Exception)\s*$", line)
            status = re.search(r"HTTP (\d{3})", diagram["lines"][offset + 1])
            if exception and status:
                number = diagram["line_start"] + 1 + offset
                extra = {"operation": title, "exception": exception.group(1), "status": status.group(1)}
                facts.append(factory.make(doc, "exception", f"{title} raises {exception.group(1)}", "http_status", status.group(1), [status.group(1), exception.group(1)], extra, number, number + 1, [title], "sequence_diagram"))
    return facts


def _param_type(schema: dict) -> str:
    if schema.get("type") == "object" and schema.get("format"):
        return str(schema["format"])
    return str(schema.get("type") or schema.get("format") or "")


def extract_spec(doc: dict, factory: FactFactory, prefer_description: Iterable[str]) -> List[dict]:
    spec = load_spec(doc["lines"])
    facts: List[dict] = []
    for path, path_item in (spec.get("paths") or {}).items():
        if path.startswith("__"):
            continue
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not operation:
                continue
            name = f"{method.upper()} {path}"
            start, end = path_item["__line__"] - 1, operation["__end__"]
            purpose = operation.get("description") if doc["project"] in prefer_description and operation.get("description") else operation.get("summary")
            if purpose:
                sentence = re.split(r"(?<=[.])\s", plain(purpose))[0]
                facts.append(factory.make(doc, *_endpoint(method.upper(), path, sentence), start, end, [name], "openapi_spec"))
            for raw in operation.get("parameters") or []:
                param = resolve_ref(spec, raw)
                ptype = _param_type(param.get("schema") or {})
                if not param.get("name") or not ptype:
                    continue
                description = plain(param.get("description") or "")
                extra = {"operation": name, "name": param["name"], "type": ptype, "required": bool(param.get("required")), "description": description, "location": param.get("in", ""), "anchor_lines": [start, end]}
                facts.append(factory.make(doc, "parameter", f"{name} parameter {param['name']}", "details", f"{ptype}, required={param.get('required')}, {description}", [ptype], extra, param["__line__"], param["__end__"], [name], "openapi_spec"))
            codes = sorted(str(code) for code in (operation.get("responses") or {}) if not str(code).startswith("__"))
            if len(codes) >= 2:
                facts.append(factory.make(doc, "responses", name, "responses", ", ".join(codes), codes, {"operation": name, "codes": codes}, start, end, [name], "openapi_spec"))
    return facts


def locate_evidence(lines: List[str], quotes: List[str]) -> Optional[Tuple[int, int]]:
    """Find every quote in a line; returns the covering line range or None."""
    numbers = []
    normalized = [normalize_text(line) for line in lines]
    for quote in quotes:
        needle = normalize_text(quote)
        found = next((i + 1 for i, line in enumerate(normalized) if needle in line), None)
        if found is None:
            return None
        numbers.append(found)
    return min(numbers), max(numbers)


def curated_fact(entry: dict, docs: Dict[str, dict], factory: FactFactory, rejected: List[dict]) -> Optional[dict]:
    identifier = entry["id"]
    doc = docs.get(entry["source_path"])
    if doc is None:
        rejected.append({"id": identifier, "reason": "curated_source_not_usable"})
        return None
    quotes = entry["evidence"] if isinstance(entry["evidence"], list) else [entry["evidence"]]
    location = locate_evidence(doc["lines"], quotes)
    if location is None:
        rejected.append({"id": identifier, "reason": "curated_evidence_not_found"})
        return None
    start, end = location
    region = normalize_text(" ".join(doc["lines"][start - 1 : end]))
    if not all(normalize_text(key) in region for key in entry["key_facts"]):
        rejected.append({"id": identifier, "reason": "curated_key_fact_not_in_evidence"})
        return None
    extra = {"train_questions": entry["train_questions"], "eval_question": entry["eval_question"], "question_type": entry.get("question_type"), "evidence_quotes": quotes, "derived_tokens": entry.get("derived_tokens", [])}
    return factory.make(doc, entry["kind"], identifier, "curated", entry["answer"], entry["key_facts"], extra, start, end, section_paths(doc["lines"])[start - 1], "curated", entry["category"])


def merged_lines(doc: dict) -> List[str]:
    """Cleaned lines with diagrams back in place, so a quote from a diagram can be located."""
    lines = list(doc["lines"])
    for diagram in doc.get("diagrams", []):
        for offset, line in enumerate(diagram["lines"]):
            index = diagram["line_start"] + offset
            if index < len(lines):
                lines[index] = line
    return lines


def multi_source_fact(entry: dict, docs: Dict[str, dict], factory: FactFactory, rejected: List[dict]) -> Optional[dict]:
    """A reasoning row: each passage must exist and carry its own key facts."""
    identifier = entry["id"]
    sources: List[dict] = []
    primary = None
    for reference in entry["sources"]:
        doc = docs.get(reference["source_path"])
        if doc is None:
            rejected.append({"id": identifier, "reason": "curated_source_not_usable"})
            return None
        lines = merged_lines(doc)
        location = locate_evidence(lines, reference["evidence"])
        if location is None:
            rejected.append({"id": identifier, "reason": "curated_evidence_not_found"})
            return None
        start, end = location
        region = normalize_text(" ".join(lines[max(start - 3, 0) : end + 2]))
        if not all(normalize_text(key) in region for key in reference["key_facts"]):
            rejected.append({"id": identifier, "reason": "curated_key_fact_not_in_evidence"})
            return None
        primary = primary or doc
        section = section_paths(doc["lines"])[start - 1]
        sources.append({"document_id": doc["document_id"], "source_file": doc["source_file"], "source_hash": doc["source_hash"], "section": " > ".join(section) or None, "line_start": start, "line_end": end, "chunk_id": None, "is_canonical": doc["is_canonical"]})
    if len({(s["document_id"], s["line_start"]) for s in sources}) < 2:
        rejected.append({"id": identifier, "reason": "curated_needs_two_passages"})
        return None
    keys = [key for reference in entry["sources"] for key in reference["key_facts"]]
    extra = {"train_questions": entry["train_questions"], "eval_question": entry["eval_question"], "sources": sources, "derived_tokens": entry.get("derived_tokens", [])}
    first = sources[0]
    fact = factory.make(primary, "multi_source", identifier, "curated", entry["answer"], keys, extra, first["line_start"], first["line_end"], (first["section"] or "").split(" > ") if first["section"] else [], "curated_multi_source", entry["category"])
    if not all(source["is_canonical"] for source in sources):
        fact["version_scope"] = VERSION_SCOPED
    return fact


def _excluded(fact: dict, rules: List[dict]) -> Optional[str]:
    for rule in rules:
        if rule["project"] == fact["project"] and rule["attribute"] == fact["attribute"] and re.fullmatch(rule["subject"], fact["subject"]):
            return rule["reason"]
    return None


def resolve_conflicts(facts: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Same subject and attribute with other value: drop inside a version, scope across versions."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for fact in facts:
        groups[(fact["project"], fact["subject"], fact["attribute"])].append(fact)
    kept, conflicts = [], []
    for key in sorted(groups):
        group = groups[key]
        canonical = [f for f in group if f["version_scope"] == CANONICAL]
        others = [f for f in group if f["version_scope"] != CANONICAL]
        if len({normalize_text(f["value"]) for f in canonical}) > 1:
            conflicts.append({"key": list(key), "type": "same_version", "action": "excluded", "values": sorted(f["value"] for f in canonical)})
            canonical, others = [], []
        else:
            canonical = canonical[:1]
        base = normalize_text(canonical[0]["value"]) if canonical else None
        unique = {normalize_text(f["value"]): f for f in others if normalize_text(f["value"]) != base}
        others = [unique[k] for k in sorted(unique)]
        if canonical and others:
            canonical[0]["version_scope"] = VERSION_SCOPED
            conflicts.append({"key": list(key), "type": "across_versions", "action": "kept_version_scoped", "values": sorted({canonical[0]["value"]} | {f["value"] for f in others})})
        kept.extend(canonical + others)
    return kept, conflicts


def extract_all(documents: List[dict], quarantine: dict, curated: dict, spanish_projects: List[str]) -> Tuple[List[dict], List[dict], List[dict]]:
    factory = FactFactory(spanish_projects)
    rejected: List[dict] = []
    raw: List[dict] = []
    prefer = quarantine.get("swagger_prefer_description", [])
    for doc in documents:
        if doc["role"] == "spec":
            raw.extend(extract_spec(doc, factory, prefer))
        else:
            raw.extend(extract_tables(doc, factory) + extract_described_endpoints(doc, factory) + extract_er_diagrams(doc, factory) + extract_exception_flows(doc, factory))
    seen_params = set()
    filtered = []
    for fact in raw:
        reason = _excluded(fact, quarantine.get("fact_exclusions", []))
        if reason:
            rejected.append({"id": f"{fact['project']}: {fact['subject']}", "reason": "excluded_by_config"})
            continue
        if fact["kind"] == "parameter":
            signature = (fact["project"], fact["version"], fact["extra"]["name"], fact["value"])
            if signature in seen_params:
                rejected.append({"id": f"{fact['project']}: {fact['subject']}", "reason": "repeated_parameter"})
                continue
            seen_params.add(signature)
        filtered.append(fact)
    facts, conflicts = resolve_conflicts(filtered)
    by_path = {doc["relative_path"]: doc for doc in documents}
    for entry in curated["facts"]:
        fact = curated_fact(entry, by_path, factory, rejected)
        if fact:
            facts.append(fact)
    for entry in curated.get("multi_source", []):
        fact = multi_source_fact(entry, by_path, factory, rejected)
        if fact:
            facts.append(fact)
    facts.sort(key=lambda fact: fact["fact_id"])
    return facts, conflicts, rejected
