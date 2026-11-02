"""Heading-aware chunks that never split a table row or a code fence.

Tables keep their header on every part. Prose gets overlap only when one section is split.
Specs and diagrams are not text a model can use, so they are rendered from their parsed values.
"""
from __future__ import annotations

import re
from typing import Dict, List

import yaml

from ..common import stable_id
from ..preprocessing.extract import HEADING, HTTP_METHODS, is_table_start, section_paths

SENTENCE_END = re.compile(r"(?<=[.!?:])\s+")
ER_ENTITY = re.compile(r"^\s*(\w+)\s*\{\s*$")
ER_COLUMN = re.compile(r"^\s*(\w+)\s+(\w+)(?:\s+(PK|FK))?\s*$")


def split_blocks(lines: List[str]) -> List[dict]:
    """Atomic blocks: heading, table, fence, paragraph or list group."""
    blocks: List[dict] = []
    sections = section_paths(lines)
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        start = index
        if line.lstrip().startswith("```"):
            index += 1
            while index < len(lines) and not lines[index].lstrip().startswith("```"):
                index += 1
            index = min(index + 1, len(lines))
            kind = "code"
        elif is_table_start(lines, index):
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                index += 1
            kind = "table"
        elif HEADING.match(line):
            index += 1
            kind = "heading"
        else:
            while index < len(lines) and lines[index].strip() and not lines[index].lstrip().startswith(("```", "#", "|")):
                index += 1
            index = max(index, start + 1)
            kind = "text"
        blocks.append({"kind": kind, "start": start + 1, "end": index, "lines": lines[start:index], "section": sections[start]})
    return blocks


def _split_table(block: dict, max_chars: int) -> List[dict]:
    header, rows = block["lines"][:2], block["lines"][2:]
    parts, current, first = [], [], block["start"] + 2
    for offset, row in enumerate(rows):
        number = block["start"] + 2 + offset
        if current and sum(len(item) + 1 for item in header + current + [row]) > max_chars:
            parts.append({**block, "lines": header + current, "start": first, "end": number - 1})
            current, first = [], number
        current.append(row)
    if current:
        parts.append({**block, "lines": header + current, "start": first, "end": block["end"]})
    return parts or [block]


def _split_text(block: dict, max_chars: int, overlap: int) -> List[dict]:
    text = " ".join(line.strip() for line in block["lines"])
    parts, current = [], ""
    for sentence in SENTENCE_END.split(text):
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current)
            current = current[-overlap:].split(" ", 1)[-1] if overlap else ""
        current = f"{current} {sentence}".strip()
    if current:
        parts.append(current)
    return [{**block, "lines": [part]} for part in parts]


def _fit(block: dict, settings: dict) -> List[dict]:
    size = sum(len(line) + 1 for line in block["lines"])
    if size <= settings["max_chars"]:
        return [block]
    if block["kind"] == "table":
        return _split_table(block, settings["max_chars"])
    if block["kind"] == "text":
        return _split_text(block, settings["max_chars"], settings["overlap_chars"])
    return [block]  # a code fence stays whole; the fence limit already bounded it


def _body(parts: List[dict]) -> str:
    return "\n\n".join("\n".join(part["lines"]) for part in parts).strip()


def chunk_markdown(lines: List[str], settings: dict) -> List[dict]:
    """Greedy packing inside a section. A new heading closes the chunk once it is big enough."""
    groups: List[List[dict]] = []
    current: List[dict] = []
    size = 0
    for block in split_blocks(lines):
        if block["kind"] == "heading" and current and size >= settings["min_chars"]:
            groups.append(current)
            current, size = [], 0
        for piece in _fit(block, settings):
            piece_size = sum(len(line) + 1 for line in piece["lines"])
            if current and size + piece_size > settings["max_chars"]:
                groups.append(current)
                current, size = [], 0
            current.append(piece)
            size += piece_size
    if current:
        groups.append(current)
    chunks = []
    for group in groups:
        content = [part for part in group if part["kind"] != "heading"]
        if not content:
            continue  # a heading alone has no context to offer
        chunks.append(
            {
                "text": _body(group),
                "line_start": group[0]["start"],
                "line_end": group[-1]["end"],
                "section": " > ".join(content[0]["section"]) or None,
                "has_table": any(part["kind"] == "table" for part in group),
                "has_code": any(part["kind"] == "code" for part in group),
                "derived_from": "markdown",
            }
        )
    return chunks


def render_er_diagram(diagram: dict) -> List[dict]:
    chunks = []
    entity, columns, first = None, [], 0
    for offset, line in enumerate(diagram["lines"]):
        number = diagram["line_start"] + 1 + offset
        start, column = ER_ENTITY.match(line), ER_COLUMN.match(line)
        if start:
            entity, columns, first = start.group(1), [], number
        elif line.strip() == "}" and entity:
            text = f"Table `{entity}` columns: " + ", ".join(f"`{name}`" for name in columns) + "."
            chunks.append({"text": text, "line_start": first, "line_end": number, "section": f"erDiagram > {entity}", "has_table": False, "has_code": False, "derived_from": "er_diagram"})
            entity = None
        elif entity and column:
            columns.append(column.group(2))
    return chunks


def render_exception_flows(diagram: dict, title: str) -> List[dict]:
    pairs = []
    for offset, line in enumerate(diagram["lines"][:-1]):
        exception = re.search(r":\s*(\w+Exception)\s*$", line)
        status = re.search(r"HTTP (\d{3})", diagram["lines"][offset + 1])
        if exception and status:
            pairs.append((diagram["line_start"] + 1 + offset, f"`{exception.group(1)}` returns HTTP {status.group(1)}"))
    if not pairs:
        return []
    text = f"`{title}` error handling: " + "; ".join(item for _, item in pairs) + "."
    return [{"text": text, "line_start": pairs[0][0], "line_end": pairs[-1][0] + 1, "section": title, "has_table": False, "has_code": False, "derived_from": "sequence_diagram"}]


class _LineLoader(yaml.SafeLoader):
    """Records where each mapping lives, so spec facts can cite swagger line numbers."""


def _construct_with_lines(loader: _LineLoader, node: yaml.MappingNode) -> dict:
    mapping = loader.construct_mapping(node, deep=True)
    mapping["__line__"] = node.start_mark.line + 1
    mapping["__end__"] = node.end_mark.line
    return mapping


_LineLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_with_lines)


def load_spec(lines: List[str]) -> dict:
    return yaml.load("\n".join(lines), Loader=_LineLoader)


def resolve_ref(spec: dict, item: dict) -> dict:
    while isinstance(item, dict) and "$ref" in item:
        node = spec
        for part in item["$ref"].lstrip("#/").split("/"):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        item = node
    return item


def plain(text: object) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", str(text)).replace("**", "").split())


def render_spec(lines: List[str], excluded_terms: List[str]) -> List[dict]:
    """One readable chunk per OpenAPI operation, built only from the spec's own values."""
    spec = load_spec(lines)
    chunks = []
    info = spec.get("info") or {}
    description = [plain(line) for line in str(info.get("description") or "").splitlines() if line.strip() and not line.lstrip().startswith("- **")]
    if info.get("title") and description:
        chunks.append({"text": f"{info['title']}: " + " ".join(description), "line_start": info["__line__"], "line_end": info["__end__"], "section": "info", "has_table": False, "has_code": False, "derived_from": "openapi_spec"})
    for path, path_item in (spec.get("paths") or {}).items():
        if path.startswith("__"):
            continue
        for method in HTTP_METHODS:
            operation = path_item.get(method)
            if not operation:
                continue
            name = f"{method.upper()} {path}"
            parts = [f"`{name}`:"]
            for field in ("summary", "description"):
                if operation.get(field):
                    sentence = plain(operation[field])
                    parts.append(sentence if sentence.endswith((".", ":", "?")) else sentence + ".")
            params = []
            for raw in operation.get("parameters") or []:
                param = resolve_ref(spec, raw)
                if param.get("name") and not any(term in str(param["name"]) for term in excluded_terms):
                    schema = param.get("schema") or {}
                    kind = schema.get("format") if schema.get("type") == "object" and schema.get("format") else schema.get("type")
                    allowed = f", allowed: {', '.join(str(v) for v in schema['enum'])}" if schema.get("enum") else ""
                    params.append(f"`{param['name']}` ({param.get('in', '')}, {kind}, required={bool(param.get('required'))}{allowed})")
            if params:
                parts.append("Parameters: " + "; ".join(params) + ".")
            codes = sorted(str(code) for code in (operation.get("responses") or {}) if not str(code).startswith("__"))
            if codes:
                parts.append("Responses: " + ", ".join(codes) + ".")
            chunks.append({"text": " ".join(parts), "line_start": path_item["__line__"] - 1, "line_end": operation["__end__"], "section": name, "has_table": False, "has_code": False, "derived_from": "openapi_spec"})
    return chunks


def finalize(chunks: List[dict], document: dict, usage: str) -> List[dict]:
    rows = []
    for chunk in chunks:
        rows.append(
            {
                **chunk,
                "chunk_id": stable_id("chunk", document["document_id"], chunk["line_start"], chunk["line_end"], chunk["text"]),
                "document_id": document["document_id"],
                "project": document["project"],
                "version": document["version"],
                "source_file": document["source_file"],
                "source_hash": document["source_hash"],
                "usage": usage,
                "char_count": len(chunk["text"]),
            }
        )
    return rows


def chunk_document(document: Dict[str, object], settings: dict, usage: str, excluded_terms: List[str]) -> List[dict]:
    lines = document["lines"]
    if document["role"] == "spec":
        return finalize(render_spec(lines, excluded_terms), document, usage)
    chunks = chunk_markdown(lines, settings)
    for diagram in document["diagrams"]:
        if diagram["kind"] == "erDiagram":
            chunks.extend(render_er_diagram(diagram))
        elif diagram["kind"] == "sequenceDiagram":
            title = next((document["raw_lines"][i].strip() for i in range(diagram["line_start"] - 2, -1, -1) if document["raw_lines"][i].strip()), "")
            if re.match(r"^(GET|POST|PUT|DELETE)\s+/", title):
                chunks.extend(render_exception_flows(diagram, title))
    return finalize(chunks, document, usage)
