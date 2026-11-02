"""Content extraction: what structure each cleaned document has."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
ENDPOINT = re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+(/[^\s`|)]*)")
TABLE_SEPARATOR = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$")
HTTP_METHODS = ("get", "post", "put", "delete", "patch")


def section_paths(lines: List[str]) -> List[List[str]]:
    """For every line, the chain of headings it lives under (fences are skipped)."""
    stack: List[Tuple[int, str]] = []
    paths: List[List[str]] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            level = len(match.group(1))
            stack = [item for item in stack if item[0] < level] + [(level, match.group(2).replace("`", ""))]
        paths.append([title for _, title in stack])
    return paths


def split_row(line: str) -> List[str]:
    cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
    return [cell.strip().replace("`", "").replace("**", "") for cell in cells]


def is_table_start(lines: List[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and lines[index].lstrip().startswith("|")
        and bool(TABLE_SEPARATOR.match(lines[index + 1]))
        and "-" in lines[index + 1]
    )


def structure_summary(lines: List[str]) -> Dict[str, object]:
    sections = 0
    code_blocks = 0
    tables = 0
    endpoints = set()
    parameters = set()
    in_fence = False
    header: List[str] = []
    for index, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            code_blocks += int(in_fence)
            continue
        for method, path in ENDPOINT.findall(line):
            endpoints.add(f"{method} {path}")
        if in_fence:
            continue
        sections += int(bool(HEADING.match(line)))
        if is_table_start(lines, index):
            tables += 1
            header = [cell.lower() for cell in split_row(line)]
        elif line.lstrip().startswith("|") and header[:1] in (["parameter"], ["header"], ["field"]):
            name = split_row(line)[0]
            if name and not set(name) <= set("-: "):
                parameters.add(name)
        elif not line.strip():
            header = []
    return {
        "section_count": sections,
        "code_block_count": code_blocks,
        "table_count": tables,
        "endpoints": sorted(endpoints),
        "parameters": sorted(parameters),
    }
