"""Markdown normalization, noise removal and redaction.

Line counts never change: a removed line becomes empty. So a line number computed on cleaned
text is also valid against the raw file, which is what makes provenance checkable.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from typing import List, Optional, Tuple

FENCE = re.compile(r"^\s*```(.*)$")
HTML_TAG = re.compile(r"</?(?:p|a|img|div|h[1-6]|i|span|br|details|summary|strong|em|ul|li|table|tr|td|th|center)\b[^>]*>", re.IGNORECASE)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]*)\]\((?:[^()]|\([^)]*\))*\)")
HTML_COMMENT = re.compile(r"<!--.*?-->")
TEMPLATE_JUNK = re.compile(r"^\s*Download .* Template\s*$")
STUB_LINE = re.compile(r"^\s*_?TO\s?DO\.?_?\s*$", re.IGNORECASE)
BASE64_BLOB = re.compile(r"[A-Za-z0-9+/=]{60,}")
GLUED_HEADING = re.compile(r"^(#{1,6})([A-Z])")
CLOSED_HEADING = re.compile(r"^(#{1,6}\s.*?)\s+#+\s*$")
BOLD_HEADING = re.compile(r"^(#{1,6}\s+)\*\*(.+?)\*\*\s*$")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏﻿]")

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
HOST = re.compile(r"(?<![\w@-])(?:\*\.)?(?:[A-Za-z0-9-]+\.)+(?:com|io|net|org)(?:\.[a-z]{2})?(?![\w-])")
SLACK_CHANNEL = re.compile(r"(?<![\w&/])#[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b")
PUBLIC_HOSTS = ("docs.aws.amazon.com",)


def normalize_lines(text: str) -> List[str]:
    """Line endings, invisible characters, entities and spacing. Tables and fences survive."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = html.unescape(line).replace("\n", " ")
        line = unicodedata.normalize("NFC", line.replace(" ", " "))
        lines.append(CONTROL.sub("", line).rstrip())
    return lines


def redact(text: str, rules: dict, counts: Counter) -> str:
    """Mask personal data and internal infrastructure."""

    def mask_host(match: "re.Match[str]") -> str:
        if match.group(0) in PUBLIC_HOSTS:
            return match.group(0)
        counts["internal_host"] += 1
        return "<internal-host>"

    text, emails = EMAIL.subn("<email>", text)
    counts["email"] += emails
    text = HOST.sub(mask_host, text)
    text, channels = SLACK_CHANNEL.subn("<slack-channel>", text)
    counts["slack_channel"] += channels
    for rule in rules.get("redactions", []):
        text, replaced = re.subn(rule["pattern"], rule["replacement"], text)
        counts[rule["label"]] += replaced
    return text


def apply_quarantine(path: str, lines: List[str], rules: dict) -> Tuple[List[str], List[dict]]:
    """Blank lines a human marked as contradictory or unsafe. Each removal is logged."""
    result, log = list(lines), []
    for entry in rules.get("lines", []):
        if entry["path"] != path:
            continue
        for index, line in enumerate(result):
            number = index + 1
            in_range = "range" in entry and entry["range"][0] <= number <= entry["range"][1]
            matches = "pattern" in entry and re.search(entry["pattern"], line)
            if line.strip() and (in_range or matches):
                result[index] = ""
                log.append({"path": path, "line": number, "reason": entry["reason"]})
    return result, log


def clean_prose(line: str) -> str:
    line = HTML_COMMENT.sub("", line)
    line = IMAGE.sub("", line)
    line = LINK.sub(r"\1", line)
    line = HTML_TAG.sub("", line)
    line = BASE64_BLOB.sub("", line)
    if TEMPLATE_JUNK.match(line) or STUB_LINE.match(line):
        return ""
    line = GLUED_HEADING.sub(r"\1 \2", line)
    line = CLOSED_HEADING.sub(r"\1", line)
    line = BOLD_HEADING.sub(r"\1\2", line)
    return line.rstrip()


def _diagram_kind(body: List[str]) -> Optional[str]:
    for line in body:
        token = line.strip().split(" ")[0] if line.strip() else ""
        if token and token[0] not in "%'}":
            return token
    return None


def strip_blocks(lines: List[str], raw: List[str], max_fence_lines: int, counts: Counter) -> Tuple[List[str], List[dict]]:
    """Clean prose and walk fences. Diagrams go apart; they are parsed, never trained on verbatim."""
    result = [""] * len(lines)
    diagrams: List[dict] = []
    index, in_comment = 0, False
    while index < len(lines):
        line = lines[index]
        if in_comment:
            in_comment = "-->" not in line
            index += 1
            continue
        if "<!--" in line and "-->" not in line:
            in_comment = True
            index += 1
            continue
        fence = FENCE.match(line)
        if not fence:
            result[index] = clean_prose(line)
            index += 1
            continue
        end = index + 1
        while end < len(lines) and not FENCE.match(lines[end]):
            end += 1
        end = min(end, len(lines) - 1)
        body = lines[index + 1 : end]
        language = fence.group(1).strip().split(" ")[0].lower()
        if language in ("mermaid", "plantuml"):
            diagrams.append({"kind": _diagram_kind(body), "line_start": index + 1, "line_end": end + 1, "lines": raw[index + 1 : end]})
            counts["diagram_blocks"] += 1
        elif not any(item.strip() for item in body):
            counts["empty_fences"] += 1
        elif len(body) > max_fence_lines:
            counts["long_fences"] += 1
        else:
            result[index : end + 1] = [line] + body + [lines[end]]
        index = end + 1
    return result, diagrams
