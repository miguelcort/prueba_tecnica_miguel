from __future__ import annotations

from collections import Counter

import pytest

from src.chunking.chunker import chunk_markdown
from src.common import PipelineError, same_question
from src.deduplication.dedup import lsh_candidates, minhash, similarity, technical_difference
from src.filtering.filters import rejection_reason
from src.ingestion.ingest import classify_role, manifest_version, resolve_canonical_folder
from src.preprocessing.normalize import normalize_lines, redact, strip_blocks
from src.sft.facts import locate_evidence, resolve_conflicts
from src.evaluation.build import answer_present
from src.provenance.provenance import Verifier
from src.sft.grounding import check_grounding
from src.sft.templates import difficulty_of, is_english
from src.sft.llm import OfflineProvider, complete_with_retries, generate_candidates, provider_from_env

CHUNKING = {"max_chars": 200, "min_chars": 20, "overlap_chars": 30}
FILTERS = {"min_unique_line_ratio": 0.3, "min_substantive_chars": 60}
TEXT = " ".join(f"word{i}" for i in range(200))


def test_canonical_uses_latest_and_never_guesses():
    assert resolve_canonical_folder("p", ["0.2.2", "latest"]) == "latest"
    assert resolve_canonical_folder("p", ["1.1.6"]) == "1.1.6"
    with pytest.raises(PipelineError):
        resolve_canonical_folder("p", ["1.0", "2.0"])


def test_manifest_reveals_the_version_behind_latest():
    assert manifest_version('{"guide": "https://x/p/0.0.3-rc-1/guide/index.html"}', "p") == "0.0.3-rc-1"


def test_roles():
    assert classify_role("p/v/guide/_sidebar.md") == "navigation"
    assert classify_role("p/v/specs/swagger.yaml") == "spec"
    assert classify_role("p/v/guide/x.md") == "markdown"


def test_normalization_preserves_line_count_and_decodes_entities():
    lines = normalize_lines("a &gt; b\r\n\r\nc d  \n\x07e")
    assert lines == ["a > b", "", "c d", "e"]


def test_strip_blocks_keeps_line_numbers_and_separates_diagrams():
    lines = ["# T", "```mermaid", "graph TD", "```", "text", "```json", "{}", "```"]
    cleaned, diagrams = strip_blocks(lines, lines, 25, Counter())
    assert len(cleaned) == len(lines) and cleaned[4] == "text" and cleaned[6] == "{}"
    assert cleaned[2] == "" and diagrams[0]["kind"] == "graph" and diagrams[0]["line_start"] == 2


def test_redaction_masks_people_and_hosts():
    counts: Counter = Counter()
    result = redact("mail a@b.com host api.internal.example.com #team-channel", {}, counts)
    assert "a@b.com" not in result and "example.com" not in result and "#team-channel" not in result


def test_minhash_finds_similar_and_separates_different():
    other = " ".join(f"other{i}" for i in range(200))
    changed = TEXT.replace("word100", "changed")
    a, b, c = (minhash(t, 5, 128, 1) for t in (TEXT, changed, other))
    assert similarity(a, b) > 0.8 and similarity(a, c) < 0.1
    assert ("a", "b") in lsh_candidates({"a": a, "b": b, "c": c}, 16)


def test_technical_difference_detects_a_new_endpoint():
    first = "Use GET /stock for stock."
    second = "Use GET /stock for stock. Use GET /stock/validation for checks."
    difference = technical_difference(first, second)
    assert any("/stock/validation" in token for token in difference["only_in_second"])


def test_short_technical_document_is_kept_and_short_noise_is_rejected():
    short = ["# Limits", "`max_size`: 50kb on POST /upload"]
    assert rejection_reason("markdown", "success", None, short, [], FILTERS) is None
    assert rejection_reason("markdown", "success", None, ["# T", "tbd"], [], FILTERS) == "stub_no_technical_content"
    assert rejection_reason("navigation", "success", None, short, [], FILTERS) == "navigation"


def test_chunks_never_split_fences_and_repeat_table_headers():
    table = ["| A | B |", "|---|---|"] + [f"| value{i} | description number {i} |" for i in range(30)]
    code = ["```", "line one", "", "line two", "```"]
    chunks = chunk_markdown(["# T", "", "intro text here"] + [""] + code + [""] + table, CHUNKING)
    assert all(chunk["text"].count("```") % 2 == 0 for chunk in chunks)
    table_chunks = [chunk for chunk in chunks if "value" in chunk["text"]]
    assert len(table_chunks) > 1 and all("| A | B |" in chunk["text"] for chunk in table_chunks)


def test_long_text_splits_with_overlap():
    sentences = " ".join(f"This is sentence number {i} of the paragraph." for i in range(20))
    chunks = chunk_markdown(["# T", "", sentences], CHUNKING)
    assert len(chunks) > 1
    assert chunks[1]["text"].split()[0] in chunks[0]["text"]


def test_evidence_quote_must_exist():
    assert locate_evidence(["x", "the **limit** is 5"], ["limit is 5"]) == (2, 2)
    assert locate_evidence(["x"], ["limit is 5"]) is None


def _fact(version, scope, value):
    return {"project": "p", "subject": "s", "attribute": "a", "version": version, "version_scope": scope, "value": value}


def test_conflicts_inside_a_version_are_dropped_and_across_versions_scoped():
    kept, conflicts = resolve_conflicts([_fact("1", "canonical", "x"), _fact("1", "canonical", "y")])
    assert not kept and conflicts[0]["type"] == "same_version"
    kept, conflicts = resolve_conflicts([_fact("2", "canonical", "x"), _fact("1", "version_scoped", "y")])
    assert len(kept) == 2 and kept[0]["version_scope"] == "version_scoped"


def test_grounding_accepts_supported_and_rejects_invented():
    source = "| GET | /stock | Returns stock | limit 50 per page"
    good = check_grounding("In svc, `GET /stock` returns stock with limit 50.", source, ["/stock"], "svc", 0.5)
    assert good["status"] == "validated"
    invented = check_grounding("In svc, `GET /orders` returns stock with limit 90.", source, [], "svc", 0.5)
    assert invented["status"] == "pending_review"
    assert any("unsupported_token" in reason for reason in invented["reasons"])
    assert check_grounding("Bananas wonderfully ripen overnight.", source, [], "svc", 0.5)["status"] == "pending_review"


class FakeProvider:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def complete(self, prompt):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


CHUNK = {"chunk_id": "c", "project": "svc", "text": "`GET /stock` returns the stock with limit 50."}


def test_llm_candidates_pass_only_when_grounded():
    provider = FakeProvider(
        [
            '{"question": "What does GET /stock do in svc?", "answer": "`GET /stock` returns the stock.", "category": "api_functionality"}',
            '{"question": "Limit in svc?", "answer": "`GET /orders` allows 900 per second.", "category": "limitations"}',
            "not json",
        ]
    )
    results = generate_candidates([CHUNK, CHUNK, CHUNK], provider, {"max_retries": 1}, 0.5)
    assert [r["validation_status"] for r in results] == ["validated", "pending_review", "rejected"]


def test_llm_retries_then_succeeds_and_gives_up():
    waits = []
    provider = FakeProvider([OSError("timeout"), "ok"])
    assert complete_with_retries(provider, "p", 3, waits.append) == "ok" and waits == [1]
    with pytest.raises(OSError):
        complete_with_retries(FakeProvider([OSError("a"), OSError("b")]), "p", 2, waits.append)


def test_llm_is_offline_unless_explicitly_configured():
    assert isinstance(provider_from_env({}), OfflineProvider)
    assert isinstance(provider_from_env({"LLM_MODE": "api"}), OfflineProvider)
    with pytest.raises(RuntimeError):
        OfflineProvider().complete("x")


def test_difficulty_follows_the_stated_rule():
    def fact(kind, keys, quotes=()):
        return {"kind": kind, "key_facts": keys, "extra": {"evidence_quotes": list(quotes)}}

    assert difficulty_of(fact("multi_source", ["a"])) == "hard"
    assert difficulty_of(fact("table_columns", ["a", "b", "c"])) == "medium"
    assert difficulty_of(fact("conceptual", ["a"], ["q1", "q2"])) == "medium"
    assert difficulty_of(fact("endpoint", ["GET", "/x"])) == "easy"


def test_template_siblings_about_other_identifiers_are_not_duplicates():
    assert same_question("What does `GET /a` do in svc?", "what does `get /a` do in svc", 0.8)
    assert not same_question("What does `GET /a` do in svc?", "What does `GET /b` do in svc?", 0.8)


def test_short_keys_do_not_prove_answer_overlap():
    assert not answer_present(["int"], "column is int", 12)
    assert answer_present(["1000 requests per minute"], "limit: 1000 requests per minute", 12)


def test_language_heuristic_flags_english_inside_spanish_services():
    assert is_english("Check if the application is online.")
    assert not is_english("Valida si la IP está bloqueada")


def test_provenance_verifier_checks_every_field(tmp_path):
    raw = tmp_path / "docs_raw" / "p" / "v" / "guide"
    raw.mkdir(parents=True)
    page = raw / "a.md"
    page.write_text("# Title\n\ntext line\n", encoding="utf-8")
    from src.common import sha256_file, stable_id

    good = {"source_file": "docs_raw/p/v/guide/a.md", "source_hash": sha256_file(page), "document_id": stable_id("doc", "p/v/guide/a.md"), "section": "Title", "line_start": 3, "line_end": 3}
    verifier = Verifier(tmp_path, "docs_raw")
    assert verifier.errors(good) == []
    for field, bad in (("source_hash", "x"), ("document_id", "x"), ("section", "Nope"), ("line_end", 99), ("source_file", "docs_raw/nope.md")):
        assert verifier.errors({**good, field: bad})
    assert verifier.errors({"source_file": "x"}) == ["missing provenance field"]
    assert verifier.errors({**good, "sources": [{**good, "source_hash": "x"}]})
