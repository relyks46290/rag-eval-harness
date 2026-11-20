"""
Tests for the parts of the pipeline that do not need a running model.

Chunking, scoring, question-set validation, and run comparison are pure
functions over text and dicts, and they are where the bugs actually live.
Retrieval quality needs a corpus, an embedding model, and judgement; this file
deliberately does not pretend to test that.

    uv run --with chromadb --with requests --with pyyaml python tests/test_pipeline.py
    uv run --with chromadb --with requests --with pyyaml --with pytest pytest
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import eval as harness  # noqa: E402
import rag  # noqa: E402


# --- fence handling, the property the whole chunker exists to protect ---

def _fence_count_even(text: str) -> bool:
    return text.count("```") % 2 == 0


def test_split_units_keeps_a_fenced_block_whole():
    """A blank line inside a fence must not end the unit. This is the single
    behaviour that separates this chunker from a naive paragraph splitter."""
    text = "Intro paragraph.\n\n```python\ndef f():\n\n    return 1\n```\n\nOutro paragraph."
    units = rag.split_units(text)
    fenced = [u for u in units if "```" in u]
    assert len(fenced) == 1, f"the code block should be one unit, got {len(units)} units"
    assert "def f():" in fenced[0] and "return 1" in fenced[0]
    assert _fence_count_even(fenced[0])


def test_split_oversized_unit_leaves_every_piece_balanced():
    body = "\n".join(f"line {i} of a long code block" for i in range(200))
    unit = f"```python\n{body}\n```"
    pieces = rag.split_oversized_unit(unit, max_chars=400)
    assert len(pieces) > 1, "a block far over budget should split"
    for i, p in enumerate(pieces):
        assert _fence_count_even(p), f"piece {i} ends inside a fence:\n{p[-120:]}"


def test_split_oversized_unit_reopens_with_the_original_fence_line():
    body = "\n".join(f"row {i}" for i in range(100))
    pieces = rag.split_oversized_unit(f"```json\n{body}\n```", max_chars=200)
    assert pieces[1].startswith("```json"), \
        "a continuation piece must reopen with the original fence, not a bare one"


def test_balance_fences_closes_a_dangling_fence():
    assert _fence_count_even(rag.balance_fences("text\n```python\ncode"))
    unchanged = "text\n```\ncode\n```"
    assert rag.balance_fences(unchanged) == unchanged


def test_no_chunk_of_the_shipped_corpus_ends_inside_a_fence():
    """The end-to-end property, run against the real corpus in this repo."""
    pages = rag.load_pages(os.path.join(REPO, "corpus"))
    assert pages, "the shipped corpus should not be empty"
    for page in pages:
        for chunk in rag.chunk_page(page):
            assert _fence_count_even(chunk["text"]), \
                f"unbalanced fence in {chunk['id']}"


# --- chunking budget ---

def test_a_short_page_stays_one_chunk():
    page = {"id": "p.md", "title": "P", "type": "concept", "tags": [],
            "body": "Short body.", "path": "p.md"}
    assert len(rag.chunk_page(page)) == 1


def test_an_oversized_first_unit_does_not_become_one_unbounded_chunk():
    """Regression test. An earlier version flushed only on a non-empty buffer, so
    an oversized unit arriving first skipped the flush and defeated the budget
    for exactly the pages that most needed splitting."""
    giant = " ".join(f"word{i}" for i in range(2000))
    page = {"id": "p.md", "title": "P", "type": "concept", "tags": [],
            "body": f"## Section\n\n{giant}\n\nshort tail.", "path": "p.md"}
    chunks = rag.chunk_page(page, max_chars=500)
    assert len(chunks) > 2, "an oversized unit should be split, not passed through whole"


def test_every_chunk_carries_its_source_metadata():
    pages = rag.load_pages(os.path.join(REPO, "corpus"))
    for page in pages:
        for chunk in rag.chunk_page(page):
            md = chunk["metadata"]
            assert md["path"] == page["path"]
            assert md["title"]
            assert chunk["text"].startswith(f"# {page['title']}"), \
                "a retrieved fragment must carry its own source header"


def test_chunk_ids_are_unique_across_the_corpus():
    pages = rag.load_pages(os.path.join(REPO, "corpus"))
    ids = [c["id"] for p in pages for c in rag.chunk_page(p)]
    assert len(ids) == len(set(ids)), "duplicate chunk ids would silently overwrite in the store"


def test_frontmatter_parses_and_bad_yaml_degrades_quietly():
    meta, body = rag.parse_frontmatter("---\ntitle: A\ntype: concept\n---\n\nBody here.")
    assert meta["title"] == "A" and body == "Body here."

    meta, body = rag.parse_frontmatter("---\n: : :\nbad\n---\n\nBody.")
    assert meta == {} and body == "Body."

    meta, body = rag.parse_frontmatter("No frontmatter at all.")
    assert meta == {} and body == "No frontmatter at all."


# --- scoring ---

def test_classify_covers_the_failure_modes():
    assert harness.classify("a" * 50, None, None) == (True, None)
    assert harness.classify("", None, None) == (False, "empty")
    assert harness.classify("   ", None, None) == (False, "empty")
    assert harness.classify("ok", None, None) == (False, "too-short")
    assert harness.classify("a" * 50, "ConnectionError: refused", None)[1] == "error"


def test_expect_gates_on_the_literal_identifier():
    long_answer = "The collection metadata key is hnsw:space and it is set to cosine here."
    passed, reason = harness.classify(long_answer, None, ["hnsw:space", "cosine"])
    assert passed and reason is None

    wrong = "The collection uses euclidean distance, configured through the space setting."
    passed, reason = harness.classify(wrong, None, ["hnsw:space", "cosine"])
    assert not passed and "missing" in reason and "hnsw:space" in reason


def test_expect_matching_is_case_insensitive():
    passed, _ = harness.classify(
        "It is configured with HNSW:SPACE set to COSINE for this collection.",
        None, ["hnsw:space", "cosine"],
    )
    assert passed


def test_an_error_outranks_a_matching_identifier():
    """An answer that arrived with an error is not a pass, whatever it contains."""
    passed, reason = harness.classify("hnsw:space cosine", "Timeout: no response", ["cosine"])
    assert not passed and reason == "error"


# --- question set validation ---

def test_the_shipped_question_set_loads_and_is_well_formed():
    name, questions = harness.load_questions(harness.DEFAULT_QUESTIONS)
    assert name and len(questions) >= 10
    exact = [q for q in questions if q.get("exact_id")]
    assert len(exact) >= 2
    for q in exact:
        assert q.get("expect"), "an exact_id question must carry an expect list"


def test_shipped_questions_reference_areas_that_exist():
    _name, questions = harness.load_questions(harness.DEFAULT_QUESTIONS)
    corpus = os.path.join(REPO, "corpus")
    for q in questions:
        area = q.get("area")
        if area:
            assert os.path.exists(os.path.join(corpus, area + ".md")), \
                f"question '{q['id']}' names a corpus page that does not exist: {area}.md"


def test_expected_identifiers_actually_appear_in_the_corpus():
    """If the corpus does not state the expected string, the gate tests nothing
    and every model fails it for the wrong reason."""
    _name, questions = harness.load_questions(harness.DEFAULT_QUESTIONS)
    pages = rag.load_pages(os.path.join(REPO, "corpus"))
    blob = "\n".join(p["body"] for p in pages).lower()
    for q in questions:
        for token in q.get("expect", []):
            assert token.lower() in blob, \
                f"question '{q['id']}' expects '{token}', which the corpus never states"


def _write_set(path: str, questions: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": "t", "questions": questions}, f)


def test_a_malformed_question_set_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        dupes = os.path.join(d, "dupes.json")
        _write_set(dupes, [{"id": "a", "question": "?"}, {"id": "a", "question": "?"}])
        try:
            harness.load_questions(dupes)
        except SystemExit as exc:
            assert "duplicate" in str(exc)
        else:
            raise AssertionError("duplicate ids should be rejected")

        naked = os.path.join(d, "naked.json")
        _write_set(naked, [{"id": "a", "question": "?", "exact_id": True}])
        try:
            harness.load_questions(naked)
        except SystemExit as exc:
            assert "expect" in str(exc)
        else:
            raise AssertionError("an exact_id question with no expect list should be rejected")


# --- comparison ---

def _run_file(
    path: str,
    rows: list[tuple[str, bool, float]],
    model: str = "m",
    extra: dict | None = None,
) -> None:
    results = [{
        "id": rid, "area": "", "exact_id": False, "expect": [],
        "question": "?", "passed": ok, "fail_reason": None if ok else "empty",
        "latency_s": lat, "answer_chars": 100 if ok else 0,
        "done_reason": "stop" if ok else "length", "error": None,
        "pages": [], "top_similarity": None, "answer": "x" * 100 if ok else "",
    } for rid, ok, lat in rows]
    payload = {
        "timestamp": "t", "question_set": "t", "chat_model": model,
        "embed_model": "e", "total_questions": len(rows),
        "passed": sum(1 for _, ok, _ in rows if ok), "results": results,
    }
    payload.update(extra or {})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _quiet_compare(before: str, after: str) -> int:
    """compare() prints a human-readable report; the tests only want its exit code."""
    with contextlib.redirect_stdout(io.StringIO()):
        return harness.compare(before, after)


def _compare_output(before: str, after: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        harness.compare(before, after)
    return buf.getvalue()


def test_compare_flags_a_regression_and_exits_nonzero():
    with tempfile.TemporaryDirectory() as d:
        before, after = os.path.join(d, "b.json"), os.path.join(d, "a.json")
        _run_file(before, [("q1", True, 1.0), ("q2", True, 1.0)])
        _run_file(after, [("q1", True, 1.0), ("q2", False, 1.0)])
        assert _quiet_compare(before, after) == 1


def test_compare_passes_when_nothing_regressed():
    with tempfile.TemporaryDirectory() as d:
        before, after = os.path.join(d, "b.json"), os.path.join(d, "a.json")
        _run_file(before, [("q1", False, 1.0), ("q2", True, 1.0)])
        _run_file(after, [("q1", True, 1.0), ("q2", True, 1.0)])
        assert _quiet_compare(before, after) == 0


def test_an_equal_pass_rate_still_reports_a_regression():
    """The reason a pass rate alone is not enough. One fix and one break nets to
    zero on the headline number and is not the same as no change."""
    with tempfile.TemporaryDirectory() as d:
        before, after = os.path.join(d, "b.json"), os.path.join(d, "a.json")
        _run_file(before, [("q1", True, 1.0), ("q2", False, 1.0)])
        _run_file(after, [("q1", False, 1.0), ("q2", True, 1.0)])
        assert _quiet_compare(before, after) == 1, \
            "1/2 before and 1/2 after is not 'no change'"


def test_compare_warns_when_a_run_was_not_seeded():
    """A seeded run and an unseeded one are not comparable. Sampling variance
    alone can move a question across the pass boundary, so the diff cannot
    attribute the change to anything. Runs predating seeding carry no key."""
    with tempfile.TemporaryDirectory() as d:
        before, after = os.path.join(d, "b.json"), os.path.join(d, "a.json")
        _run_file(before, [("q1", True, 1.0)])
        _run_file(after, [("q1", True, 1.0)], extra={"temperature": 0.0, "seed": 0})
        out = _compare_output(before, after)
        assert "temperature changed, unset -> 0.0" in out, out
        assert "seed changed, unset -> 0" in out, out


def test_compare_is_silent_when_both_runs_carry_the_same_seed():
    seeded = {"temperature": 0.0, "seed": 0}
    with tempfile.TemporaryDirectory() as d:
        before, after = os.path.join(d, "b.json"), os.path.join(d, "a.json")
        _run_file(before, [("q1", True, 1.0)], extra=seeded)
        _run_file(after, [("q1", True, 1.0)], extra=seeded)
        out = _compare_output(before, after)
        assert "temperature changed" not in out, out
        assert "seed changed" not in out, out


def test_the_shipped_pipeline_requests_greedy_decoding():
    """The determinism claim depends on these two values, so assert them rather
    than trusting that nobody reverted the options dict."""
    assert rag.TEMPERATURE == 0.0
    assert rag.SEED is not None


def main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
