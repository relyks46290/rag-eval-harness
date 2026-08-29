#!/usr/bin/env python3
"""
Reliability harness for the local RAG pipeline.

Runs a fixed question set through rag.query() and reports how often the model
returns a usable answer at all. This measures RELIABILITY, not accuracy. A query
passes if it comes back without an error and with a non-trivial answer. Nothing
here judges whether a general answer is correct.

The one exception is deliberate. A question carrying an `expect` list is checked
by string match against the answer, so an exact-identifier question is a real
correctness gate that needs no judge and no ground-truth essay. Those questions
are also the canary for a retrieval regression, because a chunking change or an
embedding-model swap degrades them before it degrades anything else.

    python eval.py                                     # default model, default questions
    python eval.py --model llama3.1:8b
    python eval.py --limit 3                           # spot check on a slow model
    python eval.py --questions questions/mine.json
    python eval.py --compare results/a.json results/b.json

Exit codes:
    0  every question passed, or a comparison found no regression
    1  at least one question failed, or a comparison found a regression
    2  bad invocation (missing file, malformed question set)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rag  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS = os.path.join(HERE, "questions", "sample-corpus.json")
DEFAULT_RESULTS_DIR = os.path.join(HERE, "results")

# Anything shorter than this is a non-answer: an empty string, a stray token, or
# a bare refusal. None of those should score as a success.
MIN_ANSWER_CHARS = 20


# --- question set ---

def load_questions(path: str) -> tuple[str, list[dict]]:
    """Load a question set. Returns (set_name, questions)."""
    if not os.path.exists(path):
        raise SystemExit(f"Question set not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        raise SystemExit(f"{path} has no 'questions' list.")

    seen = set()
    for q in questions:
        for field in ("id", "question"):
            if field not in q:
                raise SystemExit(f"{path}: a question is missing '{field}'.")
        if q["id"] in seen:
            raise SystemExit(f"{path}: duplicate question id '{q['id']}'.")
        seen.add(q["id"])
        if q.get("exact_id") and not q.get("expect"):
            raise SystemExit(
                f"{path}: question '{q['id']}' is marked exact_id but has no 'expect' list. "
                "An identifier question with nothing to match against is not checking anything."
            )
    return data.get("name", os.path.basename(path)), questions


# --- scoring ---

def classify(answer: str, error: str | None, expect: list[str] | None) -> tuple[bool, str | None]:
    """Decide pass or fail for one run.

    Reliability first. `expect` adds a string-match correctness gate on top,
    which is the only correctness claim this harness makes.
    """
    if error is not None:
        return False, "error"
    stripped = answer.strip()
    if not stripped:
        return False, "empty"
    if len(stripped) < MIN_ANSWER_CHARS:
        return False, "too-short"
    if expect:
        lowered = stripped.lower()
        missing = [s for s in expect if s.lower() not in lowered]
        if missing:
            return False, f"missing:{','.join(missing)}"
    return True, None


def run_question(item: dict, model: str, chroma_dir: str | None) -> dict:
    """Run one question. Never raises. A failed query is a data point, not a crash."""
    started = time.time()
    answer, done_reason, pages, error = "", None, [], None
    try:
        result = rag.query(item["question"], model=model, quiet=True, chroma_dir=chroma_dir)
        answer = result["answer"]
        done_reason = result["done_reason"]
        pages = result["pages"]
    except Exception as exc:  # noqa: BLE001 - a harness records failures, it does not propagate them
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    elapsed = time.time() - started

    expect = item.get("expect")
    passed, reason = classify(answer, error, expect)
    return {
        "id": item["id"],
        "area": item.get("area", ""),
        "exact_id": bool(item.get("exact_id", False)),
        "expect": expect or [],
        "question": item["question"],
        "passed": passed,
        "fail_reason": reason,
        "latency_s": round(elapsed, 2),
        "answer_chars": len(answer.strip()),
        "done_reason": done_reason,
        "error": error,
        "pages": [p["path"] for p in pages],
        "top_similarity": pages[0]["similarity"] if pages else None,
        "answer": answer.strip(),
    }


# --- reporting ---

def print_result_line(index: int, total: int, record: dict) -> None:
    status = "PASS" if record["passed"] else "FAIL"
    done = record["done_reason"] or "-"
    top = record["pages"][0] if record["pages"] else "(nothing retrieved)"
    print(
        f"[{index:>2}/{total}] {status}  {record['latency_s']:>6.1f}s  "
        f"{record['answer_chars']:>5} chars  done={done:<10} "
        f"{record['id']:<24} <- {top}"
    )
    if record["error"]:
        print(f"          error: {record['error']}")
    elif not record["passed"]:
        print(f"          {record['fail_reason']}: {record['answer'][:120]!r}")


def print_summary(records: list[dict], model: str, total_s: float, set_name: str) -> None:
    n = len(records)
    passed = [r for r in records if r["passed"]]
    failed = [r for r in records if not r["passed"]]

    print("\n" + "=" * 80)
    print(f"question set: {set_name}   model: {model}   embed: {rag.EMBED_MODEL}")
    print(f"top_k={rag.TOP_K}  num_ctx={rag.NUM_CTX}  num_predict={rag.NUM_PREDICT}  "
          f"min_answer_chars={MIN_ANSWER_CHARS}")
    print(f"\npass rate: {len(passed)}/{n} ({len(passed) / n if n else 0:.0%})"
          f"   wall clock: {total_s:.1f}s")

    if passed:
        lat = sorted(r["latency_s"] for r in passed)
        chars = [r["answer_chars"] for r in passed]
        print(f"passing latency:  min {lat[0]:.1f}s  median {statistics.median(lat):.1f}s  "
              f"max {lat[-1]:.1f}s")
        print(f"passing length:   min {min(chars)}  median {int(statistics.median(chars))}  "
              f"max {max(chars)} chars")

    exact = [r for r in records if r["exact_id"]]
    if exact:
        ok = sum(1 for r in exact if r["passed"])
        print(f"\nexact-identifier gate: {ok}/{len(exact)} passed "
              f"(string match, the only correctness claim here)")

    if failed:
        print("\nfailures:")
        for r in failed:
            print(f"  {r['id']:<24} {str(r['fail_reason']):<28} done={r['done_reason']}")
    print("=" * 80)
    print("\nReliability only. A pass means a usable answer came back, not a correct one,")
    print("except on exact-identifier questions where the expected strings were matched.")


def write_results(
    records: list[dict], model: str, total_s: float, out_dir: str, set_name: str
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_model = model.replace(":", "_").replace("/", "_")
    path = os.path.join(out_dir, f"rag-eval-{safe_model}-{stamp}.json")

    passed = [r for r in records if r["passed"]]
    lat = sorted(r["latency_s"] for r in passed)
    exact = [r for r in records if r["exact_id"]]

    payload = {
        "timestamp": stamp,
        "question_set": set_name,
        "chat_model": model,
        "embed_model": rag.EMBED_MODEL,
        "top_k": rag.TOP_K,
        "num_ctx": rag.NUM_CTX,
        "num_predict": rag.NUM_PREDICT,
        "min_answer_chars": MIN_ANSWER_CHARS,
        "total_questions": len(records),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(records), 4) if records else 0.0,
        "exact_id_total": len(exact),
        "exact_id_passed": sum(1 for r in exact if r["passed"]),
        "total_wall_clock_s": round(total_s, 2),
        "latency_passing_s": {
            "min": lat[0] if lat else None,
            "median": statistics.median(lat) if lat else None,
            "max": lat[-1] if lat else None,
        },
        "results": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


# --- comparison ---

def load_run(path: str) -> dict:
    if not os.path.exists(path):
        raise SystemExit(f"Results file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" not in data:
        raise SystemExit(f"{path} is not an eval results file (no 'results' key).")
    return data


def compare(before_path: str, after_path: str) -> int:
    """Diff two runs question by question. Returns a process exit code.

    This is what makes the harness usable in CI. A pass rate alone cannot tell
    you whether a change fixed three questions and broke three others, which
    nets to zero and is not the same as no change.
    """
    before, after = load_run(before_path), load_run(after_path)
    b = {r["id"]: r for r in before["results"]}
    a = {r["id"]: r for r in after["results"]}

    print(f"before: {os.path.basename(before_path)}  "
          f"{before['chat_model']}  {before['passed']}/{before['total_questions']}")
    print(f"after:  {os.path.basename(after_path)}  "
          f"{after['chat_model']}  {after['passed']}/{after['total_questions']}")

    if before.get("embed_model") != after.get("embed_model"):
        print(f"\nWARNING: embedding model changed, "
              f"{before.get('embed_model')} -> {after.get('embed_model')}. "
              f"Retrieval differences dominate any model comparison across this boundary.")

    only_before = sorted(set(b) - set(a))
    only_after = sorted(set(a) - set(b))
    if only_before:
        print(f"\nnot in the after run: {', '.join(only_before)}")
    if only_after:
        print(f"new in the after run: {', '.join(only_after)}")

    shared = sorted(set(a) & set(b))
    regressions = [i for i in shared if b[i]["passed"] and not a[i]["passed"]]
    fixes = [i for i in shared if not b[i]["passed"] and a[i]["passed"]]

    if fixes:
        print(f"\nfixed ({len(fixes)}):")
        for i in fixes:
            print(f"  + {i:<24} was {b[i]['fail_reason']}")

    if regressions:
        print(f"\nREGRESSIONS ({len(regressions)}):")
        for i in regressions:
            print(f"  - {i:<24} now {a[i]['fail_reason']}  done={a[i]['done_reason']}")

    both = [i for i in shared if a[i]["passed"] and b[i]["passed"]]
    if both:
        deltas = [a[i]["latency_s"] - b[i]["latency_s"] for i in both]
        median = statistics.median(deltas)
        print(f"\nlatency on the {len(both)} questions passing in both: "
              f"median {median:+.1f}s")
        worst = max(both, key=lambda i: a[i]["latency_s"] - b[i]["latency_s"])
        worst_delta = a[worst]["latency_s"] - b[worst]["latency_s"]
        if worst_delta > 0:
            print(f"  slowest regression: {worst} {worst_delta:+.1f}s")

    if regressions:
        print(f"\nFAIL: {len(regressions)} regression(s).")
        return 1
    print("\nOK: no regressions.")
    return 0


# --- entry point ---

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure local RAG reliability across a fixed question set."
    )
    parser.add_argument("--model", default=None,
                        help=f"Ollama chat model under test (default: {rag.CHAT_MODEL})")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS,
                        help="Question set JSON (default: questions/sample-corpus.json)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N questions, for a spot check on a slow model")
    parser.add_argument("--out-dir", default=DEFAULT_RESULTS_DIR,
                        help="Directory for the timestamped JSON results file (default: ./results)")
    parser.add_argument("--chroma-dir", default=None,
                        help=f"ChromaDB directory (default: {rag.CHROMA_DIR})")
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="Diff two results files instead of running. Exits 1 on any regression.")
    args = parser.parse_args()

    if args.compare:
        return compare(*args.compare)

    set_name, all_questions = load_questions(args.questions)
    questions = all_questions[: args.limit] if args.limit else all_questions
    model = rag.resolve_chat_model(args.model)

    print(f"RAG reliability eval")
    print(f"  model under test: {model}")
    print(f"  question set:     {set_name} ({len(questions)} of {len(all_questions)} questions)")
    print(f"  embedding model:  {rag.EMBED_MODEL}")
    print(f"  a query fails if it errors, or returns under {MIN_ANSWER_CHARS} chars,")
    print(f"  or omits an expected identifier.\n")

    records = []
    started = time.time()
    for i, item in enumerate(questions, 1):
        record = run_question(item, model, args.chroma_dir)
        records.append(record)
        print_result_line(i, len(questions), record)
    total_s = time.time() - started

    print_summary(records, model, total_s, set_name)
    path = write_results(records, model, total_s, args.out_dir, set_name)
    print(f"\nresults: {path}")
    print(f"compare: python eval.py --compare <previous>.json {os.path.basename(path)}")

    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
