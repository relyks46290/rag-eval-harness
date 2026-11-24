# rag-eval-harness

A reliability-first eval harness for RAG pipelines. No LLM in the scoring path.
Same inputs, same output, every time.

Ships with a local RAG pipeline (Ollama + ChromaDB over a markdown corpus) as
the default system under test. The eval itself is backend-agnostic: `eval.py`
scores answers as strings against deterministic criteria. Swap the pipeline,
keep the harness.

```
question set ──> pipeline under test ──> answers
                                           │
                               eval.py ──> results/*.json
                                           │
                               --compare ──> exit 1 on regression
```

## Why this exists

I wanted to swap the chat model behind a local RAG setup and know whether I had
made it better or worse. There was no way to tell. Model comparisons are usually
vibes, and vibes do not survive contact with a model that answers three questions
beautifully and silently returns nothing on the fourth.

So the harness measures **reliability first**, because reliability needs no
ground truth and no judge:

> A query passes if it returns without an error and with a non-trivial answer.

That sounds like a low bar. It is not. It is the bar that a stronger model failed.

## The result that motivated the whole repo

On an AMD Strix Halo box running Ollama's Vulkan backend, a larger
mixture-of-experts model produced better answers than `llama3.1:8b` when it
produced them at all. It scored **2 of 8**. `llama3.1:8b` scored **8 of 8** on
identical prompts, identical retrieval, same machine, same day.

The MoE model was not wrong. It returned empty `content` with
`done_reason: "length"`. It spent its entire generation budget on reasoning
tokens and emitted no answer. A quality-only eval would have ranked it first,
because you only score the answers that arrive, and the ones that arrive are the
good ones. That is survivorship bias, and it is the default failure mode of
eyeballing model output.

**Those two figures are historical.** They were measured on 2026-08-13 against an
8-question set on a corpus that is not in this repo. The question set here has
grown to 12 and the corpus is different, so **do not compare a run of this repo
against 2/8 and 8/8.** They are stated because they are the reason the harness
exists, not as a current benchmark. Re-measuring them on this corpus is exactly
what `eval.py` is for.

The trap underneath it is worth naming, because it cost me an afternoon:
Ollama's `"think": false` does not always disable reasoning. On some models it
only stops the runtime from separating `thinking` from `content`. The model still
reasons, still burns the budget, and now the reasoning is invisible instead of
merely unwanted. `NUM_PREDICT` is therefore set as a liveness guarantee, not as a
quality knob.

## The scoring function

`classify()` in `eval.py` is a pure function of three inputs: the answer text,
the error (or `None`), and the expect list. No model call, no network, no
randomness, no clock. It applies three checks in order:

1. Did the query error? Fail.
2. Is the answer shorter than 20 characters? Fail (`too-short` or `empty`).
3. Does the question carry an `expect` list? Every expected string must appear
   in the answer, case-insensitive. Missing any one fails the question with
   `missing:<tokens>`.

That is the entire scoring path. It is deterministic by construction, not by
configuration.

## Reproducible generation

Generation is seeded. `rag.py` sends `temperature: 0.0` and `seed: 0` to
Ollama. Without these, a single eval run is unreproducible and `--compare`
cannot separate a real regression from sampling variance.

Seeding does not guarantee identical text. Two seeded `llama3.1:8b` runs of the
12 sample questions, on the same machine with the same model build, returned
identical text for 7 of 12 answers. Pass or fail matched on all 12. Two possible
causes are the order of floating-point operations on the GPU backend and a cold
model load on the first run. No test has separated them yet. A different Ollama
build, quantization, or batch size can change output further.

`eval.py` records `temperature` and `seed` in every results file. `--compare`
warns when either value changed between two runs, including when the earlier run
predates seeding and carries no key at all.

## Quickstart

Requires [Ollama](https://ollama.com) running locally and Python 3.10+.

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b

pip install -r requirements.txt

python rag.py index                       # 9 pages -> 30 chunks, a few seconds
python rag.py query "What is hnsw:space set to?"
python rag.py query                       # interactive
```

Then measure it:

```bash
python eval.py                            # 12 questions, writes results/<timestamp>.json
python eval.py --model qwen3:30b-a3b      # same questions, different model
python eval.py --compare results/A.json results/B.json
```

`--compare` exits `1` if any question that passed in A fails in B. That is the
whole point: it is a regression gate you can put in front of a model swap, not a
score you read and forget.

## The shipped corpus

Nine markdown pages under [`corpus/`](corpus/) on RAG, chunking, and evaluation.
It is real prose, not lorem ipsum, and it is deliberately small so the repo is
runnable in a minute rather than being a dataset download.

It is also the answer key. Because I wrote the corpus, I know exactly what the
correct answer to a factual question is, which makes the next section possible.

## Two questions have an actual right answer

Reliability is judge-free but it is not correctness. Two questions in
[`questions/sample-corpus.json`](questions/sample-corpus.json) are marked
`exact_id` and carry an `expect` list:

```json
{
  "id": "chroma-distance-key",
  "question": "What metadata key sets the distance function on a Chroma collection, and what is it set to here?",
  "area": "entities/chromadb",
  "exact_id": true,
  "expect": ["hnsw:space", "cosine"]
}
```

If `hnsw:space` and `cosine` are not both in the answer, the question fails,
regardless of how confident the prose sounds. No LLM judge, no embedding
similarity, no partial credit. A string is either there or it is not.

This works only because the identifiers are unambiguous and the corpus is
authored here, so `test_expected_identifiers_actually_appear_in_the_corpus`
enforces that every `expect` token is actually stated somewhere in the corpus. An
expectation the corpus never supports would fail every model for the wrong
reason, and would look like a model problem.

Exact identifiers are also where dense retrieval is weakest. `hnsw:space` and
`hnsw_space` and "the distance setting" all embed to nearly the same vector, so
similarity search cannot tell you which one the docs actually use. Testing
exactly there is intentional.

## What the harness records

Each run writes one JSON file with per-question rows:

| Field | Why it is there |
|---|---|
| `passed`, `fail_reason` | `empty`, `too-short`, `error`, or `missing:<tokens>` |
| `done_reason` | `length` here is the runaway-generation signature |
| `answer_chars` | Distinguishes "said nothing" from "said the wrong thing" |
| `latency_s` | Compared only across questions that passed in both runs |
| `top_similarity`, `pages` | Separates a retrieval failure from a generation failure |
| `embed_model` | `--compare` warns loudly if this changed between runs |
| `temperature`, `seed` | `--compare` warns if either changed between runs |

That last row matters more than it looks. Changing the embedding model
invalidates every stored vector. The index does not error, it just quietly
retrieves against a mismatched space, and the comparison you are about to make is
meaningless. There is no migration path short of a full re-index, so the harness
warns rather than pretending the two runs are comparable.

`--compare` reports fixes and regressions separately and on purpose. One question
fixed and one broken nets to zero on the pass rate, and that is not "no change".
`test_an_equal_pass_rate_still_reports_a_regression` pins that behaviour.

## The pipeline (swappable)

The shipped pipeline in `rag.py` is one implementation the harness tests:

- Ollama for embeddings (`nomic-embed-text`) and generation (`llama3.1:8b`)
- ChromaDB for vector storage
- A fence-aware markdown chunker that keeps code blocks indivisible
- Everything on localhost, no API keys

To test a different backend, replace `rag.py` or write an adapter that produces
the same per-question record format. `eval.py` does not import `rag.py` at
scoring time. It reads JSON.

## The chunker

Fenced code blocks are indivisible. That is the one hard rule, and it is why
this is not a paragraph splitter:

1. Page fits the budget, ship it whole.
2. Otherwise split on `##` headers.
3. Otherwise pack paragraph-sized units, keeping fences and tables intact.

A unit that is still oversized gets split line-by-line, and every piece reopens
with the original fence line, so a continuation of a `json` block stays a `json`
block. When the choice is between a chunk that runs over budget and a chunk that
ends in the middle of a fence, it overshoots. A code block cut in half retrieves
as two fragments that are each syntactically broken, and a model handed a
truncated fence tends to complete it rather than answer the question.

Every chunk carries its source header (`# Page Title`) so a retrieved fragment
still says where it came from.

## Tests

```bash
python tests/test_pipeline.py       # 21 tests, no model server required
```

CI runs these on 3.10 and 3.12. They cover chunking (including the fence
invariant against the real corpus), frontmatter parsing, `classify`,
question-set validation, `compare` against synthetic fixtures, and the seeding
contract.

**They do not test retrieval quality**, and no unit test can. Whether the right
chunk comes back for a given question is what `eval.py` measures, against a real
model, and the answer changes when you change the model. The tests cover the
deterministic parts, which is also where the bugs turned out to be. The
regression test named `test_an_oversized_first_unit_does_not_become_one_unbounded_chunk`
is there because an earlier version flushed the buffer only when it was non-empty,
so an oversized unit arriving first skipped the flush and defeated the size budget
on exactly the pages that most needed splitting.

## Configuration

| Variable | Default | |
|---|---|---|
| `RAG_CORPUS` | `./corpus` | also `--corpus` |
| `RAG_CHROMA_DIR` | `./.chroma` | also `--chroma-dir` |
| `RAG_CHAT_MODEL` | `llama3.1:8b` | also `--model` |
| `RAG_EMBED_MODEL` | `nomic-embed-text` | changing this requires a re-index |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | |

Point `RAG_CORPUS` at any directory of markdown and write a question set for it.
The harness does not know or care that the shipped corpus is about RAG.

## What this is not

It is not a benchmark, it does not measure answer quality, and twelve questions
is a smoke test rather than a statistically meaningful sample. It tells you
whether a model swap broke something that used to work. That was the question I
had, and it is a question most local RAG setups cannot answer at all.

## License

MIT
