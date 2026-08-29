---
title: Exact Identifier Retrieval
type: concept
tags: [retrieval, evaluation]
---

# Exact Identifier Retrieval

Dense vector retrieval is weak on queries whose answer is a literal token: a
function name, a config key, an error constant, a version string.

## Why it fails

An embedding places text by meaning. Two identifiers that mean nearly the same
thing land near each other, which is exactly wrong when the user needs one
specific string. `keep_alive` and `keepalive` and `idle_timeout` are neighbours
in embedding space and distinct in an API. The retriever cannot tell them apart
because the property that distinguishes them, exact spelling, is the property
the embedding discards.

Lexical retrieval such as BM25 has the opposite profile: strong on exact tokens,
weak on paraphrase. Hybrid retrieval exists because the two failure modes are
complementary.

## Why these questions belong in an eval set

An identifier question is checkable by string match. No judge, no ground-truth
essay, no scoring rubric. Either the exact string appears in the answer or it
does not.

That makes identifier questions the cheapest correctness signal available in a
harness that otherwise measures only reliability. They are marked separately so
a run reports them as their own line, and they serve as the canary for a
retrieval regression: a chunking change or an embedding-model swap degrades
these before it degrades anything else.

## The trap

An identifier question that names the identifier in the question is not testing
retrieval, it is testing copying. Ask what the constant is and where it applies,
not "does the code use `keep_alive`".
