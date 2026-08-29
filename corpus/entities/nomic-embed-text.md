---
title: nomic-embed-text
type: entity
tags: [tool, embedding]
---

# nomic-embed-text

A text embedding model served locally through Ollama. It produces 768-dimension
vectors and supports a context length substantially longer than most earlier
open embedding models, which matters when chunks run long.

## Why it is the default here

It runs locally, so no document text leaves the machine, and it is small enough
that embedding a corpus is fast on CPU alone. For a pipeline whose premise is
that nothing goes to a third party, a hosted embedding API is not an option
regardless of its benchmark ranking.

## Comparability

Its output is not comparable with vectors from any other embedding model. Swapping
it means rebuilding the index from source. See [[embedding-model-lock-in]].
