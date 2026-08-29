---
title: Embedding Model Lock-In
type: concept
tags: [retrieval, indexing]
---

# Embedding Model Lock-In

Vectors produced by two different embedding models are not comparable. Their
dimensions mean different things, and in most cases there are a different number
of them.

## The practical consequence

Changing the embedding model invalidates the entire existing index. There is no
migration, no conversion, no partial re-embed. Every chunk must be embedded
again from source.

This is not a bug to be worked around. It follows directly from what an
embedding is, which is a position in a space that particular model defined.

## Why this shapes the design

Two rules follow.

**Keep the source corpus, not just the index.** A vector store without the text
it came from cannot be rebuilt. The index is derived data and should be treated
as disposable.

**Make re-indexing cheap enough to be routine.** If rebuilding takes hours,
nobody will test a candidate embedding model, and the first choice becomes
permanent through inertia rather than measurement.

## Dimensionality

Dimension count is a storage and speed tradeoff, not a quality dial. A larger
vector costs more memory and more time per comparison. It does not follow that
it retrieves better on a given corpus, and it can retrieve worse if the model
was trained for a different domain.

Measure on your own corpus. General benchmark rankings for embedding models
transfer poorly to a specific document set.
