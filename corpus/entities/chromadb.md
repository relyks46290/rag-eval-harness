---
title: ChromaDB
type: entity
tags: [tool, retrieval]
---

# ChromaDB

An embedded vector database. It runs in-process against a local directory, so
there is no service to deploy or operate.

## Why it is used here

The alternative to an embedded store is standing up a server to hold a few
thousand vectors. For a local single-user pipeline that is operational overhead
with no return.

A collection holds documents, embeddings, and metadata together, so a query
returns nearest neighbours along with the source text and its metadata in one
call. There is no separate document lookup to keep in sync.

## Distance function

The distance function is chosen per collection through collection metadata, and
the key is `hnsw:space`. It accepts `l2`, `ip`, and `cosine`.

This pipeline creates its collection with `hnsw:space` set to `cosine`. Cosine
similarity compares direction and ignores magnitude, which matches how the
embeddings are trained. Chroma returns distances, so similarity is computed as
`1 - distance`.

The key cannot be changed after a collection is created. Changing it means
deleting the collection and rebuilding.

## Persistence

`PersistentClient` writes to a directory on disk and reloads it on the next run.
That directory is derived data and should be excluded from version control; the
markdown corpus is the source of truth, and the index can always be rebuilt from
it.
