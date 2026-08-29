---
title: Chunking Strategy
type: concept
tags: [retrieval, indexing]
---

# Chunking Strategy

Chunking decides what unit of text gets embedded and therefore what unit can be
retrieved. It matters more than embedding-model choice on most corpora, because
a retrieval system can only return things it stored as retrievable units.

## Why fixed-width chunking fails on markdown

The simplest approach slices every N characters. On prose it is merely
suboptimal. On markdown it is destructive, because it cuts through structures
whose meaning depends on being whole.

A code block split across two chunks produces two useless halves. Neither one
runs, neither one reads correctly, and both now sit in the index competing for
retrieval slots against chunks that would actually answer the question. The same
applies to a table split mid-row.

## The strategy used here

Three tiers, applied in order.

1. **Whole page.** A page under the character budget is embedded as one chunk.
   Splitting it would only lose context.
2. **Split on headers.** Longer pages split on markdown headers, so each section
   becomes a chunk. Section boundaries are semantic boundaries the author
   already chose, which makes them free and better than any guess.
3. **Pack paragraph units.** A section still over budget is packed from
   paragraph units, where a fenced code block counts as a single indivisible
   unit no matter how many blank lines it contains.

The default character budget is 1500.

## Deliberate overshoot

A chunk is allowed to exceed the budget rather than cut a fence or a table row.
Only a unit that is itself over budget gets split, and when it does, each piece
is left individually fence-balanced: the piece is closed with a fence and the
next piece reopens with the original fence line.

A single line longer than the budget is never broken. This means a chunk can
still overshoot, which is the intended trade. An oversized chunk costs context
window. A table row sliced down the middle costs meaning.

## The bug this design fixes

An earlier version flushed the buffer only when the buffer was non-empty. An
oversized unit arriving first therefore skipped the flush and became a single
unbounded chunk, silently defeating the budget for exactly the pages most in
need of splitting. The fix is to handle the oversized unit before the packing
check, not inside it.
