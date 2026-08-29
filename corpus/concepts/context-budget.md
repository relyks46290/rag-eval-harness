---
title: Context Budget
type: concept
tags: [inference, retrieval]
---

# Context Budget

The advertised context window of a model is a ceiling, not a setting to adopt.

## Why the ceiling is expensive

Some runtimes allocate the key-value cache up front, sized to the configured
window rather than to actual use. A model advertising a very large window can
therefore consume tens of gigabytes of memory and a long cold-load time before
processing a single token, regardless of how short the prompt turns out to be.

Retrieval over a handful of chunks sends a couple of thousand tokens. Configuring
a window two orders of magnitude larger than that buys nothing and costs the full
allocation.

## Setting a working value

Size the window to the realistic maximum prompt, with headroom to raise the
number of retrieved chunks later. This pipeline uses 16384 tokens against a
retrieval payload of roughly 2000.

## The generation cap is a separate control

A ceiling on generated tokens is not a quality setting, it is a liveness
guarantee. A reasoning model can enter an unbounded reasoning loop on a
retrieved context and never emit an answer. Without a cap the request never
returns and the caller hangs indefinitely.

With a cap the run fails in seconds, reports why, and is recoverable. This
pipeline caps generation at 2048 tokens, well above the few hundred a real
answer consumes.

## Keeping the model resident

Unloading a model between queries means paying the full load cost on the next
one, which dominates the time an interactive session actually spends generating.
Setting a residency window trades idle memory for latency, and for an
interactive tool that is the right trade.
