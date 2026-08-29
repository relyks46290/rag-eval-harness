---
title: Ollama
type: entity
tags: [tool, inference]
---

# Ollama

A local model runtime exposing an HTTP API on a loopback port, default 11434. It
pulls quantised models, keeps them resident for a configurable period, and
serves both chat and embedding endpoints.

## Endpoints used here

`/api/embed` takes a model and an `input` field, where `input` accepts either a
single string or a list of strings. Passing a list is what makes batch embedding
possible, and batching is the difference between indexing a corpus in seconds
and indexing it in minutes.

`/api/chat` takes a message list and streams newline-delimited JSON objects back.
The final object carries `done` set to true along with a `done_reason` field,
which is the only reliable way to distinguish a completed answer from one cut
off by the token cap.

## Residency

The `keep_alive` field controls how long a model stays loaded after a request.
This pipeline sets it to `30m`. The default idle unload means paying a full
model reload on the next question, which for a large model swamps the actual
generation time.

## Reasoning models and the "think" trap

Some reasoning models stream their chain of thought into a separate `thinking`
field on each message, distinct from `content`.

Setting `"think": false` looks like it disables reasoning. On at least some
models it does not. It only stops the runtime from separating the reasoning out,
so the chain of thought lands in `content` instead, with no stop discipline, and
generation runs until the token cap ends it. The result is a request that
consumes its entire budget and returns no answer.

The correct handling is to leave the field alone, read `thinking` for progress
display only, and treat `content` as the answer.

## Backend variation

Backend selection is per-platform and per-GPU, and it affects reliability rather
than only speed. A model that answers correctly on one backend can return empty
responses on another. This is a property of the model and backend together, so
it has to be measured per combination rather than inferred from the model alone.
