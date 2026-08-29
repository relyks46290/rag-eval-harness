---
title: Retrieval-Augmented Generation
type: concept
tags: [retrieval, llm]
---

# Retrieval-Augmented Generation

Retrieve documents relevant to a query, place them in the prompt, and generate
an answer conditioned on them. Knowledge lives in a corpus rather than in the
weights, so updating what the system knows means editing documents instead of
retraining.

## Against fine-tuning

Fine-tuning adjusts weights and is appropriate for changing behaviour, format,
or style. It is a poor way to add facts. The facts are diffuse, unattributable,
and cannot be corrected without another training run.

Retrieval keeps facts inspectable. A wrong answer can be traced to the chunk
that produced it, and fixing the document fixes the answer.

## The failure modes are retrieval failures

If the chunk containing the answer is not in the context, no prompt recovers it.
The model will either say it does not know, which is the good outcome, or
confabulate from adjacent material, which is the bad one.

This is why retrieval and generation must be evaluated separately. A single
end-to-end score attributes every miss to the model and hides the fact that most
misses are indexing problems.

## Grounding instructions

Instructing the model to answer only from the supplied context, and to say so
when the context is insufficient, is the difference between a system that
reports gaps and one that fills them with plausible text. It does not eliminate
confabulation, but it substantially reduces it and it makes the failure legible.

Requiring citation of document titles makes the failure legible too, because a
citation to a document that does not support the claim is checkable.
