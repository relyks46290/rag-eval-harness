---
title: Reliability Versus Accuracy
type: concept
tags: [evaluation]
---

# Reliability Versus Accuracy

These are different measurements and conflating them produces evaluation results
that mean nothing.

**Reliability** asks whether the system returned a usable answer at all. Did the
request complete, did the stream terminate properly, did content come back.

**Accuracy** asks whether the answer was correct. This requires ground truth and
either a human judge or a model acting as one.

## Why reliability deserves its own harness

Reliability is cheap to measure, needs no ground truth, and is the axis on which
local models differ most dramatically. A model can be excellent on answer
quality and still be unusable because it returns empty content on a third of
requests under a particular inference backend.

That failure is invisible to an accuracy benchmark run on the responses that did
come back. Filtering out the empty responses before scoring is a survivorship
bias that makes the unreliable model look fine.

Measuring reliability first also fails faster. If a candidate model returns
nothing on a quarter of queries, its accuracy score is not the interesting
number about it.

## What a reliability pass means here

A query passes if it does not raise an error and returns an answer above a
minimum character threshold. The threshold exists because an empty string, a
stray token, and a bare refusal are all non-answers, and none should be scored
as a success.

Nothing in a reliability pass claims the answer was correct. Stating that limit
plainly in the harness output is part of the design. A metric that is quietly
assumed to mean more than it does is worse than no metric.

## The exception that is closest to accuracy

A question whose answer hinges on a literal identifier is a partial exception. A
wrong identifier is checkable without a judge, by string match. Those questions
function as a correctness gate on retrieval specifically, which is why this
harness marks them separately.
