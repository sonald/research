# mini-gepa2

`mini-gepa2` is a teaching-oriented reimplementation of the main GEPA loop.
It is based on:

- the paper `GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning` ([arXiv:2507.19457v2](https://arxiv.org/abs/2507.19457), revised on February 14, 2026)
- the official codebase [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa), especially `src/gepa/api.py`, `src/gepa/core/engine.py`, `src/gepa/core/state.py`, and `src/gepa/proposer/reflective_mutation/reflective_mutation.py`

This directory is intentionally not a production clone of the upstream project.
Instead, it keeps the algorithmic backbone and removes most of the engineering shell.

## What This Version Keeps

This teaching version keeps the pieces that make GEPA feel like GEPA:

1. Candidates are text artifacts represented as `dict[str, str]`.
2. A parent candidate is chosen from a Pareto-style candidate pool.
3. The parent is evaluated on a small training minibatch with traces.
4. The adapter turns those traces into a reflective dataset.
5. A reflection LM proposes a new text for one component.
6. The new candidate is re-evaluated on the same minibatch.
7. The candidate is accepted only if the minibatch score strictly improves.
8. Accepted candidates are fully evaluated on the validation set.
9. Validation scores update the per-example Pareto frontier.

That is the core `reflect -> mutate -> locally test -> globally admit` loop.

## What This Version Intentionally Removes

These features are useful in the full repository, but they hide the algorithmic spine:

1. checkpoint save/load and schema migration
2. callbacks and event buses
3. experiment tracking
4. evaluation caching
5. merge proposals
6. parallel proposal workers
7. multi-objective frontier variants
8. resume logic, file stoppers, progress bars, and many validation checks

The result is smaller, easier to read, and easier to modify while learning.

## Directory Layout

```text
mini-gepa2/
  README.md
  pyproject.toml
  src/mini_gepa2/
    __init__.py
    adapter.py
    api.py
    demo.py
    engine.py
    proposal.py
    reflection.py
    sampler.py
    selectors.py
    state.py
  tests/
    test_optimizer.py
```

## Core Design

### 1. Adapter Boundary

The optimizer never interprets task-specific trajectories itself.
It only asks the adapter for two things:

1. `evaluate(batch, candidate, capture_traces)`
2. `make_reflective_dataset(candidate, evaluation, components_to_update)`

This preserves the most important boundary from the official implementation:
GEPA owns the search loop, while the task owns execution details.

### 2. Strict Two-Stage Evaluation

The loop uses two different evaluations for two different purposes:

1. minibatch evaluation decides whether a proposal is worth accepting
2. validation evaluation decides where the accepted candidate sits in the global pool

This separation is central to GEPA and is preserved here.

### 3. Per-Example Pareto Frontier

Instead of keeping only one global best candidate, `mini-gepa2` keeps the best candidate
for each validation example. A candidate can survive because it is best on only a subset
of examples. That keeps specialized solutions alive long enough to be refined further.

### 4. Round-Robin Component Updates

The official system can optimize multiple text components. This teaching version keeps that
shape, even though the included toy demo uses only one component. The state remembers which
component should be updated next for each candidate, and the selector cycles through them.

## Quick Start

Run the toy demo:

```bash
cd /Users/siancao/work/ai/research/mini-gepa2
PYTHONPATH=src python -m mini_gepa2.demo
```

Run the tests:

```bash
cd /Users/siancao/work/ai/research/mini-gepa2
PYTHONPATH=src pytest -q
```

## How To Read The Code

Recommended reading order:

1. `adapter.py` to understand the task boundary
2. `state.py` to understand what the optimizer remembers
3. `selectors.py` and `sampler.py` to understand how a step starts
4. `reflection.py` to understand how traces become a text update
5. `engine.py` for the full optimization loop
6. `demo.py` for a concrete runnable example

## Included Demo

The demo is intentionally tiny and deterministic:

1. the task is a toy word classification problem
2. the candidate text is a prompt containing known `word => label` rules
3. the adapter evaluates whether the current prompt predicts the right label
4. the reflection LM is a heuristic function that reads reflective records and appends
   missing rules to the prompt

This is obviously much simpler than a real LLM system, but it makes the GEPA control flow
fully visible and easy to test.
