# Task Model

## Overview

Released sweeps do not hand-write prompts and seeds from scratch. Instead, each sweep assembles a task from shared building blocks under [`tasks/`](../tasks/README.md).

The common parts are:

- a goal
- an implementation bundle
- zero or more hints
- one or more `.env` fragments

The assembled task text is then written into sweep-local files such as `TASK.md` or `PROMPT.md`, depending on the method.

## Goals

Goals describe the high-level mathematical objective. The default goal lives at [`tasks/goals/default/README.md`](../tasks/goals/default/README.md).

In practice, goals say what kind of object the candidate is trying to produce and what benchmark quantity matters.

## Implementations

Implementation bundles define the coding contract for the candidate.

Current released variants include:

- `potential-family-only`
- `non-legacy-evaluator`
- `general-legacy-evaluator`

An implementation bundle typically contains:

- one or more seed `initial*.py` files
- human-readable instructions
- optionally a `.env` fragment for method runners

## Hints

Hints narrow the search space or specify the metric regime.

Common examples:

- canonical-potential structure hints
- symmetry restrictions
- metric-selection hints such as `k4_general_task`

Hints are intentionally compositional. A sweep can combine multiple hint readmes and multiple `.env` files without changing the underlying method runner.

## Generated Sweep Assets

Each released sweep directory contains a `generate_task.py`.

That script usually does two things:

1. regenerate sweep-local text files from shared task sources
2. regenerate `sweep.sh` so the run uses the current shared assets and env fragments

This is why the shared `tasks/` tree is the source of truth while the sweep folders remain generated launch surfaces.
