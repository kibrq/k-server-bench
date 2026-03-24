# Core Concepts

## Work Functions

The benchmark studies potentials over work functions for the k-server problem.

A work function assigns a value to each configuration of `k` servers. Intuitively, that value is the minimum cost needed to serve the request sequence seen so far and end in that configuration.

The evaluators operate on normalized work functions induced by precomputed metric instances. Candidate potentials are judged on how they change across transitions in those instances.

## Potentials

A potential is a function `Phi(w)` from a work function `w` to a scalar.

In this repository, potentials usually appear in one of two forms:

- a direct `Potential` class used by the non-legacy evaluator
- a combination of `PotentialFamily`, `Potential`, and `SearchEvaluator` used by the legacy evaluator

The benchmark includes a strong bias toward canonical potentials, where the candidate searches over a structured family described by:

- `n`
- `index_matrix`
- `coefs`

Those objects encode a minimization over metric space points and antipodes. The relevant hint text lives under [`tasks/hints/canonical-potential/`](../tasks/hints/canonical-potential/README.md).

## Metric Instances

Metric instances are precomputed state graphs stored as pickles under [`metrics/`](../metrics/README.md).

Each instance packages:

- a `WFContext`
- the explored nodes and edges
- Bellman-style information used during scoring

The released benchmark mostly uses small but mathematically meaningful circle and circle-taxi instances. Smaller instances are often genuine subproblems of harder ones, which is why they are useful for proxy search rather than just random approximations.

## Scoring

The main benchmark objective is minimizing `violations_k`.

At a high level, the potential is checked across many edges of the instance graph. A violation means the potential change does not certify the desired inequality on that transition.

Important score fields include:

- `violations_k`
- `violations_k_score`
- `combined_score`
- `processed_normalized_edges_score`

The ideal result is zero violations on every evaluated metric, which corresponds to a perfect multiplicative score of `1.0`.

## Search vs Final Evaluation

Many candidate programs do not expose a fully fixed potential directly. Instead, they search for good hyperparameters within a time and CPU budget.

That leads to a two-stage picture:

1. search logic proposes `potential_kwargs`
2. the evaluator instantiates the final potential and scores it on the full metrics

This separation is central to the benchmark design. The search procedure is allowed to use proxy evaluations, but the final scoring is always done by the benchmark evaluator.
