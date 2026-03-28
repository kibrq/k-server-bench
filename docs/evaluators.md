# Evaluators

## Two Evaluator Families

`k-server-bench` keeps both a released non-legacy evaluator and a released legacy evaluator because they serve different candidate interfaces.

Relevant code lives under:

- [`tools/evaluator/`](../tools/evaluator/README.md)
- [`tools/legacy-evaluator/`](../tools/legacy-evaluator/README.md)

In general, prefer the non-legacy evaluator.

The main reason is that it is less restrictive and operationally more robust. The legacy path depends on Ray-backed evaluation patterns, and in practice both "one shared Ray cluster for parallel evaluations" and "multiple Ray clusters per run" tend to be fragile. The non-legacy path avoids baking that orchestration into the benchmark contract and instead puts the responsibility for parallelization inside the candidate search procedure.

Historical note: the `k=3` experiments reported in the paper were run with the legacy evaluator. That is useful for reproducing the paper setup, but it should not be read as a recommendation for new experiments.

## Non-Legacy Evaluator

The non-legacy evaluator expects a candidate module with a `Potential` class and usually a `main(args)` search procedure.

The typical flow is:

1. run the candidate program as a subprocess
2. read the candidate JSON payload
3. extract `potential_kwargs`
4. instantiate `Potential(context, **potential_kwargs)`
5. score the resulting potential on the requested metrics

This model is simpler when you want the search procedure and the final potential definition in one file.

It is also the recommended model for new work:

- it exposes fewer benchmark-imposed structural constraints
- it leaves parallelization strategy to the agent or candidate implementation
- it avoids the more brittle Ray-cluster management assumptions used by the legacy path

## Legacy Evaluator

The legacy evaluator preserves the older three-component split:

- `PotentialFamily`
- `Potential`
- `SearchEvaluator`

That interface is more structured but also more cumbersome. It remains useful for reproducing older experiment styles and for methods already written against the legacy contract.

Use it mainly for backward compatibility and reproduction, not as the default choice for new experiments.

## Candidate Responsibilities

Regardless of evaluator family, candidate code is expected to:

- stay within the provided time and CPU budget
- return JSON-serializable outputs
- preserve a valid best-so-far result near interruption boundaries
- avoid modifying the benchmark infrastructure itself

The benchmark is designed so the evaluator owns the final scoring logic. Candidate code is responsible for proposing the object to score, not redefining the scoring rule.

## MCP Wrappers

Both evaluator families also expose MCP server wrappers for agent-driven workflows. Those wrappers live next to the evaluator entrypoints and are documented further under [`agents/docs/`](../agents/docs/README.md).
