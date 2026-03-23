# Dedup `index_matrix` (SearchEvaluator)

## Why

Many candidate `index_matrix` are equivalent under symmetries that don’t change the evaluated potential. Deduplicate by **canonicalizing** the matrix and caching results by that canonical key.

## Scope

This procedure is valid **as written** for the **canonical potential** with the **circle metric**.
If you change the **metric** or the **potential skeleton**, you must re-check which symmetries still hold (some may break).
**However, rules (1) and (2) below are always safe** because they only use commutativity / unordered configurations.

## Equivalence

Treat two matrices as the same if they differ only by:

1. **Row order** (sum over rows is commutative)
2. **Column order within a row** (a configuration is an unordered multiset)

Optional (strong dedup, typically valid for canonical potential + circle metric):
3) **Anchor relabeling** (permute indices `1..n`)
4) **Antipode flips** (for each `i`, flip sign of all `±i`)

## Canonical key (basic)

1. For each row `r`: sort entries by `(abs(x), x)`
2. Sort rows lexicographically
3. Key = `tuple(tuple(row) for row in rows)`

## Canonical key (strong, if `n_used ≤ T`)

Let `n_used = max(abs(entry))`. For all permutations `perm` of `1..n_used` and all sign vectors `flip ∈ {±1}^{n_used}`:

* Transform entry `x`:

  * `j = abs(x)`
  * `x' = sign(x) * flip[j] * perm[j]`
* Compute the **basic** key of the transformed matrix
* Canonical key = lexicographically smallest key over all transforms

Use strong dedup only when `n_used` is small (e.g. `T=6`).

## Cache

Before evaluation:

* `key = canonicalize(index_matrix)`
* return cached result if `key` seen
* else evaluate and store

Include other eval parameters in the cache key if they affect scoring: `(key, n, k, flags_hash, …)`.
