## Canonical Potential

The canonical potential has the form:

\[
\min_{a_1, \ldots, a_n}
\sum_{r} wf[a_{\text{index\_matrix}[r,1]}, \ldots, a_{\text{index\_matrix}[r,k]}]
- \sum_{i<j} \text{coef}_{i,j} \, d(i,j)
\]

If `index_matrix[r, c] < 0`, the implementation uses the antipode of `a_{abs(index_matrix[r, c])}`.

The canonical `Potential` requires these keys:

- `n`: number of abstract points `a_1, ..., a_n`
- `index_matrix`: tuple/list of rows, where every row has length exactly `k`
- `coefs`: tuple/list of pairwise coefficients of length `n * (n - 1) // 2`, or `None` (all zeros)

If these keys are missing or malformed, construction will fail.

The implementation does not enforce stronger mathematical canonicalization for you. In particular, you should still treat these as your responsibility:

- sorting entries inside each row when you want equivalent rows to collapse
- sorting rows lexicographically when you want equivalent matrices to collapse
- avoiding duplicate exploration of the same `index_matrix`
- keeping `n` small enough (avoid using n > 8) for the search to remain computationally feasible

## Row Count Guidance

For the k-server inequality, using fewer than `k + 1` rows is mathematically invalid if your goal is to achieve `violations_k = 0`. Such a potential should not be expected to score zero violations.

Using exactly `k + 1` rows is the necessary regime to target `violations_k = 0`.

Using more than `k + 1` rows is not expected to achieve `violations_k = 0` either, but it can still reduce `violations_k`. In practice, allowing extra rows can help you identify which rows are useless or redundant before shrinking back to a tighter structure.

So the practical guidance is:

- if you want a plausible zero-violation candidate, keep the number of rows exactly `k + 1`
- if you want to explore or diagnose structure, it can still make sense to test more than `k + 1` rows

## Semantics Of `index_matrix`

Each row of `index_matrix` describes one term of the work-function sum.

- positive entry `r > 0` means use point `a_r`
- negative entry `r < 0` means use the antipode of `a_{abs(r)}`
- entries are 1-based with respect to `a_1, ..., a_n`

This means every absolute index appearing in `index_matrix` must be between `1` and `n`. The implementation expects that; it does not add its own bounds-checking layer around malformed indices.

Good ways to use it:

- search over `n`
- search over `coefs`
- search over `index_matrix`
- canonicalize candidate structures before submitting them

Avoid assumptions that are not guaranteed by the implementation, such as:

- automatic deduplication of equivalent `index_matrix` values
- automatic clipping or validation of out-of-range row indices
- cheap scaling to very large `n`

Usually `coefs` that work best are small integers from -5 to 5.
