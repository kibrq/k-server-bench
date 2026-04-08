# search_n7_async_pipeline

This example contains a Codex-developed, human-guided search procedure for a fixed `n=7` canonical `index_matrix`. The script searches for a coefficient vector for that matrix using a staged evaluator pipeline:

1. evaluate on `circle_k4_m6`
2. evaluate the hard edges of `circle_taxi_k4_m6`
3. run timeout-limited evaluation on random taxi edges
4. run full `circle_taxi_k4_m6` evaluation on the best survivors

The main driver is [`search_n7_async_pipeline.py`](./search_n7_async_pipeline.py).

The fixed `index_matrix` used by this search is:

```text
-A  -A  -A  -A
 A -B1 -B2 -B3
 A  B1 -C1 -C2
 A  B2  C1  -D
 A  B3  C2   D
```

which in the script is encoded as:

```text
-1 -1 -1 -1
 1 -2 -3 -4
 1  2 -5 -6
 1  3  5 -7
 1  4  6  7
```

We selected this matrix because we noticed that in the Huang `k=3` potential, the index matrix can be merged back into the unifying Coester matrix. That suggested using the same idea in the `k=4` case, but here there is extra freedom in how the unifying matrix could have been split.

The corresponding unifying matrix is:

```text
-A -A -A -A
 A -B -B -B
 A  B -C -C
 A  B  C -D
 A  B  C  D
```

Our split rule was:

- split `B` into `B1`, `B2`, `B3`
- split `C` into `C1`, `C2`

For comparison, in the `k=3` case the unifying matrix is:

```text
-A -A -A
 A -B -B
 A  B -C
 A  B  C
```

and the split was `B -> B1, B2` with the sign pattern `B1, -B2`, which is essentially the same two-way split idea.

In practice, this search can find a 3-violation coefficient vector in about 20-30 minutes when it starts from random candidates.

To reproduce the successful setup, run from the repository root with all mutate families enabled, only `seed_dense` enabled among seed families, and startup seed candidates disabled:

```bash
python examples/search_n7_async_pipeline/search_n7_async_pipeline.py \
  --timeout 1800 \
  --n-cpus 20 \
  --families seed_dense,mutate_ck4,mutate_proxy,mutate_timed,mutate_full \
  --disable-startup-seeds \
  --tag reproduce_n7_codex
```

The exact family names and flags above come directly from the script's CLI argument parser.
