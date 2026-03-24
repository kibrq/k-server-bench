# Metrics

This folder contains serialized benchmark instances used for evaluation.

Current files include:

- circle instances such as `circle_k3_m6.pickle` and `circle_k4_m8.pickle`
- circle-taxi instances such as `circle_taxi_k4_m6.pickle`
- small auxiliary instances such as `uniform_small.pickle`

These pickles are consumed by the evaluator and by some utility scripts under [`tools/`](../tools/README.md).

The benchmark logic assumes these are precomputed artifacts. If you change them, you are changing the benchmark itself rather than just a candidate.
