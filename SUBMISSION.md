# Submission Guide

This anonymous release should be evaluated with the released non-legacy evaluator.

## Primary Evaluation Recommendation

Use:

- metric: `circle_k4_m6.pickle`
- evaluator flag: `--robustness_check`

This is the primary metric/configuration reviewers should use first.

## Setup

1. Install Git LFS and fetch the metric files.
2. Install the local `k-servers` package.

Example:

```bash
git lfs install
git lfs pull
pip install -e ./k-servers
```

## Recommended Evaluation Command

Run the released evaluator on your candidate program with robustness checking enabled.

Example:

```bash
python tools/evaluator/evaluate.py \
  --program_path <path-to-candidate-program> \
  --metrics_names circle_k4_m6.pickle \
  --robustness_check
```

If your candidate program expects the usual benchmark entrypoint, it should export `Potential` and may also implement `main(args)` to emit `potential_kwargs`.

## Example: Unifying Potential

One concrete example in this repository is the unifying potential parameterization stored in:

- potential implementation: `k-servers/src/kserver/potential/canonical_potential.py`
- kwargs: `examples/unifying_potential/kwargs.json`

Example command:

```bash
python tools/evaluator/evaluate.py \
  --program_path k-servers/src/kserver/potential/canonical_potential.py \
  --metrics_names circle_k4_m6.pickle \
  --potential_kwargs_json examples/unifying_potential/kwargs.json \
  --robustness_check
```

## Outputs

The evaluator writes:

- `correct.json`
- `metrics.json`

Reviewers should inspect `metrics.json` for the reported violation counts and robustness result.

## Notes

- Prefer the non-legacy evaluator for submission review.
- Do not modify files under `metrics/`; they are part of the benchmark dataset.
- If Git LFS files are missing, evaluation results are not meaningful.
