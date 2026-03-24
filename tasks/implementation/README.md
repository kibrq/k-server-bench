# Implementations

Implementation bundles define what the candidate is expected to write.

Available bundles:

- [`potential-family-only/`](./potential-family-only/README.long.md)
- [`non-legacy-evaluator/`](./non-legacy-evaluator/README.md)
- [`general-legacy-evaluator/`](./general-legacy-evaluator/README.long.md)

An implementation bundle may provide:

- long-form and short-form instructions
- one or more initial seed files
- environment fragments consumed by method runners

This layer is where the repository decides whether the candidate edits a simple `Potential`, a full non-legacy search program, or the legacy three-component interface.
