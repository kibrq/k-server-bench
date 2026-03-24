# Docker

This folder contains the benchmark container image definition.

Files:

- [`Dockerfile`](./Dockerfile): installs the environment and copies the released benchmark assets
- [`entrypoint.sh`](./entrypoint.sh): container startup logic, including optional Ray restart for the legacy evaluator

The Docker image is used to provide a predictable runtime for benchmark tools and agent workflows. It is not the source of truth for benchmark logic; it packages the logic defined elsewhere in the repository.
