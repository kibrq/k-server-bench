# Package Source

This folder contains the `kserver` Python package source tree.

Main subpackages:

- `kserver/context`: work-function contexts and support utilities
- `kserver/metrics`: benchmark metric families
- `kserver/potential`: concrete potential implementations
- `kserver/evaluation`: scoring helpers used by the evaluators
- `kserver/graph`: graph exploration and hashing utilities

This layer should remain library-quality code. It is benchmark infrastructure, not sweep-local generated material.
