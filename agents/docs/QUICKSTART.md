# Quickstart

## Create a new attempt
1. Copy an existing solution folder or start fresh (for example):
   - `000_baseline_linear`
2. Rename to:
   - `###_your_idea`

## Required classes
Your program must define (any subset of and provide defaults to the other components):
- `PotentialFamily`
- `Potential`
- `SearchEvaluator`

Typically put all in `main.py`.

## Run evaluation
Minimal call:

```MCP
evaluate_directory(
  directory="solutions/010_softmax_gap",
  program_filename="main.py",
)
```
