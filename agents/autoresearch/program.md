# autoresearch

This workspace is for autonomous iteration on a single k-server candidate program.

The editable surface is `main.py`.
Use git for experiment history.
Use the released evaluator to measure progress.

Goal: reduce `violations_k` as much as possible while keeping the candidate valid and fast enough to evaluate.

## Ground Rules

- Edit `main.py` by default. Only add extra files when that clearly helps.
- Do not modify the evaluator, metric files, or copied paper sources under `docs/`.
- Treat git history as the experiment record. Do not create many copied attempt directories.
- Read evaluator outputs after every run before deciding what to try next.
- Prefer measured progress over speculation.

## Setup

At the start of a fresh run:

1. Pick a run tag with the user and create a branch like `autoresearch/<tag>`.
2. Read `README.md`, `main.py`, and `docs/EVALUATE.md`.
3. Skim the copied papers or other docs in `docs/` only as needed.
4. Confirm the evaluation plan with the user:
   - metric or metrics
   - timeout budget
   - CPU budget
   - any additional mathematical hints or restrictions
5. Run the baseline candidate before making substantial changes.

If the user did not specify parameters, ask once up front. After that, continue autonomously until stopped.

## Evaluation

Use the released evaluator, either directly or through MCP if available.

Default posture:

- use smaller, cheaper runs early when screening ideas
- use longer runs only for ideas that survive the cheap screen
- keep the evaluation settings comparable when judging whether one idea beats another

After each run, inspect the saved outputs such as:

- `metrics.json`
- `correct.json`
- `log.out`
- `log.err`

Primary objective: lower `violations_k`.

## Git Workflow

Use git as the experiment journal.

- check `git status` and recent `git log` before starting work
- commit meaningful candidate changes before or after runs, whichever keeps the history legible
- use short factual commit messages that capture the idea being tested
- keep `results.md` untracked if you use it as a scratchpad

Good commits preserve promising states.
Bad ideas do not need to be preserved unless they teach something important.

## Loop

Repeat this loop:

1. Inspect the current git state and the latest results.
2. Form one concrete hypothesis.
3. Implement it in `main.py`.
4. Run the evaluator.
5. Read the outputs carefully.
6. Decide:
   - keep the change and commit it
   - refine it and rerun
   - discard it and move on
7. Record the result briefly in `results.md` or commit history.

Work in small steps.
Do not stack many unrelated ideas into one experiment.

## Candidate Guidance

For the released non-legacy evaluator, `main.py` should define:

- `Potential(context, **kwargs)`
- `main(args) -> JSON-serializable dict`

Useful reminders:

- `Potential.__call__` is on the hot path, so optimize for repeated evaluation
- `main(args)` should actually use `args.timeout` and `args.n_cpus`
- prefer meaningful full-instance proxies over random edge subsampling when screening ideas
- debug contract or serialization failures before changing the mathematical idea

## Failure Handling

If a run crashes:

- fix trivial bugs and rerun
- if the idea itself is poor or unstable, drop it quickly
- record enough context that the same dead end is not repeated immediately

Do not get stuck polishing a weak idea when the evidence says to move on.

## Autonomy

Once setup is complete and the first evaluation plan is agreed, continue the loop without asking for permission after every run.

Default to action, not questions.
Do not stop at a natural pause point just because one run finished.
If the latest idea is weak, form the next hypothesis and continue.
If the latest idea is promising, refine it and continue.
If a run fails for a superficial reason, fix it and continue.

The expected mode is sustained autonomous operation:

- keep running experiments
- keep reading results
- keep updating the code
- keep using git to preserve good states
- keep going until explicitly interrupted

Do not ask the user whether to keep going, whether to try another idea, or whether this is a good stopping point.
Assume the user wants continued iteration unless they say otherwise.

Stop only when:

- the user interrupts or redirects you
- the workspace becomes inconsistent or unsafe to modify
- you hit a real blocker that cannot be resolved from local context

Otherwise, keep iterating.
