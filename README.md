# k-server-bench

This project aims to **automate and accelerate progress** toward the [k-server conjecture](https://en.wikipedia.org/wiki/K-server_problem), a central open problem in competitive analysis, using **open-ended AI-driven discovery**.

The core task is to discover a **potential function** that satisfies a large, graph-structured system of linear inequalities. The evaluation procedure is **sound but incomplete**:

* Any violated inequality definitively refutes a candidate.
* Satisfying all inequalities does *not* constitute a formal proof of the conjecture.

Nevertheless, a candidate that satisfies all constraints would provide strong evidence toward a valid proof. To the best of our knowledge, **no known potential satisfies all constraints** under our formulation for the open circle case with $$k = 4$$.

As such, a successful candidate would already represent a meaningful contribution—and could potentially lead to a full theoretical result when paired with a formal proof.

For additional mathematical background, see the [accompanying paper](TODO).

---

## Results

Among previously human-discovered solutions for the open circle case with $$k = 4$$, the best known potential has:

* **17 violations** out of **~7 million** inequalities

Using the tools in this repository together with coding agents such as OpenAI Codex, we discovered a potential with:

* **3 violations** out of **~7 million** inequalities

This problem is an extremely challenging **optimization task**:

* The search space is vast
* Improvements are incremental and rare
* Each evaluation requires checking millions of inequalities

These factors make large-scale exploration computationally and algorithmically demanding.

---

## Target Audience

This repository may be useful as:

1. **Research tooling**
   For researchers exploring potential-function approaches to the $$k$$-server problem, particularly the open circle $$k = 4$$ case and beyond.

2. **AI discovery benchmark**
   As a benchmark for **agentic, code-based mathematical discovery**, where progress is measured via automatically computed violations.

---

## Core Concepts

### Candidate (Potential Function)

A candidate is typically implemented as a Python class defining a potential function:

```python
class Potential:

  def __init__(
    self,
    context,   # k-server parameters (metric space, number of servers, etc.)
    **kwargs   # optional hyperparameters
  ):
    self.context = context
    ...

  def __call__(
    self,
    wf: np.ndarray   # representation of a graph node
  ) -> float:
    ...
```

See [`examples`](./examples/README.md) for both human-designed and AI-discovered potentials.

---

### Evaluator

The evaluator is a Python program that:

* Computes potential values
* Checks inequalities over a precomputed dataset of instances (see [`metrics/`](./metrics/README.md))

Each instance corresponds to a metric space $$M$$ and defines a graph $$(V, E)$$, where:

* Nodes $$v \in V$$ are parameterized by vectors $$w \in \mathbb{R}^d$$
* Edges $$(u, v) \in E$$ are parameterized by requests $$r \in M$$

The goal is to find a potential $$\Phi$$ such that for every edge $$(u, v, r)$$:

$$
\Phi(v) - \Phi(u) \geq \nabla(u, r, c),
$$

where:

$$
\nabla(u, r, c) =
\max_X \big(w_v(X) - w_u(X)\big) - (c + 1)\Big(\min_X w_v(X) - \min_X w_u(X)\Big).
$$

The evaluator reports:

* Number of violations
* Additional diagnostic statistics

See [`tools/`](./tools/README.md) for details.

---

Here’s a polished and cleaned-up version of your **Quickstart** section. I improved flow, fixed grammar, clarified agent/evaluator separation, and tightened phrasing while keeping your intent intact.

---

## Quickstart

### Git LFS (required for metrics)

This repository uses Git LFS to store large precomputed metric datasets.

Before cloning the repository, ensure Git LFS is installed and initialized:

```bash
git lfs install
git clone <repo-url>
cd <repo>
git lfs pull
```

Without Git LFS, the files in [`metrics/`](./metrics/README.md) will not be downloaded correctly, and the evaluator will not function.

---

### Installation

The evaluator depends on the [`k-servers`](./k-servers/README.md) package, which can be installed in an environment **separate from the agent**:

```bash
pip install -e ./k-servers
```

> This separation is recommended when running agents in isolated or containerized environments.

---

### Coding Agents

To prepare a workspace with benchmark documentation and source materials:

```bash
bash agents/setup.sh <target-dir>
```

This setup is **agent-agnostic** and copies only the necessary benchmark-facing docs and papers into the target directory.

---

Got it—that makes the structure much cleaner and more general. Here’s a polished rewrite of that section with MCP treated as a first-class alternative:

---

#### Running the evaluator

You have two main options:

**1. Direct invocation (simplest)**
Instruct your coding agent to run:

```
/abs/path/to/tools/evaluator/evaluate.py
```

This can be added to agent context files (e.g., `AGENTS.md`, `CLAUDE.md`) so the agent can call the evaluator when needed.

---

**2. MCP integration**

Alternatively, you can expose the evaluator via MCP using the entrypoint:

```
tools/evaluator/evaluate_mcp_server.py
```

This allows agents to interact with the evaluator as a tool rather than invoking it directly.

For OpenAI Codex, we provide a convenience setup script that performs both:

* the base workspace setup
* MCP configuration

```bash
pip install tomli tomli_w mcp
bash agents/codex/setup.sh <target-dir>
```

For additional details (including split-environment setups), see [`agents/codex/README.md`](./agents/codex/README.md).

---

### Sanity Check

After setup:

1. Enter the target directory
2. Start the agent
3. Run a simple test task

Example prompt:

```text
Implement a very simple potential that always returns zero and evaluate it on circle_k3_m6.pickle
```

Expected output (truncated):

```json
{
  "correct": true,
  "error": null,
  "combined_score": 0.7285714285714286,
  "public": {
    "0/violations_k": 570,
    "0/violations_k_score": 0.7285714285714286
  }
}
```

---


### Discovery Agents

For discovery systems (e.g., AlphaEvolve), the main evaluator entry point is:

```bash
tools/evaluator/evaluate.py
```

Key parameters:

* `--program_path`: path to the candidate implementation
* `--metrics_names`: comma-separated list of metric files

See:

* [`tools/evaluator/`](./tools/evaluator/)
* [`agents/docs/`](./agents/docs/)

---

### Tasks, Hints, and Presets

The [`tasks/`](./tasks) directory contains reusable task definitions, including:

* Goals
* Implementation constraints
* Hints to reduce the search space

Predefined tasks:

* [k3-stress-test](./tasks/k3-stress-test)
* [k4-general-task](./tasks/k4-general-task)

---

### Docker

You can build a fully reproducible environment using Docker.

#### Build images

```bash
docker build -t k-server-env:latest -f docker/Dockerfile .

docker build -t codex-k-server:latest -f agents/codex/Dockerfile .
```

* `k-server-env`: base environment with dependencies and tooling
* `codex-k-server`: environment configured for running Codex agents

#### Run Codex agent container

```bash
docker run --rm -it \
  -v $HOME/.codex/auth.json:/run/secrets/codex-auth.json \
  codex-k-server:latest
```
This mounts your Codex authentication file into the container and launches an interactive session.

---

## Documentation Guide

Start here:

* [`docs/concepts.md`](./docs/concepts.md): core mathematical concepts and scoring
* [`docs/task-model.md`](./docs/task-model.md): task composition and sweeps
* [`docs/evaluators.md`](./docs/evaluators.md): evaluator interfaces
* [`docs/experiments.md`](./docs/experiments.md): experiment structure

---

## Repository Structure

* [`docs/`](./docs/README.md): detailed documentation
* [`k-servers/`](./k-servers/README.md): core Python package
* [`tasks/`](./tasks/README.md): task definitions and hints
* [`experiments/`](./experiments/README.md): experiment configurations
* [`tools/`](./tools/README.md): evaluators and utilities
* [`metrics/`](./metrics/README.md): benchmark datasets
* [`examples/`](./examples/README.md): example runs and candidates
* [`tests/`](./tests/README.md): basic validation tests
* [`agents/`](./agents/README.md): agent integrations
* [`docker/`](./docker/README.md): reproducible environments

---

## Citation

TODO
