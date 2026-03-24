#!/usr/bin/env python3
from pathlib import Path
import shutil


def read_trimmed(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def main() -> int:
    sweep_dir = Path(__file__).resolve().parent
    repo_root = sweep_dir.parents[3]
    tasks_dir = repo_root / "tasks"

    goal = read_trimmed(tasks_dir / "goals" / "default" / "README.md")
    implementation = read_trimmed(
        tasks_dir / "implementation" / "potential-family-only" / "README.long.md"
    )
    canonical_hint = read_trimmed(
        tasks_dir / "hints" / "canonical-potential" / "README.md"
    )
    metrics_hint = read_trimmed(
        tasks_dir / "hints" / "metrics" / "k3_warmup" / "README.md"
    )
    (sweep_dir / "PROMPT.md").write_text(
        f"{goal}\n\n{implementation}\n\n{canonical_hint}\n\n{metrics_hint}\n",
        encoding="utf-8",
    )
    shutil.copyfile(
        tasks_dir / "implementation" / "potential-family-only" / "initial_k3.py",
        sweep_dir / "initial.py",
    )
    base_env_path = sweep_dir / "base.env"
    base_env_lines = base_env_path.read_text(encoding="utf-8").splitlines()
    filtered_lines = [
        line
        for line in base_env_lines
        if not (
            line.startswith("SHINKA_WFA_TASK_DIR=")
            or line.startswith("SHINKA_EXPERIMENT_PROMPT_PATH=")
            or line.startswith("SHINKA_EXPERIMENT_SYSTEM_PATH=")
            or line.startswith("SHINKA_WFA_INITIAL_PATH=")
        )
    ]
    filtered_lines.append('SHINKA_EXPERIMENT_PROMPT_PATH="${SWEEP_DIR}/PROMPT.md"')
    filtered_lines.append('SHINKA_EXPERIMENT_SYSTEM_PATH="${SWEEP_DIR}/SYSTEM.md"')
    filtered_lines.append('SHINKA_WFA_INITIAL_PATH="${SWEEP_DIR}/initial.py"')
    base_env_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
    sweep_path = sweep_dir / "sweep.sh"
    command = (
        'ROOT="$(git rev-parse --show-toplevel)" && '
        "$ROOT/experiments/shinka-evolve/run_evo.sh "
        "--env $ROOT/experiments/shinka-evolve/sweeps/canonical-potential-just-potential-family/base.env "
        "--env $ROOT/tasks/implementation/potential-family-only/.env "
        "--env $ROOT/tasks/hints/canonical-potential/.env "
        "--env $ROOT/tasks/hints/metrics/k3_warmup/.env"
    )
    sweep_path.write_text("\n".join([command, command, command, ""]), encoding="utf-8")
    print(f"Generated task assets in {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
