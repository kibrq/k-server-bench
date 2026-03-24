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
        tasks_dir / "hints" / "metrics" / "k4_general_task" / "README.md"
    )
    (sweep_dir / "TASK.md").write_text(
        f"{goal}\n\n{implementation}\n\n{canonical_hint}\n\n{metrics_hint}\n",
        encoding="utf-8",
    )
    shutil.copyfile(
        tasks_dir / "implementation" / "potential-family-only" / "initial_k4.py",
        sweep_dir / "initial.py",
    )
    sweep_path = sweep_dir / "sweep.sh"
    command = (
        'ROOT="$(git rev-parse --show-toplevel)" && '
        "$ROOT/experiments/loong-flow/run_math.sh "
        "--env $ROOT/experiments/loong-flow/sweeps/canonical-potential-just-potential-family-k4-general-task/base.env "
        "--env $ROOT/tasks/implementation/potential-family-only/.env "
        "--env $ROOT/tasks/hints/canonical-potential/.env "
        "--env $ROOT/tasks/hints/metrics/k4_general_task/.env "
        "-- --config $ROOT/experiments/loong-flow/sweeps/canonical-potential-just-potential-family-k4-general-task/config.yaml "
        "--eval-file $ROOT/experiments/loong-flow/eval_program.py "
        "--initial-file $ROOT/experiments/loong-flow/sweeps/canonical-potential-just-potential-family-k4-general-task/initial.py "
        "--workspace-path ./workspace "
        "--task-file $ROOT/experiments/loong-flow/sweeps/canonical-potential-just-potential-family-k4-general-task/TASK.md "
        "--log-level DEBUG"
    )
    sweep_path.write_text("\n".join([command, command, command, ""]), encoding="utf-8")
    print(f"Generated task assets in {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
