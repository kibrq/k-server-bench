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
        tasks_dir / "implementation" / "non-legacy-evaluator" / "README.md"
    )
    canonical_hint = read_trimmed(
        tasks_dir / "hints" / "canonical-potential" / "README.md"
    )
    symmetric_hint = read_trimmed(
        tasks_dir
        / "hints"
        / "canonical-potential"
        / "symmetric-matrices"
        / "README.md"
    )
    metrics_hint = read_trimmed(
        tasks_dir / "hints" / "metrics" / "k4_general_task" / "README.md"
    )
    (sweep_dir / "TASK.md").write_text(
        f"{goal}\n\n{implementation}\n\n{canonical_hint}\n\n{symmetric_hint}\n\n{metrics_hint}\n",
        encoding="utf-8",
    )
    shutil.copyfile(
        tasks_dir / "implementation" / "non-legacy-evaluator" / "initial.py",
        sweep_dir / "initial.py",
    )
    sweep_path = sweep_dir / "sweep.sh"
    command = (
        f"{repo_root / 'experiments' / 'loong-flow' / 'run_math.sh'} "
        f"--env {sweep_dir / 'base.env'} "
        f"--env {tasks_dir / 'hints' / 'metrics' / 'k4_general_task' / '.env'} "
        f"-- --config {sweep_dir / 'config.yaml'} "
        f"--eval-file {repo_root / 'experiments' / 'loong-flow' / 'eval_program.py'} "
        f"--initial-file {sweep_dir / 'initial.py'} "
        "--workspace-path ./workspace "
        f"--task-file {sweep_dir / 'TASK.md'} "
        "--log-level DEBUG"
    )
    sweep_path.write_text("\n".join([command, command, command, ""]), encoding="utf-8")
    print(f"Generated task assets in {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
