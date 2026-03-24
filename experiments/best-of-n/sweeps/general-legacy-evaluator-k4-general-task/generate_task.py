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
        tasks_dir / "implementation" / "general-legacy-evaluator" / "README.long.md"
    )
    metrics_hint = read_trimmed(
        tasks_dir / "hints" / "metrics" / "k4_general_task" / "README.md"
    )
    (sweep_dir / "TASK.md").write_text(
        f"{goal}\n\n{implementation}\n\n{metrics_hint}\n",
        encoding="utf-8",
    )
    shutil.copyfile(
        tasks_dir / "implementation" / "general-legacy-evaluator" / "initial.py",
        sweep_dir / "initial.py",
    )
    sweep_path = sweep_dir / "sweep.sh"
    command = (
        f"{repo_root / 'experiments' / 'best-of-n' / 'run_best_of_n.sh'} "
        f"sweeps/general-legacy-evaluator-k4-general-task "
        f"--env {sweep_dir / 'base.env'} "
        f"--env {tasks_dir / 'implementation' / 'general-legacy-evaluator' / '.env'} "
        f"--env {tasks_dir / 'hints' / 'metrics' / 'k4_general_task' / '.env'} "
        f"-- --eval-script {repo_root / 'tools' / 'legacy-evaluator' / 'evaluate.py'} "
        f"--model-config {sweep_dir / 'model_config.json'}"
    )
    sweep_path.write_text(command + "\n", encoding="utf-8")
    print(f"Generated task assets in {sweep_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
