"""Math-agent evaluator bridge for k-server-bench evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def _find_eval_script() -> Path | None:
    from_env = os.getenv("KSERVER_EVALUATE_SCRIPT")
    if from_env:
        path = Path(from_env).expanduser().resolve()
        return path if path.is_file() else None

    from_home = os.getenv("K_SERVER_EVALUATE_HOME")
    if from_home:
        path = (Path(from_home).expanduser() / "evaluate.py").resolve()
        if path.is_file():
            return path

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "tools" / "evaluator" / "evaluate.py",
        repo_root / "tools" / "legacy-evaluator" / "evaluate.py",
        Path.cwd() / "tools" / "evaluator" / "evaluate.py",
        Path.cwd() / "tools" / "legacy-evaluator" / "evaluate.py",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def evaluate(program_path: str) -> dict[str, Any]:
    eval_script = _find_eval_script()
    if eval_script is None:
        return {
            "status": "framework_error",
            "score": 0.0,
            "summary": "Cannot find k-server-bench evaluator.",
            "metrics": {},
            "artifacts": {},
        }

    program_file = Path(program_path).expanduser().resolve()
    if not program_file.is_file():
        return {
            "status": "validation_failed",
            "score": 0.0,
            "summary": f"Program file not found: {program_file}",
            "metrics": {},
            "artifacts": {},
        }

    with tempfile.TemporaryDirectory(prefix="kserver_eval_") as tmpdir:
        results_dir = Path(tmpdir) / "results"
        eval_env = os.getenv("K_SERVER_EVAL_ENV", "").strip()
        eval_python = os.getenv("KSERVER_EVAL_PYTHON", "").strip()

        if eval_env:
            cmd = [
                "micromamba",
                "run",
                "-n",
                eval_env,
                "python",
                str(eval_script),
                "--program_path",
                str(program_file),
                "--results_dir",
                str(results_dir),
            ]
        else:
            python_bin = eval_python or sys.executable
            cmd = [
                python_bin,
                str(eval_script),
                "--program_path",
                str(program_file),
                "--results_dir",
                str(results_dir),
            ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        correct_data: dict[str, Any] = {}
        metrics_data: dict[str, Any] = {}
        correct_file = results_dir / "correct.json"
        metrics_file = results_dir / "metrics.json"
        if correct_file.is_file():
            correct_data = json.loads(correct_file.read_text(encoding="utf-8"))
        if metrics_file.is_file():
            metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))

        if proc.returncode != 0:
            return {
                "status": "execution_failed",
                "score": 0.0,
                "summary": "k-server-bench evaluator exited with non-zero status",
                "metrics": {"returncode": proc.returncode, **metrics_data},
                "artifacts": {
                    "stdout_tail": proc.stdout[-1000:],
                    "stderr_tail": proc.stderr[-2000:],
                    "correct": correct_data,
                },
            }

        if not correct_data.get("correct", False):
            return {
                "status": "execution_failed",
                "score": 0.0,
                "summary": correct_data.get("error", "Evaluator reported failure"),
                "metrics": metrics_data,
                "artifacts": {
                    "correct": correct_data,
                    "stdout_tail": proc.stdout[-1000:],
                    "stderr_tail": proc.stderr[-2000:],
                },
            }

        raw_score = metrics_data.get("combined_score", 0.0)
        return {
            "status": "success",
            "score": float(raw_score),
            "summary": f"k-server evaluation succeeded (raw={raw_score})",
            "metrics": metrics_data,
            "artifacts": {
                "program_file": str(program_file),
                "eval_script": str(eval_script),
            },
        }
