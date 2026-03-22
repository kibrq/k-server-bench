from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import logging
import os
import resource
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from numbers import Number
from pathlib import Path
from typing import Any

import numpy as np

EVALUATOR_HOME = Path(__file__).resolve().parent
BENCH_ROOT = EVALUATOR_HOME.parent.parent
KSERVERCLEAN_SRC = BENCH_ROOT / "k-servers" / "src"

if str(KSERVERCLEAN_SRC) not in sys.path:
    sys.path.insert(0, str(KSERVERCLEAN_SRC))

from kserverclean.evaluation import NumpyKServerInstance, compute_potential_stats  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_METRICS_ON_ERROR = {
    "combined_score": 0.0,
    "execution_time_mean": 0.0,
    "execution_time_std": 0.0,
    "num_successful_runs": 0,
    "num_valid_runs": 0,
    "num_invalid_runs": 0,
    "all_validation_errors": [],
}


class EvaluationError(Exception):
    pass


def load_program(program_path: str | os.PathLike[str], module_name: str = "user_program"):
    program_path = os.path.abspath(program_path)
    module_dir = os.path.dirname(program_path)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec from {program_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _env_default(name: str, default: Any) -> Any:
    return os.getenv(f"K_SERVER_EVALUATE_{name}", default)


def _load_json_value_or_file(value: str) -> Any:
    candidate_path = Path(value).expanduser()
    if candidate_path.is_file():
        return json.loads(candidate_path.read_text(encoding="utf-8"))
    return json.loads(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a single Potential program with a simple subprocess -> kwargs -> sequential stats pipeline."
    )
    parser.add_argument(
        "--program_path",
        type=Path,
        default=Path(_env_default("PROGRAM_PATH", "initial.py")),
        help="Path to the candidate program.",
    )
    parser.add_argument(
        "--potential_path",
        type=Path,
        default=None,
        help="Optional path to the module that defines Potential. Defaults to --program_path.",
    )
    parser.add_argument(
        "--potential_kwargs_json",
        type=str,
        default=None,
        help="Optional JSON object or path to a JSON file with potential kwargs. If provided, skips running --program_path as a candidate subprocess.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path(_env_default("RESULTS_DIR", "results")),
        help="Directory where correct.json and metrics.json are written.",
    )
    parser.add_argument(
        "--evaluate_home",
        type=Path,
        default=(Path(raw) if (raw := _env_default("HOME", str(BENCH_ROOT))) else None),
        help="Evaluation home directory. Metrics are resolved from <evaluate_home>/metrics unless --metrics_path is given.",
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=None,
        help="Optional metrics directory override. Defaults to <evaluate_home>/metrics.",
    )
    parser.add_argument(
        "--metrics_names",
        type=lambda s: [x for x in s.split(",") if x],
        default=[x for x in _env_default("METRICS_NAMES", "").split(",") if x],
        help="Comma-separated metric filenames to evaluate from the metrics directory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(_env_default("TIMEOUT", "600")),
        help="Timeout in seconds for the candidate subprocess.",
    )
    parser.add_argument(
        "--kill_after",
        type=float,
        default=float(_env_default("KILL_AFTER", "10")),
        help="Extra grace period after SIGTERM before SIGKILL.",
    )
    parser.add_argument(
        "--memory_limit_gb",
        type=float,
        default=float(_env_default("MEMORY_LIMIT_GB", "128")),
        help="Address-space limit for the candidate subprocess. Negative disables the limit.",
    )
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=int(_env_default("N_CPUS", str(os.cpu_count() // 10 or 1))),
        help="CPU count hint forwarded to the candidate program.",
    )
    parser.add_argument(
        "--exec",
        dest="exec_cmd",
        type=str,
        default=_env_default("EXEC", sys.executable),
        help="Command prefix used for the candidate subprocess, for example 'python3' or 'micromamba run -n env python'.",
    )
    parser.add_argument(
        "--compute_stats_round_digits",
        type=lambda x: None if str(x).lower() == "none" else int(x),
        default=(lambda raw: None if str(raw).lower() == "none" else int(raw))(
            _env_default("COMPUTE_STATS_ROUND_DIGITS", "4")
        ),
        help="Passed through to compute_potential_stats.",
    )
    parser.add_argument(
        "--final_evaluation_num_processes",
        type=int,
        default=int(_env_default("FINAL_EVALUATION_NUM_PROCESSES", "1")),
        help="n_processes passed to compute_potential_stats when using the mp backend.",
    )
    parser.add_argument(
        "--final_evaluation_timeout",
        type=lambda x: None if str(x).lower() == "none" else float(x),
        default=(lambda raw: None if raw is None or str(raw).lower() == "none" else float(raw))(
            _env_default("FINAL_EVALUATION_TIMEOUT", 200)
        ),
        help="Timeout passed to compute_potential_stats via compute_potential_kwargs.",
    )
    parser.add_argument(
        "--rho",
        type=lambda x: None if str(x).lower() == "none" else float(x),
        default=(lambda raw: None if raw is None or str(raw).lower() == "none" else float(raw))(
            _env_default("RHO", None)
        ),
        help="Optional rho override for compute_potential_stats.",
    )
    parser.add_argument(
        "--keep_only_violations_k",
        action="store_true",
        default=str(_env_default("KEEP_ONLY_VIOLATIONS_K", "false")).lower() in {"1", "true", "yes", "on"},
        help="Keep only violations_k metrics in compute_potential_stats output.",
    )
    parser.add_argument(
        "--robustness_check",
        action="store_true",
        default=str(_env_default("ROBUSTNESS_CHECK", "false")).lower() in {"1", "true", "yes", "on"},
        help="Run robustness checks in compute_potential_stats.",
    )
    return parser.parse_args()


def _normalize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, type):
        return obj.__name__
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_normalize(v) for v in obj]
    if isinstance(obj, Number):
        if obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    return obj


def save_json_results(
    results_dir: Path,
    metrics: dict[str, Any],
    correct: bool,
    error: str | None = None,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    correct_payload = {"correct": correct, "error": error}
    (results_dir / "correct.json").write_text(
        json.dumps(correct_payload, indent=2),
        encoding="utf-8",
    )
    (results_dir / "metrics.json").write_text(
        json.dumps(_normalize(metrics), indent=2),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_normalize(payload), indent=2),
        encoding="utf-8",
    )


def read_tail(path: Path, max_chars: int) -> str:
    if max_chars <= 0 or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def resolve_metric_paths(args: argparse.Namespace) -> list[Path]:
    if not args.metrics_names:
        raise ValueError("metrics_names must be non-empty")

    if args.metrics_path is not None:
        metrics_root = Path(args.metrics_path).expanduser().resolve()
    else:
        if args.evaluate_home is None:
            raise ValueError("Either --metrics_path or --evaluate_home must be provided")
        evaluate_home = Path(args.evaluate_home).expanduser().resolve()
        metrics_root = (evaluate_home / "metrics").resolve()

    metric_paths = [(metrics_root / name).resolve() for name in args.metrics_names]

    missing = [str(path) for path in metric_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing metric files: {missing}")
    return metric_paths


def _child_preexec_fn(memory_limit_gb: float, cpu_limit_seconds: float):
    def _fn() -> None:
        if memory_limit_gb is not None and memory_limit_gb > 0:
            memory_limit_bytes = int(memory_limit_gb * 1024 ** 3)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
        if cpu_limit_seconds is not None and cpu_limit_seconds > 0:
            cpu_limit_seconds_int = max(1, int(np.ceil(cpu_limit_seconds)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds_int, cpu_limit_seconds_int))

    return _fn


def normalize_exec_cmd(value: Any) -> list[str]:
    if isinstance(value, str):
        cmd = shlex.split(value)
    elif isinstance(value, (list, tuple)):
        cmd = [str(part) for part in value]
    else:
        raise TypeError("exec must be a string or a list/tuple of strings")

    if not cmd:
        raise ValueError("exec must not be empty")
    return cmd


def run_candidate_subprocess(
    *,
    program_path: Path,
    metric_paths: list[Path],
    output_path: Path,
    results_dir: Path,
    exec_cmd: list[str],
    timeout: float,
    n_cpus: int,
    kill_after: float,
    memory_limit_gb: float,
    cpu_limit_seconds: float,
) -> dict[str, Any]:
    cmd = [
        *exec_cmd,
        str(program_path),
        "--metrics",
        *[str(path) for path in metric_paths],
        "--timeout",
        str(timeout),
        "--n_cpus",
        str(n_cpus),
        "--output",
        str(output_path),
    ]
    logger.info("Running candidate subprocess: %s", " ".join(cmd))

    stdout_path = results_dir / "subprocess.stdout"
    stderr_path = results_dir / "subprocess.stderr"
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(program_path.parent),
            env=os.environ.copy(),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
            preexec_fn=_child_preexec_fn(memory_limit_gb, cpu_limit_seconds),
        )

        timed_out = False
        terminated_for_timeout = False
        try:
            proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            terminated_for_timeout = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.communicate(timeout=kill_after)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()

    payload = {
        "cmd": cmd,
        "cwd": str(program_path.parent),
        "returncode": proc.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_tail": read_tail(stdout_path, 2000),
        "stderr_tail": read_tail(stderr_path, 4000),
        "timed_out": timed_out,
        "terminated_for_timeout": terminated_for_timeout,
        "timeout": timeout,
        "kill_after": kill_after,
        "memory_limit_gb": memory_limit_gb,
        "cpu_limit_seconds": cpu_limit_seconds,
    }
    write_json(results_dir / "subprocess.json", payload)

    if timed_out:
        raise EvaluationError(
            f"Candidate subprocess timed out after {timeout:.2f}s. stderr tail:\n{payload['stderr_tail']}",
        )
    if proc.returncode != 0 and not (
        terminated_for_timeout and proc.returncode == -signal.SIGTERM and output_path.is_file()
    ):
        raise EvaluationError(
            "Candidate subprocess exited with return code "
            f"{proc.returncode}.\nstdout tail:\n{payload['stdout_tail']}\n\nstderr tail:\n{payload['stderr_tail']}",
        )
    if not output_path.is_file():
        raise EvaluationError("Candidate subprocess did not create its output JSON file")

    try:
        candidate_output = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvaluationError(f"Could not parse candidate output JSON: {exc}") from exc

    if not isinstance(candidate_output, dict):
        raise EvaluationError("Candidate output must be a JSON object")

    if candidate_output.get("failure") is not None:
        reason = candidate_output.get("reason")
        raise EvaluationError(f"{candidate_output['failure']}: {reason}")

    write_json(results_dir / "candidate_output.json", candidate_output)
    candidate_output["_subprocess"] = payload
    return candidate_output


def extract_potential_kwargs(candidate_output: dict[str, Any]) -> dict[str, Any]:
    if "potential_kwargs" in candidate_output:
        potential_kwargs = candidate_output["potential_kwargs"]
    elif "kwargs" in candidate_output:
        potential_kwargs = candidate_output["kwargs"]
    else:
        potential_kwargs = {
            key: value
            for key, value in candidate_output.items()
            if not str(key).startswith("_")
        }

    if not isinstance(potential_kwargs, dict):
        raise EvaluationError("Candidate output must resolve to a dict of potential kwargs")
    return potential_kwargs


def evaluate_potential(
    *,
    module: Any,
    metric_paths: list[Path],
    potential_kwargs: dict[str, Any],
    final_evaluation_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    if not hasattr(module, "Potential"):
        raise EvaluationError("Program does not define a Potential class")

    potential_cls = getattr(module, "Potential")
    if not callable(potential_cls):
        raise EvaluationError("Potential is not callable")

    instance_results: list[dict[str, Any]] = []
    instance_scores: list[float] = []
    execution_times: list[float] = []
    potential_fn = lambda context: potential_cls(context, **potential_kwargs)

    for metric_path in metric_paths:
        logger.info("Evaluating metric: %s", metric_path)
        start_time = time.perf_counter()
        kserver_instance = NumpyKServerInstance.load(metric_path)
        stats = compute_potential_stats(
            potential_fn,
            kserver_instance,
            **final_evaluation_kwargs,
        )
        elapsed = time.perf_counter() - start_time

        metrics = dict(stats.metrics)
        instance_score = float(metrics.get("violations_k_score", 0.0)) * float(
            metrics.get("processed_normalized_edges_score", 1.0)
        )

        execution_times.append(elapsed)
        instance_scores.append(instance_score)
        instance_results.append(
            {
                "metric_path": str(metric_path),
                "metric_name": metric_path.name,
                "score": instance_score,
                "unique_key": stats.unique_key,
                "metrics": metrics,
                "execution_time": elapsed,
            }
        )

    return instance_results


def build_final_evaluation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "compute_potential_backend": "mp",
        "include_wf_columns": False,
        "include_renormalized_metrics": False,
        "include_info_columns": False,
        "compute_potential_kwargs": {
            "n_processes": args.final_evaluation_num_processes,
            "timeout": args.final_evaluation_timeout,
        },
        "round_digits": args.compute_stats_round_digits,
        "rho": args.rho,
        "keep_only_violations_k": args.keep_only_violations_k,
        "robustness_check": args.robustness_check,
    }


def filter_supported_final_evaluation_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    supported = set(inspect.signature(compute_potential_stats).parameters)
    return {key: value for key, value in kwargs.items() if key in supported and value is not None}


def aggregate_metrics_old_style(
    *,
    instance_results: list[dict[str, Any]],
    potential_kwargs: dict[str, Any],
    potential_cls: str = "Potential",
) -> dict[str, Any]:
    private_only_keys = {"robustness_details"}
    combined_score = float(np.prod([result["score"] for result in instance_results], dtype=float)) if instance_results else 0.0
    execution_times = [float(result["execution_time"]) for result in instance_results]
    unique_keys = [str(result["unique_key"]) for result in instance_results]
    has_robustness_failure = False

    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    for instance_idx, result in enumerate(instance_results):
        metrics = dict(result["metrics"])
        if metrics.get("robustness", True) is False:
            has_robustness_failure = True
        for key, value in metrics.items():
            target = private if key.endswith("_time") or key in private_only_keys else public
            target[f"{instance_idx}/{key}"] = value
        private[f"{instance_idx}/metric_path"] = result["metric_path"]
        private[f"{instance_idx}/metric_name"] = result["metric_name"]
        private[f"{instance_idx}/score"] = float(result["score"])
        private[f"{instance_idx}/execution_time"] = float(result["execution_time"])

    if has_robustness_failure:
        combined_score = 0.0

    serialized_kwargs = json.dumps(_normalize(potential_kwargs), sort_keys=True)
    best_attempt = {
        "score": round(combined_score, 4),
        "kwargs": str(potential_kwargs),
        "potential_cls": potential_cls,
    }
    best_attempt_extended = {
        "score": combined_score,
        "kwargs": potential_kwargs,
        "potential_cls": potential_cls,
    }

    public.update(
        {
            "best_kwargs": str(potential_kwargs),
            "best_potential_cls": potential_cls,
            "best_kwargs_size_bytes": len(serialized_kwargs.encode("utf-8")),
            "best_kwargs_threshold_bytes": None,
            "best_kwargs_disqualified": False,
            "best_attempts_str": str([best_attempt]),
            "summary_str": str({}),
            "total_attempts": 1,
            "note": None,
            "execution_time_mean": float(np.mean(execution_times)) if execution_times else 0.0,
            "execution_time_std": float(np.std(execution_times)) if execution_times else 0.0,
            "num_successful_runs": len(instance_results),
            "num_valid_runs": len(instance_results),
            "num_invalid_runs": 0,
            "all_validation_errors": [],
        }
    )
    private.update(
        {
            "unique_key": "_".join(unique_keys),
            "best_kwargs": potential_kwargs,
            "best_potential_cls": potential_cls,
            "best_kwargs_size_bytes": len(serialized_kwargs.encode("utf-8")),
            "best_kwargs_threshold_bytes": None,
            "best_kwargs_disqualified": False,
            "best_attempts_extended": [best_attempt_extended],
            "summary": {},
        }
    )

    return {
        "combined_score": combined_score,
        "public": public,
        "private": private,
    }


def main(args: argparse.Namespace) -> tuple[dict[str, Any], bool, str | None]:
    logger.info("Evaluation args: %s", args)
    program_path = Path(args.program_path).expanduser().resolve()
    potential_path = (
        Path(args.potential_path).expanduser().resolve()
        if args.potential_path is not None
        else program_path
    )
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    metric_paths = resolve_metric_paths(args)
    exec_cmd = normalize_exec_cmd(args.exec_cmd)
    cpu_limit_seconds = (
        float(args.n_cpus) * float(args.timeout)
        if args.n_cpus is not None and args.n_cpus > 0 and args.timeout is not None and args.timeout > 0
        else -1.0
    )
    requested_final_evaluation_kwargs = build_final_evaluation_kwargs(args)
    final_evaluation_kwargs = filter_supported_final_evaluation_kwargs(
        requested_final_evaluation_kwargs
    )

    if not program_path.is_file():
        raise FileNotFoundError(f"Program file not found: {program_path}")
    if not potential_path.is_file():
        raise FileNotFoundError(f"Potential file not found: {potential_path}")

    module = load_program(potential_path, module_name=f"user_program_{int(time.time() * 1e6)}")
    if not hasattr(module, "Potential"):
        raise EvaluationError("Potential program does not define Potential")

    if args.potential_kwargs_json is not None:
        candidate_output = {"potential_kwargs": _load_json_value_or_file(args.potential_kwargs_json)}
    else:
        with tempfile.TemporaryDirectory(prefix="kserver_eval_") as tmpdir:
            output_path = Path(tmpdir) / "candidate_output.json"
            candidate_output = run_candidate_subprocess(
                program_path=program_path,
                metric_paths=metric_paths,
                output_path=output_path,
                results_dir=results_dir,
                exec_cmd=exec_cmd,
                timeout=args.timeout,
                n_cpus=args.n_cpus,
                kill_after=args.kill_after,
                memory_limit_gb=args.memory_limit_gb,
                cpu_limit_seconds=cpu_limit_seconds,
            )

    potential_kwargs = extract_potential_kwargs(candidate_output)
    instance_results = evaluate_potential(
        module=module,
        metric_paths=metric_paths,
        potential_kwargs=potential_kwargs,
        final_evaluation_kwargs=final_evaluation_kwargs,
    )
    metrics = aggregate_metrics_old_style(
        instance_results=instance_results,
        potential_kwargs=potential_kwargs,
        potential_cls="Potential",
    )
    return metrics, True, None


if __name__ == "__main__":
    args = parse_args()
    try:
        metrics, correct, error = main(args)
    except KeyboardInterrupt:
        metrics = dict(DEFAULT_METRICS_ON_ERROR)
        correct = False
        error = "Evaluation interrupted"
    except Exception:
        metrics = dict(DEFAULT_METRICS_ON_ERROR)
        metrics["traceback"] = traceback.format_exc()
        correct = False
        error = metrics["traceback"]

    save_json_results(
        results_dir=Path(args.results_dir).expanduser().resolve(),
        metrics=metrics,
        correct=correct,
        error=error,
    )
