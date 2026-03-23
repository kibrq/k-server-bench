#!/usr/bin/env python3
"""Best-of-N baseline for k-server potential search.

This runner:
1) prompts an LLM with TASK + initial solution,
2) samples N candidate solutions,
3) evaluates each candidate with evaluate.py,
4) retries failed candidates with follow-up feedback up to N_REPETITIONS,
5) keeps the best scoring successful candidate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypedDict

try:
    from litellm import acompletion
except Exception:  # pragma: no cover
    acompletion = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


DEFAULT_MODEL_CONFIG = {
    "model": "openai/openai/gpt-oss-120b",
    "url": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "temperature": 0.8,
    "max_tokens": 16000,
}

STDOUT_TAIL_CHARS = 2000
STDERR_TAIL_CHARS = 4000
METRICS_FEEDBACK_CHARS = 4000
CORRECT_FEEDBACK_CHARS = 2000
STDERR_FEEDBACK_CHARS = 1500
RUN_NAME_FMT = "%Y%m%d_%H%M%S"


class AttemptRecord(TypedDict, total=False):
    sample_idx: int
    retry_idx: int
    status: str
    score: float
    summary: str
    metrics: dict[str, Any]
    correct: dict[str, Any]
    program_file: str
    error: str


class SampleRunResult(TypedDict):
    sample_idx: int
    attempts: list[AttemptRecord]
    best: AttemptRecord | None


@dataclass
class EvalResult:
    status: str
    score: float
    summary: str
    metrics: dict[str, Any]
    correct: dict[str, Any]
    returncode: int
    stdout_tail: str
    stderr_tail: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json_dict(path: Path) -> tuple[dict[str, Any], str | None]:
    if not path.is_file():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {}, f"Failed parsing {path.name}: {e}"
    if not isinstance(data, dict):
        return {}, f"Expected JSON object in {path.name}, got {type(data).__name__}"
    return data, None


def _coerce_attempts(raw_attempts: Any) -> list[AttemptRecord]:
    if not isinstance(raw_attempts, list):
        return []
    return [a for a in raw_attempts if isinstance(a, dict)]


def _resolve_int_arg(
    arg_value: int | None,
    *,
    env_key: str,
    default: int,
    field_name: str,
    min_value: int,
) -> int:
    value = arg_value if arg_value is not None else int(os.getenv(env_key, str(default)))
    if value < min_value:
        op = ">=" if min_value == 0 else ">"
        raise SystemExit(f"{field_name} must be {op} {min_value}")
    return value


def _find_eval_script(explicit_path: str | None) -> Path | None:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        return p if p.is_file() else None

    from_env = os.getenv("KSERVER_EVALUATE_SCRIPT")
    if from_env:
        p = Path(from_env).expanduser().resolve()
        if p.is_file():
            return p

    kserver_home = os.getenv("K_SERVER_EVALUATE_HOME")
    if kserver_home:
        p = Path(kserver_home).expanduser().resolve() / "evaluate.py"
        if p.is_file():
            return p

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    candidates = [
        repo_root / "tools/evaluator/evaluate.py",
        repo_root / "tools/legacy-evaluator/evaluate.py",
        repo_root / "k-servers/potential-evaluation/evaluate.py",
        repo_root.parent / "k-servers/potential-evaluation/evaluate.py",
        Path("tools/evaluator/evaluate.py"),
        Path("tools/legacy-evaluator/evaluate.py"),
        Path("../k-servers/potential-evaluation/evaluate.py"),
        Path("k-servers/potential-evaluation/evaluate.py"),
        Path.cwd() / "tools/evaluator/evaluate.py",
        Path.cwd() / "tools/legacy-evaluator/evaluate.py",
        Path.cwd() / "../k-servers/potential-evaluation/evaluate.py",
        Path.cwd() / "k-servers/potential-evaluation/evaluate.py",
    ]
    for candidate in candidates:
        p = candidate.resolve()
        if p.is_file():
            return p
    return None


def _extract_code(text: str) -> str:
    block = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text)
    if block:
        return block[-1].strip() + "\n"
    return text.strip() + "\n"


def _load_model_config(path: Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if path is None:
        return cfg

    data: dict[str, Any]
    ext = path.suffix.lower()

    if ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif ext in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except ImportError as e:
            raise RuntimeError("PyYAML is required to read YAML model configs.") from e
        except Exception as e:
            raise ValueError(f"Failed to parse YAML model config {path}: {e}") from e
        if loaded is None:
            data = {}
        elif isinstance(loaded, dict):
            data = loaded
        else:
            raise ValueError(f"YAML model config must be a mapping, got {type(loaded).__name__}")
    else:
        raise ValueError(f"Unsupported model config extension: {path}")

    llm_cfg = data.get("llm_config") if isinstance(data, dict) else None
    if isinstance(llm_cfg, dict):
        data = llm_cfg

    cfg.update({k: v for k, v in data.items() if v is not None})
    return cfg


async def _chat_completion(
    model_cfg: dict[str, Any],
    messages: list[dict[str, str]],
    candidate_dir: Path,
) -> str:
    if acompletion is None:
        raise RuntimeError("litellm is not installed. Install it to use async model calls.")

    base_url = str(model_cfg.get("url", model_cfg.get("api_base", ""))).rstrip("/")
    if not base_url:
        raise RuntimeError("Model config is missing 'url' (or 'api_base').")

    kwargs: dict[str, Any] = {
        "model": model_cfg.get("model"),
        "messages": messages,
        "temperature": float(model_cfg.get("temperature", 0.8)),
        "max_tokens": int(model_cfg.get("max_tokens", 16000)),
        "api_base": base_url,
    }

    api_key = str(model_cfg.get("api_key", ""))
    if api_key and api_key.upper() != "EMPTY":
        kwargs["api_key"] = api_key

    timeout = model_cfg.get("timeout")
    if timeout is not None:
        kwargs["timeout"] = float(timeout)

    response = await acompletion(**kwargs)

    if hasattr(response, "model_dump") and callable(getattr(response, "model_dump")):
        raw_obj = response.model_dump()
    elif isinstance(response, dict):
        raw_obj = response
    else:
        raise RuntimeError(f"Unsupported model response type: {type(response).__name__}")
    if not isinstance(raw_obj, dict):
        raise RuntimeError(f"Model response did not decode to an object: {type(raw_obj).__name__}")

    (candidate_dir / "raw_response.json").write_text(
        json.dumps(raw_obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    choices = raw_obj.get("choices") or []
    if not choices:
        raise RuntimeError("No choices returned by model.")

    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty content.")

    return content


async def _evaluate_program(
    eval_script: Path,
    program_file: Path,
    output_dir: Path,
) -> EvalResult:
    results_dir = output_dir / "eval_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    kserver_home = os.getenv("K_SERVER_EVALUATE_HOME", str(eval_script.parent))
    subprocess_timeout = float(os.getenv("BEST_OF_N_EVAL_SUBPROCESS_TIMEOUT", "900"))

    cmd = [
        sys.executable,
        str(eval_script),
        "--program_path",
        str(program_file),
        "--results_dir",
        str(results_dir),
        "--home",
        str(kserver_home),
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=subprocess_timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return EvalResult(
            status="execution_failed",
            score=0.0,
            summary=f"Evaluator timed out after {subprocess_timeout}s",
            metrics={"timeout_seconds": subprocess_timeout},
            correct={},
            returncode=-1,
            stdout_tail="",
            stderr_tail="",
        )

    stdout = stdout_b.decode("utf-8", errors="replace") if stdout_b else ""
    stderr = stderr_b.decode("utf-8", errors="replace") if stderr_b else ""

    (output_dir / "eval_stdout.txt").write_text(stdout, encoding="utf-8")
    (output_dir / "eval_stderr.txt").write_text(stderr, encoding="utf-8")

    correct_file = results_dir / "correct.json"
    metrics_file = results_dir / "metrics.json"
    correct, correct_parse_err = _read_json_dict(correct_file)
    metrics, metrics_parse_err = _read_json_dict(metrics_file)
    parse_errors = [msg for msg in (correct_parse_err, metrics_parse_err) if msg]
    parse_suffix = f" ({'; '.join(parse_errors)})" if parse_errors else ""

    if proc.returncode != 0:
        return EvalResult(
            status="execution_failed",
            score=0.0,
            summary=f"Evaluator exited with return code {proc.returncode}{parse_suffix}",
            metrics=metrics,
            correct=correct,
            returncode=int(proc.returncode),
            stdout_tail=stdout[-STDOUT_TAIL_CHARS:],
            stderr_tail=stderr[-STDERR_TAIL_CHARS:],
        )

    if not correct.get("correct", False):
        err_msg = correct.get("error", "correct.json reported failure")
        return EvalResult(
            status="execution_failed",
            score=0.0,
            summary=f"Evaluator reported incorrect program: {err_msg}{parse_suffix}",
            metrics=metrics,
            correct=correct,
            returncode=int(proc.returncode),
            stdout_tail=stdout[-STDOUT_TAIL_CHARS:],
            stderr_tail=stderr[-STDERR_TAIL_CHARS:],
        )

    raw_score = metrics.get("combined_score", 0.0)
    try:
        score = float(raw_score)
    except Exception:
        score = 0.0

    return EvalResult(
        status="success",
        score=score,
        summary="Evaluation succeeded",
        metrics=metrics,
        correct=correct,
        returncode=int(proc.returncode),
        stdout_tail=stdout[-STDOUT_TAIL_CHARS:],
        stderr_tail=stderr[-STDERR_TAIL_CHARS:],
    )


def _build_initial_prompt(task_text: str, initial_code: str) -> str:
    return (
        f"{task_text.strip()}\n\n"
        "Here is the initial solution:\n"
        "```python\n"
        f"{initial_code.rstrip()}\n"
        "```\n\n"
        "Return an improved full Python file implementing class Potential. "
        "Respond with code only."
    )


def _build_retry_feedback(eval_result: EvalResult, attempt_idx: int, max_attempts: int) -> str:
    return (
        "The previous candidate failed evaluation. "
        f"Attempt {attempt_idx}/{max_attempts}.\n\n"
        f"Failure summary: {eval_result.summary}\n"
        f"Score: {eval_result.score}\n"
        f"Metrics: {json.dumps(eval_result.metrics, ensure_ascii=False)[:METRICS_FEEDBACK_CHARS]}\n"
        f"Correct payload: {json.dumps(eval_result.correct, ensure_ascii=False)[:CORRECT_FEEDBACK_CHARS]}\n"
        f"Stderr tail: {eval_result.stderr_tail[-STDERR_FEEDBACK_CHARS:]}\n\n"
        "Please provide a corrected full Python file implementing class Potential. "
        "Respond with code only."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-of-N baseline runner")
    parser.add_argument("task_name", nargs="?", help="Task folder name under best-of-n/")

    parser.add_argument("--task-dir", type=Path, help="Task directory containing SYSTEM.md TASK.md initial.py")
    parser.add_argument("--system-file", type=Path)
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--initial-file", type=Path)
    parser.add_argument("--model-config", type=Path, help="JSON/YAML model config")

    parser.add_argument("--n", type=int, default=None, help="Number of sampled solutions")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of samples to run concurrently",
    )
    parser.add_argument(
        "--n-repetitions",
        type=int,
        default=None,
        help="Number of retry follow-up messages for failed evaluations",
    )

    parser.add_argument("--eval-script", type=Path, help="Path to k-server evaluate.py")
    parser.add_argument("--output-root", type=Path, help="Root output directory")
    parser.add_argument("--run-name", type=str, help="Optional fixed run folder name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run by skipping completed samples",
    )
    return parser.parse_args()


async def _run_single_sample(
    *,
    sample_idx: int,
    run_dir: Path,
    model_cfg: dict[str, Any],
    system_text: str,
    task_text: str,
    initial_code: str,
    n_repetitions: int,
    eval_script: Path,
) -> SampleRunResult:
    sample_dir = run_dir / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": _build_initial_prompt(task_text, initial_code)},
    ]

    attempts: list[AttemptRecord] = []
    sample_best: AttemptRecord | None = None

    for retry_idx in range(0, n_repetitions + 1):
        attempt_dir = sample_dir / f"attempt_{retry_idx:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _write_json(attempt_dir / "messages.json", messages)

        try:
            response_text = await _chat_completion(model_cfg, messages, attempt_dir)
            (attempt_dir / "model_response.txt").write_text(response_text, encoding="utf-8")
            code = _extract_code(response_text)
            program_file = attempt_dir / "program.py"
            program_file.write_text(code, encoding="utf-8")
        except Exception as e:
            result: AttemptRecord = {
                "sample_idx": sample_idx,
                "retry_idx": retry_idx,
                "status": "model_error",
                "score": 0.0,
                "error": str(e),
            }
            attempts.append(result)
            _write_json(attempt_dir / "result.json", result)
            break

        eval_result = await _evaluate_program(eval_script, program_file, attempt_dir)

        record: AttemptRecord = {
            "sample_idx": sample_idx,
            "retry_idx": retry_idx,
            "status": eval_result.status,
            "score": eval_result.score,
            "summary": eval_result.summary,
            "metrics": eval_result.metrics,
            "correct": eval_result.correct,
            "program_file": str(program_file),
        }
        attempts.append(record)
        _write_json(attempt_dir / "result.json", record)

        if sample_best is None or eval_result.score > float(sample_best.get("score", float("-inf"))):
            sample_best = record

        if eval_result.status == "success":
            break

        if retry_idx < n_repetitions:
            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": _build_retry_feedback(
                        eval_result,
                        attempt_idx=retry_idx + 1,
                        max_attempts=n_repetitions,
                    ),
                }
            )

    return SampleRunResult(sample_idx=sample_idx, attempts=attempts, best=sample_best)


async def _run_all_samples(
    *,
    n: int,
    max_concurrent: int,
    run_dir: Path,
    model_cfg: dict[str, Any],
    system_text: str,
    task_text: str,
    initial_code: str,
    n_repetitions: int,
    eval_script: Path,
    completed_samples: set[int],
    initial_attempts: list[AttemptRecord],
    initial_best: AttemptRecord | None,
    checkpoint_callback: Callable[[list[AttemptRecord], AttemptRecord | None, set[int]], None] | None = None,
) -> tuple[list[AttemptRecord], AttemptRecord | None, set[int]]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded(sample_idx: int) -> SampleRunResult:
        async with semaphore:
            return await _run_single_sample(
                sample_idx=sample_idx,
                run_dir=run_dir,
                model_cfg=model_cfg,
                system_text=system_text,
                task_text=task_text,
                initial_code=initial_code,
                n_repetitions=n_repetitions,
                eval_script=eval_script,
            )

    async def guarded(sample_idx: int) -> tuple[int, SampleRunResult | None, Exception | None]:
        try:
            return sample_idx, await bounded(sample_idx), None
        except Exception as e:
            return sample_idx, None, e

    pending_samples = [sample_idx for sample_idx in range(1, n + 1) if sample_idx not in completed_samples]
    tasks = [asyncio.create_task(guarded(sample_idx)) for sample_idx in pending_samples]

    pbar = (
        tqdm(total=n, initial=len(completed_samples), desc="best-of-n", unit="sample")
        if tqdm is not None
        else None
    )

    attempts_log: list[AttemptRecord] = list(initial_attempts)
    best_entry: AttemptRecord | None = initial_best
    finished_samples = set(completed_samples)

    try:
        for done in asyncio.as_completed(tasks):
            sample_idx, result, task_error = await done
            if task_error is not None:
                attempts_log.append(
                    AttemptRecord(
                        sample_idx=sample_idx,
                        retry_idx=0,
                        status="runner_error",
                        score=0.0,
                        error=str(task_error),
                    )
                )
                if checkpoint_callback is not None:
                    checkpoint_callback(attempts_log, best_entry, finished_samples)
            elif result is not None:
                attempts_log.extend(result["attempts"])
                sample_best = result["best"]

                if sample_best and (
                    best_entry is None
                    or float(sample_best.get("score", float("-inf"))) > float(best_entry.get("score", float("-inf")))
                ):
                    best_entry = sample_best

                finished_samples.add(sample_idx)
                if checkpoint_callback is not None:
                    checkpoint_callback(attempts_log, best_entry, finished_samples)

            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    return attempts_log, best_entry, finished_samples


def _load_resume_state(run_dir: Path, n: int) -> tuple[list[AttemptRecord], AttemptRecord | None, set[int]]:
    summary_path = run_dir / "summary.json"
    attempts: list[AttemptRecord] = []
    best: AttemptRecord | None = None

    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            raw_attempts = data.get("attempts", [])
            attempts = _coerce_attempts(raw_attempts)
            raw_best = data.get("best")
            best = raw_best if isinstance(raw_best, dict) else None
        except Exception:
            attempts = []
            best = None

    completed: set[int] = set()
    for sample_idx in range(1, n + 1):
        sample_dir = run_dir / f"sample_{sample_idx:03d}"
        if not sample_dir.is_dir():
            continue
        if any(sample_dir.glob("attempt_*/result.json")):
            completed.add(sample_idx)

    return attempts, best, completed


def _write_summary(
    run_dir: Path,
    *,
    n: int,
    max_concurrent: int,
    n_repetitions: int,
    attempts: list[AttemptRecord],
    best: AttemptRecord | None,
    completed_samples: set[int],
    finished: bool,
) -> None:
    payload = {
        "n": n,
        "max_concurrent": max_concurrent,
        "n_repetitions": n_repetitions,
        "completed_samples": sorted(completed_samples),
        "attempts": attempts,
        "best": best,
        "status": "finished" if finished else "running",
        "finished_at": datetime.now().isoformat() if finished else None,
    }
    tmp_path = run_dir / "summary.json.tmp"
    final_path = run_dir / "summary.json"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(final_path)


def _resolve_task_paths(args: argparse.Namespace, base_dir: Path) -> tuple[Path, str, Path, Path, Path]:
    if args.task_dir:
        task_dir = args.task_dir.resolve()
        try:
            task_output_name = str(task_dir.relative_to(base_dir))
        except ValueError:
            task_output_name = task_dir.name
    elif args.task_name:
        task_dir = (base_dir / args.task_name).resolve()
        task_output_name = args.task_name
    else:
        raise SystemExit("You must provide <task_name> or --task-dir")

    if not task_dir.is_dir():
        raise SystemExit(f"Task directory not found: {task_dir}")

    system_file = (args.system_file or (task_dir / "SYSTEM.md")).resolve()
    task_file = (args.task_file or (task_dir / "TASK.md")).resolve()
    initial_file = (args.initial_file or (task_dir / "initial.py")).resolve()

    for p in (system_file, task_file, initial_file):
        if not p.is_file():
            raise SystemExit(f"Missing required file: {p}")

    return task_dir, task_output_name, system_file, task_file, initial_file


def _resolve_model_config_path(task_dir: Path, arg_path: Path | None) -> Path | None:
    if arg_path is not None:
        return arg_path
    for candidate in ("model_config.json", "model_config.yaml", "model_config.yml"):
        p = task_dir / candidate
        if p.is_file():
            return p
    return None


def _resolve_run_dir(args: argparse.Namespace, outputs_root: Path, task_output_name: str) -> Path:
    task_outputs_root = outputs_root / Path(task_output_name)

    if args.run_name:
        run_name = args.run_name
    elif args.resume and task_outputs_root.is_dir():
        existing_runs = [p for p in task_outputs_root.iterdir() if p.is_dir()]
        run_name = max((p.name for p in existing_runs), default=datetime.now().strftime(RUN_NAME_FMT))
    else:
        run_name = datetime.now().strftime(RUN_NAME_FMT)

    run_dir = task_outputs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


async def _async_main(args: argparse.Namespace) -> int:
    base_dir = Path(__file__).resolve().parent
    task_dir, task_output_name, system_file, task_file, initial_file = _resolve_task_paths(args, base_dir)
    model_config_path = _resolve_model_config_path(task_dir, args.model_config)

    model_cfg = _load_model_config(model_config_path)

    n = _resolve_int_arg(
        args.n,
        env_key="BEST_OF_N_N",
        default=10,
        field_name="N",
        min_value=1,
    )
    max_concurrent = _resolve_int_arg(
        args.max_concurrent,
        env_key="BEST_OF_N_MAX_CONCURRENT",
        default=1,
        field_name="max_concurrent",
        min_value=1,
    )
    max_concurrent = min(max_concurrent, n)

    n_repetitions = _resolve_int_arg(
        args.n_repetitions,
        env_key="BEST_OF_N_REPETITIONS",
        default=2,
        field_name="N_REPETITIONS",
        min_value=0,
    )

    eval_script = _find_eval_script(str(args.eval_script) if args.eval_script else None)
    if eval_script is None:
        raise SystemExit("Could not locate evaluate.py. Set --eval-script or K_SERVER_EVALUATE_HOME.")

    output_root_arg = args.output_root
    if output_root_arg is None:
        run_grid_outputs_dir = os.getenv("RUN_GRID_OUTPUTS_DIR", "").strip()
        if run_grid_outputs_dir:
            output_root_arg = Path(run_grid_outputs_dir)

    outputs_root = (output_root_arg or (base_dir / "outputs")).resolve()
    run_dir = _resolve_run_dir(args, outputs_root, task_output_name)

    system_text = _read_text(system_file)
    task_text = _read_text(task_file)
    initial_code = _read_text(initial_file)

    run_config = {
        "task_dir": str(task_dir),
        "system_file": str(system_file),
        "task_file": str(task_file),
        "initial_file": str(initial_file),
        "model_config_path": str(model_config_path) if model_config_path else None,
        "model_config": model_cfg,
        "n": n,
        "max_concurrent": max_concurrent,
        "n_repetitions": n_repetitions,
        "eval_script": str(eval_script),
        "started_at": datetime.now().isoformat(),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    initial_attempts: list[AttemptRecord] = []
    initial_best: AttemptRecord | None = None
    completed_samples: set[int] = set()
    if args.resume:
        initial_attempts, initial_best, completed_samples = _load_resume_state(run_dir, n)

    def checkpoint_cb(
        attempts: list[AttemptRecord],
        best: AttemptRecord | None,
        completed: set[int],
    ) -> None:
        _write_summary(
            run_dir,
            n=n,
            max_concurrent=max_concurrent,
            n_repetitions=n_repetitions,
            attempts=attempts,
            best=best,
            completed_samples=completed,
            finished=False,
        )

    attempts_log, best_entry, completed_samples = await _run_all_samples(
        n=n,
        max_concurrent=max_concurrent,
        run_dir=run_dir,
        model_cfg=model_cfg,
        system_text=system_text,
        task_text=task_text,
        initial_code=initial_code,
        n_repetitions=n_repetitions,
        eval_script=eval_script,
        completed_samples=completed_samples,
        initial_attempts=initial_attempts,
        initial_best=initial_best,
        checkpoint_callback=checkpoint_cb,
    )

    _write_summary(
        run_dir,
        n=n,
        max_concurrent=max_concurrent,
        n_repetitions=n_repetitions,
        attempts=attempts_log,
        best=best_entry,
        completed_samples=completed_samples,
        finished=True,
    )

    if best_entry:
        src_program = Path(best_entry["program_file"])
        if src_program.is_file():
            dst_program = run_dir / "best_program.py"
            dst_program.write_text(src_program.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            print(f"Warning: best program file missing, skipping copy: {src_program}")

    print(f"Run directory: {run_dir}")
    if best_entry:
        print(f"Best score: {best_entry.get('score')} from {best_entry.get('program_file')}")
        return 0

    print("No valid candidate produced.")
    return 1


def main() -> int:
    args = _parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
