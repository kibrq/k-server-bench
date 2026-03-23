import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from pydantic import Field


logger = logging.getLogger(__name__)

EVALUATOR_HOME = Path(__file__).resolve().parent
BENCH_ROOT = EVALUATOR_HOME.parent.parent

mcp = FastMCP("eval-server")

K_SERVER_MCP_EVALUATE_PATH = os.getenv(
    "K_SERVER_MCP_EVALUATE_PATH",
    str(EVALUATOR_HOME / "evaluate.py"),
)
K_SERVER_MCP_DEFAULT_FILENAME = os.getenv("K_SERVER_MCP_DEFAULT_FILENAME", "main.py")
K_SERVER_EVALUATE_HOME = os.getenv("K_SERVER_EVALUATE_HOME", str(BENCH_ROOT))


def _append_optional_flag(arguments: list[str], flag: str, value) -> None:
    if value is None:
        return
    arguments.extend([flag, str(value)])


def _append_optional_bool_flag(arguments: list[str], flag: str, value: Optional[bool]) -> None:
    if value:
        arguments.append(flag)


def _resolve_path(directory: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if os.path.isabs(value):
        return value
    return os.path.join(directory, value)


@mcp.tool()
async def evaluate_directory(
    ctx: Context[ServerSession, None],
    directory: Annotated[
        str,
        Field(description="Absolute path containing the implemented solution."),
    ],
    metrics_names: Annotated[
        Optional[str],
        Field(description="Optional comma-separated list of metric names to evaluate."),
    ] = None,
    program_filename: Annotated[
        Optional[str],
        Field(description=f"Optional relative filename to the program file. (Default: {K_SERVER_MCP_DEFAULT_FILENAME})"),
    ] = K_SERVER_MCP_DEFAULT_FILENAME,
    timeout: Annotated[
        Optional[float],
        Field(description="Candidate subprocess timeout."),
    ] = None,
    potential_qualname: Annotated[
        Optional[str],
        Field(description="Qualified name of the Potential class within the loaded program module. Defaults to Potential."),
    ] = None,
    potential_kwargs_json: Annotated[
        Optional[str],
        Field(description="Optional JSON object or path to a JSON file with potential kwargs."),
    ] = None,
    results_dir: Annotated[
        Optional[str],
        Field(description="Optional relative or absolute results directory override. Defaults to <directory>/<attempt_id>."),
    ] = None,
    evaluate_home: Annotated[
        Optional[str],
        Field(description="Optional evaluation home directory override."),
    ] = None,
    metrics_path: Annotated[
        Optional[str],
        Field(description="Optional metrics directory override."),
    ] = None,
    kill_after: Annotated[
        Optional[float],
        Field(description="Grace period after SIGTERM before SIGKILL."),
    ] = None,
    memory_limit_gb: Annotated[
        Optional[float],
        Field(description="Address-space limit for the candidate subprocess. Negative disables the limit."),
    ] = None,
    n_cpus: Annotated[
        Optional[int],
        Field(description="CPU count hint forwarded to the candidate program."),
    ] = None,
    exec_cmd: Annotated[
        Optional[str],
        Field(description="Command prefix used for the candidate subprocess, matching evaluate.py --exec."),
    ] = None,
    compute_stats_round_digits: Annotated[
        Optional[int],
        Field(description="Round digits passed through to compute_potential_stats."),
    ] = None,
    final_evaluation_timeout: Annotated[
        Optional[float],
        Field(description="Per-instance final evaluation timeout."),
    ] = None,
    final_evaluation_num_processes: Annotated[
        Optional[int],
        Field(description="n_processes passed to compute_potential_stats when using the mp backend."),
    ] = None,
    final_evaluation_max_concurrent: Annotated[
        Optional[int],
        Field(description="Deprecated alias for final_evaluation_num_processes."),
    ] = None,
    attempt_id: Annotated[
        Optional[str],
        Field(description="Optional attempt id for results; defaults to current date."),
    ] = None,
    rho: Annotated[
        Optional[float],
        Field(description="rho parameter for compute_potential_stats."),
    ] = None,
    keep_only_violations_k: Annotated[
        Optional[bool],
        Field(description="Keep only violations_k metrics in compute_potential_stats output."),
    ] = None,
    robustness_check: Annotated[
        Optional[bool],
        Field(description="Run robustness checks in compute_potential_stats."),
    ] = None,
    raise_on_missing_candidate_output: Annotated[
        Optional[bool],
        Field(description="Raise if the candidate subprocess does not create its output JSON file. By default evaluation falls back to empty potential kwargs."),
    ] = None,
):
    """
    Evaluate the program file inside the given directory using the draft-v2/260317 evaluator.
    """

    try:
        if attempt_id is None:
            attempt_id = "results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if program_filename is None:
            raise ValueError("program_filename must not be None")
        if "/" in program_filename:
            raise ValueError("program_filename should be a relative filename")

        program_path = os.path.join(directory, program_filename)
        resolved_results_dir = _resolve_path(directory, results_dir) if results_dir is not None else os.path.join(directory, attempt_id)
        resolved_evaluate_home = _resolve_path(directory, evaluate_home)
        resolved_metrics_path = _resolve_path(directory, metrics_path)
        final_evaluation_processes = final_evaluation_num_processes
        if final_evaluation_processes is None:
            final_evaluation_processes = final_evaluation_max_concurrent

        results_dir = resolved_results_dir
        logout_filename = os.path.join(results_dir, "log.out")
        logerr_filename = os.path.join(results_dir, "log.err")

        arguments = [
            "--program_path",
            program_path,
            "--results_dir",
            results_dir,
        ]
        _append_optional_flag(arguments, "--metrics_names", metrics_names)
        _append_optional_flag(arguments, "--timeout", timeout)
        _append_optional_flag(arguments, "--potential_qualname", potential_qualname)
        _append_optional_flag(arguments, "--potential_kwargs_json", potential_kwargs_json)
        _append_optional_flag(arguments, "--evaluate_home", resolved_evaluate_home)
        _append_optional_flag(arguments, "--metrics_path", resolved_metrics_path)
        _append_optional_flag(arguments, "--kill_after", kill_after)
        _append_optional_flag(arguments, "--memory_limit_gb", memory_limit_gb)
        _append_optional_flag(arguments, "--n_cpus", n_cpus)
        _append_optional_flag(arguments, "--exec", exec_cmd)
        _append_optional_flag(arguments, "--compute_stats_round_digits", compute_stats_round_digits)
        _append_optional_flag(arguments, "--final_evaluation_timeout", final_evaluation_timeout)
        _append_optional_flag(arguments, "--final_evaluation_num_processes", final_evaluation_processes)
        _append_optional_flag(arguments, "--rho", rho)
        _append_optional_bool_flag(arguments, "--keep_only_violations_k", keep_only_violations_k)
        _append_optional_bool_flag(arguments, "--robustness_check", robustness_check)
        _append_optional_bool_flag(arguments, "--raise_on_missing_candidate_output", raise_on_missing_candidate_output)

        os.makedirs(results_dir, exist_ok=True)
        await ctx.info(f"Evaluating {program_filename} inside {directory} with {K_SERVER_MCP_EVALUATE_PATH}")

        with open(logout_filename, "w") as out, open(logerr_filename, "w") as err:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                K_SERVER_MCP_EVALUATE_PATH,
                *arguments,
                stdout=out,
                stderr=err,
            )
            rc = await proc.wait()

        await ctx.info(f"Evaluation complete with return code {rc}")

        correct_path = Path(results_dir) / "correct.json"
        metrics_path = Path(results_dir) / "metrics.json"

        if correct_path.exists():
            correct_dict = json.loads(correct_path.read_text())
        else:
            correct_dict = {"correct": False, "error": "No correct.json file found"}

        if not correct_dict.get("correct"):
            return correct_dict

        if metrics_path.exists():
            metrics_dict = json.loads(metrics_path.read_text())
        else:
            metrics_dict = {"error": "No metrics.json file found"}

        return {**correct_dict, **metrics_dict}
    except Exception:
        import traceback

        return {"error": traceback.format_exc()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")
