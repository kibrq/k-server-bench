import os
from typing import Optional, List, Dict, Literal, Annotated
from datetime import datetime
import shlex

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from pydantic import Field
import logging
import sys
import json
from pathlib import Path

logger = logging.getLogger(__name__)

LEGACY_EVALUATOR_HOME = Path(__file__).resolve().parent

if str(LEGACY_EVALUATOR_HOME) not in sys.path:
    sys.path.insert(0, str(LEGACY_EVALUATOR_HOME))

# Name of your MCP server (shows up in clients like ChatGPT / Claude Desktop)
TOOL_TIMEOUT_SECONDS = float(os.getenv("K_SERVER_MCP_TOOL_TIMEOUT_SECONDS", "1"))
# mcp = FastMCP("eval-server", timeout=TOOL_TIMEOUT_SECONDS)
mcp = FastMCP("eval-server")

K_SERVER_MCP_EVALUATE_PATH = os.getenv(
    "K_SERVER_MCP_EVALUATE_PATH",
    str(LEGACY_EVALUATOR_HOME / "evaluate.py"),
)
K_SERVER_MCP_DEFAULT_FILENAME = os.getenv("K_SERVER_MCP_DEFAULT_FILENAME", "main.py")
K_SERVER_MCP_KILL_GRACE_SECONDS = float(os.getenv("K_SERVER_MCP_KILL_GRACE_SECONDS", "3.0"))
K_SERVER_MCP_CMD_PREFIX = os.getenv("K_SERVER_MCP_CMD_PREFIX", sys.executable)

from evaluate import main as evaluate_main, parse_args
import asyncio


def _split_cmd_prefix(cmd_prefix: str) -> list[str]:
    parts = shlex.split(cmd_prefix)
    if not parts:
        raise ValueError("cmd_prefix must not be empty")
    return parts


@mcp.tool()
async def evaluate_directory(
    ctx: Context[ServerSession, None],

    directory: Annotated[
        str,
        Field(description="Path to the directory containing the implemented solution."),
    ],

    metrics_names: Annotated[
        Optional[str],
        Field(description="Optional comma-separated list of metric names to evaluate."),
    ] = None,

    home: Annotated[
        Optional[str],
        Field(description="Optional home directory for the evaluation. If provided, will use the default potential family, search evaluator, and potential files in the home directory."),
    ] = None,

    use_default_potential_family: Annotated[
        bool,
        Field(description="Whether to use the default potential family file in the home directory."),
    ] = False,

    use_default_search_evaluator: Annotated[
        bool,
        Field(description="Whether to use the default search evaluator file in the home directory."),
    ] = False,

    use_default_potential: Annotated[
        bool,
        Field(description="Whether to use the default potential file in the home directory."),
    ] = False,

    potential_family_kwargs_path: Annotated[
        Optional[str],
        Field(description="Optional path to the potential family kwargs file (json)."),
    ] = None,

    shared_readonly_ray_resources_kwargs_path: Annotated[
        Optional[str],
        Field(description="Optional path to the shared readonly ray resources kwargs file (json)."),
    ] = None,

    skip_final_evaluation: Annotated[
        bool,
        Field(description="Whether to skip final evaluation."),
    ] = None,

    program_filename: Annotated[
        Optional[str],
        Field(description=f"Optional relative filename to the program file. (Default: {K_SERVER_MCP_DEFAULT_FILENAME})"),
    ] = K_SERVER_MCP_DEFAULT_FILENAME,

    potential_family_path: Annotated[
        Optional[str],
        Field(description="Optional path to the potential family file."),
    ] = None,

    potential_family_filename: Annotated[
        Optional[str],
        Field(description="Optional relative filename to the potential family file."),
    ] = None,

    potential_path: Annotated[
        Optional[str],
        Field(description="Optional path to the potential file."),
    ] = None,

    potential_filename: Annotated[
        Optional[str],
        Field(description="Optional filename to the potential file."),
    ] = None,

    search_evaluator_path: Annotated[
        Optional[str],
        Field(description="Optional path to the search evaluator file."),
    ] = None,

    search_evaluator_filename: Annotated[
        Optional[str],
        Field(description="Optional filename to the search evaluator file."),
    ] = None,

    ask_tell_max_queue_size: Annotated[
        Optional[int],
        Field(description="Max size of the ask/tell queue for search workers (env var, then default if None)."),
    ] = None,
    search_timeout: Annotated[
        Optional[float],
        Field(description="Total search timeout in seconds (None disables; env var, then default if None)."),
    ] = None,
    search_timeout_buffer: Annotated[
        Optional[float],
        Field(description="Buffer before stopping search in seconds (env var, then default if None)."),
    ] = None,
    search_max_concurrent: Annotated[
        Optional[int],
        Field(description="Max number of concurrent search tasks (env var, then default if None)."),
    ] = None,
    search_max_worker_timeout: Annotated[
        Optional[float],
        Field(description="Max per-worker timeout in seconds (env var, then default if None)."),
    ] = None,
    search_min_worker_timeout: Annotated[
        Optional[float],
        Field(description="Min per-worker timeout in seconds (env var, then default if None)."),
    ] = None,
    final_evaluation_timeout: Annotated[
        Optional[float],
        Field(description="Per-evaluation timeout in seconds (env var, then default if None)."),
    ] = None,
    final_evaluation_timeout_buffer: Annotated[
        Optional[float],
        Field(description="Buffer before stopping final evaluation in seconds (env var, then default if None)."),
    ] = None,
    final_evaluation_max_concurrent: Annotated[
        Optional[int],
        Field(description="Max concurrent final evaluation tasks (env var, then default if None)."),
    ] = None,
    final_evaluation_top_k: Annotated[
        Optional[int],
        Field(description="Number of top search results to run final evaluation on (env var, then default if None)."),
    ] = None,
    attempt_id: Annotated[
        Optional[str],
        Field(description="Optional attempt id for results; defaults to current date."),
    ] = None,
    rho: Annotated[
        Optional[float],
        Field(description="rho parameter for compute_potential_stats (env var, then default if None)."),
    ] = None,
    cmd_prefix: Annotated[
        Optional[str],
        Field(description="Command prefix used to launch the evaluator process itself. Defaults to K_SERVER_MCP_CMD_PREFIX or the current Python executable."),
    ] = None,
):
    """
    Evaluate the default program file inside the given directory and store results in the same directory.
    """

    try:

        if attempt_id is None:
            attempt_id = "results_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        assert not "/" in program_filename, "program_filename should be a relative filename"
        assert program_filename is not None, "program_filename must be not None"

        program_path = os.path.join(directory, program_filename)

        if potential_family_path is None:
            if not potential_family_filename is None:
                potential_family_path = os.path.join(directory, potential_family_filename)

        if potential_path is None:
            if not potential_filename is None:
                potential_path = os.path.join(directory, potential_filename)

        if search_evaluator_path is None:
            if not search_evaluator_filename is None:
                search_evaluator_path = os.path.join(directory, search_evaluator_filename)

        results_dir = os.path.join(directory, attempt_id)
        
        kwargs = dict(
            program_path=program_path,
            results_dir=results_dir,
            potential_family_path=potential_family_path,
            potential_path=potential_path,
            search_evaluator_path=search_evaluator_path,
            ask_tell_max_queue_size=ask_tell_max_queue_size,
            metrics_names=metrics_names,
            search_timeout=search_timeout,
            search_timeout_buffer=search_timeout_buffer,
            search_max_concurrent=search_max_concurrent,
            search_max_worker_timeout=search_max_worker_timeout,
            search_min_worker_timeout=search_min_worker_timeout,
            final_evaluation_timeout=final_evaluation_timeout,
            final_evaluation_timeout_buffer=final_evaluation_timeout_buffer,
            final_evaluation_max_concurrent=final_evaluation_max_concurrent,
            final_evaluation_top_k=final_evaluation_top_k,
            timeout=None,
            home=home,
            use_default_potential_family=use_default_potential_family,
            use_default_search_evaluator=use_default_search_evaluator,
            use_default_potential=use_default_potential,
            potential_family_kwargs_path=potential_family_kwargs_path,
            shared_readonly_ray_resources_kwargs_path=shared_readonly_ray_resources_kwargs_path,
            skip_final_evaluation=skip_final_evaluation,
            rho=rho,
        )

        arguments = [
            "--program_path", program_path,
            "--results_dir", results_dir,
        ]
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    arguments.append(f"--{key}")
                continue
            arguments.append(f"--{key}={value}")

        logout_filename = os.path.join(results_dir, "log.out")
        logerr_filename = os.path.join(results_dir, "log.err")

        os.makedirs(results_dir, exist_ok=True)

        launch_prefix = _split_cmd_prefix(cmd_prefix or K_SERVER_MCP_CMD_PREFIX)

        with open(logout_filename, "w") as out, open(logerr_filename, "w") as err:

            proc = await asyncio.create_subprocess_exec(
                *launch_prefix, K_SERVER_MCP_EVALUATE_PATH, *arguments,
                stdout=out, stderr=err,
            )
            await ctx.info(f"Evaluating {program_filename} inside {directory}")
            rc = await proc.wait()

        await ctx.info(f"Evaluation complete with return code {rc}")

        if (Path(results_dir) / "correct.json").exists():
            correct_dict = json.load((Path(results_dir) / "correct.json").open())
        else:
            correct_dict = {"correct": False, "error_msg": "No correct.json file found"}

        if not correct_dict["correct"]:
            return correct_dict

        if (Path(results_dir) / "metrics.json").exists():
            metrics_dict = json.load((Path(results_dir) / "metrics.json").open())
        else:
            metrics_dict = {"error": "No metrics.json file found"}

        return {**correct_dict, **metrics_dict}
    except Exception as e:
        import traceback
        return {"error": traceback.format_exc()}
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mcp.run(transport="stdio")
