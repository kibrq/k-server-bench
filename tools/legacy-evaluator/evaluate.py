from os import PathLike
import time
from traceback import format_exc
import numpy as np
import ray
import os
import sys
import importlib.util
import argparse
import resource
import uuid
from pathlib import Path

LEGACY_EVALUATOR_HOME = Path(__file__).resolve().parent
BENCH_ROOT = LEGACY_EVALUATOR_HOME.parent.parent
KSERVERCLEAN_SRC = BENCH_ROOT / "k-servers" / "src"
DEFAULT_METRICS_DIR = BENCH_ROOT / "metrics"

from kserver.evaluation import NumpyKServerInstance, compute_potential_stats
from ray_kserver_instance import KServerInstanceRayFriendly, to_ray_friendly
from ray_utils import connect_or_restart_ray

from typing import List, Dict, Any, Callable, Optional, Tuple
from collections import deque
import logging
from datetime import datetime

from numbers import Number
import json
from functools import partial



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)



def load_program(program_path: str, module_name: str = "user_program"):
    

    program_path = os.path.abspath(program_path)
    module_dir = os.path.dirname(program_path)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location(module_name, program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



# --- module-level cache on each Ray worker process ---

_random_cache_key = uuid.uuid4().hex

_potential_family_module_cache = None
_potential_module_cache = None
_search_evaluator_module_cache = None


def _get_shared_readonly_resource(shared_readonly_ray_resources, name: str):
    if not shared_readonly_ray_resources:
        return None
    return shared_readonly_ray_resources.get(name)

def _ensure_loaded(
    potential_family_path: str,
    potential_path: str,
    search_evaluator_path: str,
):
    """Lazy-init per *worker process*."""
    global _potential_family_module_cache, _potential_module_cache, _search_evaluator_module_cache
    logger = logging.getLogger(__name__)

    if _potential_family_module_cache is None:
        _potential_family_module_cache = {}
    if _potential_module_cache is None:
        _potential_module_cache = {}
    if _search_evaluator_module_cache is None:
        _search_evaluator_module_cache = {}

    potential_family_key = f"{potential_family_path}_{_random_cache_key}"
    potential_key = f"{potential_path}_{_random_cache_key}"
    search_evaluator_key = f"{search_evaluator_path}_{_random_cache_key}"

    if potential_family_key not in _potential_family_module_cache:
        logger.debug(f"Loading program {potential_family_path}")
        _potential_family_module_cache[potential_family_key] = load_program(
            potential_family_path, module_name="potential_family"
        )

    if potential_key not in _potential_module_cache:
        logger.debug(f"Loading program {potential_path}")
        _potential_module_cache[potential_key] = load_program(
            potential_path, module_name="potential"
        )

    if search_evaluator_key not in _search_evaluator_module_cache:
        logger.debug(f"Loading program {search_evaluator_path}")
        _search_evaluator_module_cache[search_evaluator_key] = load_program(
            search_evaluator_path, module_name="search_evaluator"
        )

    return (
        _potential_family_module_cache[potential_family_key],
        _potential_module_cache[potential_key],
        _search_evaluator_module_cache[search_evaluator_key],
    )


def load_shared_readonly_ray_resources(
    train_instances: Optional[List[PathLike]] = None,
    **np_paths: Dict[str, PathLike],
) -> Dict[Any, Any]:
    shared_resources: Dict[Any, Any] = {}

    if not train_instances:
        return shared_resources

    train_instances_loaded = []
    for inst in train_instances:
        if isinstance(inst, KServerInstanceRayFriendly):
            ray_friendly = inst
        else:
            if isinstance(inst, NumpyKServerInstance):
                raw_instance = inst
            else:
                raw_instance = NumpyKServerInstance.load(inst)
            ray_friendly = to_ray_friendly(raw_instance)
        train_instances_loaded.append(ray.put(ray_friendly))

    shared_resources["train_instances"] = train_instances_loaded

    for name, path in np_paths.items():
        shared_resources[name] = ray.put(np.load(path))

    return shared_resources


@ray.remote(max_calls=100)
def submit_fn_remote(
    potential_family_path: str,
    potential_path: str,
    search_evaluator_path: str,
    shared_readonly_ray_resources: Optional[Dict[Any, Any]] = None,
    potential_cls: str = "Potential",
    potential_kwargs: dict | None = None,
    search_evaluator_cls: str = "SearchEvaluator",
    search_evaluator_kwargs: dict | None = None,
    memory_limit: float = 10,
    timeout: float = None,
):
    try:
        start_time = time.time()

        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        if memory_limit < 0:
            logger.debug(f"Memory limit is negative, setting to unlimited")
        else:
            memory_limit_bytes = int(memory_limit * 1024 ** 3)
            logger.debug(f"Setting memory limit for worker to {memory_limit} gbs")
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, hard))

        # Lazily load once per worker process
        (
            potential_family_module,
            potential_module,
            search_evaluator_module,
        ) = _ensure_loaded(
            potential_family_path,
            potential_path,
            search_evaluator_path,
        )

        train_instances_value = _get_shared_readonly_resource(
            shared_readonly_ray_resources, "train_instances"
        )
        if train_instances_value is None:
            raise ValueError(
                "shared_readonly_ray_resources must include train_instances"
            )

        if isinstance(train_instances_value, ray.ObjectRef):
            train_instances_value = ray.get(train_instances_value)
        
        assert isinstance(train_instances_value, (list, tuple)), "train_instances must be a list or tuple"

        train_instances_value = [ray.get(inst) if isinstance(inst, ray.ObjectRef) else inst for inst in train_instances_value]

        preload_time_end = time.time()

        if potential_kwargs is None:
            potential_kwargs = {}
        if search_evaluator_kwargs is None:
            # search_evaluator_kwargs = {}
            search_evaluator_kwargs = dict(
                seed=np.random.randint(0, 1000000),
            )

        for key, value in shared_readonly_ray_resources.items():
            if not key in ["train_instances", "np_paths"]:
                search_evaluator_kwargs[key] = value

        assert isinstance(search_evaluator_cls, str), "search_evaluator_cls must be a string"
        assert hasattr(
            search_evaluator_module, search_evaluator_cls
        ), f"search_evaluator_cls {search_evaluator_cls} not found in {search_evaluator_module.__name__}"

        assert isinstance(potential_cls, str), "potential_cls must be a string"
        assert hasattr(potential_module, potential_cls), \
            f"potential_cls {potential_cls} not found in {potential_module.__name__}"
        potential_cls = getattr(potential_module, potential_cls)

        data_init_time_end = time.time()
    except KeyboardInterrupt:
        raise TimeoutError("The initialization of the train instances and context is timed out. Consider increasing the timeout...")

    try:

        evaluator = getattr(search_evaluator_module, search_evaluator_cls)(
            instances=train_instances_value,
            potential_cls=potential_cls,
            potential_kwargs=potential_kwargs,
            timeout=timeout - (time.time() - start_time) - 1 if timeout is not None else None,
            **search_evaluator_kwargs,
        )

        evaluator_init_time_end = time.time()

    except KeyboardInterrupt:
        raise TimeoutError("The constructor of the evaluator took too long... Consider increasing the timeout or optimize the initialization of the evaluator.")

    try:
        result = evaluator()
        eval_time_end = time.time()
        return result, dict(
            preload_time=preload_time_end - start_time,
            data_init_time=data_init_time_end - preload_time_end,
            evaluator_init_time=evaluator_init_time_end - data_init_time_end,
            eval_time=eval_time_end - evaluator_init_time_end,
            total_time=eval_time_end - start_time,
        )
    except KeyboardInterrupt:
        raise ValueError("The evaluator did not handle graceful cancellation (KeyboardInterrupt)")
    

@ray.remote(max_calls=1)
def evaluate_search(
    potential_family_path: str,
    potential_path: str,
    search_evaluator_path: str,
    max_concurrent: int,
    shared_readonly_ray_resources: Optional[Dict[Any, Any]] = None,
    timeout: float = 10,
    max_attempts: Optional[int] = None,
    max_queue_size: int = 100,
    memory_limit: float = 10,
    potential_family_kwargs: Optional[Dict[str, Any]] = None,
    min_worker_timeout: float = 5,
    max_worker_timeout: float = 10,
    worker_timeout_buffer: float = 1,
    worker_kwargs: Optional[Dict[str, Any]] = None,
    log_every: Optional[float] = None,
):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if memory_limit < 0:
        logger.warning(f"Memory limit is negative, setting to unlimited")
        memory_limit_bytes = 0
    else:
        memory_limit_bytes = int(memory_limit * 1024 ** 3)
        logger.info(f"Setting memory limit for ask-tell process to {memory_limit} gbs")
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, hard))

    assert max_worker_timeout >= min_worker_timeout, "max_worker_timeout must be greater than or equal to min_worker_timeout"

    if worker_kwargs is None:
        worker_kwargs = {}
    else:
        worker_kwargs = worker_kwargs.copy()

    potential_family_module = load_program(potential_family_path)
    potential_family = potential_family_module.PotentialFamily(**potential_family_kwargs)

    inflight_tasks: Dict[Any, Dict[str, Any]] = {}
    cancelled_tasks = set()
    
    start_time = time.time()
    last_log_time = time.time()
    n_asked = 0
    n_told = 0
    
    queue = deque()

    if not shared_readonly_ray_resources:
        raise ValueError("shared_readonly_ray_resources must be provided")
    worker_kwargs["shared_readonly_ray_resources"] = shared_readonly_ray_resources

    logger.info("Starting search")

    try:
        while True:
            pool_closed = False
            if timeout is not None and time.time() - start_time > timeout:
                pool_closed = True
            
            if inflight_tasks:
                # Cancel tasks over worker_timeout
                now = time.time()
                for future, meta in list(inflight_tasks.items()):
                    if (pool_closed or (now - meta["start_time"] > meta["timeout"])) and future not in cancelled_tasks:
                        ray.cancel(future)
                        cancelled_tasks.add(future)

                ready_futures, _ = ray.wait(
                    list(inflight_tasks.keys()),
                    timeout=worker_timeout_buffer,
                )

                worker_results = []
                for future in ready_futures:
                    try:
                        results, profiling_info = ray.get(future)
                    except TimeoutError:
                        logger.warning("Task timed out before starting evaluator, resubmitting with timeout_ratio 1.0...")
                        submission = inflight_tasks.pop(future)
                        queue.appendleft({
                            "worker_input": submission["worker_input"],
                            "metadata": submission["metadata"],
                            "timeout": 1.0,
                            "cancelled_expected": submission["cancelled_expected"] + 1,
                            "cancelled_unexpected": submission["cancelled_unexpected"],
                        })
                        continue
                    except ray.exceptions.TaskCancelledError:
                        logger.warning("Task was cancelled while it was pending, resubmitting with timeout_ratio 1.0...")
                        submission = inflight_tasks.pop(future)
                        queue.appendleft({
                            "worker_input": submission["worker_input"],
                            "metadata": submission["metadata"],
                            "timeout": 1.0,
                            "cancelled_expected": submission["cancelled_expected"],
                            "cancelled_unexpected": submission["cancelled_unexpected"] + 1,
                        })
                        continue

                    logger.debug(f"Got results from {future}")
                    logger.debug(f"Profiling info: {profiling_info}")

                    submission = inflight_tasks.pop(future)
                    worker_results.append(dict(
                        worker_input = submission["worker_input"],
                        metadata = submission["metadata"],
                        worker_output = results,
                        timeout = submission["timeout"],
                    ))
            else:
                worker_results = []

            for worker_result in worker_results:
                result = worker_result.pop("worker_output")
                potential_family.tell(worker_result, result)
                n_told += 1

            if not pool_closed:
                while len(inflight_tasks) + len(queue) < max_concurrent:
                    worker_inputs = potential_family.ask()
                    n_asked += len(worker_inputs)
                    if not worker_inputs:
                        break
                    if len(queue) + len(worker_inputs) > max_queue_size:
                        logger.warning(f"Queue size {len(queue) + len(worker_inputs)} exceeds max_queue_size {max_queue_size}, skipping {len(worker_inputs)} worker inputs")
                        break
                    queue.extend(worker_inputs[:max_queue_size - len(queue)])

                while len(inflight_tasks) < max_concurrent:
                    if not queue:
                        break

                    submission = queue.popleft()
                    worker_input = submission["worker_input"]
                    timeout_ratio = min(1.0, submission.get("timeout", 1.0))
                    timeout_ratio = max(0.01, timeout_ratio)
                    submission_timeout = min_worker_timeout + timeout_ratio * (max_worker_timeout - min_worker_timeout)
                    metadata = submission.get("metadata", {})

                    cancelled_expected = submission.get("cancelled_expected", 0)
                    cancelled_unexpected = submission.get("cancelled_unexpected", 0)

                    if cancelled_expected >= 5:
                        logger.error("Task was cancelled expectedly too many times, skipping...")
                        continue

                    if cancelled_unexpected >= 5:
                        logger.error("Task was cancelled unexpectedly too many times, skipping...")
                        continue

                    # Concurrency control is handled by how many tasks *you* submit
                    handle = submit_fn_remote.options(
                        # e.g. 1 CPU per eval
                        num_cpus=1,
                    ).remote(
                        potential_family_path=potential_family_path,
                        potential_path=potential_path,
                        search_evaluator_path=search_evaluator_path,
                        timeout=submission_timeout,
                        **worker_kwargs,
                        **worker_input,
                    )
                    inflight_tasks[handle] = {
                        "worker_input": worker_input,
                        "timeout": submission_timeout,
                        "metadata": metadata,
                        "start_time": time.time(),
                        "cancelled_expected": cancelled_expected,
                        "cancelled_unexpected": cancelled_unexpected,
                    }

            if not inflight_tasks:
                break

            if log_every is not None and time.time() - last_log_time > log_every:
                time_elapsed = time.time() - start_time
                logger.info(f"In flight tasks: {len(inflight_tasks)}")
                logger.info(f"Asked: {n_asked}, Told: {n_told}")
                logger.info(f"Time: {time_elapsed:.2f}s, Done: {time_elapsed/timeout * 100:.2f}%")
                last_log_time = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        logger.info(f"Cancelling {len(inflight_tasks)} inflight tasks")
        for future in inflight_tasks.keys():
            ray.cancel(future, force=True)


    recommendations, summary = potential_family.finalize()

    if not isinstance(summary, dict):
        summary = {}

    summary.setdefault("n_asked", n_asked)
    summary.setdefault("n_told", n_told)

    return summary, recommendations


@ray.remote(max_calls=1)
def _final_evaluation_one_instance(
    potential_path: str,
    potential_kwargs: Dict[str, Any],
    instance_path: PathLike,
    potential_cls: str = "Potential",
    **kwargs,
):

    module = load_program(potential_path)
    kserver_instance = NumpyKServerInstance.load(instance_path)
    context = kserver_instance.get_context()
    assert isinstance(potential_cls, str), "potential_cls must be a string"
    assert hasattr(module, potential_cls), (
        f"potential_cls {potential_cls} not found in {module.__name__}"
    )
    potential_cls = getattr(module, potential_cls)
    potential_factory = partial(potential_cls, **potential_kwargs)

    stats = compute_potential_stats(
        potential_factory,
        kserver_instance,
        **kwargs,
    )

    # to avoid keeping kserver_instance on IDLE worker
    del kserver_instance, module

    return stats

@ray.remote(max_calls=1)
def final_evaluation(
    potential_path: str,
    potential_kwargs: Dict[str, Any],
    instance_paths: List[PathLike],
    potential_cls: str = "Potential",
    **kwargs,
):

    all_handles = []
    for instance in instance_paths:
        all_handles.append(_final_evaluation_one_instance.remote(
            potential_path=potential_path,
            potential_kwargs=potential_kwargs,
            instance_path=instance,
            potential_cls=potential_cls,
            **kwargs,
        ))

    return ray.get(all_handles)


def evaluate_solution_family(
    potential_family_path: str,
    potential_path: str,
    search_evaluator_path: str,
    search_timeout: float = 10,
    search_timeout_buffer: float = 1,
    search_kwargs: Optional[Dict[str, Any]] = None,
    shared_readonly_ray_resources_kwargs: Optional[Dict[str, Any]] = None,
    skip_final_evaluation: bool = False,
    final_evaluation_timeout: float = 200,
    final_evaluation_timeout_buffer: float = 10,
    final_evaluation_kwargs: Optional[Dict[str, Any]] = None,
    top_k_final_evaluation: Optional[int] = 1,
):
    if search_kwargs is None:
        search_kwargs = {}
    if final_evaluation_kwargs is None:
        final_evaluation_kwargs = {}

    shared_readonly_ray_resources = None
    if shared_readonly_ray_resources_kwargs is not None:
        shared_readonly_ray_resources = load_shared_readonly_ray_resources(
            **shared_readonly_ray_resources_kwargs
        )
        if not shared_readonly_ray_resources:
            shared_readonly_ray_resources = None

    search_handle = evaluate_search.remote(
        potential_family_path=potential_family_path,
        potential_path=potential_path,
        search_evaluator_path=search_evaluator_path,
        shared_readonly_ray_resources=shared_readonly_ray_resources,
        timeout=search_timeout,
        **search_kwargs,
    )

    try:
        summary, search_results = ray.get(search_handle, timeout=search_timeout + search_timeout_buffer)
    except ray.exceptions.GetTimeoutError:
        logger.info(f"Search timed out, cancelling search handle")
        ray.cancel(search_handle)
        try:
            summary, search_results = ray.get(search_handle, timeout=search_timeout_buffer)
        except ray.exceptions.GetTimeoutError:
            raise TimeoutError(
                "Search timed out and search process did not exit. "
                "Most likely, the PotentialFamily step method takes too long to complete."
            )

    logger.info(f"Got {len(search_results)} search results in total")
    if len(search_results) == 0:
        raise ValueError("No scored attempts found! Please check the way the return of scored attempts is implemented.")
    
    sorted_idxes = np.argsort([r.get("score", 0) for r in search_results])[::-1]
    logger.info(f"Best attempt with score {search_results[sorted_idxes[0]].get('score', 0)}")

    if skip_final_evaluation:
        return summary, search_results, []

    if top_k_final_evaluation is not None:
        sorted_idxes = sorted_idxes[:top_k_final_evaluation]

    final_evaluation_potential_kwargs = []
    final_evaluation_handles = []
    for idx in sorted_idxes:
        potential_kwargs = search_results[idx].get("kwargs", {})
        potential_cls = search_results[idx].get("potential_cls", "Potential")
        final_evaluation_handles.append(final_evaluation.remote(
            potential_path=potential_path,
            potential_kwargs=potential_kwargs,
            potential_cls=potential_cls,
            **final_evaluation_kwargs,
        ))
        final_evaluation_potential_kwargs.append(
            {
                "potential_cls": potential_cls,
                "kwargs": potential_kwargs,
            }
        )

    try:
        logger.info(f"Waiting for {len(final_evaluation_handles)} final evaluation handles")
        final_evaluation_results = ray.get(final_evaluation_handles, timeout=final_evaluation_timeout + final_evaluation_timeout_buffer)
    except ray.exceptions.GetTimeoutError:
        raise TimeoutError("Final evaluation timed out")

    return summary, search_results, list(zip(final_evaluation_results, final_evaluation_potential_kwargs))




def env_default(name: str, fallback: str) -> str:
    
    env_name = f"K_SERVER_EVALUATE_{name}"
    value = os.getenv(env_name)
    return fallback if value is None or value == "" else value


def env_default_bool(name: str, fallback: bool = False) -> bool:
    value = env_default(name, "1" if fallback else "0")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description="k-server evaluator")
    
    parser.add_argument(
        "--program_path",
        type=str,
        default=env_default("PROGRAM_PATH", "initial.py"),
        help="Path to program to evaluate",
    )

    parser.add_argument(
        "--home",
        type=str,
        default=env_default("HOME", str(LEGACY_EVALUATOR_HOME)),
        help="Path to directory where default implementation and metrics are stored",
    )

    parser.add_argument(
        "--ray_address",
        type=str,
        default=env_default("RAY_ADDRESS", "auto"),
        help="Ray address passed to ray.init (default: 'auto')",
    )
    parser.add_argument(
        "--ray_port",
        type=int,
        default=int(env_default("RAY_PORT", "0")),
        help="If set to a positive value, overrides ray_address with localhost:<ray_port>",
    )
    parser.add_argument(
        "--ray_probe_timeout",
        type=float,
        default=float(env_default("RAY_CONNECT_TIMEOUT_S", "30")),
        help="Timeout for the subprocess Ray probe before restart",
    )
    parser.add_argument(
        "--ray_restart_path",
        type=str,
        default=env_default("RAY_RESTART_SCRIPT", str(LEGACY_EVALUATOR_HOME / "restart_ray.sh")),
        help="Path to ray_restart.sh / restart_ray.sh used when the probe times out",
    )

    parser.add_argument(
        "--potential_family_kwargs_path",
        type=str,
        default=env_default("POTENTIAL_FAMILY_KWARGS_PATH", None),
        help="Path to potential family kwargs file (json)",
    )

    parser.add_argument(
        "--shared_readonly_ray_resources_kwargs_path",
        type=str,
        default=env_default("SHARED_READONLY_RAY_RESOURCES_KWARGS_PATH", None),
        help="Path to shared READONLY ray resources kwargs file (json). Currently supports paths to npz files.",
    )

    parser.add_argument(
        "--use_default_potential_family",
        action="store_true",
        help="Use default implementation of PotentialFamily",
        default=bool(env_default("USE_DEFAULT_POTENTIAL_FAMILY", False)),
    )

    parser.add_argument(
        "--use_default_search_evaluator",
        action="store_true",
        help="Use default implementation of SearchEvaluator",
        default=bool(env_default("USE_DEFAULT_SEARCH_EVALUATOR", False)),
    )

    parser.add_argument(
        "--use_default_potential",
        action="store_true",
        help="Use default implementation of Potential",
        default=bool(env_default("USE_DEFAULT_POTENTIAL", False)),
    )

    parser.add_argument(
        "--potential_family_path",
        type=str,
        default=env_default("POTENTIAL_FAMILY_PATH", None),
        help="Path to potential family module",
    )

    parser.add_argument(
        "--potential_path",
        type=str,
        default=env_default("POTENTIAL_PATH", None),
        help="Path to potential module",
    )

    parser.add_argument(
        "--search_evaluator_path",
        type=str,
        default=env_default("SEARCH_EVALUATOR_PATH", None),
        help="Path to search evaluator module",
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default=env_default("RESULTS_DIR", None),
        help="Path to save results (metrics.json, correct.json)",
    )

    parser.add_argument(
        "--log_filename",
        type=str,
        default=env_default("LOG_FILENAME", "log"),
        help="Filename to save log",
    )

    parser.add_argument(
        "--metrics_path",
        type=str,
        default=env_default("METRICS_PATH", str(DEFAULT_METRICS_DIR)),
        help="Path to metrics directory",
    )

    metrics_names_default = env_default(
        "METRICS_NAMES",
        "circle_taxi_k4_m6.pickle",
    )
    metrics_names_default_list = [x for x in metrics_names_default.split(",") if x]
    parser.add_argument(
        "--metrics_names",
        type=lambda s: [x for x in s.split(",") if x],
        default=metrics_names_default_list,
        help="Comma-separated list of metric filenames to load",
    )

    ask_tell_max_queue_size_default = env_default("ASK_TELL_MAX_QUEUE_SIZE", 100)
    parser.add_argument(
        "--ask_tell_max_queue_size",
        type=int,
        default=ask_tell_max_queue_size_default,
        help="Maximum size of ask-tell queue",
    )

    ask_tell_memory_limit_default = env_default("ASK_TELL_MEMORY_LIMIT", -1)
    parser.add_argument(
        "--ask_tell_memory_limit",
        type=float,
        default=ask_tell_memory_limit_default,
        help="Memory limit for ask-tell process (gbs). Negative for unlimited.",
    )

    search_process_memory_limit_default = env_default("SEARCH_PROCESS_MEMORY_LIMIT", -1)
    parser.add_argument(
        "--search_process_memory_limit",
        type=float,
        default=search_process_memory_limit_default,
        help="Memory limit for search process (gbs). Negative for unlimited.",
    )

    search_timeout_default = env_default("SEARCH_TIMEOUT", "200")
    search_timeout_default = None if str(search_timeout_default).lower() == "none" else float(search_timeout_default)
    parser.add_argument(
        "--search_timeout",
        type=lambda x: None if x.lower() == "none" else float(x),
        default=search_timeout_default,
        help="Total timeout for search (seconds, 'none' to disable)",
    )

    parser.add_argument(
        "--search_timeout_buffer",
        type=float,
        default=float(env_default("SEARCH_TIMEOUT_BUFFER", "10")),
        help="Buffer before stopping search (seconds)",
    )

    parser.add_argument(
        "--search_max_concurrent",
        type=int,
        default=int(env_default("SEARCH_MAX_CONCURRENT", "10")),
        help="Maximum number of concurrent search tasks",
    )

    parser.add_argument(
        "--search_max_worker_timeout",
        type=float,
        default=float(env_default("SEARCH_MAX_WORKER_TIMEOUT", "15")),
        help="Maximum timeout per search worker (seconds)",
    )

    parser.add_argument(
        "--search_min_worker_timeout",
        type=float,
        default=float(env_default("SEARCH_MIN_WORKER_TIMEOUT", "5")),
        help="Minimum timeout per search worker (seconds)",
    )

    parser.add_argument(
        "--skip_final_evaluation",
        action="store_true",
        help="Skip final evaluation",
        default=bool(),
    )

    parser.add_argument(
        "--final_evaluation_timeout",
        type=float,
        default=float(env_default("FINAL_EVALUATION_TIMEOUT", "200")),
        help="Timeout per final evaluation (seconds)",
    )

    parser.add_argument(
        "--final_evaluation_timeout_buffer",
        type=float,
        default=float(env_default("FINAL_EVALUATION_TIMEOUT_BUFFER", "10")),
        help="Buffer before stopping final evaluation (seconds)",
    )
    
    parser.add_argument(
        "--final_evaluation_max_concurrent",
        type=int,
        default=int(env_default("FINAL_EVALUATION_MAX_CONCURRENT", "5")),
        help="Maximum number of concurrent final evaluation tasks",
    )

    parser.add_argument(
        "--final_evaluation_top_k",
        type=int,
        default=int(env_default("FINAL_EVALUATION_TOP_K", "1")),
        help="Top K final evaluation results to return",
    )

    max_best_kwargs_bytes_default = env_default("MAX_BEST_KWARGS_BYTES", "51200")
    max_best_kwargs_bytes_default = None if str(max_best_kwargs_bytes_default).lower() == "none" else int(max_best_kwargs_bytes_default)
    parser.add_argument(
        "--max_best_kwargs_bytes",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=max_best_kwargs_bytes_default,
        help="Hard threshold on serialized kwargs size (bytes). Candidates above threshold get score 0. Use 'none' to disable.",
    )
    
    compute_round_default = env_default("COMPUTE_STATS_ROUND_DIGITS", "4")
    compute_round_default = None if str(compute_round_default).lower() == "none" else int(compute_round_default)
    parser.add_argument(
        "--compute_stats_round_digits",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=compute_round_default,
        help="Round floats inside compute_potential_stats metrics (None to skip rounding)",
    )

    default_rho = env_default("RHO", None)
    try:
        default_rho = float(default_rho)
    except Exception:
        default_rho = None

    parser.add_argument(
        "--rho",
        type=float,
        default=default_rho,
        help="rho parameter for compute_potential_stats",
    )

    parser.add_argument(
        "--keep_only_violations_k",
        action="store_true",
        default=env_default_bool("KEEP_ONLY_VIOLATIONS_K", False),
        help="Keep only violations_k metrics in compute_potential_stats output",
    )
    parser.add_argument(
        "--robustness_check",
        action="store_true",
        default=env_default_bool("ROBUSTNESS_CHECK", False),
        help="Run robustness checks in compute_potential_stats (default: false)",
    )
    
    return parser.parse_args(*args, **kwargs)


DEFAULT_METRICS_ON_ERROR = {
    "combined_score": 0.0,
    "execution_time_mean": 0.0,
    "execution_time_std": 0.0,
    "num_successful_runs": 0,
    "num_valid_runs": 0,
    "num_invalid_runs": 0,
    "all_validation_errors": [],
}



def save_json_results(
    results_dir: str,
    metrics: Dict[str, Any],
    correct: bool,
    error: Optional[str] = None,
) -> None:
    
    

    os.makedirs(results_dir, exist_ok=True)

    def normalize(obj: Any):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.dtype):
            return str(obj)
        if isinstance(obj, type):
            return obj.__name__
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(normalize(v) for v in obj)
        if isinstance(obj, Number):
            if obj == float("inf") or obj == float("-inf"):
                return None
            return obj
        return str(obj)

    correct_payload = {"correct": correct, "error": error}
    correct_file = os.path.join(results_dir, "correct.json")
    with open(correct_file, "w") as f:
        json.dump(correct_payload, f, indent=4)
    logger.info("Correctness and error status saved to %s", correct_file)

    metrics = normalize(metrics)
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved to %s", metrics_file)


def aggregate_metrics(
    summary: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    final_evaluation_results: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]],
    skip_final_evaluation: bool = False,
    top_k_attempts: int = 5,
    max_best_kwargs_bytes: Optional[int] = 50 * 1024,
) -> Dict[str, Any]:
    def _serialized_size_bytes(value: Any) -> int:
        try:
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            payload = str(value)
        return len(payload.encode("utf-8"))

    global_public_metrics = {}
    global_private_metrics = {}

    sorted_idxes = np.argsort([r.get("score", 0) for r in search_results])[::-1]
    sorted_idxes = sorted_idxes[:top_k_attempts]

    best_attempts = []
    best_attempts_extended = []
    for idx in sorted_idxes:
        search_result = search_results[idx]
        score = search_result.get("score", 0)
        kwargs = search_result.get("kwargs", {})
        potential_cls = search_result.get("potential_cls", "Potential")

        best_attempts.append({
            "score": round(score, 4),
            "kwargs": str(kwargs),
            "potential_cls": potential_cls,
        })

        best_attempts_extended.append(search_result)

    if skip_final_evaluation:
        best_score = float(max([r.get("score", 0) for r in search_results]))
        return {
            "combined_score": best_score,
            "public": {
                **global_public_metrics,
                "best_attempts_str": str(best_attempts),
                "summary_str": str(summary),
                "total_attempts": len(search_results),
            },
            "private": {
                "unique_key": None,
                "best_attempts_extended": best_attempts_extended,
                "summary": summary,
            },
        }

    best_fe_score = float("-inf")
    best_fe = None
    best_fe_kwargs = None
    best_fe_potential_cls = None
    best_fe_kwargs_size_bytes = None
    best_fe_kwargs_disqualified = False
    best_fe_has_robustness_failure = False
    for fe_results, fe_submission in final_evaluation_results:
        if isinstance(fe_submission, dict):
            fe_kwargs = fe_submission.get("kwargs", fe_submission)
            fe_potential_cls = fe_submission.get("potential_cls", "Potential")
        else:
            fe_kwargs = fe_submission
            fe_potential_cls = "Potential"

        fe_kwargs_size_bytes = _serialized_size_bytes(fe_kwargs)
        kwargs_too_large = (
            max_best_kwargs_bytes is not None
            and max_best_kwargs_bytes >= 0
            and fe_kwargs_size_bytes > max_best_kwargs_bytes
        )

        if kwargs_too_large:
            score = 0.0
        else:
            score = 1.0
            for instance_idx, stats in enumerate(fe_results):
                metrics = stats.metrics
                # score = -max(metrics["violations_k"], score)
                score *= metrics.get("violations_k_score", 0) *\
                    metrics.get("processed_normalized_edges_score", 1.0)

        if score > best_fe_score:
            best_fe_score = score
            best_fe = fe_results
            best_fe_kwargs = fe_kwargs
            best_fe_potential_cls = fe_potential_cls
            best_fe_kwargs_size_bytes = fe_kwargs_size_bytes
            best_fe_kwargs_disqualified = kwargs_too_large

    if best_fe is None and not skip_final_evaluation:
        raise ValueError("No final evaluation results found")

    unique_keys = []
    for instance_idx, stats in enumerate(best_fe):
        metrics = stats.metrics
        private_only_keys = {
            "robustness_details",
        }
        if metrics.get("robustness", True) is False:
            best_fe_has_robustness_failure = True
        global_public_metrics.update({
            f"{instance_idx}/{key}": value for key, value in metrics.items() if not key.endswith("_time")
            if key not in private_only_keys
        })
        global_private_metrics.update({
            f"{instance_idx}/{key}": value for key, value in metrics.items() if (key.endswith("_time") or key in private_only_keys)
        })
        unique_keys.append(stats.unique_key)
    
    global_public_metrics["best_kwargs"] = str(best_fe_kwargs)
    global_public_metrics["best_potential_cls"] = str(best_fe_potential_cls)
    global_public_metrics["best_kwargs_size_bytes"] = int(best_fe_kwargs_size_bytes) if best_fe_kwargs_size_bytes is not None else None
    global_public_metrics["best_kwargs_threshold_bytes"] = int(max_best_kwargs_bytes) if max_best_kwargs_bytes is not None else None
    global_public_metrics["best_kwargs_disqualified"] = bool(best_fe_kwargs_disqualified)
    global_private_metrics["best_kwargs"] = best_fe_kwargs
    global_private_metrics["best_potential_cls"] = best_fe_potential_cls
    global_private_metrics["best_kwargs_size_bytes"] = int(best_fe_kwargs_size_bytes) if best_fe_kwargs_size_bytes is not None else None
    global_private_metrics["best_kwargs_threshold_bytes"] = int(max_best_kwargs_bytes) if max_best_kwargs_bytes is not None else None
    global_private_metrics["best_kwargs_disqualified"] = bool(best_fe_kwargs_disqualified)

    unique_key = "_".join(unique_keys)

    if best_fe_has_robustness_failure:
        best_fe_score = 0.0

    notes = []
    if best_fe_kwargs_disqualified:
        notes.append("You kwargs are too large, probably you are using forbidden techniques like lookup")
    if best_fe_has_robustness_failure:
        notes.append("Robustness check did not pass, probably you are using forbidden techniques like lookup")
    final_note = " ".join(notes) if notes else None

    return {
        "combined_score": best_fe_score,
        "public": {
            **global_public_metrics,
            "best_attempts_str": str(best_attempts),
            "summary_str": str(summary),
            "total_attempts": len(search_results),
            "note": final_note,
        },
        "private": {
            "unique_key": unique_key,
            **global_private_metrics,
            "best_attempts_extended": best_attempts_extended,
            "summary": summary,
        }
    }


def main(args):
    effective_ray_address = args.ray_address
    if args.ray_port and args.ray_port > 0:
        effective_ray_address = f"localhost:{args.ray_port}"

    connect_or_restart_ray(
        ray_addr=effective_ray_address,
        probe_timeout_s=args.ray_probe_timeout,
        restart_script=args.ray_restart_path or None,
    )

    args.program_path = os.path.abspath(args.program_path)

    if args.results_dir is None:
        args.results_dir = os.path.join(os.path.dirname(args.program_path), f"results-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    
    if args.home:
        args.home = os.path.abspath(args.home)

    if args.home and not args.metrics_path:
        args.metrics_path = os.path.join(
            os.path.dirname(os.path.dirname(args.home)),
            "metrics",
        )

    if args.home and args.use_default_potential_family and not args.potential_family_path:
        args.potential_family_path = os.path.join(args.home, "naive_potential_family.py")

    if args.home and args.use_default_search_evaluator and not args.search_evaluator_path:
        args.search_evaluator_path = os.path.join(args.home, "naive_search_evaluator.py")

    if args.home and args.use_default_potential and not args.potential_path:
        args.potential_path = os.path.join(args.home, "canonical_potential.py")

    if args.potential_family_path is not None:
        args.potential_family_path = os.path.abspath(args.potential_family_path)
    else:
        args.potential_family_path = args.program_path
    if args.potential_path is not None:
        args.potential_path = os.path.abspath(args.potential_path)
    else:
        args.potential_path = args.program_path
    if args.search_evaluator_path is not None:
        args.search_evaluator_path = os.path.abspath(args.search_evaluator_path)
    else:
        args.search_evaluator_path = args.program_path

    if args.shared_readonly_ray_resources_kwargs_path:
        args.shared_readonly_ray_resources_kwargs_path = os.path.abspath(args.shared_readonly_ray_resources_kwargs_path)
    
    if args.potential_family_kwargs_path:
        args.potential_family_kwargs_path = os.path.abspath(args.potential_family_kwargs_path)

    args.results_dir = os.path.abspath(args.results_dir)
    args.metrics_path = os.path.abspath(args.metrics_path)

    os.makedirs(args.results_dir, exist_ok=True)

    logger.info(str(args))

    logger.info("HOME: %s", args.home)

    logger.info("Evaluating program: %s", args.program_path)
    logger.info("Saving results to: %s", args.results_dir)
    logger.info("Using metrics from: %s", args.metrics_path)
    logger.info("Using potential family from: %s", args.potential_family_path)
    logger.info("Using search evaluator from: %s", args.search_evaluator_path)
    logger.info("Using potential from: %s", args.potential_path)

    if args.shared_readonly_ray_resources_kwargs_path:
        logger.info("Using shared readonly ray resources kwargs from: %s", args.shared_readonly_ray_resources_kwargs_path)
    if args.potential_family_kwargs_path:
        logger.info("Using potential family kwargs from: %s", args.potential_family_kwargs_path)


    train_instances = [
        os.path.join(args.metrics_path, f) for f in args.metrics_names
    ]

    start_time = time.time()

    potential_family_kwargs = {}
    if args.potential_family_kwargs_path:
        with open(args.potential_family_kwargs_path, "r") as f:
            potential_family_kwargs = json.load(f)

    if potential_family_kwargs:
        logger.info(f"Using potential family kwargs: {potential_family_kwargs}")

    shared_readonly_ray_resources_kwargs = {}
    if args.shared_readonly_ray_resources_kwargs_path:
        with open(args.shared_readonly_ray_resources_kwargs_path, "r") as f:
            shared_readonly_ray_resources_kwargs = json.load(f)

    if shared_readonly_ray_resources_kwargs:
        logger.info(f"Using shared readonly ray resources kwargs: {shared_readonly_ray_resources_kwargs}")

    try:

        potential_family_kwargs.update(dict(
            n_instances=len(train_instances),
            n_workers=args.search_max_concurrent,
            search_timeout=args.search_timeout,
            min_worker_timeout=args.search_min_worker_timeout,
            max_worker_timeout=args.search_max_worker_timeout,
        ))

        shared_readonly_ray_resources_kwargs.update(dict(
            train_instances=train_instances,
        ))


        summary, search_results, final_evaluation_results = evaluate_solution_family(
            potential_family_path=args.potential_family_path,
            potential_path=args.potential_path,
            search_evaluator_path=args.search_evaluator_path,
            search_timeout=args.search_timeout,
            search_timeout_buffer=args.search_timeout_buffer,
            
            shared_readonly_ray_resources_kwargs=shared_readonly_ray_resources_kwargs,
            
            search_kwargs=dict(
                max_concurrent=args.search_max_concurrent,
                max_attempts=None,
                max_worker_timeout=args.search_max_worker_timeout,
                min_worker_timeout=args.search_min_worker_timeout,
                max_queue_size=args.ask_tell_max_queue_size,
                memory_limit=args.ask_tell_memory_limit,
                potential_family_kwargs=potential_family_kwargs,
                worker_kwargs=dict(
                    memory_limit=args.search_process_memory_limit,
                ),
                log_every=20,
            ),

            skip_final_evaluation=args.skip_final_evaluation,

            final_evaluation_timeout=args.final_evaluation_timeout,
            final_evaluation_timeout_buffer=args.final_evaluation_timeout_buffer,

            final_evaluation_kwargs=dict(
                instance_paths=train_instances,
                compute_potential_backend="mp",
                include_wf_columns=False,
                include_renormalized_metrics=False,
                include_info_columns=False,
                compute_potential_kwargs=dict(
                    timeout = args.final_evaluation_timeout,
                    n_processes = args.final_evaluation_max_concurrent,
                ),
                round_digits=args.compute_stats_round_digits,
                rho=args.rho,
                keep_only_violations_k=args.keep_only_violations_k,
                robustness_check=args.robustness_check,
            ),
            top_k_final_evaluation=args.final_evaluation_top_k,
        )
        correct = True
        error_msg = None

        metrics = aggregate_metrics(
            summary,
            search_results,
            final_evaluation_results,
            skip_final_evaluation=args.skip_final_evaluation,
            max_best_kwargs_bytes=args.max_best_kwargs_bytes,
            # top_k_attempts=args.final_evaluation_top_k,
        )

        num_valid_runs = 1
        num_invalid_runs = 0
        all_validation_errors = []

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc(limit=10)
        logger.error("Evaluation error")
        logger.exception(e)
        # logger.exception(error_msg[:1000])
        metrics = {k: v for k, v in DEFAULT_METRICS_ON_ERROR.items()}
        correct = False

        num_valid_runs = 0
        num_invalid_runs = 1
        all_validation_errors = [error_msg]
    
    finally:
        ray.shutdown()

    end_time = time.time()

    metrics["execution_time_mean"] = float(end_time - start_time)
    metrics["execution_time_std"] = 0.0
    metrics["num_valid_runs"] = num_valid_runs
    metrics["num_invalid_runs"] = num_invalid_runs
    metrics["all_validation_errors"] = all_validation_errors

    save_json_results(
        args.results_dir,
        metrics,
        correct,
        error_msg,
    )



if __name__ == "__main__":
    args = parse_args()
    main(args)
