from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional
from time import perf_counter
import gc
import numpy as np


def _make_wandb_logger(enabled: bool):
    if not enabled:
        return None
    # try:
    import wandb  # type: ignore
    # except Exception:
    #     return None
    # if getattr(wandb, "run", None) is None:
    #     return None

    def _log(data: Dict[str, Any], step: Optional[int] = None) -> None:
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)

    return _log


def create_progress_bar_hooks(
    desc: str = "WF Graph BFS",
    unit: str = "expand",
    tqdm_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, List]:
    """
    Return main hook constructors that render a live progress bar.

    Postfix fields:
    - nodes: currently explored/known unique nodes
    - edges: currently explored/known edges
    """
    try:
        from tqdm.auto import tqdm
    except Exception:
        def _noop(_state):
            def _hook(**kwargs):
                return None
            return _hook

        return {
            "main.start": [_noop],
            "main.handle_expand.end": [_noop],
            "main.end": [_noop],
        }

    bar_key = "_progress_bar"
    extra = dict(tqdm_kwargs or {})

    update_every = 1000
    current_updates = 0

    def on_start(state):
        def hook(**kwargs):
            if state.get(bar_key) is None:
                state[bar_key] = tqdm(total=None, desc=desc, unit=unit, **extra)
        return hook

    def on_expand(state):
        def hook(num_nodes=None, num_edges=None, **kwargs):
            nonlocal current_updates
            pbar = state.get(bar_key)
            current_updates += 1
            if current_updates >= update_every:
                current_updates = 0
                if pbar is None:
                    return
                pbar.update(update_every)
                pbar.set_postfix(nodes=num_nodes, edges=num_edges)
        return hook

    def on_end(state):
        def hook(num_nodes=None, num_edges=None, **kwargs):
            pbar = state.get(bar_key)
            if pbar is None:
                return
            pbar.set_postfix(nodes=num_nodes, edges=num_edges, refresh=True)
            pbar.close()
            state[bar_key] = None
        return hook

    return {
        "main.start": [on_start],
        "main.handle_expand.end": [on_expand],
        "main.end": [on_end],
    }


def create_worker_potential_hooks(
    potential_cls,
    potential_kwargs: Optional[Dict[str, Any]] = None,
    potential_key: str = "potential",
    potential_meta_key: str = "potential_meta",
) -> Dict[str, List]:
    """
    Create worker hooks that instantiate a potential from worker.context and
    compute/store potential for newly created nodes using wf_norm.
    """

    kwargs = dict(potential_kwargs or {})

    def ensure_potential(worker):
        if not hasattr(worker, "_potential_obj"):
            worker._potential_obj = potential_cls(worker.context, **kwargs)
        return worker._potential_obj

    def compute_and_store(worker, node, wf_like):
        potential = ensure_potential(worker)
        wf_norm = node.metadata.get("wf_norm")
        if wf_norm is None:
            wf_arr = np.asarray(wf_like)
            wf_norm = wf_arr - wf_arr.min()
        value, info = potential(wf_norm)
        node.metadata[potential_key] = value
        node.metadata[potential_meta_key] = info

    def on_init_node(worker):
        def hook(node, **kwargs):
            wf = node.get_wf(context=worker.context, transitions=worker.transitions, cache=False)
            compute_and_store(worker, node, wf)
        return hook

    def on_new_expand_node(worker):
        def hook(v, vwf, is_new_node, **kwargs):
            if not is_new_node:
                return
            compute_and_store(worker, v, vwf)
        return hook

    return {
        "init_node.after": [on_init_node],
        "expand.after_transition": [on_new_expand_node],
    }


def create_main_timing_profile_hooks(
    log_every: int = 1000,
    print_fn=print,
    key: str = "_timing_profile",
    wandb_log: bool = False,
    wandb_prefix: str = "timing/main",
    max_sample_history: int = 20000,
) -> Dict[str, List]:
    """
    Create main-loop timing hooks for parallel_bfs_exploration.

    Tracks per-loop times for:
    - submit
    - wait
    - handle_expand/bookkeeping
    - total loop
    """
    if log_every < 1:
        log_every = 1
    if max_sample_history < 1:
        max_sample_history = 1
    wandb_logger = _make_wandb_logger(wandb_log)

    def _ensure(state):
        if key not in state:
            state[key] = {
                "loops": 0,
                "init_s": 0.0,
                "submit_s_total": 0.0,
                "wait_s_total": 0.0,
                "handle_s_total": 0.0,
                "loop_s_total": 0.0,
                "wait_s_max_total": 0.0,
                "wait_samples_total": deque(maxlen=max_sample_history),
                "push_s_total": 0.0,
                "push_s_max_total": 0.0,
                "push_samples_total": deque(maxlen=max_sample_history),
                "queue_get_samples_total": deque(maxlen=max_sample_history),
                "queue_put_samples_total": deque(maxlen=max_sample_history),
                "queue_wait_samples_total": deque(maxlen=max_sample_history),
                "submit_s_win": 0.0,
                "wait_s_win": 0.0,
                "handle_s_win": 0.0,
                "loop_s_win": 0.0,
                "wait_s_max_win": 0.0,
                "wait_samples_win": [],
                "push_s_win": 0.0,
                "push_s_max_win": 0.0,
                "push_samples_win": [],
                "queue_get_samples_win": [],
                "queue_put_samples_win": [],
                "queue_wait_samples_win": [],
                "batch_size_win": 0,
                "push_batch_size_win": 0,
                "win_count": 0,
                "t_init_start": None,
                "t_loop_start": None,
                "t_after_submit": None,
                "t_before_get": None,
                "t_after_get": None,
            }
        return state[key]

    def on_start(state):
        def hook(**kwargs):
            _ensure(state)
        return hook

    def on_init_start(state):
        def hook(**kwargs):
            prof = _ensure(state)
            prof["t_init_start"] = perf_counter()
        return hook

    def on_init_end(state):
        def hook(n_initial=None, n_unique=None, **kwargs):
            prof = _ensure(state)
            t0 = prof.get("t_init_start")
            if t0 is None:
                return
            prof["init_s"] = max(0.0, perf_counter() - t0)
            if n_initial is not None and n_unique is not None:
                print_fn(
                    "[timing/main/init] "
                    f"initial={n_initial} unique={n_unique} "
                    f"elapsed={1000.0 * prof['init_s']:.2f}ms"
                )
        return hook

    def on_after_push(state):
        def hook(push_ms=None, batch_size=None, **kwargs):
            prof = _ensure(state)
            if push_ms is None:
                return
            push_s = max(0.0, float(push_ms) / 1000.0)
            prof["push_s_total"] += push_s
            prof["push_s_max_total"] = max(prof["push_s_max_total"], push_s)
            prof["push_samples_total"].append(push_s)
            prof["push_s_win"] += push_s
            prof["push_s_max_win"] = max(prof["push_s_max_win"], push_s)
            prof["push_samples_win"].append(push_s)
            if batch_size is not None:
                prof["push_batch_size_win"] += int(batch_size)
        return hook

    def on_loop_start(state):
        def hook(**kwargs):
            prof = _ensure(state)
            prof["t_loop_start"] = perf_counter()
            prof["t_after_submit"] = None
            prof["t_before_get"] = None
            prof["t_after_get"] = None
        return hook

    def on_submitted(state):
        def hook(**kwargs):
            prof = _ensure(state)
            prof["t_after_submit"] = perf_counter()
        return hook

    def on_before_get(state):
        def hook(**kwargs):
            prof = _ensure(state)
            prof["t_before_get"] = perf_counter()
        return hook

    def on_after_get(state):
        def hook(batch_size=None, queue_wait_ms=None, queue_get_samples_ms=None, queue_put_samples_ms=None, **kwargs):
            prof = _ensure(state)
            prof["t_after_get"] = perf_counter()
            if batch_size is not None:
                prof["batch_size_win"] += int(batch_size)
            if queue_wait_ms is not None:
                prof["queue_wait_samples_win"].append(float(queue_wait_ms))
                prof["queue_wait_samples_total"].append(float(queue_wait_ms))
            for value in queue_get_samples_ms or []:
                prof["queue_get_samples_win"].append(float(value))
                prof["queue_get_samples_total"].append(float(value))
            for value in queue_put_samples_ms or []:
                prof["queue_put_samples_win"].append(float(value))
                prof["queue_put_samples_total"].append(float(value))
        return hook

    def on_loop_end(state):
        def hook(queue_size=None, **kwargs):
            prof = _ensure(state)
            t0 = prof.get("t_loop_start")
            t1 = prof.get("t_after_submit")
            tb = prof.get("t_before_get")
            t2 = prof.get("t_after_get")
            t3 = perf_counter()
            if t0 is None:
                return

            # If a phase event is missing, attribute 0s to that phase rather than
            # over-attributing time and making multiple phases look slow.
            submit_end = t1 if t1 is not None else t0
            get_start = tb if tb is not None else submit_end
            get_end = t2 if t2 is not None else get_start

            submit_s = max(0.0, submit_end - t0)
            wait_s = max(0.0, get_end - get_start)
            handle_s = max(0.0, t3 - get_end)
            loop_s = max(0.0, t3 - t0)

            prof["loops"] += 1
            prof["submit_s_total"] += submit_s
            prof["wait_s_total"] += wait_s
            prof["handle_s_total"] += handle_s
            prof["loop_s_total"] += loop_s
            prof["wait_s_max_total"] = max(prof["wait_s_max_total"], wait_s)
            prof["wait_samples_total"].append(wait_s)
            prof["submit_s_win"] += submit_s
            prof["wait_s_win"] += wait_s
            prof["handle_s_win"] += handle_s
            prof["loop_s_win"] += loop_s
            prof["wait_s_max_win"] = max(prof["wait_s_max_win"], wait_s)
            prof["wait_samples_win"].append(wait_s)
            prof["win_count"] += 1

            if prof["loops"] % log_every == 0:
                win_count = prof["win_count"]
                avg_submit_ms = 1000.0 * prof["submit_s_win"] / win_count
                avg_wait_ms = 1000.0 * prof["wait_s_win"] / win_count
                avg_handle_ms = 1000.0 * prof["handle_s_win"] / win_count
                avg_batch = prof["batch_size_win"] / win_count
                avg_push_ms = 1000.0 * prof["push_s_win"] / win_count
                avg_push_batch = prof["push_batch_size_win"] / win_count
                max_wait_ms = 1000.0 * prof["wait_s_max_win"]
                max_push_ms = 1000.0 * prof["push_s_max_win"]
                q05_ms, q50_ms, q95_ms = (
                    1000.0 * q for q in np.quantile(prof["wait_samples_win"], [0.05, 0.5, 0.95])
                )
                if prof["push_samples_win"]:
                    pq05_ms, pq50_ms, pq95_ms = (
                        1000.0 * q for q in np.quantile(prof["push_samples_win"], [0.05, 0.5, 0.95])
                    )
                else:
                    pq05_ms = pq50_ms = pq95_ms = 0.0
                if prof["queue_get_samples_win"]:
                    qg_avg_ms = float(np.mean(prof["queue_get_samples_win"]))
                    qg_max_ms = float(np.max(prof["queue_get_samples_win"]))
                    qg05_ms, qg50_ms, qg95_ms = (
                        float(q) for q in np.quantile(prof["queue_get_samples_win"], [0.05, 0.5, 0.95])
                    )
                else:
                    qg_avg_ms = qg_max_ms = qg05_ms = qg50_ms = qg95_ms = 0.0
                if prof["queue_wait_samples_win"]:
                    qw_avg_ms = float(np.mean(prof["queue_wait_samples_win"]))
                    qw_max_ms = float(np.max(prof["queue_wait_samples_win"]))
                    qw05_ms, qw50_ms, qw95_ms = (
                        float(q) for q in np.quantile(prof["queue_wait_samples_win"], [0.05, 0.5, 0.95])
                    )
                else:
                    qw_avg_ms = qw_max_ms = qw05_ms = qw50_ms = qw95_ms = 0.0
                if prof["queue_put_samples_win"]:
                    qp_avg_ms = float(np.mean(prof["queue_put_samples_win"]))
                    qp_max_ms = float(np.max(prof["queue_put_samples_win"]))
                    qp05_ms, qp50_ms, qp95_ms = (
                        float(q) for q in np.quantile(prof["queue_put_samples_win"], [0.05, 0.5, 0.95])
                    )
                else:
                    qp_avg_ms = qp_max_ms = qp05_ms = qp50_ms = qp95_ms = 0.0
                rate = win_count / prof["loop_s_win"] if prof["loop_s_win"] > 0 else float("inf")
                print_fn(
                    "[timing/main] "
                    f"loops={prof['loops']} window={win_count} rate={rate:.1f}it/s "
                    f"avg_submit_last={avg_submit_ms:.2f}ms "
                    f"avg_wait_last={avg_wait_ms:.2f}ms "
                    f"max_wait_last={max_wait_ms:.2f}ms "
                    f"q05_wait_last={q05_ms:.2f}ms "
                    f"q50_wait_last={q50_ms:.2f}ms "
                    f"q95_wait_last={q95_ms:.2f}ms "
                    f"avg_push_last={avg_push_ms:.2f}ms "
                    f"max_push_last={max_push_ms:.2f}ms "
                    f"q05_push_last={pq05_ms:.2f}ms "
                    f"q50_push_last={pq50_ms:.2f}ms "
                    f"q95_push_last={pq95_ms:.2f}ms "
                    f"avg_qget_last={qg_avg_ms:.2f}ms "
                    f"max_qget_last={qg_max_ms:.2f}ms "
                    f"q05_qget_last={qg05_ms:.2f}ms "
                    f"q50_qget_last={qg50_ms:.2f}ms "
                    f"q95_qget_last={qg95_ms:.2f}ms "
                    f"avg_qwait_last={qw_avg_ms:.2f}ms "
                    f"max_qwait_last={qw_max_ms:.2f}ms "
                    f"q05_qwait_last={qw05_ms:.2f}ms "
                    f"q50_qwait_last={qw50_ms:.2f}ms "
                    f"q95_qwait_last={qw95_ms:.2f}ms "
                    f"avg_qput_last={qp_avg_ms:.2f}ms "
                    f"max_qput_last={qp_max_ms:.2f}ms "
                    f"q05_qput_last={qp05_ms:.2f}ms "
                    f"q50_qput_last={qp50_ms:.2f}ms "
                    f"q95_qput_last={qp95_ms:.2f}ms "
                    f"avg_handle_last={avg_handle_ms:.2f}ms "
                    f"avg_batch_last={avg_batch:.2f} "
                    f"avg_push_batch_last={avg_push_batch:.2f} "
                    f"queue={queue_size}"
                )
                if wandb_logger is not None:
                    wandb_logger(
                        {
                            f"{wandb_prefix}/window_loops": win_count,
                            f"{wandb_prefix}/rate_last_it_s": rate,
                            f"{wandb_prefix}/avg_submit_last_ms": avg_submit_ms,
                            f"{wandb_prefix}/avg_wait_last_ms": avg_wait_ms,
                            f"{wandb_prefix}/max_wait_last_ms": max_wait_ms,
                            f"{wandb_prefix}/q05_wait_last_ms": q05_ms,
                            f"{wandb_prefix}/q50_wait_last_ms": q50_ms,
                            f"{wandb_prefix}/q95_wait_last_ms": q95_ms,
                            f"{wandb_prefix}/avg_push_last_ms": avg_push_ms,
                            f"{wandb_prefix}/max_push_last_ms": max_push_ms,
                            f"{wandb_prefix}/q05_push_last_ms": pq05_ms,
                            f"{wandb_prefix}/q50_push_last_ms": pq50_ms,
                            f"{wandb_prefix}/q95_push_last_ms": pq95_ms,
                            f"{wandb_prefix}/avg_qget_last_ms": qg_avg_ms,
                            f"{wandb_prefix}/max_qget_last_ms": qg_max_ms,
                            f"{wandb_prefix}/q05_qget_last_ms": qg05_ms,
                            f"{wandb_prefix}/q50_qget_last_ms": qg50_ms,
                            f"{wandb_prefix}/q95_qget_last_ms": qg95_ms,
                            f"{wandb_prefix}/avg_qwait_last_ms": qw_avg_ms,
                            f"{wandb_prefix}/max_qwait_last_ms": qw_max_ms,
                            f"{wandb_prefix}/q05_qwait_last_ms": qw05_ms,
                            f"{wandb_prefix}/q50_qwait_last_ms": qw50_ms,
                            f"{wandb_prefix}/q95_qwait_last_ms": qw95_ms,
                            f"{wandb_prefix}/avg_qput_last_ms": qp_avg_ms,
                            f"{wandb_prefix}/max_qput_last_ms": qp_max_ms,
                            f"{wandb_prefix}/q05_qput_last_ms": qp05_ms,
                            f"{wandb_prefix}/q50_qput_last_ms": qp50_ms,
                            f"{wandb_prefix}/q95_qput_last_ms": qp95_ms,
                            f"{wandb_prefix}/avg_handle_last_ms": avg_handle_ms,
                            f"{wandb_prefix}/avg_batch_last": avg_batch,
                            f"{wandb_prefix}/avg_push_batch_last": avg_push_batch,
                            f"{wandb_prefix}/queue": queue_size,
                        },
                        step=prof["loops"],
                    )
                prof["submit_s_win"] = 0.0
                prof["wait_s_win"] = 0.0
                prof["handle_s_win"] = 0.0
                prof["loop_s_win"] = 0.0
                prof["wait_s_max_win"] = 0.0
                prof["wait_samples_win"] = []
                prof["push_s_win"] = 0.0
                prof["push_s_max_win"] = 0.0
                prof["push_samples_win"] = []
                prof["queue_get_samples_win"] = []
                prof["queue_put_samples_win"] = []
                prof["queue_wait_samples_win"] = []
                prof["batch_size_win"] = 0
                prof["push_batch_size_win"] = 0
                prof["win_count"] = 0
        return hook

    def on_end(state):
        def hook(num_nodes=None, num_edges=None, **kwargs):
            prof = _ensure(state)
            loops = prof["loops"]
            if loops == 0:
                return
            avg_submit_ms = 1000.0 * prof["submit_s_total"] / loops
            avg_wait_ms = 1000.0 * prof["wait_s_total"] / loops
            avg_handle_ms = 1000.0 * prof["handle_s_total"] / loops
            avg_loop_ms = 1000.0 * prof["loop_s_total"] / loops
            avg_push_ms = 1000.0 * prof["push_s_total"] / loops
            max_wait_ms = 1000.0 * prof["wait_s_max_total"]
            max_push_ms = 1000.0 * prof["push_s_max_total"]
            q05_ms, q50_ms, q95_ms = (
                1000.0 * q for q in np.quantile(prof["wait_samples_total"], [0.05, 0.5, 0.95])
            )
            if prof["push_samples_total"]:
                pq05_ms, pq50_ms, pq95_ms = (
                    1000.0 * q for q in np.quantile(prof["push_samples_total"], [0.05, 0.5, 0.95])
                )
            else:
                pq05_ms = pq50_ms = pq95_ms = 0.0
            if prof["queue_get_samples_total"]:
                qg_avg_ms = float(np.mean(prof["queue_get_samples_total"]))
                qg_max_ms = float(np.max(prof["queue_get_samples_total"]))
                qg05_ms, qg50_ms, qg95_ms = (
                    float(q) for q in np.quantile(prof["queue_get_samples_total"], [0.05, 0.5, 0.95])
                )
            else:
                qg_avg_ms = qg_max_ms = qg05_ms = qg50_ms = qg95_ms = 0.0
            if prof["queue_wait_samples_total"]:
                qw_avg_ms = float(np.mean(prof["queue_wait_samples_total"]))
                qw_max_ms = float(np.max(prof["queue_wait_samples_total"]))
                qw05_ms, qw50_ms, qw95_ms = (
                    float(q) for q in np.quantile(prof["queue_wait_samples_total"], [0.05, 0.5, 0.95])
                )
            else:
                qw_avg_ms = qw_max_ms = qw05_ms = qw50_ms = qw95_ms = 0.0
            if prof["queue_put_samples_total"]:
                qp_avg_ms = float(np.mean(prof["queue_put_samples_total"]))
                qp_max_ms = float(np.max(prof["queue_put_samples_total"]))
                qp05_ms, qp50_ms, qp95_ms = (
                    float(q) for q in np.quantile(prof["queue_put_samples_total"], [0.05, 0.5, 0.95])
                )
            else:
                qp_avg_ms = qp_max_ms = qp05_ms = qp50_ms = qp95_ms = 0.0
            rate = loops / prof["loop_s_total"] if prof["loop_s_total"] > 0 else float("inf")
            print_fn(
                "[timing/main/final] "
                f"loops={loops} rate={rate:.1f}it/s "
                f"avg_submit_overall={avg_submit_ms:.2f}ms "
                f"avg_wait_overall={avg_wait_ms:.2f}ms "
                f"max_wait_overall={max_wait_ms:.2f}ms "
                f"q05_wait_overall={q05_ms:.2f}ms "
                f"q50_wait_overall={q50_ms:.2f}ms "
                f"q95_wait_overall={q95_ms:.2f}ms "
                f"avg_push_overall={avg_push_ms:.2f}ms "
                f"max_push_overall={max_push_ms:.2f}ms "
                f"q05_push_overall={pq05_ms:.2f}ms "
                f"q50_push_overall={pq50_ms:.2f}ms "
                f"q95_push_overall={pq95_ms:.2f}ms "
                f"avg_qget_overall={qg_avg_ms:.2f}ms "
                f"max_qget_overall={qg_max_ms:.2f}ms "
                f"q05_qget_overall={qg05_ms:.2f}ms "
                f"q50_qget_overall={qg50_ms:.2f}ms "
                f"q95_qget_overall={qg95_ms:.2f}ms "
                f"avg_qwait_overall={qw_avg_ms:.2f}ms "
                f"max_qwait_overall={qw_max_ms:.2f}ms "
                f"q05_qwait_overall={qw05_ms:.2f}ms "
                f"q50_qwait_overall={qw50_ms:.2f}ms "
                f"q95_qwait_overall={qw95_ms:.2f}ms "
                f"avg_qput_overall={qp_avg_ms:.2f}ms "
                f"max_qput_overall={qp_max_ms:.2f}ms "
                f"q05_qput_overall={qp05_ms:.2f}ms "
                f"q50_qput_overall={qp50_ms:.2f}ms "
                f"q95_qput_overall={qp95_ms:.2f}ms "
                f"avg_handle_overall={avg_handle_ms:.2f}ms "
                f"avg_loop_overall={avg_loop_ms:.2f}ms "
                f"init_overall={1000.0 * prof['init_s']:.2f}ms "
                f"nodes={num_nodes} edges={num_edges}"
            )
            if wandb_logger is not None:
                wandb_logger(
                    {
                        f"{wandb_prefix}/loops_overall": loops,
                        f"{wandb_prefix}/rate_overall_it_s": rate,
                        f"{wandb_prefix}/avg_submit_overall_ms": avg_submit_ms,
                        f"{wandb_prefix}/avg_wait_overall_ms": avg_wait_ms,
                        f"{wandb_prefix}/max_wait_overall_ms": max_wait_ms,
                        f"{wandb_prefix}/q05_wait_overall_ms": q05_ms,
                        f"{wandb_prefix}/q50_wait_overall_ms": q50_ms,
                        f"{wandb_prefix}/q95_wait_overall_ms": q95_ms,
                        f"{wandb_prefix}/avg_push_overall_ms": avg_push_ms,
                        f"{wandb_prefix}/max_push_overall_ms": max_push_ms,
                        f"{wandb_prefix}/q05_push_overall_ms": pq05_ms,
                        f"{wandb_prefix}/q50_push_overall_ms": pq50_ms,
                        f"{wandb_prefix}/q95_push_overall_ms": pq95_ms,
                        f"{wandb_prefix}/avg_qget_overall_ms": qg_avg_ms,
                        f"{wandb_prefix}/max_qget_overall_ms": qg_max_ms,
                        f"{wandb_prefix}/q05_qget_overall_ms": qg05_ms,
                        f"{wandb_prefix}/q50_qget_overall_ms": qg50_ms,
                        f"{wandb_prefix}/q95_qget_overall_ms": qg95_ms,
                        f"{wandb_prefix}/avg_qwait_overall_ms": qw_avg_ms,
                        f"{wandb_prefix}/max_qwait_overall_ms": qw_max_ms,
                        f"{wandb_prefix}/q05_qwait_overall_ms": qw05_ms,
                        f"{wandb_prefix}/q50_qwait_overall_ms": qw50_ms,
                        f"{wandb_prefix}/q95_qwait_overall_ms": qw95_ms,
                        f"{wandb_prefix}/avg_qput_overall_ms": qp_avg_ms,
                        f"{wandb_prefix}/max_qput_overall_ms": qp_max_ms,
                        f"{wandb_prefix}/q05_qput_overall_ms": qp05_ms,
                        f"{wandb_prefix}/q50_qput_overall_ms": qp50_ms,
                        f"{wandb_prefix}/q95_qput_overall_ms": qp95_ms,
                        f"{wandb_prefix}/avg_handle_overall_ms": avg_handle_ms,
                        f"{wandb_prefix}/avg_loop_overall_ms": avg_loop_ms,
                        f"{wandb_prefix}/init_overall_ms": 1000.0 * prof["init_s"],
                        f"{wandb_prefix}/nodes": num_nodes,
                        f"{wandb_prefix}/edges": num_edges,
                    },
                    step=loops,
                )
        return hook

    return {
        "main.start": [on_start],
        "main.init.start": [on_init_start],
        "main.init.end": [on_init_end],
        "main.init.after_push": [on_after_push],
        "main.loop.start": [on_loop_start],
        "main.loop.submitted": [on_submitted],
        "main.loop.before_get": [on_before_get],
        "main.loop.after_get": [on_after_get],
        "main.loop.after_push": [on_after_push],
        "main.loop.end": [on_loop_end],
        "main.end": [on_end],
    }


def create_worker_timing_profile_hooks(
    log_every: int = 1000,
    print_fn=print,
    key: str = "_timing_profile",
    wandb_log: bool = False,
    wandb_prefix: str = "timing/worker",
) -> Dict[str, List]:
    """
    Create worker timing hooks for per-actor expand() runtime.
    """
    if log_every < 1:
        log_every = 1
    wandb_logger = _make_wandb_logger(wandb_log)

    def _ensure(worker):
        if not hasattr(worker, key):
            setattr(
                worker,
                key,
                {
                    "expands": 0,
                    "expand_s_total": 0.0,
                    "expand_s_win": 0.0,
                    "win_count": 0,
                    "t_expand_start": None,
                },
            )
        return getattr(worker, key)

    def on_expand_start(worker):
        def hook(**kwargs):
            prof = _ensure(worker)
            prof["t_expand_start"] = perf_counter()
        return hook

    def on_expand_end(worker):
        def hook(result=None, **kwargs):
            prof = _ensure(worker)
            t0 = prof.get("t_expand_start")
            if t0 is None:
                return
            dt = max(0.0, perf_counter() - t0)
            prof["expands"] += 1
            prof["expand_s_total"] += dt
            prof["expand_s_win"] += dt
            prof["win_count"] += 1
            expands = prof["expands"]
            if expands % log_every == 0:
                win_count = prof["win_count"]
                avg_ms = 1000.0 * prof["expand_s_win"] / win_count
                rate = win_count / prof["expand_s_win"] if prof["expand_s_win"] > 0 else float("inf")
                n_neighbors = len(result.neighbors) if result is not None else None
                print_fn(
                    "[timing/worker] "
                    f"expands={expands} window={win_count} rate={rate:.1f}it/s "
                    f"avg_expand_last={avg_ms:.2f}ms "
                    f"neighbors={n_neighbors}"
                )
                if wandb_logger is not None:
                    wandb_logger(
                        {
                            f"{wandb_prefix}/expands": expands,
                            f"{wandb_prefix}/window_expands": win_count,
                            f"{wandb_prefix}/rate_last_it_s": rate,
                            f"{wandb_prefix}/avg_expand_last_ms": avg_ms,
                            f"{wandb_prefix}/neighbors_last": n_neighbors,
                        },
                        step=expands,
                    )
                prof["expand_s_win"] = 0.0
                prof["win_count"] = 0
        return hook

    return {
        "expand.start": [on_expand_start],
        "expand.end": [on_expand_end],
    }


def create_main_gc_profile_hooks(
    log_every: int = 1000,
    print_fn=print,
    key: str = "_gc_profile",
    wandb_log: bool = False,
    wandb_prefix: str = "gc/main",
    type_sample_every: Optional[int] = None,
    tracked_type_names: Optional[List[str]] = None,
) -> Dict[str, List]:
    """
    Create main hooks for:
    1) GC event telemetry via gc.callbacks.
    2) Sampled census of GC-tracked object types via gc.get_objects().
    """
    if log_every < 1:
        log_every = 1
    if type_sample_every is None:
        type_sample_every = log_every
    if type_sample_every < 1:
        type_sample_every = 1
    wandb_logger = _make_wandb_logger(wandb_log)
    selected_types = tuple(
        tracked_type_names
        or [
            "Node",
            "WFPath",
            "ExpandResult",
            "ExpandNeighbor",
            "dict",
            "list",
            "tuple",
            "set",
            "deque",
        ]
    )
    selected_set = set(selected_types)

    def _ensure(state):
        if key not in state:
            state[key] = {
                "loops": 0,
                "events_total": 0,
                "events_win": 0,
                "pause_s_total": 0.0,
                "pause_s_win": 0.0,
                "pause_s_max_total": 0.0,
                "pause_s_max_win": 0.0,
                "collected_total": 0,
                "collected_win": 0,
                "uncollectable_total": 0,
                "uncollectable_win": 0,
                "start_times": {},
                "by_gen_total": {},
                "by_gen_win": {},
                "callback": None,
                "callback_registered": False,
                "last_sample_total_tracked": 0,
                "last_sample_selected_counts": {name: 0 for name in selected_types},
            }
        return state[key]

    def _gc_callback(state):
        def callback(phase, info):
            prof = _ensure(state)
            gen = int(info.get("generation", -1))
            if phase == "start":
                prof["start_times"][gen] = perf_counter()
                return

            t0 = prof["start_times"].pop(gen, None)
            if t0 is None:
                return
            dt = max(0.0, perf_counter() - t0)
            collected = int(info.get("collected", 0) or 0)
            uncollectable = int(info.get("uncollectable", 0) or 0)

            prof["events_total"] += 1
            prof["events_win"] += 1
            prof["pause_s_total"] += dt
            prof["pause_s_win"] += dt
            prof["pause_s_max_total"] = max(prof["pause_s_max_total"], dt)
            prof["pause_s_max_win"] = max(prof["pause_s_max_win"], dt)
            prof["collected_total"] += collected
            prof["collected_win"] += collected
            prof["uncollectable_total"] += uncollectable
            prof["uncollectable_win"] += uncollectable

            gen_total = prof["by_gen_total"].setdefault(
                gen, {"events": 0, "pause_s": 0.0, "pause_s_max": 0.0, "collected": 0, "uncollectable": 0}
            )
            gen_win = prof["by_gen_win"].setdefault(
                gen, {"events": 0, "pause_s": 0.0, "pause_s_max": 0.0, "collected": 0, "uncollectable": 0}
            )
            for bucket in (gen_total, gen_win):
                bucket["events"] += 1
                bucket["pause_s"] += dt
                bucket["pause_s_max"] = max(bucket["pause_s_max"], dt)
                bucket["collected"] += collected
                bucket["uncollectable"] += uncollectable

        return callback

    def on_start(state):
        def hook(**kwargs):
            prof = _ensure(state)
            if prof["callback_registered"]:
                return
            cb = _gc_callback(state)
            prof["callback"] = cb
            gc.callbacks.append(cb)
            prof["callback_registered"] = True
        return hook

    def on_loop_end(state):
        def hook(**kwargs):
            prof = _ensure(state)
            prof["loops"] += 1
            loops = prof["loops"]

            if loops % type_sample_every == 0:
                selected_counts = {name: 0 for name in selected_types}
                tracked_total = 0
                for obj in gc.get_objects():
                    tracked_total += 1
                    name = type(obj).__name__
                    if name in selected_set:
                        selected_counts[name] += 1
                prof["last_sample_total_tracked"] = tracked_total
                prof["last_sample_selected_counts"] = selected_counts

            if loops % log_every != 0:
                return

            events = prof["events_win"]
            pause_avg_ms = (1000.0 * prof["pause_s_win"] / events) if events > 0 else 0.0
            pause_max_ms = 1000.0 * prof["pause_s_max_win"]
            gc_count = gc.get_count()
            thresholds = gc.get_threshold()
            selected_counts = prof["last_sample_selected_counts"]
            selected_fragment = " ".join(f"{name}={selected_counts.get(name, 0)}" for name in selected_types)

            print_fn(
                "[gc/main] "
                f"loops={loops} events_win={events} pause_avg_win={pause_avg_ms:.3f}ms "
                f"pause_max_win={pause_max_ms:.3f}ms collected_win={prof['collected_win']} "
                f"uncollectable_win={prof['uncollectable_win']} gc_count={gc_count} "
                f"gc_threshold={thresholds} tracked_total={prof['last_sample_total_tracked']} "
                f"{selected_fragment}"
            )
            if wandb_logger is not None:
                payload = {
                    f"{wandb_prefix}/loops": loops,
                    f"{wandb_prefix}/events_win": events,
                    f"{wandb_prefix}/pause_avg_win_ms": pause_avg_ms,
                    f"{wandb_prefix}/pause_max_win_ms": pause_max_ms,
                    f"{wandb_prefix}/collected_win": prof["collected_win"],
                    f"{wandb_prefix}/uncollectable_win": prof["uncollectable_win"],
                    f"{wandb_prefix}/gc_count_gen0": gc_count[0],
                    f"{wandb_prefix}/gc_count_gen1": gc_count[1],
                    f"{wandb_prefix}/gc_count_gen2": gc_count[2],
                    f"{wandb_prefix}/tracked_total": prof["last_sample_total_tracked"],
                }
                for name in selected_types:
                    payload[f"{wandb_prefix}/type_count/{name}"] = selected_counts.get(name, 0)
                wandb_logger(payload, step=loops)

            prof["events_win"] = 0
            prof["pause_s_win"] = 0.0
            prof["pause_s_max_win"] = 0.0
            prof["collected_win"] = 0
            prof["uncollectable_win"] = 0
            prof["by_gen_win"] = {}

        return hook

    def on_end(state):
        def hook(**kwargs):
            prof = _ensure(state)
            cb = prof.get("callback")
            if cb is not None and prof.get("callback_registered"):
                try:
                    gc.callbacks.remove(cb)
                except ValueError:
                    pass
                prof["callback_registered"] = False
                prof["callback"] = None

            events = prof["events_total"]
            pause_avg_ms = (1000.0 * prof["pause_s_total"] / events) if events > 0 else 0.0
            pause_max_ms = 1000.0 * prof["pause_s_max_total"]
            by_gen_parts = []
            for gen in sorted(prof["by_gen_total"].keys()):
                bucket = prof["by_gen_total"][gen]
                gen_avg_ms = (1000.0 * bucket["pause_s"] / bucket["events"]) if bucket["events"] > 0 else 0.0
                by_gen_parts.append(
                    f"gen{gen}:events={bucket['events']},avg_ms={gen_avg_ms:.3f},max_ms={1000.0 * bucket['pause_s_max']:.3f}"
                )
            by_gen_fragment = " ".join(by_gen_parts) if by_gen_parts else "no_gc_events"

            print_fn(
                "[gc/main/final] "
                f"loops={prof['loops']} events_total={events} pause_avg_total={pause_avg_ms:.3f}ms "
                f"pause_max_total={pause_max_ms:.3f}ms collected_total={prof['collected_total']} "
                f"uncollectable_total={prof['uncollectable_total']} {by_gen_fragment}"
            )

        return hook

    return {
        "main.start": [on_start],
        "main.loop.end": [on_loop_end],
        "main.end": [on_end],
    }
