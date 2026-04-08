from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import queue
import random
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/home/kserver/workspace")
KSERVER_SRC = ROOT.parent / "k-server-bench" / "k-servers" / "src"
CK4_METRIC = "/home/kserver/k-server-bench/metrics/circle_k4_m6.pickle"
TAXI_METRIC = "/home/kserver/k-server-bench/metrics/circle_taxi_k4_m6.pickle"
N_COEFS = 21
FROZEN_ZERO_IDXS = tuple(range(6))
ACTIVE_IDXS = tuple(range(6, N_COEFS))
VALUE_SET = (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
MATRIX = [
    [-1, -1, -1, -1],
    [1, -2, -3, -4],
    [1, 2, -5, -6],
    [1, 3, 5, -7],
    [1, 4, 6, 7],
]
SEED_VECTORS = [
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, -1, -1, 0, 0),
    (0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, -1, 1, -1),
    (0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, -1, 0, 1, 0, -1, 0, -1, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, -1, 0, 0, -1, -1, 0),
    (0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1, 0, 0, 1, 0, 0, 0, -1, 1, 1),
]
STAGE_B_PROXY_THRESHOLD = 5
STAGE_C_TIMEOUT_SECONDS = 30
STAGE_C_TIMEOUT_MIN_SECONDS = 10
STAGE_C_EDGE_RATIO = 0.05
STAGE_C_TIMEOUT_TRIGGER = 7
STAGE_D_SHORTLIST = 12
STAGE_D_UNIQUE_EDGE_GATE_INITIAL = 17
STAGE_D_UNIQUE_EDGE_GATE_MIN = 1
STAGE_D_UNIQUE_EDGE_COLLECT_LIMIT = STAGE_D_UNIQUE_EDGE_GATE_INITIAL + 1
STAGE_C_MAX_PENDING = 128
STAGE_D_MAX_PENDING = 24
STAGE_C_FRONTIER_ESTIMATE = 20.0
STAGE_C_DISAGREEMENT_ESTIMATE = 200.0
STAGE_D_FRONTIER_FULL = 20
STAGE_D_DISAGREEMENT_FULL = 100
STAGE_EDGE_COLLECT_LIMIT = 128
FAST_WORKERS_TOTAL = 6
FAST_STAGE_A_WORKERS = 2
FAST_STAGE_B_WORKERS = 4
HEAVY_CPU_BUDGET = 14
STAGE_C_MAX_CPUS = 6
STAGE_D_MIN_CPUS = 6
STAGE_C_BATCH_SIZE = 4
STAGE_D_BATCH_SIZE = 2
CHECKPOINT_INTERVAL = 30.0
ADAPTIVE_EDGE_MAX = 4096
CORE_EDGE_BUDGET = 1024
RECENT_EDGE_BUDGET = 2048
FREQUENT_EDGE_BUDGET = 2048
DISAGREEMENT_EDGE_BUDGET = 1024
FRONTIER_EDGE_BUDGET = 1024
STAGE_B_NODE_CACHE_MAX_BYTES = 768 * 1024 * 1024
LOG_PATH: Path | None = None


@dataclass
class SearchConfig:
    timeout_s: float
    n_cpus: int
    output_path: Path
    checkpoint_path: Path
    event_log_path: Path
    seed: int = 20260401
    stage_a_backlog_factor: int = 6
    proposal_support_bias: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    checkpoint_interval_s: float = CHECKPOINT_INTERVAL
    final_full_taxi_budget: int = STAGE_D_SHORTLIST


@dataclass
class CandidateRecord:
    candidate_id: int
    coefs: tuple[int, ...]
    family: str
    parent_id: int | None
    created_at: float
    stage_a_violations: int | None = None
    stage_b_proxy: int | None = None
    stage_b_edge_idxes: list[int] | None = None
    stage_c_metrics: dict[str, float] | None = None
    stage_d_metrics: dict[str, Any] | None = None


@dataclass
class StageStats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    wait_times: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)
    queue_peak: int = 0
    queue_last_nonempty_ts: float | None = None


def log(msg: str) -> None:
    line = f"[n7-async] {msg}"
    print(line, flush=True)
    if LOG_PATH is not None:
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def set_log_path(path: Path) -> None:
    global LOG_PATH
    LOG_PATH = path
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")


def load_main_module():
    spec = importlib.util.spec_from_file_location("candidate_main", ROOT / "main.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_ck4_cache():
    mod = load_main_module()
    inst = mod.NumpyKServerInstance.load(CK4_METRIC)
    kwargs = {"n": 7, "index_matrix": MATRIX, "coefs": [0] * N_COEFS}
    context = inst.get_context()
    potential = mod.Potential(context, **kwargs)
    nodes = inst.get_nodes()
    edges = inst.get_edges()
    B = np.empty((len(nodes), potential.tmp.shape[0]), dtype=np.int32)
    penalties_backup = potential.penalties.copy()
    potential.penalties[:] = 0
    for i, node in enumerate(nodes):
        wf = np.asarray(node["wf_norm"], dtype=potential.coefs_dtype)
        potential._compute_candidate_values(wf)
        B[i] = potential.tmp
    potential.penalties[:] = penalties_backup
    return mod, context, potential.distances.astype(np.int16, copy=False), B, edges


def clone_potential_from_base(mod, base_potential, coefs: tuple[int, ...]):
    potential = object.__new__(mod.Potential)
    potential.context = base_potential.context
    potential.kwargs = dict(base_potential.kwargs)
    potential.kwargs["coefs"] = list(coefs)
    potential.k = base_potential.k
    potential.n = base_potential.n
    potential.index_matrix = base_potential.index_matrix
    potential.coefs = np.asarray(coefs, dtype=np.int32)
    potential.support = list(base_potential.support)
    potential.config_idxes_dtype = base_potential.config_idxes_dtype
    potential.coefs_dtype = base_potential.coefs_dtype
    potential.distances = base_potential.distances
    potential.config_idxes = base_potential.config_idxes
    potential.penalties = np.dot(
        potential.distances,
        np.asarray(coefs, dtype=potential.coefs_dtype),
    ).astype(potential.coefs_dtype, copy=False)
    potential.tmp = np.zeros(base_potential.tmp.shape, dtype=base_potential.coefs_dtype)
    potential.scratch = np.zeros(base_potential.scratch.shape, dtype=base_potential.coefs_dtype)
    return potential


def build_zero_penalty_node_matrix(base_potential, nodes, node_idxes: list[int]) -> np.ndarray:
    B = np.empty((len(node_idxes), base_potential.tmp.shape[0]), dtype=np.int32)
    for row_idx, node_idx in enumerate(node_idxes):
        wf = np.asarray(nodes[node_idx]["wf_norm"], dtype=base_potential.coefs_dtype)
        base_potential._compute_candidate_values(wf)
        B[row_idx] = base_potential.tmp
    return B


def compute_node_potentials_from_base(mod, instance, base_potential, coefs: tuple[int, ...], node_idxes: list[int] | None = None) -> dict[int, float]:
    nodes = instance.get_nodes()
    if node_idxes is None:
        node_idxes = list(range(len(nodes)))
    potential = clone_potential_from_base(mod, base_potential, coefs)
    potentials: dict[int, float] = {}
    for node_idx in node_idxes:
        value = potential(nodes[node_idx]["wf_norm"])
        if isinstance(value, tuple):
            value = value[0]
        potentials[int(node_idx)] = float(value)
    return potentials


def evaluate_ck4_cached(mod, context, distances, B, edges, coefs: tuple[int, ...]) -> int:
    coef_arr = np.array(coefs, dtype=np.int16)
    penalty = np.dot(distances, coef_arr).astype(np.int32, copy=False)
    mins = np.min(B - penalty, axis=1)
    viol = 0
    for edge in edges:
        if mod.is_violation(
            float(mins[int(edge["from"])]),
            float(mins[int(edge["to"])]),
            edge["d_min"],
            edge["ext"],
            context.k,
        ):
            viol += 1
    return viol


def collect_violated_edge_idxes(mod, inst, potential_kwargs: dict[str, Any], *, cutoff: int | None = None) -> list[int]:
    context = inst.get_context()
    edges = inst.get_edges()
    potentials = mod.compute_potentials_for_nodes(inst, potential_kwargs)
    violated = []
    for edge_idx, edge in enumerate(edges):
        if mod.is_violation(
            u_potential=potentials[int(edge["from"])],
            v_potential=potentials[int(edge["to"])],
            d_min=edge["d_min"],
            ext=edge["ext"],
            rho=context.k,
        ):
            violated.append(edge_idx)
            if cutoff is not None and len(violated) >= cutoff:
                break
    return violated


def build_baseline_taxi_edge_idxes() -> list[int]:
    mod = load_main_module()
    inst = mod.NumpyKServerInstance.load(TAXI_METRIC)
    baseline_kwargs = mod.default_canonical_kwargs()
    hard_edge_cache = mod.build_hard_edge_cache([inst], baseline_kwargs)
    violated = collect_violated_edge_idxes(mod, inst, baseline_kwargs)
    ordered = list(hard_edge_cache[0]["edge_idxes"])
    seen = set(ordered)
    for edge_idx in violated:
        if edge_idx not in seen:
            ordered.append(edge_idx)
            seen.add(edge_idx)
    return ordered[:ADAPTIVE_EDGE_MAX]


def score_candidate_on_edge_subset_with_violations(
    mod,
    instance,
    potential_kwargs: dict[str, Any],
    edge_idxes: list[int],
    *,
    cutoff: int | None = None,
    collect_limit: int = STAGE_EDGE_COLLECT_LIMIT,
) -> dict[str, Any]:
    context = instance.get_context()
    edges = instance.get_edges()
    node_idxes = sorted(
        {int(edges[edge_idx]["from"]) for edge_idx in edge_idxes}
        | {int(edges[edge_idx]["to"]) for edge_idx in edge_idxes}
    )
    potentials = mod.compute_potentials_for_nodes(instance, potential_kwargs, node_idxes)

    n_violations = 0
    processed = 0
    violated_edge_idxes: list[int] = []
    for edge_idx in edge_idxes:
        edge = edges[edge_idx]
        if mod.is_violation(
            u_potential=potentials[int(edge["from"])],
            v_potential=potentials[int(edge["to"])],
            d_min=edge["d_min"],
            ext=edge["ext"],
            rho=context.k,
        ):
            n_violations += 1
            if len(violated_edge_idxes) < collect_limit:
                violated_edge_idxes.append(int(edge_idx))
            if cutoff is not None and n_violations > cutoff:
                return {
                    "n_violations": n_violations,
                    "edges_processed": processed + 1,
                    "edges_total": len(edge_idxes),
                    "stopped_early": True,
                    "violated_edge_idxes": violated_edge_idxes,
                }
        processed += 1

    return {
        "n_violations": n_violations,
        "edges_processed": processed,
        "edges_total": len(edge_idxes),
        "stopped_early": False,
        "violated_edge_idxes": violated_edge_idxes,
    }


def score_candidate_on_edge_subset_with_cache(
    mod,
    instance,
    base_potential,
    edge_idxes: list[int],
    coefs: tuple[int, ...],
    *,
    cutoff: int | None = None,
    collect_limit: int = STAGE_EDGE_COLLECT_LIMIT,
    cache_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = instance.get_context()
    edges = instance.get_edges()
    if cache_payload is not None:
        coef_arr = np.asarray(coefs, dtype=np.int16)
        penalties = np.dot(cache_payload["distances"], coef_arr).astype(np.int32, copy=False)
        mins = np.min(cache_payload["B"] - penalties, axis=1)
        n_violations = 0
        processed = 0
        violated_edge_idxes: list[int] = []
        for local_u, local_v, d_min, ext, edge_idx in cache_payload["local_edges"]:
            if mod.is_violation(
                float(mins[local_u]),
                float(mins[local_v]),
                d_min,
                ext,
                context.k,
            ):
                n_violations += 1
                if len(violated_edge_idxes) < collect_limit:
                    violated_edge_idxes.append(int(edge_idx))
                if cutoff is not None and n_violations > cutoff:
                    return {
                        "n_violations": n_violations,
                        "edges_processed": processed + 1,
                        "edges_total": len(cache_payload["local_edges"]),
                        "stopped_early": True,
                        "violated_edge_idxes": violated_edge_idxes,
                        "used_node_cache": True,
                        "cached_nodes": int(cache_payload["B"].shape[0]),
                    }
            processed += 1
        return {
            "n_violations": n_violations,
            "edges_processed": processed,
            "edges_total": len(cache_payload["local_edges"]),
            "stopped_early": False,
            "violated_edge_idxes": violated_edge_idxes,
            "used_node_cache": True,
            "cached_nodes": int(cache_payload["B"].shape[0]),
        }

    node_idxes = sorted(
        {int(edges[edge_idx]["from"]) for edge_idx in edge_idxes}
        | {int(edges[edge_idx]["to"]) for edge_idx in edge_idxes}
    )
    potentials = compute_node_potentials_from_base(mod, instance, base_potential, coefs, node_idxes)
    n_violations = 0
    processed = 0
    violated_edge_idxes: list[int] = []
    for edge_idx in edge_idxes:
        edge = edges[edge_idx]
        if mod.is_violation(
            u_potential=potentials[int(edge["from"])],
            v_potential=potentials[int(edge["to"])],
            d_min=edge["d_min"],
            ext=edge["ext"],
            rho=context.k,
        ):
            n_violations += 1
            if len(violated_edge_idxes) < collect_limit:
                violated_edge_idxes.append(int(edge_idx))
            if cutoff is not None and n_violations > cutoff:
                return {
                    "n_violations": n_violations,
                    "edges_processed": processed + 1,
                    "edges_total": len(edge_idxes),
                    "stopped_early": True,
                    "violated_edge_idxes": violated_edge_idxes,
                    "used_node_cache": False,
                    "cached_nodes": len(node_idxes),
                }
        processed += 1

    return {
        "n_violations": n_violations,
        "edges_processed": processed,
        "edges_total": len(edge_idxes),
        "stopped_early": False,
        "violated_edge_idxes": violated_edge_idxes,
        "used_node_cache": False,
        "cached_nodes": len(node_idxes),
    }


def take_top_keys(score_items: list[tuple[int, float]], budget: int, seen: set[int]) -> list[int]:
    out: list[int] = []
    for edge_idx, _ in sorted(score_items, key=lambda item: (-item[1], item[0])):
        if edge_idx in seen:
            continue
        out.append(edge_idx)
        seen.add(edge_idx)
        if len(out) >= budget:
            break
    return out


def rebuild_stage_b_edge_cache(
    core_edge_idxes: list[int],
    recent_last_seen: dict[int, int],
    frequent_counts: dict[int, int],
    disagreement_counts: dict[int, int],
    frontier_counts: dict[int, int],
) -> tuple[list[int], dict[str, int]]:
    selected: list[int] = []
    seen: set[int] = set()

    for edge_idx in core_edge_idxes[:CORE_EDGE_BUDGET]:
        if edge_idx not in seen:
            selected.append(edge_idx)
            seen.add(edge_idx)

    selected.extend(take_top_keys(list(recent_last_seen.items()), RECENT_EDGE_BUDGET, seen))
    selected.extend(take_top_keys(list(frequent_counts.items()), FREQUENT_EDGE_BUDGET, seen))
    selected.extend(take_top_keys(list(disagreement_counts.items()), DISAGREEMENT_EDGE_BUDGET, seen))
    selected.extend(take_top_keys(list(frontier_counts.items()), FRONTIER_EDGE_BUDGET, seen))
    selected = selected[:ADAPTIVE_EDGE_MAX]

    bucket_sizes = {
        "core": min(len(core_edge_idxes), CORE_EDGE_BUDGET),
        "recent_pool": len(recent_last_seen),
        "frequent_pool": len(frequent_counts),
        "disagreement_pool": len(disagreement_counts),
        "frontier_pool": len(frontier_counts),
        "selected_total": len(selected),
    }
    return selected, bucket_sizes


def init_fast_worker(task_queue, result_queue):
    global FAST_TASK_QUEUE, FAST_RESULT_QUEUE, FAST_MOD, FAST_CONTEXT, FAST_DISTANCES, FAST_B, FAST_EDGES
    global FAST_TAXI_INST, FAST_TAXI_CONTEXT, FAST_TAXI_NODES, FAST_TAXI_EDGES, FAST_TAXI_BASE_POTENTIAL
    global FAST_STAGE_B_CACHE_KEY, FAST_STAGE_B_CACHE_PAYLOAD
    started_at = time.time()
    FAST_TASK_QUEUE = task_queue
    FAST_RESULT_QUEUE = result_queue
    FAST_MOD, FAST_CONTEXT, FAST_DISTANCES, FAST_B, FAST_EDGES = build_ck4_cache()
    FAST_TAXI_INST = FAST_MOD.NumpyKServerInstance.load(TAXI_METRIC)
    FAST_TAXI_CONTEXT = FAST_TAXI_INST.get_context()
    FAST_TAXI_NODES = FAST_TAXI_INST.get_nodes()
    FAST_TAXI_EDGES = FAST_TAXI_INST.get_edges()
    FAST_TAXI_BASE_POTENTIAL = FAST_MOD.Potential(FAST_TAXI_CONTEXT, n=7, index_matrix=MATRIX, coefs=[0] * N_COEFS)
    FAST_STAGE_B_CACHE_KEY = None
    FAST_STAGE_B_CACHE_PAYLOAD = None
    print(
        f"[n7-async-worker] pid={mp.current_process().pid} cache_ready "
        f"elapsed={time.time()-started_at:.1f}s nodes={FAST_B.shape[0]} latent={FAST_B.shape[1]}",
        flush=True,
    )


def fast_worker_loop(task_queue, result_queue):
    global FAST_STAGE_B_CACHE_KEY, FAST_STAGE_B_CACHE_PAYLOAD
    init_fast_worker(task_queue, result_queue)
    while True:
        task = task_queue.get()
        if task is None:
            return
        stage = task["stage"]
        started_at = time.time()
        try:
            if stage == "stage_a":
                ck4 = evaluate_ck4_cached(
                    FAST_MOD,
                    FAST_CONTEXT,
                    FAST_DISTANCES,
                    FAST_B,
                    FAST_EDGES,
                    tuple(task["coefs"]),
                )
                payload = {"ck4_violations": int(ck4)}
            elif stage == "stage_b":
                edge_idxes = list(task["edge_idxes"])
                edge_cache_key = tuple(edge_idxes)
                cache_payload = None
                if FAST_STAGE_B_CACHE_KEY != edge_cache_key:
                    node_idxes = sorted(
                        {int(FAST_TAXI_EDGES[edge_idx]["from"]) for edge_idx in edge_idxes}
                        | {int(FAST_TAXI_EDGES[edge_idx]["to"]) for edge_idx in edge_idxes}
                    )
                    estimated_bytes = len(node_idxes) * FAST_TAXI_BASE_POTENTIAL.tmp.shape[0] * np.dtype(np.int32).itemsize
                    if estimated_bytes <= STAGE_B_NODE_CACHE_MAX_BYTES:
                        local_node_map = {node_idx: local_idx for local_idx, node_idx in enumerate(node_idxes)}
                        local_edges = [
                            (
                                local_node_map[int(FAST_TAXI_EDGES[edge_idx]["from"])],
                                local_node_map[int(FAST_TAXI_EDGES[edge_idx]["to"])],
                                float(FAST_TAXI_EDGES[edge_idx]["d_min"]),
                                float(FAST_TAXI_EDGES[edge_idx]["ext"]),
                                int(edge_idx),
                            )
                            for edge_idx in edge_idxes
                        ]
                        FAST_STAGE_B_CACHE_PAYLOAD = {
                            "B": build_zero_penalty_node_matrix(FAST_TAXI_BASE_POTENTIAL, FAST_TAXI_NODES, node_idxes),
                            "distances": FAST_TAXI_BASE_POTENTIAL.distances.astype(np.int16, copy=False),
                            "local_edges": local_edges,
                        }
                    else:
                        FAST_STAGE_B_CACHE_PAYLOAD = None
                    FAST_STAGE_B_CACHE_KEY = edge_cache_key
                cache_payload = FAST_STAGE_B_CACHE_PAYLOAD
                result = score_candidate_on_edge_subset_with_cache(
                    FAST_MOD,
                    FAST_TAXI_INST,
                    FAST_TAXI_BASE_POTENTIAL,
                    edge_idxes,
                    tuple(task["coefs"]),
                    cutoff=task.get("cutoff"),
                    cache_payload=cache_payload,
                )
                payload = {
                    "taxi_proxy_violations": int(result["n_violations"]),
                    "edges_processed": int(result["edges_processed"]),
                    "edges_total": int(result["edges_total"]),
                    "stopped_early": bool(result["stopped_early"]),
                    "violated_edge_idxes": list(result["violated_edge_idxes"]),
                    "used_node_cache": bool(result["used_node_cache"]),
                    "cached_nodes": int(result["cached_nodes"]),
                }
            else:
                raise ValueError(f"Unknown fast stage {stage}")
            result_queue.put(
                {
                    "ok": True,
                    "worker_stage": stage,
                    "task_id": task["task_id"],
                    "candidate_id": task["candidate_id"],
                    "family": task["family"],
                    "parent_id": task["parent_id"],
                    "submitted_at": task["submitted_at"],
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "payload": payload,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            result_queue.put(
                {
                    "ok": False,
                    "worker_stage": stage,
                    "task_id": task["task_id"],
                    "candidate_id": task["candidate_id"],
                    "family": task["family"],
                    "parent_id": task["parent_id"],
                    "submitted_at": task["submitted_at"],
                    "started_at": started_at,
                    "finished_at": time.time(),
                    "error": repr(exc),
                }
            )


def heavy_stage_process(
    stage: str,
    candidate_id: int,
    coefs: tuple[int, ...],
    n_processes: int,
    result_queue,
    stage_timeout_s: float | None = None,
):
    try:
        if str(KSERVER_SRC) not in sys.path:
            sys.path.insert(0, str(KSERVER_SRC))
        from kserver.evaluation import NumpyKServerInstance
        from kserver.evaluation.evaluation import (
            _unwrap_potential_result,
            _wf_key,
            _unique_wf_dict,
            _accumulate_unique_wf,
            batch_compute_potential_mp,
        )

        mod = load_main_module()
        inst = NumpyKServerInstance.load(TAXI_METRIC)
        context = inst.get_context()
        base_potential = mod.Potential(context, n=7, index_matrix=MATRIX, coefs=[0] * N_COEFS)
        payload = evaluate_heavy_candidate(
            mod,
            inst,
            context,
            base_potential,
            stage,
            candidate_id,
            coefs,
            n_processes,
            stage_timeout_s,
            _unwrap_potential_result,
            _wf_key,
            _unique_wf_dict,
            _accumulate_unique_wf,
            batch_compute_potential_mp,
        )
        if payload["processed_normalized_edges_score"] > 0:
            payload["estimated_total_violations"] = payload["violations_k"] / payload["processed_normalized_edges_score"]
        else:
            payload["estimated_total_violations"] = float("inf")
        result_queue.put(
            {
                "ok": True,
                "worker_stage": stage,
                "candidate_id": candidate_id,
                "finished_at": time.time(),
                "payload": payload,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put(
            {
                "ok": False,
                "worker_stage": stage,
                "candidate_id": candidate_id,
                "finished_at": time.time(),
                "error": repr(exc),
            }
        )


def evaluate_heavy_candidate(
    mod,
    inst,
    context,
    base_potential,
    stage: str,
    candidate_id: int,
    coefs: tuple[int, ...],
    n_processes: int,
    stage_timeout_s: float | None,
    _unwrap_potential_result,
    _wf_key,
    _unique_wf_dict,
    _accumulate_unique_wf,
    batch_compute_potential_mp,
) -> dict[str, Any]:
    started_at = time.time()
    potential = clone_potential_from_base(mod, base_potential, coefs)
    nodes = inst.get_nodes()
    edges = inst.get_edges()

    if stage == "stage_c":
        unique_wfs = _unique_wf_dict()
        edge_seed = int((time.time_ns() ^ (candidate_id * 1000003)) & 0xFFFFFFFF)
        rng = np.random.default_rng(edge_seed)
        edge_order = rng.permutation(len(edges))
        sampled_edge_count = max(1, int(len(edges) * STAGE_C_EDGE_RATIO))
        for edge_idx in edge_order[:sampled_edge_count]:
            edge = edges[int(edge_idx)]
            _accumulate_unique_wf(unique_wfs, nodes[int(edge["from"])]["wf_norm"])
            _accumulate_unique_wf(unique_wfs, nodes[int(edge["to"])]["wf_norm"])
        what_to_compute = list(unique_wfs.values())
        cache_start = time.time()
        cache = batch_compute_potential_mp(
            what_to_compute,
            potential,
            n_processes=n_processes,
            timeout=stage_timeout_s if stage_timeout_s is not None else STAGE_C_TIMEOUT_SECONDS,
            chunk_size=50,
        )
        cache_time = time.time() - cache_start

        node_potential = np.full(len(nodes), np.nan, dtype=float)
        for idx, node in enumerate(nodes):
            val = cache.get(_wf_key(node["wf_norm"]), None)
            p, _ = _unwrap_potential_result(val)
            node_potential[idx] = p

        processed_edges = 0
        violations_k = 0
        violated_edge_idxes: list[int] = []
        for edge_idx, edge in enumerate(edges):
            u = int(edge["from"])
            v = int(edge["to"])
            if np.isnan(node_potential[u]) or np.isnan(node_potential[v]):
                continue
            processed_edges += 1
            slack = node_potential[v] - node_potential[u] + (context.k + 1) * edge["d_min"] - edge["ext"]
            if slack < 0:
                violations_k += 1
                if len(violated_edge_idxes) < STAGE_EDGE_COLLECT_LIMIT:
                    violated_edge_idxes.append(int(edge_idx))

        processed_score = processed_edges / len(edges) if len(edges) else 0.0
        payload = {
            "violations_k": int(violations_k),
            "processed_normalized_edges_score": float(processed_score),
            "cache_time": float(cache_time),
            "build_dataframes_time": 0.0,
            "edge_seed": int(edge_seed),
            "timeout_s": float(stage_timeout_s if stage_timeout_s is not None else STAGE_C_TIMEOUT_SECONDS),
            "violated_edge_idxes": violated_edge_idxes,
            "edge_ratio": float(STAGE_C_EDGE_RATIO),
            "sampled_edge_count": int(sampled_edge_count),
            "sampled_unique_wfs": int(len(what_to_compute)),
            "elapsed_s": time.time() - started_at,
        }
        return payload

    unique_wfs = _unique_wf_dict()
    for node in nodes:
        _accumulate_unique_wf(unique_wfs, node["wf_norm"])
    what_to_compute = list(unique_wfs.values())
    cache_start = time.time()
    cache = batch_compute_potential_mp(
        what_to_compute,
        potential,
        n_processes=n_processes,
        timeout=None,
        chunk_size=50,
    )
    cache_time = time.time() - cache_start

    node_potential = np.full(len(nodes), np.nan, dtype=float)
    for idx, node in enumerate(nodes):
        val = cache.get(_wf_key(node["wf_norm"]), None)
        p, _ = _unwrap_potential_result(val)
        node_potential[idx] = p

    processed_edges = 0
    violations_k = 0
    violated_edge_idxes: list[int] = []
    for edge_idx, edge in enumerate(edges):
        u = int(edge["from"])
        v = int(edge["to"])
        if np.isnan(node_potential[u]) or np.isnan(node_potential[v]):
            continue
        processed_edges += 1
        slack = node_potential[v] - node_potential[u] + (context.k + 1) * edge["d_min"] - edge["ext"]
        if slack < 0:
            violations_k += 1
            if len(violated_edge_idxes) < STAGE_EDGE_COLLECT_LIMIT:
                violated_edge_idxes.append(int(edge_idx))
    processed_score = processed_edges / len(edges) if len(edges) else 0.0
    return {
        "violations_k": int(violations_k),
        "processed_normalized_edges_score": float(processed_score),
        "cache_time": float(cache_time),
        "build_dataframes_time": 0.0,
        "violated_edge_idxes": violated_edge_idxes,
        "elapsed_s": time.time() - started_at,
    }


def heavy_stage_batch_process(stage: str, items: list[dict[str, Any]], n_processes: int, result_queue):
    try:
        if str(KSERVER_SRC) not in sys.path:
            sys.path.insert(0, str(KSERVER_SRC))
        from kserver.evaluation import NumpyKServerInstance
        from kserver.evaluation.evaluation import (
            _unwrap_potential_result,
            _wf_key,
            _unique_wf_dict,
            _accumulate_unique_wf,
            batch_compute_potential_mp,
        )

        mod = load_main_module()
        inst = NumpyKServerInstance.load(TAXI_METRIC)
        context = inst.get_context()
        base_potential = mod.Potential(context, n=7, index_matrix=MATRIX, coefs=[0] * N_COEFS)
        for item in items:
            candidate_id = int(item["candidate_id"])
            coefs = tuple(item["coefs"])
            stage_timeout_s = item.get("stage_timeout_s")
            started_at = time.time()
            payload = evaluate_heavy_candidate(
                mod,
                inst,
                context,
                base_potential,
                stage,
                candidate_id,
                coefs,
                n_processes,
                stage_timeout_s,
                _unwrap_potential_result,
                _wf_key,
                _unique_wf_dict,
                _accumulate_unique_wf,
                batch_compute_potential_mp,
            )
            if payload["processed_normalized_edges_score"] > 0:
                payload["estimated_total_violations"] = payload["violations_k"] / payload["processed_normalized_edges_score"]
            else:
                payload["estimated_total_violations"] = float("inf")
            payload["elapsed_s"] = time.time() - started_at
            result_queue.put(
                {
                    "ok": True,
                    "worker_stage": stage,
                    "candidate_id": candidate_id,
                    "finished_at": time.time(),
                    "payload": payload,
                }
            )
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put(
            {
                "ok": False,
                "worker_stage": stage,
                "candidate_id": None,
                "finished_at": time.time(),
                "error": repr(exc),
            }
        )


def heavy_worker_loop(stage: str, worker_id: str, n_processes: int, task_queue, result_queue):
    try:
        if str(KSERVER_SRC) not in sys.path:
            sys.path.insert(0, str(KSERVER_SRC))
        from kserver.evaluation import NumpyKServerInstance
        from kserver.evaluation.evaluation import (
            _unwrap_potential_result,
            _wf_key,
            _unique_wf_dict,
            _accumulate_unique_wf,
            batch_compute_potential_mp,
        )

        mod = load_main_module()
        inst = NumpyKServerInstance.load(TAXI_METRIC)
        context = inst.get_context()
        base_potential = mod.Potential(context, n=7, index_matrix=MATRIX, coefs=[0] * N_COEFS)
        while True:
            task = task_queue.get()
            if task is None:
                break
            candidate_id = int(task["candidate_id"])
            coefs = tuple(task["coefs"])
            stage_timeout_s = task.get("stage_timeout_s")
            started_at = time.time()
            try:
                payload = evaluate_heavy_candidate(
                    mod,
                    inst,
                    context,
                    base_potential,
                    stage,
                    candidate_id,
                    coefs,
                    n_processes,
                    stage_timeout_s,
                    _unwrap_potential_result,
                    _wf_key,
                    _unique_wf_dict,
                    _accumulate_unique_wf,
                    batch_compute_potential_mp,
                )
                if payload["processed_normalized_edges_score"] > 0:
                    payload["estimated_total_violations"] = payload["violations_k"] / payload["processed_normalized_edges_score"]
                else:
                    payload["estimated_total_violations"] = float("inf")
                payload["elapsed_s"] = time.time() - started_at
                result_queue.put(
                    {
                        "ok": True,
                        "worker_stage": stage,
                        "worker_id": worker_id,
                        "candidate_id": candidate_id,
                        "finished_at": time.time(),
                        "payload": payload,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                result_queue.put(
                    {
                        "ok": False,
                        "worker_stage": stage,
                        "worker_id": worker_id,
                        "candidate_id": candidate_id,
                        "finished_at": time.time(),
                        "error": repr(exc),
                    }
                )
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put(
            {
                "ok": False,
                "worker_stage": stage,
                "worker_id": worker_id,
                "candidate_id": None,
                "finished_at": time.time(),
                "error": repr(exc),
            }
        )


def sparsity_stats(coefs: tuple[int, ...]) -> tuple[int, int]:
    l0 = sum(1 for x in coefs if x != 0)
    l1 = sum(abs(x) for x in coefs)
    return l0, l1


def rank_key(record: CandidateRecord) -> tuple[Any, ...]:
    l0, l1 = sparsity_stats(record.coefs)
    timed = None if record.stage_c_metrics is None else record.stage_c_metrics.get("estimated_total_violations")
    proxy = record.stage_b_proxy
    full = None if record.stage_d_metrics is None else record.stage_d_metrics.get("violations_k")
    if full is not None:
        return (0, full, timed if timed is not None else float("inf"), proxy if proxy is not None else 10**9, l0, l1, record.coefs)
    if timed is not None:
        return (1, timed, proxy if proxy is not None else 10**9, l0, l1, record.coefs)
    if proxy is not None:
        return (2, proxy, l0, l1, record.coefs)
    return (3, record.stage_a_violations if record.stage_a_violations is not None else 10**9, l0, l1, record.coefs)


def top_records(records: dict[int, CandidateRecord], *, limit: int, predicate) -> list[CandidateRecord]:
    items = [rec for rec in records.values() if predicate(rec)]
    items.sort(key=rank_key)
    return items[:limit]


def choose_weighted_family(rng: random.Random, family_rewards: dict[str, float], families: list[str]) -> str:
    family_base_weights = {
        "seed_sparse": 1.0,
        "seed_dense": 1.35,
        "seed_peak5": 1.15,
        "mutate_ck4": 1.0,
        "mutate_proxy": 1.0,
        "mutate_timed": 1.0,
        "mutate_full": 1.0,
    }
    weights = [family_base_weights.get(f, 1.0) * max(0.1, family_rewards.get(f, 1.0)) for f in families]
    total = sum(weights)
    draw = rng.random() * total
    acc = 0.0
    for family, weight in zip(families, weights):
        acc += weight
        if draw <= acc:
            return family
    return families[-1]


def mutate_vector(parent: tuple[int, ...], rng: random.Random, family: str) -> tuple[int, ...]:
    vec = list(parent)
    if family == "seed_sparse":
        support = rng.choice((2, 3, 4, 5, 6))
        vec = [0] * N_COEFS
        for idx in rng.sample(list(ACTIVE_IDXS), support):
            vec[idx] = rng.choice((-5, -4, -3, -2, -1, 1, 2, 3, 4, 5))
    elif family == "seed_peak5":
        vec = [0] * N_COEFS
        peak_idxs = set(rng.sample(list(ACTIVE_IDXS), 2))
        for idx in ACTIVE_IDXS:
            if idx in peak_idxs:
                vec[idx] = rng.choice((-5, 5))
            else:
                vec[idx] = rng.choice((-2, -1, 0, 1, 2))
    elif family == "seed_dense":
        for idx in ACTIVE_IDXS:
            vec[idx] = rng.choice(VALUE_SET)
    else:
        n_edits = {
            "mutate_proxy": rng.randint(1, 3),
            "mutate_timed": rng.randint(1, 2),
            "mutate_full": rng.randint(1, 2),
            "mutate_ck4": rng.randint(2, 5),
        }.get(family, rng.randint(1, 4))
        for _ in range(n_edits):
            idx = rng.choice(ACTIVE_IDXS)
            op = rng.random()
            if op < 0.30:
                vec[idx] = rng.choice(VALUE_SET)
            elif op < 0.60:
                vec[idx] = max(-2, min(2, vec[idx] + rng.choice((-1, 1))))
            elif op < 0.80:
                vec[idx] = max(-2, min(2, vec[idx] + rng.choice((-2, 2))))
            else:
                vec[idx] = 0
    for idx in FROZEN_ZERO_IDXS:
        vec[idx] = 0
    return tuple(vec)


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "avg": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "avg": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def append_event(log_path: Path, payload: dict[str, Any]) -> None:
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def summarize_stage(stats: StageStats) -> dict[str, Any]:
    waits = percentiles(stats.wait_times)
    durs = percentiles(stats.durations)
    throughput = stats.completed / max(1e-9, sum(stats.durations)) if stats.durations else 0.0
    return {
        "submitted": stats.submitted,
        "completed": stats.completed,
        "failed": stats.failed,
        "queue_peak": stats.queue_peak,
        "wait_time": waits,
        "duration": durs,
        "avg_jobs_per_sec": throughput,
    }


def safe_rate(num: float, den: float) -> float:
    return 0.0 if den == 0 else float(num) / float(den)


def threshold_counts(values: list[float], thresholds: list[float]) -> dict[str, int]:
    out = {}
    for thr in thresholds:
        out[f"le_{thr}"] = int(sum(1 for v in values if v <= thr))
    out["gt_max"] = int(sum(1 for v in values if v > thresholds[-1])) if thresholds else len(values)
    return out


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def build_funnel_summary(records: dict[int, CandidateRecord]) -> dict[str, Any]:
    all_records = list(records.values())
    stage_a_done = [rec for rec in all_records if rec.stage_a_violations is not None]
    stage_a_pass = [rec for rec in stage_a_done if rec.stage_a_violations == 0]
    stage_b_done = [rec for rec in stage_a_pass if rec.stage_b_proxy is not None]
    stage_b_pass = [rec for rec in stage_b_done if rec.stage_b_proxy <= STAGE_B_PROXY_THRESHOLD]
    stage_c_done = [rec for rec in stage_b_pass if rec.stage_c_metrics is not None]
    stage_d_done = [rec for rec in stage_c_done if rec.stage_d_metrics is not None]

    b_values = [float(rec.stage_b_proxy) for rec in stage_b_done]
    c_est = [float(rec.stage_c_metrics["estimated_total_violations"]) for rec in stage_c_done]
    c_proc = [float(rec.stage_c_metrics["processed_normalized_edges_score"]) for rec in stage_c_done]
    d_values = [float(rec.stage_d_metrics["violations_k"]) for rec in stage_d_done]

    overlap_bc = [(float(rec.stage_b_proxy), float(rec.stage_c_metrics["estimated_total_violations"])) for rec in stage_c_done]
    overlap_cd = [
        (float(rec.stage_c_metrics["estimated_total_violations"]), float(rec.stage_d_metrics["violations_k"]))
        for rec in stage_d_done
    ]

    return {
        "counts": {
            "created": len(all_records),
            "stage_a_done": len(stage_a_done),
            "stage_a_pass_zero": len(stage_a_pass),
            "stage_b_done": len(stage_b_done),
            "stage_b_pass_threshold": len(stage_b_pass),
            "stage_c_done": len(stage_c_done),
            "stage_d_done": len(stage_d_done),
        },
        "pass_rates": {
            "stage_a_zero_given_done": safe_rate(len(stage_a_pass), len(stage_a_done)),
            "stage_b_threshold_given_stage_a_zero": safe_rate(len(stage_b_pass), len(stage_a_pass)),
            "stage_c_done_given_stage_b_pass": safe_rate(len(stage_c_done), len(stage_b_pass)),
            "stage_d_done_given_stage_c_done": safe_rate(len(stage_d_done), len(stage_c_done)),
        },
        "stage_b_proxy": {
            "stats": percentiles(b_values),
            "threshold_counts": threshold_counts(b_values, [1, 3, 5, 7, 10]) if b_values else {},
        },
        "stage_c_estimate": {
            "stats": percentiles(c_est),
            "threshold_counts": threshold_counts(c_est, [5, 10, 20, 50, 100, 1000]) if c_est else {},
        },
        "stage_c_processed_fraction": {
            "stats": percentiles(c_proc),
            "threshold_counts": threshold_counts(c_proc, [0.05, 0.1, 0.15, 0.2, 0.3]) if c_proc else {},
        },
        "stage_d_full": {
            "stats": percentiles(d_values),
            "threshold_counts": threshold_counts(d_values, [5, 10, 20, 50, 100, 1000]) if d_values else {},
        },
        "correlations": {
            "stage_b_proxy_vs_stage_c_estimate": None if not overlap_bc else pearson_corr(
                [x for x, _ in overlap_bc], [y for _, y in overlap_bc]
            ),
            "stage_c_estimate_vs_stage_d_full": None if not overlap_cd else pearson_corr(
                [x for x, _ in overlap_cd], [y for _, y in overlap_cd]
            ),
        },
    }


def build_family_summary(family_stats: dict[str, dict[str, float]]) -> dict[str, Any]:
    out = {}
    for family, stats in family_stats.items():
        created = stats.get("created", 0.0)
        stage_a_done = stats.get("stage_a_done", 0.0)
        stage_a_pass = stats.get("stage_a_pass", 0.0)
        stage_b_done = stats.get("stage_b_done", 0.0)
        stage_b_pass = stats.get("stage_b_pass", 0.0)
        stage_c_done = stats.get("stage_c_done", 0.0)
        stage_d_done = stats.get("stage_d_done", 0.0)
        out[family] = {
            **stats,
            "rates": {
                "stage_a_done_per_created": safe_rate(stage_a_done, created),
                "stage_a_pass_per_done": safe_rate(stage_a_pass, stage_a_done),
                "stage_b_pass_per_done": safe_rate(stage_b_pass, stage_b_done),
                "stage_c_done_per_stage_b_pass": safe_rate(stage_c_done, stage_b_pass),
                "stage_d_done_per_stage_c_done": safe_rate(stage_d_done, stage_c_done),
            },
        }
    return out


def current_best_summary(records: dict[int, CandidateRecord]) -> dict[str, Any]:
    best_full = top_records(records, limit=1, predicate=lambda rec: rec.stage_d_metrics is not None)
    best_timed = top_records(records, limit=1, predicate=lambda rec: rec.stage_c_metrics is not None)
    best_proxy = top_records(records, limit=1, predicate=lambda rec: rec.stage_b_proxy is not None and rec.stage_a_violations == 0)
    best_ck4 = top_records(records, limit=1, predicate=lambda rec: rec.stage_a_violations is not None)
    return {
        "best_full": None if not best_full else {
            "candidate_id": best_full[0].candidate_id,
            "taxi_violations": best_full[0].stage_d_metrics["violations_k"],
        },
        "best_timed": None if not best_timed else {
            "candidate_id": best_timed[0].candidate_id,
            "estimated_total_violations": best_timed[0].stage_c_metrics["estimated_total_violations"],
        },
        "best_proxy": None if not best_proxy else {
            "candidate_id": best_proxy[0].candidate_id,
            "taxi_proxy_violations": best_proxy[0].stage_b_proxy,
        },
        "best_ck4": None if not best_ck4 else {
            "candidate_id": best_ck4[0].candidate_id,
            "ck4_violations": best_ck4[0].stage_a_violations,
        },
    }


def write_summary(
    config: SearchConfig,
    records: dict[int, CandidateRecord],
    stage_stats: dict[str, StageStats],
    family_stats: dict[str, dict[str, float]],
    adaptive_edge_idxes: list[int],
    adaptive_edge_bucket_sizes: dict[str, int],
    started_at: float,
    latest_path: Path,
    queue_snapshot: dict[str, int] | None = None,
) -> None:
    top_full = top_records(records, limit=25, predicate=lambda rec: rec.stage_d_metrics is not None)
    top_timed = top_records(records, limit=50, predicate=lambda rec: rec.stage_c_metrics is not None)
    top_proxy = top_records(records, limit=50, predicate=lambda rec: rec.stage_b_proxy is not None)
    funnel = build_funnel_summary(records)
    summary = {
        "elapsed_s": time.time() - started_at,
        "matrix": MATRIX,
        "coef_values": list(VALUE_SET),
        "frozen_zero_idxs": list(FROZEN_ZERO_IDXS),
        "config": {
            "timeout_s": config.timeout_s,
            "n_cpus": config.n_cpus,
            "final_full_taxi_budget": config.final_full_taxi_budget,
        },
        "adaptive_edge_cache_size": len(adaptive_edge_idxes),
        "adaptive_edge_buckets": adaptive_edge_bucket_sizes,
        "queue_snapshot": queue_snapshot or {},
        "candidate_counts": {
            "unique_candidates": len(records),
        },
        "stage_stats": {stage: summarize_stage(stats) for stage, stats in stage_stats.items()},
        "funnel": funnel,
        "family_stats": build_family_summary(family_stats),
        "top_stage_d": [
            {
                "candidate_id": rec.candidate_id,
                "coefs": list(rec.coefs),
                "taxi_violations": rec.stage_d_metrics["violations_k"],
                "taxi_proxy_violations": rec.stage_b_proxy,
                "timed_estimated_total_violations": None if rec.stage_c_metrics is None else rec.stage_c_metrics["estimated_total_violations"],
            }
            for rec in top_full
        ],
        "top_stage_c": [
            {
                "candidate_id": rec.candidate_id,
                "coefs": list(rec.coefs),
                **rec.stage_c_metrics,
                "taxi_proxy_violations": rec.stage_b_proxy,
            }
            for rec in top_timed
        ],
        "top_stage_b": [
            {
                "candidate_id": rec.candidate_id,
                "coefs": list(rec.coefs),
                "ck4_violations": rec.stage_a_violations,
                "taxi_proxy_violations": rec.stage_b_proxy,
            }
            for rec in top_proxy
        ],
    }
    latest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_worker_test() -> None:
    set_log_path(ROOT / "n7_async_worker_test.log")
    log("running worker self-test")
    expected_mod, expected_context, expected_distances, expected_B, expected_edges = build_ck4_cache()
    test_vectors = [
        tuple([0] * N_COEFS),
        SEED_VECTORS[0],
        SEED_VECTORS[1],
        (0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1),
    ]
    expected = {
        tuple(coefs): evaluate_ck4_cached(
            expected_mod,
            expected_context,
            expected_distances,
            expected_B,
            expected_edges,
            tuple(coefs),
        )
        for coefs in test_vectors
    }

    fast_task_queue: mp.Queue = mp.Queue()
    fast_result_queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=fast_worker_loop, args=(fast_task_queue, fast_result_queue))
    proc.start()

    for idx, coefs in enumerate(test_vectors, start=1):
        fast_task_queue.put(
            {
                "task_id": idx,
                "candidate_id": idx,
                "stage": "stage_a",
                "coefs": list(coefs),
                "family": "worker_test",
                "parent_id": None,
                "submitted_at": time.time(),
            }
        )

    seen = {}
    deadline = time.time() + 300
    while len(seen) < len(test_vectors):
        timeout = max(0.1, deadline - time.time())
        if timeout <= 0:
            raise RuntimeError("worker self-test timed out")
        result = fast_result_queue.get(timeout=timeout)
        if not result["ok"]:
            raise RuntimeError(f"worker self-test failed: {result}")
        seen[result["candidate_id"]] = result["payload"]["ck4_violations"]
        log(
            f"worker result candidate={result['candidate_id']} "
            f"ck4={result['payload']['ck4_violations']} "
            f"duration={result['finished_at']-result['started_at']:.3f}s"
        )

    fast_task_queue.put(None)
    proc.join(timeout=10.0)
    mismatches = []
    for idx, coefs in enumerate(test_vectors, start=1):
        got = seen[idx]
        want = expected[tuple(coefs)]
        if got != want:
            mismatches.append({"candidate_id": idx, "got": got, "want": want, "coefs": list(coefs)})

    payload = {
        "test": "worker",
        "passed": not mismatches,
        "vectors_tested": len(test_vectors),
        "mismatches": mismatches,
        "expected": [{"coefs": list(coefs), "ck4_violations": score} for coefs, score in expected.items()],
    }
    out = ROOT / "n7_async_worker_test.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"worker self-test wrote {out}")
    print(out)


def run_stage_a_controller_test(timeout_s: float, n_cpus: int, tag: str) -> None:
    set_log_path(ROOT / f"n7_async_stagea_{tag}.log")
    log("running stage_a-only controller test")
    out_base = ROOT / f"n7_async_stagea_{tag}"
    output_path = out_base.with_suffix(".json")
    event_log_path = out_base.with_suffix(".events.jsonl")
    checkpoint_path = out_base.with_suffix(".checkpoint.json")

    fast_task_queue: mp.Queue = mp.Queue()
    fast_result_queue: mp.Queue = mp.Queue()
    workers = []
    for _ in range(max(1, n_cpus)):
        proc = mp.Process(target=fast_worker_loop, args=(fast_task_queue, fast_result_queue))
        proc.start()
        workers.append(proc)

    rng = random.Random(20260401)
    started_at = time.time()
    deadline = started_at + timeout_s
    next_task_id = 1
    inflight = {}
    submitted = 0
    completed = 0
    ck4_zero = 0
    best = None

    def submit(coefs: tuple[int, ...]):
        nonlocal next_task_id, submitted
        inflight[next_task_id] = {"coefs": coefs, "submitted_at": time.time()}
        fast_task_queue.put(
            {
                "task_id": next_task_id,
                "candidate_id": next_task_id,
                "stage": "stage_a",
                "coefs": list(coefs),
                "family": "stagea_test",
                "parent_id": None,
                "submitted_at": inflight[next_task_id]["submitted_at"],
            }
        )
        next_task_id += 1
        submitted += 1

    while time.time() < deadline and len(inflight) < max(4, n_cpus * 4):
        submit(mutate_vector(tuple([0] * N_COEFS), rng, "seed_sparse"))

    while time.time() < deadline or inflight:
        while time.time() < deadline and len(inflight) < max(4, n_cpus * 4):
            submit(mutate_vector(tuple([0] * N_COEFS), rng, "seed_sparse"))
        try:
            result = fast_result_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        task = inflight.pop(result["task_id"], None)
        if task is None:
            continue
        completed += 1
        ck4 = result["payload"]["ck4_violations"] if result["ok"] else None
        if ck4 == 0:
            ck4_zero += 1
        if best is None or (ck4 is not None and ck4 < best["ck4_violations"]):
            best = {"coefs": list(task["coefs"]), "ck4_violations": ck4}
        append_event(event_log_path, {"ts": time.time(), "event": "stage_a_result", "ck4_violations": ck4, "coefs": list(task["coefs"])})

    for _ in workers:
        fast_task_queue.put(None)
    for proc in workers:
        proc.join(timeout=10.0)

    elapsed = time.time() - started_at
    payload = {
        "test": "stage_a_controller",
        "elapsed_s": elapsed,
        "submitted": submitted,
        "completed": completed,
        "throughput_candidates_per_sec": completed / max(elapsed, 1e-9),
        "ck4_zero_count": ck4_zero,
        "best": best,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"stage_a-only controller test wrote {output_path}")
    print(output_path)


def run_stage_b_test(n_cpus: int, tag: str) -> None:
    set_log_path(ROOT / f"n7_async_stageb_{tag}.log")
    log("running stage_b-only proxy test")
    out_base = ROOT / f"n7_async_stageb_{tag}"
    output_path = out_base.with_suffix(".json")
    fast_task_queue: mp.Queue = mp.Queue()
    fast_result_queue: mp.Queue = mp.Queue()
    workers = []
    for _ in range(max(1, n_cpus)):
        proc = mp.Process(target=fast_worker_loop, args=(fast_task_queue, fast_result_queue))
        proc.start()
        workers.append(proc)

    adaptive_edge_idxes = build_baseline_taxi_edge_idxes()
    started_at = time.time()
    submitted = 0
    completed = 0
    results = []
    test_vectors = SEED_VECTORS + [
        (0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1),
    ]
    for idx, coefs in enumerate(test_vectors, start=1):
        fast_task_queue.put(
            {
                "task_id": idx,
                "candidate_id": idx,
                "stage": "stage_b",
                "coefs": list(coefs),
                "family": "stageb_test",
                "parent_id": None,
                "submitted_at": time.time(),
                "edge_idxes": list(adaptive_edge_idxes),
                "cutoff": STAGE_B_PROXY_THRESHOLD,
            }
        )
        submitted += 1

    while completed < submitted:
        result = fast_result_queue.get(timeout=600)
        if not result["ok"]:
            raise RuntimeError(f"stage_b test failed: {result}")
        payload = result["payload"]
        completed += 1
        results.append(
            {
                "candidate_id": result["candidate_id"],
                "coefs": list(test_vectors[result["candidate_id"] - 1]),
                **payload,
                "duration_s": result["finished_at"] - result["started_at"],
            }
        )
        log(
            f"stage_b result candidate={result['candidate_id']} proxy={payload['taxi_proxy_violations']} "
            f"edges={payload['edges_processed']}/{payload['edges_total']} "
            f"duration={result['finished_at']-result['started_at']:.3f}s"
        )

    for _ in workers:
        fast_task_queue.put(None)
    for proc in workers:
        proc.join(timeout=10.0)

    elapsed = time.time() - started_at
    payload = {
        "test": "stage_b_proxy",
        "elapsed_s": elapsed,
        "submitted": submitted,
        "completed": completed,
        "throughput_candidates_per_sec": completed / max(elapsed, 1e-9),
        "adaptive_edge_cache_size": len(adaptive_edge_idxes),
        "results": sorted(results, key=lambda item: (item["taxi_proxy_violations"], item["candidate_id"])),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"stage_b-only proxy test wrote {output_path}")
    print(output_path)


def run_stage_c_test(n_cpus: int, tag: str) -> None:
    set_log_path(ROOT / f"n7_async_stagec_{tag}.log")
    log("running stage_c-only timed proxy test")
    out_base = ROOT / f"n7_async_stagec_{tag}"
    output_path = out_base.with_suffix(".json")
    heavy_result_queue: mp.Queue = mp.Queue()
    test_vectors = [
        (0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1),
        SEED_VECTORS[0],
        SEED_VECTORS[1],
    ]
    started_at = time.time()
    results = []
    completed = 0
    for idx, coefs in enumerate(test_vectors, start=1):
        submitted_at = time.time()
        proc = mp.Process(
            target=heavy_stage_process,
            args=("stage_c", idx, tuple(coefs), max(1, int(n_cpus)), heavy_result_queue),
        )
        proc.start()
        result = heavy_result_queue.get(timeout=3600)
        completed += 1
        if not result["ok"]:
            proc.join(timeout=10.0)
            raise RuntimeError(f"stage_c test failed: {result}")
        payload = result["payload"]
        results.append(
            {
                "candidate_id": idx,
                "coefs": list(coefs),
                **payload,
                "wall_time_s": result["finished_at"] - submitted_at,
            }
        )
        log(
            f"stage_c result candidate={idx} est={payload['estimated_total_violations']:.3f} "
            f"processed={payload['processed_normalized_edges_score']:.4f} "
            f"cache_time={payload['cache_time']:.3f}s"
        )
        proc.join(timeout=10.0)

    elapsed = time.time() - started_at
    payload = {
        "test": "stage_c_timed_proxy",
        "elapsed_s": elapsed,
        "submitted": len(test_vectors),
        "completed": completed,
        "throughput_candidates_per_sec": completed / max(elapsed, 1e-9),
        "results": sorted(results, key=lambda item: (item["estimated_total_violations"], item["candidate_id"])),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"stage_c-only timed proxy test wrote {output_path}")
    print(output_path)


def run_stage_d_test(n_cpus: int, tag: str) -> None:
    set_log_path(ROOT / f"n7_async_staged_{tag}.log")
    log("running stage_d-only full taxi test")
    out_base = ROOT / f"n7_async_staged_{tag}"
    output_path = out_base.with_suffix(".json")
    heavy_result_queue: mp.Queue = mp.Queue()
    test_vectors = [
        SEED_VECTORS[0],
        SEED_VECTORS[1],
        (0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1),
    ]
    started_at = time.time()
    results = []
    completed = 0
    for idx, coefs in enumerate(test_vectors, start=1):
        submitted_at = time.time()
        proc = mp.Process(
            target=heavy_stage_process,
            args=("stage_d", idx, tuple(coefs), max(1, int(n_cpus)), heavy_result_queue),
        )
        proc.start()
        result = heavy_result_queue.get(timeout=7200)
        completed += 1
        if not result["ok"]:
            proc.join(timeout=10.0)
            raise RuntimeError(f"stage_d test failed: {result}")
        payload = result["payload"]
        results.append(
            {
                "candidate_id": idx,
                "coefs": list(coefs),
                **payload,
                "wall_time_s": result["finished_at"] - submitted_at,
            }
        )
        log(
            f"stage_d result candidate={idx} taxi={payload['violations_k']} "
            f"cache_time={payload['cache_time']:.3f}s "
            f"wall={result['finished_at']-submitted_at:.3f}s"
        )
        proc.join(timeout=10.0)

    elapsed = time.time() - started_at
    payload = {
        "test": "stage_d_full_taxi",
        "elapsed_s": elapsed,
        "submitted": len(test_vectors),
        "completed": completed,
        "throughput_candidates_per_sec": completed / max(elapsed, 1e-9),
        "results": sorted(results, key=lambda item: (item["violations_k"], item["candidate_id"])),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"stage_d-only full taxi test wrote {output_path}")
    print(output_path)


def choose_parent(rng: random.Random, records: dict[int, CandidateRecord], family: str) -> tuple[int | None, tuple[int, ...]]:
    if family == "seed_sparse":
        return None, tuple([0] * N_COEFS)
    if family == "seed_dense":
        return None, tuple([0] * N_COEFS)
    if family == "mutate_full":
        pool = top_records(records, limit=12, predicate=lambda rec: rec.stage_d_metrics is not None)
    elif family == "mutate_timed":
        pool = top_records(records, limit=20, predicate=lambda rec: rec.stage_c_metrics is not None)
    elif family == "mutate_proxy":
        pool = top_records(records, limit=30, predicate=lambda rec: rec.stage_b_proxy is not None and rec.stage_a_violations == 0)
    else:
        pool = top_records(records, limit=30, predicate=lambda rec: rec.stage_a_violations == 0)
    if not pool:
        return None, tuple([0] * N_COEFS)
    chosen = rng.choice(pool)
    return chosen.candidate_id, chosen.coefs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=4 * 3600)
    parser.add_argument("--n-cpus", type=int, default=20)
    parser.add_argument("--tag", type=str, default="n7-async")
    parser.add_argument(
        "--families",
        type=str,
        default="",
        help="Comma-separated family override, e.g. seed_dense or seed_sparse,seed_dense",
    )
    parser.add_argument(
        "--disable-startup-seeds",
        action="store_true",
        help="Do not enqueue the hard-coded SEED_VECTORS at startup.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=("worker-test", "stage-a-test", "stage-b-test", "stage-c-test", "stage-d-test", "full"),
    )
    args = parser.parse_args()

    if args.mode == "worker-test":
        run_worker_test()
        return
    if args.mode == "stage-a-test":
        run_stage_a_controller_test(timeout_s=float(args.timeout), n_cpus=max(1, int(args.n_cpus)), tag=args.tag)
        return
    if args.mode == "stage-b-test":
        run_stage_b_test(n_cpus=max(1, int(args.n_cpus)), tag=args.tag)
        return
    if args.mode == "stage-c-test":
        run_stage_c_test(n_cpus=max(1, int(args.n_cpus)), tag=args.tag)
        return
    if args.mode == "stage-d-test":
        run_stage_d_test(n_cpus=max(1, int(args.n_cpus)), tag=args.tag)
        return

    out_base = ROOT / f"n7_async_pipeline_{args.tag}"
    set_log_path(out_base.with_suffix(".log"))
    output_path = out_base.with_suffix(".json")
    checkpoint_path = out_base.with_suffix(".checkpoint.json")
    event_log_path = out_base.with_suffix(".events.jsonl")
    config = SearchConfig(
        timeout_s=float(args.timeout),
        n_cpus=max(1, int(args.n_cpus)),
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        event_log_path=event_log_path,
    )

    random.seed(config.seed)
    rng = random.Random(config.seed)

    adaptive_edge_idxes = build_baseline_taxi_edge_idxes()
    core_edge_idxes = list(adaptive_edge_idxes)
    adaptive_recent_last_seen: dict[int, int] = {}
    adaptive_frequent_counts: dict[int, int] = defaultdict(int)
    adaptive_disagreement_counts: dict[int, int] = defaultdict(int)
    adaptive_frontier_counts: dict[int, int] = defaultdict(int)
    adaptive_edge_bucket_sizes = {
        "core": min(len(core_edge_idxes), CORE_EDGE_BUDGET),
        "recent_pool": 0,
        "frequent_pool": 0,
        "disagreement_pool": 0,
        "frontier_pool": 0,
        "selected_total": len(adaptive_edge_idxes),
    }
    adaptive_step = 0
    taxi_mod = load_main_module()
    taxi_inst = taxi_mod.NumpyKServerInstance.load(TAXI_METRIC)
    fast_task_queue: mp.Queue = mp.Queue()
    fast_result_queue: mp.Queue = mp.Queue()
    heavy_result_queue: mp.Queue = mp.Queue()

    fast_workers: list[mp.Process] = []
    desired_fast_workers = FAST_WORKERS_TOTAL
    heavy_workers: dict[str, dict[str, Any]] = {}

    records: dict[int, CandidateRecord] = {}
    coefs_to_id: dict[tuple[int, ...], int] = {}
    family_rewards: dict[str, float] = defaultdict(lambda: 1.0)
    family_stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    stage_stats = {
        "stage_a": StageStats(),
        "stage_b": StageStats(),
        "stage_c": StageStats(),
        "stage_d": StageStats(),
    }
    pending_stage_b: deque[int] = deque()
    pending_stage_c: deque[int] = deque()
    pending_stage_d: deque[int] = deque()
    inflight_fast: dict[int, dict[str, Any]] = {}
    next_candidate_id = 1
    next_task_id = 1
    last_checkpoint = 0.0
    stage_d_unique_edge_gate = STAGE_D_UNIQUE_EDGE_GATE_INITIAL

    def spawn_fast_worker():
        proc = mp.Process(target=fast_worker_loop, args=(fast_task_queue, fast_result_queue))
        proc.start()
        fast_workers.append(proc)

    def set_fast_workers(target: int):
        nonlocal desired_fast_workers
        desired_fast_workers = max(0, target)
        while len(fast_workers) < desired_fast_workers:
            spawn_fast_worker()
            log(f"spawned fast worker count={len(fast_workers)} target={desired_fast_workers}")
        while len(fast_workers) > desired_fast_workers:
            fast_task_queue.put(None)
            fast_workers.pop()
            log(f"retired fast worker count={len(fast_workers)} target={desired_fast_workers}")

    def ensure_candidate(coefs: tuple[int, ...], family: str, parent_id: int | None) -> int:
        nonlocal next_candidate_id
        if coefs in coefs_to_id:
            return coefs_to_id[coefs]
        candidate_id = next_candidate_id
        next_candidate_id += 1
        rec = CandidateRecord(
            candidate_id=candidate_id,
            coefs=coefs,
            family=family,
            parent_id=parent_id,
            created_at=time.time(),
        )
        records[candidate_id] = rec
        coefs_to_id[coefs] = candidate_id
        append_event(config.event_log_path, {"ts": time.time(), "event": "candidate_created", "candidate_id": candidate_id, "family": family, "parent_id": parent_id, "coefs": list(coefs)})
        family_stats[family]["created"] += 1
        return candidate_id

    def credit_family(candidate_id: int, amount: float):
        decay = 1.0
        current = records.get(candidate_id)
        while current is not None and decay > 0.1:
            family_rewards[current.family] += amount * decay
            family_stats[current.family]["reward"] += amount * decay
            current = records.get(current.parent_id) if current.parent_id is not None else None
            decay *= 0.5

    def refresh_stage_b_cache():
        nonlocal adaptive_edge_idxes, adaptive_edge_bucket_sizes
        adaptive_edge_idxes, adaptive_edge_bucket_sizes = rebuild_stage_b_edge_cache(
            core_edge_idxes,
            adaptive_recent_last_seen,
            dict(adaptive_frequent_counts),
            dict(adaptive_disagreement_counts),
            dict(adaptive_frontier_counts),
        )

    def update_edge_bucket(edge_idxes: list[int] | None, bucket: str):
        nonlocal adaptive_step
        if not edge_idxes:
            return
        adaptive_step += 1
        for edge_idx in edge_idxes:
            adaptive_recent_last_seen[int(edge_idx)] = adaptive_step
            adaptive_frequent_counts[int(edge_idx)] += 1
            if bucket == "disagreement":
                adaptive_disagreement_counts[int(edge_idx)] += 1
            elif bucket == "frontier":
                adaptive_frontier_counts[int(edge_idx)] += 1
        refresh_stage_b_cache()

    def maybe_collect_full_taxi_edges(rec: CandidateRecord, bucket: str):
        kwargs = {"n": 7, "index_matrix": MATRIX, "coefs": list(rec.coefs)}
        edge_idxes = collect_violated_edge_idxes(
            taxi_mod,
            taxi_inst,
            kwargs,
            cutoff=STAGE_EDGE_COLLECT_LIMIT,
        )
        update_edge_bucket(edge_idxes, bucket)

    def queue_fast_task(stage: str, candidate_id: int, **payload):
        nonlocal next_task_id
        rec = records[candidate_id]
        task = {
            "task_id": next_task_id,
            "candidate_id": candidate_id,
            "stage": stage,
            "coefs": list(rec.coefs),
            "family": rec.family,
            "parent_id": rec.parent_id,
            "submitted_at": time.time(),
            **payload,
        }
        next_task_id += 1
        inflight_fast[task["task_id"]] = task
        fast_task_queue.put(task)
        stage_stats[stage].submitted += 1
        if stage == "stage_b":
            log(
                f"queued stage_b candidate={candidate_id} inflight_fast={len(inflight_fast)} "
                f"edge_count={len(payload.get('edge_idxes', []))}"
            )

    def spawn_heavy_worker(stage: str, worker_id: str, n_cpus: int):
        task_queue: mp.Queue = mp.Queue()
        proc = mp.Process(
            target=heavy_worker_loop,
            args=(stage, worker_id, min(n_cpus, config.n_cpus), task_queue, heavy_result_queue),
        )
        proc.start()
        heavy_workers[worker_id] = {
            "worker_id": worker_id,
            "process": proc,
            "queue": task_queue,
            "stage": stage,
            "n_cpus": min(n_cpus, config.n_cpus),
            "busy": False,
            "candidate_id": None,
            "submitted_at": None,
            "started_at": None,
            "timeout_s": None,
        }
        log(f"spawned heavy worker worker_id={worker_id} stage={stage} n_cpus={min(n_cpus, config.n_cpus)} pid={proc.pid}")

    def idle_heavy_workers(stage: str) -> list[dict[str, Any]]:
        return [info for info in heavy_workers.values() if info["stage"] == stage and not info["busy"] and info["process"].is_alive()]

    def active_heavy_jobs(stage: str | None = None) -> int:
        return sum(1 for info in heavy_workers.values() if info["busy"] and (stage is None or info["stage"] == stage))

    def dispatch_heavy(stage: str, candidate_id: int, worker: dict[str, Any]) -> bool:
        rec = records[candidate_id]
        stage_timeout_s = None
        if stage == "stage_c":
            proxy = rec.stage_b_proxy if rec.stage_b_proxy is not None else STAGE_B_PROXY_THRESHOLD
            proxy = max(0, min(STAGE_B_PROXY_THRESHOLD, int(proxy)))
            stage_timeout_s = STAGE_C_TIMEOUT_MIN_SECONDS + (
                (STAGE_B_PROXY_THRESHOLD - proxy)
                * (STAGE_C_TIMEOUT_SECONDS - STAGE_C_TIMEOUT_MIN_SECONDS)
                / STAGE_B_PROXY_THRESHOLD
            )
        worker["queue"].put(
            {
                "candidate_id": candidate_id,
                "coefs": list(rec.coefs),
                "stage_timeout_s": stage_timeout_s,
            }
        )
        worker["busy"] = True
        worker["candidate_id"] = candidate_id
        worker["submitted_at"] = time.time()
        worker["started_at"] = worker["submitted_at"]
        worker["timeout_s"] = stage_timeout_s
        stage_stats[stage].submitted += 1
        log(
            f"dispatched {stage} candidate={candidate_id} worker_id={worker['worker_id']} "
            f"n_cpus={worker['n_cpus']}"
            + ("" if stage_timeout_s is None else f" timeout_s={stage_timeout_s:.1f}")
        )
        return True

    def refresh_worker_budget():
        set_fast_workers(FAST_WORKERS_TOTAL)

    def admit_stage_c_candidate(candidate_id: int) -> tuple[bool, int | None]:
        rec = records[candidate_id]
        if rec.stage_b_proxy == 0:
            if len(pending_stage_c) < STAGE_C_MAX_PENDING:
                pending_stage_c.append(candidate_id)
                return True, None
            nonzero_pending = [cid for cid in pending_stage_c if (records[cid].stage_b_proxy or 10**9) > 0]
            if nonzero_pending:
                worst_cid = max(
                    nonzero_pending,
                    key=lambda cid: (
                        records[cid].stage_b_proxy if records[cid].stage_b_proxy is not None else 10**9,
                        sparsity_stats(records[cid].coefs)[0],
                        sparsity_stats(records[cid].coefs)[1],
                        cid,
                    ),
                )
                pending_stage_c.remove(worst_cid)
                pending_stage_c.append(candidate_id)
                return True, worst_cid
            pending_stage_c.append(candidate_id)
            return True, None
        if len(pending_stage_c) < STAGE_C_MAX_PENDING:
            pending_stage_c.append(candidate_id)
            return True, None
        return False, None

    def stage_c_priority_key(candidate_id: int) -> tuple[Any, ...]:
        rec = records[candidate_id]
        proxy = rec.stage_b_proxy if rec.stage_b_proxy is not None else 10**9
        l0, l1 = sparsity_stats(rec.coefs)
        return (proxy, l0, l1, candidate_id)

    def stage_d_priority_key(candidate_id: int) -> tuple[Any, ...]:
        rec = records[candidate_id]
        proxy = rec.stage_b_proxy if rec.stage_b_proxy is not None else 10**9
        l0, l1 = sparsity_stats(rec.coefs)
        timed = None if rec.stage_c_metrics is None else rec.stage_c_metrics.get("estimated_total_violations")
        timed_key = float("inf") if timed is None else float(timed)
        return (timed_key, proxy, l0, l1, candidate_id)

    set_fast_workers(FAST_WORKERS_TOTAL)
    spawn_heavy_worker("stage_c", "stage_c_0", STAGE_C_MAX_CPUS)
    spawn_heavy_worker("stage_c", "stage_c_1", STAGE_C_MAX_CPUS)
    spawn_heavy_worker("stage_d", "stage_d_0", STAGE_D_MIN_CPUS)

    default_families = ["seed_sparse", "seed_dense", "mutate_ck4", "mutate_proxy", "mutate_timed", "mutate_full"]
    families = [f.strip() for f in args.families.split(",") if f.strip()] if args.families else default_families
    seed_candidate_ids: list[int] = []
    if not args.disable_startup_seeds:
        for seed in SEED_VECTORS:
            seed_candidate_ids.append(ensure_candidate(tuple(seed), "seed_sparse", None))

    started_at = time.time()
    deadline = started_at + config.timeout_s
    log(
        f"controller started timeout_s={config.timeout_s} n_cpus={config.n_cpus} "
        f"adaptive_edge_cache={len(adaptive_edge_idxes)} seed_candidates={len(records)} "
        f"startup_seeds_enabled={not args.disable_startup_seeds}"
    )

    for candidate_id in seed_candidate_ids:
        if records[candidate_id].stage_a_violations is None:
            queue_fast_task("stage_a", candidate_id)
    if seed_candidate_ids:
        log(f"queued initial seed candidates count={len(seed_candidate_ids)}")

    while time.time() < deadline or inflight_fast or pending_stage_b or pending_stage_c or pending_stage_d or active_heavy_jobs():
        while True:
            try:
                result = fast_result_queue.get_nowait()
            except queue.Empty:
                break
            task = inflight_fast.pop(result["task_id"], None)
            if task is None:
                continue
            stage = result["worker_stage"]
            stats = stage_stats[stage]
            stats.completed += int(result["ok"])
            stats.failed += int(not result["ok"])
            stats.wait_times.append(max(0.0, result["started_at"] - task["submitted_at"]))
            stats.durations.append(max(0.0, result["finished_at"] - result["started_at"]))
            rec = records[result["candidate_id"]]
            if not result["ok"]:
                append_event(config.event_log_path, {"ts": time.time(), "event": "stage_failed", "stage": stage, "candidate_id": rec.candidate_id, "error": result["error"]})
                continue
            payload = result["payload"]
            if stage == "stage_a":
                rec.stage_a_violations = payload["ck4_violations"]
                family_stats[rec.family]["stage_a_done"] += 1
                if rec.stage_a_violations == 0:
                    family_stats[rec.family]["stage_a_pass"] += 1
                    credit_family(rec.candidate_id, 1.0)
                    pending_stage_b.append(rec.candidate_id)
                    stage_stats["stage_b"].queue_peak = max(stage_stats["stage_b"].queue_peak, len(pending_stage_b))
                    log(
                        f"stage_a pass candidate={rec.candidate_id} ck4=0 "
                        f"pending_b={len(pending_stage_b)}"
                    )
                append_event(config.event_log_path, {"ts": time.time(), "event": "stage_a_done", "candidate_id": rec.candidate_id, "ck4_violations": rec.stage_a_violations})
            elif stage == "stage_b":
                rec.stage_b_proxy = payload["taxi_proxy_violations"]
                rec.stage_b_edge_idxes = list(payload.get("violated_edge_idxes", []))
                family_stats[rec.family]["stage_b_done"] += 1
                update_edge_bucket(rec.stage_b_edge_idxes, "recent")
                if rec.stage_b_proxy <= STAGE_B_PROXY_THRESHOLD:
                    family_stats[rec.family]["stage_b_pass"] += 1
                    credit_family(rec.candidate_id, 2.0)
                    admitted, displaced_cid = admit_stage_c_candidate(rec.candidate_id)
                    if admitted:
                        stage_stats["stage_c"].queue_peak = max(stage_stats["stage_c"].queue_peak, len(pending_stage_c))
                        if displaced_cid is None:
                            log(
                                f"stage_b pass candidate={rec.candidate_id} proxy={rec.stage_b_proxy} "
                                f"pending_c={len(pending_stage_c)}"
                            )
                        else:
                            family_stats[records[displaced_cid].family]["stage_c_capped"] += 1
                            append_event(
                                config.event_log_path,
                                {
                                    "ts": time.time(),
                                    "event": "stage_c_evicted",
                                    "evicted_candidate_id": displaced_cid,
                                    "evicted_proxy": records[displaced_cid].stage_b_proxy,
                                    "admitted_candidate_id": rec.candidate_id,
                                    "admitted_proxy": rec.stage_b_proxy,
                                },
                            )
                            log(
                                f"stage_b pass candidate={rec.candidate_id} proxy={rec.stage_b_proxy} "
                                f"evicted_stage_c_candidate={displaced_cid} pending_c={len(pending_stage_c)}"
                            )
                    else:
                        family_stats[rec.family]["stage_c_capped"] += 1
                        append_event(config.event_log_path, {"ts": time.time(), "event": "stage_c_capped", "candidate_id": rec.candidate_id, "proxy": rec.stage_b_proxy})
                        log(
                            f"stage_b pass candidate={rec.candidate_id} proxy={rec.stage_b_proxy} "
                            f"skipped_stage_c_cap={STAGE_C_MAX_PENDING}"
                        )
                else:
                    log(f"stage_b reject candidate={rec.candidate_id} proxy={rec.stage_b_proxy}")
                append_event(config.event_log_path, {"ts": time.time(), "event": "stage_b_done", "candidate_id": rec.candidate_id, **payload})

        while True:
            try:
                result = heavy_result_queue.get_nowait()
            except queue.Empty:
                break
            stage = result["worker_stage"]
            candidate_id = result["candidate_id"]
            rec = records[candidate_id]
            stats = stage_stats[stage]
            stats.completed += int(result["ok"])
            stats.failed += int(not result["ok"])
            worker_id = result.get("worker_id")
            worker_info = heavy_workers.get(worker_id)
            if worker_info is not None:
                stats.wait_times.append(max(0.0, worker_info["started_at"] - worker_info["submitted_at"]))
                stats.durations.append(max(0.0, time.time() - worker_info["started_at"]))
                worker_info["busy"] = False
                worker_info["candidate_id"] = None
                worker_info["submitted_at"] = None
                worker_info["started_at"] = None
                worker_info["timeout_s"] = None
            payload = result.get("payload", {})
            if stage == "stage_c" and result["ok"]:
                rec.stage_c_metrics = payload
                family_stats[rec.family]["stage_c_done"] += 1
                credit_family(rec.candidate_id, max(0.5, 50.0 / max(1.0, payload["estimated_total_violations"])))
                update_edge_bucket(payload.get("violated_edge_idxes", []), "recent")
                if payload["estimated_total_violations"] <= STAGE_C_FRONTIER_ESTIMATE:
                    update_edge_bucket(rec.stage_b_edge_idxes, "frontier")
                    update_edge_bucket(payload.get("violated_edge_idxes", []), "frontier")
                elif payload["estimated_total_violations"] >= STAGE_C_DISAGREEMENT_ESTIMATE:
                    update_edge_bucket(rec.stage_b_edge_idxes, "disagreement")
                    update_edge_bucket(payload.get("violated_edge_idxes", []), "disagreement")
                stage_b_edge_set = set(rec.stage_b_edge_idxes or ())
                stage_c_edge_set = set(payload.get("violated_edge_idxes", []) or ())
                unique_union_edges = len(stage_b_edge_set | stage_c_edge_set)
                if unique_union_edges <= stage_d_unique_edge_gate and len(pending_stage_d) < STAGE_D_MAX_PENDING:
                    pending_stage_d.append(rec.candidate_id)
                    stage_stats["stage_d"].queue_peak = max(stage_stats["stage_d"].queue_peak, len(pending_stage_d))
                log(
                    f"stage_c done candidate={candidate_id} est={payload['estimated_total_violations']:.2f} "
                    f"processed={payload['processed_normalized_edges_score']:.4f} "
                    f"stage_bc_unique_edges={unique_union_edges} stage_d_gate={stage_d_unique_edge_gate} "
                    f"pending_d={len(pending_stage_d)}"
                )
                append_event(config.event_log_path, {"ts": time.time(), "event": "stage_c_done", "candidate_id": candidate_id, **payload})
            elif stage == "stage_d" and result["ok"]:
                rec.stage_d_metrics = payload
                family_stats[rec.family]["stage_d_done"] += 1
                credit_family(rec.candidate_id, max(1.0, 200.0 / max(1.0, payload["violations_k"])))
                stage_d_unique_edge_gate = max(
                    STAGE_D_UNIQUE_EDGE_GATE_MIN,
                    min(stage_d_unique_edge_gate, int(payload["violations_k"])),
                )
                update_edge_bucket(payload.get("violated_edge_idxes", []), "recent")
                if payload["violations_k"] <= STAGE_D_FRONTIER_FULL:
                    maybe_collect_full_taxi_edges(rec, "frontier")
                elif payload["violations_k"] >= STAGE_D_DISAGREEMENT_FULL:
                    maybe_collect_full_taxi_edges(rec, "disagreement")
                log(
                    f"stage_d done candidate={candidate_id} taxi={payload['violations_k']} "
                    f"stage_d_gate={stage_d_unique_edge_gate}"
                )
                append_event(config.event_log_path, {"ts": time.time(), "event": "stage_d_done", "candidate_id": candidate_id, **payload})
            else:
                log(f"{stage} failed candidate={candidate_id} error={result.get('error')}")
                append_event(config.event_log_path, {"ts": time.time(), "event": "stage_failed", "stage": stage, "candidate_id": candidate_id, "error": result.get("error")})

        for worker_id, info in list(heavy_workers.items()):
            proc = info["process"]
            if not proc.is_alive():
                proc.join(timeout=0.1)
                if not info["busy"]:
                    continue
                info["busy"] = False
                info["candidate_id"] = None
        refresh_worker_budget()

        inflight_fast_stage_b = sum(1 for task in inflight_fast.values() if task["stage"] == "stage_b")
        inflight_fast_stage_a = sum(1 for task in inflight_fast.values() if task["stage"] == "stage_a")
        total_fast_capacity = max(1, len(fast_workers) * config.stage_a_backlog_factor)
        reserved_stage_b_capacity = max(1, FAST_STAGE_B_WORKERS * config.stage_a_backlog_factor)

        # Stage B fast proxy jobs get reserved fast-worker capacity and launch first.
        while (
            pending_stage_b
            and inflight_fast_stage_b < reserved_stage_b_capacity
            and len(inflight_fast) < total_fast_capacity
        ):
            candidate_id = pending_stage_b.popleft()
            if records[candidate_id].stage_b_proxy is not None:
                continue
            queue_len = len(pending_stage_b)
            stage_stats["stage_b"].queue_peak = max(stage_stats["stage_b"].queue_peak, queue_len)
            queue_fast_task("stage_b", candidate_id, edge_idxes=list(adaptive_edge_idxes), cutoff=STAGE_B_PROXY_THRESHOLD)
            inflight_fast_stage_b += 1

        # Proposal / Stage A fill.
        stage_a_target = max(1, FAST_STAGE_A_WORKERS * config.stage_a_backlog_factor)
        while (
            time.time() < deadline
            and inflight_fast_stage_a < stage_a_target
            and len(inflight_fast) < total_fast_capacity
        ):
            family = choose_weighted_family(rng, family_rewards, families)
            parent_id, parent = choose_parent(rng, records, family)
            coefs = mutate_vector(parent, rng, family)
            candidate_id = ensure_candidate(coefs, family, parent_id)
            rec = records[candidate_id]
            if rec.stage_a_violations is not None:
                continue
            queue_fast_task("stage_a", candidate_id)
            stage_stats["stage_a"].queue_peak = max(stage_stats["stage_a"].queue_peak, len(inflight_fast))
            inflight_fast_stage_a += 1

        # Launch heavy jobs asynchronously based on best promoted candidates.
        active_stage_c_jobs = active_heavy_jobs("stage_c")
        active_stage_d_jobs = active_heavy_jobs("stage_d")

        shortlist = top_records(
            records,
            limit=config.final_full_taxi_budget,
            predicate=lambda rec: rec.stage_c_metrics is not None and rec.stage_d_metrics is None,
        )
        shortlist_ids = {rec.candidate_id for rec in shortlist}
        stage_d_candidates = [
            cid for cid in pending_stage_d
            if cid in shortlist_ids and records[cid].stage_d_metrics is None
        ]
        stage_d_candidates.sort(key=stage_d_priority_key)

        while stage_d_candidates and active_stage_d_jobs < 1:
            idle = idle_heavy_workers("stage_d")
            if not idle:
                break
            candidate_id = stage_d_candidates.pop(0)
            pending_stage_d.remove(candidate_id)
            dispatch_heavy("stage_d", candidate_id, idle[0])
            active_stage_d_jobs += 1
            break

        eligible_stage_c = [
            cid for cid in pending_stage_c
            if records[cid].stage_c_metrics is None and (records[cid].stage_b_proxy or 10**9) <= STAGE_C_TIMEOUT_TRIGGER
        ]
        eligible_stage_c.sort(key=stage_c_priority_key)
        while eligible_stage_c and active_stage_c_jobs < 2:
            idle = idle_heavy_workers("stage_c")
            if not idle:
                break
            cid = eligible_stage_c.pop(0)
            if cid not in pending_stage_c:
                continue
            pending_stage_c.remove(cid)
            dispatch_heavy("stage_c", cid, idle[0])
            active_stage_c_jobs += 1

        if time.time() - last_checkpoint >= config.checkpoint_interval_s:
            write_summary(
                config,
                records,
                stage_stats,
                family_stats,
                adaptive_edge_idxes,
                adaptive_edge_bucket_sizes,
                started_at,
                config.checkpoint_path,
                queue_snapshot={
                    "inflight_fast": len(inflight_fast),
                    "pending_stage_b": len(pending_stage_b),
                    "pending_stage_c": len(pending_stage_c),
                    "pending_stage_d": len(pending_stage_d),
                    "heavy_jobs": active_heavy_jobs(),
                    "fast_workers": len(fast_workers),
                },
            )
            best = current_best_summary(records)
            stage_snapshot = {
                stage: {
                    "done": stats.completed,
                    "fail": stats.failed,
                    "qpeak": stats.queue_peak,
                    "avg_dur": round(percentiles(stats.durations)["avg"], 3),
                }
                for stage, stats in stage_stats.items()
            }
            log(
                "status "
                f"elapsed={time.time()-started_at:.1f}s inflight_fast={len(inflight_fast)} "
                f"pending_b={len(pending_stage_b)} pending_c={len(pending_stage_c)} pending_d={len(pending_stage_d)} "
                f"heavy={active_heavy_jobs()} edge_buckets={json.dumps(adaptive_edge_bucket_sizes, sort_keys=True)} "
                f"stage_stats={json.dumps(stage_snapshot, sort_keys=True)} "
                f"best={json.dumps(best, sort_keys=True)}"
            )
            last_checkpoint = time.time()

        time.sleep(0.02)

    for _ in range(len(fast_workers)):
        fast_task_queue.put(None)
    for proc in fast_workers:
        proc.join(timeout=2.0)
    for info in heavy_workers.values():
        try:
            info["queue"].put(None)
        except Exception:
            pass
    for info in heavy_workers.values():
        info["process"].join(timeout=2.0)
        if info["process"].is_alive():
            info["process"].terminate()
            info["process"].join(timeout=1.0)

    write_summary(
        config,
        records,
        stage_stats,
        family_stats,
        adaptive_edge_idxes,
        adaptive_edge_bucket_sizes,
        started_at,
        config.output_path,
        queue_snapshot={
            "inflight_fast": len(inflight_fast),
            "pending_stage_b": len(pending_stage_b),
            "pending_stage_c": len(pending_stage_c),
            "pending_stage_d": len(pending_stage_d),
            "heavy_jobs": active_heavy_jobs(),
            "fast_workers": len(fast_workers),
        },
    )
    best = current_best_summary(records)
    log(f"finished elapsed={time.time()-started_at:.1f}s best={json.dumps(best, sort_keys=True)}")
    print(config.output_path)


if __name__ == "__main__":
    main()
