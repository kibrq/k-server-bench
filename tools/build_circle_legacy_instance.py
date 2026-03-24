#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import pickle
from pathlib import Path
from typing import Any, Dict
import wandb

import numpy as np

from kserver.context import WFContext, all_multicombinations, k_taxi_update
from kserver.graph import (
    create_circle_symmetry_hash_fn,
    create_main_gc_profile_hooks,
    create_main_timing_profile_hooks,
    create_progress_bar_hooks,
    create_worker_timing_profile_hooks,
    parallel_bfs_exploration,
)
from kserver.metrics.circle import Circle


def _merge_hook_maps(*hook_maps: Dict[str, list]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for hm in hook_maps:
        for name, ctors in hm.items():
            out.setdefault(name, []).extend(ctors)
    return out


def create_finished_hook(message: str = "finished") -> Dict[str, list]:
    def on_finished(_state):
        def hook(**kwargs):
            print(message)

        return hook

    return {"main.finished": [on_finished]}


def create_normalized_sha256_hash_no_metadata(context):
    import hashlib

    def hash_fn(wf):
        wf_ = np.asarray(wf)
        wf_ = wf_ - wf_.min()
        return hashlib.sha256(wf_.tobytes()).hexdigest(), {}

    return hash_fn


def build_hash_factory(include_k_taxi: bool, no_symmetry_hash: bool):
    if no_symmetry_hash:
        return create_normalized_sha256_hash_no_metadata

    rotation_step = 2 if include_k_taxi else 1

    def create_hash_fn(context):
        return create_circle_symmetry_hash_fn(context, rotation_step=rotation_step)

    return create_hash_fn


def edge_metrics_worker_hook(worker):
    def hook(uwf, vwf, transition_idx, edge_meta, **kwargs):
        uwf = np.asarray(uwf)
        vwf = np.asarray(vwf)
        d_min = int(vwf.min() - uwf.min())
        ext = int((vwf - uwf).max())
        edge_meta["request"] = int(transition_idx)
        edge_meta["d_min"] = d_min
        edge_meta["ext"] = ext
        edge_meta["weight"] = float((worker.k + 1) * d_min - ext)

    return hook


def build_initial_nodes(k: int, m: int, include_k_taxi: bool):
    if not include_k_taxi:
        return list(all_multicombinations(m, k=k))
    even_points = list(range(0, 2 * m, 2))
    return [tuple(cfg) for cfg in combinations_with_replacement(even_points, k)]


def build_transitions(m: int, include_k_taxi: bool):
    if not include_k_taxi:
        return [lambda context, wf, r=r: context.update_wf(wf, r) for r in range(m)]

    transitions = []
    even_points = list(range(0, 2 * m, 2))
    odd_points = list(range(1, 2 * m, 2))

    for t in even_points:
        transitions.append(lambda context, wf, t=t: context.update_wf(wf, t))
    for s in odd_points:
        for t in even_points:
            transitions.append(lambda context, wf, s=s, t=t: k_taxi_update(context, wf, (s, t)))
    return transitions


def build_legacy_payload(
    result: Dict[str, Any],
    k: int,
    m: int,
    include_k_taxi: bool,
    halve_work_functions: bool,
    distance_matrix: np.ndarray,
    transitions,
) -> Dict[str, Any]:
    node_bookkeeper = result["node_bookkeeper"]
    edge_bookkeeper = result["edge_bookkeeper"]

    nodes_in_order = list(node_bookkeeper.nodes.values())
    node_id: Dict[Any, int] = {node.hsh: idx for idx, node in enumerate(nodes_in_order)}

    nodes = []
    context = WFContext(k=k, distance_matrix=distance_matrix)
    wf_projection = None
    projected_size = None
    if halve_work_functions and include_k_taxi:
        base_context = WFContext(k=k, distance_matrix=Circle.discrete(m=m))
        projected_size = len(base_context._idx_to_config)
        wf_projection = {}
        for doubled_idx, cfg in enumerate(context._idx_to_config):
            if any((x % 2) != 0 for x in cfg):
                continue
            base_cfg = tuple(int(x // 2) for x in cfg)
            wf_projection[doubled_idx] = base_context.config_to_idx(base_cfg)
        if len(wf_projection) != projected_size:
            raise ValueError(
                f"even-config projection mismatch: got={len(wf_projection)} expected={projected_size}"
            )
    for idx, node in enumerate(nodes_in_order):
        requests = node.path.get_requests() if node.path is not None else ()
        depth = int(len(requests))
        wf = node.get_wf(context=context, transitions=transitions, cache=False)
        wf = np.asarray(wf)
        if wf_projection is not None:
            reduced = np.empty(projected_size, dtype=wf.dtype)
            for doubled_idx, base_idx in wf_projection.items():
                reduced[base_idx] = wf[doubled_idx]
            wf = reduced
        wf_norm = tuple((wf - wf.min()).tolist())
        nodes.append({"depth": depth, "wf_norm": wf_norm, "id": idx})

    edges = []
    for u, v, metadata in getattr(edge_bookkeeper, "edges", []):
        edges.append(
            {
                "request": int(metadata["request"]),
                "ext": int(metadata["ext"]),
                "d_min": int(metadata["d_min"]),
                "weight": float(metadata["weight"]),
                "from": int(node_id[u.hsh]),
                "to": int(node_id[v.hsh]),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "edge_count": int(len(edge_bookkeeper)),
        "k": int(k),
        "distance_matrix": np.asarray(distance_matrix),
        "bellman": [0.0] * len(nodes),
    }


class CountingOnlyEdgeBookkeeper:
    def __init__(self) -> None:
        self._count = 0

    def add(self, u, v, metadata=None) -> None:
        self._count += 1

    def __len__(self) -> int:
        return self._count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build legacy-format circle WF graph instance pickle.")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--ray-local", action="store_true", help="Start/attach local Ray instead of cluster default.")
    parser.add_argument("--include-k-taxi", action="store_true", help="Use 2*m circle with even standard requests and odd->even k-taxi requests.")
    parser.add_argument("--no-symmetry-hash", action="store_true", help="Disable circle symmetry hash and use plain normalized hash.")
    parser.add_argument("--profile-timing", action="store_true", help="Print periodic BFS timing profile.")
    parser.add_argument("--timing-log-every", type=int, default=1000, help="Print timing stats every N iterations.")
    parser.add_argument("--profile-gc", action="store_true", help="Print periodic GC and tracked-object census stats.")
    parser.add_argument("--gc-log-every", type=int, default=1000, help="Print GC stats every N main-loop iterations.")
    parser.add_argument(
        "--gc-type-sample-every",
        type=int,
        default=1000,
        help="Sample tracked-object type counts every N main-loop iterations.",
    )
    parser.add_argument("--expand-batch-size", type=int, default=1, help="Number of expand tasks grouped per worker submission.")
    parser.add_argument("--disable-gc-during-run", action="store_true", help="Disable Python cyclic GC during BFS run (re-enabled at end).")
    parser.add_argument(
        "--halve-work-functions",
        action="store_true",
        help="When used with --include-k-taxi, project node work-functions from 2*m even configs to base m config order before saving.",
    )
    args = parser.parse_args()
    if args.halve_work_functions and not args.include_k_taxi:
        print("[warn] --halve-work-functions is ignored unless --include-k-taxi is enabled.")

    wandb.init()

    metric_m = 2 * args.m if args.include_k_taxi else args.m
    distance_matrix = Circle.discrete(m=metric_m)
    initial_nodes = build_initial_nodes(args.k, args.m, args.include_k_taxi)
    transitions = build_transitions(args.m, args.include_k_taxi)

    worker_hooks = {"expand.after_transition": [edge_metrics_worker_hook]}
    main_hooks = create_progress_bar_hooks(desc=f"circle k={args.k} m={args.m}", unit="expand")
    main_hooks = _merge_hook_maps(main_hooks, create_finished_hook())
    if args.profile_timing:
        # worker_hooks = _merge_hook_maps(
        #     worker_hooks,
        #     create_worker_timing_profile_hooks(log_every=args.timing_log_every),
        # )
        def empty_print(*args, **kwargs):
            pass
        main_hooks = _merge_hook_maps(
            main_hooks,
            create_main_timing_profile_hooks(log_every=args.timing_log_every, wandb_log=True, print_fn=empty_print),
        )
    if args.profile_gc:
        main_hooks = _merge_hook_maps(
            main_hooks,
            create_main_gc_profile_hooks(
                log_every=args.gc_log_every,
                type_sample_every=args.gc_type_sample_every,
                wandb_log=True,
                print_fn=print,
            ),
        )
    create_hash_fn = build_hash_factory(args.include_k_taxi, args.no_symmetry_hash)

    ray_kwargs = {"address": "local"} if args.ray_local and args.n_workers > 1 else None
    result = parallel_bfs_exploration(
        k=args.k,
        distance_matrix=distance_matrix,
        n_workers=args.n_workers,
        initial_nodes=initial_nodes,
        transitions=transitions,
        create_hash_fn=create_hash_fn,
        worker_hook_constructors=worker_hooks,
        main_hook_constructors=main_hooks,
        return_wfs=True,
        return_paths=True,
        ray_kwargs=ray_kwargs,
        edge_bookkeeper_constructor=CountingOnlyEdgeBookkeeper,
        expand_batch_size=args.expand_batch_size,
        disable_gc_during_run=args.disable_gc_during_run,
        pool_backend="subprocess",
    )

    payload = build_legacy_payload(
        result,
        k=args.k,
        m=args.m,
        include_k_taxi=args.include_k_taxi,
        halve_work_functions=args.halve_work_functions,
        distance_matrix=distance_matrix,
        transitions=transitions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Wrote: {args.output}")
    print(f"nodes={len(payload['nodes'])} edges_materialized={len(payload['edges'])} edge_count={payload['edge_count']}")


if __name__ == "__main__":
    main()
