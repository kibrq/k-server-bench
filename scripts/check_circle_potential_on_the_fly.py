#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Dict

import numpy as np

from kserverclean.context import all_multicombinations, k_taxi_update
from kserverclean.graph import (
    NodeBookkeeper,
    create_circle_symmetry_hash_fn,
    create_main_gc_profile_hooks,
    create_main_timing_profile_hooks,
    create_normalized_sha256_hash_fn,
    create_progress_bar_hooks,
    create_worker_potential_hooks,
    create_worker_timing_profile_hooks,
    parallel_bfs_exploration,
)
from kserverclean.metrics.circle import Circle


class BloomNodeBookkeeper(NodeBookkeeper):
    def __init__(self, bloom_filter_size: int, bloom_filter_fp_rate: float, factor: float = 0.9):
        super().__init__()
        try:
            from rbloom import Bloom  # type: ignore
        except ImportError as exc:
            raise ImportError("BloomNodeBookkeeper requires `rbloom`. Install with `pip install rbloom`.") from exc
        self._Bloom = Bloom
        self.bloom_filter_size = int(bloom_filter_size)
        self.bloom_filter_fp_rate = float(bloom_filter_fp_rate)
        self.blooms = [self._Bloom(self.bloom_filter_size, self.bloom_filter_fp_rate)]
        self.last_update_size = 0
        self.factor = float(factor)
        self.n_nodes = 0

    def _add(self, uhash, **kwargs):
        if self.n_nodes - self.last_update_size > int(self.bloom_filter_size * self.factor):
            self.blooms.append(self._Bloom(self.bloom_filter_size, self.bloom_filter_fp_rate))
            self.last_update_size = self.n_nodes

        for bloom in self.blooms:
            if uhash in bloom:
                return False

        self.blooms[-1].add(uhash)
        return True

    def add(self, u) -> bool:
        if self._add(u.hsh):
            self.n_nodes += 1
            return True
        return False

    def __len__(self) -> int:
        return self.n_nodes


def _merge_hook_maps(*hook_maps: Dict[str, list]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for hm in hook_maps:
        for name, ctors in hm.items():
            out.setdefault(name, []).extend(ctors)
    return out


def _load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_potential_cls(potential_file: Path, potential_class: str):
    module = _load_module_from_path(potential_file)
    if not hasattr(module, potential_class):
        raise ValueError(f"Class '{potential_class}' not found in {potential_file}")
    return getattr(module, potential_class)


def _load_potential_kwargs(args) -> Dict[str, Any]:
    if args.potential_kwargs_file is not None:
        with args.potential_kwargs_file.open("r", encoding="utf-8") as f:
            return dict(json.load(f))
    if args.potential_kwargs_json is not None:
        return dict(json.loads(args.potential_kwargs_json))
    return {}


def edge_metrics_worker_hook(worker):
    def hook(uwf, vwf, transition_idx, edge_meta, **kwargs):
        uwf = np.asarray(uwf)
        vwf = np.asarray(vwf)
        d_min = float(vwf.min() - uwf.min())
        ext = float((vwf - uwf).max())
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


def build_hash_factory(include_k_taxi: bool, no_symmetry_hash: bool):
    if no_symmetry_hash:
        return create_normalized_sha256_hash_fn

    rotation_step = 2 if include_k_taxi else 1

    def create_hash_fn(context):
        return create_circle_symmetry_hash_fn(context, rotation_step=rotation_step)

    return create_hash_fn


def potential_inequality_main_hooks(rho):
    def on_start(state):
        def hook(**kwargs):
            state["_violations"] = 0
        return hook

    def on_edge_added(state):
        def hook(u, v, neighbor, **kwargs):
            if "potential" not in u.metadata or "potential" not in v.metadata:
                return
            pu = float(u.metadata["potential"])
            pv = float(v.metadata["potential"])
            d_min = float(neighbor.metadata["d_min"])
            ext = float(neighbor.metadata["ext"])
            slack = pv - pu + (rho + 1.0) * d_min - ext

            if slack < 0:
                state["_violations"] += 1
                u_initial = None
                u_requests = ()
                if u.path is not None:
                    u_initial = tuple(u.path.initial)
                    u_requests = u.path.get_requests()
                transition = neighbor.metadata.get("request")
                print(
                    f"[VIOLATION] req={neighbor.metadata.get('request')} "
                    f"u={u.hsh} v={v.hsh} slack={slack:.6f} "
                    f"(pv={pv:.6f}, pu={pu:.6f}, d_min={d_min:.6f}, ext={ext:.6f}) "
                    f"path_u_initial={u_initial} path_u_requests={u_requests} next_transition={transition}"
                )
        return hook

    def on_end(state):
        def hook(**kwargs):
            print(f"violations={state.get('_violations', 0)}")
        return hook

    return {
        "main.start": [on_start],
        "main.handle_expand.edge_added": [on_edge_added],
        "main.end": [on_end],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check potential inequality online during circle graph exploration.")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--ray-local", action="store_true")
    parser.add_argument("--include-k-taxi", action="store_true", help="Use 2*m circle with even standard requests and odd->even k-taxi requests.")
    parser.add_argument("--no-symmetry-hash", action="store_true", help="Disable circle symmetry hash and use plain normalized hash.")

    parser.add_argument("--potential-file", type=Path, required=True)
    parser.add_argument("--potential-class", type=str, default="Potential")
    parser.add_argument("--potential-kwargs-file", type=Path, default=None)
    parser.add_argument("--potential-kwargs-json", type=str, default=None)
    parser.add_argument("--rho", type=float, default=None, help="Inequality uses pv - pu + (rho+1)*d_min - ext >= 0")
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
    parser.add_argument(
        "--node-bookkeeper",
        choices=("memory", "bloom"),
        default="memory",
        help="Bookkeeper backend for visited-node deduplication.",
    )
    parser.add_argument(
        "--bloom-filter-size",
        type=int,
        default=1_000_000,
        help="Bloom filter capacity for bloom node bookkeeper.",
    )
    parser.add_argument(
        "--bloom-filter-fp-rate",
        type=float,
        default=1e-6,
        help="Bloom filter false-positive rate.",
    )
    parser.add_argument(
        "--bloom-factor",
        type=float,
        default=0.9,
        help="Rotate to a fresh bloom when inserted nodes exceed size*factor since last rotation.",
    )
    args = parser.parse_args()

    if args.potential_kwargs_file is not None and args.potential_kwargs_json is not None:
        raise ValueError("Use only one of --potential-kwargs-file or --potential-kwargs-json")
    if args.bloom_filter_size <= 0:
        raise ValueError("--bloom-filter-size must be > 0")
    if not (0.0 < args.bloom_filter_fp_rate < 1.0):
        raise ValueError("--bloom-filter-fp-rate must be in (0, 1)")
    if args.bloom_factor <= 0.0:
        raise ValueError("--bloom-factor must be > 0")

    potential_cls = _load_potential_cls(args.potential_file, args.potential_class)
    potential_kwargs = _load_potential_kwargs(args)
    rho = float(args.k if args.rho is None else args.rho)

    if args.include_k_taxi and "support" not in potential_kwargs:
        potential_kwargs["support"] = list(range(0, 2 * args.m, 2))

    metric_m = 2 * args.m if args.include_k_taxi else args.m
    distance_matrix = Circle.discrete(m=metric_m)
    initial_nodes = build_initial_nodes(args.k, args.m, args.include_k_taxi)
    transitions = build_transitions(args.m, args.include_k_taxi)
    create_hash_fn = build_hash_factory(args.include_k_taxi, args.no_symmetry_hash)

    worker_hooks = _merge_hook_maps(
        {"expand.after_transition": [edge_metrics_worker_hook]},
        create_worker_potential_hooks(potential_cls=potential_cls, potential_kwargs=potential_kwargs),
    )
    main_hooks = _merge_hook_maps(
        create_progress_bar_hooks(desc=f"potential check k={args.k} m={args.m}", unit="expand"),
        potential_inequality_main_hooks(
            rho=rho,
        ),
    )
    if args.profile_timing:
        worker_hooks = _merge_hook_maps(
            worker_hooks,
            create_worker_timing_profile_hooks(log_every=args.timing_log_every),
        )
        main_hooks = _merge_hook_maps(
            main_hooks,
            create_main_timing_profile_hooks(log_every=args.timing_log_every),
        )
    if args.profile_gc:
        main_hooks = _merge_hook_maps(
            main_hooks,
            create_main_gc_profile_hooks(
                log_every=args.gc_log_every,
                type_sample_every=args.gc_type_sample_every,
                print_fn=print,
            ),
        )

    node_bookkeeper_constructor = None
    if args.node_bookkeeper == "bloom":
        node_bookkeeper_constructor = lambda: BloomNodeBookkeeper(
            bloom_filter_size=args.bloom_filter_size,
            bloom_filter_fp_rate=args.bloom_filter_fp_rate,
            factor=args.bloom_factor,
        )

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
        node_bookkeeper_constructor=node_bookkeeper_constructor,
        ray_kwargs=ray_kwargs,
        return_wfs=False,
        return_paths=True,
    )

    print(f"done: nodes={len(result['node_bookkeeper'])} edges={len(result['edge_bookkeeper'])}")


if __name__ == "__main__":
    main()
