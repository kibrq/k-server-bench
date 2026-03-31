from __future__ import annotations

import hashlib
import multiprocessing as mp
import pickle
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from kserver.context.numpy_wf_context import WFContext

from .numpy_kserver_instance import NumpyKServerInstance


Potential = Callable[[Sequence[float]], Union[float, tuple[float, dict]]]
PotentialFactory = Callable[[WFContext], Potential]


@dataclass
class KServerPotentialStats:
    df_nodes: pd.DataFrame
    df_edges: pd.DataFrame
    metrics: dict
    unique_key: str

    def dump(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(asdict(self), f)

    @classmethod
    def load(cls, path: str | Path) -> "KServerPotentialStats":
        with open(path, "rb") as f:
            return cls(**pickle.load(f))


def _extract_idx_to_config(context: WFContext) -> list[tuple[int, ...]]:
    if hasattr(context, "idx_to_config"):
        return [tuple(cfg) for cfg in getattr(context, "idx_to_config")]
    if hasattr(context, "_idx_to_config"):
        return [tuple(cfg) for cfg in getattr(context, "_idx_to_config")]
    raise AttributeError("Context does not expose idx_to_config")


def _unwrap_potential_result(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def _wf_to_numpy(wf) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(wf, dtype=float))


def _wf_key(wf) -> bytes:
    return _wf_to_numpy(wf).tobytes()


def _unique_wf_dict() -> dict[bytes, np.ndarray]:
    return {}


def _accumulate_unique_wf(unique_wfs: dict[bytes, np.ndarray], wf) -> np.ndarray:
    wf_array = _wf_to_numpy(wf)
    key = wf_array.tobytes()
    if key not in unique_wfs:
        unique_wfs[key] = wf_array
    return wf_array


def batch_compute_potential_simple(
    wfs: list[np.ndarray],
    potential_fn: Potential,
    context: WFContext = None,
    tqdm_kwargs: dict | None = None,
    timeout: float | None = None,
):
    del context
    cache = {}
    start_time = time.time()
    for wf in tqdm(wfs, **(tqdm_kwargs or dict(disable=True))):
        if timeout is not None and time.time() - start_time > timeout:
            break
        key = _wf_key(wf)
        if key in cache:
            continue
        cache[key] = potential_fn(wf)
    return cache


_potential_fn = None


def _init_worker(potential_fn):
    global _potential_fn
    _potential_fn = potential_fn


def _compute_one(wf):
    return _wf_key(wf), _potential_fn(wf)


def batch_compute_potential_mp(
    wfs,
    potential_fn,
    n_processes=2,
    timeout=None,
    chunk_size=50,
):
    results = []
    start_time = time.time()

    pool = mp.Pool(
        processes=n_processes,
        initializer=_init_worker,
        initargs=(potential_fn,),
    )
    async_iter = pool.imap_unordered(_compute_one, wfs, chunksize=chunk_size)

    try:
        while True:
            if timeout is None:
                results.append(next(async_iter))
                continue

            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                raise mp.TimeoutError
            results.append(async_iter.next(timeout=remaining))
    except StopIteration:
        pass
    except mp.TimeoutError:
        time.sleep(0.2)
        pool.terminate()
    finally:
        if timeout is None:
            pool.close()
        else:
            pool.terminate()
        pool.join()

    return {key: value for key, value in results if key is not None}


def _estimate_opt_upper_bound(
    wfs,
    potential,
    compute_potential_fn,
    ts_to_estimate_at=None,
):
    from collections import defaultdict
    from itertools import combinations

    if ts_to_estimate_at is None:
        ts_to_estimate_at = [0, 1, 2, 3, 10000]

    flattened_wfs = []
    unique_wfs = _unique_wf_dict()
    for i, wf in enumerate(wfs):
        wf_array = _wf_to_numpy(wf)
        for t in ts_to_estimate_at:
            shifted_wf = _accumulate_unique_wf(unique_wfs, wf_array + t)
            flattened_wfs.append((i, t, _wf_key(shifted_wf)))
    cache = compute_potential_fn(list(unique_wfs.values()), potential)

    cache_grouped_by_i = defaultdict(list)
    for i, t, wf_key in flattened_wfs:
        if wf_key in cache:
            p, info = _unwrap_potential_result(cache[wf_key])
            del info
            cache_grouped_by_i[i].append((t, p))

    estimates = defaultdict(lambda: float("-inf"))
    for i, ts_and_potentials in cache_grouped_by_i.items():
        for (t1, p1), (t2, p2) in combinations(ts_and_potentials, 2):
            estimate = (p2 - p1) / (t2 - t1)
            estimates[i] = max(estimates[i], estimate)

    return max(estimates.values())


def _make_robustness_context(
    context: WFContext,
    factor: float = 1.0,
    permutation: Sequence[int] | None = None,
) -> WFContext:
    idx_to_config = _extract_idx_to_config(context)
    if permutation is None:
        permutation = np.arange(len(idx_to_config), dtype=int)
    permutation = np.asarray(permutation, dtype=int)
    permuted_idx_to_config = [idx_to_config[i] for i in permutation]
    return WFContext(
        k=context.k,
        distance_matrix=np.asarray(context.distance_matrix, dtype=float) * float(factor),
        idx_to_config=permuted_idx_to_config,
    )


def compute_potential_stats(
    potential_factory: PotentialFactory,
    kserver_instance: NumpyKServerInstance,
    cache=None,
    opt_upper_bound_estimate_sample_size: Optional[int] = 100,
    ts_to_estimate_at=None,
    tqdm_kwargs: dict | None = None,
    seed: int = 42,
    compute_potential_backend: str = "simple",
    compute_potential_kwargs: dict | None = None,
    compute_potential_estimate_ub_backend: Optional[str] = None,
    compute_potential_estimate_ub_kwargs: Optional[dict] = None,
    compute_potential_robustness_backend: Optional[str] = None,
    compute_potential_robustness_kwargs: Optional[dict] = None,
    robustness_check: bool = False,
    compute_bellman_closure: bool = False,
    include_wf_columns: bool = True,
    include_info_columns: bool = True,
    include_renormalized_metrics: bool = True,
    round_digits: Optional[int] = None,
    rho: Optional[float] = None,
    keep_only_violations_k: bool = False,
):
    if compute_bellman_closure:
        compute_bellman_closure = False

    nodes = kserver_instance.get_nodes()
    edges = kserver_instance.get_edges()
    context = kserver_instance.get_context()
    bellman_potential = np.asarray(kserver_instance.get_bellman(), dtype=float)
    k = kserver_instance.k

    def _build_compute_potential_fn(backend_name, compute_kwargs, tqdm_kwargs=None):
        if backend_name == "simple":
            return partial(
                batch_compute_potential_simple,
                tqdm_kwargs=(tqdm_kwargs or {"disable": True}),
                **(compute_kwargs or {}),
            )
        if backend_name == "mp":
            return partial(batch_compute_potential_mp, **(compute_kwargs or {}))
        raise ValueError(f"Invalid backend {backend_name}")

    compute_potential_fn = _build_compute_potential_fn(
        compute_potential_backend,
        compute_potential_kwargs,
        tqdm_kwargs,
    )
    compute_potential_estimate_ub_fn = _build_compute_potential_fn(
        compute_potential_estimate_ub_backend or compute_potential_backend,
        compute_potential_estimate_ub_kwargs or compute_potential_kwargs,
        tqdm_kwargs,
    )
    compute_potential_robustness_fn = _build_compute_potential_fn(
        compute_potential_robustness_backend or compute_potential_backend,
        compute_potential_robustness_kwargs or compute_potential_kwargs,
        tqdm_kwargs,
    )

    potential = potential_factory(context)

    if cache is None:
        what_to_compute_start_time = time.time()
        unique_wfs = _unique_wf_dict()
        for node in nodes:
            _accumulate_unique_wf(unique_wfs, node["wf_norm"])

        if include_renormalized_metrics:
            for edge in edges:
                v = edge["to"]
                _accumulate_unique_wf(unique_wfs, np.asarray(nodes[v]["wf_norm"]) + edge["d_min"])

        rng = np.random.default_rng(seed)
        what_to_compute = list(unique_wfs.values())
        idxes = rng.permutation(len(what_to_compute))
        what_to_compute = [what_to_compute[i] for i in idxes]
        what_to_compute_time = time.time() - what_to_compute_start_time

        start_time = time.time()
        cache = compute_potential_fn(what_to_compute, potential)
        cache_time = time.time() - start_time
    else:
        what_to_compute_time = 0.0
        cache_time = 0.0

    build_dataframes_start_time = time.time()

    df_nodes = pd.DataFrame([{"id": node["id"], "depth": node["depth"]} for node in nodes])
    df_nodes["bellman"] = bellman_potential

    node_potential = np.full(len(nodes), np.nan, dtype=float)
    node_info = np.empty(len(nodes), dtype=object)
    for idx, node in enumerate(nodes):
        val = cache.get(_wf_key(node["wf_norm"]), None)
        p, p_info = _unwrap_potential_result(val)
        node_potential[idx] = p
        node_info[idx] = p_info

    df_nodes["potential"] = node_potential
    df_nodes["potential_info"] = node_info if include_info_columns else None

    if include_wf_columns:
        df_nodes["wf_norm"] = [tuple(float(x) for x in node["wf_norm"]) for node in nodes]

    df_edges = pd.DataFrame([dict(edge) for edge in edges])
    u_idx = df_edges["from"].to_numpy()
    v_idx = df_edges["to"].to_numpy()

    df_edges["u_bellman"] = bellman_potential[u_idx]
    df_edges["v_bellman"] = bellman_potential[v_idx]
    df_edges["u_potential"] = node_potential[u_idx]
    df_edges["v_potential"] = node_potential[v_idx]

    if include_info_columns:
        df_edges["u_potential_info"] = node_info[u_idx]
        df_edges["v_potential_info"] = node_info[v_idx]
    else:
        df_edges["u_potential_info"] = None
        df_edges["v_potential_info"] = None

    if include_renormalized_metrics:
        v_renorm_potential = np.full(len(df_edges), np.nan, dtype=float)
        v_renorm_info = np.empty(len(df_edges), dtype=object)
        for idx, edge in enumerate(edges):
            v = edge["to"]
            d_min = edge["d_min"]
            val = cache.get(_wf_key(np.asarray(nodes[v]["wf_norm"]) + d_min), None)
            pv, pv_info = _unwrap_potential_result(val)
            v_renorm_potential[idx] = pv
            if include_info_columns:
                v_renorm_info[idx] = pv_info
        df_edges["v_renorm_potential"] = v_renorm_potential
        df_edges["v_renorm_info"] = v_renorm_info if include_info_columns else None
    else:
        df_edges["v_renorm_potential"] = None
        df_edges["v_renorm_info"] = None

    if include_wf_columns:
        df_edges["u_wf"] = [tuple(float(x) for x in nodes[u]["wf_norm"]) for u in u_idx]
        df_edges["v_wf"] = [tuple(float(x) for x in nodes[v]["wf_norm"]) for v in v_idx]
        df_edges["v_renorm_wf"] = [
            tuple(float(x + edges[i]["d_min"]) for x in nodes[v]["wf_norm"]) for i, v in enumerate(v_idx)
        ]

    build_dataframes_time = time.time() - build_dataframes_start_time

    processed_edges_mask = ~(df_edges["u_potential"].isna() | df_edges["v_potential"].isna())
    processed_renorm_edges_mask = ~(df_edges["u_potential"].isna() | df_edges["v_renorm_potential"].isna())
    processed_nodes_mask = ~df_nodes["potential"].isna()

    if opt_upper_bound_estimate_sample_size is not None:
        df_edges["opt_upper_bound"] = np.nan
        df = df_nodes[processed_nodes_mask]
        if df.shape[0] > 0 and opt_upper_bound_estimate_sample_size > 0:
            sample_size = min(int(opt_upper_bound_estimate_sample_size), int(df.shape[0]))
            node_idxes = np.random.choice(df.index, sample_size, replace=False)
            wfs = [np.asarray(nodes[i]["wf_norm"], dtype=float) for i in node_idxes]
            opt_upper_bound = _estimate_opt_upper_bound(
                wfs,
                potential,
                compute_potential_estimate_ub_fn,
                ts_to_estimate_at,
            )
            for node_idx in node_idxes:
                mask = (df_edges["from"] == node_idx) | (df_edges["to"] == node_idx)
                df_edges.loc[mask, "opt_upper_bound"] = float(opt_upper_bound)
    else:
        opt_upper_bound = None

    df_processed_edges = df_edges[processed_edges_mask]
    df_processed_renorm_edges = df_edges[processed_renorm_edges_mask]
    df_processed_nodes = df_nodes[processed_nodes_mask]

    unprocessed_normalized_edges = df_edges.shape[0] - df_processed_edges.shape[0]
    processed_normalized_edges_score = df_processed_edges.shape[0] / df_edges.shape[0]

    if include_renormalized_metrics:
        unprocessed_renormalized_edges = df_edges.shape[0] - df_processed_renorm_edges.shape[0]
        processed_renormalized_edges_score = df_processed_renorm_edges.shape[0] / df_edges.shape[0]
    else:
        df_processed_renorm_edges = None
        unprocessed_renormalized_edges = None
        processed_renormalized_edges_score = None

    if rho is None:
        rho = k

    df = df_processed_edges
    series = df["v_potential"] - df["u_potential"] + (rho + 1) * df["d_min"] - df["ext"]
    violations_k = (series < 0).sum()
    violations_k_score = 1 - (series < 0).mean()

    df = df_processed_edges[df_processed_edges["d_min"] == 0]
    series = df["v_potential"] - df["u_potential"] - df["ext"]
    violations_dmin_0 = (series < 0).sum()
    violations_dmin_0_score = 1 - (series < 0).mean()

    df = df_processed_edges[(df_processed_edges["d_min"] == 0) & (df_processed_edges["ext"] > 0)]
    series = df["v_potential"] - df["u_potential"]
    undetected_dmin_0 = (series == 0).sum()
    undetected_dmin_0_score = 1 - (series == 0).mean()

    if include_renormalized_metrics:
        df = df_processed_renorm_edges
        series = df["v_renorm_potential"] - df["u_potential"] - df["ext"]
        violations_renorm = (series < 0).sum()
        violations_renorm_score = 1 - (series < 0).mean()
    else:
        violations_renorm = None
        violations_renorm_score = None

    df = df_processed_edges[df_processed_edges["d_min"] != 0]
    strong_hypothesis_rho = ((df["ext"] - (df["v_potential"] - df["u_potential"])) / (df["d_min"])).max()
    min_value = -len(nodes[0]["wf_norm"])
    max_value = -(k + 1)
    strong_hypothesis_rho_score = (-strong_hypothesis_rho - min_value) / (max_value - min_value)

    if opt_upper_bound_estimate_sample_size is not None:
        opt_upper_bound = df_processed_edges["opt_upper_bound"].fillna(-float("inf")).max()
        min_value = -len(nodes[0]["wf_norm"])
        max_value = -(k + 1)
        opt_upper_bound_score = (-opt_upper_bound - min_value) / (max_value - min_value)
    else:
        opt_upper_bound_score = None

    if df_processed_edges.shape[0] > 1:
        bellman_edge_corr = np.corrcoef(
            df_processed_edges["v_bellman"] - df_processed_edges["u_bellman"],
            df_processed_edges["v_potential"] - df_processed_edges["u_potential"],
        )[0, 1]
        bellman_edge_mse = (
            (
                df_processed_edges["v_bellman"]
                - df_processed_edges["u_bellman"]
                - (df_processed_edges["v_potential"] - df_processed_edges["u_potential"])
            )
            ** 2
        ).mean()
        bellman_edge_r2 = r2_score(
            df_processed_edges["v_bellman"] - df_processed_edges["u_bellman"],
            df_processed_edges["v_potential"] - df_processed_edges["u_potential"],
        )
    else:
        bellman_edge_corr = None
        bellman_edge_mse = None
        bellman_edge_r2 = None

    if include_renormalized_metrics:
        unique_key = hashlib.sha256(
            (df_processed_renorm_edges["v_renorm_potential"] - df_processed_renorm_edges["u_potential"]).to_numpy().tobytes()
        ).hexdigest()
    else:
        unique_key = hashlib.sha256(
            (df_processed_edges["v_potential"] - df_processed_edges["u_potential"]).to_numpy().tobytes()
        ).hexdigest()

    if df_processed_nodes.shape[0] > 1:
        bellman_node_corr = df_processed_nodes[["bellman", "potential"]].corr().iloc[0, 1]
        bellman_node_spearman = float(df_processed_nodes[["bellman", "potential"]].corr(method="spearman").iloc[0, 1])
        bellman_node_spearman = np.maximum(bellman_node_spearman, 0.0)
    else:
        bellman_node_corr = None
        bellman_node_spearman = None

    df = df_processed_nodes[df_processed_nodes["depth"] > 0]
    if df.shape[0] > 1:
        bellman_non_initial_node_corr = df[["bellman", "potential"]].corr().iloc[0, 1]
        bellman_non_initial_node_spearman = float(df[["bellman", "potential"]].corr(method="spearman").iloc[0, 1])
        bellman_non_initial_node_spearman = np.maximum(bellman_non_initial_node_spearman, 0.0)
    else:
        bellman_non_initial_node_corr = None
        bellman_non_initial_node_spearman = None

    slack_k = (
        df_processed_edges["v_potential"]
        - df_processed_edges["u_potential"]
        + (rho + 1) * df_processed_edges["d_min"]
        - df_processed_edges["ext"]
    )
    neg_k = np.minimum(slack_k, 0.0)
    violations_k_l1 = float(-neg_k.sum())
    violations_k_l2 = float(np.sqrt((neg_k**2).sum()))
    violations_k_linf = float((-neg_k).max())

    def _round_value(val):
        if round_digits is None or val is None:
            return val
        if isinstance(val, float):
            return round(val, round_digits)
        if isinstance(val, dict):
            return {k: _round_value(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            container = [_round_value(v) for v in val]
            return tuple(container) if isinstance(val, tuple) else container
        return val

    public_metrics = {
        "violations_k": int(violations_k),
        "violations_k_score": float(violations_k_score),
        "violations_dmin_0": int(violations_dmin_0),
        "violations_dmin_0_score": float(violations_dmin_0_score),
        "undetected_dmin_0": int(undetected_dmin_0),
        "undetected_dmin_0_score": float(undetected_dmin_0_score),
        "violations_renorm": int(violations_renorm) if violations_renorm is not None else None,
        "violations_renorm_score": float(violations_renorm_score) if violations_renorm_score is not None else None,
        "strong_hypothesis_rho": float(strong_hypothesis_rho),
        "strong_hypothesis_rho_score": float(strong_hypothesis_rho_score),
        "opt_upper_bound": float(opt_upper_bound) if opt_upper_bound is not None else None,
        "opt_upper_bound_score": float(opt_upper_bound_score) if opt_upper_bound_score is not None else None,
        "bellman_node_corr": float(bellman_node_corr) if bellman_node_corr is not None else None,
        "bellman_edge_corr": float(bellman_edge_corr) if bellman_edge_corr is not None else None,
        "bellman_edge_mse": float(bellman_edge_mse) if bellman_edge_mse is not None else None,
        "bellman_edge_r2": float(bellman_edge_r2) if bellman_edge_r2 is not None else None,
        "bellman_node_spearman": float(bellman_node_spearman) if bellman_node_spearman is not None else None,
        "bellman_non_initial_node_corr": float(bellman_non_initial_node_corr) if bellman_non_initial_node_corr is not None else None,
        "bellman_non_initial_node_spearman": float(bellman_non_initial_node_spearman) if bellman_non_initial_node_spearman is not None else None,
        "violations_k_l1": float(violations_k_l1),
        "violations_k_l2": float(violations_k_l2),
        "violations_k_linf": float(violations_k_linf),
        "unprocessed_normalized_edges": int(unprocessed_normalized_edges),
        "processed_normalized_edges_score": float(processed_normalized_edges_score),
        "unprocessed_renormalized_edges": int(unprocessed_renormalized_edges) if unprocessed_renormalized_edges is not None else None,
        "processed_renormalized_edges_score": float(processed_renormalized_edges_score) if processed_renormalized_edges_score is not None else None,
        "total_edges": len(edges),
        "total_nodes": len(nodes),
        "rho": float(rho),
        "what_to_compute_time": float(what_to_compute_time),
        "cache_time": float(cache_time),
        "build_dataframes_time": float(build_dataframes_time),
    }

    robustness_metrics = None
    robustness_pass = None
    if robustness_check and violations_k == 0:
        edges_for_robustness = df_processed_edges.reset_index(drop=True)
        robust_u_idx = edges_for_robustness["from"].to_numpy(dtype=int)
        robust_v_idx = edges_for_robustness["to"].to_numpy(dtype=int)

        node_wf_count = len(nodes[0]["wf_norm"])
        node_wf_by_node = np.empty(len(nodes), dtype=object)
        for node_idx in range(len(nodes)):
            node_wf_by_node[node_idx] = tuple(float(nodes[node_idx]["wf_norm"][i]) for i in range(node_wf_count))
        node_wf_matrix = np.asarray([node_wf_by_node[node_idx] for node_idx in range(len(nodes))], dtype=float)

        permutation_rng = np.random.default_rng(seed + 12345 if seed is not None else None)
        permutation = permutation_rng.permutation(len(_extract_idx_to_config(context)))

        transform_configs = [{"factor": 2.0, "permutation": permutation, "shift": 1.0}]
        robustness_metrics = {}

        def _cfg_key(cfg):
            has_perm = cfg["permutation"] is not None
            return f"factor={cfg['factor']}_perm={int(has_perm)}_shift={cfg['shift']}"

        for cfg in transform_configs:
            cfg_node_wfs_matrix = node_wf_matrix.copy()
            cfg_node_wfs_matrix *= cfg["factor"]
            if cfg["permutation"] is not None:
                cfg_node_wfs_matrix = cfg_node_wfs_matrix[:, cfg["permutation"]]
            cfg_node_wfs_matrix = cfg_node_wfs_matrix + cfg["shift"]

            cfg_potential = potential_factory(
                _make_robustness_context(
                    context=context,
                    factor=cfg["factor"],
                    permutation=cfg["permutation"],
                )
            )

            cfg_node_wfs = np.empty(len(nodes), dtype=object)
            unique_wfs = _unique_wf_dict()
            for node_idx in range(len(nodes)):
                wf = _wf_to_numpy(cfg_node_wfs_matrix[node_idx])
                cfg_node_wfs[node_idx] = wf
                unique_wfs[_wf_key(wf)] = wf

            cfg_cache = compute_potential_robustness_fn(list(unique_wfs.values()), cfg_potential) if len(unique_wfs) > 0 else {}

            cfg_node_potential = np.full(len(nodes), np.nan, dtype=float)
            for node_idx, wf in enumerate(cfg_node_wfs):
                p, _ = _unwrap_potential_result(cfg_cache.get(_wf_key(wf), None))
                cfg_node_potential[node_idx] = p

            df_edges_case = edges_for_robustness.copy()
            df_edges_case["u_potential"] = cfg_node_potential[robust_u_idx]
            df_edges_case["v_potential"] = cfg_node_potential[robust_v_idx]

            rhs = (
                df_edges_case["v_potential"]
                - df_edges_case["u_potential"]
                + (rho + 1) * cfg["factor"] * df_edges_case["d_min"]
                - cfg["factor"] * df_edges_case["ext"]
            )
            robustness_metrics[_cfg_key(cfg)] = int(np.less(rhs, 0).sum())

        def _no_violations(value):
            if isinstance(value, dict):
                return all(_no_violations(v) for v in value.values())
            if isinstance(value, (list, tuple, np.ndarray)):
                return all(_no_violations(v) for v in value)
            if isinstance(value, (int, np.integer, float, np.floating)):
                return int(value) == 0
            return False

        robustness_pass = _no_violations(robustness_metrics)

    public_metrics["robustness"] = robustness_pass
    public_metrics["robustness_details"] = robustness_metrics
    public_metrics = {k: _round_value(v) if k != "violations_k_score" else v for k, v in public_metrics.items()}

    if keep_only_violations_k:
        public_metrics = {
            "violations_k": public_metrics["violations_k"],
            "violations_k_score": public_metrics["violations_k_score"],
            "robustness": public_metrics["robustness"],
        }

    return KServerPotentialStats(df_nodes, df_edges, public_metrics, unique_key)
