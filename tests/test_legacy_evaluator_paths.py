from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "k-servers" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kserver.evaluation import NumpyKServerInstance


LEGACY_EVALUATOR = ROOT / "tools" / "legacy-evaluator" / "evaluate.py"
LEGACY_EVALUATOR_MCP = ROOT / "tools" / "legacy-evaluator" / "evaluate_mcp_server.py"
LEGACY_RAY_INSTANCE = ROOT / "tools" / "legacy-evaluator" / "ray_kserver_instance.py"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_legacy_evaluator_defaults_resolve_to_bench_paths() -> None:
    module = _load_module(LEGACY_EVALUATOR)
    args = module.parse_args(["--program_path", "submission.py"])

    assert Path(args.home) == ROOT / "tools" / "legacy-evaluator"
    assert Path(args.metrics_path) == ROOT / "metrics"


def test_legacy_evaluator_home_default_modules() -> None:
    module = _load_module(LEGACY_EVALUATOR)
    args = module.parse_args(
        [
            "--program_path",
            "submission.py",
            "--use_default_potential_family",
            "--use_default_search_evaluator",
            "--use_default_potential",
        ]
    )
    args.results_dir = "tmp-results"
    module.ray.shutdown = lambda: None

    def _stub_connect_or_restart_ray(**kwargs):
        return None

    module.connect_or_restart_ray = _stub_connect_or_restart_ray
    module.evaluate_solution_family = lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("stop after path resolution")
    )

    try:
        module.main(args)
    except RuntimeError as exc:
        assert str(exc) == "stop after path resolution"

    assert Path(args.potential_family_path) == ROOT / "tools" / "legacy-evaluator" / "naive_potential_family.py"
    assert Path(args.search_evaluator_path) == ROOT / "tools" / "legacy-evaluator" / "naive_search_evaluator.py"
    assert Path(args.potential_path) == ROOT / "tools" / "legacy-evaluator" / "canonical_potential.py"


def test_legacy_evaluator_mcp_defaults_to_local_evaluate_script() -> None:
    try:
        module = _load_module(LEGACY_EVALUATOR_MCP)
    except ModuleNotFoundError as exc:
        pytest.skip(f"MCP dependencies unavailable: {exc}")

    assert Path(module.K_SERVER_MCP_EVALUATE_PATH) == ROOT / "tools" / "legacy-evaluator" / "evaluate.py"


def test_legacy_ray_instance_bridge_uses_numpy_instance() -> None:
    ray = pytest.importorskip("ray")
    module = _load_module(LEGACY_RAY_INSTANCE)

    payload = {
        "k": 2,
        "distance_matrix": np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=int),
        "nodes": [
            {"id": 0, "depth": 0, "wf_norm": np.asarray([0.0, 1.0, 2.0])},
            {"id": 1, "depth": 1, "wf_norm": np.asarray([1.0, 0.0, 1.0])},
        ],
        "edges": [
            {"from": 0, "to": 1, "ext": 0.0, "d_min": 1.0, "weight": 1.0},
        ],
        "bellman": np.asarray([0.0, 1.0]),
    }
    instance = NumpyKServerInstance.from_legacy_dict(payload)

    ray.init(local_mode=True, ignore_reinit_error=True)
    try:
        friendly = module.to_ray_friendly(instance)
        context = friendly.get_context()
        nodes = friendly.get_nodes()
        edges = friendly.get_edges()

        assert context.k == instance.k
        assert int(nodes[1]["id"]) == 1
        np.testing.assert_array_equal(np.asarray(nodes[0]["wf_norm"]), instance.node_wf_norm[0])
        assert float(edges[0]["d_min"]) == 1.0
    finally:
        ray.shutdown()
