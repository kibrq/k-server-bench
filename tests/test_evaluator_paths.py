from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "k-servers" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EVALUATOR = ROOT / "tools" / "evaluator" / "evaluate.py"
EVALUATOR_MCP = ROOT / "tools" / "evaluator" / "evaluate_mcp_server.py"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluator_defaults_resolve_to_bench_root() -> None:
    module = _load_module(EVALUATOR)
    old_argv = sys.argv
    sys.argv = [str(EVALUATOR), "--program_path", "submission.py"]
    try:
        args = module.parse_args()
    finally:
        sys.argv = old_argv

    assert Path(args.evaluate_home) == ROOT


def test_evaluator_resolves_metrics_from_bench_root() -> None:
    module = _load_module(EVALUATOR)
    old_argv = sys.argv
    sys.argv = [
        str(EVALUATOR),
        "--program_path",
        "submission.py",
        "--metrics_names",
        "circle_k4_m6.pickle",
    ]
    try:
        args = module.parse_args()
    finally:
        sys.argv = old_argv

    metric_paths = module.resolve_metric_paths(args)
    assert metric_paths == [ROOT / "metrics" / "circle_k4_m6.pickle"]


def test_evaluator_accepts_potential_path_and_inline_kwargs() -> None:
    module = _load_module(EVALUATOR)
    old_argv = sys.argv
    sys.argv = [
        str(EVALUATOR),
        "--program_path",
        "submission.py",
        "--potential_path",
        "potential.py",
        "--potential_kwargs_json",
        '{"x": 1}',
    ]
    try:
        args = module.parse_args()
    finally:
        sys.argv = old_argv

    assert Path(args.potential_path) == Path("potential.py")
    assert args.potential_kwargs_json == '{"x": 1}'


def test_evaluator_loads_potential_kwargs_from_inline_json_or_file() -> None:
    module = _load_module(EVALUATOR)

    assert module._load_json_value_or_file('{"x": 1}') == {"x": 1}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "kwargs.json"
        path.write_text('{"y": 2}', encoding="utf-8")
        assert module._load_json_value_or_file(str(path)) == {"y": 2}


def test_evaluator_mcp_defaults_to_local_paths() -> None:
    try:
        module = _load_module(EVALUATOR_MCP)
    except ModuleNotFoundError as exc:
        pytest.skip(f"MCP dependencies unavailable: {exc}")

    assert Path(module.K_SERVER_MCP_EVALUATE_PATH) == ROOT / "tools" / "evaluator" / "evaluate.py"
    assert Path(module.K_SERVER_EVALUATE_HOME) == ROOT
