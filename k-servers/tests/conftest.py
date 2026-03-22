import os
import sys
import pytest


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


def pytest_addoption(parser):
    parser.addoption(
        "--run-legacy-compat",
        action="store_true",
        default=False,
        help="run legacy compatibility tests (slow, optional)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "legacy_compat: marks legacy compatibility tests that are not run by default",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-legacy-compat"):
        return

    keep = []
    deselected = []
    for item in items:
        if "legacy_compat" in item.keywords:
            deselected.append(item)
        else:
            keep.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep
