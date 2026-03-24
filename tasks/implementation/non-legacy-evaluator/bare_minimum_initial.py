from __future__ import annotations

from argparse import ArgumentParser, Namespace
import json
import signal
import time
import traceback
from itertools import product
from typing import Dict, Any

import numpy as np

from kserver.potential import KServerInstance



# EVOLVE-BLOCK-START


class Potential:
    def __init__(self, context):
        pass

    def __call__(self, wf: np.ndarray) -> float:
        return 0.0


def main(args) -> Dict[str, Any]:
    instances = [KServerInstance.load(path) for path in args.metrics]
    if not instances:
        raise ValueError("No metrics were provided")
        
    return {}


# EVOLVE-BLOCK-END


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def _install_interrupt_handlers() -> None:
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)



if __name__ == "__main__":
    _install_interrupt_handlers()
    parser = ArgumentParser()
    parser.add_argument("--metrics", type=str, nargs="+", help="Path to .pickle files")
    parser.add_argument("--output", type=str, help="Path to the result file")
    parser.add_argument("--timeout", type=float, help="timeout in seconds")
    parser.add_argument("--n_cpus", type=int, default=None, help="CPU count hint")

    args = parser.parse_args()

    output_filename = str(args.output)

    try:
        result = main(args)
    except KeyboardInterrupt:
        result = dict(
            failure="Search Failed",
            reason="Not handled interruption",
        )
    except Exception:
        result = dict(
            failure="Search Failed",
            reason=traceback.format_exc(),
        )

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f)
