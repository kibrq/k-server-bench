import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import ray


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@contextmanager
def file_lock(
    lock_path: str,
    timeout_s: float = 120.0,
    poll_s: float = 0.2,
) -> Iterator[None]:
    import fcntl

    lock_dir = os.path.dirname(lock_path) or "."
    os.makedirs(lock_dir, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > timeout_s:
                    raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
                time.sleep(poll_s)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _tcp_connect_ok(host: str, port: int, timeout_s: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def probe_ray_cluster(ray_addr: str, timeout_s: float) -> None:
    probe_code = (
        "import sys\n"
        "import ray\n"
        "addr = sys.argv[1]\n"
        "ray.init(address=addr)\n"
        "ray.shutdown()\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code, ray_addr],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Timed out probing ray.init(address={ray_addr!r}) after {timeout_s:.1f}s"
        ) from exc

    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        if details:
            raise RuntimeError(
                f"Probe ray.init(address={ray_addr!r}) failed: {details}"
            )
        raise RuntimeError(
            f"Probe ray.init(address={ray_addr!r}) failed with exit code {proc.returncode}"
        )


def restart_ray_via_script(
    script_path: str,
    env: Optional[Dict[str, str]] = None,
    timeout_s: float = 600.0,
) -> None:
    cmd = ["/bin/bash", script_path]
    proc = subprocess.run(
        cmd,
        env={**os.environ, **(env or {})},
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Ray restart script failed.\n"
            f"cmd={cmd}\n"
            "See forwarded stdout/stderr above.\n"
        )


def _default_restart_script_from_home() -> Optional[str]:
    home = os.getenv("K_SERVER_EVALUATE_HOME")
    if not home:
        return None
    # Prefer the name requested by users; keep backward-compatible fallback.
    candidate_ray_restart = os.path.join(home, "ray_restart.sh")
    candidate_restart_ray = os.path.join(home, "restart_ray.sh")
    if os.path.exists(candidate_ray_restart):
        return candidate_ray_restart
    if os.path.exists(candidate_restart_ray):
        return candidate_restart_ray
    return candidate_ray_restart


def connect_or_restart_ray(
    ray_addr: str = "auto",
    probe_timeout_s: float = 30.0,
    restart_script: Optional[str] = None,
) -> None:
    script_path = (
        restart_script
        or os.getenv("K_SERVER_EVALUATE_RAY_RESTART_SCRIPT")
        or _default_restart_script_from_home()
    )

    try:
        probe_ray_cluster(ray_addr, timeout_s=probe_timeout_s)
    except TimeoutError:
        if not script_path:
            raise RuntimeError(
                "Ray probe timed out and no restart script was provided. "
                "Set restart_script or K_SERVER_EVALUATE_RAY_RESTART_SCRIPT."
            )
        restart_ray_via_script(
            script_path=script_path,
            timeout_s=float(os.getenv("K_SERVER_EVALUATE_RAY_RESTART_TIMEOUT_S", "600")),
        )

    ray.init(address=ray_addr)
