#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path


def parse_lines_spec(spec: str) -> set[int]:
    """Parse line selector like '3,5,10-14' into a set of 1-based line numbers."""
    selected: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            pieces = token.split("-", maxsplit=1)
            if len(pieces) != 2:
                raise ValueError(f"Invalid line range: {token}")
            start_s, end_s = pieces
            if not start_s.isdigit() or not end_s.isdigit():
                raise ValueError(f"Invalid line range: {token}")
            start = int(start_s)
            end = int(end_s)
            if start < 1 or end < 1 or start > end:
                raise ValueError(f"Invalid line range: {token}")
            selected.update(range(start, end + 1))
            continue
        if not token.isdigit():
            raise ValueError(f"Invalid line number: {token}")
        value = int(token)
        if value < 1:
            raise ValueError(f"Invalid line number: {token}")
        selected.add(value)
    return selected


def default_outputs_dir_for(bash_file: Path) -> Path:
    """Build default output dir: outputs/<sweep_dir_without_sweep(s)>/<timestamp>."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_parts = bash_file.parent.parts

    start_idx = None
    for i, part in enumerate(parent_parts):
        if part in {"sweep", "sweeps"}:
            start_idx = i + 1
            break

    if start_idx is not None and start_idx < len(parent_parts):
        rel_parts = parent_parts[start_idx:]
    else:
        rel_parts = (bash_file.parent.name,)

    return Path("outputs").joinpath(*rel_parts, stamp)


def load_commands(path: Path) -> list[tuple[int, str]]:
    commands: list[tuple[int, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        cmd = raw.strip()
        if not cmd or cmd.startswith("#"):
            continue
        commands.append((line_no, cmd))
    return commands


async def run_command(
    *,
    line_no: int,
    command: str,
    output_dir: Path,
    sem: asyncio.Semaphore,
) -> tuple[int, int]:
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"

    async with sem:
        with stdout_path.open("w", encoding="utf-8") as out_fp, stderr_path.open("w", encoding="utf-8") as err_fp:
            env = os.environ.copy()
            env["RUN_GRID_OUTPUTS_DIR"] = str(output_dir)
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(output_dir),
                env=env,
                stdout=out_fp,
                stderr=err_fp,
            )
            exit_code = await proc.wait()
    return line_no, exit_code


async def async_main(args: argparse.Namespace) -> int:
    bash_file = Path(args.bash_file).resolve()
    if not bash_file.is_file():
        raise SystemExit(f"Bash file not found: {bash_file}")

    commands = load_commands(bash_file)
    selected_lines: set[int] | None = None
    if args.lines:
        selected_lines = parse_lines_spec(args.lines)

    filtered_commands: list[tuple[int, str]] = []
    for line_no, command in commands:
        if selected_lines is not None and line_no not in selected_lines:
            continue
        filtered_commands.append((line_no, command))
    commands = filtered_commands

    if not commands:
        print("No commands found after applying filters.")
        return 0

    outputs_dir_source = "explicit"
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir).resolve()
    else:
        outputs_dir = default_outputs_dir_for(bash_file).resolve()
        outputs_dir_source = "default"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sem = asyncio.Semaphore(args.max_concurrent)
    print(f"bash_file: {bash_file}")
    print(f"commands: {len(commands)}")
    if args.lines:
        print(f"line_selection: {args.lines}")
    print(f"max_concurrent: {args.max_concurrent}")
    print(f"outputs_dir ({outputs_dir_source}): {outputs_dir}")

    tasks = []
    for idx, (line_no, command) in enumerate(commands, start=1):
        job_dir = outputs_dir / f"{stamp}-{idx:04d}"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "command.sh").write_text(command + "\n", encoding="utf-8")
        tasks.append(
            asyncio.create_task(
                run_command(line_no=line_no, command=command, output_dir=job_dir, sem=sem)
            )
        )

    failures = 0
    for completed in asyncio.as_completed(tasks):
        line_no, exit_code = await completed
        if exit_code != 0:
            failures += 1
        print(f"line {line_no}: exit={exit_code}")

    print(f"outputs_dir: {outputs_dir}")
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run commands from a bash file (one command per line) with async concurrency."
    )
    parser.add_argument("bash_file", help="Path to bash file (one command per line)")
    parser.add_argument(
        "outputs_dir",
        nargs="?",
        help=(
            "Directory where per-command output folders are created. "
            "Default: outputs/<bash_file_parent_without_sweep_or_sweeps>/<YYYYmmdd_HHMMSS>"
        ),
    )
    parser.add_argument(
        "--lines",
        help="Run only specific source lines, e.g. '3,8,10-14'.",
    )
    parser.add_argument("--max-concurrent", type=int, default=1, help="Maximum concurrent commands")
    args = parser.parse_args()
    if args.max_concurrent < 1:
        parser.error("--max-concurrent must be >= 1")
    if args.lines:
        try:
            parse_lines_spec(args.lines)
        except ValueError as exc:
            parser.error(f"--lines: {exc}")
    return args


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
