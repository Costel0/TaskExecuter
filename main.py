from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
import sys
from typing import TextIO

from task_runtime.registry import execute_task, list_tasks


class _Tee:
    """Write task output to both the terminal and a persistent log file."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run long-lived TaskExecuter tasks locally or on the VM."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list", help="List registered tasks")

    run_parser = subparsers.add_parser("run", help="Run a registered task")
    run_parser.add_argument("task", help="Task name")
    run_parser.add_argument(
        "task_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the task",
    )
    return parser


def _print_tasks() -> None:
    tasks = tuple(list_tasks())
    if not tasks:
        print("No tasks registered.")
        return

    print("Available tasks:")
    for task in tasks:
        print(f"  {task.name:<24} {task.description}")


def _run_task(task_name: str, task_args: list[str]) -> int:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{timestamp}_{task_name}.log"

    with log_path.open("a", encoding="utf-8") as log_file:
        stdout = _Tee(sys.stdout, log_file)
        stderr = _Tee(sys.stderr, log_file)

        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"Starting task: {task_name}")
            print(f"Log file: {log_path}")
            try:
                exit_code = execute_task(task_name, task_args)
            except Exception as exc:
                print(f"Task failed: {exc}", file=sys.stderr)
                return 1

            print(f"Task finished with exit code {exit_code}")
            return exit_code


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "list":
        _print_tasks()
        return 0

    if args.command == "run":
        return _run_task(args.task, args.task_args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
