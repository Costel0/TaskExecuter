from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .battle_dataset import merge_battle_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogia-merge-battle-datasets",
        description=(
            "Merge every OGIA .jsonl battle dataset in a directory into one "
            "validated JSONL dataset."
        ),
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the .jsonl battle datasets to merge.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Defaults to <directory>/merged_battles.jsonl."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also include .jsonl files inside subdirectories.",
    )
    return parser


def run(args: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    options = parser.parse_args(list(args or []))

    directory = options.directory.expanduser()
    output = (
        options.output.expanduser()
        if options.output is not None
        else directory / "merged_battles.jsonl"
    )

    source_file_count, battle_count = merge_battle_directory(
        directory,
        output,
        recursive=bool(options.recursive),
    )

    print(f"Source files merged: {source_file_count:,}")
    print(f"Battles merged: {battle_count:,}")
    print(f"Output: {output}")
    return 0
