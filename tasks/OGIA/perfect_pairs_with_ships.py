from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Sequence

from . import perfect_pairs as _base
from .defender import generate_random_defender


DEFAULT_SHIP_ONLY_RATIO = 0.0


def _ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _parse_task_options(
    args: Sequence[str] | None,
) -> tuple[float, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--ship-only-ratio",
        type=_ratio,
        default=DEFAULT_SHIP_ONLY_RATIO,
    )
    options, remaining = parser.parse_known_args(list(args or []))
    return float(options.ship_only_ratio), remaining


def _mode_for_sequence_index(
    sequence_index: int,
    ship_only_ratio: float,
) -> str:
    """Distribute accepted pairs as evenly as possible across both modes.

    The cumulative ship-only count after N accepted pairs is round(N * ratio),
    so for 200 pairs at 0.60 the result is exactly 120 ship-only / 80 mixed.
    Failed optimizer attempts keep the same sequence index and therefore do not
    distort the requested accepted-pair ratio.
    """

    current_target = int(
        math.floor(float(sequence_index) * ship_only_ratio + 0.5)
    )
    previous_target = int(
        math.floor(float(sequence_index - 1) * ship_only_ratio + 0.5)
    )
    return "ship_only" if current_target > previous_target else "mixed"


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        Path("data")
        / "OGIA"
        / "perfect_pairs"
        / f"perfect_pairs_with_ships_{timestamp}.jsonl"
    )


def run(args: Sequence[str] | None = None) -> int:
    """Generate ship-aware perfect pairs with a configurable defender mix.

    By default this preserves the previous behavior and generates only mixed
    defenders. ``--ship-only-ratio`` can reserve a fraction of accepted pairs
    for defenders made exclusively from ships; the remainder stays mixed with
    at least one ship and one static defensive unit.
    """

    ship_only_ratio, base_args = _parse_task_options(args)
    current_mode = {"value": "mixed"}

    original_defense_generator = _base.generate_random_defense
    original_pair_generator = _base._generate_one_pair
    original_output_path = _base._default_output_path

    def generate_configured_defender(*generator_args, **generator_kwargs):
        options = dict(generator_kwargs)
        if current_mode["value"] == "ship_only":
            options.update(
                include_defenses=False,
                include_ships=True,
                require_ship=True,
                require_defense=False,
                include_shield_domes=False,
            )
        else:
            options.update(
                include_defenses=True,
                include_ships=True,
                require_ship=True,
                require_defense=True,
                include_shield_domes=True,
            )
        return generate_random_defender(*generator_args, **options)

    def generate_one_pair(*pair_args, **pair_kwargs):
        sequence_index = int(pair_kwargs["sequence_index"])
        mode = _mode_for_sequence_index(sequence_index, ship_only_ratio)
        current_mode["value"] = mode
        record = original_pair_generator(*pair_args, **pair_kwargs)
        if record is not None:
            record["generation"]["defender_mode"] = mode
            record["generation"]["ship_only_ratio"] = float(ship_only_ratio)
        return record

    _base.generate_random_defense = generate_configured_defender
    _base._generate_one_pair = generate_one_pair
    _base._default_output_path = _default_output_path
    try:
        print(
            "Defender generation among accepted pairs: "
            f"{ship_only_ratio:.1%} ship-only / "
            f"{1.0 - ship_only_ratio:.1%} mixed.",
            flush=True,
        )
        print(
            "Ship-only means no static defenses and no shield domes; "
            "mixed means at least one ship and one static defense.",
            flush=True,
        )
        return _base.run(base_args)
    finally:
        _base.generate_random_defense = original_defense_generator
        _base._generate_one_pair = original_pair_generator
        _base._default_output_path = original_output_path
