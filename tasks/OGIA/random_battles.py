from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Sequence
import uuid

import numpy as np

from .OgameBattleSimulator import CombatConfig, TechLevels
from .OgameUtils import (
    generate_fleet_from_percentages,
    generate_random_defense,
    generate_random_fleet_percentages,
    simulate_battle_with_profit,
)


DEFAULT_COUNT = 1_000
DEFAULT_MIN_POINTS = 100.0
DEFAULT_MAX_POINTS = 20_000.0
DEFAULT_ATTACKER_RATIO_MIN = 0.40
DEFAULT_ATTACKER_RATIO_MAX = 2.50
DEFAULT_PROGRESS_EVERY = 10


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogia-generate-random-battles",
        description=(
            "Generate random OGame battles across a broad point range and "
            "save each completed simulation incrementally as JSONL."
        ),
    )
    parser.add_argument(
        "--count",
        type=_positive_int,
        default=DEFAULT_COUNT,
        help=f"Number of successfully saved battles (default: {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--min-points",
        type=_positive_float,
        default=DEFAULT_MIN_POINTS,
        help=f"Minimum battle scale in OGame points (default: {DEFAULT_MIN_POINTS:g}).",
    )
    parser.add_argument(
        "--max-points",
        type=_positive_float,
        default=DEFAULT_MAX_POINTS,
        help=f"Maximum battle scale in OGame points (default: {DEFAULT_MAX_POINTS:g}).",
    )
    parser.add_argument(
        "--attacker-ratio-min",
        type=_positive_float,
        default=DEFAULT_ATTACKER_RATIO_MIN,
        help=(
            "Minimum attacker/defender target-point ratio "
            f"(default: {DEFAULT_ATTACKER_RATIO_MIN:g})."
        ),
    )
    parser.add_argument(
        "--attacker-ratio-max",
        type=_positive_float,
        default=DEFAULT_ATTACKER_RATIO_MAX,
        help=(
            "Maximum attacker/defender target-point ratio "
            f"(default: {DEFAULT_ATTACKER_RATIO_MAX:g})."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=_positive_int,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            "Print and fsync progress every N saved battles "
            f"(default: {DEFAULT_PROGRESS_EVERY})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible dataset generation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "JSONL output path. If omitted, a timestamped file is created "
            "under data/OGIA/random_battles/."
        ),
    )
    return parser


def _sample_log_uniform(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
) -> float:
    if minimum == maximum:
        return float(minimum)
    return float(math.exp(rng.uniform(math.log(minimum), math.log(maximum))))


def _sample_tech_levels(rng: np.random.Generator) -> TechLevels:
    return TechLevels(
        weapons=int(rng.integers(0, 21)),
        shielding=int(rng.integers(0, 21)),
        armour=int(rng.integers(0, 21)),
    )


def _sample_defender_resources(
    rng: np.random.Generator,
    defender_points: float,
) -> dict[str, int]:
    # One OGame point represents 1,000 invested resources. Planetary resources
    # are sampled around that scale and then split across metal/crystal/deuterium.
    total_scale = max(1.0, defender_points * 1_000.0)
    total_resources = int(
        round(total_scale * _sample_log_uniform(rng, 0.10, 2.00))
    )
    shares = rng.dirichlet(np.asarray([1.2, 1.2, 1.0], dtype=float))

    metal = int(round(total_resources * float(shares[0])))
    crystal = int(round(total_resources * float(shares[1])))
    deuterium = max(0, total_resources - metal - crystal)

    return {
        "metal": max(0, metal),
        "crystal": max(0, crystal),
        "deuterium": int(deuterium),
    }


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        Path("data")
        / "OGIA"
        / "random_battles"
        / f"random_battles_{timestamp}.jsonl"
    )


def _generate_one_battle(
    rng: np.random.Generator,
    *,
    min_points: float,
    max_points: float,
    attacker_ratio_min: float,
    attacker_ratio_max: float,
    run_seed: int | None,
    sequence_index: int,
) -> dict:
    defender_target_points = _sample_log_uniform(rng, min_points, max_points)
    attacker_ratio = _sample_log_uniform(
        rng,
        attacker_ratio_min,
        attacker_ratio_max,
    )
    attacker_target_points = min(
        max_points,
        max(min_points, defender_target_points * attacker_ratio),
    )

    defense_details = generate_random_defense(
        target_points=defender_target_points,
        min_unit_types=2,
        max_unit_types=6,
        include_shield_domes=True,
        rng=rng,
        return_details=True,
    )

    fleet_percentages = generate_random_fleet_percentages(
        min_ship_types=2,
        max_ship_types=6,
        include_large_cargo=True,
        minimum_percentage=3.0,
        rng=rng,
    )
    fleet_details = generate_fleet_from_percentages(
        fleet_percentages,
        points=attacker_target_points,
        percentage_basis="points",
        return_details=True,
        solver_threads=1,
    )

    attacker = fleet_details["fleet"]
    defender = defense_details["defense"]

    attacker_tech = _sample_tech_levels(rng)
    defender_tech = _sample_tech_levels(rng)
    defender_resources = _sample_defender_resources(
        rng,
        float(defense_details["actual_points"]),
    )
    battle_seed = int(rng.integers(0, 2**32 - 1))

    config = CombatConfig()
    result = simulate_battle_with_profit(
        attacker=attacker,
        defender=defender,
        attacker_tech=attacker_tech,
        defender_tech=defender_tech,
        config=config,
        seed=battle_seed,
        defender_resources=defender_resources,
        loot_percentage=0.75,
    )

    return {
        "schema_version": 1,
        "battle_id": uuid.uuid4().hex,
        "sequence_index": int(sequence_index),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "run_seed": run_seed,
            "battle_seed": battle_seed,
            "defender_target_points": float(defender_target_points),
            "attacker_target_points": float(attacker_target_points),
            "attacker_to_defender_target_ratio": float(attacker_ratio),
        },
        "inputs": {
            "attacker": attacker,
            "defender": defender,
            "attacker_tech": asdict(attacker_tech),
            "defender_tech": asdict(defender_tech),
            "defender_resources": defender_resources,
            "loot_percentage": 0.75,
            "combat_config": asdict(config),
        },
        "composition_details": {
            "attacker": fleet_details,
            "defender": defense_details,
        },
        "result": asdict(result),
    }


def run(args: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    options = parser.parse_args(list(args or []))

    min_points = float(options.min_points)
    max_points = float(options.max_points)
    ratio_min = float(options.attacker_ratio_min)
    ratio_max = float(options.attacker_ratio_max)

    if max_points < min_points:
        parser.error("--max-points must be greater than or equal to --min-points")
    if ratio_max < ratio_min:
        parser.error(
            "--attacker-ratio-max must be greater than or equal to "
            "--attacker-ratio-min"
        )

    output_path = (options.output or _default_output_path()).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(options.seed)
    saved = 0
    failed = 0
    attempts = 0
    max_attempts = max(options.count * 10, options.count + 100)

    print("OGIA random battle generator")
    print(f"Target battles: {options.count:,}")
    print(f"Point range: {min_points:g} - {max_points:g}")
    print(f"Attacker/defender ratio range: {ratio_min:g} - {ratio_max:g}")
    print(f"Output: {output_path}")
    print(f"Seed: {options.seed if options.seed is not None else 'random'}")

    try:
        with output_path.open("a", encoding="utf-8") as output_file:
            while saved < options.count:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        "Too many failed generation attempts "
                        f"({failed} failures for {saved} saved battles)."
                    )

                try:
                    record = _generate_one_battle(
                        rng,
                        min_points=min_points,
                        max_points=max_points,
                        attacker_ratio_min=ratio_min,
                        attacker_ratio_max=ratio_max,
                        run_seed=options.seed,
                        sequence_index=saved + 1,
                    )
                except Exception as exc:
                    failed += 1
                    print(
                        f"[warning] attempt {attempts:,} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                output_file.flush()
                saved += 1

                if (
                    saved == 1
                    or saved % options.progress_every == 0
                    or saved == options.count
                ):
                    os.fsync(output_file.fileno())
                    print(
                        f"Progress: {saved:,}/{options.count:,} "
                        f"battles generated and saved "
                        f"(failed attempts: {failed:,})",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print(
            f"\nStopped by user. {saved:,} battles were generated and saved "
            f"to {output_path}.",
            flush=True,
        )
        return 130

    print(
        f"Completed. {saved:,} battles generated and saved to {output_path}. "
        f"Failed attempts: {failed:,}.",
        flush=True,
    )
    return 0
