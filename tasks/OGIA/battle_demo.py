from __future__ import annotations

import argparse
from typing import Sequence

from .OgameBattleSimulator import CombatConfig, TechLevels, simulate_battle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small deterministic OGame battle demo.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used by the combat simulator (default: 42).",
    )
    return parser


def run(args: Sequence[str] | None = None) -> int:
    parsed = _build_parser().parse_args(list(args or []))

    attacker = {
        "light_fighter": 100,
        "cruiser": 20,
    }
    defender = {
        "rocket_launcher": 75,
        "light_laser": 25,
        "heavy_laser": 5,
    }

    attacker_tech = TechLevels(weapons=10, shielding=10, armour=10)
    defender_tech = TechLevels(weapons=10, shielding=10, armour=10)

    config = CombatConfig(
        fleet_debris_fraction=0.30,
        defense_debris_fraction=0.00,
        rebuild_defense=True,
    )

    print("=== OGIA battle demo ===")
    print(f"Seed: {parsed.seed}")
    print(f"Attacker: {attacker}")
    print(f"Defender: {defender}")

    result = simulate_battle(
        attacker=attacker,
        defender=defender,
        attacker_tech=attacker_tech,
        defender_tech=defender_tech,
        config=config,
        seed=parsed.seed,
        defender_resources={
            "metal": 1_000_000,
            "crystal": 500_000,
            "deuterium": 250_000,
        },
        loot_percentage=0.75,
    )

    print("\n=== Result ===")
    print(f"Winner: {result.winner}")
    print(f"Rounds: {result.rounds}")
    print(f"Attacker survivors: {result.attacker_survivors}")
    print(
        "Defender survivors before rebuild: "
        f"{result.defender_survivors_before_rebuild}"
    )
    print(f"Defender rebuilt: {result.defender_rebuilt}")
    print(f"Defender survivors after rebuild: {result.defender_survivors}")
    print(f"Loot: {result.loot}")
    print(f"Debris generated: {result.debris_generated}")
    print(f"Debris remaining: {result.debris_remaining}")

    return 0
