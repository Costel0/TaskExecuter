from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..defender import defender_points
from ..fast_fleet import generate_fleet_from_percentages_fast
from .models import AttackGenome


@dataclass(frozen=True)
class DecodedAttack:
    genome: AttackGenome
    fleet: Mapping[str, int]
    defender_points: float
    target_attacker_points: float
    actual_attacker_points: float


def decode_attack_genome(
    genome: AttackGenome,
    defender: Mapping[str, int],
    *,
    allowed_ships: Sequence[str],
) -> DecodedAttack:
    """Convert the compact genome into concrete integer ship counts.

    This is deterministic and deliberately separate from fitness. Fitness can
    therefore change without changing the genome representation or allocation
    logic. The defender may contain both static defenses and ships.
    """

    defender_total_points = defender_points(defender)
    target_attacker_points = defender_total_points * float(genome.points_multiplier)

    details = generate_fleet_from_percentages_fast(
        genome.percentages(allowed_ships),
        points=target_attacker_points,
        percentage_basis="points",
        return_details=True,
        preserve_selected_types=True,
    )

    return DecodedAttack(
        genome=genome,
        fleet=details["fleet"],
        defender_points=float(defender_total_points),
        target_attacker_points=float(target_attacker_points),
        actual_attacker_points=float(details["actual_points"]),
    )
