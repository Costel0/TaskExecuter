from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .OgameData import UNIT_SPECS


SHIELD_DOMES = ("small_shield_dome", "large_shield_dome")

DEFAULT_RANDOM_DEFENDER_DEFENSES: tuple[str, ...] = tuple(
    name
    for name, spec in UNIT_SPECS.items()
    if spec.is_defense and name not in SHIELD_DOMES
)
DEFAULT_RANDOM_DEFENDER_SHIPS: tuple[str, ...] = tuple(
    name for name, spec in UNIT_SPECS.items() if not spec.is_defense
)


def _unit_points(unit_name: str) -> float:
    if unit_name not in UNIT_SPECS:
        raise KeyError(f"Unidad desconocida: {unit_name!r}")
    return float(UNIT_SPECS[unit_name].cost) / 1_000.0


def defender_points(defender: Mapping[str, int]) -> float:
    """Return total OGame points in every unit defending the position.

    A defender may contain both static defenses and ships. This intentionally
    differs from the older ``OgameUtils.defense_points`` helper, which only
    accepts units whose category is ``defense``.
    """

    total = 0.0
    for unit_name, quantity in defender.items():
        if unit_name not in UNIT_SPECS:
            raise KeyError(f"Unidad desconocida: {unit_name!r}")
        if not isinstance(quantity, (int, np.integer)) or int(quantity) < 0:
            raise ValueError(
                f"La cantidad de {unit_name!r} debe ser un entero no negativo."
            )
        total += _unit_points(unit_name) * int(quantity)
    return float(total)


def _resolve_rng(
    *,
    seed: int | None,
    rng: np.random.Generator | None,
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Indica 'seed' o 'rng', pero no ambos.")
    return rng if rng is not None else np.random.default_rng(seed)


def _validated_ship_pool(allowed_ships: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for unit_name in allowed_ships:
        if unit_name not in UNIT_SPECS:
            raise KeyError(f"Nave desconocida: {unit_name!r}")
        if UNIT_SPECS[unit_name].is_defense:
            raise ValueError(f"{unit_name!r} no es una nave.")
        if unit_name not in result:
            result.append(str(unit_name))
    return tuple(result)


def generate_random_defender(
    target_points: float | None = None,
    *,
    min_points: float = 5_000,
    max_points: float = 40_000,
    min_unit_types: int = 3,
    max_unit_types: int = 6,
    include_ships: bool = True,
    require_ship: bool = False,
    require_defense: bool = False,
    allowed_ships: Sequence[str] = DEFAULT_RANDOM_DEFENDER_SHIPS,
    include_shield_domes: bool = True,
    shield_dome_probability: float = 0.35,
    concentration: float = 1.3,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_details: bool = False,
):
    """Generate a random defending composition containing defenses and/or ships.

    All ship types present in ``UNIT_SPECS`` are eligible by default. Units more
    expensive than the complete target budget are omitted for that sample, so
    very expensive ships such as Deathstars only appear at an appropriate scale.

    ``require_ship`` and ``require_defense`` can be used to force mixed samples;
    the normal random generator can still create defense-only, ship-only or mixed
    compositions when neither requirement is enabled.
    """

    generator = _resolve_rng(seed=seed, rng=rng)

    min_points = float(min_points)
    max_points = float(max_points)
    if not math.isfinite(min_points) or min_points <= 0:
        raise ValueError("min_points debe ser finito y mayor que cero.")
    if not math.isfinite(max_points) or max_points < min_points:
        raise ValueError("max_points debe ser finito y >= min_points.")

    if target_points is None:
        target_points = float(
            math.exp(generator.uniform(math.log(min_points), math.log(max_points)))
        )
    else:
        target_points = float(target_points)
    if not math.isfinite(target_points) or target_points <= 0:
        raise ValueError("target_points debe ser finito y mayor que cero.")

    if require_ship and not include_ships:
        raise ValueError("require_ship=True requiere include_ships=True.")
    if not math.isfinite(float(concentration)) or concentration <= 0:
        raise ValueError("concentration debe ser mayor que cero.")
    if not 0 <= shield_dome_probability <= 1:
        raise ValueError("shield_dome_probability debe estar entre 0 y 1.")

    ship_pool = _validated_ship_pool(allowed_ships) if include_ships else ()
    eligible_defenses = tuple(
        unit_name
        for unit_name in DEFAULT_RANDOM_DEFENDER_DEFENSES
        if _unit_points(unit_name) <= target_points
    )
    eligible_ships = tuple(
        unit_name
        for unit_name in ship_pool
        if _unit_points(unit_name) <= target_points
    )

    if require_defense and not eligible_defenses:
        raise ValueError("No hay defensas elegibles para el presupuesto solicitado.")
    if require_ship and not eligible_ships:
        raise ValueError("No hay naves elegibles para el presupuesto solicitado.")

    available_units = tuple(dict.fromkeys(eligible_defenses + eligible_ships))
    if not available_units:
        raise ValueError("No hay unidades elegibles para el presupuesto solicitado.")

    min_unit_types = int(min_unit_types)
    max_unit_types = int(max_unit_types)
    required_count = int(require_defense) + int(require_ship)
    if min_unit_types < max(1, required_count):
        raise ValueError(
            "min_unit_types es incompatible con las categorías obligatorias."
        )
    max_unit_types = min(max_unit_types, len(available_units))
    if max_unit_types < min_unit_types:
        raise ValueError(
            "max_unit_types debe ser >= min_unit_types y compatible con las "
            "unidades elegibles."
        )

    n_types = int(generator.integers(min_unit_types, max_unit_types + 1))
    selected_units: list[str] = []

    if require_defense:
        selected_units.append(str(generator.choice(eligible_defenses)))
    if require_ship:
        selected_units.append(str(generator.choice(eligible_ships)))

    remaining_pool = [
        unit_name for unit_name in available_units if unit_name not in selected_units
    ]
    n_remaining = n_types - len(selected_units)
    if n_remaining > 0:
        selected_units.extend(
            str(unit_name)
            for unit_name in generator.choice(
                remaining_pool,
                size=n_remaining,
                replace=False,
            )
        )

    point_shares = generator.dirichlet(
        np.full(len(selected_units), float(concentration))
    )

    defender: dict[str, int] = {}
    for unit_name, share in zip(selected_units, point_shares):
        allocated_points = target_points * float(share)
        quantity = max(1, int(round(allocated_points / _unit_points(unit_name))))
        defender[unit_name] = quantity

    if include_shield_domes:
        if generator.random() < shield_dome_probability:
            defender["small_shield_dome"] = 1
        if generator.random() < shield_dome_probability:
            defender["large_shield_dome"] = 1

    actual_points = defender_points(defender)
    if not return_details:
        return defender

    selected_ship_units = [
        unit_name for unit_name in selected_units if not UNIT_SPECS[unit_name].is_defense
    ]
    selected_defense_units = [
        unit_name for unit_name in selected_units if UNIT_SPECS[unit_name].is_defense
    ]

    return {
        "defense": defender,
        "target_points": float(target_points),
        "actual_points": float(actual_points),
        "point_difference": float(actual_points - target_points),
        "selected_regular_units": list(selected_units),
        "selected_defense_units": selected_defense_units,
        "selected_ship_units": selected_ship_units,
        "contains_ships": bool(selected_ship_units),
        "contains_static_defense": bool(selected_defense_units),
        "requested_point_shares": {
            unit_name: float(share * 100.0)
            for unit_name, share in zip(selected_units, point_shares)
        },
    }
