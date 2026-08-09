from __future__ import annotations

import math
from typing import Dict, Mapping

from .OgameData import UNIT_SPECS


def _unit_points(unit_name: str) -> float:
    if unit_name not in UNIT_SPECS:
        raise KeyError(f"Nave desconocida: {unit_name!r}")
    spec = UNIT_SPECS[unit_name]
    if spec.category != "ship":
        raise ValueError(f"{unit_name!r} no pertenece a la categoría ship.")
    return float(spec.metal + spec.crystal + spec.deuterium) / 1_000.0


def _normalise_percentages(
    ship_percentages: Mapping[str, float],
) -> tuple[list[str], list[float]]:
    if not ship_percentages:
        raise ValueError("Debes indicar al menos un tipo de nave.")

    ship_names: list[str] = []
    weights: list[float] = []

    for ship_name, raw_value in ship_percentages.items():
        point_value = _unit_points(ship_name)
        if point_value <= 0:
            raise ValueError(f"{ship_name!r} debe tener un coste positivo.")

        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(
                f"El porcentaje de {ship_name!r} debe ser un número finito."
            )
        if value < 0:
            raise ValueError(
                f"El porcentaje de {ship_name!r} no puede ser negativo."
            )
        if value == 0:
            continue

        ship_names.append(ship_name)
        weights.append(value)

    if not weights:
        raise ValueError("La suma de los porcentajes debe ser mayor que cero.")

    total = sum(weights)
    shares = [value / total for value in weights]
    return ship_names, shares


def _actual_shares(
    counts: list[int],
    point_values: list[float],
    percentage_basis: str,
) -> list[float]:
    if percentage_basis == "points":
        values = [count * points for count, points in zip(counts, point_values)]
    else:
        values = [float(count) for count in counts]

    total = sum(values)
    if total <= 0:
        return [0.0] * len(values)
    return [value / total for value in values]


def _allocation_score(
    counts: list[int],
    point_values: list[float],
    requested_shares: list[float],
    target_points: float,
    percentage_basis: str,
) -> float:
    actual_points = sum(
        count * points for count, points in zip(counts, point_values)
    )
    point_scale = max(target_points, min(point_values), 1.0)
    point_error = abs(actual_points - target_points) / point_scale

    actual_shares = _actual_shares(counts, point_values, percentage_basis)
    composition_error = sum(
        abs(actual - requested)
        for actual, requested in zip(actual_shares, requested_shares)
    )

    # Staying near the requested fleet size matters most. The composition term
    # prevents the cheap correction pass from drifting far from the sampled
    # percentages merely to save a handful of points.
    return 3.0 * point_error + composition_error


def generate_fleet_from_percentages_fast(
    ship_percentages: Mapping[str, float],
    points: float,
    *,
    percentage_basis: str = "points",
    return_details: bool = False,
    allow_empty: bool = False,
    preserve_selected_types: bool = True,
    max_adjustments: int = 64,
):
    """Build an approximately sized fleet without integer optimisation.

    This function is intended for large random simulation datasets where the
    requested point budget and composition percentages are targets rather than
    hard constraints. It keeps the same output schema as
    ``OgameUtils.generate_fleet_from_percentages`` but avoids the two MILP
    solves previously required for every generated battle.

    The initial allocation is a direct arithmetic approximation. A short greedy
    correction pass then adds or removes individual ships only when doing so
    improves a combined point/composition score.

    When ``preserve_selected_types`` is enabled and the target budget can afford
    at least one unit of every selected ship type, every selected type is kept in
    the resulting fleet. This preserves the structural diversity sampled by the
    random fleet generator.
    """
    target_points = float(points)
    if not math.isfinite(target_points) or target_points < 0:
        raise ValueError("'points' debe ser un número finito no negativo.")

    if percentage_basis not in {"units", "points"}:
        raise ValueError("'percentage_basis' debe ser 'units' o 'points'.")

    if not isinstance(max_adjustments, int) or max_adjustments < 0:
        raise ValueError("'max_adjustments' debe ser un entero no negativo.")

    ship_names, requested_shares = _normalise_percentages(ship_percentages)
    point_values = [_unit_points(ship_name) for ship_name in ship_names]

    if target_points == 0:
        counts = [0] * len(ship_names)
    elif percentage_basis == "points":
        counts = [
            max(0, int(round(target_points * share / point_value)))
            for share, point_value in zip(requested_shares, point_values)
        ]
    else:
        weighted_points_per_ship = sum(
            share * point_value
            for share, point_value in zip(requested_shares, point_values)
        )
        estimated_total_units = target_points / weighted_points_per_ship
        counts = [
            max(0, int(round(estimated_total_units * share)))
            for share in requested_shares
        ]

    minimum_counts = [0] * len(ship_names)
    one_of_each_points = sum(point_values)
    if (
        preserve_selected_types
        and target_points > 0
        and target_points >= one_of_each_points
    ):
        minimum_counts = [1] * len(ship_names)
        counts = [max(1, count) for count in counts]

    if not allow_empty and target_points > 0 and not any(counts):
        # Pick the requested type whose single unit is closest to the target.
        best_index = min(
            range(len(ship_names)),
            key=lambda index: abs(point_values[index] - target_points),
        )
        counts[best_index] = 1

    current_score = _allocation_score(
        counts,
        point_values,
        requested_shares,
        target_points,
        percentage_basis,
    )

    # Rounding each type independently already lands close to the requested
    # budget. The bounded greedy pass only needs to repair the small residue, so
    # its cost depends on the number of selected ship types, not fleet size.
    for _ in range(max_adjustments):
        best_score = current_score
        best_change: tuple[int, int] | None = None

        for index in range(len(counts)):
            for delta in (-1, 1):
                candidate_count = counts[index] + delta
                if candidate_count < minimum_counts[index] or candidate_count < 0:
                    continue

                counts[index] = candidate_count
                candidate_score = _allocation_score(
                    counts,
                    point_values,
                    requested_shares,
                    target_points,
                    percentage_basis,
                )
                counts[index] -= delta

                if candidate_score + 1e-12 < best_score:
                    best_score = candidate_score
                    best_change = (index, delta)

        if best_change is None:
            break

        index, delta = best_change
        counts[index] += delta
        current_score = best_score

    fleet = {
        ship_name: int(count)
        for ship_name, count in zip(ship_names, counts)
        if count > 0
    }

    actual_points = sum(
        count * point_value
        for count, point_value in zip(counts, point_values)
    )
    point_difference = actual_points - target_points
    actual_shares = _actual_shares(counts, point_values, percentage_basis)

    if not return_details:
        return fleet

    requested_percentages = {
        ship_name: float(share * 100.0)
        for ship_name, share in zip(ship_names, requested_shares)
    }
    actual_percentages = {
        ship_name: float(share * 100.0)
        for ship_name, share in zip(ship_names, actual_shares)
    }
    per_ship = {
        ship_name: {
            "count": int(count),
            "points_per_unit": float(point_value),
            "total_points": float(count * point_value),
            "requested_percentage": requested_percentages[ship_name],
            "actual_percentage": actual_percentages[ship_name],
        }
        for ship_name, count, point_value in zip(
            ship_names,
            counts,
            point_values,
        )
    }

    return {
        "fleet": fleet,
        "target_points": target_points,
        "actual_points": float(actual_points),
        "point_difference": float(point_difference),
        "absolute_point_difference": float(abs(point_difference)),
        "percentage_basis": percentage_basis,
        "requested_percentages": requested_percentages,
        "actual_percentages": actual_percentages,
        "per_ship": per_ship,
    }
