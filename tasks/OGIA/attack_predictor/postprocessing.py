from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PredictionPostprocessConfig:
    """Convert dense neural outputs into optimizer-friendly attack genomes."""

    min_ship_weight: float = 0.01
    max_ship_types: int = 5
    multiplier_safety_margin: float = 0.03

    def __post_init__(self) -> None:
        if not 0 <= self.min_ship_weight < 1:
            raise ValueError("min_ship_weight must be in [0, 1).")
        if self.max_ship_types < 1:
            raise ValueError("max_ship_types must be at least 1.")
        if self.multiplier_safety_margin < 0:
            raise ValueError("multiplier_safety_margin cannot be negative.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "PredictionPostprocessConfig":
        payload = payload or {}
        return cls(
            min_ship_weight=float(payload.get("min_ship_weight", 0.01)),
            max_ship_types=int(payload.get("max_ship_types", 5)),
            multiplier_safety_margin=float(
                payload.get("multiplier_safety_margin", 0.03)
            ),
        )


def sparsify_ship_weights(
    weights: Mapping[str, float],
    *,
    attack_ships: Sequence[str],
    config: PredictionPostprocessConfig,
) -> dict[str, float]:
    """Drop softmax dust, keep only the strongest useful ship types, renormalize."""

    values = {
        ship: max(0.0, float(weights.get(ship, 0.0)))
        for ship in attack_ships
    }
    if not values:
        raise ValueError("attack_ships cannot be empty.")

    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    selected = [
        (ship, value)
        for ship, value in ranked
        if value >= config.min_ship_weight
    ][: config.max_ship_types]

    # Softmax always has a maximum, but keep this fallback for future heads.
    if not selected:
        selected = [ranked[0]]

    total = sum(value for _, value in selected)
    if total <= 0:
        return {
            ship: (1.0 if ship == selected[0][0] else 0.0)
            for ship in attack_ships
        }

    selected_map = {ship: value / total for ship, value in selected}
    return {
        ship: float(selected_map.get(ship, 0.0))
        for ship in attack_ships
    }


def postprocess_prediction(
    weights: Mapping[str, float],
    multiplier: float,
    *,
    attack_ships: Sequence[str],
    config: PredictionPostprocessConfig,
    max_multiplier: float,
) -> tuple[dict[str, float], float]:
    clean_weights = sparsify_ship_weights(
        weights,
        attack_ships=attack_ships,
        config=config,
    )
    clean_multiplier = min(
        float(max_multiplier),
        float(multiplier) + config.multiplier_safety_margin,
    )
    return clean_weights, float(clean_multiplier)
