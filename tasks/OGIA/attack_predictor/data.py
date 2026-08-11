from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..OgameData import UNIT_SPECS
from ..optimizer import EvolutionConfig


DEFENSE_UNITS: tuple[str, ...] = tuple(
    name for name, spec in UNIT_SPECS.items() if spec.is_defense
)
ATTACK_SHIPS: tuple[str, ...] = EvolutionConfig().allowed_ships


@dataclass(frozen=True)
class PerfectPairExample:
    pair_id: str
    defender: Mapping[str, int]
    ship_weights: Mapping[str, float]
    points_multiplier: float
    source_path: str
    line_number: int
    attacker_tech: Mapping[str, Any] = field(default_factory=dict)
    defender_tech: Mapping[str, Any] = field(default_factory=dict)
    combat_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureScaler":
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("FeatureScaler.fit expects a non-empty 2D array.")
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return ((values - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureScaler":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


def feature_names(defense_units: Sequence[str] = DEFENSE_UNITS) -> tuple[str, ...]:
    return tuple(f"share_{unit}" for unit in defense_units) + (
        "log1p_total_defense_points",
    )


def defender_to_features(
    defender: Mapping[str, int],
    *,
    defense_units: Sequence[str] = DEFENSE_UNITS,
) -> np.ndarray:
    allowed = set(defense_units)
    unknown = set(defender) - allowed
    if unknown:
        raise ValueError(f"Defender contains unsupported units: {sorted(unknown)}")

    point_values: list[float] = []
    for unit_name in defense_units:
        quantity = defender.get(unit_name, 0)
        if not isinstance(quantity, (int, np.integer)) or int(quantity) < 0:
            raise ValueError(f"Invalid quantity for {unit_name!r}: {quantity!r}")
        spec = UNIT_SPECS[unit_name]
        points = (float(spec.cost) / 1_000.0) * int(quantity)
        point_values.append(points)

    total_points = float(sum(point_values))
    if total_points <= 0:
        raise ValueError("Defender must contain at least one defensive unit.")

    shares = np.asarray(point_values, dtype=np.float32) / total_points
    return np.concatenate(
        [shares, np.asarray([math.log1p(total_points)], dtype=np.float32)]
    ).astype(np.float32)


def defense_group_key(
    example: PerfectPairExample,
    *,
    defense_units: Sequence[str] = DEFENSE_UNITS,
) -> tuple[int, ...]:
    """Key by exactly what the predictor sees as the physical defense.

    Tech levels are deliberately not part of this key because they are not model
    inputs. Repeated versions of the same defense must therefore stay in the same
    cross-validation fold to avoid leakage.
    """

    return tuple(int(example.defender.get(unit, 0)) for unit in defense_units)


def _normalise_ship_weights(
    values: Mapping[str, Any],
    *,
    attack_ships: Sequence[str],
) -> dict[str, float]:
    unknown = set(values) - set(attack_ships)
    if unknown:
        raise ValueError(f"Target contains unsupported ships: {sorted(unknown)}")

    weights: dict[str, float] = {}
    for ship_name in attack_ships:
        value = float(values.get(ship_name, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid target weight for {ship_name!r}: {value!r}")
        weights[ship_name] = value

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Target ship weights must sum to a positive value.")
    return {name: value / total for name, value in weights.items()}


def _mapping_field(
    parent: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    value = parent.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"inputs.{name} must be a mapping when present.")
    return {str(key): item for key, item in value.items()}


def _parse_pair(
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    line_number: int,
    attack_ships: Sequence[str],
) -> PerfectPairExample:
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"Unsupported perfect-pair schema_version={schema_version}")

    inputs = payload.get("inputs")
    target = payload.get("target")
    if not isinstance(inputs, Mapping) or not isinstance(target, Mapping):
        raise ValueError("Perfect pair must contain mapping 'inputs' and 'target' fields.")

    defender = inputs.get("defender")
    ship_weights = target.get("ship_weights")
    if not isinstance(defender, Mapping) or not isinstance(ship_weights, Mapping):
        raise ValueError(
            "Perfect pair is missing inputs.defender or target.ship_weights."
        )

    clean_defender = {
        str(name): int(quantity) for name, quantity in defender.items()
    }
    defender_to_features(clean_defender)

    multiplier = float(target.get("points_multiplier"))
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f"Invalid target points_multiplier={multiplier!r}")

    return PerfectPairExample(
        pair_id=str(payload.get("pair_id") or f"{source_path.name}:{line_number}"),
        defender=clean_defender,
        ship_weights=_normalise_ship_weights(
            ship_weights,
            attack_ships=attack_ships,
        ),
        points_multiplier=multiplier,
        source_path=str(source_path),
        line_number=line_number,
        attacker_tech=_mapping_field(inputs, "attacker_tech"),
        defender_tech=_mapping_field(inputs, "defender_tech"),
        combat_config=_mapping_field(inputs, "combat_config"),
    )


def _stable_mapping(mapping: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(mapping),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def perfect_pair_fingerprint(
    example: PerfectPairExample,
    *,
    attack_ships: Sequence[str] = ATTACK_SHIPS,
) -> tuple[Any, ...]:
    """Identify exact repeated optimizer results independent of pair_id/file."""

    return (
        defense_group_key(example),
        tuple(
            round(float(example.ship_weights.get(ship, 0.0)), 12)
            for ship in attack_ships
        ),
        round(float(example.points_multiplier), 12),
        _stable_mapping(example.attacker_tech),
        _stable_mapping(example.defender_tech),
        _stable_mapping(example.combat_config),
    )


def deduplicate_perfect_pairs(
    examples: Iterable[PerfectPairExample],
    *,
    attack_ships: Sequence[str] = ATTACK_SHIPS,
) -> tuple[PerfectPairExample, ...]:
    """Keep the first occurrence of every exact defense/attack training pair."""

    unique: list[PerfectPairExample] = []
    seen: set[tuple[Any, ...]] = set()
    for example in examples:
        fingerprint = perfect_pair_fingerprint(
            example,
            attack_ships=attack_ships,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(example)
    return tuple(unique)


def resolve_pair_files(path: str | Path) -> tuple[Path, ...]:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() != ".jsonl":
            raise ValueError(f"Expected a .jsonl file, got: {path}")
        return (path,)
    if path.is_dir():
        files = tuple(
            sorted(
                candidate
                for candidate in path.glob("*.jsonl")
                if candidate.is_file()
            )
        )
        if not files:
            raise FileNotFoundError(f"No .jsonl perfect-pair files found in {path}")
        return files
    raise FileNotFoundError(f"Perfect-pair input path does not exist: {path}")


def load_perfect_pairs(
    path: str | Path,
    *,
    attack_ships: Sequence[str] = ATTACK_SHIPS,
    deduplicate: bool = True,
) -> tuple[PerfectPairExample, ...]:
    rows: list[PerfectPairExample] = []
    for source_path in resolve_pair_files(path):
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    rows.append(
                        _parse_pair(
                            payload,
                            source_path=source_path,
                            line_number=line_number,
                            attack_ships=attack_ships,
                        )
                    )
                except Exception as exc:
                    raise ValueError(
                        f"Invalid perfect pair at {source_path}:{line_number}: {exc}"
                    ) from exc
    if not rows:
        raise ValueError(f"No perfect pairs found in {path}")
    if deduplicate:
        return deduplicate_perfect_pairs(rows, attack_ships=attack_ships)
    return tuple(rows)


def examples_to_arrays(
    examples: Iterable[PerfectPairExample],
    *,
    defense_units: Sequence[str] = DEFENSE_UNITS,
    attack_ships: Sequence[str] = ATTACK_SHIPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = tuple(examples)
    if not rows:
        raise ValueError("At least one perfect-pair example is required.")

    features = np.stack(
        [
            defender_to_features(
                row.defender,
                defense_units=defense_units,
            )
            for row in rows
        ]
    ).astype(np.float32)
    compositions = np.asarray(
        [
            [row.ship_weights.get(ship, 0.0) for ship in attack_ships]
            for row in rows
        ],
        dtype=np.float32,
    )
    multipliers = np.asarray(
        [row.points_multiplier for row in rows],
        dtype=np.float32,
    )
    return features, compositions, multipliers
