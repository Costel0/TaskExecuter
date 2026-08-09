from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .OgameUtils import (
    BattleResultWithProfit,
    CombatConfig,
    TechLevels,
)


@dataclass(frozen=True)
class GenerationMetadata:
    """Random-generation metadata used to create one stored battle."""

    run_seed: int | None
    battle_seed: int
    defender_target_points: float
    attacker_target_points: float
    attacker_to_defender_target_ratio: float


@dataclass(frozen=True)
class BattleInputs:
    """Fully interpreted simulator inputs for one battle."""

    attacker: dict[str, int]
    defender: dict[str, int]
    attacker_tech: TechLevels
    defender_tech: TechLevels
    defender_resources: dict[str, int]
    loot_percentage: float
    combat_config: CombatConfig


@dataclass(frozen=True)
class ShipCompositionDetails:
    count: int
    points_per_unit: float
    total_points: float
    requested_percentage: float
    actual_percentage: float


@dataclass(frozen=True)
class FleetCompositionDetails:
    fleet: dict[str, int]
    target_points: float
    actual_points: float
    point_difference: float
    absolute_point_difference: float
    percentage_basis: str
    requested_percentages: dict[str, float]
    actual_percentages: dict[str, float]
    per_ship: dict[str, ShipCompositionDetails]


@dataclass(frozen=True)
class DefenseCompositionDetails:
    defense: dict[str, int]
    target_points: float
    actual_points: float
    point_difference: float
    selected_regular_units: list[str]
    requested_point_shares: dict[str, float]


@dataclass(frozen=True)
class CompositionDetails:
    attacker: FleetCompositionDetails
    defender: DefenseCompositionDetails


@dataclass(frozen=True)
class BattleRecord:
    """One interpreted OGIA battle ready for use by other project code.

    JSONL is only the storage format. Once loaded, nested structures are turned
    into actual domain objects such as ``TechLevels``, ``CombatConfig`` and
    ``BattleResultWithProfit``.
    """

    schema_version: int
    battle_id: str
    sequence_index: int
    generated_at_utc: datetime
    generation: GenerationMetadata
    inputs: BattleInputs
    composition_details: CompositionDetails
    result: BattleResultWithProfit

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BattleRecord":
        """Build a typed battle from one JSON-compatible dictionary."""
        generation_data = _require_mapping(data, "generation")
        inputs_data = _require_mapping(data, "inputs")
        composition_data = _require_mapping(data, "composition_details")
        result_data = _require_mapping(data, "result")

        attacker_composition_data = _require_mapping(
            composition_data,
            "attacker",
        )
        defender_composition_data = _require_mapping(
            composition_data,
            "defender",
        )
        per_ship_data = _require_mapping(
            attacker_composition_data,
            "per_ship",
        )

        generation = GenerationMetadata(
            run_seed=generation_data.get("run_seed"),
            battle_seed=int(generation_data["battle_seed"]),
            defender_target_points=float(
                generation_data["defender_target_points"]
            ),
            attacker_target_points=float(
                generation_data["attacker_target_points"]
            ),
            attacker_to_defender_target_ratio=float(
                generation_data["attacker_to_defender_target_ratio"]
            ),
        )

        inputs = BattleInputs(
            attacker=_int_mapping(_require_mapping(inputs_data, "attacker")),
            defender=_int_mapping(_require_mapping(inputs_data, "defender")),
            attacker_tech=TechLevels(
                **_require_mapping(inputs_data, "attacker_tech")
            ),
            defender_tech=TechLevels(
                **_require_mapping(inputs_data, "defender_tech")
            ),
            defender_resources=_int_mapping(
                _require_mapping(inputs_data, "defender_resources")
            ),
            loot_percentage=float(inputs_data["loot_percentage"]),
            combat_config=CombatConfig(
                **_require_mapping(inputs_data, "combat_config")
            ),
        )

        per_ship = {
            str(ship_name): ShipCompositionDetails(
                count=int(ship_details["count"]),
                points_per_unit=float(ship_details["points_per_unit"]),
                total_points=float(ship_details["total_points"]),
                requested_percentage=float(
                    ship_details["requested_percentage"]
                ),
                actual_percentage=float(ship_details["actual_percentage"]),
            )
            for ship_name, raw_ship_details in per_ship_data.items()
            for ship_details in [_ensure_dict(raw_ship_details, "per_ship entry")]
        }

        attacker_composition = FleetCompositionDetails(
            fleet=_int_mapping(
                _require_mapping(attacker_composition_data, "fleet")
            ),
            target_points=float(attacker_composition_data["target_points"]),
            actual_points=float(attacker_composition_data["actual_points"]),
            point_difference=float(
                attacker_composition_data["point_difference"]
            ),
            absolute_point_difference=float(
                attacker_composition_data["absolute_point_difference"]
            ),
            percentage_basis=str(
                attacker_composition_data["percentage_basis"]
            ),
            requested_percentages=_float_mapping(
                _require_mapping(
                    attacker_composition_data,
                    "requested_percentages",
                )
            ),
            actual_percentages=_float_mapping(
                _require_mapping(
                    attacker_composition_data,
                    "actual_percentages",
                )
            ),
            per_ship=per_ship,
        )

        defender_composition = DefenseCompositionDetails(
            defense=_int_mapping(
                _require_mapping(defender_composition_data, "defense")
            ),
            target_points=float(defender_composition_data["target_points"]),
            actual_points=float(defender_composition_data["actual_points"]),
            point_difference=float(
                defender_composition_data["point_difference"]
            ),
            selected_regular_units=[
                str(unit_name)
                for unit_name in defender_composition_data[
                    "selected_regular_units"
                ]
            ],
            requested_point_shares=_float_mapping(
                _require_mapping(
                    defender_composition_data,
                    "requested_point_shares",
                )
            ),
        )

        result = BattleResultWithProfit(**result_data)

        return cls(
            schema_version=int(data["schema_version"]),
            battle_id=str(data["battle_id"]),
            sequence_index=int(data["sequence_index"]),
            generated_at_utc=_parse_datetime(data["generated_at_utc"]),
            generation=generation,
            inputs=inputs,
            composition_details=CompositionDetails(
                attacker=attacker_composition,
                defender=defender_composition,
            ),
            result=result,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the interpreted battle back to the JSON storage schema."""
        data = asdict(self)
        data["generated_at_utc"] = self.generated_at_utc.isoformat()
        return data


class BattleDataset(Sequence[BattleRecord]):
    """In-memory collection of interpreted battles.

    It behaves like a normal Python sequence, so it can be iterated, indexed,
    sliced and passed to future analysis/training code without exposing JSONL
    parsing details.
    """

    def __init__(self, battles: Iterable[BattleRecord] = ()) -> None:
        self._battles = list(battles)

    def __len__(self) -> int:
        return len(self._battles)

    def __iter__(self) -> Iterator[BattleRecord]:
        return iter(self._battles)

    def __getitem__(self, index):
        return self._battles[index]

    @property
    def battles(self) -> list[BattleRecord]:
        """Return the underlying list of interpreted battles."""
        return self._battles

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        limit: int | None = None,
    ) -> "BattleDataset":
        return cls(load_battles(path, limit=limit))

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        recursive: bool = False,
        limit: int | None = None,
    ) -> "BattleDataset":
        return cls(
            load_battles_from_directory(
                directory,
                recursive=recursive,
                limit=limit,
            )
        )


def _ensure_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected {label} to be a JSON object, got {type(value).__name__}."
        )
    return value


def _require_mapping(container: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in container:
        raise ValueError(f"Missing required battle field: {key}")
    return _ensure_dict(container[key], key)


def _int_mapping(data: dict[str, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in data.items()}


def _float_mapping(data: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in data.items()}


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("generated_at_utc must be an ISO-8601 string")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"Invalid generated_at_utc timestamp: {value!r}"
        ) from exc


def _parse_jsonl_line(
    raw_line: str,
    *,
    dataset_path: Path,
    line_number: int,
) -> BattleRecord:
    try:
        parsed = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {dataset_path} at line {line_number}: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object in {dataset_path} at line "
            f"{line_number}, got {type(parsed).__name__}."
        )

    try:
        return BattleRecord.from_dict(parsed)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid battle record in {dataset_path} at line "
            f"{line_number}: {exc}"
        ) from exc


def iter_battles(path: str | Path) -> Iterator[BattleRecord]:
    """Yield interpreted battles from one OGIA JSONL file."""
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Battle dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            yield _parse_jsonl_line(
                line,
                dataset_path=dataset_path,
                line_number=line_number,
            )


def find_battle_files(
    directory: str | Path,
    *,
    recursive: bool = False,
    exclude: Iterable[str | Path] = (),
) -> list[Path]:
    """Return sorted JSONL battle files in a directory."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"Battle dataset directory not found: {root}")

    excluded = {
        Path(path).expanduser().resolve()
        for path in exclude
    }
    iterator = root.rglob("*.jsonl") if recursive else root.glob("*.jsonl")

    return sorted(
        path
        for path in iterator
        if path.is_file() and path.resolve() not in excluded
    )


def iter_battles_from_directory(
    directory: str | Path,
    *,
    recursive: bool = False,
) -> Iterator[BattleRecord]:
    """Yield interpreted battles across every JSONL file in a directory."""
    files = find_battle_files(directory, recursive=recursive)
    for dataset_path in files:
        yield from iter_battles(dataset_path)


def load_battles(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[BattleRecord]:
    """Load one JSONL file into a list of interpreted battle objects."""
    _validate_limit(limit)

    battles: list[BattleRecord] = []
    for battle in iter_battles(path):
        battles.append(battle)
        if limit is not None and len(battles) >= limit:
            break
    return battles


def load_battles_from_directory(
    directory: str | Path,
    *,
    recursive: bool = False,
    limit: int | None = None,
) -> list[BattleRecord]:
    """Combine all JSONL files in a folder into one Python battle list."""
    _validate_limit(limit)

    battles: list[BattleRecord] = []
    for battle in iter_battles_from_directory(
        directory,
        recursive=recursive,
    ):
        battles.append(battle)
        if limit is not None and len(battles) >= limit:
            break
    return battles


def write_battles(
    battles: Iterable[BattleRecord],
    path: str | Path,
) -> int:
    """Write interpreted battles to JSONL and return the number written."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for battle in battles:
            output_file.write(
                json.dumps(
                    battle.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def merge_battle_directory(
    directory: str | Path,
    output_path: str | Path,
    *,
    recursive: bool = False,
) -> tuple[int, int]:
    """Merge all JSONL battle files in a folder into one validated JSONL file.

    Returns ``(number_of_source_files, number_of_battles)``. The output file is
    excluded from the input set, so the command can safely be re-run in the
    same directory without merging its previous output into itself.
    """
    output = Path(output_path).expanduser()
    source_files = find_battle_files(
        directory,
        recursive=recursive,
        exclude=[output],
    )
    if not source_files:
        raise FileNotFoundError(
            f"No .jsonl battle datasets found in {Path(directory).expanduser()}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")

    battle_count = 0
    try:
        with temporary_output.open("w", encoding="utf-8") as output_file:
            for source_path in source_files:
                for battle in iter_battles(source_path):
                    output_file.write(
                        json.dumps(
                            battle.to_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    battle_count += 1
        temporary_output.replace(output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    return len(source_files), battle_count


def _validate_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("limit must be a positive integer or None")


__all__ = [
    "BattleDataset",
    "BattleInputs",
    "BattleRecord",
    "CompositionDetails",
    "DefenseCompositionDetails",
    "FleetCompositionDetails",
    "GenerationMetadata",
    "ShipCompositionDetails",
    "find_battle_files",
    "iter_battles",
    "iter_battles_from_directory",
    "load_battles",
    "load_battles_from_directory",
    "merge_battle_directory",
    "write_battles",
]
