#!/usr/bin/env python
# coding: utf-8

# # OGame Battle Simulator — reference implementation
#
# Motor de combate modular en Python. Los datos estáticos de unidades y fuego
# rápido se importan desde `OgameData`. El estado de combate se almacena de
# forma comprimida para poder manejar flotas grandes sin crear un objeto Python
# por cada nave/defensa ni reconstruir la lista de objetivos vivos por disparo.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional
import importlib
import math
import random
import statistics
import subprocess
import sys

__all__ = [
    "UnitSpec", "TechLevels", "CombatConfig", "BattleResult",
    "UNIT_SPECS", "RAPID_FIRE", "simulate_battle", "simulate_many",
    "build_side", "fleet_cost", "fleet_cargo_capacity", "survivor_counts",
    "calculate_loot", "validate_reference_data"
]


# ## 1. Importar los datos comunes


def _project_directory() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def _find_data_notebook() -> Path:
    local_candidate = _project_directory() / "OgameData.ipynb"
    if local_candidate.exists():
        return local_candidate

    drive_root = Path("/content/drive")
    if drive_root.exists():
        matches = list(drive_root.rglob("OgameData.ipynb"))
        if matches:
            matches.sort(key=lambda path: ("OWiki" not in path.parts, len(path.parts)))
            return matches[0]

    raise FileNotFoundError(
        "No se ha encontrado OgameData.py ni OgameData.ipynb. "
        "Convierte OgameData.ipynb a Python o añade su carpeta a sys.path."
    )


def _import_ogame_data():
    project_dir = _project_directory()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    try:
        return importlib.import_module("OgameData")
    except ModuleNotFoundError as exc:
        if exc.name != "OgameData":
            raise

        data_notebook = _find_data_notebook()
        data_dir = data_notebook.parent
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "python",
                str(data_notebook),
                "--output-dir",
                str(data_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if str(data_dir) not in sys.path:
            sys.path.insert(0, str(data_dir))

        return importlib.import_module("OgameData")


_ogame_data = _import_ogame_data()
UnitSpec = _ogame_data.UnitSpec
UNIT_SPECS = _ogame_data.UNIT_SPECS
RAPID_FIRE = _ogame_data.RAPID_FIRE
RESOURCE_KEYS = _ogame_data.RESOURCE_KEYS
validate_reference_data = _ogame_data.validate_reference_data


# ## 2. Tecnologías, configuración y estado interno


@dataclass(frozen=True)
class TechLevels:
    weapons: int = 0
    shielding: int = 0
    armour: int = 0

    def weapon_multiplier(self) -> float:
        return 1.0 + 0.1 * self.weapons

    def shield_multiplier(self) -> float:
        return 1.0 + 0.1 * self.shielding

    def hull_multiplier(self) -> float:
        return 1.0 + 0.1 * self.armour


@dataclass(frozen=True)
class CombatConfig:
    max_rounds: int = 6
    shield_bounce_fraction: float = 0.01
    explosion_hull_threshold: float = 0.70
    defense_rebuild_probability: float = 0.70
    use_rapid_fire: bool = True
    rebuild_defense: bool = True

    # Universe economy settings. Standard universes usually generate fleet
    # debris from metal and crystal only. Some universes use other fractions.
    fleet_debris_fraction: float = 0.30
    defense_debris_fraction: float = 0.00

    # A surviving Reaper fleet can automatically collect up to this fraction
    # of the debris generated in the battle, constrained by Reaper cargo.
    reaper_harvest_fraction: float = 0.25


@dataclass(slots=True)
class _UnitTypeState:
    """Compressed combat state for all living units of one type.

    A pristine unit does not need a Python object of its own. Only units whose
    current hull or shield differs from the default state are represented in
    ``damaged``. Its integer keys are dense logical indices in ``range(count)``.
    """

    kind: str
    count: int
    max_hull: float
    max_shield: float
    weapon: float
    is_defense: bool
    damaged: Dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass(slots=True)
class BattleSideState:
    """Compressed state of one side of a battle."""

    unit_types: list[_UnitTypeState]
    total_alive: int


@dataclass
class BattleResult:
    winner: str
    rounds: int
    attacker_initial: Dict[str, int]
    defender_initial: Dict[str, int]
    attacker_survivors: Dict[str, int]
    defender_survivors_before_rebuild: Dict[str, int]
    defender_survivors: Dict[str, int]
    defender_rebuilt: Dict[str, int]
    shots_by_attacker: int
    shots_by_defender: int

    defender_resources: Dict[str, int]
    loot_percentage: float
    attacker_cargo_capacity: int
    attacker_cargo_used_by_reapers: int
    attacker_cargo_available_for_loot: int
    loot: Dict[str, int]

    debris_generated: Dict[str, int]
    attacker_reaper_harvest: Dict[str, int]
    defender_reaper_harvest: Dict[str, int]
    debris_remaining: Dict[str, int]

    seed: Optional[int] = None

    @property
    def attacker_destroyed(self) -> Dict[str, int]:
        return _subtract_counts(self.attacker_initial, self.attacker_survivors)

    @property
    def defender_destroyed_before_rebuild(self) -> Dict[str, int]:
        return _subtract_counts(
            self.defender_initial,
            self.defender_survivors_before_rebuild,
        )

    @property
    def loot_total(self) -> int:
        return sum(self.loot.values())

    @property
    def debris_generated_total(self) -> int:
        return sum(self.debris_generated.values())

    @property
    def debris_remaining_total(self) -> int:
        return sum(self.debris_remaining.values())


def _subtract_counts(initial: Mapping[str, int], survivors: Mapping[str, int]) -> Dict[str, int]:
    return {
        k: int(v - survivors.get(k, 0))
        for k, v in initial.items()
        if v - survivors.get(k, 0) > 0
    }


def _normalise_fraction(value: float, name: str) -> float:
    """Accept either 0.75 or 75 and return a fraction in [0, 1]."""
    value = float(value)
    if 1.0 < value <= 100.0:
        value /= 100.0
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, or between 0 and 100")
    return value


def _normalise_resources(resources: Optional[Mapping[str, int]]) -> Dict[str, int]:
    resources = resources or {}
    unknown = set(resources) - set(RESOURCE_KEYS)
    if unknown:
        raise KeyError(f"Unknown resource keys: {sorted(unknown)}")

    out: Dict[str, int] = {}
    for key in RESOURCE_KEYS:
        value = resources.get(key, 0)
        if int(value) != value or value < 0:
            raise ValueError(f"Invalid amount for {key}: {value}")
        out[key] = int(value)
    return out


def build_side(composition: Mapping[str, int], tech: TechLevels) -> BattleSideState:
    """Build one combat side without allocating one object per unit."""
    unit_types: list[_UnitTypeState] = []
    total_alive = 0

    for kind, count in composition.items():
        if kind not in UNIT_SPECS:
            raise KeyError(f"Unknown unit: {kind}")
        if int(count) != count or count < 0:
            raise ValueError(f"Invalid count for {kind}: {count}")

        count = int(count)
        if count == 0:
            continue

        spec = UNIT_SPECS[kind]
        unit_types.append(
            _UnitTypeState(
                kind=kind,
                count=count,
                max_hull=spec.hull * tech.hull_multiplier(),
                max_shield=spec.shield * tech.shield_multiplier(),
                weapon=spec.weapon * tech.weapon_multiplier(),
                is_defense=spec.is_defense,
            )
        )
        total_alive += count

    return BattleSideState(unit_types=unit_types, total_alive=total_alive)


def survivor_counts(side: BattleSideState) -> Dict[str, int]:
    return {
        unit_type.kind: unit_type.count
        for unit_type in side.unit_types
        if unit_type.count > 0
    }


def fleet_cost(composition: Mapping[str, int]) -> Dict[str, int]:
    out = {"metal": 0, "crystal": 0, "deuterium": 0, "total": 0}
    for kind, count in composition.items():
        if kind not in UNIT_SPECS:
            raise KeyError(f"Unknown unit: {kind}")
        spec = UNIT_SPECS[kind]
        out["metal"] += spec.metal * count
        out["crystal"] += spec.crystal * count
        out["deuterium"] += spec.deuterium * count
    out["total"] = out["metal"] + out["crystal"] + out["deuterium"]
    return out


def fleet_cargo_capacity(composition: Mapping[str, int]) -> int:
    """Base cargo capacity of a fleet composition."""
    total = 0
    for kind, count in composition.items():
        if kind not in UNIT_SPECS:
            raise KeyError(f"Unknown unit: {kind}")
        if int(count) != count or count < 0:
            raise ValueError(f"Invalid count for {kind}: {count}")
        total += UNIT_SPECS[kind].cargo_capacity * int(count)
    return int(total)


def calculate_loot(
    defender_resources: Mapping[str, int],
    cargo_capacity: int,
    loot_percentage: float = 0.75,
) -> Dict[str, int]:
    """Apply OGame's metal/crystal/deuterium plunder loading order."""
    resources = _normalise_resources(defender_resources)
    fraction = _normalise_fraction(loot_percentage, "loot_percentage")

    if int(cargo_capacity) != cargo_capacity or cargo_capacity < 0:
        raise ValueError("cargo_capacity must be a non-negative integer")
    remaining_capacity = int(cargo_capacity)

    available = {
        key: int(math.floor(resources[key] * fraction))
        for key in RESOURCE_KEYS
    }
    loot = {key: 0 for key in RESOURCE_KEYS}

    amount = min(available["metal"], remaining_capacity // 3)
    loot["metal"] += amount
    remaining_capacity -= amount

    amount = min(available["crystal"], remaining_capacity // 2)
    loot["crystal"] += amount
    remaining_capacity -= amount

    amount = min(available["deuterium"], remaining_capacity)
    loot["deuterium"] += amount
    remaining_capacity -= amount

    metal_left = available["metal"] - loot["metal"]
    amount = min(metal_left, remaining_capacity // 2)
    loot["metal"] += amount
    remaining_capacity -= amount

    crystal_left = available["crystal"] - loot["crystal"]
    amount = min(crystal_left, remaining_capacity)
    loot["crystal"] += amount

    return loot


# ## 3. Motor de una batalla


def _regenerate_shields(side: BattleSideState) -> None:
    """Restore shields while touching only units with non-default state."""
    for unit_type in side.unit_types:
        if not unit_type.damaged:
            continue

        # A unit that only had shield damage becomes pristine again. Units with
        # hull damage keep only that hull value and recover their full shield.
        for index, (hull, _shield) in list(unit_type.damaged.items()):
            if hull == unit_type.max_hull:
                del unit_type.damaged[index]
            else:
                unit_type.damaged[index] = (hull, unit_type.max_shield)


def _shooters_snapshot(side: BattleSideState) -> list[tuple[str, float, int]]:
    """Snapshot every unit that is entitled to fire in the current round."""
    return [
        (unit_type.kind, unit_type.weapon, unit_type.count)
        for unit_type in side.unit_types
        if unit_type.count > 0
    ]


def _choose_target(
    side: BattleSideState,
    rng: random.Random,
) -> tuple[_UnitTypeState, int, float, float]:
    """Choose one living unit uniformly with a single random index.

    Every living ship/defense has probability ``1 / total_alive``. Mapping the
    chosen global index to a unit type gives the same target distribution as an
    explicit flat list, without rebuilding that list for every shot.
    """
    if side.total_alive <= 0:
        raise IndexError("cannot choose a target from an empty side")

    target_index = rng.randrange(side.total_alive)
    for unit_type in side.unit_types:
        if target_index < unit_type.count:
            hull, shield = unit_type.damaged.get(
                target_index,
                (unit_type.max_hull, unit_type.max_shield),
            )
            return unit_type, target_index, hull, shield
        target_index -= unit_type.count

    raise RuntimeError("compressed battle state is inconsistent")


def _remove_unit(
    side: BattleSideState,
    unit_type: _UnitTypeState,
    index: int,
) -> None:
    """Delete one living unit in O(1) with swap-delete inside its unit type."""
    last_index = unit_type.count - 1
    if index < 0 or index > last_index:
        raise IndexError("unit index out of range")

    if index != last_index:
        last_state = unit_type.damaged.pop(last_index, None)
        if last_state is None:
            unit_type.damaged.pop(index, None)
        else:
            unit_type.damaged[index] = last_state
    else:
        unit_type.damaged.pop(index, None)

    unit_type.count -= 1
    side.total_alive -= 1


def _store_unit_state(
    unit_type: _UnitTypeState,
    index: int,
    hull: float,
    shield: float,
) -> None:
    if hull == unit_type.max_hull and shield == unit_type.max_shield:
        unit_type.damaged.pop(index, None)
    else:
        unit_type.damaged[index] = (hull, shield)


def _apply_hit(
    targets: BattleSideState,
    unit_type: _UnitTypeState,
    index: int,
    hull: float,
    shield: float,
    damage: float,
    rng: random.Random,
    config: CombatConfig,
) -> None:
    # OGame bounce rule: very weak shots do not affect an active shield when
    # their power is below 1% of that target's maximum shield.
    if shield > 0:
        if damage < config.shield_bounce_fraction * unit_type.max_shield:
            return
        absorbed = min(shield, damage)
        shield -= absorbed
        damage -= absorbed

    if damage > 0:
        hull -= damage

    if hull <= 0:
        _remove_unit(targets, unit_type, index)
        return

    # Below 70% hull, every damaging hit can make the unit explode with
    # probability 1 - current_hull/max_hull.
    hull_fraction = hull / unit_type.max_hull
    if hull_fraction < config.explosion_hull_threshold:
        if rng.random() < (1.0 - hull_fraction):
            _remove_unit(targets, unit_type, index)
            return

    _store_unit_state(unit_type, index, hull, shield)


def _fire_phase(
    shooters_snapshot: list[tuple[str, float, int]],
    targets: BattleSideState,
    rng: random.Random,
    config: CombatConfig,
) -> int:
    """Resolve one side's shots without rebuilding target lists.

    Shooters are still processed individually. Rapid Fire is also resolved one
    extra shot at a time, preserving the official random-target and RF rules.
    """
    shots = 0

    for shooter_kind, shooter_weapon, shooter_count in shooters_snapshot:
        for _ in range(shooter_count):
            if targets.total_alive <= 0:
                return shots

            while targets.total_alive > 0:
                target_type, target_index, hull, shield = _choose_target(targets, rng)
                target_kind = target_type.kind

                _apply_hit(
                    targets,
                    target_type,
                    target_index,
                    hull,
                    shield,
                    shooter_weapon,
                    rng,
                    config,
                )
                shots += 1

                # Since OGame v10 the RF calculation stops once there are no
                # targets left. Do not even roll for another RF shot in that case.
                if not config.use_rapid_fire or targets.total_alive <= 0:
                    break

                rf = RAPID_FIRE.get((shooter_kind, target_kind), 1)
                if rf <= 1 or rng.random() >= (1.0 - 1.0 / rf):
                    break

    return shots


def _debris_from_destroyed(
    destroyed_attacker: Mapping[str, int],
    destroyed_defender: Mapping[str, int],
    config: CombatConfig,
) -> Dict[str, int]:
    fleet_fraction = _normalise_fraction(
        config.fleet_debris_fraction,
        "fleet_debris_fraction",
    )
    defense_fraction = _normalise_fraction(
        config.defense_debris_fraction,
        "defense_debris_fraction",
    )

    metal = 0.0
    crystal = 0.0
    for destroyed in (destroyed_attacker, destroyed_defender):
        for kind, count in destroyed.items():
            spec = UNIT_SPECS[kind]
            fraction = defense_fraction if spec.is_defense else fleet_fraction
            metal += spec.metal * count * fraction
            crystal += spec.crystal * count * fraction

    # Deuterium is not part of standard debris fields.
    return {
        "metal": int(math.floor(metal + 1e-9)),
        "crystal": int(math.floor(crystal + 1e-9)),
        "deuterium": 0,
    }


def _harvest_debris(
    debris: Mapping[str, int],
    cargo_capacity: int,
    max_fraction: float,
) -> Dict[str, int]:
    """Harvest a capped fraction of a metal/crystal debris field."""
    fraction = _normalise_fraction(max_fraction, "reaper_harvest_fraction")
    capacity = max(0, int(cargo_capacity))

    eligible = {
        "metal": int(math.floor(debris.get("metal", 0) * fraction)),
        "crystal": int(math.floor(debris.get("crystal", 0) * fraction)),
        "deuterium": 0,
    }
    harvested = {key: 0 for key in RESOURCE_KEYS}

    metal_take = min(eligible["metal"], capacity // 2)
    crystal_take = min(eligible["crystal"], capacity // 2)
    harvested["metal"] = metal_take
    harvested["crystal"] = crystal_take
    capacity -= metal_take + crystal_take

    if capacity > 0:
        metal_left = eligible["metal"] - harvested["metal"]
        extra = min(metal_left, capacity)
        harvested["metal"] += extra
        capacity -= extra

    if capacity > 0:
        crystal_left = eligible["crystal"] - harvested["crystal"]
        extra = min(crystal_left, capacity)
        harvested["crystal"] += extra

    return harvested


def _subtract_resources(
    resources: Mapping[str, int],
    *subtractions: Mapping[str, int],
) -> Dict[str, int]:
    return {
        key: max(
            0,
            int(resources.get(key, 0))
            - sum(int(values.get(key, 0)) for values in subtractions),
        )
        for key in RESOURCE_KEYS
    }


def simulate_battle(
    attacker: Mapping[str, int],
    defender: Mapping[str, int],
    attacker_tech: TechLevels = TechLevels(),
    defender_tech: TechLevels = TechLevels(),
    config: CombatConfig = CombatConfig(),
    seed: Optional[int] = None,
    defender_resources: Optional[Mapping[str, int]] = None,
    loot_percentage: float = 0.75,
) -> BattleResult:
    """Simulate one battle and its immediate economic consequences."""
    loot_fraction = _normalise_fraction(loot_percentage, "loot_percentage")
    planet_resources = _normalise_resources(defender_resources)

    rng = random.Random(seed)
    a_units = build_side(attacker, attacker_tech)
    d_units = build_side(defender, defender_tech)

    initial_a = {k: int(v) for k, v in attacker.items()}
    initial_d = {k: int(v) for k, v in defender.items()}
    shots_a = shots_d = 0
    rounds = 0

    for round_idx in range(1, config.max_rounds + 1):
        if a_units.total_alive <= 0 or d_units.total_alive <= 0:
            break
        rounds = round_idx

        # Shields regenerate fully at the start of every round.
        _regenerate_shields(a_units)
        _regenerate_shields(d_units)

        # Both snapshots are taken before either side fires. A unit alive at the
        # start of the round therefore still fires even if the other side destroys
        # it during its firing phase.
        a_snapshot = _shooters_snapshot(a_units)
        d_snapshot = _shooters_snapshot(d_units)
        shots_a += _fire_phase(a_snapshot, d_units, rng, config)
        shots_d += _fire_phase(d_snapshot, a_units, rng, config)

    a_survivors = survivor_counts(a_units)
    d_survivors_before_rebuild = survivor_counts(d_units)

    if a_survivors and not d_survivors_before_rebuild:
        winner = "attacker"
    elif d_survivors_before_rebuild and not a_survivors:
        winner = "defender"
    else:
        winner = "draw"

    destroyed_a = _subtract_counts(initial_a, a_survivors)
    destroyed_d = _subtract_counts(initial_d, d_survivors_before_rebuild)
    debris_generated = _debris_from_destroyed(destroyed_a, destroyed_d, config)

    reaper_fraction = _normalise_fraction(
        config.reaper_harvest_fraction,
        "reaper_harvest_fraction",
    )
    attacker_reaper_capacity = (
        a_survivors.get("reaper", 0) * UNIT_SPECS["reaper"].cargo_capacity
    )
    defender_reaper_capacity = (
        d_survivors_before_rebuild.get("reaper", 0)
        * UNIT_SPECS["reaper"].cargo_capacity
    )

    attacker_reaper_harvest = _harvest_debris(
        debris_generated,
        attacker_reaper_capacity,
        reaper_fraction,
    )
    defender_reaper_harvest = _harvest_debris(
        debris_generated,
        defender_reaper_capacity,
        reaper_fraction,
    )

    for key in ("metal", "crystal"):
        overflow = (
            attacker_reaper_harvest[key]
            + defender_reaper_harvest[key]
            - debris_generated[key]
        )
        if overflow > 0:
            reduction = min(overflow, defender_reaper_harvest[key])
            defender_reaper_harvest[key] -= reduction
            overflow -= reduction
            if overflow > 0:
                attacker_reaper_harvest[key] -= overflow

    debris_remaining = _subtract_resources(
        debris_generated,
        attacker_reaper_harvest,
        defender_reaper_harvest,
    )

    total_attacker_cargo = fleet_cargo_capacity(a_survivors)
    cargo_used_by_reapers = sum(attacker_reaper_harvest.values())
    cargo_for_loot = max(0, total_attacker_cargo - cargo_used_by_reapers)

    if winner == "attacker":
        loot = calculate_loot(
            planet_resources,
            cargo_for_loot,
            loot_fraction,
        )
    else:
        loot = {key: 0 for key in RESOURCE_KEYS}

    d_survivors = dict(d_survivors_before_rebuild)
    rebuilt: Dict[str, int] = {}
    if config.rebuild_defense:
        for kind, count in destroyed_d.items():
            if UNIT_SPECS[kind].is_defense:
                rebuilt_count = sum(
                    rng.random() < config.defense_rebuild_probability
                    for _ in range(count)
                )
                if rebuilt_count:
                    rebuilt[kind] = rebuilt_count
                    d_survivors[kind] = d_survivors.get(kind, 0) + rebuilt_count

    return BattleResult(
        winner=winner,
        rounds=rounds,
        attacker_initial=initial_a,
        defender_initial=initial_d,
        attacker_survivors=a_survivors,
        defender_survivors_before_rebuild=d_survivors_before_rebuild,
        defender_survivors=d_survivors,
        defender_rebuilt=rebuilt,
        shots_by_attacker=shots_a,
        shots_by_defender=shots_d,
        defender_resources=planet_resources,
        loot_percentage=loot_fraction,
        attacker_cargo_capacity=total_attacker_cargo,
        attacker_cargo_used_by_reapers=cargo_used_by_reapers,
        attacker_cargo_available_for_loot=cargo_for_loot,
        loot=loot,
        debris_generated=debris_generated,
        attacker_reaper_harvest=attacker_reaper_harvest,
        defender_reaper_harvest=defender_reaper_harvest,
        debris_remaining=debris_remaining,
        seed=seed,
    )


# ## 4. Monte Carlo: muchas batallas


def simulate_many(
    attacker: Mapping[str, int],
    defender: Mapping[str, int],
    n: int = 1000,
    attacker_tech: TechLevels = TechLevels(),
    defender_tech: TechLevels = TechLevels(),
    config: CombatConfig = CombatConfig(),
    seed: Optional[int] = None,
    defender_resources: Optional[Mapping[str, int]] = None,
    loot_percentage: float = 0.75,
) -> Dict[str, object]:
    if n <= 0:
        raise ValueError("n must be positive")

    master_rng = random.Random(seed)
    results = [
        simulate_battle(
            attacker,
            defender,
            attacker_tech=attacker_tech,
            defender_tech=defender_tech,
            config=config,
            seed=master_rng.randrange(2**63),
            defender_resources=defender_resources,
            loot_percentage=loot_percentage,
        )
        for _ in range(n)
    ]

    winners = Counter(r.winner for r in results)
    all_a_kinds = set(attacker)
    all_d_kinds = set(defender)

    mean_a_survivors = {
        kind: statistics.fmean(r.attacker_survivors.get(kind, 0) for r in results)
        for kind in sorted(all_a_kinds)
    }
    mean_d_survivors = {
        kind: statistics.fmean(r.defender_survivors.get(kind, 0) for r in results)
        for kind in sorted(all_d_kinds)
    }

    def mean_resources(attribute: str) -> Dict[str, float]:
        return {
            key: statistics.fmean(getattr(r, attribute).get(key, 0) for r in results)
            for key in RESOURCE_KEYS
        }

    return {
        "n": n,
        "win_probability": {
            "attacker": winners["attacker"] / n,
            "defender": winners["defender"] / n,
            "draw": winners["draw"] / n,
        },
        "mean_rounds": statistics.fmean(r.rounds for r in results),
        "mean_attacker_survivors": mean_a_survivors,
        "mean_defender_survivors_after_rebuild": mean_d_survivors,
        "mean_loot": mean_resources("loot"),
        "mean_debris_generated": mean_resources("debris_generated"),
        "mean_attacker_reaper_harvest": mean_resources("attacker_reaper_harvest"),
        "mean_defender_reaper_harvest": mean_resources("defender_reaper_harvest"),
        "mean_debris_remaining": mean_resources("debris_remaining"),
        "mean_attacker_cargo_capacity": statistics.fmean(
            r.attacker_cargo_capacity for r in results
        ),
        "mean_attacker_cargo_available_for_loot": statistics.fmean(
            r.attacker_cargo_available_for_loot for r in results
        ),
        "raw_results": results,
    }
