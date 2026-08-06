#!/usr/bin/env python
# coding: utf-8

# # OGame Battle Simulator — reference implementation
# 
# Motor de combate modular en Python. Los datos estáticos de unidades y fuego rápido se importan desde `OgameData`, de modo que el simulador contiene únicamente reglas y estado de combate.
# 
# El notebook puede convertirse a `OgameBattleSimulator.py` e importarse desde otros Colabs.

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
from collections import Counter
import math
import random
import statistics

__all__ = [
    "UnitSpec", "TechLevels", "CombatConfig", "BattleResult",
    "UNIT_SPECS", "RAPID_FIRE", "simulate_battle", "simulate_many",
    "build_side", "fleet_cost", "fleet_cargo_capacity", "survivor_counts",
    "calculate_loot", "validate_reference_data"
]


# ## 1. Importar los datos comunes
# 
# `UnitSpec`, `UNIT_SPECS`, `RAPID_FIRE` y las validaciones viven en `OgameData`. El bloque siguiente mantiene compatibilidad mientras los módulos todavía se desarrollan como notebooks: si no existe `OgameData.py`, convierte automáticamente `OgameData.ipynb`.

# In[ ]:


from pathlib import Path
import importlib
import subprocess
import sys


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

# In[ ]:


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


@dataclass
class UnitState:
    kind: str
    hull: float
    shield: float
    max_hull: float
    max_shield: float
    weapon: float
    is_defense: bool
    alive: bool = True


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


def build_side(composition: Mapping[str, int], tech: TechLevels) -> List[UnitState]:
    units: List[UnitState] = []
    for kind, count in composition.items():
        if kind not in UNIT_SPECS:
            raise KeyError(f"Unknown unit: {kind}")
        if int(count) != count or count < 0:
            raise ValueError(f"Invalid count for {kind}: {count}")
        spec = UNIT_SPECS[kind]
        max_hull = spec.hull * tech.hull_multiplier()
        max_shield = spec.shield * tech.shield_multiplier()
        weapon = spec.weapon * tech.weapon_multiplier()
        units.extend(
            UnitState(
                kind,
                max_hull,
                max_shield,
                max_hull,
                max_shield,
                weapon,
                spec.is_defense,
            )
            for _ in range(int(count))
        )
    return units


def survivor_counts(units: Iterable[UnitState]) -> Dict[str, int]:
    return dict(Counter(u.kind for u in units if u.alive))


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
    """Base cargo capacity of a fleet composition.

    Fuel already spent on the mission is not modelled here. Hyperspace,
    class and lifeform cargo bonuses can later be represented by modifying
    the unit data or adding an external multiplier.
    """
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
    """Apply OGame's metal/crystal/deuterium plunder loading order.

    The loot percentage limits how much of each planetary resource can be
    taken. Cargo capacity can reduce the amount below that limit.
    """
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

    # Standard OGame plunder loading order:
    # 1) up to one third of total cargo with metal;
    # 2) up to half the remaining cargo with crystal;
    # 3) fill the remaining cargo with deuterium;
    # 4) half the remaining cargo with additional metal;
    # 5) the rest with additional crystal.
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

# In[ ]:


def _living(units: List[UnitState]) -> List[UnitState]:
    return [u for u in units if u.alive]


def _apply_hit(target: UnitState, damage: float, rng: random.Random, config: CombatConfig) -> None:
    if not target.alive:
        return

    # OGame bounce rule: very weak shots do not affect a shield when their power
    # is below 1% of the target's maximum shield.
    if target.shield > 0:
        if damage < config.shield_bounce_fraction * target.max_shield:
            return
        absorbed = min(target.shield, damage)
        target.shield -= absorbed
        damage -= absorbed

    if damage > 0:
        target.hull -= damage

    if target.hull <= 0:
        target.alive = False
        return

    # When hull falls below 70%, the unit may explode. The probability is
    # 1 - current_hull/max_hull.
    hull_fraction = target.hull / target.max_hull
    if hull_fraction < config.explosion_hull_threshold:
        if rng.random() < (1.0 - hull_fraction):
            target.alive = False


def _fire_phase(
    shooters_snapshot: List[UnitState],
    targets: List[UnitState],
    rng: random.Random,
    config: CombatConfig,
) -> int:
    shots = 0
    for shooter in shooters_snapshot:
        # Simultaneous rounds: a unit alive at the beginning of the round still fires,
        # even if it was destroyed during the opponent's phase.
        while _living(targets):
            target = rng.choice(_living(targets))
            _apply_hit(target, shooter.weapon, rng, config)
            shots += 1

            if not config.use_rapid_fire:
                break
            rf = RAPID_FIRE.get((shooter.kind, target.kind), 1)
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
    """Harvest a capped fraction of a metal/crystal debris field.

    When cargo is insufficient, metal and crystal are loaded as evenly as
    possible, matching the usual debris harvesting behaviour.
    """
    fraction = _normalise_fraction(max_fraction, "reaper_harvest_fraction")
    capacity = max(0, int(cargo_capacity))

    eligible = {
        "metal": int(math.floor(debris.get("metal", 0) * fraction)),
        "crystal": int(math.floor(debris.get("crystal", 0) * fraction)),
        "deuterium": 0,
    }
    harvested = {key: 0 for key in RESOURCE_KEYS}

    # Try to split capacity evenly between metal and crystal.
    metal_take = min(eligible["metal"], capacity // 2)
    crystal_take = min(eligible["crystal"], capacity // 2)
    harvested["metal"] = metal_take
    harvested["crystal"] = crystal_take
    capacity -= metal_take + crystal_take

    # If one resource is scarce, use the remaining capacity for the other.
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
    """Simulate one battle and its immediate economic consequences.

    Parameters
    ----------
    defender_resources:
        Planetary metal, crystal and deuterium available before the attack.
        Missing keys default to zero.
    loot_percentage:
        Maximum fraction of each planetary resource that can be pillaged.
        Both ``0.75`` and ``75`` are accepted. The default is 75%.

    Notes
    -----
    Loot is only obtained when the attacker wins. It is constrained by the
    surviving attacker's total cargo capacity. Debris collected automatically
    by surviving attacking Reapers occupies cargo space before loot is loaded.
    """
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
        if not _living(a_units) or not _living(d_units):
            break
        rounds = round_idx

        # Shields regenerate fully at the start of every round.
        for unit in _living(a_units) + _living(d_units):
            unit.shield = unit.max_shield

        a_snapshot = list(_living(a_units))
        d_snapshot = list(_living(d_units))
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

    # Each side's surviving Reapers may collect up to the configured fraction
    # of the initial debris. At the standard 25%, both sides combined can never
    # remove more than half of the field.
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

    # Guard against non-standard configurations in which the two independent
    # caps could exceed the available field.
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

# In[ ]:


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


# ## 5. Ejemplo con recursos, loot y Reapers
# 
# ```python
# result = simulate_battle(
#     attacker={"reaper": 5, "large_cargo": 20},
#     defender={"rocket_launcher": 100, "small_cargo": 10},
#     defender_resources={
#         "metal": 1_000_000,
#         "crystal": 500_000,
#         "deuterium": 250_000,
#     },
#     loot_percentage=0.75,  # también se acepta 75
#     seed=42,
# )
# 
# print(result.loot)
# print(result.debris_generated)
# print(result.attacker_reaper_harvest)
# print(result.debris_remaining)
# ```
# 
# Los porcentajes de escombros y el límite de reciclaje de los Reapers se pueden adaptar a cada universo mediante `CombatConfig`.
# 
