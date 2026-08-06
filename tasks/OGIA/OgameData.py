#!/usr/bin/env python
# coding: utf-8

# # OGame Data
# 
# Fuente única de datos estáticos compartidos por el proyecto:
# 
# - `UnitSpec`
# - `UNIT_SPECS`
# - `RAPID_FIRE`
# - `RESOURCE_KEYS`
# - validaciones de referencia
# 
# Este módulo no contiene lógica de simulación ni optimización. Tanto `OgameBattleSimulator` como `OgameUtils` dependen de él, pero no dependen entre sí.

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import math

__all__ = [
    "UnitSpec",
    "UNIT_SPECS",
    "RAPID_FIRE",
    "RESOURCE_KEYS",
    "validate_reference_data",
]

RESOURCE_KEYS: Tuple[str, str, str] = ("metal", "crystal", "deuterium")

@dataclass(frozen=True)
class UnitSpec:
    name: str
    metal: int
    crystal: int
    deuterium: int
    weapon: float
    shield: float
    hull: float
    is_defense: bool = False
    cargo_capacity: int = 0

    @property
    def cost(self) -> int:
        return self.metal + self.crystal + self.deuterium

    @property
    def category(self) -> str:
        return "defense" if self.is_defense else "ship"


# Base combat values audited against Gameforge/OGame reference information.
# `hull` is the value used internally by the combat engine:
# 10% of the official Structural Integrity (equivalent to metal + crystal cost / 10).
UNIT_SPECS: Dict[str, UnitSpec] = {
    # Ships and civil units
    "small_cargo": UnitSpec("Small Cargo", 2_000, 2_000, 0, 5, 10, 400, cargo_capacity=5_000),
    "large_cargo": UnitSpec("Large Cargo", 6_000, 6_000, 0, 5, 25, 1_200, cargo_capacity=25_000),
    "light_fighter": UnitSpec("Light Fighter", 3_000, 1_000, 0, 50, 10, 400, cargo_capacity=50),
    "heavy_fighter": UnitSpec("Heavy Fighter", 6_000, 4_000, 0, 150, 25, 1_000, cargo_capacity=100),
    "cruiser": UnitSpec("Cruiser", 20_000, 7_000, 2_000, 400, 50, 2_700, cargo_capacity=800),
    "battleship": UnitSpec("Battleship", 45_000, 15_000, 0, 1_000, 200, 6_000, cargo_capacity=1_500),
    "colony_ship": UnitSpec("Colony Ship", 10_000, 20_000, 10_000, 50, 100, 3_000, cargo_capacity=7_500),
    "recycler": UnitSpec("Recycler", 10_000, 6_000, 2_000, 1, 10, 1_600, cargo_capacity=20_000),
    "espionage_probe": UnitSpec("Espionage Probe", 0, 1_000, 0, 0.01, 0.01, 100, cargo_capacity=5),
    "bomber": UnitSpec("Bomber", 50_000, 25_000, 15_000, 1_000, 500, 7_500, cargo_capacity=500),
    "solar_satellite": UnitSpec("Solar Satellite", 0, 2_000, 500, 1, 1, 200, cargo_capacity=0),
    "destroyer": UnitSpec("Destroyer", 60_000, 50_000, 15_000, 2_000, 500, 11_000, cargo_capacity=2_000),
    "deathstar": UnitSpec(
        "Deathstar",
        5_000_000,
        4_000_000,
        1_000_000,
        200_000,
        50_000,
        900_000,
        cargo_capacity=1_000_000,
    ),
    "battlecruiser": UnitSpec("Battlecruiser", 30_000, 40_000, 15_000, 700, 400, 7_000, cargo_capacity=750),
    "reaper": UnitSpec("Reaper", 85_000, 55_000, 20_000, 2_800, 700, 14_000, cargo_capacity=10_000),
    "pathfinder": UnitSpec("Pathfinder", 8_000, 15_000, 8_000, 200, 100, 2_300, cargo_capacity=10_000),
    "crawler": UnitSpec("Crawler", 2_000, 2_000, 1_000, 1, 1, 400, cargo_capacity=0),

    # Defenses
    "rocket_launcher": UnitSpec("Rocket Launcher", 2_000, 0, 0, 80, 20, 200, True),
    "light_laser": UnitSpec("Light Laser", 1_500, 500, 0, 100, 25, 200, True),
    "heavy_laser": UnitSpec("Heavy Laser", 6_000, 2_000, 0, 250, 100, 800, True),
    "gauss_cannon": UnitSpec("Gauss Cannon", 20_000, 15_000, 2_000, 1_100, 200, 3_500, True),
    "ion_cannon": UnitSpec("Ion Cannon", 5_000, 3_000, 0, 150, 500, 800, True),
    "plasma_turret": UnitSpec("Plasma Turret", 50_000, 50_000, 30_000, 3_000, 300, 10_000, True),
    "small_shield_dome": UnitSpec("Small Shield Dome", 10_000, 10_000, 0, 1, 2_000, 2_000, True),
    "large_shield_dome": UnitSpec("Large Shield Dome", 50_000, 50_000, 0, 1, 10_000, 10_000, True),
}


# RF=N means the shooter receives another shot with probability 1 - 1/N.
# The table below contains the base rapid-fire relations currently used by OGame.
RAPID_FIRE: Dict[Tuple[str, str], int] = {
    # Classical special interactions
    ("heavy_fighter", "small_cargo"): 3,
    ("cruiser", "light_fighter"): 6,
    ("cruiser", "rocket_launcher"): 10,

    ("battleship", "pathfinder"): 5,

    ("bomber", "rocket_launcher"): 20,
    ("bomber", "light_laser"): 20,
    ("bomber", "heavy_laser"): 10,
    ("bomber", "gauss_cannon"): 5,
    ("bomber", "ion_cannon"): 10,
    ("bomber", "plasma_turret"): 5,

    ("destroyer", "light_laser"): 10,
    ("destroyer", "battlecruiser"): 2,

    ("battlecruiser", "small_cargo"): 3,
    ("battlecruiser", "large_cargo"): 3,
    ("battlecruiser", "heavy_fighter"): 4,
    ("battlecruiser", "cruiser"): 4,
    ("battlecruiser", "battleship"): 7,

    # Class ships
    ("reaper", "battleship"): 7,
    ("reaper", "bomber"): 4,
    ("reaper", "destroyer"): 3,

    ("pathfinder", "light_fighter"): 3,
    ("pathfinder", "heavy_fighter"): 2,
    ("pathfinder", "cruiser"): 3,

    # Defensive rapid fire
    ("ion_cannon", "reaper"): 2,

    # Deathstar. Version 10 capped Probe, Crawler and Solar Satellite at RF 250.
    ("deathstar", "small_cargo"): 250,
    ("deathstar", "large_cargo"): 250,
    ("deathstar", "light_fighter"): 200,
    ("deathstar", "heavy_fighter"): 100,
    ("deathstar", "cruiser"): 33,
    ("deathstar", "battleship"): 30,
    ("deathstar", "colony_ship"): 250,
    ("deathstar", "recycler"): 250,
    ("deathstar", "espionage_probe"): 250,
    ("deathstar", "bomber"): 25,
    ("deathstar", "solar_satellite"): 250,
    ("deathstar", "destroyer"): 5,
    ("deathstar", "battlecruiser"): 15,
    ("deathstar", "reaper"): 10,
    ("deathstar", "pathfinder"): 30,
    ("deathstar", "crawler"): 250,
    ("deathstar", "rocket_launcher"): 200,
    ("deathstar", "light_laser"): 200,
    ("deathstar", "heavy_laser"): 100,
    ("deathstar", "gauss_cannon"): 50,
    ("deathstar", "ion_cannon"): 100,
}


# All movable ships other than the Espionage Probe and Deathstar have RF 5
# against the three fragile economic/utility units.
_ECONOMIC_RF_SHOOTERS: Tuple[str, ...] = (
    "small_cargo",
    "large_cargo",
    "light_fighter",
    "heavy_fighter",
    "cruiser",
    "battleship",
    "colony_ship",
    "recycler",
    "bomber",
    "destroyer",
    "battlecruiser",
    "reaper",
    "pathfinder",
)

for _shooter in _ECONOMIC_RF_SHOOTERS:
    for _target in ("espionage_probe", "solar_satellite", "crawler"):
        RAPID_FIRE[(_shooter, _target)] = 5


def validate_reference_data() -> None:
    """Fail early if an audited base statistic or RF relation regresses."""

    # Unit statistics corrected during the official-data audit.
    assert UNIT_SPECS["espionage_probe"].weapon == 0.01
    assert UNIT_SPECS["espionage_probe"].shield == 0.01
    assert UNIT_SPECS["crawler"] == UnitSpec(
        "Crawler", 2_000, 2_000, 1_000, 1, 1, 400, cargo_capacity=0
    )
    assert (
        UNIT_SPECS["ion_cannon"].metal,
        UNIT_SPECS["ion_cannon"].crystal,
    ) == (5_000, 3_000)

    # The internal hull scale must remain 10% of metal + crystal cost.
    for _unit_key, _spec in UNIT_SPECS.items():
        expected_hull = (_spec.metal + _spec.crystal) / 10
        assert math.isclose(
            _spec.hull,
            expected_hull,
            rel_tol=0.0,
            abs_tol=1e-9,
        ), f"Invalid hull for {_unit_key}: {_spec.hull} != {expected_hull}"

    # Corrected/missing RF relations.
    expected_rf = {
        ("heavy_fighter", "small_cargo"): 3,
        ("battleship", "pathfinder"): 5,
        ("bomber", "plasma_turret"): 5,
        ("destroyer", "battlecruiser"): 2,
        ("reaper", "battleship"): 7,
        ("reaper", "bomber"): 4,
        ("reaper", "destroyer"): 3,
        ("pathfinder", "cruiser"): 3,
        ("ion_cannon", "reaper"): 2,
        ("deathstar", "espionage_probe"): 250,
        ("deathstar", "solar_satellite"): 250,
        ("deathstar", "crawler"): 250,
        ("deathstar", "reaper"): 10,
        ("deathstar", "pathfinder"): 30,
    }
    for _relation, _rf in expected_rf.items():
        assert RAPID_FIRE.get(_relation) == _rf, (
            f"Invalid RF for {_relation}: "
            f"{RAPID_FIRE.get(_relation)} != {_rf}"
        )

    # These relations existed in the previous draft but are not current Reaper RF.
    invalid_reaper_relations = (
        ("reaper", "small_cargo"),
        ("reaper", "large_cargo"),
        ("reaper", "heavy_fighter"),
        ("reaper", "cruiser"),
        ("reaper", "battlecruiser"),
    )
    for _relation in invalid_reaper_relations:
        assert _relation not in RAPID_FIRE, f"Obsolete RF relation present: {_relation}"

    # Complete RF5 coverage for utility/economic targets.
    for _shooter in _ECONOMIC_RF_SHOOTERS:
        for _target in ("espionage_probe", "solar_satellite", "crawler"):
            assert RAPID_FIRE.get((_shooter, _target)) == 5


validate_reference_data()


# ## Importación
# 
# Después de convertir el notebook a Python:
# 
# ```python
# from OgameData import UNIT_SPECS, RAPID_FIRE, UnitSpec
# ```
