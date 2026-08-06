#!/usr/bin/env python
# coding: utf-8

# # OGame Utils
# 
# Funciones auxiliares reutilizables para los notebooks del proyecto.
# 
# Incluye utilidades para generar composiciones de flota a partir de porcentajes y para enriquecer los resultados de combate con métricas económicas.

# ## Importar los datos comunes
# 
# El notebook importa `UNIT_SPECS` desde `OgameData`. Mientras el proyecto siga en formato notebook, convertirá automáticamente `OgameData.ipynb` cuando todavía no exista el módulo `.py`.

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping, Any, Dict, Optional
import importlib
import subprocess
import sys
import warnings


def _project_directory() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd()


def _find_project_notebook(filename: str) -> Path:
    local_candidate = _project_directory() / filename
    if local_candidate.exists():
        return local_candidate

    drive_root = Path("/content/drive")
    if drive_root.exists():
        matches = list(drive_root.rglob(filename))
        if matches:
            matches.sort(
                key=lambda path: (
                    "OWiki" not in path.parts,
                    len(path.parts),
                )
            )
            return matches[0]

    raise FileNotFoundError(
        f"No se ha encontrado {filename}. "
        "Ejecuta GeneratePythonModules.ipynb o añade la carpeta del proyecto a sys.path."
    )


def _import_project_module(
    module_name: str,
    notebook_name: str,
):
    project_dir = _project_directory()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

        notebook_path = _find_project_notebook(notebook_name)
        module_dir = notebook_path.parent

        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "python",
                str(notebook_path),
                "--output-dir",
                str(module_dir),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        return importlib.import_module(module_name)


_ogame_data = _import_project_module(
    "OgameData",
    "OgameData.ipynb",
)
_ogame_simulator = _import_project_module(
    "OgameBattleSimulator",
    "OgameBattleSimulator.ipynb",
)

UnitSpec = _ogame_data.UnitSpec
UNIT_SPECS = _ogame_data.UNIT_SPECS

TechLevels = _ogame_simulator.TechLevels
CombatConfig = _ogame_simulator.CombatConfig
BattleResult = _ogame_simulator.BattleResult
simulate_battle = _ogame_simulator.simulate_battle


# ## Cálculo de puntos y generación de flotas

# In[ ]:


import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds



def unit_points(unit_name: str) -> float:
    """
    Return the general OGame points contributed by one completed unit.

    One point corresponds to 1,000 resources invested.
    """
    if unit_name not in UNIT_SPECS:
        raise KeyError(f"Unidad desconocida: {unit_name!r}")

    spec = UNIT_SPECS[unit_name]
    return float(spec.metal + spec.crystal + spec.deuterium) / 1_000.0


def fleet_points(fleet: Mapping[str, int]) -> float:
    """Return the total general points represented by a fleet."""
    total = 0.0

    for unit_name, quantity in fleet.items():
        if not isinstance(quantity, (int, np.integer)) or quantity < 0:
            raise ValueError(
                f"La cantidad de {unit_name!r} debe ser un entero no negativo."
            )

        if unit_name not in UNIT_SPECS:
            raise KeyError(f"Unidad desconocida: {unit_name!r}")

        if UNIT_SPECS[unit_name].category != "ship":
            raise ValueError(
                f"{unit_name!r} no es una nave y no puede incluirse en la flota."
            )

        total += unit_points(unit_name) * int(quantity)

    return float(total)


def _normalise_percentages(
    ship_percentages: Mapping[str, float],
) -> tuple[list[str], np.ndarray]:
    """
    Validate and normalise percentages.

    Values may be written as fractions (0.5) or percentages (50).
    Only their relative proportions matter.
    """
    if not ship_percentages:
        raise ValueError("Debes indicar al menos un tipo de nave.")

    ship_names: list[str] = []
    weights: list[float] = []

    for ship_name, raw_value in ship_percentages.items():
        if ship_name not in UNIT_SPECS:
            raise KeyError(f"Nave desconocida: {ship_name!r}")

        ship_category = UNIT_SPECS[ship_name].category
        if ship_category != "ship":
            raise ValueError(
                f"{ship_name!r} pertenece a la categoría "
                f"{ship_category!r}, no a 'ship'."
            )

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

    shares = np.asarray(weights, dtype=float)
    shares /= shares.sum()

    return ship_names, shares


def _solve_minimum_point_difference(
    point_values: np.ndarray,
    target_points: float,
    *,
    allow_empty: bool,
    time_limit: float | None,
    threads: int | None,
):
    """First MILP stage: minimise the absolute difference in total points."""
    n_ships = len(point_values)
    n_variables = n_ships + 2

    # Variables:
    # [ship counts..., positive point deviation, negative point deviation]
    objective = np.concatenate(
        [np.zeros(n_ships, dtype=float), np.ones(2, dtype=float)]
    )

    total_row = np.zeros(n_variables, dtype=float)
    total_row[:n_ships] = point_values
    total_row[n_ships] = -1.0
    total_row[n_ships + 1] = 1.0

    rows = [total_row]
    lower_bounds = [target_points]
    upper_bounds = [target_points]

    if not allow_empty:
        non_empty_row = np.zeros(n_variables, dtype=float)
        non_empty_row[:n_ships] = 1.0
        rows.append(non_empty_row)
        lower_bounds.append(1.0)
        upper_bounds.append(np.inf)

    options: dict[str, Any] = {
        "presolve": True,
        "mip_rel_gap": 0.0,
    }
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if threads is not None:
        options["threads"] = int(threads)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected: .*threads.*",
            category=RuntimeWarning,
        )
        result = milp(
        c=objective,
        integrality=np.concatenate(
            [np.ones(n_ships, dtype=int), np.zeros(2, dtype=int)]
        ),
        bounds=Bounds(
            np.zeros(n_variables, dtype=float),
            np.full(n_variables, np.inf, dtype=float),
        ),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower_bounds, dtype=float),
            np.asarray(upper_bounds, dtype=float),
        ),
            options=options,
        )

    if not result.success:
        raise RuntimeError(
            "No se pudo resolver la primera etapa de optimización: "
            f"{result.message}"
        )

    return result


def _solve_best_composition(
    point_values: np.ndarray,
    shares: np.ndarray,
    target_points: float,
    minimum_point_difference: float,
    *,
    percentage_basis: str,
    allow_empty: bool,
    time_limit: float | None,
    threads: int | None,
):
    """
    Second MILP stage: preserve the percentages while retaining the optimal
    point difference found in the first stage.
    """
    n_ships = len(point_values)

    # Variables:
    # [ship counts...,
    #  positive total deviation, negative total deviation,
    #  positive composition deviations...,
    #  negative composition deviations...]
    positive_total_index = n_ships
    negative_total_index = n_ships + 1
    positive_composition_start = n_ships + 2
    negative_composition_start = positive_composition_start + n_ships
    n_variables = n_ships + 2 + 2 * n_ships

    objective = np.zeros(n_variables, dtype=float)
    objective[
        positive_composition_start : positive_composition_start + n_ships
    ] = 1.0
    objective[
        negative_composition_start : negative_composition_start + n_ships
    ] = 1.0

    rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    # Total points equality with positive and negative deviation variables.
    total_row = np.zeros(n_variables, dtype=float)
    total_row[:n_ships] = point_values
    total_row[positive_total_index] = -1.0
    total_row[negative_total_index] = 1.0
    rows.append(total_row)
    lower_bounds.append(target_points)
    upper_bounds.append(target_points)

    # Preserve the minimum total point difference found in stage one.
    deviation_row = np.zeros(n_variables, dtype=float)
    deviation_row[positive_total_index] = 1.0
    deviation_row[negative_total_index] = 1.0
    rows.append(deviation_row)
    lower_bounds.append(-np.inf)

    tolerance = max(1e-7, abs(target_points) * 1e-9)
    upper_bounds.append(minimum_point_difference + tolerance)

    if not allow_empty:
        non_empty_row = np.zeros(n_variables, dtype=float)
        non_empty_row[:n_ships] = 1.0
        rows.append(non_empty_row)
        lower_bounds.append(1.0)
        upper_bounds.append(np.inf)

    # Composition constraints.
    #
    # Unit basis:
    #   count_i - requested_share_i * total_ship_count
    #
    # Point basis:
    #   points_i - requested_share_i * actual_total_points
    for index in range(n_ships):
        composition_row = np.zeros(n_variables, dtype=float)

        if percentage_basis == "units":
            composition_row[:n_ships] = -shares[index]
            composition_row[index] += 1.0
        else:
            composition_row[:n_ships] = -shares[index] * point_values
            composition_row[index] += point_values[index]

        composition_row[positive_composition_start + index] = -1.0
        composition_row[negative_composition_start + index] = 1.0

        rows.append(composition_row)
        lower_bounds.append(0.0)
        upper_bounds.append(0.0)

    options: dict[str, Any] = {
        "presolve": True,
        "mip_rel_gap": 0.0,
    }
    if time_limit is not None:
        options["time_limit"] = float(time_limit)
    if threads is not None:
        options["threads"] = int(threads)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unrecognized options detected: .*threads.*",
            category=RuntimeWarning,
        )
        result = milp(
        c=objective,
        integrality=np.concatenate(
            [
                np.ones(n_ships, dtype=int),
                np.zeros(2 + 2 * n_ships, dtype=int),
            ]
        ),
        bounds=Bounds(
            np.zeros(n_variables, dtype=float),
            np.full(n_variables, np.inf, dtype=float),
        ),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower_bounds, dtype=float),
            np.asarray(upper_bounds, dtype=float),
        ),
            options=options,
        )

    if not result.success:
        raise RuntimeError(
            "No se pudo resolver la segunda etapa de optimización: "
            f"{result.message}"
        )

    return result



# -----------------------------------------------------------------------------
# Random compositions for large simulation studies
# -----------------------------------------------------------------------------

DEFAULT_RANDOM_ATTACK_SHIPS = (
    "small_cargo",
    "large_cargo",
    "light_fighter",
    "heavy_fighter",
    "cruiser",
    "battleship",
    "bomber",
    "destroyer",
    "battlecruiser",
    "reaper",
    "pathfinder",
)

DEFAULT_RANDOM_DEFENSE_UNITS = (
    "rocket_launcher",
    "light_laser",
    "heavy_laser",
    "gauss_cannon",
    "ion_cannon",
    "plasma_turret",
)


def _resolve_rng(
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.random.Generator:
    """Return a NumPy Generator while preventing ambiguous double seeding."""
    if rng is not None and seed is not None:
        raise ValueError("Indica 'seed' o 'rng', pero no ambos.")

    if rng is not None:
        return rng

    return np.random.default_rng(seed)


def defense_points(defense: Mapping[str, int]) -> float:
    """Return the total general points represented by defensive units."""
    total = 0.0

    for unit_name, quantity in defense.items():
        if unit_name not in UNIT_SPECS:
            raise KeyError(f"Unidad desconocida: {unit_name!r}")

        if UNIT_SPECS[unit_name].category != "defense":
            raise ValueError(
                f"{unit_name!r} no es una defensa y no puede incluirse aquí."
            )

        if not isinstance(quantity, (int, np.integer)) or quantity < 0:
            raise ValueError(
                f"La cantidad de {unit_name!r} debe ser un entero no negativo."
            )

        total += unit_points(unit_name) * int(quantity)

    return float(total)


def generate_random_defense(
    target_points: float | None = None,
    *,
    min_points: float = 5_000,
    max_points: float = 40_000,
    min_unit_types: int = 3,
    max_unit_types: int = 6,
    include_shield_domes: bool = True,
    shield_dome_probability: float = 0.35,
    concentration: float = 1.3,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_details: bool = False,
):
    """
    Generate a random defensive composition using absolute unit counts.

    The requested point scale is distributed randomly among several defensive
    unit types. Shield domes, when enabled, are added independently with a
    configurable probability and are always capped at one unit each.

    Parameters
    ----------
    target_points:
        Approximate defensive point budget. When omitted, a value is sampled
        log-uniformly between ``min_points`` and ``max_points``.

    min_points, max_points:
        Range used only when ``target_points`` is omitted.

    min_unit_types, max_unit_types:
        Number of regular defensive unit types selected for the composition.

    concentration:
        Dirichlet concentration controlling how balanced the point allocation
        is. Values below 1 create more specialised defenses; values above 1
        create more balanced defenses.

    return_details:
        When True, return target/actual points and selected point shares in
        addition to the absolute composition.
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
            math.exp(
                generator.uniform(
                    math.log(min_points),
                    math.log(max_points),
                )
            )
        )
    else:
        target_points = float(target_points)

    if not math.isfinite(target_points) or target_points <= 0:
        raise ValueError("target_points debe ser finito y mayor que cero.")

    available_units = list(DEFAULT_RANDOM_DEFENSE_UNITS)

    min_unit_types = int(min_unit_types)
    max_unit_types = int(max_unit_types)

    if min_unit_types < 1:
        raise ValueError("min_unit_types debe ser al menos 1.")

    max_unit_types = min(max_unit_types, len(available_units))
    if max_unit_types < min_unit_types:
        raise ValueError(
            "max_unit_types debe ser >= min_unit_types y compatible "
            "con las defensas disponibles."
        )

    if not math.isfinite(float(concentration)) or concentration <= 0:
        raise ValueError("concentration debe ser mayor que cero.")

    if not 0 <= shield_dome_probability <= 1:
        raise ValueError(
            "shield_dome_probability debe estar entre 0 y 1."
        )

    n_types = int(
        generator.integers(
            min_unit_types,
            max_unit_types + 1,
        )
    )

    selected_units = [
        str(unit_name)
        for unit_name in generator.choice(
            available_units,
            size=n_types,
            replace=False,
        )
    ]

    point_shares = generator.dirichlet(
        np.full(n_types, float(concentration))
    )

    defense: dict[str, int] = {}

    for unit_name, share in zip(selected_units, point_shares):
        points_per_unit = unit_points(unit_name)
        allocated_points = target_points * float(share)
        quantity = max(
            1,
            int(round(allocated_points / points_per_unit)),
        )
        defense[unit_name] = quantity

    if include_shield_domes:
        if generator.random() < shield_dome_probability:
            defense["small_shield_dome"] = 1

        if generator.random() < shield_dome_probability:
            defense["large_shield_dome"] = 1

    actual_points = defense_points(defense)

    if not return_details:
        return defense

    return {
        "defense": defense,
        "target_points": float(target_points),
        "actual_points": float(actual_points),
        "point_difference": float(actual_points - target_points),
        "selected_regular_units": selected_units,
        "requested_point_shares": {
            unit_name: float(share * 100.0)
            for unit_name, share in zip(selected_units, point_shares)
        },
    }


def generate_random_fleet_percentages(
    *,
    min_ship_types: int = 3,
    max_ship_types: int = 6,
    allowed_ships: tuple[str, ...] = DEFAULT_RANDOM_ATTACK_SHIPS,
    include_large_cargo: bool = True,
    minimum_percentage: float = 5.0,
    concentration: float = 1.2,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> Dict[str, float]:
    """
    Generate random attacker point percentages summing exactly to 100.

    The output is directly compatible with ``generate_fleet_from_percentages``
    using ``percentage_basis='points'``.

    By default a large-cargo component is always included so the generated
    attacking fleet has realistic loot capacity for profitability studies.
    """
    generator = _resolve_rng(seed=seed, rng=rng)

    allowed = []
    for ship_name in allowed_ships:
        if ship_name not in UNIT_SPECS:
            raise KeyError(f"Nave desconocida: {ship_name!r}")
        if UNIT_SPECS[ship_name].category != "ship":
            raise ValueError(f"{ship_name!r} no pertenece a la categoría ship.")
        if ship_name not in allowed:
            allowed.append(ship_name)

    if not allowed:
        raise ValueError("allowed_ships no puede estar vacío.")

    min_ship_types = int(min_ship_types)
    max_ship_types = int(max_ship_types)

    if min_ship_types < 1:
        raise ValueError("min_ship_types debe ser al menos 1.")

    max_ship_types = min(max_ship_types, len(allowed))
    if max_ship_types < min_ship_types:
        raise ValueError(
            "max_ship_types debe ser >= min_ship_types y compatible "
            "con allowed_ships."
        )

    minimum_percentage = float(minimum_percentage)
    if not math.isfinite(minimum_percentage) or minimum_percentage < 0:
        raise ValueError("minimum_percentage debe ser finito y no negativo.")

    if not math.isfinite(float(concentration)) or concentration <= 0:
        raise ValueError("concentration debe ser mayor que cero.")

    n_types = int(
        generator.integers(
            min_ship_types,
            max_ship_types + 1,
        )
    )

    mandatory: list[str] = []
    if include_large_cargo:
        if "large_cargo" not in allowed:
            raise ValueError(
                "include_large_cargo=True requiere 'large_cargo' "
                "en allowed_ships."
            )
        mandatory.append("large_cargo")

    optional_pool = [
        ship_name
        for ship_name in allowed
        if ship_name not in mandatory
    ]

    n_optional = n_types - len(mandatory)
    if n_optional < 0 or n_optional > len(optional_pool):
        raise ValueError(
            "El número de tipos solicitado es incompatible con las naves "
            "obligatorias y allowed_ships."
        )

    selected = mandatory + [
        str(ship_name)
        for ship_name in generator.choice(
            optional_pool,
            size=n_optional,
            replace=False,
        )
    ]

    n_selected = len(selected)
    reserved_percentage = minimum_percentage * n_selected

    if reserved_percentage >= 100:
        raise ValueError(
            "minimum_percentage es demasiado alto para el número de tipos "
            "de nave seleccionados."
        )

    random_shares = generator.dirichlet(
        np.full(n_selected, float(concentration))
    )
    remaining_percentage = 100.0 - reserved_percentage
    percentages = (
        minimum_percentage
        + remaining_percentage * random_shares
    )

    # Correct the final floating-point residue so the public output sums to 100.
    percentages[-1] += 100.0 - float(percentages.sum())

    return {
        ship_name: float(percentage)
        for ship_name, percentage in zip(selected, percentages)
    }


def generate_fleet_from_percentages(
    ship_percentages: Mapping[str, float],
    points: float,
    *,
    percentage_basis: str = "units",
    return_details: bool = False,
    allow_empty: bool = False,
    solver_time_limit: float | None = None,
    solver_threads: int | None = None,
):
    """
    Generate the integer fleet closest to a target number of OGame points.

    Parameters
    ----------
    ship_percentages:
        Mapping from simulator ship keys to desired percentages.

        Values may use either scale:
        - {"light_fighter": 0.5, "cruiser": 0.5}
        - {"light_fighter": 50, "cruiser": 50}

    points:
        Target general OGame points. One point equals 1,000 resources invested.

    percentage_basis:
        "units" (default):
            Percentages refer to the number of ships.

        "points":
            Percentages refer to the proportion of fleet points allocated to
            each ship type.

    return_details:
        False:
            Return only the fleet dictionary.

        True:
            Return a dictionary containing the fleet, point difference,
            requested percentages, actual percentages and a per-ship breakdown.

    allow_empty:
        When False, at least one ship must be returned.

    solver_time_limit:
        Optional maximum number of seconds for each optimisation stage.

    solver_threads:
        Optional number of HiGHS threads per optimisation. Use 1 when many
        independent Python processes solve fleets concurrently.

    Returns
    -------
    dict
        Fleet dictionary, or a detailed result when return_details=True.
    """
    target_points = float(points)

    if not math.isfinite(target_points) or target_points < 0:
        raise ValueError("'points' debe ser un número finito no negativo.")

    if percentage_basis not in {"units", "points"}:
        raise ValueError(
            "'percentage_basis' debe ser 'units' o 'points'."
        )

    if solver_threads is not None:
        if (
            not isinstance(solver_threads, (int, np.integer))
            or int(solver_threads) <= 0
        ):
            raise ValueError("'solver_threads' debe ser un entero positivo o None.")
        solver_threads = int(solver_threads)

    ship_names, requested_shares = _normalise_percentages(ship_percentages)
    point_values = np.asarray(
        [unit_points(ship_name) for ship_name in ship_names],
        dtype=float,
    )

    stage_one = _solve_minimum_point_difference(
        point_values,
        target_points,
        allow_empty=allow_empty,
        time_limit=solver_time_limit,
        threads=solver_threads,
    )

    minimum_point_difference = float(stage_one.fun)

    stage_two = _solve_best_composition(
        point_values,
        requested_shares,
        target_points,
        minimum_point_difference,
        percentage_basis=percentage_basis,
        allow_empty=allow_empty,
        time_limit=solver_time_limit,
        threads=solver_threads,
    )

    counts = np.rint(stage_two.x[: len(ship_names)]).astype(int)
    counts = np.maximum(counts, 0)

    fleet = {
        ship_name: int(count)
        for ship_name, count in zip(ship_names, counts)
        if count > 0
    }

    actual_points = float(np.dot(point_values, counts))
    point_difference = actual_points - target_points

    if percentage_basis == "units":
        composition_values = counts.astype(float)
    else:
        composition_values = point_values * counts

    composition_total = float(composition_values.sum())

    if composition_total > 0:
        actual_shares = composition_values / composition_total
    else:
        actual_shares = np.zeros_like(requested_shares)

    if not return_details:
        return fleet

    requested_percentages_normalised = {
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
            "points_per_unit": float(points_per_unit),
            "total_points": float(count * points_per_unit),
            "requested_percentage": requested_percentages_normalised[ship_name],
            "actual_percentage": actual_percentages[ship_name],
        }
        for ship_name, count, points_per_unit in zip(
            ship_names,
            counts,
            point_values,
        )
    }

    return {
        "fleet": fleet,
        "target_points": target_points,
        "actual_points": actual_points,
        "point_difference": point_difference,
        "absolute_point_difference": abs(point_difference),
        "percentage_basis": percentage_basis,
        "requested_percentages": requested_percentages_normalised,
        "actual_percentages": actual_percentages,
        "per_ship": per_ship,
    }


# ## Consumo de deuterio de una flota
# 
# Las funciones de esta sección calculan el deuterio consumido al lanzar una flota.
# 
# Por defecto se utiliza:
# 
# - Distancia: 70 sistemas dentro de la misma galaxia.
# - Velocidad de envío: 100 %.
# - Motores: combustión 15, impulso 15 e hiperespacial 15.
# - Multiplicador de consumo: 1,0.
# 
# El cálculo considera la velocidad real de cada nave y la velocidad de la nave más lenta de la flota. El consumo devuelto es el deuterio descontado al lanzar la misión completa.

# In[ ]:


@dataclass(frozen=True)
class DriveLevels:
    """Niveles de los tres motores utilizados para calcular velocidades."""

    combustion: int = 15
    impulse: int = 15
    hyperspace: int = 15

    def __post_init__(self):
        for field_name in (
            "combustion",
            "impulse",
            "hyperspace",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"{field_name} debe ser un entero no negativo."
                )


# Estadísticas base de movimiento.
#
# Algunas naves cambian de motor al alcanzar determinados niveles. Esos
# cambios se aplican en _resolve_ship_flight_spec().
SHIP_FLIGHT_SPECS = {
    "small_cargo": {
        "base_speed": 5_000,
        "fuel_consumption": 10,
        "drive": "combustion",
    },
    "large_cargo": {
        "base_speed": 7_500,
        "fuel_consumption": 50,
        "drive": "combustion",
    },
    "light_fighter": {
        "base_speed": 12_500,
        "fuel_consumption": 20,
        "drive": "combustion",
    },
    "heavy_fighter": {
        "base_speed": 10_000,
        "fuel_consumption": 75,
        "drive": "impulse",
    },
    "cruiser": {
        "base_speed": 15_000,
        "fuel_consumption": 300,
        "drive": "impulse",
    },
    "battleship": {
        "base_speed": 10_000,
        "fuel_consumption": 500,
        "drive": "hyperspace",
    },
    "colony_ship": {
        "base_speed": 2_500,
        "fuel_consumption": 1_000,
        "drive": "impulse",
    },
    "recycler": {
        "base_speed": 2_000,
        "fuel_consumption": 300,
        "drive": "combustion",
    },
    "espionage_probe": {
        "base_speed": 100_000_000,
        "fuel_consumption": 1,
        "drive": "combustion",
    },
    "bomber": {
        "base_speed": 4_000,
        "fuel_consumption": 700,
        "drive": "impulse",
    },
    "destroyer": {
        "base_speed": 5_000,
        "fuel_consumption": 1_000,
        "drive": "hyperspace",
    },
    "deathstar": {
        "base_speed": 100,
        "fuel_consumption": 1,
        "drive": "hyperspace",
    },
    "battlecruiser": {
        "base_speed": 10_000,
        "fuel_consumption": 250,
        "drive": "hyperspace",
    },
    "reaper": {
        "base_speed": 7_000,
        "fuel_consumption": 1_100,
        "drive": "hyperspace",
    },
    "pathfinder": {
        "base_speed": 12_000,
        "fuel_consumption": 300,
        "drive": "hyperspace",
    },
}


_DRIVE_SPEED_BONUSES = {
    "combustion": 0.10,
    "impulse": 0.20,
    "hyperspace": 0.30,
}


def _resolve_ship_flight_spec(
    ship_name: str,
    drive_levels: DriveLevels,
) -> Dict[str, float]:
    """Resuelve motor, velocidad base y consumo tras mejoras de motor."""
    if ship_name not in SHIP_FLIGHT_SPECS:
        if ship_name in UNIT_SPECS:
            raise ValueError(
                f"{ship_name!r} no puede desplazarse como parte de una flota."
            )
        raise KeyError(f"Nave desconocida: {ship_name!r}")

    spec = dict(SHIP_FLIGHT_SPECS[ship_name])

    # Nave pequeña de carga: cambia a impulso a nivel 5.
    if ship_name == "small_cargo" and drive_levels.impulse >= 5:
        spec.update(
            {
                "base_speed": 10_000,
                "fuel_consumption": 20,
                "drive": "impulse",
            }
        )

    # Bombardero: cambia a hiperespacial a nivel 8.
    if ship_name == "bomber" and drive_levels.hyperspace >= 8:
        spec.update(
            {
                "base_speed": 5_000,
                "drive": "hyperspace",
            }
        )

    # Reciclador: hiperespacial tiene prioridad sobre impulso.
    if ship_name == "recycler":
        if drive_levels.hyperspace >= 15:
            spec.update(
                {
                    "base_speed": 6_000,
                    "fuel_consumption": 900,
                    "drive": "hyperspace",
                }
            )
        elif drive_levels.impulse >= 17:
            spec.update(
                {
                    "base_speed": 4_000,
                    "fuel_consumption": 600,
                    "drive": "impulse",
                }
            )

    drive_name = str(spec["drive"])
    drive_level = getattr(drive_levels, drive_name)
    speed_bonus = _DRIVE_SPEED_BONUSES[drive_name]

    spec["actual_speed"] = float(
        spec["base_speed"]
        * (1.0 + speed_bonus * drive_level)
    )

    return spec


def system_distance_to_flight_distance(
    system_distance: int = 70,
) -> int:
    """
    Convierte una separación entre sistemas de la misma galaxia a distancia.

    Distancia = 2.700 + 95 × diferencia de sistemas
    """
    if not isinstance(system_distance, int):
        raise TypeError("system_distance debe ser un entero.")

    if system_distance <= 0:
        raise ValueError(
            "system_distance debe ser mayor que cero. "
            "Esta función representa viajes entre sistemas distintos."
        )

    return 2_700 + 95 * system_distance


def calculate_fleet_deuterium_cost(
    fleet: Mapping[str, int],
    system_distance: int = 70,
    *,
    speed_percentage: float = 100.0,
    drive_levels: DriveLevels = DriveLevels(),
    deuterium_consumption_factor: float = 1.0,
    return_details: bool = False,
):
    """
    Calcula el consumo de deuterio de una flota.

    Parameters
    ----------
    fleet:
        Diccionario con las cantidades de cada nave.

    system_distance:
        Diferencia de sistemas dentro de la misma galaxia.

    speed_percentage:
        Porcentaje de velocidad seleccionado al enviar la flota.

    drive_levels:
        Niveles de combustión, impulso e hiperespacial.

    deuterium_consumption_factor:
        Multiplicador total aplicado al consumo.

        Ejemplos:
        - 1.0: consumo estándar.
        - 0.5: universo con consumo deuterio 50 %.
        - 0.75: reducción equivalente al 25 %.
        - 0.375: combinación de 50 % de universo y 25 % de reducción.

    return_details:
        Cuando es True devuelve también el desglose por nave.

    Notes
    -----
    La fórmula redondea la suma total y añade una unidad de deuterio, igual
    que el cálculo de lanzamiento del juego.
    """
    if not fleet:
        if return_details:
            return {
                "total_deuterium": 0,
                "flight_distance": system_distance_to_flight_distance(
                    system_distance
                ),
                "slowest_ship_speed": 0.0,
                "speed_percentage": float(speed_percentage),
                "deuterium_consumption_factor": float(
                    deuterium_consumption_factor
                ),
                "per_ship": {},
            }
        return 0

    speed_percentage = float(speed_percentage)
    if not math.isfinite(speed_percentage):
        raise ValueError("speed_percentage debe ser finito.")

    if speed_percentage <= 0 or speed_percentage > 100:
        raise ValueError(
            "speed_percentage debe estar entre 0 y 100."
        )

    deuterium_consumption_factor = float(
        deuterium_consumption_factor
    )
    if (
        not math.isfinite(deuterium_consumption_factor)
        or deuterium_consumption_factor < 0
    ):
        raise ValueError(
            "deuterium_consumption_factor debe ser finito "
            "y no negativo."
        )

    flight_distance = system_distance_to_flight_distance(
        system_distance
    )

    resolved_specs = {}
    quantities = {}

    for ship_name, raw_quantity in fleet.items():
        quantity = int(raw_quantity)
        if quantity != raw_quantity or quantity < 0:
            raise ValueError(
                f"La cantidad de {ship_name!r} debe ser "
                "un entero no negativo."
            )

        if quantity == 0:
            continue

        resolved_specs[ship_name] = _resolve_ship_flight_spec(
            ship_name,
            drive_levels,
        )
        quantities[ship_name] = quantity

    if not quantities:
        if return_details:
            return {
                "total_deuterium": 0,
                "flight_distance": flight_distance,
                "slowest_ship_speed": 0.0,
                "speed_percentage": speed_percentage,
                "deuterium_consumption_factor": (
                    deuterium_consumption_factor
                ),
                "per_ship": {},
            }
        return 0

    slowest_ship_speed = min(
        spec["actual_speed"]
        for spec in resolved_specs.values()
    )

    raw_total = 0.0
    per_ship = {}

    for ship_name, quantity in quantities.items():
        spec = resolved_specs[ship_name]
        ship_speed = float(spec["actual_speed"])
        base_fuel = float(spec["fuel_consumption"])

        relative_speed_factor = (
            speed_percentage
            / 100.0
            * math.sqrt(slowest_ship_speed / ship_speed)
        )

        raw_consumption = (
            quantity
            * base_fuel
            * flight_distance
            / 35_000.0
            * (1.0 + relative_speed_factor) ** 2
        )

        raw_total += raw_consumption

        per_ship[ship_name] = {
            "quantity": quantity,
            "drive": spec["drive"],
            "base_speed": float(spec["base_speed"]),
            "actual_speed": ship_speed,
            "base_fuel_consumption": base_fuel,
            "raw_deuterium": raw_consumption,
        }

    adjusted_raw_total = (
        raw_total * deuterium_consumption_factor
    )

    # El ROUND de OGame es redondeo aritmético, no bankers rounding.
    total_deuterium = 1 + math.floor(
        adjusted_raw_total + 0.5
    )

    if not return_details:
        return int(total_deuterium)

    return {
        "total_deuterium": int(total_deuterium),
        "raw_deuterium": float(raw_total),
        "adjusted_raw_deuterium": float(adjusted_raw_total),
        "flight_distance": int(flight_distance),
        "system_distance": int(system_distance),
        "slowest_ship_speed": float(slowest_ship_speed),
        "speed_percentage": float(speed_percentage),
        "drive_levels": {
            "combustion": drive_levels.combustion,
            "impulse": drive_levels.impulse,
            "hyperspace": drive_levels.hyperspace,
        },
        "deuterium_consumption_factor": float(
            deuterium_consumption_factor
        ),
        "per_ship": per_ship,
    }


# ## Simulación con métricas de beneficio
# 
# `simulate_battle_with_profit()` acepta los mismos argumentos y conserva todos los campos de salida de la simulación normal.
# 
# Añade:
# 
# - `minimum_profit`: loot obtenido menos el coste de las naves atacantes destruidas.
# - `maximum_profit`: beneficio mínimo más todos los escombros generados.
# - `minimum_profit_total` y `maximum_profit_total`: suma total de recursos de cada escenario.

# In[ ]:


@dataclass
class BattleResultWithProfit(BattleResult):
    """BattleResult ampliado con métricas económicas del atacante."""

    minimum_profit: Dict[str, int] = field(default_factory=dict)
    maximum_profit: Dict[str, int] = field(default_factory=dict)

    @property
    def minimum_profit_total(self) -> int:
        return int(sum(self.minimum_profit.values()))

    @property
    def maximum_profit_total(self) -> int:
        return int(sum(self.maximum_profit.values()))


def _composition_resource_cost(
    composition: Mapping[str, int],
) -> Dict[str, int]:
    """Devuelve el coste completo en recursos de una composición."""
    resources = {
        "metal": 0,
        "crystal": 0,
        "deuterium": 0,
    }

    for unit_name, quantity in composition.items():
        if unit_name not in UNIT_SPECS:
            raise KeyError(f"Unidad desconocida: {unit_name!r}")

        count = int(quantity)
        if count != quantity or count < 0:
            raise ValueError(
                f"La cantidad de {unit_name!r} debe ser un entero no negativo."
            )

        spec = UNIT_SPECS[unit_name]
        resources["metal"] += int(spec.metal * count)
        resources["crystal"] += int(spec.crystal * count)
        resources["deuterium"] += int(spec.deuterium * count)

    return resources


def _subtract_resources(
    left: Mapping[str, int],
    right: Mapping[str, int],
) -> Dict[str, int]:
    return {
        resource: int(left.get(resource, 0)) - int(right.get(resource, 0))
        for resource in ("metal", "crystal", "deuterium")
    }


def _add_resources(
    left: Mapping[str, int],
    right: Mapping[str, int],
) -> Dict[str, int]:
    return {
        resource: int(left.get(resource, 0)) + int(right.get(resource, 0))
        for resource in ("metal", "crystal", "deuterium")
    }


def simulate_battle_with_profit(
    attacker: Mapping[str, int],
    defender: Mapping[str, int],
    attacker_tech: TechLevels = TechLevels(),
    defender_tech: TechLevels = TechLevels(),
    config: CombatConfig = CombatConfig(),
    seed: Optional[int] = None,
    defender_resources: Optional[Mapping[str, int]] = None,
    loot_percentage: float = 0.75,
) -> BattleResultWithProfit:
    """
    Ejecuta una batalla y añade los beneficios mínimo y máximo del atacante.

    Beneficio mínimo:
        loot - coste completo de las naves atacantes destruidas

    Beneficio máximo:
        beneficio mínimo + todos los escombros generados
    """
    result = simulate_battle(
        attacker=attacker,
        defender=defender,
        attacker_tech=attacker_tech,
        defender_tech=defender_tech,
        config=config,
        seed=seed,
        defender_resources=defender_resources,
        loot_percentage=loot_percentage,
    )

    attacker_loss_cost = _composition_resource_cost(
        result.attacker_destroyed
    )

    minimum_profit = _subtract_resources(
        result.loot,
        attacker_loss_cost,
    )

    maximum_profit = _add_resources(
        minimum_profit,
        result.debris_generated,
    )

    original_result_fields = {
        result_field.name: getattr(result, result_field.name)
        for result_field in fields(BattleResult)
    }

    return BattleResultWithProfit(
        **original_result_fields,
        minimum_profit=minimum_profit,
        maximum_profit=maximum_profit,
    )


# ## Ejemplo
# 
# En este ejemplo se solicita una flota de aproximadamente 10.000 puntos con:
# 
# - 50 % de cazas ligeros.
# - 30 % de cruceros.
# - 20 % de naves de batalla.
# 
# Como `percentage_basis="units"`, los porcentajes se refieren al número de naves.

# In[ ]:


if __name__ == "__main__":
    example = generate_fleet_from_percentages(
        {
            "light_fighter": 50,
            "cruiser": 30,
            "battleship": 20,
        },
        points=10_000,
        percentage_basis="units",
        return_details=True,
    )

    print("Flota:", example["fleet"])
    print("Puntos objetivo:", example["target_points"])
    print("Puntos reales:", example["actual_points"])
    print("Diferencia:", example["point_difference"])
    print("Porcentajes reales:", example["actual_percentages"])


# ## Importación desde otro Colab
# 
# ```python
# from OgameUtils import (
#     DriveLevels,
#     calculate_fleet_deuterium_cost,
#     defense_points,
#     generate_random_defense,
#     generate_random_fleet_percentages,
#     generate_fleet_from_percentages,
#     simulate_battle_with_profit,
# )
# ```

# ## Notas sobre el significado de los porcentajes
# 
# ### Por número de unidades
# 
# ```python
# percentage_basis="units"
# ```
# 
# Un 50 % de cazas ligeros y un 50 % de cruceros intenta producir cantidades similares de ambos tipos, aunque los cruceros cuesten más puntos.
# 
# ### Por puntos invertidos
# 
# ```python
# percentage_basis="points"
# ```
# 
# Un 50 % de cazas ligeros y un 50 % de cruceros intenta dedicar aproximadamente la mitad de los puntos a cada tipo. En este caso habrá muchos más cazas que cruceros.
