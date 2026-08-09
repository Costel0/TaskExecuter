from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..OgameData import UNIT_SPECS
from .interfaces import PopulationInitializer
from .models import AttackGenome, AttackProblem, EvolutionConfig


@dataclass(frozen=True)
class HybridInitializerConfig:
    """Mixture used to construct generation zero.

    The 85% guided/diversified share follows the random-battle analysis while
    the remaining 15% deliberately ignores it so the optimizer can discover
    solutions outside the observed random-battle distribution.
    """

    defense_guided_fraction: float = 0.35
    efficiency_guided_fraction: float = 0.20
    aggressive_fraction: float = 0.15
    safe_fraction: float = 0.15
    exploration_fraction: float = 0.15

    defense_prior_weight: float = 0.70
    guided_concentration: float = 24.0
    diversified_concentration: float = 10.0

    aggressive_multiplier_min: float = 1.30
    aggressive_multiplier_max: float = 2.20
    safe_multiplier_min: float = 4.00
    safe_multiplier_max: float = 5.20
    guided_multiplier_sigma: float = 0.55
    efficiency_multiplier_sigma: float = 0.60

    min_active_ship_types: int = 2
    max_active_ship_types: int = 6
    exploration_specialist_probability: float = 0.50

    def __post_init__(self) -> None:
        fractions = (
            self.defense_guided_fraction,
            self.efficiency_guided_fraction,
            self.aggressive_fraction,
            self.safe_fraction,
            self.exploration_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("Initializer fractions cannot be negative.")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("Initializer fractions must sum to 1.0.")
        if not 0 <= self.defense_prior_weight <= 1:
            raise ValueError("defense_prior_weight must be between 0 and 1.")
        if self.guided_concentration <= 0 or self.diversified_concentration <= 0:
            raise ValueError("Dirichlet concentrations must be positive.")
        if self.min_active_ship_types < 1:
            raise ValueError("min_active_ship_types must be at least 1.")
        if self.max_active_ship_types < self.min_active_ship_types:
            raise ValueError("max_active_ship_types must be >= min_active_ship_types.")
        if not 0 <= self.exploration_specialist_probability <= 1:
            raise ValueError(
                "exploration_specialist_probability must be between 0 and 1."
            )


class HybridAnalysisPopulationInitializer(PopulationInitializer):
    """Generation-zero initializer informed by the random-battle analysis.

    The analysis is used only as a prior. Large Cargo is never forced and no
    ship type is removed from the search space, because the source dataset was
    biased by always including Large Cargo and did not contain optimal attacks.
    """

    def __init__(
        self,
        analysis: Mapping[str, Any],
        *,
        initializer_config: HybridInitializerConfig = HybridInitializerConfig(),
    ) -> None:
        self.analysis = dict(analysis)
        self.initializer_config = initializer_config

        ship_rows = {
            str(row["ship"]): row
            for row in self.analysis.get("ship_rows", [])
            if isinstance(row, Mapping) and "ship" in row
        }
        self._efficiency_prior = {
            ship_name: max(
                0.0,
                float(row.get("mean_point_share_top_efficiency_pct") or 0.0),
            )
            for ship_name, row in ship_rows.items()
        }

        self._defense_rows = {
            str(row["defense"]): row
            for row in self.analysis.get("defense_rows", [])
            if isinstance(row, Mapping) and "defense" in row
        }

        summary = self.analysis.get("success_summary", {})
        self._overall_success_multiplier = float(
            summary.get("mean_multiplier") or 3.35
        )
        self._efficient_multiplier = float(
            summary.get("top_efficiency_mean_multiplier") or 3.10
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        initializer_config: HybridInitializerConfig = HybridInitializerConfig(),
    ) -> "HybridAnalysisPopulationInitializer":
        analysis_path = Path(path).expanduser()
        with analysis_path.open("r", encoding="utf-8") as handle:
            analysis = json.load(handle)
        if not isinstance(analysis, dict):
            raise ValueError("Battle analysis JSON must contain a JSON object.")
        return cls(analysis, initializer_config=initializer_config)

    def create_population(
        self,
        problem: AttackProblem,
        config: EvolutionConfig,
        rng: np.random.Generator,
    ) -> Sequence[AttackGenome]:
        allowed_ships = tuple(config.allowed_ships)
        if not allowed_ships:
            raise ValueError("EvolutionConfig.allowed_ships cannot be empty.")

        seed_prior, defense_multiplier = self._problem_prior(
            problem,
            allowed_ships,
        )
        efficiency_prior = self._normalized_prior(
            self._efficiency_prior,
            allowed_ships,
        )

        fractions = (
            self.initializer_config.defense_guided_fraction,
            self.initializer_config.efficiency_guided_fraction,
            self.initializer_config.aggressive_fraction,
            self.initializer_config.safe_fraction,
            self.initializer_config.exploration_fraction,
        )
        counts = self._allocate_counts(config.population_size, fractions)

        population: list[AttackGenome] = []

        for _ in range(counts[0]):
            population.append(
                AttackGenome(
                    ship_weights=self._sample_guided_composition(
                        seed_prior,
                        allowed_ships,
                        rng,
                        concentration=self.initializer_config.guided_concentration,
                    ),
                    points_multiplier=self._sample_centered_multiplier(
                        defense_multiplier,
                        self.initializer_config.guided_multiplier_sigma,
                        config,
                        rng,
                    ),
                )
            )

        for _ in range(counts[1]):
            population.append(
                AttackGenome(
                    ship_weights=self._sample_guided_composition(
                        efficiency_prior,
                        allowed_ships,
                        rng,
                        concentration=self.initializer_config.guided_concentration,
                    ),
                    points_multiplier=self._sample_centered_multiplier(
                        self._efficient_multiplier,
                        self.initializer_config.efficiency_multiplier_sigma,
                        config,
                        rng,
                    ),
                )
            )

        for _ in range(counts[2]):
            population.append(
                AttackGenome(
                    ship_weights=self._sample_guided_composition(
                        seed_prior,
                        allowed_ships,
                        rng,
                        concentration=self.initializer_config.diversified_concentration,
                    ),
                    points_multiplier=self._sample_uniform_multiplier(
                        self.initializer_config.aggressive_multiplier_min,
                        self.initializer_config.aggressive_multiplier_max,
                        config,
                        rng,
                    ),
                )
            )

        for _ in range(counts[3]):
            population.append(
                AttackGenome(
                    ship_weights=self._sample_guided_composition(
                        seed_prior,
                        allowed_ships,
                        rng,
                        concentration=self.initializer_config.diversified_concentration,
                    ),
                    points_multiplier=self._sample_uniform_multiplier(
                        self.initializer_config.safe_multiplier_min,
                        self.initializer_config.safe_multiplier_max,
                        config,
                        rng,
                    ),
                )
            )

        for _ in range(counts[4]):
            population.append(
                AttackGenome(
                    ship_weights=self._sample_exploration_composition(
                        allowed_ships,
                        rng,
                    ),
                    points_multiplier=float(
                        rng.uniform(
                            config.min_points_multiplier,
                            config.max_points_multiplier,
                        )
                    ),
                )
            )

        rng.shuffle(population)
        return population

    def _problem_prior(
        self,
        problem: AttackProblem,
        allowed_ships: Sequence[str],
    ) -> tuple[dict[str, float], float]:
        defense_point_values: dict[str, float] = {}
        for defense_name, raw_count in problem.defender.items():
            if defense_name not in self._defense_rows:
                continue
            count = int(raw_count)
            if count <= 0:
                continue
            spec = UNIT_SPECS.get(defense_name)
            if spec is None or not spec.is_defense:
                continue
            defense_point_values[defense_name] = (
                float(spec.cost) / 1_000.0 * count
            )

        total_defense_points = sum(defense_point_values.values())
        defense_prior = {ship_name: 0.0 for ship_name in allowed_ships}
        multiplier_weighted_sum = 0.0

        if total_defense_points > 0:
            for defense_name, point_value in defense_point_values.items():
                defense_share = point_value / total_defense_points
                row = self._defense_rows[defense_name]
                ship_shares = row.get("mean_success_ship_point_shares_pct", {})
                if isinstance(ship_shares, Mapping):
                    for ship_name in allowed_ships:
                        defense_prior[ship_name] += defense_share * max(
                            0.0,
                            float(ship_shares.get(ship_name, 0.0)),
                        )
                multiplier_weighted_sum += defense_share * float(
                    row.get("mean_success_multiplier")
                    or self._overall_success_multiplier
                )
        else:
            multiplier_weighted_sum = self._overall_success_multiplier

        efficiency_prior = self._normalized_prior(
            self._efficiency_prior,
            allowed_ships,
        )
        defense_prior = self._normalized_prior(defense_prior, allowed_ships)

        if not any(value > 0 for value in defense_prior.values()):
            defense_prior = dict(efficiency_prior)

        defense_weight = self.initializer_config.defense_prior_weight
        combined_prior = {
            ship_name: (
                defense_weight * defense_prior.get(ship_name, 0.0)
                + (1.0 - defense_weight)
                * efficiency_prior.get(ship_name, 0.0)
            )
            for ship_name in allowed_ships
        }
        combined_prior = self._normalized_prior(combined_prior, allowed_ships)

        return combined_prior, float(multiplier_weighted_sum)

    @staticmethod
    def _normalized_prior(
        prior: Mapping[str, float],
        allowed_ships: Sequence[str],
    ) -> dict[str, float]:
        values = {
            ship_name: max(0.0, float(prior.get(ship_name, 0.0)))
            for ship_name in allowed_ships
        }
        total = sum(values.values())
        if total <= 0:
            uniform = 1.0 / len(allowed_ships)
            return {ship_name: uniform for ship_name in allowed_ships}
        return {
            ship_name: value / total
            for ship_name, value in values.items()
        }

    @staticmethod
    def _allocate_counts(
        total: int,
        fractions: Sequence[float],
    ) -> tuple[int, ...]:
        raw = [total * float(fraction) for fraction in fractions]
        counts = [int(np.floor(value)) for value in raw]
        remainder = total - sum(counts)
        order = sorted(
            range(len(raw)),
            key=lambda index: raw[index] - counts[index],
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
        return tuple(counts)

    def _sample_guided_composition(
        self,
        prior: Mapping[str, float],
        allowed_ships: Sequence[str],
        rng: np.random.Generator,
        *,
        concentration: float,
    ) -> dict[str, float]:
        max_types = min(
            self.initializer_config.max_active_ship_types,
            len(allowed_ships),
        )
        min_types = min(
            self.initializer_config.min_active_ship_types,
            max_types,
        )
        n_types = int(rng.integers(min_types, max_types + 1))

        probabilities = np.asarray(
            [max(0.0, float(prior.get(ship_name, 0.0))) for ship_name in allowed_ships],
            dtype=float,
        )
        if probabilities.sum() <= 0:
            probabilities = np.ones(len(allowed_ships), dtype=float)
        probabilities /= probabilities.sum()

        selected_indices = rng.choice(
            len(allowed_ships),
            size=n_types,
            replace=False,
            p=probabilities,
        )
        selected_ships = [allowed_ships[int(index)] for index in selected_indices]
        selected_prior = np.asarray(
            [max(0.0, float(prior.get(ship_name, 0.0))) for ship_name in selected_ships],
            dtype=float,
        )
        if selected_prior.sum() <= 0:
            selected_prior = np.ones(n_types, dtype=float)
        selected_prior /= selected_prior.sum()

        alpha = 0.35 + selected_prior * float(concentration)
        sampled = rng.dirichlet(alpha)
        return {
            ship_name: float(weight)
            for ship_name, weight in zip(selected_ships, sampled)
        }

    def _sample_exploration_composition(
        self,
        allowed_ships: Sequence[str],
        rng: np.random.Generator,
    ) -> dict[str, float]:
        max_types = min(
            self.initializer_config.max_active_ship_types,
            len(allowed_ships),
        )
        n_types = int(rng.integers(1, max_types + 1))
        selected = [
            allowed_ships[int(index)]
            for index in rng.choice(
                len(allowed_ships),
                size=n_types,
                replace=False,
            )
        ]

        if n_types == 1:
            return {selected[0]: 1.0}

        if rng.random() < self.initializer_config.exploration_specialist_probability:
            focal_index = int(rng.integers(0, n_types))
            focal_share = float(rng.uniform(0.55, 0.85))
            remainder = 1.0 - focal_share
            other_indices = [index for index in range(n_types) if index != focal_index]
            other_shares = rng.dirichlet(np.ones(len(other_indices))) * remainder
            weights = np.zeros(n_types, dtype=float)
            weights[focal_index] = focal_share
            for index, share in zip(other_indices, other_shares):
                weights[index] = share
        else:
            concentration = float(rng.uniform(0.4, 1.5))
            weights = rng.dirichlet(np.full(n_types, concentration))

        return {
            ship_name: float(weight)
            for ship_name, weight in zip(selected, weights)
        }

    @staticmethod
    def _sample_centered_multiplier(
        center: float,
        sigma: float,
        config: EvolutionConfig,
        rng: np.random.Generator,
    ) -> float:
        value = float(rng.normal(float(center), float(sigma)))
        return float(
            np.clip(
                value,
                config.min_points_multiplier,
                config.max_points_multiplier,
            )
        )

    @staticmethod
    def _sample_uniform_multiplier(
        lower: float,
        upper: float,
        config: EvolutionConfig,
        rng: np.random.Generator,
    ) -> float:
        lower = max(float(lower), config.min_points_multiplier)
        upper = min(float(upper), config.max_points_multiplier)
        if upper < lower:
            return float(np.clip(lower, config.min_points_multiplier, config.max_points_multiplier))
        if upper == lower:
            return float(lower)
        return float(rng.uniform(lower, upper))
