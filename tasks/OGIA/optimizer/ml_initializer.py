from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .interfaces import PopulationInitializer
from .models import AttackGenome, AttackProblem, EvolutionConfig
from .operators import canonicalize_genome

GenomePredictor = Callable[[Mapping[str, int]], AttackGenome]


@dataclass(frozen=True)
class MLSeededInitializerConfig:
    """Generation-zero refinement around a learned attack prediction.

    Every generated individual is either the exact ML prediction, a perturbation
    of it, or a blend in which the ML prediction remains the dominant component.
    There are deliberately no model-free restarts in this initializer.
    """

    local_fraction: float = 0.70
    broad_fraction: float = 0.20
    analysis_blend_fraction: float = 0.10
    exact_seed_count: int = 1

    local_log_weight_sigma: float = 0.20
    broad_log_weight_sigma: float = 0.65
    local_multiplier_sigma: float = 0.04
    broad_multiplier_sigma: float = 0.14

    local_new_ship_probability: float = 0.05
    broad_new_ship_probability: float = 0.20
    broad_drop_ship_probability: float = 0.10
    new_ship_weight_min: float = 0.005
    new_ship_weight_max: float = 0.05
    min_ship_weight: float = 0.004
    max_active_ship_types: int = 6

    analysis_model_weight_min: float = 0.70
    analysis_model_weight_max: float = 0.90

    def __post_init__(self) -> None:
        fractions = (
            self.local_fraction,
            self.broad_fraction,
            self.analysis_blend_fraction,
        )
        if any(value < 0 for value in fractions):
            raise ValueError("ML initializer fractions cannot be negative.")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("ML initializer fractions must sum to 1.0.")
        if self.exact_seed_count < 1:
            raise ValueError("exact_seed_count must be at least 1.")
        if self.local_log_weight_sigma < 0 or self.broad_log_weight_sigma < 0:
            raise ValueError("Weight perturbation sigmas cannot be negative.")
        if self.local_multiplier_sigma < 0 or self.broad_multiplier_sigma < 0:
            raise ValueError("Multiplier perturbation sigmas cannot be negative.")
        for field_name in (
            "local_new_ship_probability",
            "broad_new_ship_probability",
            "broad_drop_ship_probability",
        ):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1.")
        if self.new_ship_weight_min < 0:
            raise ValueError("new_ship_weight_min cannot be negative.")
        if self.new_ship_weight_max < self.new_ship_weight_min:
            raise ValueError("new_ship_weight_max must be >= new_ship_weight_min.")
        if self.min_ship_weight < 0:
            raise ValueError("min_ship_weight cannot be negative.")
        if self.max_active_ship_types < 1:
            raise ValueError("max_active_ship_types must be at least 1.")
        if not 0 <= self.analysis_model_weight_min <= 1:
            raise ValueError("analysis_model_weight_min must be in [0, 1].")
        if not 0 <= self.analysis_model_weight_max <= 1:
            raise ValueError("analysis_model_weight_max must be in [0, 1].")
        if self.analysis_model_weight_max < self.analysis_model_weight_min:
            raise ValueError(
                "analysis_model_weight_max must be >= analysis_model_weight_min."
            )


class MLSeededPopulationInitializer(PopulationInitializer):
    """Initialize the optimizer around an M0/M1 prediction for each defense."""

    def __init__(
        self,
        predict_genome: GenomePredictor,
        *,
        analysis_initializer: PopulationInitializer | None = None,
        initializer_config: MLSeededInitializerConfig = MLSeededInitializerConfig(),
    ) -> None:
        self.predict_genome = predict_genome
        self.analysis_initializer = analysis_initializer
        self.initializer_config = initializer_config

    def create_population(
        self,
        problem: AttackProblem,
        config: EvolutionConfig,
        rng: np.random.Generator,
    ) -> Sequence[AttackGenome]:
        ml_seed = canonicalize_genome(
            self.predict_genome(problem.defender),
            config,
        )

        population: list[AttackGenome] = []
        exact_count = min(
            self.initializer_config.exact_seed_count,
            config.population_size,
        )
        population.extend(ml_seed for _ in range(exact_count))

        remaining = config.population_size - len(population)
        if remaining <= 0:
            return population

        counts = self._allocate_counts(
            remaining,
            (
                self.initializer_config.local_fraction,
                self.initializer_config.broad_fraction,
                self.initializer_config.analysis_blend_fraction,
            ),
        )

        population.extend(
            self._perturb(
                ml_seed,
                config,
                rng,
                log_weight_sigma=self.initializer_config.local_log_weight_sigma,
                multiplier_sigma=self.initializer_config.local_multiplier_sigma,
                new_ship_probability=self.initializer_config.local_new_ship_probability,
                drop_ship_probability=0.0,
            )
            for _ in range(counts[0])
        )
        population.extend(
            self._perturb(
                ml_seed,
                config,
                rng,
                log_weight_sigma=self.initializer_config.broad_log_weight_sigma,
                multiplier_sigma=self.initializer_config.broad_multiplier_sigma,
                new_ship_probability=self.initializer_config.broad_new_ship_probability,
                drop_ship_probability=self.initializer_config.broad_drop_ship_probability,
            )
            for _ in range(counts[1])
        )

        if counts[2] > 0:
            population.extend(
                self._analysis_blends(
                    ml_seed,
                    problem,
                    config,
                    rng,
                    count=counts[2],
                )
            )

        if len(population) != config.population_size:
            raise RuntimeError(
                "MLSeededPopulationInitializer created "
                f"{len(population)} genomes, expected {config.population_size}."
            )
        rng.shuffle(population)
        return population

    def _analysis_blends(
        self,
        ml_seed: AttackGenome,
        problem: AttackProblem,
        config: EvolutionConfig,
        rng: np.random.Generator,
        *,
        count: int,
    ) -> list[AttackGenome]:
        if self.analysis_initializer is None:
            return [
                self._perturb(
                    ml_seed,
                    config,
                    rng,
                    log_weight_sigma=self.initializer_config.broad_log_weight_sigma,
                    multiplier_sigma=self.initializer_config.broad_multiplier_sigma,
                    new_ship_probability=self.initializer_config.broad_new_ship_probability,
                    drop_ship_probability=self.initializer_config.broad_drop_ship_probability,
                )
                for _ in range(count)
            ]

        analysis_population = list(
            self.analysis_initializer.create_population(problem, config, rng)
        )
        if not analysis_population:
            raise ValueError("analysis_initializer returned an empty population.")

        results: list[AttackGenome] = []
        for index in range(count):
            other = canonicalize_genome(
                analysis_population[index % len(analysis_population)],
                config,
            )
            model_weight = float(
                rng.uniform(
                    self.initializer_config.analysis_model_weight_min,
                    self.initializer_config.analysis_model_weight_max,
                )
            )
            results.append(
                self._blend(ml_seed, other, model_weight, config)
            )
        return results

    def _perturb(
        self,
        seed: AttackGenome,
        config: EvolutionConfig,
        rng: np.random.Generator,
        *,
        log_weight_sigma: float,
        multiplier_sigma: float,
        new_ship_probability: float,
        drop_ship_probability: float,
    ) -> AttackGenome:
        base = seed.normalized_weights(config.allowed_ships)
        weights: dict[str, float] = {}

        for ship_name in config.allowed_ships:
            value = float(base.get(ship_name, 0.0))
            if value > 0:
                if drop_ship_probability > 0 and rng.random() < drop_ship_probability:
                    value = 0.0
                elif log_weight_sigma > 0:
                    value *= float(np.exp(rng.normal(0.0, log_weight_sigma)))
            elif rng.random() < new_ship_probability:
                value = float(
                    rng.uniform(
                        self.initializer_config.new_ship_weight_min,
                        self.initializer_config.new_ship_weight_max,
                    )
                )
            weights[ship_name] = max(0.0, value)

        weights = self._sparsify(weights, config.allowed_ships)
        multiplier = float(
            np.clip(
                float(seed.points_multiplier)
                + float(rng.normal(0.0, multiplier_sigma)),
                config.min_points_multiplier,
                config.max_points_multiplier,
            )
        )
        return canonicalize_genome(
            AttackGenome(weights, multiplier),
            config,
            fallback=seed,
        )

    def _sparsify(
        self,
        weights: Mapping[str, float],
        allowed_ships: Sequence[str],
    ) -> dict[str, float]:
        positive = {
            ship: max(0.0, float(weights.get(ship, 0.0)))
            for ship in allowed_ships
        }
        kept = {
            ship: value
            for ship, value in positive.items()
            if value >= self.initializer_config.min_ship_weight
        }
        if not kept:
            best = max(positive, key=positive.get)
            kept = {best: max(positive[best], 1.0)}

        if len(kept) > self.initializer_config.max_active_ship_types:
            ranked = sorted(kept.items(), key=lambda item: item[1], reverse=True)
            kept = dict(ranked[: self.initializer_config.max_active_ship_types])

        total = sum(kept.values())
        return {ship: value / total for ship, value in kept.items()}

    @staticmethod
    def _blend(
        model: AttackGenome,
        other: AttackGenome,
        model_weight: float,
        config: EvolutionConfig,
    ) -> AttackGenome:
        model_values = model.normalized_weights(config.allowed_ships)
        other_values = other.normalized_weights(config.allowed_ships)
        weights = {
            ship: (
                model_weight * model_values.get(ship, 0.0)
                + (1.0 - model_weight) * other_values.get(ship, 0.0)
            )
            for ship in config.allowed_ships
        }
        multiplier = (
            model_weight * float(model.points_multiplier)
            + (1.0 - model_weight) * float(other.points_multiplier)
        )
        return canonicalize_genome(
            AttackGenome(weights, multiplier),
            config,
            fallback=model,
        )

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
