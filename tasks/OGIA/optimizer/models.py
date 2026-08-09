from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

from ..OgameUtils import CombatConfig, DEFAULT_RANDOM_ATTACK_SHIPS, TechLevels


@dataclass(frozen=True)
class AttackGenome:
    """Compact attack representation evolved by the optimizer.

    ``ship_weights`` stores relative point shares, not absolute ship counts.
    ``points_multiplier`` scales the defender's points to obtain the attacker's
    target point budget.
    """

    ship_weights: Mapping[str, float]
    points_multiplier: float

    def normalized_weights(
        self,
        allowed_ships: Sequence[str] | None = None,
    ) -> dict[str, float]:
        allowed = tuple(allowed_ships or self.ship_weights.keys())
        weights: dict[str, float] = {}

        for ship_name in allowed:
            raw_weight = float(self.ship_weights.get(ship_name, 0.0))
            if not math.isfinite(raw_weight) or raw_weight < 0:
                raise ValueError(
                    f"Invalid weight for {ship_name!r}: {raw_weight!r}."
                )
            weights[ship_name] = raw_weight

        total = sum(weights.values())
        if total <= 0:
            raise ValueError("An attack genome must contain at least one positive ship weight.")

        # Keep explicit zeroes when an allowed ship list is provided. This makes
        # persisted optimizer labels directly convertible to fixed-size ML
        # vectors while remaining compatible with fleet decoders that ignore 0%.
        return {
            ship_name: weight / total
            for ship_name, weight in weights.items()
        }

    def percentages(
        self,
        allowed_ships: Sequence[str] | None = None,
    ) -> dict[str, float]:
        return {
            ship_name: share * 100.0
            for ship_name, share in self.normalized_weights(allowed_ships).items()
        }


@dataclass(frozen=True)
class AttackProblem:
    """Everything fixed while optimizing one defender."""

    defender: Mapping[str, int]
    attacker_tech: TechLevels = field(default_factory=TechLevels)
    defender_tech: TechLevels = field(default_factory=TechLevels)
    defender_resources: Mapping[str, int] = field(default_factory=dict)
    combat_config: CombatConfig = field(default_factory=CombatConfig)
    loot_percentage: float = 0.75


@dataclass(frozen=True)
class FitnessEvaluation:
    """One optimizer evaluation. The evolutionary engine maximizes ``score``."""

    score: float
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)):
            raise ValueError("Fitness score must be finite.")


@dataclass(frozen=True)
class EvaluatedIndividual:
    genome: AttackGenome
    evaluation: FitnessEvaluation


@dataclass(frozen=True)
class GenerationStats:
    generation: int
    best_score: float
    mean_score: float
    worst_score: float
    best_genome: AttackGenome


@dataclass(frozen=True)
class OptimizationResult:
    best: EvaluatedIndividual
    history: tuple[GenerationStats, ...]
    final_population: tuple[EvaluatedIndividual, ...]


@dataclass(frozen=True)
class EvolutionConfig:
    """Structural parameters of the evolutionary search."""

    population_size: int = 128
    generations: int = 100
    elite_count: int = 8
    tournament_size: int = 4
    crossover_rate: float = 0.85
    per_gene_mutation_rate: float = 0.20
    multiplier_mutation_rate: float = 0.35
    weight_mutation_sigma: float = 0.10
    multiplier_mutation_sigma: float = 0.10
    min_points_multiplier: float = 0.25
    max_points_multiplier: float = 6.0
    allowed_ships: tuple[str, ...] = DEFAULT_RANDOM_ATTACK_SHIPS

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.generations < 0:
            raise ValueError("generations cannot be negative.")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [0, population_size).")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be in [2, population_size].")

        for field_name in (
            "crossover_rate",
            "per_gene_mutation_rate",
            "multiplier_mutation_rate",
        ):
            value = float(getattr(self, field_name))
            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1.")

        if self.weight_mutation_sigma < 0 or self.multiplier_mutation_sigma < 0:
            raise ValueError("Mutation sigmas cannot be negative.")
        if self.min_points_multiplier <= 0:
            raise ValueError("min_points_multiplier must be positive.")
        if self.max_points_multiplier < self.min_points_multiplier:
            raise ValueError("max_points_multiplier must be >= min_points_multiplier.")
        if not self.allowed_ships:
            raise ValueError("allowed_ships cannot be empty.")
        if len(set(self.allowed_ships)) != len(self.allowed_ships):
            raise ValueError("allowed_ships cannot contain duplicates.")
