from __future__ import annotations

from collections.abc import Sequence
import statistics
from typing import Callable

import numpy as np

from .interfaces import FitnessEvaluator, PopulationInitializer
from .models import (
    AttackGenome,
    AttackProblem,
    EvaluatedIndividual,
    EvolutionConfig,
    GenerationStats,
    OptimizationResult,
)
from .operators import canonicalize_genome, crossover, mutate, tournament_select

GenerationCallback = Callable[[GenerationStats], None]


class EvolutionaryOptimizer:
    """Generic natural-selection loop for one OGame defense.

    The two domain-specific decisions intentionally live outside this class:
    generation-zero construction (``PopulationInitializer``) and battle value
    (``FitnessEvaluator``).
    """

    def __init__(
        self,
        *,
        config: EvolutionConfig,
        initializer: PopulationInitializer,
        fitness_evaluator: FitnessEvaluator,
        on_generation: GenerationCallback | None = None,
    ) -> None:
        self.config = config
        self.initializer = initializer
        self.fitness_evaluator = fitness_evaluator
        self.on_generation = on_generation

    def optimize(
        self,
        problem: AttackProblem,
        *,
        seed: int | None = None,
    ) -> OptimizationResult:
        rng = np.random.default_rng(seed)

        genomes = list(
            self.initializer.create_population(problem, self.config, rng)
        )
        self._validate_initial_population(genomes)
        genomes = [
            canonicalize_genome(genome, self.config)
            for genome in genomes
        ]

        population = self._evaluate(problem, genomes, rng)
        history: list[GenerationStats] = [
            self._generation_stats(0, population)
        ]
        self._notify(history[-1])

        for generation in range(1, self.config.generations + 1):
            ranked = sorted(
                population,
                key=lambda individual: individual.evaluation.score,
                reverse=True,
            )
            elites = ranked[: self.config.elite_count]

            offspring_genomes: list[AttackGenome] = []
            required_offspring = self.config.population_size - len(elites)

            while len(offspring_genomes) < required_offspring:
                parent_a = tournament_select(population, self.config, rng)
                parent_b = tournament_select(population, self.config, rng)
                child_a, child_b = crossover(parent_a, parent_b, self.config, rng)

                offspring_genomes.append(mutate(child_a, self.config, rng))
                if len(offspring_genomes) < required_offspring:
                    offspring_genomes.append(mutate(child_b, self.config, rng))

            offspring = self._evaluate(problem, offspring_genomes, rng)
            population = [*elites, *offspring]

            stats = self._generation_stats(generation, population)
            history.append(stats)
            self._notify(stats)

        final_population = tuple(sorted(
            population,
            key=lambda individual: individual.evaluation.score,
            reverse=True,
        ))
        return OptimizationResult(
            best=final_population[0],
            history=tuple(history),
            final_population=final_population,
        )

    def _evaluate(
        self,
        problem: AttackProblem,
        genomes: Sequence[AttackGenome],
        rng: np.random.Generator,
    ) -> list[EvaluatedIndividual]:
        evaluations = list(
            self.fitness_evaluator.evaluate_population(problem, genomes, rng)
        )
        if len(evaluations) != len(genomes):
            raise ValueError(
                "FitnessEvaluator must return exactly one evaluation per genome."
            )

        return [
            EvaluatedIndividual(genome=genome, evaluation=evaluation)
            for genome, evaluation in zip(genomes, evaluations)
        ]

    def _validate_initial_population(
        self,
        genomes: Sequence[AttackGenome],
    ) -> None:
        if len(genomes) != self.config.population_size:
            raise ValueError(
                "PopulationInitializer returned "
                f"{len(genomes)} genomes, expected {self.config.population_size}."
            )

    @staticmethod
    def _generation_stats(
        generation: int,
        population: Sequence[EvaluatedIndividual],
    ) -> GenerationStats:
        best = max(
            population,
            key=lambda individual: individual.evaluation.score,
        )
        scores = [float(individual.evaluation.score) for individual in population]
        return GenerationStats(
            generation=generation,
            best_score=float(best.evaluation.score),
            mean_score=float(statistics.fmean(scores)),
            worst_score=float(min(scores)),
            best_genome=best.genome,
        )

    def _notify(self, stats: GenerationStats) -> None:
        if self.on_generation is not None:
            self.on_generation(stats)
