from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models import AttackGenome, EvaluatedIndividual, EvolutionConfig


def canonicalize_genome(
    genome: AttackGenome,
    config: EvolutionConfig,
    *,
    fallback: AttackGenome | None = None,
) -> AttackGenome:
    """Clamp the genome to the search domain and normalize composition weights."""

    weights = {
        ship_name: max(0.0, float(genome.ship_weights.get(ship_name, 0.0)))
        for ship_name in config.allowed_ships
    }

    total = sum(weights.values())
    if total <= 0:
        if fallback is None:
            raise ValueError("Genome has no positive ship weights and no fallback was supplied.")
        weights = {
            ship_name: max(0.0, float(fallback.ship_weights.get(ship_name, 0.0)))
            for ship_name in config.allowed_ships
        }
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Fallback genome also has no positive ship weights.")

    # Normalize first so the optional threshold always means point-share rather
    # than depending on the arbitrary raw scale produced by crossover/mutation.
    weights = {
        ship_name: weight / total
        for ship_name, weight in weights.items()
        if weight > 0
    }

    if config.min_ship_weight > 0:
        filtered = {
            ship_name: weight
            for ship_name, weight in weights.items()
            if weight >= config.min_ship_weight
        }
        if filtered:
            weights = filtered
        else:
            best_ship = max(weights, key=weights.get)
            weights = {best_ship: weights[best_ship]}

    if (
        config.max_active_ship_types is not None
        and len(weights) > config.max_active_ship_types
    ):
        ranked = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        weights = dict(ranked[: config.max_active_ship_types])

    total = sum(weights.values())
    weights = {
        ship_name: weight / total
        for ship_name, weight in weights.items()
        if weight > 0
    }

    multiplier = float(np.clip(
        float(genome.points_multiplier),
        config.min_points_multiplier,
        config.max_points_multiplier,
    ))

    return AttackGenome(
        ship_weights=weights,
        points_multiplier=multiplier,
    )


def tournament_select(
    population: Sequence[EvaluatedIndividual],
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> AttackGenome:
    indices = rng.choice(
        len(population),
        size=config.tournament_size,
        replace=False,
    )
    winner = max(
        (population[int(index)] for index in indices),
        key=lambda individual: individual.evaluation.score,
    )
    return winner.genome


def crossover(
    parent_a: AttackGenome,
    parent_b: AttackGenome,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> tuple[AttackGenome, AttackGenome]:
    """Blend two genomes while preserving the compact representation."""

    if rng.random() >= config.crossover_rate:
        return parent_a, parent_b

    child_a_weights: dict[str, float] = {}
    child_b_weights: dict[str, float] = {}

    for ship_name in config.allowed_ships:
        weight_a = float(parent_a.ship_weights.get(ship_name, 0.0))
        weight_b = float(parent_b.ship_weights.get(ship_name, 0.0))
        alpha = float(rng.random())
        child_a_weights[ship_name] = alpha * weight_a + (1.0 - alpha) * weight_b
        child_b_weights[ship_name] = alpha * weight_b + (1.0 - alpha) * weight_a

    alpha = float(rng.random())
    child_a_multiplier = (
        alpha * parent_a.points_multiplier
        + (1.0 - alpha) * parent_b.points_multiplier
    )
    child_b_multiplier = (
        alpha * parent_b.points_multiplier
        + (1.0 - alpha) * parent_a.points_multiplier
    )

    child_a = canonicalize_genome(
        AttackGenome(child_a_weights, child_a_multiplier),
        config,
        fallback=parent_a,
    )
    child_b = canonicalize_genome(
        AttackGenome(child_b_weights, child_b_multiplier),
        config,
        fallback=parent_b,
    )
    return child_a, child_b


def mutate(
    genome: AttackGenome,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> AttackGenome:
    """Apply small local changes to composition and total attack size."""

    weights = {
        ship_name: float(genome.ship_weights.get(ship_name, 0.0))
        for ship_name in config.allowed_ships
    }

    for ship_name in config.allowed_ships:
        if rng.random() < config.per_gene_mutation_rate:
            weights[ship_name] = max(
                0.0,
                weights[ship_name] + float(rng.normal(0.0, config.weight_mutation_sigma)),
            )

    multiplier = float(genome.points_multiplier)
    if rng.random() < config.multiplier_mutation_rate:
        # Multiplicative/log-normal mutation behaves similarly at different
        # defender scales and cannot accidentally create a negative multiplier.
        multiplier *= float(np.exp(rng.normal(0.0, config.multiplier_mutation_sigma)))

    return canonicalize_genome(
        AttackGenome(weights, multiplier),
        config,
        fallback=genome,
    )
