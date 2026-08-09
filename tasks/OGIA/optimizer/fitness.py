from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..OgameUtils import fleet_points, simulate_battle
from .decoder import DecodedAttack, decode_attack_genome
from .interfaces import FitnessEvaluator
from .models import AttackGenome, AttackProblem, EvolutionConfig, FitnessEvaluation


@dataclass(frozen=True)
class DefenseFitnessConfig:
    """Fitness settings for reliable, compact attacks.

    Fitness is intentionally not economic. It first requires a sufficiently
    reliable win rate and, only inside that feasible region, rewards smaller
    attack multipliers and lower attacker losses.
    """

    simulations_per_genome: int = 12
    reliable_win_rate: float = 0.90
    reliable_score_base: float = 100.0
    reliable_loss_penalty: float = 0.25
    reliable_win_bonus: float = 0.01
    unreliable_loss_penalty: float = 0.05
    unreliable_multiplier_penalty: float = 0.001

    def __post_init__(self) -> None:
        if self.simulations_per_genome < 1:
            raise ValueError("simulations_per_genome must be at least 1.")
        if not 0 < self.reliable_win_rate <= 1:
            raise ValueError("reliable_win_rate must be in (0, 1].")
        if self.reliable_score_base <= 1:
            raise ValueError("reliable_score_base must be > 1.")
        for field_name in (
            "reliable_loss_penalty",
            "reliable_win_bonus",
            "unreliable_loss_penalty",
            "unreliable_multiplier_penalty",
        ):
            if float(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative.")


class ReliableDefenseFitnessEvaluator(FitnessEvaluator):
    """Evaluate attacks against one fixed defense using common combat seeds.

    Every genome in every generation is evaluated on exactly the same combat
    seeds. Elites therefore remain comparable with descendants even though the
    engine does not re-evaluate preserved elites each generation.
    """

    def __init__(
        self,
        *,
        evolution_config: EvolutionConfig,
        fitness_config: DefenseFitnessConfig = DefenseFitnessConfig(),
        combat_seed: int | None = None,
    ) -> None:
        self.evolution_config = evolution_config
        self.fitness_config = fitness_config
        generator = np.random.default_rng(combat_seed)
        self._training_seeds = tuple(
            int(value)
            for value in generator.integers(
                0,
                2**32 - 1,
                size=fitness_config.simulations_per_genome,
                dtype=np.uint64,
            )
        )
        self._cache: dict[tuple[tuple[str, int], ...], FitnessEvaluation] = {}

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return self._training_seeds

    def evaluate_population(
        self,
        problem: AttackProblem,
        genomes: Sequence[AttackGenome],
        rng: np.random.Generator,
    ) -> Sequence[FitnessEvaluation]:
        del rng  # Seeds are fixed at evaluator construction time by design.
        return [self.evaluate_genome(problem, genome) for genome in genomes]

    def evaluate_genome(
        self,
        problem: AttackProblem,
        genome: AttackGenome,
        *,
        combat_seeds: Sequence[int] | None = None,
        use_cache: bool = True,
    ) -> FitnessEvaluation:
        decoded = decode_attack_genome(
            genome,
            problem.defender,
            allowed_ships=self.evolution_config.allowed_ships,
        )
        seeds = tuple(
            int(seed)
            for seed in (
                self._training_seeds if combat_seeds is None else combat_seeds
            )
        )
        if not seeds:
            raise ValueError("At least one combat seed is required.")

        cache_key = tuple(sorted((str(name), int(count)) for name, count in decoded.fleet.items()))
        if combat_seeds is None and use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        evaluation = self._evaluate_decoded(problem, decoded, seeds)
        if combat_seeds is None and use_cache:
            self._cache[cache_key] = evaluation
        return evaluation

    def _evaluate_decoded(
        self,
        problem: AttackProblem,
        decoded: DecodedAttack,
        combat_seeds: Sequence[int],
    ) -> FitnessEvaluation:
        if decoded.actual_attacker_points <= 0:
            raise ValueError("Decoded attack must contain positive attacker points.")

        wins = 0
        loss_ratios: list[float] = []

        for combat_seed in combat_seeds:
            result = simulate_battle(
                attacker=decoded.fleet,
                defender=problem.defender,
                attacker_tech=problem.attacker_tech,
                defender_tech=problem.defender_tech,
                config=problem.combat_config,
                seed=int(combat_seed),
                defender_resources=(problem.defender_resources or None),
                loot_percentage=problem.loot_percentage,
            )
            if result.winner == "attacker":
                wins += 1

            destroyed_points = fleet_points(result.attacker_destroyed)
            loss_ratios.append(
                min(
                    1.0,
                    max(0.0, destroyed_points / decoded.actual_attacker_points),
                )
            )

        simulations = len(combat_seeds)
        win_rate = wins / simulations
        mean_loss_ratio = float(np.mean(loss_ratios)) if loss_ratios else 0.0
        actual_multiplier = (
            decoded.actual_attacker_points / decoded.defender_points
            if decoded.defender_points > 0
            else float("inf")
        )

        score = self._score(
            win_rate=win_rate,
            mean_loss_ratio=mean_loss_ratio,
            actual_multiplier=actual_multiplier,
        )

        return FitnessEvaluation(
            score=float(score),
            metrics={
                "wins": int(wins),
                "simulations": int(simulations),
                "win_rate": float(win_rate),
                "mean_loss_ratio": float(mean_loss_ratio),
                "requested_multiplier": float(decoded.genome.points_multiplier),
                "actual_multiplier": float(actual_multiplier),
                "defender_points": float(decoded.defender_points),
                "target_attacker_points": float(decoded.target_attacker_points),
                "actual_attacker_points": float(decoded.actual_attacker_points),
                "fleet": dict(decoded.fleet),
                "reliable": bool(win_rate >= self.fitness_config.reliable_win_rate),
            },
        )

    def _score(
        self,
        *,
        win_rate: float,
        mean_loss_ratio: float,
        actual_multiplier: float,
    ) -> float:
        cfg = self.fitness_config

        if win_rate < cfg.reliable_win_rate:
            # Before reliability is reached, learning to win dominates. Losses
            # and fleet size are only tiny tie-breakers between equal win rates.
            return (
                float(win_rate)
                - cfg.unreliable_loss_penalty * float(mean_loss_ratio)
                - cfg.unreliable_multiplier_penalty * float(actual_multiplier)
            )

        # Any reliable attack always outranks every unreliable attack. Inside
        # this region lower real multiplier is the main objective; losses are a
        # secondary objective and extra reliability is only a small tie-breaker.
        return (
            cfg.reliable_score_base
            - float(actual_multiplier)
            - cfg.reliable_loss_penalty * float(mean_loss_ratio)
            + cfg.reliable_win_bonus * float(win_rate)
        )


def make_combat_seeds(
    count: int,
    *,
    seed: int | None,
) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("count must be at least 1.")
    generator = np.random.default_rng(seed)
    return tuple(
        int(value)
        for value in generator.integers(
            0,
            2**32 - 1,
            size=int(count),
            dtype=np.uint64,
        )
    )
