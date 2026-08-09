from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .models import (
    AttackGenome,
    AttackProblem,
    EvolutionConfig,
    FitnessEvaluation,
)


class PopulationInitializer(ABC):
    """Creates generation zero.

    Intentionally left without a concrete implementation. The next design step
    can decide whether generation zero is random, data-driven, seeded from
    battle-analysis heuristics, or a mixture of several strategies.
    """

    @abstractmethod
    def create_population(
        self,
        problem: AttackProblem,
        config: EvolutionConfig,
        rng: np.random.Generator,
    ) -> Sequence[AttackGenome]:
        raise NotImplementedError


class FitnessEvaluator(ABC):
    """Scores a complete population for one fixed defender.

    The batch-oriented API is deliberate: the concrete implementation can later
    parallelize simulations, evaluate several combat seeds per genome, cache
    repeated fleets, or use vectorized/precomputed information.
    """

    @abstractmethod
    def evaluate_population(
        self,
        problem: AttackProblem,
        genomes: Sequence[AttackGenome],
        rng: np.random.Generator,
    ) -> Sequence[FitnessEvaluation]:
        raise NotImplementedError
