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
    """Strategy interface for constructing generation zero."""

    @abstractmethod
    def create_population(
        self,
        problem: AttackProblem,
        config: EvolutionConfig,
        rng: np.random.Generator,
    ) -> Sequence[AttackGenome]:
        raise NotImplementedError


class FitnessEvaluator(ABC):
    """Score a complete population for one fixed defender.

    The batch-oriented API lets implementations share combat seeds, cache
    equivalent fleets and later add multiprocessing without changing the
    evolutionary engine.
    """

    @abstractmethod
    def evaluate_population(
        self,
        problem: AttackProblem,
        genomes: Sequence[AttackGenome],
        rng: np.random.Generator,
    ) -> Sequence[FitnessEvaluation]:
        raise NotImplementedError
