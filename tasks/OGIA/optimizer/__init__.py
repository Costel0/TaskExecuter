"""Evolutionary optimizer architecture for OGIA Phase A.

Concrete generation-zero and fitness strategies are intentionally not included
until those decisions are designed from the battle-analysis results.
"""

from .decoder import DecodedAttack, decode_attack_genome
from .engine import EvolutionaryOptimizer
from .interfaces import FitnessEvaluator, PopulationInitializer
from .models import (
    AttackGenome,
    AttackProblem,
    EvaluatedIndividual,
    EvolutionConfig,
    FitnessEvaluation,
    GenerationStats,
    OptimizationResult,
)

__all__ = [
    "AttackGenome",
    "AttackProblem",
    "DecodedAttack",
    "EvaluatedIndividual",
    "EvolutionConfig",
    "EvolutionaryOptimizer",
    "FitnessEvaluation",
    "FitnessEvaluator",
    "GenerationStats",
    "OptimizationResult",
    "PopulationInitializer",
    "decode_attack_genome",
]
