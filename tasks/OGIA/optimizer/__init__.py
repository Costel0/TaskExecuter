"""Evolutionary optimizer for OGIA Phase A."""

from .decoder import DecodedAttack, decode_attack_genome
from .engine import EvolutionaryOptimizer
from .fitness import (
    DefenseFitnessConfig,
    ReliableDefenseFitnessEvaluator,
    make_combat_seeds,
)
from .initializer import (
    HybridAnalysisPopulationInitializer,
    HybridInitializerConfig,
)
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
    "DefenseFitnessConfig",
    "EvaluatedIndividual",
    "EvolutionConfig",
    "EvolutionaryOptimizer",
    "FitnessEvaluation",
    "FitnessEvaluator",
    "GenerationStats",
    "HybridAnalysisPopulationInitializer",
    "HybridInitializerConfig",
    "OptimizationResult",
    "PopulationInitializer",
    "ReliableDefenseFitnessEvaluator",
    "decode_attack_genome",
    "make_combat_seeds",
]
