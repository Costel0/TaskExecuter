"""Phase B: supervised prediction of promising OGIA attack genomes."""

from .data import (
    ATTACK_SHIPS,
    DEFENSE_UNITS,
    FeatureScaler,
    PerfectPairExample,
    defender_to_features,
    examples_to_arrays,
    load_perfect_pairs,
)
from .model import AttackPredictorConfig, AttackPredictorNet
from .predictor import AttackPrediction, AttackPredictor
from .training import (
    TrainingArtifacts,
    TrainingConfig,
    train_attack_predictor,
)

__all__ = [
    "ATTACK_SHIPS",
    "DEFENSE_UNITS",
    "AttackPrediction",
    "AttackPredictor",
    "AttackPredictorConfig",
    "AttackPredictorNet",
    "FeatureScaler",
    "PerfectPairExample",
    "TrainingArtifacts",
    "TrainingConfig",
    "defender_to_features",
    "examples_to_arrays",
    "load_perfect_pairs",
    "train_attack_predictor",
]
