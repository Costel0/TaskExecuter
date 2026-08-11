"""Phase B: supervised prediction of promising OGIA attack genomes."""

from .combat_evaluation import (
    CombatEvaluationConfig,
    evaluate_oof_against_oracle,
)
from .data import (
    ATTACK_SHIPS,
    DEFENSE_UNITS,
    FeatureScaler,
    PerfectPairExample,
    deduplicate_perfect_pairs,
    defense_group_key,
    defender_to_features,
    examples_to_arrays,
    load_perfect_pairs,
)
from .model import AttackPredictorConfig, AttackPredictorNet
from .postprocessing import (
    PredictionPostprocessConfig,
    postprocess_prediction,
    sparsify_ship_weights,
)
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
    "CombatEvaluationConfig",
    "FeatureScaler",
    "PerfectPairExample",
    "PredictionPostprocessConfig",
    "TrainingArtifacts",
    "TrainingConfig",
    "deduplicate_perfect_pairs",
    "defense_group_key",
    "defender_to_features",
    "evaluate_oof_against_oracle",
    "examples_to_arrays",
    "load_perfect_pairs",
    "postprocess_prediction",
    "sparsify_ship_weights",
    "train_attack_predictor",
]
