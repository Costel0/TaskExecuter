from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Callable, Mapping, Sequence

import numpy as np

from ..OgameUtils import CombatConfig, TechLevels
from ..optimizer import (
    AttackGenome,
    AttackProblem,
    DefenseFitnessConfig,
    EvolutionConfig,
    ReliableDefenseFitnessEvaluator,
    make_combat_seeds,
)
from .data import ATTACK_SHIPS, PerfectPairExample


FIXED_TECH_LEVEL = 15


@dataclass(frozen=True)
class CombatEvaluationConfig:
    """Settings for out-of-fold combat evaluation against the Phase-A oracle."""

    simulations_per_attack: int = 128
    reliable_win_rate: float = 0.90
    seed: int = 123_456
    progress_every: int = 10

    def __post_init__(self) -> None:
        if self.simulations_per_attack < 1:
            raise ValueError("simulations_per_attack must be at least 1.")
        if not 0 < self.reliable_win_rate <= 1:
            raise ValueError("reliable_win_rate must be in (0, 1].")
        if self.progress_every < 1:
            raise ValueError("progress_every must be at least 1.")


def _evaluation_payload(evaluation) -> dict:
    metrics = dict(evaluation.metrics)
    return {
        "score": float(evaluation.score),
        "wins": int(metrics["wins"]),
        "simulations": int(metrics["simulations"]),
        "win_rate": float(metrics["win_rate"]),
        "mean_loss_ratio": float(metrics["mean_loss_ratio"]),
        "requested_multiplier": float(metrics["requested_multiplier"]),
        "actual_multiplier": float(metrics["actual_multiplier"]),
        "defender_points": float(metrics["defender_points"]),
        "target_attacker_points": float(metrics["target_attacker_points"]),
        "actual_attacker_points": float(metrics["actual_attacker_points"]),
        "fleet": {
            str(name): int(count)
            for name, count in dict(metrics["fleet"]).items()
        },
        "reliable": bool(metrics["reliable"]),
    }


def _prediction_by_pair_id(
    oof_predictions: Sequence[Mapping],
) -> dict[str, Mapping]:
    result: dict[str, Mapping] = {}
    for row in oof_predictions:
        pair_id = str(row["pair_id"])
        if pair_id in result:
            raise ValueError(f"Duplicate OOF prediction for pair_id={pair_id!r}.")
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping):
            raise ValueError(f"Missing OOF prediction for pair_id={pair_id!r}.")
        result[pair_id] = prediction
    return result


def _tech_levels(values: Mapping) -> TechLevels:
    if not values:
        return TechLevels(
            weapons=FIXED_TECH_LEVEL,
            shielding=FIXED_TECH_LEVEL,
            armour=FIXED_TECH_LEVEL,
        )
    return TechLevels(
        weapons=int(values.get("weapons", 0)),
        shielding=int(values.get("shielding", 0)),
        armour=int(values.get("armour", 0)),
    )


def _combat_config(values: Mapping) -> CombatConfig:
    if not values:
        return CombatConfig()
    allowed = set(CombatConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Perfect pair contains unsupported CombatConfig fields: {sorted(unknown)}"
        )
    return CombatConfig(**{str(name): value for name, value in values.items()})


def _aggregate(rows: Sequence[dict]) -> dict:
    if not rows:
        return {
            "pair_count": 0,
            "oracle_reliable_rate": 0.0,
            "prediction_reliable_rate": 0.0,
            "prediction_reliable_when_oracle_reliable_rate": 0.0,
            "mean_prediction_win_rate": 0.0,
            "mean_oracle_win_rate": 0.0,
            "mean_oracle_efficiency": 0.0,
            "median_oracle_efficiency": 0.0,
            "oracle_efficiency_at_least_90pct": 0.0,
            "oracle_efficiency_at_least_95pct": 0.0,
            "oracle_efficiency_at_least_99pct": 0.0,
            "mean_fitness_score_ratio": 0.0,
            "prediction_beats_oracle_score_rate": 0.0,
        }

    oracle_reliable = np.asarray(
        [bool(row["oracle"]["reliable"]) for row in rows],
        dtype=bool,
    )
    prediction_reliable = np.asarray(
        [bool(row["prediction"]["reliable"]) for row in rows],
        dtype=bool,
    )
    prediction_win_rates = np.asarray(
        [float(row["prediction"]["win_rate"]) for row in rows],
        dtype=float,
    )
    oracle_win_rates = np.asarray(
        [float(row["oracle"]["win_rate"]) for row in rows],
        dtype=float,
    )

    comparable = [row for row in rows if bool(row["oracle"]["reliable"])]
    efficiencies = [
        float(row["oracle_efficiency"])
        for row in comparable
        if row["oracle_efficiency"] is not None
    ]
    score_ratios = [
        float(row["fitness_score_ratio"])
        for row in comparable
        if row["fitness_score_ratio"] is not None
    ]

    if efficiencies:
        efficiency_array = np.asarray(efficiencies, dtype=float)
        mean_efficiency = float(np.mean(efficiency_array))
        median_efficiency = float(median(efficiencies))
        at_90 = float(np.mean(efficiency_array >= 0.90))
        at_95 = float(np.mean(efficiency_array >= 0.95))
        at_99 = float(np.mean(efficiency_array >= 0.99))
    else:
        mean_efficiency = median_efficiency = 0.0
        at_90 = at_95 = at_99 = 0.0

    if comparable:
        reliable_when_oracle_reliable = float(
            np.mean(
                [bool(row["prediction"]["reliable"]) for row in comparable]
            )
        )
        beats_oracle = float(
            np.mean(
                [
                    float(row["prediction"]["score"])
                    > float(row["oracle"]["score"])
                    for row in comparable
                ]
            )
        )
    else:
        reliable_when_oracle_reliable = 0.0
        beats_oracle = 0.0

    return {
        "pair_count": int(len(rows)),
        "oracle_reliable_pairs": int(np.sum(oracle_reliable)),
        "oracle_reliable_rate": float(np.mean(oracle_reliable)),
        "prediction_reliable_pairs": int(np.sum(prediction_reliable)),
        "prediction_reliable_rate": float(np.mean(prediction_reliable)),
        "prediction_reliable_when_oracle_reliable_rate": (
            reliable_when_oracle_reliable
        ),
        "mean_prediction_win_rate": float(np.mean(prediction_win_rates)),
        "mean_oracle_win_rate": float(np.mean(oracle_win_rates)),
        "mean_oracle_efficiency": mean_efficiency,
        "median_oracle_efficiency": median_efficiency,
        "oracle_efficiency_at_least_90pct": at_90,
        "oracle_efficiency_at_least_95pct": at_95,
        "oracle_efficiency_at_least_99pct": at_99,
        "mean_fitness_score_ratio": (
            float(np.mean(score_ratios)) if score_ratios else 0.0
        ),
        "prediction_beats_oracle_score_rate": beats_oracle,
    }


def evaluate_oof_against_oracle(
    examples: Sequence[PerfectPairExample],
    oof_predictions: Sequence[Mapping],
    *,
    config: CombatEvaluationConfig = CombatEvaluationConfig(),
    progress: Callable[[str], None] | None = print,
) -> dict:
    """Re-simulate OOF predictions and oracle attacks on fresh common seeds.

    Every defense receives fresh deterministic combat seeds. Its OOF prediction
    and its Phase-A oracle attack are evaluated on exactly the same seeds, so
    their comparison is not distorted by simulator randomness.

    ``oracle_efficiency`` is deliberately task-oriented rather than a distance
    between attack vectors. If the predicted attack is not reliable it is 0.
    Otherwise it is ``oracle_actual_multiplier / predicted_actual_multiplier``.
    Therefore 1.0 means the model achieved oracle-level point efficiency, 0.9
    means it needed about 11% more attacker points, and values above 1.0 mean it
    found a smaller reliable attack on this independent simulation sample.
    """

    rows = tuple(examples)
    prediction_map = _prediction_by_pair_id(oof_predictions)
    missing = [row.pair_id for row in rows if row.pair_id not in prediction_map]
    if missing:
        raise ValueError(
            "Missing OOF predictions for perfect pairs: "
            + ", ".join(missing[:5])
        )

    evolution_config = EvolutionConfig(allowed_ships=ATTACK_SHIPS)
    fitness_config = DefenseFitnessConfig(
        simulations_per_genome=config.simulations_per_attack,
        reliable_win_rate=config.reliable_win_rate,
    )
    rng = np.random.default_rng(config.seed)

    evaluated_rows: list[dict] = []
    for index, example in enumerate(rows, start=1):
        prediction = prediction_map[example.pair_id]
        predicted_weights = prediction.get("ship_weights")
        if not isinstance(predicted_weights, Mapping):
            raise ValueError(
                f"OOF prediction for {example.pair_id!r} has no ship_weights."
            )

        oracle_genome = AttackGenome(
            ship_weights=example.ship_weights,
            points_multiplier=float(example.points_multiplier),
        )
        predicted_genome = AttackGenome(
            ship_weights={
                ship: float(predicted_weights.get(ship, 0.0))
                for ship in ATTACK_SHIPS
            },
            points_multiplier=float(prediction["points_multiplier"]),
        )
        problem = AttackProblem(
            defender=example.defender,
            attacker_tech=_tech_levels(example.attacker_tech),
            defender_tech=_tech_levels(example.defender_tech),
            combat_config=_combat_config(example.combat_config),
        )

        pair_seed = int(rng.integers(0, 2**32 - 1, dtype=np.uint64))
        combat_seeds = make_combat_seeds(
            config.simulations_per_attack,
            seed=pair_seed,
        )
        evaluator = ReliableDefenseFitnessEvaluator(
            evolution_config=evolution_config,
            fitness_config=fitness_config,
            combat_seed=pair_seed,
        )
        oracle_evaluation = evaluator.evaluate_genome(
            problem,
            oracle_genome,
            combat_seeds=combat_seeds,
            use_cache=False,
        )
        prediction_evaluation = evaluator.evaluate_genome(
            problem,
            predicted_genome,
            combat_seeds=combat_seeds,
            use_cache=False,
        )

        oracle = _evaluation_payload(oracle_evaluation)
        predicted = _evaluation_payload(prediction_evaluation)

        if oracle["reliable"]:
            if predicted["reliable"] and predicted["actual_multiplier"] > 0:
                oracle_efficiency = float(
                    oracle["actual_multiplier"]
                    / predicted["actual_multiplier"]
                )
            else:
                oracle_efficiency = 0.0
        else:
            oracle_efficiency = None

        if float(oracle["score"]) > 0:
            fitness_score_ratio = float(
                max(0.0, float(predicted["score"]))
                / float(oracle["score"])
            )
        else:
            fitness_score_ratio = None

        evaluated_rows.append(
            {
                "pair_id": example.pair_id,
                "combat_seed": pair_seed,
                "oracle_efficiency": oracle_efficiency,
                "fitness_score_ratio": fitness_score_ratio,
                "oracle": oracle,
                "prediction": predicted,
            }
        )

        if progress and (
            index == 1
            or index == len(rows)
            or index % config.progress_every == 0
        ):
            efficiency_text = (
                "n/a"
                if oracle_efficiency is None
                else f"{oracle_efficiency:.2%}"
            )
            progress(
                f"[combat {index}/{len(rows)}] "
                f"model_win={predicted['win_rate']:.1%} "
                f"oracle_win={oracle['win_rate']:.1%} "
                f"oracle_efficiency={efficiency_text}"
            )

    return {
        "schema_version": 1,
        "definition": (
            "oracle_efficiency = oracle actual attacker multiplier / model "
            "actual attacker multiplier when both attacks are reliable; "
            "0 when the model attack is unreliable"
        ),
        "config": {
            "simulations_per_attack": config.simulations_per_attack,
            "reliable_win_rate": config.reliable_win_rate,
            "seed": config.seed,
            "combat_inputs": "recorded per perfect pair; Phase-A tech-15/default-config fallback",
        },
        "summary": _aggregate(evaluated_rows),
        "pairs": evaluated_rows,
    }
