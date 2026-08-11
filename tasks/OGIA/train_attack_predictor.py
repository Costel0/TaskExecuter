from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .attack_predictor import (
    ATTACK_SHIPS,
    CombatEvaluationConfig,
    PredictionPostprocessConfig,
    TrainingConfig,
    deduplicate_perfect_pairs,
    defense_group_key,
    evaluate_oof_against_oracle,
    load_perfect_pairs,
    train_attack_predictor,
)


DEFAULT_INPUT = Path("data/OGIA/perfect_pairs")
DEFAULT_MODEL_DIR = Path("data/OGIA/models")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _default_output_paths() -> tuple[Path, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stem = f"attack_predictor_m0_{timestamp}"
    return (
        DEFAULT_MODEL_DIR / f"{stem}.pt",
        DEFAULT_MODEL_DIR / f"{stem}_report.json",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogia-train-attack-predictor",
        description=(
            "Train Phase-B M0 from validated perfect pairs and evaluate "
            "out-of-fold attacks with the OGame combat engine."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="Perfect-pair JSONL file or directory.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--folds", type=_positive_int, default=5)
    parser.add_argument("--epochs", type=_positive_int, default=500)
    parser.add_argument("--patience", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--weight-decay", type=_non_negative_float, default=1e-4)
    parser.add_argument(
        "--multiplier-loss-weight",
        type=_non_negative_float,
        default=1.0,
    )
    parser.add_argument(
        "--multiplier-underprediction-weight",
        type=_positive_float,
        default=3.0,
        help="Extra loss multiplier when M0 predicts too few points (default: 3).",
    )
    parser.add_argument(
        "--multiplier-loss-beta",
        type=_positive_float,
        default=0.10,
        help="Smooth-L1 beta in real multiplier units (default: 0.10).",
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=_positive_int,
        default=[128, 64],
    )
    parser.add_argument("--dropout", type=_non_negative_float, default=0.05)
    parser.add_argument("--min-multiplier", type=_positive_float, default=0.50)
    parser.add_argument("--max-multiplier", type=_positive_float, default=6.00)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")

    postprocess = parser.add_argument_group("prediction postprocessing")
    postprocess.add_argument(
        "--min-ship-weight",
        type=_non_negative_float,
        default=0.01,
        help="Drop predicted ship shares below this value (default: 0.01).",
    )
    postprocess.add_argument(
        "--max-ship-types",
        type=_positive_int,
        default=5,
        help="Maximum active ship types after sparsification (default: 5).",
    )
    postprocess.add_argument(
        "--multiplier-safety-margin",
        type=_non_negative_float,
        default=0.03,
        help="Small deployment margin added after prediction (default: 0.03x).",
    )

    combat = parser.add_argument_group("combat-aware OOF evaluation")
    combat.add_argument("--combat-simulations", type=_positive_int, default=128)
    combat.add_argument(
        "--combat-reliable-win-rate",
        type=_probability,
        default=0.90,
    )
    combat.add_argument("--combat-seed", type=int, default=123_456)
    combat.add_argument("--combat-progress-every", type=_positive_int, default=10)
    combat.add_argument("--skip-combat-evaluation", action="store_true")
    return parser


def run(args=None) -> int:
    parsed = _build_parser().parse_args(args)
    if parsed.dropout >= 1:
        raise ValueError("--dropout must be smaller than 1.")
    if parsed.max_multiplier <= parsed.min_multiplier:
        raise ValueError("--max-multiplier must be greater than --min-multiplier.")
    if parsed.min_ship_weight >= 1:
        raise ValueError("--min-ship-weight must be smaller than 1.")
    if parsed.max_ship_types > len(ATTACK_SHIPS):
        raise ValueError(
            f"--max-ship-types cannot exceed {len(ATTACK_SHIPS)}."
        )
    if parsed.multiplier_underprediction_weight < 1:
        raise ValueError("--multiplier-underprediction-weight must be >= 1.")

    default_output, default_report = _default_output_paths()
    output_path = parsed.output or default_output
    report_path = parsed.report or (
        default_report
        if parsed.output is None
        else parsed.output.with_name(f"{parsed.output.stem}_report.json")
    )

    raw_examples = load_perfect_pairs(parsed.input, deduplicate=False)
    examples = deduplicate_perfect_pairs(raw_examples)
    duplicate_count = len(raw_examples) - len(examples)
    defense_count = len({defense_group_key(row) for row in examples})
    source_files = sorted({row.source_path for row in raw_examples})

    print(f"Raw JSONL pairs:       {len(raw_examples)}", flush=True)
    print(f"Exact duplicates:      {duplicate_count}", flush=True)
    print(f"Training pairs:        {len(examples)}", flush=True)
    print(f"Distinct defenses:     {defense_count}", flush=True)
    print("CV split:              grouped by defense", flush=True)
    print(f"Source files:          {len(source_files)}", flush=True)
    for source in source_files:
        print(f"  - {source}", flush=True)
    print(f"Model output:          {output_path}", flush=True)
    print(f"Report output:         {report_path}", flush=True)

    config = TrainingConfig(
        folds=parsed.folds,
        max_epochs=parsed.epochs,
        patience=parsed.patience,
        batch_size=parsed.batch_size,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        multiplier_loss_weight=parsed.multiplier_loss_weight,
        multiplier_underprediction_weight=(
            parsed.multiplier_underprediction_weight
        ),
        multiplier_loss_beta=parsed.multiplier_loss_beta,
        hidden_dims=tuple(parsed.hidden_dims),
        dropout=parsed.dropout,
        min_multiplier=parsed.min_multiplier,
        max_multiplier=parsed.max_multiplier,
        seed=parsed.seed,
    )
    postprocess_config = PredictionPostprocessConfig(
        min_ship_weight=parsed.min_ship_weight,
        max_ship_types=parsed.max_ship_types,
        multiplier_safety_margin=parsed.multiplier_safety_margin,
    )

    artifacts = train_attack_predictor(
        examples,
        output_path=output_path,
        config=config,
        postprocess_config=postprocess_config,
        device=parsed.device,
    )

    report = dict(artifacts.report)
    report["dataset"] = {
        "raw_pair_count": len(raw_examples),
        "exact_duplicate_count": duplicate_count,
        "training_pair_count": len(examples),
        "distinct_defense_count": defense_count,
        "deduplicated": True,
        "cv_grouped_by_defense": True,
    }
    report["source_files"] = source_files
    report["checkpoint_path"] = str(artifacts.checkpoint_path)

    if not parsed.skip_combat_evaluation:
        print("", flush=True)
        print("Combat-aware OOF evaluation against Phase-A oracle:", flush=True)
        report["combat_evaluation"] = evaluate_oof_against_oracle(
            examples,
            report["oof_predictions"],
            config=CombatEvaluationConfig(
                simulations_per_attack=parsed.combat_simulations,
                reliable_win_rate=parsed.combat_reliable_win_rate,
                seed=parsed.combat_seed,
                progress_every=parsed.combat_progress_every,
            ),
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metrics = report["oof_metrics"]
    print("", flush=True)
    print("Cross-validated M0 imitation metrics:", flush=True)
    print(
        f"  composition L1 distance: {metrics['composition_l1_distance']:.6f}",
        flush=True,
    )
    print(
        f"  composition cosine:      {metrics['composition_cosine_similarity']:.6f}",
        flush=True,
    )
    print(
        f"  top ship accuracy:       {metrics['top_ship_accuracy']:.2%}",
        flush=True,
    )
    print(
        f"  multiplier MAE:          {metrics['multiplier_mae']:.6f}",
        flush=True,
    )
    print(
        "  multiplier signed error: "
        f"{metrics['multiplier_mean_signed_error']:+.6f}",
        flush=True,
    )
    print(
        "  multiplier under-rate:   "
        f"{metrics['multiplier_underprediction_rate']:.2%}",
        flush=True,
    )

    combat_report = report.get("combat_evaluation")
    if combat_report:
        summary = combat_report["summary"]
        print("", flush=True)
        print("Combat-aware OOF metrics (primary):", flush=True)
        print(
            f"  model reliable attacks:   {summary['prediction_reliable_rate']:.2%}",
            flush=True,
        )
        print(
            f"  oracle reliable attacks:  {summary['oracle_reliable_rate']:.2%}",
            flush=True,
        )
        print(
            f"  mean oracle efficiency:   {summary['mean_oracle_efficiency']:.2%}",
            flush=True,
        )
        print(
            f"  median oracle efficiency: {summary['median_oracle_efficiency']:.2%}",
            flush=True,
        )
        print(
            "  >= 90% oracle efficiency: "
            f"{summary['oracle_efficiency_at_least_90pct']:.2%}",
            flush=True,
        )
        print(
            "  >= 95% oracle efficiency: "
            f"{summary['oracle_efficiency_at_least_95pct']:.2%}",
            flush=True,
        )
        print(
            "  >= 99% oracle efficiency: "
            f"{summary['oracle_efficiency_at_least_99pct']:.2%}",
            flush=True,
        )
        print(
            f"  mean fitness score ratio: {summary['mean_fitness_score_ratio']:.2%}",
            flush=True,
        )

    print("", flush=True)
    print(f"Saved checkpoint: {artifacts.checkpoint_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
