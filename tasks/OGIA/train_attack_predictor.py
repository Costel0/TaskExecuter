from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .attack_predictor import (
    CombatEvaluationConfig,
    TrainingConfig,
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
        raise argparse.ArgumentTypeError(
            "must be a non-negative finite number"
        )
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            "must be a positive finite number"
        )
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
            "Train the Phase-B M0 attack predictor from validated "
            "perfect-pair JSONL files and evaluate its out-of-fold attacks "
            "with the real OGame combat engine."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Perfect-pair JSONL file or directory "
            "(default: data/OGIA/perfect_pairs)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pt checkpoint path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Output JSON training/evaluation report path.",
    )
    parser.add_argument("--folds", type=_positive_int, default=5)
    parser.add_argument("--epochs", type=_positive_int, default=500)
    parser.add_argument("--patience", type=_positive_int, default=50)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight-decay",
        type=_non_negative_float,
        default=1e-4,
    )
    parser.add_argument(
        "--multiplier-loss-weight",
        type=_non_negative_float,
        default=1.0,
    )
    parser.add_argument(
        "--hidden-dims",
        nargs="+",
        type=_positive_int,
        default=[128, 64],
    )
    parser.add_argument(
        "--dropout",
        type=_non_negative_float,
        default=0.05,
    )
    parser.add_argument(
        "--min-multiplier",
        type=_positive_float,
        default=0.50,
    )
    parser.add_argument(
        "--max-multiplier",
        type=_positive_float,
        default=6.00,
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Torch device: auto, cpu, cuda, cuda:0, ... "
            "(default: auto)."
        ),
    )

    combat = parser.add_argument_group("combat-aware OOF evaluation")
    combat.add_argument(
        "--combat-simulations",
        type=_positive_int,
        default=128,
        help=(
            "Fresh combat simulations for each model attack and oracle attack "
            "(default: 128)."
        ),
    )
    combat.add_argument(
        "--combat-reliable-win-rate",
        type=_probability,
        default=0.90,
        help=(
            "Win-rate threshold that makes an attack reliable during M0 "
            "evaluation (default: 0.90)."
        ),
    )
    combat.add_argument(
        "--combat-seed",
        type=int,
        default=123_456,
        help="Seed for fresh common combat simulations (default: 123456).",
    )
    combat.add_argument(
        "--combat-progress-every",
        type=_positive_int,
        default=10,
        help="Print combat evaluation progress every N pairs (default: 10).",
    )
    combat.add_argument(
        "--skip-combat-evaluation",
        action="store_true",
        help="Train M0 without running the real combat-engine OOF evaluation.",
    )
    return parser


def run(args=None) -> int:
    parsed = _build_parser().parse_args(args)
    if parsed.dropout >= 1:
        raise ValueError("--dropout must be smaller than 1.")
    if parsed.max_multiplier <= parsed.min_multiplier:
        raise ValueError(
            "--max-multiplier must be greater than --min-multiplier."
        )

    default_output, default_report = _default_output_paths()
    output_path = parsed.output or default_output
    report_path = parsed.report or (
        default_report
        if parsed.output is None
        else parsed.output.with_name(
            f"{parsed.output.stem}_report.json"
        )
    )

    examples = load_perfect_pairs(parsed.input)
    source_files = sorted({row.source_path for row in examples})
    print(f"Perfect pairs: {len(examples)}", flush=True)
    print(f"Source files: {len(source_files)}", flush=True)
    for source in source_files:
        print(f"  - {source}", flush=True)
    print(f"Model output: {output_path}", flush=True)
    print(f"Report output: {report_path}", flush=True)

    config = TrainingConfig(
        folds=parsed.folds,
        max_epochs=parsed.epochs,
        patience=parsed.patience,
        batch_size=parsed.batch_size,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        multiplier_loss_weight=parsed.multiplier_loss_weight,
        hidden_dims=tuple(parsed.hidden_dims),
        dropout=parsed.dropout,
        min_multiplier=parsed.min_multiplier,
        max_multiplier=parsed.max_multiplier,
        seed=parsed.seed,
    )
    artifacts = train_attack_predictor(
        examples,
        output_path=output_path,
        config=config,
        device=parsed.device,
    )

    report = dict(artifacts.report)
    report["source_files"] = source_files
    report["checkpoint_path"] = str(artifacts.checkpoint_path)

    if not parsed.skip_combat_evaluation:
        print("", flush=True)
        print(
            "Combat-aware OOF evaluation against the Phase-A oracle:",
            flush=True,
        )
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
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    metrics = report["oof_metrics"]
    print("", flush=True)
    print("Cross-validated M0 imitation metrics:", flush=True)
    print(
        "  composition L1 distance: "
        f"{metrics['composition_l1_distance']:.6f}",
        flush=True,
    )
    print(
        "  composition cosine:      "
        f"{metrics['composition_cosine_similarity']:.6f}",
        flush=True,
    )
    print(
        "  top ship accuracy:       "
        f"{metrics['top_ship_accuracy']:.2%}",
        flush=True,
    )
    print(
        "  multiplier MAE:          "
        f"{metrics['multiplier_mae']:.6f}",
        flush=True,
    )
    print(
        "  multiplier RMSE:         "
        f"{metrics['multiplier_rmse']:.6f}",
        flush=True,
    )

    combat_report = report.get("combat_evaluation")
    if combat_report:
        summary = combat_report["summary"]
        print("", flush=True)
        print("Combat-aware OOF metrics (primary):", flush=True)
        print(
            "  model reliable attacks:  "
            f"{summary['prediction_reliable_rate']:.2%}",
            flush=True,
        )
        print(
            "  oracle reliable attacks: "
            f"{summary['oracle_reliable_rate']:.2%}",
            flush=True,
        )
        print(
            "  mean oracle efficiency:  "
            f"{summary['mean_oracle_efficiency']:.2%}",
            flush=True,
        )
        print(
            "  median oracle efficiency:"
            f" {summary['median_oracle_efficiency']:.2%}",
            flush=True,
        )
        print(
            "  >= 90% oracle efficiency:"
            f" {summary['oracle_efficiency_at_least_90pct']:.2%}",
            flush=True,
        )
        print(
            "  >= 95% oracle efficiency:"
            f" {summary['oracle_efficiency_at_least_95pct']:.2%}",
            flush=True,
        )
        print(
            "  >= 99% oracle efficiency:"
            f" {summary['oracle_efficiency_at_least_99pct']:.2%}",
            flush=True,
        )
        print(
            "  mean fitness score ratio:"
            f" {summary['mean_fitness_score_ratio']:.2%}",
            flush=True,
        )

    print("", flush=True)
    print(
        f"Saved checkpoint: {artifacts.checkpoint_path}",
        flush=True,
    )
    print(f"Saved report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
