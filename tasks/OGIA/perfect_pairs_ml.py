from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from .attack_predictor import AttackPredictor
from .optimizer import (
    DefenseFitnessConfig,
    EvolutionConfig,
    HybridAnalysisPopulationInitializer,
    HybridInitializerConfig,
    MLSeededInitializerConfig,
    MLSeededPopulationInitializer,
)
from .perfect_pairs import _generate_one_pair


DEFAULT_COUNT = 500
DEFAULT_MIN_POINTS = 200.0
DEFAULT_MAX_POINTS = 200_000.0
DEFAULT_POPULATION_SIZE = 48
DEFAULT_GENERATIONS = 30
DEFAULT_TRAINING_SIMULATIONS = 12
DEFAULT_SEARCH_WIN_RATE = 0.90
DEFAULT_VALIDATION_SIMULATIONS = 128
DEFAULT_VALIDATION_WIN_RATE = 0.95
DEFAULT_VALIDATION_CANDIDATES = 20
DEFAULT_RESTARTS = 2
DEFAULT_PROGRESS_EVERY = 1
DEFAULT_MIN_MULTIPLIER = 0.50
DEFAULT_MAX_MULTIPLIER = 6.00
DEFAULT_ANALYSIS = Path("data/OGIA/Analisis/merged_battles_1_analysis.json")
DEFAULT_STAGNATION_PATIENCE = 8
DEFAULT_MIN_GENERATIONS_BEFORE_STOPPING = 12
DEFAULT_STAGNATION_SCORE_TOLERANCE = 1e-4
DEFAULT_WEIGHT_MUTATION_SIGMA = 0.04
DEFAULT_MULTIPLIER_MUTATION_SIGMA = 0.04


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        Path("data")
        / "OGIA"
        / "perfect_pairs"
        / f"perfect_pairs_ml_{timestamp}.jsonl"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ogia-generate-perfect-pairs-ml",
        description=(
            "Generate validated perfect pairs by refining an attack-predictor "
            "checkpoint with the evolutionary optimizer."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Attack predictor .pt checkpoint used to seed every optimizer restart.",
    )
    parser.add_argument("--count", type=_positive_int, default=DEFAULT_COUNT)
    parser.add_argument("--min-points", type=_positive_float, default=DEFAULT_MIN_POINTS)
    parser.add_argument("--max-points", type=_positive_float, default=DEFAULT_MAX_POINTS)
    parser.add_argument("--population-size", type=_positive_int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--generations", type=_non_negative_int, default=DEFAULT_GENERATIONS)
    parser.add_argument(
        "--training-simulations",
        type=_positive_int,
        default=DEFAULT_TRAINING_SIMULATIONS,
    )
    parser.add_argument(
        "--search-win-rate",
        type=_probability,
        default=DEFAULT_SEARCH_WIN_RATE,
    )
    parser.add_argument(
        "--validation-simulations",
        type=_positive_int,
        default=DEFAULT_VALIDATION_SIMULATIONS,
    )
    parser.add_argument(
        "--validation-win-rate",
        type=_probability,
        default=DEFAULT_VALIDATION_WIN_RATE,
    )
    parser.add_argument(
        "--validation-candidates",
        type=_positive_int,
        default=DEFAULT_VALIDATION_CANDIDATES,
    )
    parser.add_argument("--restarts", type=_positive_int, default=DEFAULT_RESTARTS)
    parser.add_argument(
        "--min-multiplier",
        type=_positive_float,
        default=DEFAULT_MIN_MULTIPLIER,
    )
    parser.add_argument(
        "--max-multiplier",
        type=_positive_float,
        default=DEFAULT_MAX_MULTIPLIER,
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        default=DEFAULT_ANALYSIS,
        help="Random-battle analysis JSON used only for the small ML/analysis blend.",
    )
    parser.add_argument(
        "--stagnation-patience",
        type=_positive_int,
        default=DEFAULT_STAGNATION_PATIENCE,
        help="Stop a restart after this many generations without a meaningful improvement.",
    )
    parser.add_argument(
        "--min-generations-before-stopping",
        type=_non_negative_int,
        default=DEFAULT_MIN_GENERATIONS_BEFORE_STOPPING,
    )
    parser.add_argument(
        "--stagnation-score-tolerance",
        type=_non_negative_float,
        default=DEFAULT_STAGNATION_SCORE_TOLERANCE,
    )
    parser.add_argument(
        "--weight-mutation-sigma",
        type=_non_negative_float,
        default=DEFAULT_WEIGHT_MUTATION_SIGMA,
    )
    parser.add_argument(
        "--multiplier-mutation-sigma",
        type=_non_negative_float,
        default=DEFAULT_MULTIPLIER_MUTATION_SIGMA,
    )
    parser.add_argument("--progress-every", type=_positive_int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path. Defaults to data/OGIA/perfect_pairs/perfect_pairs_ml_<timestamp>.jsonl.",
    )
    return parser


def run(args: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    options = parser.parse_args(list(args or []))

    min_points = float(options.min_points)
    max_points = float(options.max_points)
    if max_points < min_points:
        parser.error("--max-points must be greater than or equal to --min-points")
    if options.population_size < 4:
        parser.error("--population-size must be at least 4")
    if options.max_multiplier < options.min_multiplier:
        parser.error("--max-multiplier must be >= --min-multiplier")
    if options.validation_win_rate < options.search_win_rate:
        parser.error("--validation-win-rate must be >= --search-win-rate")
    if options.min_generations_before_stopping > options.generations:
        parser.error("--min-generations-before-stopping cannot exceed --generations")

    model_path = options.model.expanduser()
    if not model_path.is_file():
        parser.error(f"Attack predictor checkpoint not found: {model_path}")

    analysis_path = options.analysis.expanduser()
    if not analysis_path.is_file():
        parser.error(
            f"Analysis JSON not found: {analysis_path}. Run ogia-analyze-battles first."
        )

    output_path = (options.output or _default_output_path()).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictor = AttackPredictor.load(model_path, device="cpu")

    analysis_initializer_config = HybridInitializerConfig()
    analysis_initializer = HybridAnalysisPopulationInitializer.from_json(
        analysis_path,
        initializer_config=analysis_initializer_config,
    )
    ml_initializer_config = MLSeededInitializerConfig()
    initializer = MLSeededPopulationInitializer(
        predictor.predict_genome,
        analysis_initializer=analysis_initializer,
        initializer_config=ml_initializer_config,
    )

    elite_count = max(1, min(6, options.population_size // 12 or 1))
    tournament_size = min(4, options.population_size)
    evolution_config = EvolutionConfig(
        population_size=int(options.population_size),
        generations=int(options.generations),
        elite_count=int(elite_count),
        tournament_size=int(tournament_size),
        per_gene_mutation_rate=0.18,
        multiplier_mutation_rate=0.45,
        weight_mutation_sigma=float(options.weight_mutation_sigma),
        multiplier_mutation_sigma=float(options.multiplier_mutation_sigma),
        min_points_multiplier=float(options.min_multiplier),
        max_points_multiplier=float(options.max_multiplier),
        stagnation_patience=int(options.stagnation_patience),
        min_generations_before_stopping=int(options.min_generations_before_stopping),
        stagnation_score_tolerance=float(options.stagnation_score_tolerance),
    )
    fitness_config = DefenseFitnessConfig(
        simulations_per_genome=int(options.training_simulations),
        reliable_win_rate=float(options.search_win_rate),
    )

    rng = np.random.default_rng(options.seed)
    saved = 0
    failed = 0
    attempts = 0
    max_attempts = max(options.count * 5, options.count + 20)

    print("OGIA ML-seeded perfect-pair generator")
    print(f"Target accepted pairs: {options.count:,}")
    print(f"Defense point range: {min_points:g} - {max_points:g}")
    print(f"ML checkpoint: {model_path}")
    print(
        "Generation zero: exact M0 + "
        f"{ml_initializer_config.local_fraction:.0%} local + "
        f"{ml_initializer_config.broad_fraction:.0%} broad + "
        f"{ml_initializer_config.analysis_blend_fraction:.0%} ML/analysis blend"
    )
    print(f"Population: {evolution_config.population_size}")
    print(f"Max generations: {evolution_config.generations}")
    print(
        "Early stop: after "
        f"{evolution_config.stagnation_patience} stagnant generations, "
        f"not before generation {evolution_config.min_generations_before_stopping}"
    )
    print(f"Training simulations/genome: {fitness_config.simulations_per_genome}")
    print(f"Search reliability threshold: {fitness_config.reliable_win_rate:.1%}")
    print(
        f"Validation: {options.validation_simulations} battles at "
        f"{options.validation_win_rate:.1%}"
    )
    print(f"ML-seeded restarts/defense: {options.restarts}")
    print(f"Analysis priors: {analysis_path}")
    print(f"Output: {output_path}")
    print(f"Seed: {options.seed if options.seed is not None else 'random'}")

    try:
        with output_path.open("a", encoding="utf-8") as output_file:
            while saved < options.count:
                attempts += 1
                if attempts > max_attempts:
                    raise RuntimeError(
                        "Too many optimizer attempts without enough validated pairs: "
                        f"{saved} saved, {failed} failed."
                    )

                try:
                    record = _generate_one_pair(
                        rng,
                        run_seed=options.seed,
                        sequence_index=saved + 1,
                        attempt_index=attempts,
                        min_points=min_points,
                        max_points=max_points,
                        initializer=initializer,
                        initializer_config=ml_initializer_config,
                        evolution_config=evolution_config,
                        fitness_config=fitness_config,
                        validation_simulations=int(options.validation_simulations),
                        validation_win_rate=float(options.validation_win_rate),
                        validation_candidates=int(options.validation_candidates),
                        restarts=int(options.restarts),
                        analysis_path=analysis_path,
                    )
                except Exception as exc:
                    failed += 1
                    print(
                        f"[warning] optimizer attempt {attempts} failed with "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue

                if record is None:
                    failed += 1
                    print(
                        f"[warning] attempt {attempts}: no candidate reached "
                        f"{options.validation_win_rate:.1%} validation win rate.",
                        flush=True,
                    )
                    continue

                defender = record["inputs"]["defender"]
                ml_prediction = predictor.predict(defender)
                record["generation"].update(
                    {
                        "search_mode": "ml_seeded_refinement",
                        "seeded_model_path": str(model_path),
                        "seeded_model_prediction": {
                            "ship_weights": dict(ml_prediction.ship_weights),
                            "points_multiplier": float(ml_prediction.points_multiplier),
                        },
                    }
                )
                record["optimizer"]["ml_seeded"] = True
                record["optimizer"]["analysis_initializer_config"] = asdict(
                    analysis_initializer_config
                )

                output_file.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                output_file.flush()
                saved += 1

                if (
                    saved == 1
                    or saved % options.progress_every == 0
                    or saved == options.count
                ):
                    os.fsync(output_file.fileno())
                    validation_metrics = record["validation"]["selected_metrics"]
                    selected_history = record["optimizer"]["selected_history"]
                    executed_generations = (
                        int(selected_history[-1]["generation"])
                        if selected_history
                        else 0
                    )
                    print(
                        f"Saved {saved:,}/{options.count:,} · "
                        f"validation win rate={validation_metrics['win_rate']:.1%} · "
                        f"multiplier={validation_metrics['actual_multiplier']:.3f}x · "
                        f"selected generations={executed_generations} · "
                        f"failed defenses={failed:,}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print(
            f"\nStopped by user. {saved:,} validated pairs were saved to {output_path}.",
            flush=True,
        )
        return 130

    print(
        f"Completed. {saved:,} validated ML-seeded pairs saved to {output_path}. "
        f"Failed optimizer attempts: {failed:,}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
