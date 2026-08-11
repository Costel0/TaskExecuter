from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from pathlib import Path
from statistics import median
from typing import Callable, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import (
    ATTACK_SHIPS,
    DEFENSE_UNITS,
    FeatureScaler,
    PerfectPairExample,
    defense_group_key,
    examples_to_arrays,
    feature_names,
)
from .model import AttackPredictorConfig, AttackPredictorNet
from .postprocessing import (
    PredictionPostprocessConfig,
    postprocess_prediction,
)


@dataclass(frozen=True)
class TrainingConfig:
    folds: int = 5
    max_epochs: int = 500
    patience: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    multiplier_loss_weight: float = 1.0
    multiplier_underprediction_weight: float = 3.0
    multiplier_loss_beta: float = 0.10
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.05
    min_multiplier: float = 0.50
    max_multiplier: float = 6.00
    seed: int = 123

    def __post_init__(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least 2.")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("epochs, patience and batch_size must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer hyperparameters.")
        if self.multiplier_loss_weight < 0:
            raise ValueError("multiplier_loss_weight cannot be negative.")
        if self.multiplier_underprediction_weight < 1:
            raise ValueError("multiplier_underprediction_weight must be >= 1.")
        if self.multiplier_loss_beta <= 0:
            raise ValueError("multiplier_loss_beta must be positive.")
        if self.max_multiplier <= self.min_multiplier:
            raise ValueError("max_multiplier must be greater than min_multiplier.")


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: Path
    report: dict


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return resolved


def _make_grouped_folds(
    rows: Sequence[PerfectPairExample],
    folds: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    """Split by physical defense so identical model inputs never cross folds."""

    if len(rows) < 2:
        raise ValueError("At least two perfect pairs are required for training.")

    groups: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(defense_group_key(row), []).append(index)

    if len(groups) < 2:
        raise ValueError("At least two distinct defenses are required for CV.")

    fold_count = min(int(folds), len(groups))
    rng = np.random.default_rng(seed)
    grouped_indices = list(groups.values())
    rng.shuffle(grouped_indices)
    grouped_indices.sort(key=len, reverse=True)

    buckets: list[list[int]] = [[] for _ in range(fold_count)]
    bucket_sizes = [0] * fold_count
    for group in grouped_indices:
        target = min(range(fold_count), key=lambda idx: bucket_sizes[idx])
        buckets[target].extend(group)
        bucket_sizes[target] += len(group)

    return tuple(
        np.asarray(sorted(bucket), dtype=np.int64)
        for bucket in buckets
    )


def _loss(
    output: dict[str, torch.Tensor],
    target_composition: torch.Tensor,
    target_multiplier: torch.Tensor,
    *,
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probabilities = F.log_softmax(output["composition_logits"], dim=-1)
    composition_loss = -(
        target_composition * log_probabilities
    ).sum(dim=-1).mean()

    # Multiplier errors are measured in real x-units. The first M0 run showed
    # that the former normalization by the full 0.5x-6x range made a 0.1x miss
    # almost irrelevant. Underprediction is additionally more costly because
    # attacks close to the reliability frontier can collapse when slightly too
    # small, while a small overprediction is still a useful optimizer seed.
    multiplier_per_row = F.smooth_l1_loss(
        output["multiplier"],
        target_multiplier,
        beta=config.multiplier_loss_beta,
        reduction="none",
    )
    asymmetry = torch.where(
        output["multiplier"] < target_multiplier,
        torch.full_like(
            multiplier_per_row,
            config.multiplier_underprediction_weight,
        ),
        torch.ones_like(multiplier_per_row),
    )
    multiplier_loss = (multiplier_per_row * asymmetry).mean()

    total = composition_loss + config.multiplier_loss_weight * multiplier_loss
    return total, composition_loss, multiplier_loss


def _predict_arrays(
    model: AttackPredictorNet,
    features: np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(features.astype(np.float32)).to(device)
        output = model(tensor)
    return (
        output["composition"].detach().cpu().numpy().astype(np.float32),
        output["multiplier"].detach().cpu().numpy().astype(np.float32),
    )


def _metrics(
    target_composition: np.ndarray,
    predicted_composition: np.ndarray,
    target_multiplier: np.ndarray,
    predicted_multiplier: np.ndarray,
) -> dict[str, float]:
    delta = predicted_composition - target_composition
    composition_mae = float(np.mean(np.abs(delta)))
    composition_l1 = float(np.mean(np.sum(np.abs(delta), axis=1)))

    numerator = np.sum(target_composition * predicted_composition, axis=1)
    denominator = (
        np.linalg.norm(target_composition, axis=1)
        * np.linalg.norm(predicted_composition, axis=1)
    )
    cosine = float(
        np.mean(numerator / np.clip(denominator, 1e-8, None))
    )
    top_ship_accuracy = float(
        np.mean(
            np.argmax(target_composition, axis=1)
            == np.argmax(predicted_composition, axis=1)
        )
    )

    multiplier_delta = predicted_multiplier - target_multiplier
    return {
        "composition_mae": composition_mae,
        "composition_l1_distance": composition_l1,
        "composition_cosine_similarity": cosine,
        "top_ship_accuracy": top_ship_accuracy,
        "multiplier_mae": float(np.mean(np.abs(multiplier_delta))),
        "multiplier_rmse": float(
            math.sqrt(np.mean(np.square(multiplier_delta)))
        ),
        "multiplier_mean_signed_error": float(np.mean(multiplier_delta)),
        "multiplier_underprediction_rate": float(
            np.mean(multiplier_delta < 0)
        ),
    }


def _train_one_model(
    train_features: np.ndarray,
    train_composition: np.ndarray,
    train_multiplier: np.ndarray,
    *,
    model_config: AttackPredictorConfig,
    training_config: TrainingConfig,
    device: torch.device,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    exact_epochs: int | None,
    seed: int,
) -> tuple[AttackPredictorNet, int, float | None]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = AttackPredictorNet(model_config).to(device)

    with torch.no_grad():
        mean_composition = np.mean(train_composition, axis=0)
        composition_logits = np.log(np.clip(mean_composition, 1e-6, None))
        composition_logits -= float(np.mean(composition_logits))
        model.composition_head.weight.zero_()
        model.composition_head.bias.copy_(
            torch.as_tensor(
                composition_logits,
                dtype=torch.float32,
                device=device,
            )
        )

        multiplier_range = (
            model_config.max_multiplier - model_config.min_multiplier
        )
        mean_fraction = (
            float(np.mean(train_multiplier)) - model_config.min_multiplier
        ) / multiplier_range
        mean_fraction = float(np.clip(mean_fraction, 1e-4, 1.0 - 1e-4))
        multiplier_bias = math.log(mean_fraction / (1.0 - mean_fraction))
        model.multiplier_head.weight.zero_()
        model.multiplier_head.bias.fill_(multiplier_bias)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    dataset = TensorDataset(
        torch.from_numpy(train_features.astype(np.float32)),
        torch.from_numpy(train_composition.astype(np.float32)),
        torch.from_numpy(train_multiplier.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(training_config.batch_size, len(dataset)),
        shuffle=True,
        generator=generator,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    epochs_to_run = exact_epochs or training_config.max_epochs

    for epoch in range(1, epochs_to_run + 1):
        model.train()
        for batch_features, batch_composition, batch_multiplier in loader:
            batch_features = batch_features.to(device)
            batch_composition = batch_composition.to(device)
            batch_multiplier = batch_multiplier.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(batch_features)
            total_loss, _, _ = _loss(
                output,
                batch_composition,
                batch_multiplier,
                config=training_config,
            )
            total_loss.backward()
            optimizer.step()

        if validation is None:
            best_epoch = epoch
            continue

        validation_features, validation_composition, validation_multiplier = validation
        model.eval()
        with torch.inference_mode():
            output = model(torch.from_numpy(validation_features).to(device))
            validation_loss, _, _ = _loss(
                output,
                torch.from_numpy(validation_composition).to(device),
                torch.from_numpy(validation_multiplier).to(device),
                config=training_config,
            )
            current_loss = float(validation_loss.detach().cpu().item())

        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (
        model,
        best_epoch,
        best_loss if validation is not None else None,
    )


def train_attack_predictor(
    examples: Sequence[PerfectPairExample],
    *,
    output_path: str | Path,
    config: TrainingConfig = TrainingConfig(),
    postprocess_config: PredictionPostprocessConfig = PredictionPostprocessConfig(),
    device: str = "auto",
    progress: Callable[[str], None] | None = print,
) -> TrainingArtifacts:
    rows = tuple(examples)
    features, target_composition, target_multiplier = examples_to_arrays(rows)

    if np.any(target_multiplier < config.min_multiplier) or np.any(
        target_multiplier > config.max_multiplier
    ):
        minimum = float(target_multiplier.min())
        maximum = float(target_multiplier.max())
        raise ValueError(
            "Training target multiplier falls outside configured model bounds: "
            f"data=[{minimum:.4f}, {maximum:.4f}] "
            f"model=[{config.min_multiplier:.4f}, {config.max_multiplier:.4f}]"
        )

    resolved_device = _resolve_device(device)
    folds = _make_grouped_folds(rows, config.folds, config.seed)
    oof_composition = np.zeros_like(target_composition)
    oof_multiplier = np.zeros_like(target_multiplier)
    fold_reports: list[dict] = []
    best_epochs: list[int] = []

    model_config = AttackPredictorConfig(
        input_dim=features.shape[1],
        output_ships=target_composition.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
        min_multiplier=config.min_multiplier,
        max_multiplier=config.max_multiplier,
    )

    all_indices = np.arange(len(rows), dtype=np.int64)
    for fold_index, validation_indices in enumerate(folds, start=1):
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[validation_indices] = False
        train_indices = all_indices[train_mask]
        if len(train_indices) == 0:
            raise ValueError("Cross-validation produced an empty training fold.")

        train_groups = {
            defense_group_key(rows[int(index)]) for index in train_indices
        }
        validation_groups = {
            defense_group_key(rows[int(index)]) for index in validation_indices
        }
        if train_groups & validation_groups:
            raise RuntimeError("Grouped CV leaked a defense across train/validation.")

        scaler = FeatureScaler.fit(features[train_indices])
        scaled_train = scaler.transform(features[train_indices])
        scaled_validation = scaler.transform(features[validation_indices])

        if progress:
            progress(
                f"[fold {fold_index}/{len(folds)}] "
                f"train={len(train_indices)} ({len(train_groups)} defenses) "
                f"validation={len(validation_indices)} "
                f"({len(validation_groups)} defenses)"
            )

        model, best_epoch, best_loss = _train_one_model(
            scaled_train,
            target_composition[train_indices],
            target_multiplier[train_indices],
            model_config=model_config,
            training_config=config,
            device=resolved_device,
            validation=(
                scaled_validation,
                target_composition[validation_indices],
                target_multiplier[validation_indices],
            ),
            exact_epochs=None,
            seed=config.seed + fold_index,
        )
        predicted_composition, predicted_multiplier = _predict_arrays(
            model,
            scaled_validation,
            device=resolved_device,
        )
        oof_composition[validation_indices] = predicted_composition
        oof_multiplier[validation_indices] = predicted_multiplier
        fold_metrics = _metrics(
            target_composition[validation_indices],
            predicted_composition,
            target_multiplier[validation_indices],
            predicted_multiplier,
        )
        best_epochs.append(best_epoch)
        fold_reports.append(
            {
                "fold": fold_index,
                "train_pairs": int(len(train_indices)),
                "train_defenses": int(len(train_groups)),
                "validation_pairs": int(len(validation_indices)),
                "validation_defenses": int(len(validation_groups)),
                "best_epoch": int(best_epoch),
                "best_validation_loss": float(best_loss),
                "metrics": fold_metrics,
            }
        )
        if progress:
            progress(
                f"[fold {fold_index}/{len(folds)}] "
                f"epoch={best_epoch} "
                f"composition_l1={fold_metrics['composition_l1_distance']:.4f} "
                f"multiplier_mae={fold_metrics['multiplier_mae']:.4f} "
                f"under={fold_metrics['multiplier_underprediction_rate']:.1%}"
            )

    overall_metrics = _metrics(
        target_composition,
        oof_composition,
        target_multiplier,
        oof_multiplier,
    )
    final_epochs = max(1, int(round(median(best_epochs))))
    final_scaler = FeatureScaler.fit(features)
    scaled_all = final_scaler.transform(features)

    if progress:
        progress(
            f"[final] training on all {len(rows)} pairs "
            f"for {final_epochs} epochs"
        )

    final_model, _, _ = _train_one_model(
        scaled_all,
        target_composition,
        target_multiplier,
        model_config=model_config,
        training_config=config,
        device=resolved_device,
        validation=None,
        exact_epochs=final_epochs,
        seed=config.seed + 10_000,
    )
    final_model = final_model.to("cpu")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 2,
        "model_config": model_config.to_dict(),
        "state_dict": final_model.state_dict(),
        "feature_scaler": final_scaler.to_dict(),
        "feature_spec": {
            "defense_units": list(DEFENSE_UNITS),
            "attack_ships": list(ATTACK_SHIPS),
            "feature_names": list(feature_names()),
        },
        "postprocess_config": postprocess_config.to_dict(),
        "training": {
            "pair_count": len(rows),
            "defense_group_count": len(
                {defense_group_key(row) for row in rows}
            ),
            "seed": config.seed,
            "final_epochs": final_epochs,
            "cv_metrics": overall_metrics,
        },
    }
    torch.save(checkpoint, output_path)

    oof_rows: list[dict] = []
    for index, row in enumerate(rows):
        raw_weights = {
            ship: float(oof_composition[index, ship_index])
            for ship_index, ship in enumerate(ATTACK_SHIPS)
        }
        raw_multiplier = float(oof_multiplier[index])
        processed_weights, processed_multiplier = postprocess_prediction(
            raw_weights,
            raw_multiplier,
            attack_ships=ATTACK_SHIPS,
            config=postprocess_config,
            max_multiplier=config.max_multiplier,
        )
        oof_rows.append(
            {
                "pair_id": row.pair_id,
                "target": {
                    "ship_weights": {
                        ship: float(target_composition[index, ship_index])
                        for ship_index, ship in enumerate(ATTACK_SHIPS)
                    },
                    "points_multiplier": float(target_multiplier[index]),
                },
                "prediction": {
                    "raw_ship_weights": raw_weights,
                    "raw_points_multiplier": raw_multiplier,
                    "ship_weights": processed_weights,
                    "points_multiplier": processed_multiplier,
                    "active_ship_types": int(
                        sum(value > 0 for value in processed_weights.values())
                    ),
                },
            }
        )

    report = {
        "schema_version": 2,
        "pair_count": len(rows),
        "defense_group_count": len({defense_group_key(row) for row in rows}),
        "device": str(resolved_device),
        "training_config": {
            "folds": config.folds,
            "grouped_cv_by_defense": True,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "multiplier_loss_weight": config.multiplier_loss_weight,
            "multiplier_underprediction_weight": (
                config.multiplier_underprediction_weight
            ),
            "multiplier_loss_beta": config.multiplier_loss_beta,
            "hidden_dims": list(config.hidden_dims),
            "dropout": config.dropout,
            "min_multiplier": config.min_multiplier,
            "max_multiplier": config.max_multiplier,
            "seed": config.seed,
        },
        "postprocess_config": postprocess_config.to_dict(),
        "model_config": model_config.to_dict(),
        "feature_spec": checkpoint["feature_spec"],
        "folds": fold_reports,
        "oof_metrics": overall_metrics,
        "final_training_epochs": final_epochs,
        "oof_predictions": oof_rows,
    }
    return TrainingArtifacts(
        checkpoint_path=output_path,
        report=report,
    )
