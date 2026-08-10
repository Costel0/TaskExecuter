from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from ..optimizer import AttackGenome
from .data import FeatureScaler, defender_to_features
from .model import AttackPredictorConfig, AttackPredictorNet


@dataclass(frozen=True)
class AttackPrediction:
    ship_weights: Mapping[str, float]
    points_multiplier: float

    def to_genome(self) -> AttackGenome:
        return AttackGenome(
            ship_weights=dict(self.ship_weights),
            points_multiplier=float(self.points_multiplier),
        )


class AttackPredictor:
    """Loadable Phase-B model that maps a defense directly to an AttackGenome."""

    def __init__(
        self,
        *,
        model: AttackPredictorNet,
        scaler: FeatureScaler,
        defense_units: tuple[str, ...],
        attack_ships: tuple[str, ...],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.scaler = scaler
        self.defense_units = defense_units
        self.attack_ships = attack_ships
        self.device = device

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "AttackPredictor":
        resolved_device = torch.device(device)
        checkpoint = torch.load(
            Path(path),
            map_location=resolved_device,
            weights_only=True,
        )
        if int(checkpoint.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported attack-predictor checkpoint schema.")

        config = AttackPredictorConfig.from_dict(checkpoint["model_config"])
        model = AttackPredictorNet(config)
        model.load_state_dict(checkpoint["state_dict"])

        feature_spec = checkpoint["feature_spec"]
        defense_units = tuple(
            str(value) for value in feature_spec["defense_units"]
        )
        attack_ships = tuple(
            str(value) for value in feature_spec["attack_ships"]
        )
        if config.output_ships != len(attack_ships):
            raise ValueError(
                "Checkpoint output size does not match attack ship metadata."
            )

        return cls(
            model=model,
            scaler=FeatureScaler.from_dict(checkpoint["feature_scaler"]),
            defense_units=defense_units,
            attack_ships=attack_ships,
            device=resolved_device,
        )

    def predict(self, defender: Mapping[str, int]) -> AttackPrediction:
        features = defender_to_features(
            defender,
            defense_units=self.defense_units,
        )
        scaled = self.scaler.transform(features[None, :])
        tensor = torch.from_numpy(
            np.asarray(scaled, dtype=np.float32)
        ).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)

        composition = output["composition"][0].detach().cpu().numpy()
        multiplier = float(
            output["multiplier"][0].detach().cpu().item()
        )
        weights = {
            ship_name: float(composition[index])
            for index, ship_name in enumerate(self.attack_ships)
        }
        return AttackPrediction(
            ship_weights=weights,
            points_multiplier=multiplier,
        )

    def predict_genome(self, defender: Mapping[str, int]) -> AttackGenome:
        return self.predict(defender).to_genome()
