from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class AttackPredictorConfig:
    input_dim: int
    output_ships: int
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.05
    min_multiplier: float = 0.50
    max_multiplier: float = 6.00

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_ships <= 0:
            raise ValueError("input_dim and output_ships must be positive.")
        if not self.hidden_dims or any(size <= 0 for size in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive sizes.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.min_multiplier <= 0 or self.max_multiplier <= self.min_multiplier:
            raise ValueError("Invalid multiplier bounds.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hidden_dims"] = list(self.hidden_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AttackPredictorConfig":
        return cls(
            input_dim=int(payload["input_dim"]),
            output_ships=int(payload["output_ships"]),
            hidden_dims=tuple(
                int(value)
                for value in payload.get("hidden_dims", (128, 64))
            ),
            dropout=float(payload.get("dropout", 0.05)),
            min_multiplier=float(payload.get("min_multiplier", 0.50)),
            max_multiplier=float(payload.get("max_multiplier", 6.00)),
        )


class AttackPredictorNet(nn.Module):
    def __init__(self, config: AttackPredictorConfig):
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        previous = config.input_dim
        for hidden in config.hidden_dims:
            layers.append(nn.Linear(previous, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous = hidden

        self.backbone = nn.Sequential(*layers)
        self.composition_head = nn.Linear(previous, config.output_ships)
        self.multiplier_head = nn.Linear(previous, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        composition_logits = self.composition_head(hidden)
        composition = torch.softmax(composition_logits, dim=-1)
        raw_multiplier = self.multiplier_head(hidden).squeeze(-1)
        multiplier_fraction = torch.sigmoid(raw_multiplier)
        multiplier = self.config.min_multiplier + multiplier_fraction * (
            self.config.max_multiplier - self.config.min_multiplier
        )
        return {
            "composition_logits": composition_logits,
            "composition": composition,
            "multiplier": multiplier,
        }
