from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from . import perfect_pairs as _base
from .defender import generate_random_defender


def _generate_mixed_defender(*args, **kwargs):
    return generate_random_defender(
        *args,
        include_ships=True,
        require_ship=True,
        require_defense=True,
        **kwargs,
    )


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return (
        Path("data")
        / "OGIA"
        / "perfect_pairs"
        / f"perfect_pairs_with_ships_{timestamp}.jsonl"
    )


def run(args: Sequence[str] | None = None) -> int:
    """Generate perfect pairs whose defender contains ships and static defense.

    The evolutionary perfect-pair pipeline is reused unchanged; only defender
    sampling and the default filename are specialized. This deliberately avoids
    using the current ML seed until a model has been retrained with ship-aware
    defender features.
    """

    original_generator = _base.generate_random_defense
    original_output_path = _base._default_output_path
    _base.generate_random_defense = _generate_mixed_defender
    _base._default_output_path = _default_output_path
    try:
        print(
            "Defender generation: mixed static defense + ships "
            "(at least one of each).",
            flush=True,
        )
        return _base.run(args)
    finally:
        _base.generate_random_defense = original_generator
        _base._default_output_path = original_output_path
