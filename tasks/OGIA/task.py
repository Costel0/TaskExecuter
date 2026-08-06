from __future__ import annotations

from typing import Sequence


def run(args: Sequence[str] | None = None) -> int:
    """Small diagnostic entrypoint proving the OGIA modules are VM-ready."""
    del args

    from . import OgameBattleSimulator  # noqa: F401
    from . import OgameData
    from . import OgameUtils  # noqa: F401

    OgameData.validate_reference_data()

    print("OGIA engine loaded successfully")
    print(f"Units loaded: {len(OgameData.UNIT_SPECS)}")
    print(f"Rapid-fire relations loaded: {len(OgameData.RAPID_FIRE)}")
    return 0
