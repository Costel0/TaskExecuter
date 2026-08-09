from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


class AttrDict(dict):
    """Dictionary that also exposes JSON keys through attribute access.

    Values are converted recursively, so nested objects can be read either as
    normal dictionaries or with dot notation::

        battle["result"]["winner"]
        battle.result.winner
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = _to_attr_value(value)

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class BattleRecord(AttrDict):
    """One stored OGIA battle, preserving every field from the JSONL record."""


def _to_attr_value(value: Any) -> Any:
    if isinstance(value, AttrDict):
        return value
    if isinstance(value, dict):
        return AttrDict(
            (key, _to_attr_value(nested_value))
            for key, nested_value in value.items()
        )
    if isinstance(value, list):
        return [_to_attr_value(item) for item in value]
    return value


def _record_from_dict(data: dict[str, Any]) -> BattleRecord:
    return BattleRecord(
        (key, _to_attr_value(value))
        for key, value in data.items()
    )


def iter_battles(path: str | Path) -> Iterator[BattleRecord]:
    """Yield battles from an OGIA JSONL dataset one at a time.

    This is the preferred API for very large datasets because only one battle
    needs to be held in memory at once.
    """
    dataset_path = Path(path).expanduser()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Battle dataset not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, raw_line in enumerate(dataset_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {dataset_path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Expected a JSON object in {dataset_path} at line "
                    f"{line_number}, got {type(parsed).__name__}."
                )

            yield _record_from_dict(parsed)


def load_battles(
    path: str | Path,
    *,
    limit: int | None = None,
) -> list[BattleRecord]:
    """Load an OGIA JSONL dataset into a list of battle records.

    Parameters
    ----------
    path:
        Path to the ``.jsonl`` file produced by
        ``ogia-generate-random-battles``.

    limit:
        Optional maximum number of battles to load. Useful when inspecting a
        very large dataset interactively.

    Returns
    -------
    list[BattleRecord]
        A list containing every loaded battle. Every JSON field is preserved,
        and nested JSON objects support both dictionary and attribute access.
    """
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer or None")

    battles: list[BattleRecord] = []
    for battle in iter_battles(path):
        battles.append(battle)
        if limit is not None and len(battles) >= limit:
            break

    return battles


__all__ = [
    "AttrDict",
    "BattleRecord",
    "iter_battles",
    "load_battles",
]
