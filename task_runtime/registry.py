from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TaskSpec:
    name: str
    module: str
    description: str


_TASKS = {
    "ogia-engine-check": TaskSpec(
        name="ogia-engine-check",
        module="tasks.OGIA.task",
        description="Validate that the OGIA engine and its dependencies load correctly.",
    ),
    "ogia-battle-demo": TaskSpec(
        name="ogia-battle-demo",
        module="tasks.OGIA.battle_demo",
        description="Run a small OGame battle using the OGIA simulator.",
    ),
    "ogia-generate-random-battles": TaskSpec(
        name="ogia-generate-random-battles",
        module="tasks.OGIA.random_battles",
        description="Generate and incrementally save random OGame battles as JSONL.",
    ),
    "ogia-merge-battle-datasets": TaskSpec(
        name="ogia-merge-battle-datasets",
        module="tasks.OGIA.merge_battle_datasets",
        description="Merge every OGIA JSONL battle file in a directory into one dataset.",
    ),
    "ogia-analyze-battles": TaskSpec(
        name="ogia-analyze-battles",
        module="tasks.OGIA.analyze_battles",
        description="Analyze a merged OGIA battle dataset and generate an HTML report.",
    ),
    "ogia-generate-perfect-pairs": TaskSpec(
        name="ogia-generate-perfect-pairs",
        module="tasks.OGIA.perfect_pairs",
        description="Generate validated defense/optimized-attack pairs with the evolutionary optimizer.",
    ),
    "ogia-generate-perfect-pairs-with-ships": TaskSpec(
        name="ogia-generate-perfect-pairs-with-ships",
        module="tasks.OGIA.perfect_pairs_with_ships",
        description="Generate validated perfect pairs for mixed defenders containing both static defenses and ships.",
    ),
    "ogia-generate-perfect-pairs-ml": TaskSpec(
        name="ogia-generate-perfect-pairs-ml",
        module="tasks.OGIA.perfect_pairs_ml",
        description="Generate validated perfect pairs by refining an ML-predicted attack with the optimizer.",
    ),
    "ogia-train-attack-predictor": TaskSpec(
        name="ogia-train-attack-predictor",
        module="tasks.OGIA.train_attack_predictor",
        description="Train the Phase-B M0 attack predictor from validated perfect pairs.",
    ),
}


def list_tasks() -> Iterable[TaskSpec]:
    return tuple(sorted(_TASKS.values(), key=lambda task: task.name))


def get_task(name: str) -> TaskSpec:
    try:
        return _TASKS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TASKS)) or "none"
        raise KeyError(f"Unknown task '{name}'. Available tasks: {available}") from exc


def execute_task(name: str, task_args: Sequence[str] | None = None) -> int:
    spec = get_task(name)
    module = import_module(spec.module)

    runner = getattr(module, "run", None)
    if runner is None or not callable(runner):
        raise RuntimeError(f"Task module '{spec.module}' must expose a callable run(args) function")

    result = runner(list(task_args or []))
    return int(result or 0)
