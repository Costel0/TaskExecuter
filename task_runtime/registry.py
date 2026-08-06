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
