# TaskExecuter

Python project for running long-lived tasks locally and, eventually, on a Hetzner VM.

## Local setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Task commands

List the registered tasks:

```powershell
python main.py list
```

Run a task:

```powershell
python main.py run <task-name>
```

Every execution writes its terminal output to `logs/<timestamp>_<task-name>.log` as well as showing it in the terminal.

## Current tasks

Diagnostic for the OGIA engine:

```powershell
python main.py run ogia-engine-check
```

Small OGame battle using the OGIA simulator:

```powershell
python main.py run ogia-battle-demo
```

The demo accepts an optional seed:

```powershell
python main.py run ogia-battle-demo --seed 123
```

## Adding a new task

1. Create a Python module under `tasks/` (or inside the relevant task package).
2. Expose a callable entrypoint with this shape:

```python
def run(args=None) -> int:
    # task code
    return 0
```

3. Register the task in `task_runtime/registry.py` by adding a `TaskSpec` to `_TASKS`.
4. Add any new third-party Python packages to `requirements.txt`.
5. Pull/install dependencies on the machine that will execute the task.
6. Verify registration:

```powershell
python main.py list
```

7. Run it:

```powershell
python main.py run <task-name>
```

## Project structure

```text
TaskExecuter/
├── main.py
├── requirements.txt
├── task_runtime/
│   ├── __init__.py
│   └── registry.py
└── tasks/
    └── OGIA/
        ├── __init__.py
        ├── task.py
        ├── battle_demo.py
        ├── OgameData.py
        ├── OgameBattleSimulator.py
        └── OgameUtils.py
```

`task_runtime` contains generic execution infrastructure. Each real task lives under `tasks/` and exposes a `run(args)` entrypoint. Reusable code specific to that task can live alongside its entrypoint.
