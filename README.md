# TaskExecuter

Python project for running long-lived tasks locally and on an OVH VPS.

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

Generate a random OGame battle dataset:

```powershell
python main.py run ogia-generate-random-battles
```

By default it generates 1,000 battles with defender sizes sampled logarithmically between 200 and 200,000 OGame points. The attacker target size is sampled between 1.2x and 5x the defender target size, so the largest attacker target can reach 1,000,000 points. Each completed battle is saved incrementally as JSONL under `data/OGIA/random_battles/`.

Useful arguments:

```powershell
python main.py run ogia-generate-random-battles --count 100
python main.py run ogia-generate-random-battles --count 100 --min-points 200 --max-points 200000
python main.py run ogia-generate-random-battles --count 100 --attacker-ratio-min 1.2 --attacker-ratio-max 5
python main.py run ogia-generate-random-battles --count 100 --seed 123
python main.py run ogia-generate-random-battles --count 100 --output data/OGIA/random_battles/test.jsonl
```

The generated dataset directory is ignored by Git so large simulation outputs remain on the machine that generated them.

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
        ├── random_battles.py
        ├── OgameData.py
        ├── OgameBattleSimulator.py
        └── OgameUtils.py
```

`task_runtime` contains generic execution infrastructure. Each real task lives under `tasks/` and exposes a `run(args)` entrypoint. Reusable code specific to that task can live alongside its entrypoint.
