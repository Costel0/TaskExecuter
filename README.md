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

## Current diagnostic task

The first registered entry is a diagnostic for the OGIA engine:

```powershell
python main.py run ogia-engine-check
```

This validates the OGame reference data and checks that `OgameData.py`, `OgameBattleSimulator.py`, `OgameUtils.py` and their Python dependencies can be imported correctly.

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
        ├── OgameData.py
        ├── OgameBattleSimulator.py
        └── OgameUtils.py
```

`task_runtime` contains generic execution infrastructure. Each real task lives under `tasks/` and exposes a `run(args)` entrypoint. Reusable code specific to that task can live alongside its entrypoint.
