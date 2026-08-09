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

## Evolutionary perfect-pair generation

After generating and analyzing the random battle dataset, optimized defense/attack pairs can be produced with:

```powershell
python main.py run ogia-generate-perfect-pairs
```

The task expects optimizer priors at:

```text
data/OGIA/Analisis/merged_battles_1_analysis.json
```

and writes accepted pairs incrementally as JSONL under:

```text
data/OGIA/perfect_pairs/
```

Weapons, shielding and armour are fixed to level 15 for attacker and defender. Generation zero is 85% analysis-guided/diversified and 15% broad exploration. The evolutionary fitness first seeks a reliable attack and then minimizes real attacker/defender points and attacker losses. All individuals use common combat seeds during evolution. Final candidates are re-tested on independent common seeds and a pair is saved only if it reaches the validation threshold.

Production defaults:

- 64 individuals;
- 50 generations;
- 12 common training simulations per genome;
- 90% search reliability threshold;
- 2 optimizer restarts per defense;
- 20 diverse final candidates checked;
- 128 independent validation battles;
- 95% minimum validation win rate;
- attacker multiplier search range 0.5x–6.0x.

Start with a small smoke run before a long VM job:

```powershell
python main.py run ogia-generate-perfect-pairs --count 1 --population-size 16 --generations 5 --training-simulations 4 --validation-simulations 32 --validation-win-rate 0.90 --restarts 1
```

Generate a custom number of production pairs:

```powershell
python main.py run ogia-generate-perfect-pairs --count 100 --seed 123
```

The generated pair dataset directory is ignored by Git.

## Typed battle datasets

`tasks/OGIA/battle_dataset.py` separates the JSONL storage format from the Python objects used by the rest of the project.

Loading a file produces interpreted `BattleRecord` objects. Technologies, combat configuration and battle results are restored as the same domain classes used by the simulator:

```python
from tasks.OGIA.battle_dataset import BattleDataset

battles = BattleDataset.from_file(
    "data/OGIA/random_battles/random_battles_20260808_173848.jsonl"
)

for battle in battles:
    print(battle.inputs.attacker)
    print(battle.inputs.attacker_tech.weapons)
    print(battle.inputs.combat_config.max_rounds)
    print(battle.result.winner)
    print(battle.result.attacker_destroyed)
    print(battle.result.minimum_profit_total)
```

Important interpreted types include:

- `BattleRecord`
- `GenerationMetadata`
- `BattleInputs`
- `TechLevels`
- `CombatConfig`
- `CompositionDetails`
- `FleetCompositionDetails`
- `DefenseCompositionDetails`
- `BattleResultWithProfit`

`generated_at_utc` is restored as a Python `datetime`.

If a plain list is preferred:

```python
from tasks.OGIA.battle_dataset import load_battles

battles = load_battles("data/OGIA/random_battles/battles.jsonl")
```

For very large files, iterate without loading the full dataset into RAM:

```python
from tasks.OGIA.battle_dataset import iter_battles

for battle in iter_battles("data/OGIA/random_battles/battles.jsonl"):
    print(battle.result.winner)
```

## Combining datasets from several runs or machines

All `.jsonl` files in a directory can be interpreted and combined directly into one Python list:

```python
from tasks.OGIA.battle_dataset import load_battles_from_directory

battles = load_battles_from_directory(
    "data/OGIA/random_battles"
)
```

Or use the console task to consolidate all `.jsonl` files in a directory into one validated JSONL file:

```powershell
python main.py run ogia-merge-battle-datasets data/OGIA/random_battles
```

The default output is:

```text
data/OGIA/random_battles/merged_battles.jsonl
```

A custom output path can be supplied:

```powershell
python main.py run ogia-merge-battle-datasets data/OGIA/random_battles --output data/OGIA/all_battles.jsonl
```

Use `--recursive` to also include JSONL files in subdirectories:

```powershell
python main.py run ogia-merge-battle-datasets data/OGIA --recursive
```

The selected output file is excluded from its own input set, so the merge command can safely be re-run with the same output path.

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
        ├── battle_dataset.py
        ├── merge_battle_datasets.py
        ├── analyze_battles.py
        ├── perfect_pairs.py
        ├── optimizer/
        │   ├── __init__.py
        │   ├── decoder.py
        │   ├── engine.py
        │   ├── fitness.py
        │   ├── initializer.py
        │   ├── interfaces.py
        │   ├── models.py
        │   ├── operators.py
        │   └── README.md
        ├── OgameData.py
        ├── OgameBattleSimulator.py
        └── OgameUtils.py
```

`task_runtime` contains generic execution infrastructure. Each real task lives under `tasks/` and exposes a `run(args)` entrypoint. Reusable OGIA data models and dataset utilities live alongside the OGIA task modules.
