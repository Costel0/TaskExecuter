# OGIA Phase B — Attack Predictor M0

This package trains a supervised predictor from the validated JSONL pairs produced by `ogia-generate-perfect-pairs` and evaluates out-of-fold predictions with the real OGame combat engine.

## Representation

The defender is encoded as defensive point shares for all defense types plus `log1p(total_defense_points)`. The model predicts the optimizer genome: attack-ship point shares and the requested attacker/defender point multiplier.

M0 is a small PyTorch MLP with a shared 128/64 backbone, a softmax composition head, and a bounded multiplier head.

## Data hygiene and cross-validation

Training automatically removes exact repeated optimizer pairs. Repeated appearances of the same physical defense are also assigned to the same cross-validation fold, even if their targets differ. This prevents the same model input from appearing in both train and validation.

The command prints:

- raw JSONL rows;
- exact duplicates removed;
- final training-pair count;
- number of distinct defenses.

## Multiplier loss

The multiplier loss is measured directly in attacker/defender multiplier units. By default, predicting below the optimizer target is weighted 3x more heavily than predicting above it. This reflects the fact that attacks close to the reliability frontier can fail sharply when slightly undersized.

Defaults:

- Smooth-L1 beta: `0.10x`;
- underprediction weight: `3.0`;
- multiplier loss weight: `1.0`.

## Sparse deployment output

The neural softmax is dense and never produces exact zeros. Passing that output directly to the existing fleet decoder made tiny probability noise look like deliberately selected ship types and could force one unit of nearly every ship into small attacks.

Before combat evaluation or optimizer use, predictions are therefore postprocessed:

- ship shares below `1%` are dropped;
- at most the strongest 5 ship types are retained;
- remaining shares are renormalized;
- a small `+0.03x` multiplier safety margin is added.

The report preserves both the raw neural output and the postprocessed deployable output.

## Cross-validation and combat-aware evaluation

M0 uses 5-fold grouped cross-validation by default. Every defense is predicted by a model that has never trained on that defense.

After training, the task automatically re-simulates each postprocessed OOF attack and its Phase-A oracle attack with exactly the same fresh combat seeds.

The primary task metric is `oracle_efficiency`:

```text
oracle actual attacker multiplier / model actual attacker multiplier
```

It is positive only when the model attack reaches the configured reliability threshold. Values mean:

- `100%`: model reached oracle attacker-point efficiency;
- `90%`: model needed roughly 11% more attacker points;
- `>100%`: model found a smaller reliable attack on the independent sample;
- `0%`: model attack was not reliable.

The report also includes reliability, win rates, loss ratios, decoded fleets, optimizer-fitness ratios, and percentages reaching at least 90%, 95%, and 99% oracle efficiency.

## Training

Install/update dependencies:

```powershell
pip install -r requirements.txt
```

Train from all JSONL files under the default perfect-pair directory:

```powershell
python main.py run ogia-train-attack-predictor
```

No merge step is required for fragmented JSONL files.

Useful diagnostic options:

```powershell
python main.py run ogia-train-attack-predictor --combat-simulations 32
python main.py run ogia-train-attack-predictor --skip-combat-evaluation
```

Relevant tuning options include:

```text
--multiplier-underprediction-weight 3.0
--multiplier-loss-beta 0.10
--min-ship-weight 0.01
--max-ship-types 5
--multiplier-safety-margin 0.03
```

Production defaults use 5-fold CV, up to 500 epochs, patience 50, batch size 16, hidden layers 128/64, multiplier bounds 0.5x-6.0x, and 128 fresh combat simulations for both M0 and oracle per defense.

Generated artifacts are written under `data/OGIA/models/`:

- `attack_predictor_m0_<timestamp>.pt`: deployable checkpoint;
- `attack_predictor_m0_<timestamp>_report.json`: dataset stats, fold metrics, raw/processed OOF predictions, and combat evaluation.

## Loading a model

```python
from tasks.OGIA.attack_predictor import AttackPredictor

predictor = AttackPredictor.load(
    "data/OGIA/models/attack_predictor_m0_YYYYMMDD_HHMMSS.pt"
)

genome = predictor.predict_genome(defender)
```

`predict_genome()` returns the sparse postprocessed `AttackGenome` intended for combat evaluation and future evolutionary-optimizer seeding. `predict_raw()` remains available for diagnostics.
