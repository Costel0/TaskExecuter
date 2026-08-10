# OGIA Phase B — Attack Predictor M0

This package trains a supervised model from the validated JSONL pairs produced by `ogia-generate-perfect-pairs`.

## Representation

The defender is encoded as:

- defensive point share for every defensive unit type, including shield domes;
- `log1p(total_defense_points)` as an explicit scale feature.

The target is the optimizer genome itself:

- point-share weights for every allowed attack ship;
- requested attacker/defender points multiplier.

Predicting the optimizer genome instead of absolute ship counts keeps the output space compact and makes the model directly reusable as an evolutionary-search seed.

## Model

M0 is a small PyTorch MLP with a shared backbone and two heads:

- composition head: softmax over allowed attack ships, so all weights are non-negative and sum to one;
- multiplier head: bounded sigmoid mapped to the configured multiplier interval.

The composition loss is soft cross-entropy against the optimizer point-share distribution. The multiplier uses Smooth L1 loss after normalization to the configured range.

Each fold initializes the output heads from the empirical training-fold target distribution. This improves stability for the intentionally small first M0 datasets without leaking validation targets.

## Training

Install/update dependencies:

```powershell
pip install -r requirements.txt
```

Train from every JSONL file under the default perfect-pair directory:

```powershell
python main.py run ogia-train-attack-predictor
```

Or train from one specific file:

```powershell
python main.py run ogia-train-attack-predictor data/OGIA/perfect_pairs/perfect_pairs_20260809_180613.jsonl
```

Useful smoke configuration:

```powershell
python main.py run ogia-train-attack-predictor --folds 3 --epochs 50 --patience 10
```

Production defaults are 5-fold cross-validation, up to 500 epochs, early-stopping patience 50, batch size 16, hidden layers 128/64, and multiplier bounds 0.5x–6.0x.

The task writes two generated artifacts under `data/OGIA/models/`:

- `attack_predictor_m0_<timestamp>.pt`: loadable model checkpoint;
- `attack_predictor_m0_<timestamp>_report.json`: fold metrics, out-of-fold metrics and out-of-fold predictions for every pair.

The final deployable model is retrained on all available pairs for the median best epoch found during cross-validation.

## Loading a model

```python
from tasks.OGIA.attack_predictor import AttackPredictor

predictor = AttackPredictor.load(
    "data/OGIA/models/attack_predictor_m0_YYYYMMDD_HHMMSS.pt"
)

defender = {
    "rocket_launcher": 10000,
    "light_laser": 3000,
    "heavy_laser": 1000,
    "gauss_cannon": 500,
    "plasma_turret": 150,
}

genome = predictor.predict_genome(defender)
print(genome.ship_weights)
print(genome.points_multiplier)
```

`predict_genome()` returns the same `AttackGenome` type used by the Phase A optimizer. The next integration step is therefore to mix model-seeded genomes with analysis-guided and broad-exploration genomes in generation zero.
