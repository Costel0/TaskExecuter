# OGIA Phase B — Attack Predictor M0

This package trains a supervised model from the validated JSONL pairs produced by `ogia-generate-perfect-pairs` and evaluates its out-of-fold predictions with the real OGame combat engine.

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

## Cross-validation and combat-aware evaluation

M0 uses 5-fold cross-validation by default. Every perfect pair is therefore predicted once by a model that was not trained on that pair. These out-of-fold (OOF) predictions are the predictions used for the real combat evaluation.

After training, the task automatically runs each OOF model attack through the OGame combat simulator. For every defense it also re-simulates the Phase-A oracle attack. The two attacks use exactly the same fresh combat seeds, so random combat variance does not favor either side.

The primary metric is `oracle_efficiency`:

```text
oracle actual attacker multiplier / model actual attacker multiplier
```

It is only positive when the model attack reaches the configured reliable win-rate threshold. An unreliable model attack receives 0 efficiency. Therefore:

- `100%`: model achieved the oracle's attacker-point efficiency;
- `90%`: model required roughly 11% more attacker points than the oracle;
- `>100%`: model found a smaller reliable attack on the independent combat sample;
- `0%`: model attack was not reliable.

The report also keeps the optimizer fitness-score ratio, win rates, loss ratios, decoded integer fleets, and the percentage of defenses where M0 reaches at least 90%, 95%, and 99% oracle efficiency.

The combat engine is used as an evaluator, not as a differentiable training loss. M0 still learns by supervised backpropagation against the optimized genomes; the simulator tells us whether differences from the oracle actually matter in combat.

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
python main.py run ogia-train-attack-predictor --folds 3 --epochs 50 --patience 10 --combat-simulations 16
```

Production defaults are:

- 5-fold cross-validation;
- up to 500 epochs;
- early-stopping patience 50;
- batch size 16;
- hidden layers 128/64;
- multiplier bounds 0.5x–6.0x;
- 128 fresh combat simulations for the model and 128 for the oracle on every defense;
- 90% reliable win-rate threshold during M0 combat evaluation.

For 100 pairs, the default combat evaluation runs about 25,600 battles in total: `100 defenses x 2 attacks x 128 simulations`. This is tiny compared with the evolutionary search used to create the perfect pairs.

Combat evaluation can be adjusted or skipped:

```powershell
python main.py run ogia-train-attack-predictor --combat-simulations 64
python main.py run ogia-train-attack-predictor --combat-reliable-win-rate 0.95
python main.py run ogia-train-attack-predictor --skip-combat-evaluation
```

The task writes two generated artifacts under `data/OGIA/models/`:

- `attack_predictor_m0_<timestamp>.pt`: loadable model checkpoint;
- `attack_predictor_m0_<timestamp>_report.json`: fold metrics, OOF predictions, and combat-aware oracle evaluation for every pair.

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
