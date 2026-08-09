# OGIA Phase A — Evolutionary Optimizer

This package searches for a compact, reliable attack against one fixed OGame defense.

## Genome

An individual contains:

- relative point weights for allowed ship types;
- `points_multiplier`, the requested attacker points divided by defender points.

`decode_attack_genome()` converts the compact representation into integer ship counts with the existing fast fleet allocator.

## Generation zero

`HybridAnalysisPopulationInitializer` reads the JSON produced by `ogia-analyze-battles` and builds a mixed population:

- 35% defense-guided;
- 20% guided by the top-efficiency attacks from the random dataset;
- 15% aggressive/small multipliers;
- 15% safe/large multipliers;
- 15% broad exploration, including specialist fleets.

The defense-guided prior is a weighted combination of the analysis rows for every defensive unit present, using their share of defender points. It is mixed 70/30 with the global top-efficiency prior.

The analysis is only a prior. No ship type is removed from the search space. Large Cargo is additionally neutralized in the analysis-based priors by replacing its observed weight with the median weight of the other ships, because the source random-battle generator included Large Cargo in virtually every attack. Evolution remains free to drive it to zero or to a large share.

## Fitness

`ReliableDefenseFitnessEvaluator` uses fixed common combat seeds for every genome and generation.

The optimization is deliberately non-economic:

1. before the configured reliability threshold (default 90%), win rate dominates;
2. once reliable, lower real attacker/defender multiplier is the main objective;
3. attacker loss ratio is secondary;
4. extra win rate is only a small tie-breaker.

This prevents the optimizer from solving the problem by simply sending the largest possible fleet.

The generator fixes weapons, shielding and armour to level 15 for both sides.

## Final validation

The console task does not save the evolutionary winner directly. It gathers diverse candidates from the final populations of several restarts and evaluates them on a new common block of combat seeds. A pair is written only if it reaches the independent validation threshold (default 95% wins over 128 battles).

Among validated candidates, selection is lexicographic:

1. smallest real attacker/defender multiplier;
2. lowest mean attacker loss ratio;
3. highest validation win rate.

## Generate pairs

The task is registered as:

```bash
python main.py run ogia-generate-perfect-pairs
```

Default input priors:

```text
data/OGIA/Analisis/merged_battles_1_analysis.json
```

Default output:

```text
data/OGIA/perfect_pairs/perfect_pairs_<timestamp>.jsonl
```

A small smoke run can be launched with:

```bash
python main.py run ogia-generate-perfect-pairs --count 1 --population-size 16 --generations 5 --training-simulations 4 --validation-simulations 32 --validation-win-rate 0.90 --restarts 1
```

The production defaults are intentionally more expensive: population 64, 50 generations, 12 training simulations per genome, two restarts, and 128 independent validation simulations.
