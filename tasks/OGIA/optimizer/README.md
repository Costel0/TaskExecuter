# OGIA Phase A — Evolutionary Optimizer

This package contains the architecture for searching the best attack against one fixed defense by natural selection.

## Genome

An individual is intentionally compact:

- relative point weight for each allowed ship type;
- attacker-points multiplier relative to the defender's points.

Example conceptually:

```text
light_fighter = 0.30
heavy_fighter = 0.10
destroyer = 0.40
large_cargo = 0.20
points_multiplier = 1.30
```

`decode_attack_genome()` converts that representation into integer ship counts using the existing fast fleet allocator.

## Evolution loop

`EvolutionaryOptimizer` already owns the generic loop:

1. obtain generation zero;
2. evaluate fitness;
3. preserve elites;
4. select parents by tournament;
5. crossover;
6. mutate;
7. evaluate the next generation;
8. repeat and keep generation statistics.

## Deliberately pending

Two decisions are intentionally interfaces only:

- `PopulationInitializer`: how generation zero is constructed;
- `FitnessEvaluator`: what makes one attack better than another.

The fitness API evaluates a whole population at once so its later implementation can efficiently run several combat seeds, multiprocessing, caching, or other batch optimizations.

No console task is registered yet because, without those two concrete strategies, the optimizer should not pretend to be runnable.
