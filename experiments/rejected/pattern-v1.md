# Pattern V1

Status: rejected on 2026-06-10.

## Hypothesis

Immediate-score greedy undervalues unfinished Bear pairs and empty placements
that can extend Fox, Elk, Salmon, or Hawk patterns.

## Treatment

For every complete legal action:

```text
value = resulting base score
      + 3.50 * isolated Bear components
      + 0.35 * distinct adjacent wildlife around empty Fox-compatible tiles
      + 0.45 * adjacent Elk around empty Elk-compatible tiles
      + 0.60 * empty Salmon-compatible tiles adjacent to exactly one Salmon
      + 0.25 * empty Hawk-compatible tiles adjacent to no Hawk
```

Ties used the same deterministic per-seat RNG as the baseline. Nature-token
market wipes were disabled in both treatment and baseline.

## Protocol And Result

Command:

```bash
target/release/cascadia-v2 benchmark \
  --games 10 --first-seed 0 --strategy pattern
```

AAAAA, four symmetric seats, no habitat bonus:

- mean: 78.65
- 95% game-block interval: 77.058-80.242
- Bear: 0.10
- elapsed: 22.72 seconds

The canonical 50-game greedy baseline is 86.265. The pilot regression was too
large to justify the planned full paired trial.

## Conclusion

Absolute handcrafted setup rewards can dominate draft selection and destroy
the very completion behavior they intend to encourage. The code was removed
from production. Future smooth-value work must be learned or calibrated
against action rankings, not assigned by intuition.
