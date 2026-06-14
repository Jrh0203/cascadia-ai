# Exact MLX Root-Candidate Pilot

Experiment: `exact-mlx-root-candidate-k32-k64-pilot-v1-20260612`

## Result

Rejected. Increasing retained root capacity from K32 to K64 did not clear the
+0.50 paired-gain gate and materially reduced habitat score.

| Metric | K32 | K64 | Delta |
|---|---:|---:|---:|
| Mean base score | 96.583 | 96.667 | +0.083 |
| Wildlife | 59.417 | 59.750 | +0.333 |
| Habitat | 33.917 | 32.417 | -1.500 |
| Nature Tokens | 3.250 | 4.500 | +1.250 |
| Seconds/game | 150.01 | 155.51 | +5.50 |

Paired 95% CI: `[-5.151,+5.318]`. Record: two wins, zero ties, one loss.
The paired game deltas were -5.25, +2.50, and +3.00.

Both arms completed all 240 canonical actions legally with zero bridge or
neural fallback, remained below the 240-second runtime ceiling, and shut down
cleanly. K64 passed the absolute-mean, wildlife, Nature Token, integrity, and
runtime gates, but failed the paired-gain and habitat gates.

The seed-34,099 smoke had shown +1.750 at 95.500 K64 mean, but that signal did
not reproduce on the frozen pilot seeds. Generic root widening is therefore
closed as the next strength bottleneck; subsequent work must improve
candidate semantics, value quality, or planning structure.

Machine-readable report:
`exact-mlx-root-candidate-k32-k64-pilot3.json`

BLAKE3:
`00bdd3b79663b03680c2a1c7fcb4911d2e9dd7ddeb3d696a7120206d18131c4c`
