# Exact MLX Habitat-Candidate Pilot

Experiment: `exact-mlx-habitat-candidate-h6-pilot-v1-20260612`

## Result

Rejected. Unioning six canonical habitat-cohesion candidates into the exact
MLX K32/R600 root frontier improved mean score by 0.333, below the
preregistered +0.500 advancement gate.

| Metric | K32 | K32+H6 | Delta |
|---|---:|---:|---:|
| Mean base score | 96.167 | 96.500 | +0.333 |
| Wildlife | 61.333 | 61.417 | +0.083 |
| Habitat | 31.167 | 31.417 | +0.250 |
| Nature Tokens | 3.667 | 3.667 | +0.000 |
| Seconds/game | 151.83 | 150.27 | -1.56 |

Paired 95% CI: `[-1.485,+2.152]`. Record: two wins, zero ties, one loss.
The paired game deltas were +1.50, +1.00, and -1.50.

The semantic frontier generated 1,440 H6 candidates, of which 873 were novel,
645 survived the unchanged MLX prefilter, and 27 were selected. Both arms
completed all 240 selections legally with zero bridge or neural fallback,
remained below the 240-second runtime ceiling, and shut down cleanly.

H6 passed the absolute-mean, wildlife, habitat, Nature Token, integrity, and
runtime gates. It failed only the paired-gain gate. The candidate mechanism
has measurable category value, but its total-score effect is too small and
unstable to justify a larger confirmation.

Machine-readable report:
`exact-mlx-habitat-candidate-h6-pilot3.json`

BLAKE3:
`90f9438d119eefad4d01d17889341dec7f729aa75546c6737200be6bfdfe5c65`
