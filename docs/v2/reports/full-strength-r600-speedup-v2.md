# Full-Strength R600 Speedup V2

- Protocol: `cascadia-aaaaa-4p-base-v1`
- Search: K32, R600 sequential halving, LMR and diverse prefilter enabled
- Rollout limit: none
- Learned leaf: none
- Policy fallbacks: zero

## Strength

| Metric | Value |
|---|---:|
| Fresh games / seat scores | 50 / 200 |
| Mean score | **96.345** |
| Game-block 95% CI | `[95.892,96.798]` |
| Historical champion estimate | 95.940 |
| Paired control mean | 91.920 |
| Paired delta | +4.425 |
| Paired 95% CI | `[+3.814,+5.036]` |
| Paired record | 48-1-1 |

All 2,306,322 rollout samples remained terminal. No sample was bootstrapped,
no policy fallback occurred, and every MLX service shut down cleanly.

## Performance

| Metric | Value |
|---|---:|
| Initial seed-34400 treatment time | 141.027 s |
| Optimized seed-34400 treatment time | 37.457 s |
| Mean optimized treatment time, 50 games | 36.335 s |
| Projected single-Mac speedup | **3.88x** |
| Three-node treatment wall | 656.483 s |
| Treatment throughput speedup | **10.74x** |
| Complete paired-report wall | 705.655 s |
| End-to-end paired speedup | **9.99x** |

The speedup comes from exact reuse of repeated public-state candidate work,
in-place board transitions, exact habitat previews, local potential updates,
shared tile and wildlife calculations, and rotation-invariant memoization.

The separate two-turn MLX bootstrap screen is not part of this result. It is
faster, but matched-seed evidence showed a 3.688-point loss versus full search.
