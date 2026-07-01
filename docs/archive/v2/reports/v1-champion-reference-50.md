# V1 Champion Reference

This report independently measures the frozen v1 `mce_wide_v1` strategy with
`nnue_weights_v4opp_modal_iter3.bin`. It is a legacy reference, not a canonical
v2 result.

## Result

- Games: 50
- Scored seats: 200
- Mean base score: 95.895
- Game-block standard deviation: 1.496
- Standard error: 0.212
- 95% confidence interval: 95.480-96.310
- P10 / P50 / P90: 92 / 96 / 99

## Mean Breakdown

| Habitat | Wildlife | Nature Tokens | Bear | Elk | Salmon | Hawk | Fox |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30.985 | 61.075 | 3.835 | 11.615 | 11.050 | 12.580 | 11.110 | 14.720 |

## Method

The legacy CLI prints a summary derived from seat 0 even when all four seats
use the same strategy. `tools/v1_champion_benchmark.py` therefore executes one
deterministic game per `CASCADIA_SEED_OFFSET`, requires exactly four
`SYMPLAYER` rows, computes game-block statistics from all seats, appends each
completed seed to durable JSONL, and writes the final report atomically.

```bash
uv run python tools/v1_champion_benchmark.py \
  --games 50 \
  --jobs 8 \
  --progress artifacts/v1-champion-50/progress.jsonl \
  --output docs/v2/reports/v1-champion-reference-50.json
```

Binary SHA-256:
`029ec9b587e50abfe6dcf2f93ec306f4c6d05cd6ff7fcbe747003214e49ff853`

Weights SHA-256:
`f40627623d3686d7d2d6a2f8f109445f54e449f0d7045552ebe831f955a58f48`

Wall time is retained in the JSON for audit but is not a latency claim because
games ran in parallel and shared the machine with compilation.
