# Common-Random-Number Sequential Halving Pilot

ADR 0071 tested one mechanism change in the qualified exact-MLX historical
teacher: alive root candidates share the same ordered rollout-seed prefix
within each sequential-halving round. K32, R600, LMR, the model, candidate
frontier, rollout and opponent policies, scoring, elimination, and tie order
were unchanged.

## Implementation Evidence

- Four focused `nnue_batch` tests passed, including deterministic full-search
  replay and exact independent/CRN seed schedules.
- Nine differential bridge tests passed.
- The legacy-teacher binary passed focused strict Clippy.
- Native Rust and exact MLX independent search matched with zero error over
  80 R32 decisions and three R600 spots.
- R32 live-service qualification completed 80 actions per arm with zero
  fallback and clean shutdown.

## R600 Smoke

Seed 35,699:

| Metric | Independent | CRN | Delta |
|---|---:|---:|---:|
| Mean score | 95.00 | 96.25 | +1.25 |
| Wildlife | 60.25 | 62.00 | +1.75 |
| Habitat | 30.50 | 30.50 | 0.00 |
| Nature Tokens | 4.25 | 3.75 | -0.50 |
| Seconds/game | 152.50 | 154.46 | +1.96 |

Both arms completed 80 legal actions with zero bridge or policy fallback and
clean shutdown. Every smoke gate passed.

## Three-Game Pilot

Seeds 35,700-35,702:

| Metric | Independent | CRN | Delta |
|---|---:|---:|---:|
| Mean score | 95.9167 | 97.0833 | +1.1667 |
| 95% paired CI |  |  | `[+0.5778,+1.7556]` |
| Wildlife | 60.9167 | 61.5000 | +0.5833 |
| Habitat | 30.9167 | 30.8333 | -0.0833 |
| Nature Tokens | 4.0833 | 4.7500 | +0.6667 |
| Seconds/game | 155.26 | 155.69 | +0.43 |

Per-seed deltas were +1.00, +1.75, and +0.75. CRN won all three pairs. All
480 actions per arm were legal, fallback-free, and served locally through
MLX. Every preregistered pilot gate passed.

## Artifact Integrity

- R32 qualification:
  `27dbfac246a905c6910822165899816d3ebfee7bb41a223bb883d96cff140b69`
- R600 smoke:
  `c28e26ed34048fca3683cb3dc3c5a3c3f91a39fd9f2cf9d5a44125a05c7d5251`
- three-game pilot:
  `4adae7292649326e524be6c9a651f0f73ddae51c91688fa97fb33b02d1c0b37b`
- independent parity:
  `ff10f31941e3a49b6dc9acfc06c34a1d0e7fba5e680ad42494af283c3aafc4dc`

## Conclusion

Same-budget CRN is the first search-process change in this campaign to show a
clean, positive pilot on the exact MLX teacher. It is not promoted from three
games. ADR 0072 preregisters the required 20-game confirmation on fresh seeds.
