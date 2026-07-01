# Overnight Report — Apr 19→20, 2026
_Generated: 2026-04-20 08:25_

## TL;DR

- **Cross-binary HH** (37 games): v4opp = 95.27 ± 0.40, v6peak = 94.96 ± 0.30.
  Δ = +0.31 pts (z = +0.62σ). **Champion: v4opp_modal_iter3**.
- **vs-greedy parity bench**: v4opp = 95.80, v6peak = 95.20, Δ = +0.60.
- **v6-peak training**: 20 iters complete, final RMSE = 4.8932 at LR=1e-06.

## 1. Cross-binary HH — v6peak_iter20 vs v4opp_modal_iter3

### Cross-binary HH — 37 games

| | v4opp (A) | v6peak (B) | Δ |
|---|---|---|---|
| Seat-games (n) | 74 | 74 | |
| Mean (no bonus) | **95.27 ± 0.40** | **94.96 ± 0.30** | +0.31 (z = +0.62σ) |
| Mean (w/ bonus) | 99.95 ± 0.47 | 99.89 ± 0.40 | +0.05 |
| sd / p10 / p90 | 3.47 / 91 / 99 | 2.61 / 92 / 99 | |
| min / max | 84 / 104 | 89 / 102 | |

**Win rate (best-seat per side, with bonus + Cascadia tiebreakers — leftover nature tokens, then wildlife sum, then habitat sum):**
- v4opp: 19/37 (51.4%)  [2 on tiebreak]
- v6peak: 18/37 (48.6%)  [2 on tiebreak]
- Ties: 0/37 (0.0%) (still equal after all tiebreakers)

**Per-animal mean:**
| Animal | v4opp | v6peak | Δ |
|---|---|---|---|
| bear | 11.23 | 11.68 | -0.45 |
| elk | 11.24 | 11.35 | -0.11 |
| salmon | 12.92 | 11.96 | +0.96 |
| hawk | 11.23 | 11.23 | +0.00 |
| fox | 14.39 | 15.30 | -0.91 |

**Verdict**: v4opp ahead by 0.31 pts; within noise (|z| ≤ 1.5).

## 2. vs-greedy parity bench (apples-to-apples)

Both models played the SAME bench: 50 games, mce_wide_v1 strategy, vs 3× greedy opponents, Card A scoring.

| Model | Mean | +bonus | Median | P10 | P90 | Wall |
|---|---|---|---|---|---|---|
| v4opp_modal_iter3 | **95.80** | 100.7 | 96 | 92 | 99 | 3442s |
| v6peak_iter20 | **95.20** | 99.8 | 95 | 92 | 99 | 1908s |

Per-animal mean (vs-greedy bench):

| Animal | v4opp | v6peak | Δ |
|---|---|---|---|
| bear | 24.3 | 22.8 | +1.50 |
| elk | 7.1 | 7.9 | -0.80 |
| salmon | 7.7 | 8.6 | -0.90 |
| hawk | 10.0 | 9.7 | +0.30 |
| fox | 13.1 | 13.3 | -0.20 |

## 3. v6-peak training trajectory

### v6-peak training trajectory

| Iter | Phase | LR | RMSE |
|---|---|---|---|
| 1 | 1: bootstrap | 1.00e-04 | 5.4501 |
| 2 | 1: bootstrap | 1.00e-04 | 5.2394 |
| 3 | 1: bootstrap | 1.00e-04 | 5.1101 |
| 4 | 1: bootstrap | 1.00e-04 | 5.1814 |
| 5 | 1: bootstrap | 1.00e-04 | 5.1137 |
| 6 | 2: mid | 5.00e-05 | 5.0687 |
| 7 | 2: mid | 4.56e-05 | 5.0649 |
| 8 | 2: mid | 4.11e-05 | 5.0270 |
| 9 | 2: mid | 3.67e-05 | 5.0104 |
| 10 | 2: mid | 3.22e-05 | 4.9751 |
| 11 | 2: mid | 2.78e-05 | 4.9937 |
| 12 | 2: mid | 2.33e-05 | 4.9878 |
| 13 | 2: mid | 1.89e-05 | 4.9734 |
| 14 | 2: mid | 1.44e-05 | 4.9605 |
| 15 | 2: mid | 1.00e-05 | 4.9144 |
| 16 | 3: refine | 3.00e-06 | 4.9482 |
| 17 | 3: refine | 2.50e-06 | 4.9166 |
| 18 | 3: refine | 2.00e-06 | 4.9226 |
| 19 | 3: refine | 1.50e-06 | 4.9133 |
| 20 | 3: refine | 1.00e-06 | 4.8932 |

## 4. Web app changes

- **Scoring card variant selector** added to right-panel (`#scoring-cards` UI block). Each animal has an A/B/C/D dropdown; selections persist to `localStorage`.
- **All 12 missing scoring variants** (Bear B/C/D, Elk B/C/D, Salmon B/C/D, Hawk B/C/D, Fox B/C/D) implemented in `crates/cascadia-core/src/scoring/wildlife/` with 90+ unit tests.
- Server `GET /api/state` accepts `?display_cards=A,B,C,A,D` (Bear,Elk,Salmon,Hawk,Fox order); recomputes the score breakdown with that override while leaving the game's actual `scoring_cards` (always `all_a()`) and AI logic untouched.
- Frontend re-fetches state on every selector change AND after every move, so right-panel scores always reflect the chosen cards.
- Two web binaries built: `target-web-v4/release/cascadia-web` (loads v4opp weights) and `target-web-v6/release/cascadia-web` (loads v6peak weights). Run whichever matches the champion.

## 5. Cross-binary infrastructure

New components added to enable per-binary head-to-head:
- **`cascadia-cli --daemon --weights <path>`** — long-lived daemon mode. Maintains one `GameState` internally; reads line-based commands from stdin (`INIT/PICK/APPLY/HASH/GAMEOVER/CURPLAYER/SCORES/BREAKDOWN/QUIT`); writes responses to stdout.
- **`overnight/cross_bin_hh.py`** — Python coordinator. Spawns one daemon per binary, keeps both states in lockstep by replaying every action via APPLY, hash-verifies sync every 10 moves, runs N games with rotating seat ownership for fairness, appends per-game JSONL.
- Cross-binary state-hash determinism verified: same seed → identical hash across binaries; same action sequence → identical post-move hash.

## 6. Wildlife scoring B/C/D — implementation notes

From the user-supplied tables; 65+ unit tests covering edge cases.
- **Bear B**: 10 pts per group of EXACTLY 3 (other sizes score 0).
- **Bear C**: 1=2, 2=5, 3=8 + 3-pt bonus for having all three sizes.
- **Bear D**: 2=5, 3=8, 4=14; sizes 1 and 5+ score 0.
- **Elk B**: shape-based — single (2), pair (5), triangle-3 (9), triangle+1 / rhombus (13). Bitmask-DP partitioning; line-of-3 best-partitions to 7 (pair + single).
- **Elk C**: 1..8 → 2/4/7/10/14/18/23/28; bitmask-DP over connected sub-groups.
- **Elk D**: any hex point as a center, ring score 1..6 → 2/5/8/12/16/21; rings can span otherwise-disconnected components; each elk in ≤ 1 ring; bitmask-DP picks best assignment.
- **Salmon B**: same chain rule as A, table 1=2/2=4/3=9/4=11/5+=17.
- **Salmon C**: same chain rule, min size 3, table 3=10/4=12/5+=15.
- **Salmon D**: 1 pt per salmon in run + 1 pt per UNIQUE adjacent non-salmon token (counted once even if next to multiple salmon).
- **Hawk B**: count hawks with LOS to a NON-adjacent hawk; table 2=5/3=9/4=12/5=16/6=20/7=24/8+=28.
- **Hawk C**: 3 pts per non-adjacent LOS pair (each pair counted once).
- **Hawk D**: max-weight matching of non-adjacent LOS pairs, weight = #unique-non-hawk-types in cells between, table 1=4/2=7/3+=9.
- **Fox B**: per fox, count adjacent non-fox types appearing ≥2×, table 1=3/2=5/3+=7.
- **Fox C**: per fox, count of the most-frequent adjacent non-fox type (foxes excluded).
- **Fox D**: max-weight matching of adjacent fox pairs, weight by #unique non-fox pair-types in 8 surrounding cells, table 1=5/2=7/3=9/4=11. Other foxes never count.

## 7. How to launch the right web binary

**Champion (per cross-bin HH + vs-greedy parity): `v4opp_modal_iter3`.** 
v6-peak's 17,608-feature redesign did NOT pay off; it lands ~0.3 pts behind in HH and ~0.6 pts behind vs greedy. RMSE 4.89 (v6peak) vs prior champion's deeper convergence is consistent with a feature-engineering regression.

→ Launch the **champion** web binary:
```bash
./target-web-v4/release/cascadia-web
# Cascadia web UI running at http://localhost:3000
```

To try the alternate binary in the UI for qualitative comparison:
```bash
./target-web-v6/release/cascadia-web
```

Both binaries:
- Serve the same UI (`crates/cascadia-web/src/index.html`) with the new scoring-card variant selector.
- Honor `?display_cards=A,B,C,A,D` on `/api/state` (Bear, Elk, Salmon, Hawk, Fox order); selectors at the top of the right panel persist your choices to `localStorage`.
- Are feature-gated to load only weights they can run natively (the previous default-features web binary was inadvertently loading v6peak weights with truncated columns — fixed in this build).

## 8. Files to know

| Path | Purpose |
|---|---|
| `target-web-v4/release/cascadia-web` | **Champion** web server |
| `target-web-v6/release/cascadia-web` | Alt web server |
| `target-mid-v4/release/cascadia-cli` | v4opp CLI with new `--daemon` mode |
| `target-mid-v6/release/cascadia-cli` | v6peak CLI with new `--daemon` mode |
| `overnight/cross_bin_hh.py` | Cross-binary HH coordinator (with tiebreaker logic) |
| `overnight/v6peak/cross_bin_hh.jsonl` | Per-game results |
| `overnight/v6peak/cross_bin_hh.log` | Per-game progress log |
| `overnight/v6peak/bench_v4opp_vs_greedy.log` | v4opp baseline 50-game bench |
| `overnight/v6peak/bench_v6_iter20_vs_greedy.log` | v6peak baseline 50-game bench |
| `overnight/v6peak/orchestrator.log` | v6-peak training trajectory |
| `overnight/generate_overnight_report.py` | Re-run anytime to refresh this report |
| `crates/cascadia-core/src/scoring/wildlife/{bear,elk,salmon,hawk,fox}.rs` | All 12 new B/C/D variants + 65+ tests |
| `crates/cascadia-core/src/scoring/wildlife/matching.rs` | Bitmask-DP max-weight matching (Hawk D, Fox D) |
| `crates/cascadia-cli/src/main.rs` (run_daemon / daemon_pick / daemon_apply / state_hash) | Daemon mode protocol implementation |

## 9. Honest assessment

**v6-peak missed.** The bigger feature shape (17,608 vs 11,231) trained to a worse RMSE plateau (4.89 vs prior champion's deeper convergence) and the regression carried through to play strength: −0.3 pts in HH, −0.6 pts vs greedy. Both deltas are within noise individually but consistently directional. **Do not promote v6-peak.** v4opp_modal_iter3 remains champion.

The overnight cross-binary HH was the right test to run — without it, "v6peak vs greedy = 95.2" looked indistinguishable from "v4opp HH = 95.94" because they were measured in different conditions. The parity bench (v4opp vs greedy = 95.80) plus the head-to-head (95.27 vs 94.96) jointly confirm: same conditions, v4opp wins by a hair.

**Recommended next directions** (high-priority, based on this result):
- The level of effort on v6-peak (17K-feature redesign + 20 iters of training) yielded zero. Future work should NOT chase bigger feature counts. The "only new feature SIGNAL moves the needle" pattern from the Apr 17 log holds.
- If cycles are available, the queued ideas worth trying: **OPP×MARKET cross features** (named in CLAUDE.md as a probable step-function lever), **HIDDEN1=1024 from scratch** (capacity, not features), **cross-turn MCTS tree reuse** (orthogonal to value function).
