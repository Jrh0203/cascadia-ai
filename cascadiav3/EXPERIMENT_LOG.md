# Cascadia v3 Transformer Experiment Log

This log records v3 transformer architecture experiments as they run. Entries
distinguish implementation health from model merit; dry-run experiments are not
promotion evidence.

## 2026-07-02 - `gumbel-selfplay-cycle2-v1` (EI-3) — RESULTS: rejected

Cycle 2 (400+60 seeds, w=0.75, replay window, warm start from cycle 1) is a
**rejected candidate**: no-search `91.8475` (cycle 1: `91.705`, same seeds)
and Gumbel n=64 `94.4725` (cycle 1: `94.53`; paired vs honest control
`-0.9275`, CI `[-1.31, -0.54]`, n=100). Locked-val final-Q regret improved
`0.7934 -> 0.2135` (866 steps over ~42k examples) without moving gameplay.
Cycle 1 remains champion; cycle 2 joins the opponent pool.

Interpretation branches under test: (a) serving budget n=64 may be
saturated for this model class — budget sweep n=128/256 x25 seeds running
on the champion; (b) the w=0.75 blend ramp may have degraded teacher
labels — cycle 3 reverts generation to w=0.5 at full scale.

Generated on the optimized stack at ~278 games/h (vs 40 baseline):
generation 400+60 seeds took ~2h total including a restart.

## 2026-07-02 - `gumbel-selfplay-cycle2-v1` (EI-3) — launch notes

Status: running on john0 (relaunched 14:50 on owned 6-session after a
shared-bridge throughput experiment regressed: one Python collate pipeline
vs six — see PERFORMANCE.md "Self-Play Generation Throughput"). The shared
aggregated bridge (`--shared-model-session`) is merged and tested but not
production until feature extraction moves to Rust.

Config: 400 train + 60 val seeds (fresh blocks 2026720000/2026820000),
warm start from the cycle-1 checkpoint, `GUMBEL_BLEND_WEIGHT=0.75` (ramp
per campaign; value head earned trust in cycle 1), replay window = cycle-1
shard at weight 0.5 via `EXTRA_TRAIN_TAIL_TENSORS`. ETA ~11h at ~40
games/h (6 sessions).

## 2026-07-02 - `gumbel-cycle1-gumbel-gate-candidate-v1`

Status: complete. **Cycle-1 checkpoint at Gumbel n=64/m=16/w=0.5:
`94.53`** (p50 95.0, p90 98.0, `2.49s`/decision) on the Phase A seed set —
versus EI-1's `93.36` on the same seeds and the stored honest control
`95.40`. The search gap closed from `-2.05` to about `-0.87` in one small
cycle while staying ~4.4x cheaper per decision than the rollout control.
(Aggregate comparison; the harness now exports per-seed rows so future
candidate-only runs pair offline.)

## 2026-07-02 - `gumbel-selfplay-cycle1-v1` (EI-2) — RESULTS

Status: complete. First positive evidence for the real-outcome value
training direction.

- Corpus: 120 train seeds x 80 plies = 9,600 v2 roots (+ 30 val seeds),
  all-seat self-play at n=64/m=16/w=0.5, exploration on, root menu 256.
- Training: warm start EI-1, `gumbel-selfplay` objective, steps clamped by
  the 4-pass guard to 200 (batch 192), selection metric locked-val final-Q
  regret `0.7934`.
- **100-game no-search gate: q-head `91.705` vs greedy `87.85`
  (+3.855)** — new best no-search score; EI-1 measured `90.76` (100g) /
  `90.065` (500g) with greedy anchors ~87.5. One small cycle gained
  roughly +1 point of no-search strength from 9.6k self-play roots and
  ~4 minutes of training.
- Gumbel gate (candidate side on the Phase A seed set, paired offline
  against the stored honest control 95.40): running.
- Generation throughput facts: ~40 games/h at 6 bridge sessions;
  12 sessions near-stalled the box (CUDA context thrash) — the
  shared-bridge server is the identified lever for full-scale cycles.

## 2026-07-02 - `gumbel-selfplay-cycle1-v1` (EI-2) — launch notes

Status: superseded by results above (relaunched 06:15 after throughput
sizing).

Purpose: first Gumbel self-play expert-iteration cycle. All-seat self-play
with exploration, n=64/top-m 16/w=0.5/root-menu 256, warm start from EI-1
best_locked_val, objective `gumbel-selfplay`, `MAX_EXAMPLE_PASSES=4`.

Overnight sizing decisions:

- Measured uncontended generation ~58 games/h at 6 sessions (~4.3s/decision
  — self-play with record export is ~4x slower per decision than benchmark
  games). A 1,375-seed cycle would take ~24h; resized to
  `MODEL_SESSIONS=12`, `TRAIN_SEED_COUNT=280`, `VAL_SEED_COUNT=60`
  (~22.4k train roots — EI-0 scale, far better labels) to land the full
  generate->train loop before morning. Scale-up to 1,250 seeds is the next
  daytime run once the loop is proven.
- Phase B full probe DEFERRED: after the OOM fix, a verified single
  512-sim w=1.0 game scored seat mean `94.75` at `8.84s`/decision (p50) —
  provisional model-bound evidence consistent with Phase A. The full
  20-seed probe reruns after cycle-1 training, where it measures the
  retrained value head (more decision-relevant).
- Operational lesson: the 512-sim probe and self-play generation strangle
  each other through GPU round-trip queueing (28 games/h combined). Jobs on
  john0 run strictly sequentially from now on.
- CUDA OOM root cause fixed mid-night: late-game full legal menus reach
  thousands of compound actions; the CGAB relation bias materializes
  [rows, actions, seq, d] (a 35.26 GiB single allocation was attempted).
  Fixes: `--gumbel-root-menu 256` enumeration cap, cell-budget-aware eval
  chunking in the bridge, `expandable_segments` allocator config.

## 2026-07-02 - `gumbel-phase-a-gate-v1`

Status: complete — Gumbel loses at the serving budget; proceed per the lose
branch (Phase B probe, then Phase C with the search infrastructure as the
teacher). Two headline findings:

- **Honest control = `95.4000`** (100 games, K64/R16 with
  `--rollout-determinize`) versus the leaky `96.9750`: the hidden-order peek
  inflated historical search numbers by roughly `1.6` points.
- **Gumbel n=64/m=16/w=0.5: `93.3550`** (p50 `94.0`, p90 `97.0`), paired
  delta `-2.045`, 95% CI `[-2.46, -1.63]`, n=100 — a real loss, but at
  `1.07s`/decision versus the control's `10.91s`: 10x less compute for -2
  points. Decision-time p50/p95: `0.956` / `1.204` s.

Interpretation: the current q-head (trained on rollout-mean targets) is not
yet strong enough to replace long rollouts at small budgets — exactly the
weakness Phase C's real-outcome value training targets. The search
infrastructure itself is sound and an order of magnitude cheaper per
decision.

Artifacts: `reports/gumbel_phase_a_gate.json` /
`gumbel_phase_a_gate_summary.md`.

Purpose: Phase A gate of the Gumbel campaign — 100 paired games, Gumbel
search (n=64, top-m 16, w=0.5, depth 1, 4 determinizations, full legal root
menus) versus the honest full K64/R16 rollout-search control
(`--rollout-determinize`, no hidden-order peek). Checkpoint: EI-1
`full_v3_ei1_model_state_k32_r4/best_locked_val` (strongest incumbent:
no-search q `90.065` over 500 games vs EI-0's `89.62`).

Pre-launch smoke (1 game, same config, seed 2026994000): complete game via
the batched cuda bridge, seat mean `94.0`, decision time `1.84s` at n=64 —
about 4.8x faster per decision than the legacy 8.8s full rollout search.

Gate: promote the Gumbel serving path iff the paired delta (gumbel minus
honest control) is positive with the 95% CI excluding zero
(`paired_delta_stats` in the report). Branches per
`docs/v3/GUMBEL_SELFPLAY_CAMPAIGN.md` Phase A.

Artifacts: `reports/gumbel_phase_a_gate.json`,
`reports/gumbel_phase_a_gate_summary.md`,
`logs/gumbel_phase_a_gate_job.log`.

## 2026-07-02 - `gumbel-selfplay-stack-implementation-v1`

Status: implementation complete and locally verified; remote Phase A pending.

Purpose: replace the one-ply sampled-greedy-rollout teacher with Gumbel
AlphaZero-style search using batched neural leaf values, fix the
serving-search hidden-information leak, remove greedy-ranked menu bias from
labeling, and stand up self-play data generation at 5x corpus scale. Full
plan: `docs/v3/GUMBEL_SELFPLAY_CAMPAIGN.md`.

Implementation:

- `real-root-exporter/src/gumbel.rs`: Gumbel top-m + sequential halving over
  the full legal root menu, model priors + derived-final-Q leaf values,
  hidden redeterminization before every simulated root action (no-peek by
  construction, unit-tested), max^n interior advancement, blended
  rollout/bootstrap leaf values, completed-Q + improved-policy outputs.
- `real-root-exporter/src/model_bridge.rs`: extracted bridge with
  `eval_batch` protocol (hello `protocol_features` detection, sequential
  fallback); Python bridge answers `eval_batch_request` with one collated
  forward per 32-root chunk and now returns the 4-seat `value` vector;
  relation matrices build in numpy.
- New exporter modes: `--gumbel-policy-game` (all-seat search games,
  decision JSONL) and `--gumbel-selfplay-tensor-corpus` (schema
  `cascadiav3.expert_tensor_shard.v2` with `improved_policy` +
  `search_root_value`, real-outcome value labels backfilled at terminal).
- `--rollout-determinize` makes the legacy rollout path
  public-information-legal for honest rebaselines; afterstate reuse removes
  the per-rollout clone+re-apply (golden-equality test keeps default-path
  labels bit-identical).
- Trainer: `gumbel-selfplay` objective (soft improved-policy targets,
  up-weighted real-outcome value loss), `--max-example-passes` overfit
  guard; loader/filter/materialize handle v2 alongside v1.
- Evaluation: `torch_benchmark_stats.paired_delta_stats` (t + bootstrap CI)
  wired into the search benchmark; new
  `torch_cascadiaformer_gumbel_benchmark` paired harness; runners
  `run_gumbel_phase_a_gate.sh`, `run_gumbel_ceiling_probe.sh`,
  `run_gumbel_selfplay_cycle.sh`.

Evidence:

- `cargo test` (exporter): 18 tests including golden equality, no-peek
  invariance for both rollout and Gumbel paths, determinism, sequential
  halving budget accounting, v2 shard roundtrip, and a full mock-bridge
  Gumbel game.
- Python suite: 54 tests including batch-vs-single eval equivalence, numpy
  relation-matrix equivalence, v2 load/filter/materialize, soft-target loss,
  a CPU training smoke on the v2 fixture with the passes clamp firing
  (50 -> 12 steps), and t-quantile/CI reference checks.
- Fixtures: `cascadiav3/fixtures/gumbel_tiny_tensor.npz` (v2, mock bridge).

Decision: EI-1 (`cascadiaformer-ei1-model-state-k32-r4-v1`) is terminated in
favor of this line — it kept both binding constraints (greedy rollout labels,
greedy-ranked K32 menus). Its partial artifacts are fetched and retained as
teacher-comparison evidence only, not model-promotion evidence.

## 2026-07-01 - `cascadiaformer-ei1-model-state-k32-r4-v1`

Status: terminated 2026-07-02 in favor of the Gumbel self-play campaign (see
`gumbel-selfplay-stack-implementation-v1` above); partial artifacts fetched
and retained as teacher-comparison evidence.

Purpose: start the first true model-state expert-iteration bootstrap. EI-0 was
trained on greedy-state roots and reached useful no-search/search strength, but
the search ceiling tests show that retained-set width and raw rollout count are
not the immediate 100-point bottleneck. EI-1 should expose the model to states
caused by its own q-policy while still labeling roots with sampled rollout
search.

Configuration:

- Expert tensor mode: `model_state_search_bootstrap`.
- Behavior policy: current EI-0 checkpoint q-head, advancing by model-selected
  action.
- Teacher: sampled greedy rollout mean per retained action.
- Init manifest:
  `checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json`.
- Train roots: `20,000` from 250 seeds x 80 plies.
- Validation roots: `4,000` from 50 seeds x 80 plies.
- Action menu: K32 greedy-ranked legal actions.
- Rollouts/action: `4`.
- Rollout top-k: `4`.
- Objective: `expert`, with no greedy-retention loss.
- Data mix: 70% new model-state roots, 30% EI-0 greedy-state bootstrap roots.
- Selection metric: minimum `locked_val_final_q_regret`.
- Training: 25,000 steps, batch size 192, LR `3e-5`, AdamW weight decay
  `0.05`, cosine schedule, SWA final 20%.
- Export parallelism: relaunched with `RAYON_THREADS=16` after confirming GPU
  memory headroom and CPU under-utilization in the first attempt. The first
  attempt reached `50 / 250` train seeds at 8-way generation before restart.
- Exporter implementation: model-state tensor export now chunks seeds so each
  chunk can reuse a Python model service across multiple seeds, reducing model
  startup churn in future runs.

Artifacts:

- Runbook report:
  `reports/full_v3_ei1_model_state_k32_r4_runbook.json`.
- Training report:
  `reports/full_v3_ei1_model_state_k32_r4_train.json`.
- Metrics:
  `reports/full_v3_ei1_model_state_k32_r4_metrics.jsonl`.
- Checkpoints:
  `checkpoints/full_v3_ei1_model_state_k32_r4/`.

Success readout:

- Infrastructure success: tensor invariants pass with no selected-action drops
  and no model fallback.
- Training success: locked validation final-Q regret improves from the init
  checkpoint on model-state validation roots without q/policy collapse.
- Gameplay success after training: no-search q should beat EI-0 q's `89.6175`,
  and search-integrated K56/K64 should move above the `96-97` band. The 100
  target still requires gameplay evidence, not loss alone.

## 2026-07-01 - `cascadiaformer-ei0-k64-r32-game20`

Status: completed; negative search-depth result.

Purpose: test whether the immediate 100-point bottleneck is sampled-search
strength rather than model retained-set width. The completed K56 run narrowed
the retained-search gap but both K56 and full K64 at 16 rollouts/action stayed
below the 100-point target. This run keeps the full K64 action set and doubles
rollout samples per action.

Configuration:

- Manifest:
  `checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json`.
- Seeds: same 20-game set as the K56/K64-R16 benchmark,
  starting at `2026995000`.
- Selection head: `q`.
- Retain K: `64` of max `64`.
- Rollouts/action: `32`.
- Rollout top-k: `4`.
- Candidate workers: `8`.
- Shadow full search: disabled.
- Full baseline: disabled; compare against the previous full K64
  16-rollout control on the same seeds.

Artifacts:

- Report: `reports/cascadiaformer_ei0_k64_r32_game20.json`.
- Decisions:
  `reports/cascadiaformer_ei0_k64_r32_game20_decisions.jsonl`.
- Summary: `reports/cascadiaformer_ei0_k64_r32_game20_summary.md`.
- Remote log:
  `/home/john0/cascadia/cascadiav3/logs/cascadiaformer_ei0_k64_r32_game20_job.log`.

Success readout:

- Primary: mean complete-game score moves materially toward `100` versus the
  previous full K64 R16 mean of `96.9750`.
- Strong signal: per-game means at or above `100` appear on this same seed set,
  not merely isolated high-scoring seats.
- Negative signal: the mean remains in the `96-97` band, implying that more
  one-ply rollout samples alone are not enough and the next step should improve
  rollout policy/value training rather than just spend more CPU.

Result:

- K64/R32 mean seat score: `96.8375`.
- K64/R32 P50/P90 seat scores: `97.0000` / `100.0000`.
- Previous same-seed full K64/R16 control mean: `96.9750`.
- Paired delta K64/R32 minus K64/R16: `-0.1375`.
- Bootstrap 95% CI for paired delta: approximately `[-0.9750, 0.6375]`.
- Mean total decision seconds: `18.4531`, versus K64/R16 at `8.8327`.
- Per-game means at or above `100`: `0 / 20`.
- Individual seats at or above `100`: `12 / 80`.
- Candidate workers: `8`.
- Wall-clock behavior: first wave completed around `27` minutes; total run was
  about `74` minutes. The final tail underutilized john0 because only a few
  game workers remained active.

Decision:

- Do not scale rollout count as the next 100-point path. Doubling samples per
  action roughly doubled runtime and did not move the score mean upward.
- For future CPU-bound search benchmarks on john0, use enough workers to fill
  the machine for the intended game count, typically `16-20` workers for
  20-game probes when thermals allow.
- The next scientific step should improve the learned policy/value target or
  rollout policy, then re-evaluate with K56/K64 search. Retained-set width and
  raw rollout count are not the immediate bottlenecks on current evidence.

## 2026-07-01 - `cascadiaformer-ei0-search-trace-forensics-v1`

Status: completed; K56 non-shadow follow-up completed; stronger K64 rollout
ceiling test is the next step.

Purpose: mine the completed EI-0 K32 shadow-search decision trace to choose the
next retained-search width on evidence rather than guessing.

Implementation:

- Added `cascadiav3.analyze_search_decision_trace`.
- Added runner support for non-shadow retained-search benchmarks:
  `SEARCH_SHADOW_FULL_SEARCH=0`.
- The default benchmark wrapper behavior remains unchanged:
  `SEARCH_SHADOW_FULL_SEARCH=1` and
  `SEARCH_INCLUDE_FULL_SEARCH_BASELINE=1`.

Evidence:

- Input decision trace:
  `reports/cascadiaformer_ei0_search_game20_decisions.jsonl`.
- Forensics JSON:
  `reports/cascadiaformer_ei0_search_game20_trace_forensics.json`.
- Forensics summary:
  `reports/cascadiaformer_ei0_search_game20_trace_forensics_summary.md`.
- Follow-up K56 non-shadow benchmark:
  `reports/cascadiaformer_ei0_k56_nonshadow_game20.json`.

Result:

- Candidate decisions analyzed: `1,600`.
- Full-search winner model-rank:
  - mean `17.1088`;
  - p50 `10`;
  - p90 `46`;
  - p95 `53`;
  - p99 `61`;
  - max `64`.
- Full-search winner retention by model top-K:
  - K32: `79.1875%`;
  - K40: `86.6250%`;
  - K48: `91.6875%`;
  - K56: `96.8750%`;
  - K64: `100%`.
- Phase recall at K56:
  - opening `95.25%`;
  - early-mid `96.25%`;
  - late-mid `97.25%`;
  - endgame `98.75%`.

Decision:

- K48 is the minimum width that clears `90%` overall retention, but opening
  retention is only `88.5%`.
- K56 is the next serious 100-point-path experiment because it clears `95%`
  retention in every phase while still reducing non-shadow rollout work to
  `87.5%` of K64.
- `cascadiaformer-ei0-k56-nonshadow-game20` completed on john0 with K56
  non-shadow treatment and matched K64 full-search control.
- K56 scored `96.4125` mean / `100.0000` P90 over 20 games, versus full K64
  control at `96.9750` mean / `100.0000` P90.
- Mean paired delta was `-0.5625`, improving materially on K32's `-1.1750`.
- Mean total decision seconds were `7.8031` for K56 and `8.8327` for full
  K64, for a passing treatment/control ratio of `0.8834`.
- K56 saved the expected `12.5%` of non-shadow rollout work.
- Neither K56 nor full K64 produced a per-game mean at or above `100` in this
  20-game sample, so this result supports K56 as a cheaper retained-search
  serving width but not as the final 100-point path.

Decision:

- Do not spend the next compute block merely expanding K56 confidence. K56 is
  close enough to full K64 that the immediate bottleneck is now search strength,
  not retained-set width.
- Launch the next ceiling test as all-action K64 with `32` rollouts/action and
  rollout top-k `4` on the same 20 seeds, with no redundant matched control.
  Compare it against the existing full K64 `16` rollouts/action control to see
  whether more samples move the score distribution toward the 100-point target.

## 2026-07-01 - `cascadiaformer-ei0-greedy-search-bootstrap-v1`

Status: completed, infrastructure pass, no-search gameplay pass,
search-integrated timing pass, K32 retained search trails full K64 search.

Purpose: make the first useful move beyond greedy without leaving greedy's
state distribution. Roots are generated from greedy self-play states, the action
menu is strict greedy-ranked K32, and sampled rollout search supplies
search-improved supervised targets while the objective preserves greedy
retention.

Implementation:

- Runner: `scripts/run_cascadiaformer_ei0_greedy_search_bootstrap.sh`.
- Benchmark runner: `scripts/run_cascadiaformer_ei0_benchmark_suite.sh`.
- Source runbook:
  `docs/v3/EI0_GREEDY_SEARCH_BOOTSTRAP_RUNBOOK.md`.
- Expert tensor mode: `greedy_search_bootstrap`.
- Objective: `search-improved-greedy-retention`.
- Filter: `greedy-prefix-strict`, K32.
- Model: CascadiaFormer-S.
- Warm start:
  `checkpoints/full_v3_greedy_k32_retention/best_locked_val.manifest.json`.
- Promotion checkpoint used for gameplay:
  `checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json`.

Run defaults:

- Corpus: 20,000 train roots and 4,000 locked validation roots generated from
  greedy self-play states.
- Action menu: K32 retained actions/root.
- Search target generation: 4 rollouts/action, rollout top-k 4.
- Training: 25,000 steps, batch size 192, learning rate 1e-4.
- Checkpoint selection: guarded retention-safe best, selected at step 7,250.

Evidence:

- Full runbook report:
  `reports/full_v3_ei0_greedy_search_bootstrap_runbook.json`.
- Training report:
  `reports/full_v3_ei0_greedy_search_bootstrap_train.json`.
- Metrics JSONL:
  `reports/full_v3_ei0_greedy_search_bootstrap_metrics.jsonl`.
- Tensor invariants:
  `reports/full_v3_ei0_greedy_search_bootstrap_train_tensor_invariants.json`
  and `reports/full_v3_ei0_greedy_search_bootstrap_val_tensor_invariants.json`.
- Strict K32 tensor invariants:
  `reports/full_v3_ei0_greedy_search_bootstrap_train_tensor_top32_invariants.json`
  and `reports/full_v3_ei0_greedy_search_bootstrap_val_tensor_top32_invariants.json`.
- No-search complete-game benchmark:
  `reports/cascadiaformer_ei0_no_search_game100.json`.
- Search-integrated complete-game benchmark:
  `reports/cascadiaformer_ei0_search_game20.json`.

Result:

- Runbook status: `pass`.
- Generation: 1,569 s total, 1,282 s train generation, 287 s validation
  generation.
- Throughput: `15.2964` roots/s and `1,957.9350` rollout evals/s.
- Training: 2,457 s total, `0.09828` s/step.
- Tensor invariants: train and validation raw/top32 shards all passed with
  selected-action drops `0` and max absolute Q invariant error `0.0`.
- Guarded checkpoint: step 7,250, `locked_val_total=5.8410`,
  `locked_val_greedy_top1=0.69375`, `locked_val_mean_greedy_rank=1.8860`,
  `locked_val_teacher_top1=0.13025`, and
  `locked_val_teacher_advantage_over_greedy=2.1672`.
- Final/SWA checkpoint was not selected because validation greedy retention
  collapsed relative to the guarded checkpoint.
- 100-game no-search benchmark:
  - greedy mean `87.5575`, P90 `92.0000`;
  - CascadiaFormer policy mean `87.7925`, P90 `92.0000`, paired delta
    `+0.2350`, exact greedy-action match `70.1125%`;
  - CascadiaFormer q mean `89.6175`, P90 `94.0000`, paired delta `+2.0600`,
    exact greedy-action match `29.8125%`.
- 20-game search-integrated benchmark:
  - CascadiaFormer-search mean `95.8000`, P90 `99.0000`;
  - full-search control mean `96.9750`, P90 `100.0000`;
  - paired delta candidate-control `-1.1750`;
  - treatment/control time ratio `1.0044`, passing the `<=1.20` gate;
  - full-search winner retained rate `79.1875%`;
  - mean shadow search regret `0.0916`;
  - mean model score time/root `0.0170s` versus mean search time/root
    `8.8088s`, so this benchmark is CPU rollout-bound, not GPU-bound.

Interpretation: EI-0 is the first CascadiaFormer run with positive no-search
gameplay evidence against greedy. The q head is clearly useful without search,
and K32 retained search reaches a strong absolute mean above `95`, but the
retained set still trails matched full K64 search by `1.175` points. Treat this
as real merit and a good bootstrap, not as proof that K32 model-retained search
is stronger than full search. The next improvement should raise full-search
winner retention or train/evaluate a retained set larger than K32 before scaling
to a broader expert-iteration loop.

## 2026-07-01 - `cascadiaformer-k32-greedy-retention-v1`

Status: completed, infrastructure pass, model is near-greedy but not a full
greedy clone.

Purpose: redo the earlier "copy greedy" test inside the actual
CascadiaFormer-S expert-root serving surface. The previous full expert run used
rollout-selected teacher labels and Q/value auxiliaries, then failed to retain
greedy behavior under complete-game K32 evaluation. This run makes greedy
retention explicit before trying to improve beyond it.

Implementation:

- `torch_train_cascadiaformer.py` now supports objective presets:
  `expert`, `k32-greedy-retention`, and `pure-greedy-retention`.
- Every expert batch carries `greedy_action_index=0`, matching the
  greedy-ranked K32 action menu served by the Rust interactive game bridge.
- The trainer now logs `greedy_policy`, `greedy_margin`, `greedy_top1`, and
  `mean_greedy_rank` separately from rollout teacher policy metrics.
- Expert tensor filtering now supports strict
  `--filter-mode greedy-prefix-strict` for exact served K32 greedy-copy tests,
  plus `greedy-prefix-with-selected` for later blended teacher-retention runs.
- `scripts/run_cascadiaformer_k32_greedy_retention.sh` wraps the full pipeline
  with K32 relation-tail tensors, `OBJECTIVE=pure-greedy-retention`, and
  checkpoint selection by `locked_val_greedy_policy`.
- `real-root-exporter --greedy-expert-tensor-corpus` writes
  `cascadiav3.expert_tensor_shard.v1` directly from greedy self-play states, so
  the copy-greedy diagnostic no longer depends on rollout/expert trajectory
  states.
- `scripts/run_cascadiaformer_greedy_k32_retention.sh` wraps the corrected
  on-policy greedy-state run with `EXPERT_TENSOR_MODE=greedy`,
  strict K32 filtering, relation-tail materialization, and pure
  greedy-retention training.
- The generic full-v3 fetch helper is scoped to the active run's reports,
  manifests, log, and checkpoint directory instead of syncing every historical
  checkpoint.

Run defaults:

- Profile: `greedy_k32_retention` for the corrected run.
- Corpus: 10,000 train roots and 2,000 locked validation roots generated from
  greedy self-play states.
- Filter: strict K32 greedy prefix.
- Model: CascadiaFormer-S.
- Steps: 1,500.
- Batch size: 128.
- Learning rate: 5e-4.
- Validation: full locked validation set.
- Selection metric: minimum `locked_val_greedy_policy`.

Evidence:

- Rollout/expert-state offline run:
  `reports/full_v3_phase0_bootstrap_jsonl_k32_greedy_retention_train.json`
- Rollout/expert-state failed complete-game check:
  `reports/cascadiaformer_k32_greedy_retention_game20_benchmark.json`
- Corrected greedy-state training:
  `reports/full_v3_greedy_k32_retention_train.json`
- Corrected greedy-state runbook:
  `reports/full_v3_greedy_k32_retention_runbook.json`
- Corrected greedy-state 20-game benchmark:
  `reports/cascadiaformer_greedy_k32_retention_game20_benchmark.json`
- Corrected greedy-state 100-game benchmark:
  `reports/cascadiaformer_greedy_k32_retention_game100_benchmark.json`
- Human benchmark summary:
  `reports/cascadiaformer_game_benchmark_summary.md`

Result:

- Rollout/expert-state strict K32 run selected step 1,350 with
  `locked_val_greedy_policy=3.93e-08`, `locked_val_greedy_top1=1.0`, and
  `locked_val_mean_greedy_rank=1.0`.
- That same checkpoint failed complete-game evaluation: 20 games scored
  CascadiaFormer policy `73.45` versus greedy `87.8375`, with only
  `11.0625%` exact greedy-action match. Interpretation: offline copying on
  rollout/expert trajectory roots did not transfer to self-play states.
- Corrected greedy-state run generated train/val tensors with zero strict K32
  selected-action drops and selected step 1,500.
- Corrected locked validation metrics: `locked_val_greedy_policy=1.3817`,
  `locked_val_greedy_top1=0.6780`, and `locked_val_mean_greedy_rank=2.1640`.
- Corrected 20-game benchmark: CascadiaFormer policy `87.0000`, greedy
  `87.4125`, paired delta `-0.4125`, exact greedy-action match `69.3125%`.
- Corrected 100-game benchmark: CascadiaFormer policy `86.7800`, greedy
  `87.5875`, paired delta `-0.8075`, exact greedy-action match `67.3625%`,
  mean greedy rank in model `2.14825`.

Interpretation: the corrected CascadiaFormer-S no-search policy is now close
to greedy on score, which validates the serving path, tensor path, objective,
and benchmark. It is not a solved behavior clone. The next credible step is
not expert iteration yet; first either improve copy-greedy fidelity with more
greedy states/capacity/objective tuning or add an explicit distilled greedy
score/rank target that makes the hand-coded ranker's decision boundary easier
to learn.

## 2026-06-30 - `greedy-policy-rust-tensor-shards-v1`

Status: completed, infrastructure pass.

Purpose: replace large greedy behavior-cloning JSONL corpora with a lean
training artifact before scaling to hundreds of thousands of games.

Implementation:

- New compact shard module: `src/cascadiav3/greedy_tensor_shards.py`.
- Shard format: versioned `.npz`, `float16` public-token tensors, `float16`
  semantic action tensors, per-root offsets, and selected greedy action index.
- Trainer support: `torch_greedy_policy_pretrain.py` now accepts
  `--train-format npz` and `--val-format npz`.
- Exporter support: `real-root-exporter --greedy-policy-corpus --out -` streams
  canonical JSONL records on stdout, with diagnostics on stderr.
- Rust-native exporter support:
  `real-root-exporter --greedy-policy-tensor-corpus` writes trainer-ready `.npz`
  shards directly from simulator records without JSONL or Python feature
  extraction in the generation path.
- Compression control: `--tensor-compression deflate|stored`; the runner exposes
  this as `TENSOR_COMPRESSION`.
- Runner support: `scripts/run_greedy_policy_pretrain.sh` defaults to streamed
  Rust-native `.npz` corpus generation and only persists JSONL when
  `KEEP_JSONL=1`, `CORPUS_FORMAT=jsonl`, or an explicit Python exporter mode is
  selected.
- Source sync guard: the `john0` runner excludes remote `fixtures/`, `reports/`,
  `checkpoints/`, `logs/`, and `target/` from `rsync --delete` so generated
  artifacts survive future launches.

Evidence:

- Raw generation benchmark: `reports/greedy_policy_generation_benchmark.json`
- JSONL-file conversion report:
  `reports/greedy_policy_bench_t32_128_tensor_shard.json`
- Streamed exporter-to-NPZ report:
  `reports/greedy_policy_bench_stream_t32_128_tensor_shard.json`
- Rust deflated 1,024-game benchmark:
  `reports/greedy_policy_bench_rust_tensor_t32_1024.json`
- Rust deflated timing:
  `reports/greedy_policy_bench_rust_tensor_t32_1024_time.txt`
- Rust stored 1,024-game benchmark:
  `reports/greedy_policy_bench_rust_tensor_stored_t32_1024.json`
- Rust stored timing:
  `reports/greedy_policy_bench_rust_tensor_stored_t32_1024_time.txt`
- Rust/Python parity: `reports/greedy_policy_rust_tensor_parity_seq16.json`
- Deflated loader smoke: `reports/greedy_policy_rust_tensor_loader_smoke.json`
- Stored loader smoke:
  `reports/greedy_policy_rust_tensor_stored_loader_smoke.json`
- Human summary: `reports/greedy_policy_compact_tensor_shard_summary.md`

Result:

- Raw JSONL, 128 games: 10,240 roots, 794 MiB, 2.39s generation time.
- JSONL file to `float16` `.npz`: 9.0 MiB, 21.27s conversion time,
  914 bytes/root.
- Streamed exporter stdout to `float16` `.npz`: 9.0 MiB, 20.60s total wall,
  915 bytes/root, no raw JSONL persisted.
- Rust-native deflated `.npz`, 1,024 games: 81,920 roots, 1:35.36 wall,
  71,413,962 bytes, 871.8 bytes/root.
- Rust-native stored `.npz`, 1,024 games: 81,920 roots, 0:18.28 wall,
  1,248,396,657 bytes, 15,239.2 bytes/root.
- Stored/uncompressed `.npz` is `5.22x` faster than deflated `.npz` and
  `17.48x` larger on the measured 1,024-game shard.
- The deterministic 16-game parity audit matched the Python feature path
  exactly: token/action tensors, offsets, and selected action indices all match;
  max absolute token/action diff is `0.0`.
- RTX 5090 loader smokes passed for both deflated and stored shards with CUDA,
  checkpoint round-trip, and train/validation record counts of 81,920 each.
- RTX 5090 stored-shard consumption benchmark reached 4,553 roots/s at batch
  512, with data fetch plus host-to-device copy at about `8.6%` of step time
  and GPU compute at about `91.4%`.

Interpretation: the Rust feature-extraction migration is complete for the greedy
behavior-cloning corpus. Use deflated `.npz` when storage or transfer cost
matters. Use `TENSOR_COMPRESSION=stored` when generation throughput matters and
disk is available: it projects to about 29.8 minutes and 121.9 GB per 100,000
games, versus about 2.59 hours and 7.0 GB for deflated shards.

## 2026-06-30 - `crt-mini-action-query-merit-v1`

Status: completed, `has_merit=false`.

Purpose: test whether a tiny action-query TransformerEncoder over scalar
state/action features can beat an immediate-score baseline and a same-feature
MLP on held-out simulator root action ranking.

Data:

- Train: `fixtures/crt_merit_train.jsonl`, 400 roots, 16 retained actions/root.
- Validation: `fixtures/crt_merit_val.jsonl`, 100 roots, 16 retained actions/root.
- Labels: `canonical_simulator_greedy_rollout_dry_run`.

Evidence:

- Default report: `reports/crt_merit_pilot.json`
- Lower-learning-rate retry: `reports/crt_merit_pilot_lr3e4.json`
- Human summary: `reports/crt_merit_pilot_summary.md`

Result:

- Default transformer mean regret 5.14 vs immediate 4.85 and MLP 4.50.
- Lower-learning-rate transformer mean regret 5.15 vs immediate 4.85 and MLP 4.66.
- Checkpoint round-trip passed for both runs.

Interpretation: the scalar feature pilot is healthy as infrastructure but does
not justify expert iteration. Next step is real public-state/action tokenization
and relation-aware inputs.

## 2026-06-30 - `crt-public-token-query-merit-v1`

Status: completed, `has_merit=false`.

Purpose: test whether simulator-exported public tokens and C-GAB-style relation
summaries improve held-out action ranking versus immediate-score and token-pooled
MLP baselines.

Data:

- Train: `fixtures/crt_token_merit_train.jsonl`, 400 roots, 16 retained actions/root.
- Validation: `fixtures/crt_token_merit_val.jsonl`, 100 roots, 16 retained actions/root.
- Labels: `canonical_simulator_greedy_rollout_dry_run`.
- Generated on `john0` CPU.

Implementation:

- Enriched Rust exporter sidecar: `public_tokens.schema_id = cascadiav3.public_tokens.v1`.
- Public tokens: players, placed tiles for all seats, frontiers, market tiles,
  market wildlife, and public supply.
- Relations: directed `adjacent_hex` and bidirectional `same_market_slot`
  relation templates.
- Trainer: `src/cascadiav3/torch_public_token_merit.py`.
- Runner: `scripts/run_crt_public_token_pilot.sh`.

Evidence:

- Report: `reports/crt_public_token_pilot.json`
- Checkpoint: `checkpoints/crt_public_token_pilot.pt`
- Human summary: `reports/crt_public_token_pilot_summary.md`

Result:

- immediate-score baseline: top-1 0.13, top-4 0.27, mean regret 4.93.
- token-pooled MLP: top-1 0.08, top-4 0.33, mean regret 4.77.
- public-token transformer: top-1 0.08, top-4 0.32, mean regret 4.73.
- `has_merit=false`: regret improved only 4.1% versus immediate score and
  top-1 was 5pp worse than immediate score.
- Checkpoint round-trip passed.

Decision gate:

- `has_merit=true` requires at least 10% lower regret or at least 5pp top-1 gain
  versus immediate-score baseline, plus nonregression versus the token-pooled
  MLP baseline.

Interpretation: the public-token bridge, relation sidecar, CUDA training path,
and artifact round-trip are healthy. The model result is still a red light for
full expert iteration. The next credible transformer test needs stronger
teacher labels and actual relation-aware attention bias rather than only
scalarized relation-degree summaries.

## 2026-06-30 - `crt-relation-bias-query-merit-v1`

Status: completed, `has_merit=false`.

Purpose: test whether C-GAB-style learned additive attention bias over public
state and action-query relations improves held-out action ranking versus
immediate-score, token-pooled MLP, and same-run vanilla public-token Transformer
baselines.

Data:

- Train: `fixtures/crt_token_merit_train.jsonl`, 400 roots, 16 retained actions/root.
- Validation: `fixtures/crt_token_merit_val.jsonl`, 100 roots, 16 retained actions/root.
- Labels: `canonical_simulator_greedy_rollout_dry_run`.
- Generated on `john0` CPU.

Implementation:

- Relation ids: same-owner board, adjacent hex, terrain-match adjacency,
  same-market slot, action-to-tile-slot, action-to-wildlife-slot,
  action-to-tile-frontier, and action-to-wildlife-cell.
- Trainer: `src/cascadiav3/torch_relation_bias_merit.py`.
- Runner: `scripts/run_crt_relation_bias_pilot.sh`.

Evidence:

- Report: `reports/crt_relation_bias_pilot.json`
- Checkpoint: `checkpoints/crt_relation_bias_pilot.pt`
- Human summary: `reports/crt_relation_bias_pilot_summary.md`

Result:

- immediate-score baseline: top-1 0.13, top-4 0.27, mean regret 4.93.
- token-pooled MLP: top-1 0.10, top-4 0.29, mean regret 4.92.
- vanilla public-token Transformer: top-1 0.12, top-4 0.30, mean regret 4.93.
- relation-bias Transformer: top-1 0.10, top-4 0.33, mean regret 4.69.
- `has_merit=false`: regret improved 4.9% versus immediate score and same-run
  vanilla Transformer, but top-1 remained 3pp worse than immediate score.
- Checkpoint round-trip passed.

Decision gate:

- `has_merit=true` requires the existing CRT gate versus immediate-score and
  MLP baselines, plus nonregression versus the same-run vanilla public-token
  Transformer.

Interpretation: learned relation bias is directionally useful, but the dry-run
teacher labels remain the likely bottleneck. More architecture churn on the same
greedy-rollout labels is lower value than improving the teacher target quality.

## 2026-06-30 - `crt-sampled-teacher-relation-bias-v1`

Status: completed, `has_merit=true`.

Purpose: retest the relation-bias Transformer after improving target quality
from one deterministic greedy terminal rollout per retained action to mean
labels over repeated sampled top-k greedy continuations.

Data:

- Train: `fixtures/crt_sampled_teacher_train.jsonl`, 400 roots, 16 retained actions/root.
- Validation: `fixtures/crt_sampled_teacher_val.jsonl`, 100 roots, 16 retained actions/root.
- Labels: 4 rollout samples per retained action, top-4 sampled greedy
  continuation policy.
- Generated on `john0` CPU with per-seed parallelism.

Implementation:

- Exporter flags: `--rollouts-per-action`, `--rollout-top-k`.
- New optional fields: `per_action_Q_variance`, `per_action_Q_count`.
- Runner: `scripts/run_crt_sampled_teacher_relation_bias_pilot.sh`.

Evidence:

- Report: `reports/crt_sampled_teacher_relation_bias_pilot.json`
- Checkpoint: `checkpoints/crt_sampled_teacher_relation_bias_pilot.pt`
- Human summary: `reports/crt_sampled_teacher_relation_bias_pilot_summary.md`

Result:

- immediate-score baseline: top-1 0.05, top-4 0.26, mean regret 3.2800.
- token-pooled MLP: top-1 0.06, top-4 0.35, mean regret 2.7400.
- vanilla public-token Transformer: top-1 0.09, top-4 0.35, mean regret 2.7525.
- relation-bias Transformer: top-1 0.12, top-4 0.36, mean regret 2.5975.
- `has_merit=true`: regret improved 20.8% and top-1 improved 7pp versus
  immediate score; regret improved 5.6% and top-1 improved 3pp versus same-run
  vanilla public-token Transformer.
- Checkpoint round-trip passed.

Decision gate:

- Same as `crt-relation-bias-query-merit-v1`; this run tests whether better
  supervision makes the relation-aware architecture worth scaling.

Interpretation: this is the first v3 Transformer green light. It is offline
teacher-ranking merit, not gameplay strength. The next credible step is a
larger sampled-teacher shard, stronger rollout counts, and a larger
relation-bias model evaluated for top-K retention/prefilter usefulness.

## 2026-06-30 - `crt-scaled-sampled-teacher-relation-bias-v1`

Status: completed, `has_merit=true`.

Purpose: scale the first green-light recipe to a larger sampled-teacher shard,
stronger per-action labels, and a wider relation-bias Transformer while
measuring top-K prefilter retention.

Data:

- Train: `fixtures/crt_scaled_sampled_teacher_train.jsonl`, 1600 roots,
  16 retained actions/root.
- Validation: `fixtures/crt_scaled_sampled_teacher_val.jsonl`, 400 roots,
  16 retained actions/root.
- Labels: 8 rollout samples per retained action, top-4 sampled greedy
  continuation policy.
- Generated on `john0` CPU with per-seed parallelism.

Implementation:

- Runner: `scripts/run_crt_scaled_sampled_teacher_relation_bias_pilot.sh`.
- Model: 4-layer relation-bias Transformer, hidden size 256, 8 heads, MLP 512.
- New metrics: top-2/top-4/top-8/top-16 prefilter recall and oracle regret.
- First generation attempt exposed a sampled-continuation resource-exhaustion
  edge case (`WildlifeBagEmpty`). The exporter now treats `WildlifeBagEmpty` and
  `TileStackEmpty` during teacher continuations as scored truncated samples and
  records `per_action_truncated_count`.

Evidence:

- Report: `reports/crt_scaled_sampled_teacher_relation_bias_pilot.json`
- Checkpoint: `checkpoints/crt_scaled_sampled_teacher_relation_bias_pilot.pt`
- Human summary: `reports/crt_scaled_sampled_teacher_relation_bias_pilot_summary.md`

Result:

- immediate-score baseline: top-1 0.0800, top-4 0.3025, top-8 0.5300,
  mean regret 2.2441.
- token-pooled MLP: top-1 0.1325, top-4 0.3750, top-8 0.6175,
  mean regret 2.0000.
- vanilla public-token Transformer: top-1 0.1325, top-4 0.3975, top-8 0.6100,
  mean regret 1.9831.
- relation-bias Transformer: top-1 0.1675, top-4 0.4050, top-8 0.6500,
  mean regret 1.7838.
- Prefilter oracle regret for relation-bias: top-4 0.7200, top-8 0.2963.
- `has_merit=true`: regret improved 20.5% and top-1 improved 8.75pp versus
  immediate score; regret improved 10.1% versus same-run vanilla Transformer.
- Checkpoint round-trip passed.
- Truncation accounting: 2 truncated train rollout samples out of 204800,
  0 validation truncations out of 51200.

Decision gate:

- `has_merit=true` remains the offline action-ranking gate. The prefilter signal
  is separately judged by top-K recall and top-K oracle regret.

Interpretation: scaling strengthened the relation-bias result, especially
top-8 oracle regret. It is still not gameplay strength. The prefilter signal is
promising but not yet narrow enough; next work should evaluate transformer
prefiltering against stronger teacher labels and/or larger retained action sets.

## 2026-06-30 - `crt-wide32-sampled-teacher-relation-bias-v1`

Status: completed, `has_merit=true`.

Purpose: test whether the sampled-teacher relation-bias Transformer still has
offline merit and useful prefilter retention when the retained legal-action
surface widens from 16 to 32 greedy-ranked candidates per root.

Data:

- Train: `fixtures/crt_wide32_sampled_teacher_train.jsonl`, 1200 roots,
  32 retained actions/root.
- Validation: `fixtures/crt_wide32_sampled_teacher_val.jsonl`, 300 roots,
  32 retained actions/root.
- Labels: 8 rollout samples per retained action, top-4 sampled greedy
  continuation policy.
- Generated on `john0` CPU with per-seed parallelism.

Implementation:

- Runner: `scripts/run_crt_wide32_sampled_teacher_relation_bias_pilot.sh`.
- Model: 4-layer relation-bias Transformer, hidden size 256, 8 heads, MLP 512.
- Metrics include top-2/top-4/top-8/top-16/top-32 prefilter recall and oracle
  regret.

Evidence:

- Report: `reports/crt_wide32_sampled_teacher_relation_bias_pilot.json`
- Checkpoint: `checkpoints/crt_wide32_sampled_teacher_relation_bias_pilot.pt`
- Human summary: `reports/crt_wide32_sampled_teacher_relation_bias_pilot_summary.md`

Result:

- immediate-score baseline: top-1 0.0500, top-4 0.2133, top-8 0.3233,
  top-16 0.6100, mean regret 2.5129.
- token-pooled MLP: top-1 0.0633, top-4 0.2200, top-8 0.4000,
  top-16 0.6400, mean regret 2.3813.
- vanilla public-token Transformer: top-1 0.0800, top-4 0.2667,
  top-8 0.4200, top-16 0.6433, mean regret 2.3421.
- relation-bias Transformer: top-1 0.0833, top-4 0.2767, top-8 0.4333,
  top-16 0.6767, mean regret 2.2408.
- Prefilter oracle regret for relation-bias: top-4 1.0150, top-8 0.5658,
  top-16 0.2542.
- `has_merit=true`: regret improved 10.8% versus immediate score and 4.3%
  versus same-run vanilla Transformer.
- Checkpoint round-trip passed.
- Truncation accounting: 1 truncated train rollout sample out of 307200,
  0 validation truncations out of 76800.

Decision gate:

- `has_merit=true` remains the offline action-ranking gate. The main prefilter
  question is whether top-8/top-16 recall and oracle regret stay useful over 32
  retained actions.

Interpretation: relation-bias remains the best tested architecture on a wider
candidate set, but top-8 is too narrow for reliable teacher-best preservation
over 32 retained actions. Top-16 is more plausible as a prefilter candidate.
Next work should test stronger teacher labels and/or a serving-shaped top-16 or
top-24 prefilter rather than claiming direct gameplay strength.

## 2026-06-30 - `crt-wide32-relation-bias-prefilter-eval-v1`

Status: completed, serving gate passes at K=24.

Purpose: replay the wide-32 relation-bias checkpoint as a serving-shaped
prefilter and produce per-root retained action sets for a future search bridge.
This checks whether the trained Transformer can safely reduce 32 retained
candidates before downstream search/evaluation.

Data:

- Validation: `fixtures/crt_wide32_sampled_teacher_val.jsonl`, 300 roots,
  32 retained actions/root.
- Labels: same 8 rollout samples per retained action from
  `crt-wide32-sampled-teacher-relation-bias-v1`.
- No new training data generated.

Implementation:

- Evaluator: `src/cascadiav3/torch_prefilter_eval.py`.
- Runner: `scripts/run_crt_wide32_prefilter_eval.sh`.
- Checkpoint:
  `checkpoints/crt_wide32_sampled_teacher_relation_bias_pilot.pt`.
- Gate: smallest K with teacher-best recall >= 0.750 and mean oracle regret
  <= 0.250 sampled-teacher points.
- New metric contract: K=24 is now emitted alongside K=4/8/16/32 in the
  action-query, public-token, and relation-bias evaluators.

Evidence:

- Report: `reports/crt_wide32_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_prefilter_eval_roots.jsonl`
- Human summary: `reports/crt_wide32_prefilter_eval_summary.md`

Result:

- relation-bias K=8: recall 0.4333, mean oracle regret 0.5658.
- relation-bias K=16: recall 0.6767, mean oracle regret 0.2542.
- relation-bias K=24: recall 0.8567, mean oracle regret 0.1058.
- relation-bias K=32: recall 1.0000, mean oracle regret 0.0000.
- Immediate-score K=24 also passes the same safety gate: recall 0.7900,
  mean oracle regret 0.1229.
- Remote verification on `john0`: RTX 5090 visible, Torch
  `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 15 unit/schema tests passed.

Decision gate:

- K=24 is the smallest passing serving candidate for the relation-bias
  Transformer.
- K=16 does not pass: it narrowly misses oracle regret and materially misses
  teacher-best recall.
- This is a 25% candidate-pruning result, not a direct gameplay-strength result
  and not a decisive transformer-only serving win because immediate-score K=24
  also passes the same safety gate.

Interpretation: the current relation-bias checkpoint is safe enough to test as a
top-24 prefilter over 32 retained actions, but top-16 remains the practical
research target. The next useful move is either end-to-end K=24 integration to
measure real search cost/strength or a stronger prefilter training objective
aimed specifically at top-16 retention.

## 2026-06-30 - `crt-wide32-top16-margin-relation-bias-v1`

Status: completed, `has_merit=true`, top-16 serving gate still fails.

Purpose: test whether an objective closer to the literature-first proposal's
`teacher_weighted_distributional_policy + pairwise_action_margin` recipe can
turn the existing wide-32 relation-bias Transformer into a viable top-16
prefilter.

Data:

- Train: `fixtures/crt_wide32_sampled_teacher_train.jsonl`, 1200 roots,
  32 retained actions/root.
- Validation: `fixtures/crt_wide32_sampled_teacher_val.jsonl`, 300 roots,
  32 retained actions/root.
- Labels: 8 rollout samples per retained action, top-4 sampled greedy
  continuation policy.
- No new roots were generated for this run.

Implementation:

- Loss mode: `top16-prefilter`.
- Loss terms: weighted normalized-Q regression, sharpened listwise policy
  imitation, and teacher-best pairwise margin.
- Label reliability: rollout-count and root-normalized variance weighting,
  clamped to `[0.05, 1.0]`.
- Runner: `scripts/run_crt_wide32_top16_margin_relation_bias_pilot.sh`.
- Code: `src/cascadiav3/torch_relation_bias_merit.py`.

Evidence:

- Training report:
  `reports/crt_wide32_top16_margin_relation_bias_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_top16_margin_relation_bias_pilot.pt`
- Serving eval:
  `reports/crt_wide32_top16_margin_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_top16_margin_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_top16_margin_relation_bias_pilot_summary.md`

Result:

- relation-bias point metrics: top-1 0.1000, mean regret 2.2108.
- relation-bias K=8: recall 0.4067, mean oracle regret 0.6613.
- relation-bias K=16: recall 0.6833, mean oracle regret 0.2704.
- relation-bias K=24: recall 0.8633, mean oracle regret 0.0800.
- `has_merit=true`: regret improved 12.0% versus immediate score and 15.4%
  versus same-run vanilla Transformer; top-1 gained 5pp versus immediate.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 16 unit/schema tests
  passed.

Decision gate:

- Top-16 serving gate still fails: recall is below 0.750 and oracle regret is
  above 0.250.
- Top-24 still passes.
- Compared with the previous standard relation-bias checkpoint, the margin loss
  improves top-1 and mean regret but worsens K=16 oracle regret.

Interpretation: a teacher-best margin term is not enough to create a reliable
top-16 prefilter on the current 8-rollout/action labels. The next credible move
is stronger sampled-teacher labels rather than more objective shaping on the same
noisy target.

## 2026-06-30 - `crt-wide32-r16-sampled-teacher-relation-bias-v1`

Status: completed, `has_merit=false` by the same-run vanilla nonregression
gate, but strongly improved prefilter retention.

Purpose: test whether stronger sampled-teacher labels solve the top-16
prefilter problem better than objective shaping on the old 8-rollout/action
labels.

Data:

- Train: `fixtures/crt_wide32_r16_sampled_teacher_train.jsonl`, 1200 roots,
  32 retained actions/root.
- Validation: `fixtures/crt_wide32_r16_sampled_teacher_val.jsonl`, 300 roots,
  32 retained actions/root.
- Labels: 16 rollout samples per retained action, top-4 sampled greedy
  continuation policy.
- Generated fresh on `john0` CPU with per-seed parallelism.
- Train rollout samples: 614400; validation rollout samples: 153600.

Implementation:

- Runner: `scripts/run_crt_wide32_r16_sampled_teacher_relation_bias_pilot.sh`.
- Model: 4-layer relation-bias Transformer, hidden size 256, 8 heads, MLP 512.
- Training: 5200 steps, LR 0.00035, standard loss.

Evidence:

- Train fixture:
  `fixtures/crt_wide32_r16_sampled_teacher_train.jsonl`
- Validation fixture:
  `fixtures/crt_wide32_r16_sampled_teacher_val.jsonl`
- Training report:
  `reports/crt_wide32_r16_sampled_teacher_relation_bias_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16_sampled_teacher_relation_bias_pilot.pt`
- Serving eval: `reports/crt_wide32_r16_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16_sampled_teacher_relation_bias_pilot_summary.md`

Result:

- relation-bias point metrics: top-1 0.1333, mean regret 1.5683.
- relation-bias K=8: recall 0.5133, mean oracle regret 0.3821.
- relation-bias K=16: recall 0.7233, mean oracle regret 0.1552.
- relation-bias K=24: recall 0.9167, mean oracle regret 0.0381.
- Same-run vanilla public-token Transformer: top-1 0.1533, mean regret 1.4700,
  K=16 recall 0.6900, K=16 oracle regret 0.1831.
- `has_merit=false`: relation-bias beats immediate and MLP, but does not beat
  same-run vanilla point regret/top-1.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 16 unit/schema tests
  passed.

Decision gate:

- Top-16 oracle-regret gate now passes comfortably: 0.1552 <= 0.250.
- Top-16 recall still misses: 0.7233 < 0.750.
- Top-24 remains a strong pass.

Interpretation: stronger labels moved the true prefilter target much more than
the margin-loss objective did. The remaining top-16 failure is recall, not
oracle regret. Relation-bias remains the best same-run K=16 prefilter, while
vanilla remains the better point selector on this R16 shard.

## 2026-06-30 - `crt-wide32-r16-prefilter-blend-eval-v1`

Status: completed, validation top-16 serving gate still fails.

Purpose: test whether simple serving calibration can recover the remaining
top-16 recall gap by blending normalized relation, vanilla, MLP, and immediate
scores. Weights are selected on the R16 training shard and evaluated once on the
held-out R16 validation shard.

Implementation:

- Evaluator: `src/cascadiav3/torch_prefilter_blend_eval.py`.
- Runner: `scripts/run_crt_wide32_r16_prefilter_blend_eval.sh`.
- Checkpoint:
  `checkpoints/crt_wide32_r16_sampled_teacher_relation_bias_pilot.pt`.
- Grid: simplex weights over relation/vanilla/MLP/immediate with step 0.1.
- Target: K=16 recall >= 0.750 and oracle regret <= 0.250.

Evidence:

- Report: `reports/crt_wide32_r16_prefilter_blend_eval.json`
- Human summary: `reports/crt_wide32_r16_prefilter_blend_eval_summary.md`

Result:

- Selected train weights: relation 0.5, vanilla 0.3, MLP 0.0, immediate 0.2.
- Train K=16: recall 0.7983, mean oracle regret 0.1045.
- Validation K=16: recall 0.7133, mean oracle regret 0.1471.
- Validation K=24: recall 0.9033, mean oracle regret 0.0371.

Decision gate:

- The selected blend passes top-16 on train but fails on validation.
- K=24 remains the smallest validated serving width.

Interpretation: simple normalized score blending overfits the train shard and
does not solve the held-out top-16 recall gap. The next credible top-16 path is
more data/stronger structure/recall-specific modeling, not post-hoc blending.

## 2026-06-30 - `crt-wide32-r16-top16-margin-relation-bias-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails.

Purpose: retest the top-16-focused pairwise/listwise objective on the cleaner
R16 sampled-teacher labels. This checks whether the objective failed earlier
because the R8 labels were too noisy.

Data:

- Train: `fixtures/crt_wide32_r16_sampled_teacher_train.jsonl`, 1200 roots,
  32 retained actions/root.
- Validation: `fixtures/crt_wide32_r16_sampled_teacher_val.jsonl`, 300 roots,
  32 retained actions/root.
- Labels: 16 rollout samples per retained action.
- No new roots generated.

Implementation:

- Runner: `scripts/run_crt_wide32_r16_top16_margin_relation_bias_pilot.sh`.
- Loss mode: `top16-prefilter`.
- Model: 4-layer relation-bias Transformer, hidden size 256, 8 heads, MLP 512.
- Training: 6200 steps, LR 0.00025.

Evidence:

- Training report:
  `reports/crt_wide32_r16_top16_margin_relation_bias_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16_top16_margin_relation_bias_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16_top16_margin_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16_top16_margin_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16_top16_margin_relation_bias_pilot_summary.md`

Result:

- relation-bias point metrics: top-1 0.1267, mean regret 1.6456.
- relation-bias K=8: recall 0.4467, mean oracle regret 0.4671.
- relation-bias K=16: recall 0.6633, mean oracle regret 0.2110.
- relation-bias K=24: recall 0.8567, mean oracle regret 0.0681.
- `has_merit=false`: relation-bias beats immediate and same-run vanilla, but not
  the same-run MLP baseline.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 16 unit/schema tests
  passed.

Decision gate:

- Top-16 fails by recall: 0.6633 < 0.750.
- Top-24 passes but is worse than the R16 standard model.
- The R16 standard objective remains the best tested R16 relation-bias
  prefilter.

Interpretation: the teacher-best margin/listwise objective regressed held-out
prefilter quality even on cleaner R16 labels. More objective shaping of this
form is lower priority than more data, better structure, or a different
recall-specific training target.

## 2026-06-30 - `crt-wide32-r16x2-sampled-teacher-relation-bias-v1`

Status: completed, `has_merit=true`, top-16 serving gate still fails by recall.

Purpose: test whether the remaining R16 top-16 recall miss was mostly
small-shard variance by doubling both the train and validation root counts while
keeping the standard relation-bias objective.

Data:

- Train: `fixtures/crt_wide32_r16x2_sampled_teacher_train.jsonl`, 2400 roots,
  32 retained actions/root.
- Validation: `fixtures/crt_wide32_r16x2_sampled_teacher_val.jsonl`, 600 roots,
  32 retained actions/root.
- Labels: 16 rollout samples per retained action.
- Train rollout samples: 1228800.
- Validation rollout samples: 307200.
- Truncation: 2 train action labels, 0 validation action labels.

Implementation:

- Runner:
  `scripts/run_crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.sh`.
- Loss mode: `standard`.
- Model: 4-layer relation-bias Transformer, hidden size 256, 8 heads, MLP 512.
- Training: 7600 steps, LR 0.00032.

Evidence:

- Training report:
  `reports/crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16x2_sampled_teacher_relation_bias_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16x2_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16x2_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16x2_sampled_teacher_relation_bias_pilot_summary.md`

Result:

- relation-bias point metrics: top-1 0.1283, mean regret 1.5090.
- relation-bias K=8: recall 0.5283, mean oracle regret 0.3471.
- relation-bias K=16: recall 0.7400, mean oracle regret 0.1475.
- relation-bias K=24: recall 0.9083, mean oracle regret 0.0448.
- Same-run vanilla public-token Transformer: top-1 0.1267, mean regret 1.5338,
  K=16 recall 0.7400, K=16 oracle regret 0.1458, K=24 recall 0.8817.
- Same-run MLP: top-1 0.1133, mean regret 1.5916, K=16 recall 0.7250.
- Immediate-score baseline: top-1 0.0717, mean regret 1.9894, K=16 recall
  0.5817.
- `has_merit=true`: relation-bias beats immediate, MLP, and same-run vanilla on
  the configured point/regret criterion.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 16 unit/schema tests
  passed before training and before serving replay.

Decision gate:

- Top-16 fails by recall: 0.7400 < 0.750.
- Top-16 oracle regret passes: 0.1475 <= 0.250.
- Top-24 passes strongly: 0.9083 recall and 0.0448 oracle regret.
- K=16 misses by 6 teacher-best actions on the 600-root validation split.

Interpretation: doubling data moved K=16 recall in the right direction
(`0.7233 -> 0.7400`) and produced the first clean positive relation-bias merit
decision at this scale, but it did not validate K=16 serving. K=24 remains the
smallest defensible serving width. The next top-16 attempt should add richer
action-conditioned structure rather than rely on another simple objective
variant or normalized score blend.

## 2026-06-30 - `crt-wide32-r16x2-semantic-relation-bias-v1`

Status: completed, `has_merit=true`, top-16 serving gate passes on the opening
R16x2 shard.

Purpose: test whether richer action-conditioned structure closes the remaining
K=16 recall gap. This keeps the public-token/relation-bias encoder but widens
each action from 33 raw/public features to 61 features by appending semantic
signals for habitat matching, drafted-species market/supply context, opponent
species count, and Card-A wildlife pattern opportunities.

Implementation:

- Feature/training module:
  `src/cascadiav3/torch_semantic_relation_bias_merit.py`.
- Backward-compatible collate hook:
  `src/cascadiav3/torch_public_token_merit.py`.
- Semantic-aware checkpoint replay:
  `src/cascadiav3/torch_prefilter_eval.py`.
- Runner:
  `scripts/run_crt_wide32_r16x2_semantic_relation_bias_pilot.sh`.
- Action feature dim: 61 = 33 base + 28 semantic.
- Data: reused `crt_wide32_r16x2_sampled_teacher_{train,val}.jsonl`,
  2400 train roots and 600 validation roots, 32 actions/root, 16 rollout
  samples/action.

Evidence:

- Training report:
  `reports/crt_wide32_r16x2_semantic_relation_bias_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16x2_semantic_relation_bias_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16x2_semantic_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16x2_semantic_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16x2_semantic_relation_bias_pilot_summary.md`

Result:

- Semantic relation-bias top-1: 0.1317, mean regret 1.5082.
- Semantic relation-bias K=16: recall 0.7767, oracle regret 0.1233.
- Semantic relation-bias K=24: recall 0.9433, oracle regret 0.0304.
- Semantic vanilla Transformer K=16: recall 0.7533, oracle regret 0.1422.
- Semantic MLP K=16: recall 0.7550, oracle regret 0.1320.
- Immediate-score K=16 remains poor: recall 0.5817, oracle regret 0.3163.
- `has_merit=true`: semantic relation-bias beats immediate, MLP, and same-run
  vanilla on the configured point/regret criterion.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 17 unit/schema tests
  passed before training and before serving replay.

Decision gate:

- Top-16 passes: 0.7767 recall >= 0.750, 0.1233 oracle regret <= 0.250.
- Top-24 passes strongly: 0.9433 recall and 0.0304 oracle regret.
- Recommended serving width on this shard: K=16.

Caveat and next action: this is still an opening-root shard because
`PLIES_PER_SEED=4`; many wildlife-adjacency semantic features remain zero. The
next credible test is a phase-diverse semantic run with deeper plies per seed
so bear/elk/salmon/hawk/fox pattern features are actually exercised.

## 2026-06-30 - `crt-wide32-r16p20-semantic-relation-bias-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails on a
phase-diverse deeper-ply shard.

Purpose: retest the semantic action-conditioned relation-bias architecture on
roots that are not just opening positions. This preserves the 32-action,
16-rollout/action setup but generates 20 plies per seed so wildlife adjacency
and Card-A pattern features are actually active.

Data:

- Train: `fixtures/crt_wide32_r16p20_semantic_train.jsonl`, 2400 roots,
  32 actions/root.
- Validation: `fixtures/crt_wide32_r16p20_semantic_val.jsonl`, 600 roots,
  32 actions/root.
- Labels: 16 rollout samples per retained action, top-4 sampled greedy
  continuation policy.
- Generation: `120` train seeds and `30` validation seeds, `20` plies/seed.
- Validation semantic feature activity: `wildlife_adjacent_any_wildlife_count`
  mean `0.1780`, `wildlife_adjacent_same_species_count` mean `0.0395`,
  `fox_unique_adjacent_species_count` mean `0.0552`, `bear_pair_signal` mean
  `0.0290`, `hawk_line_of_sight_count` mean `0.0068`.

Implementation:

- Feature/training module:
  `src/cascadiav3/torch_semantic_relation_bias_merit.py`.
- Runner:
  `scripts/run_crt_wide32_r16p20_semantic_relation_bias_pilot.sh`.
- Action feature dim: 61 = 33 base + 28 semantic.

Evidence:

- Training report:
  `reports/crt_wide32_r16p20_semantic_relation_bias_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_relation_bias_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16p20_semantic_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16p20_semantic_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_relation_bias_pilot_summary.md`

Result:

- Semantic relation-bias top-1: 0.0983, mean regret 1.6415.
- Semantic relation-bias K=16: recall 0.6867, oracle regret 0.1825.
- Semantic relation-bias K=24: recall 0.8617, oracle regret 0.0688.
- Same-run vanilla Transformer K=16: recall 0.7300, oracle regret 0.1556.
- Same-run vanilla Transformer K=24: recall 0.8867, oracle regret 0.0565.
- Same-run token-pooled MLP K=16: recall 0.7150, oracle regret 0.1615.
- Same-run token-pooled MLP K=24: recall 0.8683, oracle regret 0.0659.
- Immediate-score K=16: recall 0.6117, oracle regret 0.2715.
- Immediate-score K=24: recall 0.8117, oracle regret 0.1140.
- `has_merit=false`: relation-bias fails the configured merit gate and
  underperforms the same-run vanilla Transformer and token-pooled MLP.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 17 unit/schema tests
  passed before training and before serving replay.

Decision gate:

- Top-16 fails by recall: 0.6867 < 0.750.
- Top-16 oracle regret passes: 0.1825 <= 0.250.
- Top-24 passes: 0.8617 recall and 0.0688 oracle regret.
- Recommended serving width on phase-diverse roots remains K=24.

Interpretation: the opening-shard semantic K=16 pass did not generalize to
deeper roots. The deeper shard is more credible because semantic pattern
features are nonzero, and on that split the relation-bias structure is not the
best tested representation. The next GPU run should change the learning target
or phase conditioning rather than simply re-run the same relation-bias recipe.

## 2026-06-30 - `crt-wide32-r16p20-semantic-topk-retention-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails.

Purpose: replace scalar-Q-first training with a direct K=16 retention objective.
The loss pushes the teacher top-16 action set above the bottom-16 set, with
auxiliary Q regression and policy losses.

Implementation:

- Reusable loss mode: `topk-retention` in
  `src/cascadiav3/torch_relation_bias_merit.py`.
- Runner:
  `scripts/run_crt_wide32_r16p20_semantic_retention_pilot.sh`.
- Replay support: existing semantic checkpoint replay path.

Evidence:

- Training report:
  `reports/crt_wide32_r16p20_semantic_retention_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_retention_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16p20_semantic_retention_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16p20_semantic_retention_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_retention_pilot_summary.md`

Result:

- Relation-bias K=16: recall 0.6967, oracle regret 0.1864.
- Relation-bias K=24: recall 0.8883, oracle regret 0.0493.
- Same-run vanilla Transformer K=16: recall 0.6867, oracle regret 0.1955.
- Same-run vanilla Transformer K=24: recall 0.8767, oracle regret 0.0676.
- Same-run token-pooled MLP K=16: recall 0.7417, oracle regret 0.1469.
- Same-run token-pooled MLP K=24: recall 0.8767, oracle regret 0.0593.
- Immediate-score K=16: recall 0.6117, oracle regret 0.2715.
- `has_merit=false`: relation-bias beats same-run vanilla but does not beat
  immediate/MLP under the configured merit gate.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 17 unit/schema tests
  passed before training and serving replay.

Decision gate:

- Transformer K=16 still fails by recall: 0.6967 < 0.750.
- Transformer K=24 passes: 0.8883 recall and 0.0493 oracle regret.
- Best same-run K=16 signal is the MLP at 0.7417, five teacher-best hits short
  of the 0.750 recall gate on 600 roots.

Interpretation: direct retention supervision is useful but not sufficient for
the current relation-bias Transformer. The near-pass MLP says action-conditioned
semantic features carry most of the current phase-diverse signal; attention over
public tokens is not yet helping enough.

## 2026-06-30 - `crt-wide32-r16p20-semantic-cross-attention-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails.

Purpose: test a cleaner action-query decoder from the architecture proposal:
encode public state tokens once, then let each legal action query cross-attend
to that state without action-action self-attention.

Implementation:

- Model/training module:
  `src/cascadiav3/torch_semantic_cross_attention_merit.py`.
- Checkpoint replay support:
  `src/cascadiav3/torch_prefilter_eval.py`.
- Runner:
  `scripts/run_crt_wide32_r16p20_semantic_cross_attention_pilot.sh`.
- Loss mode: `topk-retention`.

Evidence:

- Training report:
  `reports/crt_wide32_r16p20_semantic_cross_attention_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_cross_attention_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16p20_semantic_cross_attention_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16p20_semantic_cross_attention_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_cross_attention_pilot_summary.md`

Result:

- Cross-attention Transformer K=16: recall 0.6733, oracle regret 0.2136.
- Cross-attention Transformer K=24: recall 0.8600, oracle regret 0.0697.
- Same-run vanilla Transformer K=16: recall 0.7233, oracle regret 0.1509.
- Same-run vanilla Transformer K=24: recall 0.8833, oracle regret 0.0596.
- Same-run token-pooled MLP K=16: recall 0.7350, oracle regret 0.1531.
- Same-run token-pooled MLP K=24: recall 0.8767, oracle regret 0.0570.
- Immediate-score K=16: recall 0.6117, oracle regret 0.2715.
- `has_merit=false`: cross-attention underperforms immediate on point
  selection, same-run vanilla, and same-run MLP.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 18 unit/schema tests
  passed before training and serving replay.

Decision gate:

- Cross-attention K=16 fails by recall: 0.6733 < 0.750.
- Cross-attention K=24 passes: 0.8600 recall and 0.0697 oracle regret.
- Decision: K=24 remains the only validated serving width on phase-diverse
  roots.

Interpretation: the Perceiver-style cross-attention decoder did not fix the
phase-diverse failure. The current next best direction is not a larger version
of this decoder; it is forensic analysis of the MLP near-miss and better
state/action features or calibration that preserve the MLP signal while adding
public-token context only where it helps.

## 2026-06-30 - `crt-wide32-r16p20-semantic-residual-attention-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails.

Purpose: test whether an MLP-anchored residual attention branch can preserve
the strong semantic MLP signal while allowing public-token context to add only
an incremental correction. This starts from the semantic token-pooled MLP action
score, encodes public state tokens, lets each legal action cross-attend to that
state, and adds a scaled residual score.

Implementation:

- Model/training module:
  `src/cascadiav3/torch_semantic_residual_attention_merit.py`.
- Checkpoint replay support:
  `src/cascadiav3/torch_prefilter_eval.py`.
- Blend replay support:
  `src/cascadiav3/torch_prefilter_blend_eval.py`.
- Runner:
  `scripts/run_crt_wide32_r16p20_semantic_residual_attention_pilot.sh`.
- Loss mode: `topk-retention`.
- Residual scale: `0.25`.

Evidence:

- Training report:
  `reports/crt_wide32_r16p20_semantic_residual_attention_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_residual_attention_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16p20_semantic_residual_attention_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16p20_semantic_residual_attention_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_residual_attention_pilot_summary.md`

Result:

- Residual-attention Transformer K=16: recall 0.7350, oracle regret 0.1502.
- Residual-attention Transformer K=24: recall 0.8683, oracle regret 0.0649.
- Same-run vanilla Transformer K=16: recall 0.7267, oracle regret 0.1595.
- Same-run vanilla Transformer K=24: recall 0.8733, oracle regret 0.0672.
- Same-run token-pooled MLP K=16: recall 0.7383, oracle regret 0.1473.
- Same-run token-pooled MLP K=24: recall 0.8800, oracle regret 0.0590.
- Immediate-score K=16: recall 0.6117, oracle regret 0.2715.
- `has_merit=false`: residual-attention improves over vanilla at K=16 but does
  not beat the same-run MLP and fails the top-16 recall gate.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 19 unit/schema tests
  passed before training and serving replay.

Decision gate:

- Residual-attention K=16 fails by recall: 0.7350 < 0.750.
- Residual-attention K=24 passes: 0.8683 recall and 0.0649 oracle regret.
- Decision: K=24 remains the only validated serving width on phase-diverse
  roots.

Interpretation: anchoring on the MLP helped avoid the severe cross-attention
regression, but the residual branch still did not add enough held-out signal to
beat the MLP. This is a useful negative architecture result: preserving the MLP
signal is necessary, but not sufficient.

## 2026-06-30 - `crt-wide32-r16p20-semantic-residual-attention-blend-eval-v1`

Status: completed, validation top-16 serving gate still fails.

Purpose: test whether replay-only score calibration over residual, vanilla,
MLP, and immediate scores can close the remaining top-16 recall gap on the
phase-diverse R16p20 shard.

Implementation:

- Evaluator: `src/cascadiav3/torch_prefilter_blend_eval.py`.
- Runner: `scripts/run_crt_wide32_r16_prefilter_blend_eval.sh`.
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_residual_attention_pilot.pt`.
- Grid: simplex weights over residual/vanilla/MLP/immediate with step 0.05.
- Target: K=16 recall >= 0.750 and oracle regret <= 0.250.

Evidence:

- Report:
  `reports/crt_wide32_r16p20_semantic_residual_attention_blend_eval.json`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_residual_attention_blend_eval_summary.md`

Result:

- Selected train weights: residual 0.0, vanilla 0.75, MLP 0.1, immediate 0.15.
- Train K=16: recall 0.7892, mean oracle regret 0.1114.
- Validation K=16: recall 0.7333, mean oracle regret 0.1521.
- Validation K=24: recall 0.8783, mean oracle regret 0.0632.

Decision gate:

- The selected blend passes top-16 on train but fails on validation.
- The residual branch receives zero selected weight.
- K=24 remains the smallest validated serving width.

Interpretation: simple calibration still overfits train and does not recover
held-out K=16 recall. The zero residual weight is especially informative: the
trained residual-attention branch is not contributing useful signal under this
serving blend test.

## 2026-06-30 - `crt-wide32-r16p20-semantic-action-set-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails.

Purpose: test whether the current MLP near-miss is missing action-set context
rather than public-token context. This model pools public tokens into one root
context token, then applies Transformer self-attention only across the legal
action set. It is a Set Transformer-style ranking model over the 32 retained
actions.

Implementation:

- Model/training module:
  `src/cascadiav3/torch_semantic_action_set_merit.py`.
- Checkpoint replay support:
  `src/cascadiav3/torch_prefilter_eval.py`.
- Blend replay support:
  `src/cascadiav3/torch_prefilter_blend_eval.py`.
- Runner:
  `scripts/run_crt_wide32_r16p20_semantic_action_set_pilot.sh`.
- Loss mode: `topk-retention`.

Evidence:

- Training report:
  `reports/crt_wide32_r16p20_semantic_action_set_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16p20_semantic_action_set_prefilter_eval.json`
- Per-root retained action sets:
  `reports/crt_wide32_r16p20_semantic_action_set_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_action_set_pilot_summary.md`

Result:

- Action-set Transformer K=16: recall 0.7133, oracle regret 0.1628.
- Action-set Transformer K=24: recall 0.8867, oracle regret 0.0583.
- Same-run vanilla Transformer K=16: recall 0.7300, oracle regret 0.1590.
- Same-run vanilla Transformer K=24: recall 0.8733, oracle regret 0.0636.
- Same-run token-pooled MLP K=16: recall 0.7417, oracle regret 0.1456.
- Same-run token-pooled MLP K=24: recall 0.8667, oracle regret 0.0654.
- Immediate-score K=16: recall 0.6117, oracle regret 0.2715.
- `has_merit=false`: action-set does not beat immediate, same-run vanilla, or
  same-run MLP under the configured merit gate.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 20 unit/schema tests
  passed before training and serving replay.

Decision gate:

- Action-set K=16 fails by recall: 0.7133 < 0.750.
- Action-set K=24 passes: 0.8867 recall and 0.0583 oracle regret.
- Decision: K=24 remains the only validated serving width on phase-diverse
  roots.

Interpretation: legal-action self-attention is not enough to fix the top-16
recall problem. The MLP still carries the best phase-diverse K=16 signal. The
next useful step is miss-set forensics and/or richer target/features, not a
larger action-set Transformer.

## 2026-06-30 - `crt-wide32-r16p20-semantic-action-set-blend-eval-v1`

Status: completed, validation top-16 serving gate still fails.

Purpose: test whether replay-only score calibration over action-set, vanilla,
MLP, and immediate scores can recover K=16 recall even though the action-set
model underperforms alone.

Implementation:

- Evaluator: `src/cascadiav3/torch_prefilter_blend_eval.py`.
- Runner: `scripts/run_crt_wide32_r16_prefilter_blend_eval.sh`.
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt`.
- Grid: simplex weights over action_set/vanilla/MLP/immediate with step 0.05.
- Target: K=16 recall >= 0.750 and oracle regret <= 0.250.

Evidence:

- Report:
  `reports/crt_wide32_r16p20_semantic_action_set_blend_eval.json`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_action_set_blend_eval_summary.md`

Result:

- Selected train weights: action_set 0.55, vanilla 0.20, MLP 0.25,
  immediate 0.00.
- Train K=16: recall 0.8042, mean oracle regret 0.0978.
- Validation K=16: recall 0.7350, mean oracle regret 0.1489.
- Validation K=24: recall 0.8933, mean oracle regret 0.0582.

Decision gate:

- The selected blend passes top-16 on train but fails on validation.
- K=24 remains the smallest validated serving width.

Interpretation: the action-set branch has some complementary train signal, but
it does not generalize into a K=16 serving pass. The repeated pattern is now
clear: K=24 is robust, while K=16 needs better data/features rather than more
Transformer wrapper shapes around the same labels.

## 2026-06-30 - `crt-wide32-r16p20-semantic-prefilter-forensics-v1`

Status: completed.

Purpose: explain the held-out K=16 miss sets from the latest phase-diverse
action-set checkpoint. This replays the action_set, vanilla, MLP, and immediate
sources from the same checkpoint and compares which validation roots each source
misses.

Implementation:

- Analyzer: `src/cascadiav3/torch_prefilter_forensics.py`.
- Runner: `scripts/run_crt_wide32_r16p20_prefilter_forensics.sh`.
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt`.

Evidence:

- Report:
  `reports/crt_wide32_r16p20_semantic_prefilter_forensics.json`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_prefilter_forensics_summary.md`

Result:

- MLP K=16 recall: 0.7417, misses 155, needs 5 more hits for the 0.750 gate.
- Action-set K=16 recall: 0.7133, misses 172, needs 22 more hits.
- Vanilla K=16 recall: 0.7300, misses 162, needs 12 more hits.
- Immediate K=16 recall: 0.6117, misses 233, needs 83 more hits.
- MLP misses recovered by other sources: action_set 48, vanilla 34,
  immediate 72.
- Consensus misses across all four sources: 48 roots.
- Largest MLP misses are mainly early/mid phase roots involving elk, hawk,
  salmon, and fox teacher-best actions with teacher-best ranks often in the
  high teens through low thirties.
- Remote verification on `john0`: RTX 5090 visible, Torch `2.11.0+cu128`,
  CUDA 12.8, driver 591.86, and 20 unit/schema tests passed before replay.

Interpretation: the miss sets are complementary, so there is signal outside the
MLP. But the overlap and train/validation instability mean that any learned or
handwritten gate must be validated strictly. This motivates source-union and
future learned-gating tests, but does not itself validate K=16 serving.

## 2026-06-30 - `crt-wide32-r16p20-semantic-source-union-prefilter-v1`

Status: completed, validation top-16 serving gate fails.

Purpose: test whether the complementary miss sets exposed by forensics can be
used by a simple train-selected K=16 source-quota union. The rule chooses source
quotas on the train shard, then evaluates the selected rule once on held-out
validation.

Implementation:

- Evaluator: `src/cascadiav3/torch_prefilter_union_eval.py`.
- Runner: `scripts/run_crt_wide32_r16p20_source_union_prefilter.sh`.
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt`.
- Candidate rule: union top-N rankings from action_set, vanilla, MLP, and
  immediate sources, then fill to K=16.

Evidence:

- Report:
  `reports/crt_wide32_r16p20_semantic_source_union_prefilter.json`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_source_union_prefilter_summary.md`

Result:

- Selected train quotas: action_set 15, vanilla 1, MLP 0, immediate 0.
- Fill source: action_set.
- Train K=16: recall 0.8004, mean oracle regret 0.1032.
- Validation K=16: recall 0.7233, mean oracle regret 0.1556.
- Remote verification on `john0`: RTX 5090 visible, Torch `2.11.0+cu128`,
  CUDA 12.8, driver 591.86, and 20 unit/schema tests passed before replay.

Decision gate:

- Validation K=16 fails by recall: 0.7233 < 0.750.
- This is worse than the same-run MLP's validation recall of 0.7417.

Interpretation: source complementarity is real, but naive quota selection
overfits the train shard and regresses validation. The next credible move is
not a more flexible train-selected combiner on the same split; it is either more
diverse train/validation roots, a cross-validated gating setup, or features
that directly target the consensus misses.

## 2026-06-30 - `crt-wide32-r16p20-semantic-learned-source-gate-v1`

Status: completed, validation top-16 serving gate fails.

Purpose: test the strict learned-gating version of the source-complementarity
idea. The gate uses only serving-safe per-action source score/rank features from
the action_set, vanilla, MLP, and immediate sources. It trains on the training
shard, chooses the best checkpoint by an inner tune split, and evaluates once on
the held-out validation shard.

Implementation:

- Evaluator: `src/cascadiav3/torch_prefilter_gate_eval.py`.
- Runner: `scripts/run_crt_wide32_r16p20_learned_source_gate.sh`.
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_action_set_pilot.pt`.
- Features: per-source z-score, rank score, top-4/top-8/top-16 flags, plus
  aggregate source mean/max/min/std/range and top-16 vote count.
- Model: two-layer MLP gate, hidden size 64, dropout 0.10.
- Objective: top-16 membership BCE plus top-16-vs-rest pairwise retention loss.

Evidence:

- Report:
  `reports/crt_wide32_r16p20_semantic_learned_source_gate.json`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_learned_source_gate_summary.md`

Result:

- Fit split K=16: recall 0.7948, mean oracle regret 0.1069.
- Tune split K=16: recall 0.8229, mean oracle regret 0.1014.
- Held-out validation K=16: recall 0.7250, mean oracle regret 0.1644.
- Held-out source baselines: action_set recall 0.7133, vanilla 0.7300, MLP
  0.7417, immediate 0.6117.
- Remote verification on `john0`: RTX 5090 visible, Torch `2.11.0+cu128`,
  CUDA 12.8, driver 591.86, and 20 unit/schema tests passed before training.

Decision gate:

- Validation K=16 fails by recall: 0.7250 < 0.750.
- The learned gate also underperforms the same-run token-pooled MLP's validation
  recall of 0.7417.

Interpretation: a flexible source gate can exploit the training distribution,
but it does not transfer to held-out phase-diverse roots. This is the second
post-hoc combiner failure after source-union, so K=16 should not be pursued via
more train-selected source mixing on this shard. The next credible branch is
larger/diverser roots or new features that attack the consensus miss set
directly.

## 2026-06-30 - `crt-wide32-r16p20-semantic-species-moe-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails.

Purpose: attack the wildlife-specific phase-diverse miss pattern directly.
Forensics showed many large MLP misses on elk, hawk, and salmon teacher-best
actions. This model keeps the semantic relation-bias encoder but adds
per-wildlife residual scoring heads for no-wildlife, bear, elk, salmon, hawk,
and fox actions.

Implementation:

- Model/evaluator: `src/cascadiav3/torch_semantic_species_moe_merit.py`.
- Runner: `scripts/run_crt_wide32_r16p20_semantic_species_moe_pilot.sh`.
- Data: reused `fixtures/crt_wide32_r16p20_semantic_{train,val}.jsonl`,
  2400 train roots and 600 validation roots, 32 actions/root, 16 rollout
  samples/action.
- Model: 4-layer semantic relation-bias encoder, hidden size 256, 8 heads, MLP
  512, plus species embeddings and species residual Q/policy heads.
- Objective: standard scalar-Q/listwise/pairwise recipe.

Evidence:

- Report:
  `reports/crt_wide32_r16p20_semantic_species_moe_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_species_moe_pilot.pt`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_species_moe_pilot_summary.md`

Result:

- Species-MoE top-1: 0.1217, mean regret 1.5729.
- Species-MoE K=16: recall 0.7033, oracle regret 0.1715.
- Species-MoE K=24: recall 0.8850, oracle regret 0.0558.
- Same-run vanilla Transformer K=16: recall 0.7317, oracle regret 0.1536.
- Same-run token-pooled MLP K=16: recall 0.7367, oracle regret 0.1445.
- Immediate-score K=16: recall 0.6117, oracle regret 0.2715.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 21 unit/schema tests
  passed before training.

Decision gate:

- Top-16 fails by recall: 0.7033 < 0.750.
- `has_merit=false`: point regret improved 9.3% versus immediate, just short of
  the 10% gate, and K=16 retention regressed versus both same-run vanilla and
  MLP.

Interpretation: species-specific residual heads slightly help scalar action
selection but make K=16 retention worse. The miss pattern is not solved by
per-species head capacity alone.

## 2026-06-30 - `crt-wide32-r16p20-semantic-species-moe-retention-v1`

Status: completed, `has_merit=false`, top-16 serving gate fails.

Purpose: retest the species-MoE architecture with direct top-K retention
supervision, since the standard objective improved scalar selection but
regressed top-16 retention.

Implementation:

- Model/evaluator: `src/cascadiav3/torch_semantic_species_moe_merit.py`.
- Runner: `scripts/run_crt_wide32_r16p20_semantic_species_moe_pilot.sh` with
  `LOSS_MODE=topk-retention`.
- Loss weights: Q 0.15, top-K policy 0.25, retention 1.5, margin 0.15,
  policy temperature 0.75.
- Same phase-diverse R16p20 train/validation shard as the standard species-MoE
  run.

Evidence:

- Report:
  `reports/crt_wide32_r16p20_semantic_species_moe_retention_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p20_semantic_species_moe_retention_pilot.pt`
- Human summary:
  `reports/crt_wide32_r16p20_semantic_species_moe_retention_pilot_summary.md`

Result:

- Species-MoE retention top-1: 0.1083, mean regret 1.5624.
- Species-MoE retention K=16: recall 0.7133, oracle regret 0.1610.
- Species-MoE retention K=24: recall 0.8683, oracle regret 0.0703.
- Same-run vanilla Transformer K=16: recall 0.7417, oracle regret 0.1461.
- Same-run token-pooled MLP K=16: recall 0.7317, oracle regret 0.1473.
- Immediate-score K=16: recall 0.6117, oracle regret 0.2715.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 21 unit/schema tests
  passed before training.

Decision gate:

- Top-16 fails by recall: 0.7133 < 0.750.
- The same-run vanilla Transformer again reaches the best K=16 recall in this
  run at 0.7417, still five hits short of the gate.

Interpretation: direct retention supervision does not rescue the species-MoE
architecture. At this point, K=16 on phase-diverse roots is unlikely to come
from another small head/routing variant. The next credible branch should be new
training data or labels: larger/diverser roots, stronger action labels, or a
curriculum that over-samples the consensus miss state/action patterns without
using the validation shard as training data.

## 2026-06-30 - `crt-wide32-r16p80-semantic-relation-bias-v1`

Status: completed, `has_merit=false`, top-16 transformer serving gate fails.

Purpose: correct the phase coverage mistake in the R16p20 run. R16p20 gave
2400 roots, but only active tile counts 3-7. This run keeps the same 2400-root
train scale while using 80 plies/seed so each active player is represented from
tile count 3 through tile count 22.

Implementation:

- Runner: `scripts/run_crt_wide32_r16p80_semantic_relation_bias_detached.sh`.
- Data: generated fresh all-phase `fixtures/crt_wide32_r16p80_semantic_train.jsonl`
  and `fixtures/crt_wide32_r16p80_semantic_val.jsonl` on `john0` CPU.
- Train roots: 2400, 32 actions/root, 16 rollout samples/action.
- Validation roots: 640, 32 actions/root, 16 rollout samples/action.
- Phase coverage: active tile counts 3-22, 120 train roots and 32 validation
  roots per active tile count.
- Model: 4-layer semantic relation-bias Transformer, hidden size 256, 8 heads,
  MLP 512, 61 action features, standard objective.
- Detached job artifacts: `logs/r16p80_semantic_relation_bias_job.log`.

Evidence:

- Report:
  `reports/crt_wide32_r16p80_semantic_relation_bias_pilot.json`
- Checkpoint:
  `checkpoints/crt_wide32_r16p80_semantic_relation_bias_pilot.pt`
- Serving eval:
  `reports/crt_wide32_r16p80_semantic_prefilter_eval.json`
- Per-root serving eval:
  `reports/crt_wide32_r16p80_semantic_prefilter_eval_roots.jsonl`
- Human summary:
  `reports/crt_wide32_r16p80_semantic_relation_bias_pilot_summary.md`

Result:

- Semantic relation-bias top-1: 0.1375, mean regret 1.2553.
- Semantic relation-bias K=16: recall 0.7125, oracle regret 0.1482.
- Semantic relation-bias K=24: recall 0.8563, oracle regret 0.0623.
- Same-run vanilla Transformer K=16: recall 0.7000, oracle regret 0.1598.
- Same-run token-pooled MLP K=16: recall 0.7609, oracle regret 0.0995.
- Immediate-score K=16: recall 0.6562, oracle regret 0.2164.
- Same-run token-pooled MLP K=24: recall 0.9047, oracle regret 0.0321.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 21 unit/schema tests
  passed before training.

Decision gate:

- Top-16 fails for the transformer by recall: 0.7125 < 0.750.
- Top-24 passes for the transformer: recall 0.8563 and oracle regret 0.0623.
- `has_merit=false`: relation-bias beats same-run vanilla slightly, but loses
  to immediate score and the same-run MLP.
- The token-pooled MLP is the only K=16 pass on this all-phase run.

Interpretation: the transformer did not regain the opening-shard K=16 signal
once the data covered the whole active-player game arc. The surprising and
useful result is that the cheap semantic MLP clears K=16 on all-phase roots.
The next branch should treat that as evidence about representation: either use
the MLP as the first serving prefilter candidate, or add explicit phase/scalar
conditioning so the public-token transformer can match what the dense semantic
features are already capturing.

## 2026-06-30 - `crt-wide32-r16p80-semantic-residual-attention-v1`

Status: completed, `has_merit=false`, top-16 serving gate is borderline by
single seed and passes with fixed 3-seed ensembling.

Purpose: test the representation lesson from the all-phase relation-bias run.
The token-pooled semantic MLP could clear K=16, while public-token attention
damaged the signal. This architecture keeps the semantic MLP as an anchor and
adds a bounded public-token cross-attention residual.

Implementation:

- Model/evaluator: `src/cascadiav3/torch_semantic_residual_attention_merit.py`.
- Seed-ensemble evaluator:
  `src/cascadiav3/torch_prefilter_seed_ensemble_eval.py`.
- Runner:
  `scripts/run_crt_wide32_r16p80_semantic_residual_attention_pilot.sh`.
- Seed-ensemble runner:
  `scripts/run_crt_wide32_r16p80_residual_seed_ensemble_eval.sh`.
- Data: reused `fixtures/crt_wide32_r16p80_semantic_{train,val}.jsonl`.
- Train roots: 2400, 32 actions/root, 16 rollout samples/action.
- Validation roots: 640, 32 actions/root, 16 rollout samples/action.
- Phase coverage: active tile counts 3-22, 120 train roots and 32 validation
  roots per active tile count.
- Model: 4-layer semantic residual-attention Transformer, hidden size 256,
  8 heads, MLP 512, residual scale 0.25, top-k-retention objective.

Evidence:

- Seed 20260630 report:
  `reports/crt_wide32_r16p80_semantic_residual_attention_pilot.json`
- Seed 20260631 report:
  `reports/crt_wide32_r16p80_semantic_residual_attention_seed31_pilot.json`
- Seed 20260632 report:
  `reports/crt_wide32_r16p80_semantic_residual_attention_seed32_pilot.json`
- Fixed 3-seed ensemble report:
  `reports/crt_wide32_r16p80_residual_seed_ensemble_3x_eval.json`
- Fixed 3-seed ensemble summary:
  `reports/crt_wide32_r16p80_residual_seed_ensemble_3x_eval_summary.md`
- Seed-sweep summary:
  `reports/crt_wide32_r16p80_semantic_residual_attention_seed_sweep_summary.md`

Result:

- Seed 20260630 residual K=16: recall 0.7531, oracle regret 0.1039.
- Seed 20260631 residual K=16: recall 0.7484, oracle regret 0.1021.
- Seed 20260632 residual K=16: recall 0.7500, oracle regret 0.1010.
- Fixed 2-seed residual ensemble K=16: recall 0.7531, oracle regret 0.1012.
- Fixed 3-seed residual ensemble K=16: recall 0.7578, oracle regret 0.0999.
- Fixed 3-seed residual ensemble K=24: recall 0.9156, oracle regret 0.0265.
- Vanilla public-token Transformer single-seed K=16 recalls stayed far lower:
  0.6734, 0.6734, and 0.6984.
- Same-run MLP K=16 varied around the threshold: 0.7375, 0.7484, and 0.7344.
- Remote verification on `john0`: exporter tests passed, RTX 5090 visible,
  Torch `2.11.0+cu128`, CUDA 12.8, driver 591.86, and 21 unit/schema tests
  passed before every training/eval run.

Decision gate:

- Single-seed residual-attention is promising but not robust enough by itself:
  one pass, one miss by one hit, and one exact-threshold pass.
- Fixed 3-seed residual-attention ensemble passes the K=16 serving prefilter
  gate with recall 0.7578 and oracle regret 0.0999.
- `has_merit=false` remains true under the scalar/top-1 offline merit gate; the
  result is specifically a top-K serving prefilter signal.

Interpretation: the MLP anchor was the missing stabilizer. Public-token
attention alone still underperforms, but bounded residual attention on top of
dense semantic features can recover enough complementary misses to make K=16
plausible on all-phase roots. The next credible step is not another small head
variant; it is a larger/diverser all-phase shard, then a search-prefilter trial
that measures actual downstream score impact.

## 2026-06-30 - `crt-wide32-r16p80x2-semantic-residual-attention-v1`

Status: completed on `john0`; K=16 failed the larger/diverser hardening gate,
K=24 passed.

Purpose: harden the narrow all-phase residual-attention K=16 prefilter signal
before any search integration claim. The prior p80 run used 2400 train roots and
640 validation roots; this sweep doubles both seed counts while preserving the
same all-phase 80-ply coverage and 16 rollout samples/action.

Implementation:

- Runner:
  `scripts/run_crt_wide32_r16p80x2_semantic_residual_attention_sweep.sh`.
- Detached remote log:
  `logs/r16p80x2_semantic_residual_attention_sweep_job.log`.
- Train roots target:
  `fixtures/crt_wide32_r16p80x2_semantic_train.jsonl`.
- Validation roots target:
  `fixtures/crt_wide32_r16p80x2_semantic_val.jsonl`.
- Train seed count: 60.
- Validation seed count: 16.
- Plies per seed: 80.
- Retained actions/root: 32.
- Rollout samples/action: 16.
- Residual seeds: 20260640, 20260641, 20260642.
- Model: same 4-layer MLP-anchored residual-attention Transformer as the p80
  sweep, hidden size 256, 8 heads, MLP 512, residual scale 0.25,
  top-k-retention objective.

Evidence:

- Per-seed reports:
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed*_pilot.json`.
- Per-seed checkpoint replays:
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed*_prefilter_eval.json`
  and corresponding `_roots.jsonl` files.
- Fixed 3-seed ensemble report:
  `reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval.json`.
- Human summaries:
  `reports/crt_wide32_r16p80x2_residual_seed_ensemble_3x_eval_summary.md`
  and
  `reports/crt_wide32_r16p80x2_semantic_residual_attention_seed_sweep_summary.md`.

Result:

- Train roots: 4800.
- Validation roots: 1280.
- Seed 20260640 residual K=16: recall 0.7312, oracle regret 0.1098.
- Seed 20260641 residual K=16: recall 0.7453, oracle regret 0.1064.
- Seed 20260642 residual K=16: recall 0.7383, oracle regret 0.1149.
- Same-run semantic MLP K=16 recalls: 0.7406, 0.7563, and 0.7469.
- Fixed 3-seed residual ensemble K=16: recall 0.7367, oracle regret 0.1084.
- Fixed 3-seed residual ensemble K=24: recall 0.8945, oracle regret 0.0328.
- Recommended serving K from the ensemble gate: 24.
- `has_merit=false` for all three residual checkpoints under the scalar/top-1
  offline merit gate.

Launch evidence:

- Initial launch was killed before GPU work because the runner's embedded
  markdown summary used shell-sensitive backticks in an unquoted heredoc.
- The runner was patched to remove those backticks, syntax-checked with
  `bash -n`, checked for remaining backticks, and relaunched with
  `REGENERATE_ROOTS=1` so the partial training JSONL from the killed job cannot
  be reused.
- The first cleanup only killed the parent shell, revealing an orphaned exporter
  child. The runner was then hardened to launch the detached job with `setsid`
  and expose a `stop` action that kills the process group plus matching stale
  p80x2 exporter/training/evaluator processes.
- Clean relaunch pid after process-group hardening: 8567.
- Exporter unit tests passed; train-root generation completed with 4800 roots,
  validation-root generation completed with 1280 roots, RTX 5090 was visible,
  and 21 unit/schema tests passed on `john0` before training.

Decision gate:

- Same serving gate as the p80 result: smallest K with teacher-best recall
  >= 0.750 and mean oracle regret <= 0.250 sampled-teacher points.
- The fixed residual-attention ensemble does not hold K=16 on the larger
  validation set: 0.7367 < 0.750 recall.
- K=24 passes comfortably, but K=24 is a wider and less valuable search
  prefilter than the targeted K=16.
- This means the earlier p80 K=16 pass was too narrow to justify search
  integration for residual attention.
- The next transformer iteration should not be a search integration of this
  checkpoint. The evidence points toward either treating the semantic MLP as the
  stronger baseline to beat, or changing the transformer structure so attention
  cannot damage the dense semantic action signal.

## 2026-06-30 - `crt-wide32-r16p80x2-checkpoint-member-ensembles-v1`

Status: completed on `john0`.

Purpose: replay the non-primary members already stored in the p80x2 residual
checkpoints. The residual primary missed K=16, but the same reports showed the
MLP and vanilla public-token Transformer baselines were competitive. This probe
exports per-root rankings for those checkpoint members and runs the same fixed
3-seed ensemble gate without retraining.

Implementation:

- Evaluator extension: `src/cascadiav3/torch_prefilter_eval.py` now accepts
  `--checkpoint-member primary|mlp|vanilla|immediate` and writes per-root rows
  for the selected member.
- Runner:
  `scripts/run_crt_wide32_r16p80x2_mlp_member_ensemble_eval.sh`.
- MLP command:
  `bash scripts/run_crt_wide32_r16p80x2_mlp_member_ensemble_eval.sh`.
- Vanilla command:
  `CHECKPOINT_MEMBER=vanilla bash scripts/run_crt_wide32_r16p80x2_mlp_member_ensemble_eval.sh`.
- Validation set:
  `fixtures/crt_wide32_r16p80x2_semantic_val.jsonl`, 1280 roots.
- Checkpoints:
  `checkpoints/crt_wide32_r16p80x2_semantic_residual_attention_seed20260640_pilot.pt`,
  `checkpoints/crt_wide32_r16p80x2_semantic_residual_attention_seed20260641_pilot.pt`,
  and
  `checkpoints/crt_wide32_r16p80x2_semantic_residual_attention_seed20260642_pilot.pt`.

Evidence:

- MLP member ensemble:
  `reports/crt_wide32_r16p80x2_mlp_seed_ensemble_3x_eval.json`.
- MLP member summary:
  `reports/crt_wide32_r16p80x2_mlp_seed_ensemble_3x_eval_summary.md`.
- Vanilla member ensemble:
  `reports/crt_wide32_r16p80x2_vanilla_seed_ensemble_3x_eval.json`.
- Vanilla member summary:
  `reports/crt_wide32_r16p80x2_vanilla_seed_ensemble_3x_eval_summary.md`.

Result:

- Fixed 3-seed MLP member ensemble K=16: recall 0.7461, oracle regret 0.1048;
  fail by recall.
- Fixed 3-seed MLP member ensemble K=24: recall 0.9000, oracle regret 0.0337;
  pass.
- Vanilla member single-seed K=16 recalls: 0.7508, 0.7328, and 0.7516.
- Fixed 3-seed vanilla public-token Transformer member ensemble K=16: recall
  0.7570, oracle regret 0.1125; pass.
- Fixed 3-seed vanilla public-token Transformer member ensemble K=24: recall
  0.9062, oracle regret 0.0321; pass.
- Remote verification: RTX 5090 visible and 21 unit/schema tests passed before
  both member-ensemble runs.

Decision:

- Residual attention is not the current K=16 path.
- The semantic MLP is close but also misses K=16 under fixed seed ensembling.
- The vanilla public-token Transformer member is now the strongest dry-run
  K=16 serving prefilter signal on p80x2.
- Next credible experiment: train/evaluate the vanilla public-token Transformer
  as a first-class checkpoint family rather than as a side member of the
  residual run, then consider a search-prefilter pilot only if that dedicated
  vanilla family also survives a strict held-out gate.

## 2026-06-30 - `crt-wide32-r16p80x2-semantic-vanilla-public-token-v1`

Status: completed on `john0`; fixed 3-seed ensemble passes the strict K=16
serving gate.

Purpose: isolate the vanilla public-token Transformer signal found in the
p80x2 residual checkpoints. This run trains the same semantic 61-feature
public-token Transformer as a primary checkpoint family, then replays each
checkpoint and a fixed equal-weight seed ensemble through the serving-shaped
prefilter gate.

Implementation:

- New model module:
  `src/cascadiav3/torch_semantic_vanilla_public_token_merit.py`.
- Evaluator support:
  `src/cascadiav3/torch_prefilter_eval.py` now recognizes
  `CRT-semantic-vanilla-public-token-*` checkpoints as primary
  `vanilla_public_token_transformer` models.
- Runner:
  `scripts/run_crt_wide32_r16p80x2_semantic_vanilla_public_token_sweep.sh`.
- Detached remote log:
  `logs/r16p80x2_semantic_vanilla_public_token_sweep_job.log`.
- Data: reused `fixtures/crt_wide32_r16p80x2_semantic_train.jsonl` and
  `fixtures/crt_wide32_r16p80x2_semantic_val.jsonl`.
- Train roots: 4800.
- Validation roots: 1280.
- Retained actions/root: 32.
- Rollout samples/action: 16.
- Phase coverage: active tile counts 3-22, 64 validation roots per active tile
  count.
- Seeds: 20260650, 20260651, 20260652.
- Model: 4-layer semantic public-token Transformer, hidden size 256, 8 heads,
  MLP 512, top-k-retention objective.

Evidence:

- Per-seed reports:
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed*_pilot.json`.
- Per-seed checkpoints:
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed*_pilot.pt`.
- Per-seed checkpoint replays:
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed*_prefilter_eval.json`
  and corresponding `_roots.jsonl` files.
- Fixed 3-seed ensemble report:
  `reports/crt_wide32_r16p80x2_vanilla_public_token_seed_ensemble_3x_eval.json`.
- Fixed 3-seed ensemble rows:
  `reports/crt_wide32_r16p80x2_vanilla_public_token_seed_ensemble_3x_eval_roots.jsonl`.
- Human summaries:
  `reports/crt_wide32_r16p80x2_vanilla_public_token_seed_ensemble_3x_eval_summary.md`
  and
  `reports/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed_sweep_summary.md`.

Result:

- Seed 20260650 K=16: recall 0.7367, oracle regret 0.1216; recommended K=24.
- Seed 20260651 K=16: recall 0.7375, oracle regret 0.1297; recommended K=24.
- Seed 20260652 K=16: recall 0.7656, oracle regret 0.1181; recommended K=16.
- Fixed 3-seed ensemble K=16: recall 0.7672, oracle regret 0.1146.
- Fixed 3-seed ensemble K=24: recall 0.8992, oracle regret 0.0368.
- Fixed 3-seed ensemble top-1: 0.1555, mean regret 1.0996.
- Recommended serving K from the ensemble gate: 16.
- The first-class ensemble improves K=16 recall over the earlier same-run
  vanilla side-member ensemble: 0.7672 vs 0.7570, while oracle regret is
  slightly worse: 0.1146 vs 0.1125.
- Remote verification: exporter tests passed, RTX 5090 was visible, all 22
  unit/schema tests passed on `john0`, every per-root replay file contains 1280
  rows, artifacts fetched locally, and the GPU was idle after completion.

Decision:

- Dedicated vanilla public-token ensembling is now the strongest dry-run p80x2
  K=16 serving prefilter candidate.
- Single checkpoints are still not reliable enough for K=16 by themselves: two
  of three miss the recall threshold and one passes.
- Residual attention remains demoted for this branch; it failed K=16 on the
  same p80x2 validation set, and the simpler vanilla ensemble now wins on K=16
  recall.
- Next credible experiment: a search-prefilter pilot that inserts the dedicated
  vanilla ensemble as a top-16 candidate filter before the existing downstream
  search/value stage, then measures actual score, latency, and missed-teacher
  forensics. This run is still dry-run sampled-teacher evidence, not gameplay
  strength.

## 2026-06-30 - `crt-wide32-r16p80x2-vanilla-prefilter-game-pilot-v1`

Status: completed on `john0`; interactive search-prefilter bridge works.

Purpose: run the first complete-game v3 integration of the dedicated vanilla
public-token Transformer ensemble. A Python/Torch controller scores every
Rust-streamed simulator root on the RTX 5090, retains K16 actions from 32
greedy-ranked legal candidates, and lets the Rust side run sampled rollout
search inside that retained set. Shadow full-search is enabled to measure
decision-level missed-winner/regret telemetry along the actual prefilter
trajectory.

Implementation:

- Rust bridge:
  `real-root-exporter --interactive-policy-game`.
- Python controller:
  `src/cascadiav3/torch_prefilter_game_pilot.py`.
- Remote runner:
  `scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh`.
- Model ensemble:
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260650_pilot.pt`,
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260651_pilot.pt`,
  and
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260652_pilot.pt`.
- The controller now streams per-decision JSONL incrementally as games finish,
  so a longer future run preserves partial telemetry even if interrupted.

Config:

- Seeds: 2026170000, 2026170001, 2026170002, 2026170003.
- Game shape: 4-player Card A/no-bonus `research_aaaaa`, 80 decisions/game.
- Candidate surface: 32 greedy-ranked actions/root.
- Retained actions: K16 for prefilter-search.
- Downstream search: 16 rollout samples/action, top-4 sampled greedy
  continuations.
- Shadow full-search: enabled for prefilter-search, so every prefilter decision
  also evaluates all 32 candidates before applying the filtered choice.
- Paired full-search baseline: enabled on the same four seeds with all 32
  candidates retained.

Evidence:

- Report:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.json`.
- Per-decision rows:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_decisions.jsonl`.
- Human summary:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_pilot_summary.md`.
- Detached remote log:
  `logs/r16p80x2_vanilla_prefilter_game_pilot_job.log`.

Result:

- Prefilter-search: 4 games, 320 decisions, mean seat score 96.0625,
  p50 95.5, p90 100.0.
- Full-search baseline: 4 games, 320 decisions, mean seat score 95.0625,
  p50 95.5, p90 96.5.
- Mean paired delta, prefilter minus full search: +1.0000.
- Per-seed paired deltas: +0.50, +2.00, +0.75, +0.75.
- Shadow full-search winner retained rate: 0.778125.
- Shadow mean search regret: 0.1142578125.
- Shadow p95 search regret: 0.8125.
- Shadow zero-regret decision rate: 0.803125.
- Mean model scoring latency: 0.01355 seconds/decision.
- Estimated non-shadow rollout fraction: 0.5.
- Estimated non-shadow rollout savings: 0.5.
- Per-decision rows fetched locally: 640.

Remote verification:

- Rust exporter tests passed.
- Release build succeeded.
- RTX 5090 visible to Torch: `2.11.0+cu128`, CUDA 12.8, driver 591.86.
- Full Torch-enabled unit/schema suite passed: 22 tests.
- Artifacts fetched locally after completion.
- GPU was idle after the run and no simulator/controller process remained.

Decision:

- The end-to-end prefilter architecture is now real: Rust owns exact game state
  and legal actions; Python/Torch owns model ranking; Rust owns downstream
  rollout search and scoring.
- The tiny paired result is encouraging and does not show immediate harm from
  K16 filtering, but it is not statistically meaningful strength evidence.
- Shadow mode intentionally pays full-search CPU cost, so the run estimates
  speed savings from retained/candidate counts rather than measuring actual
  non-shadow throughput.
- Next credible evidence run: larger paired seed set and/or non-shadow K16
  prefilter-search to measure real score and wall-clock savings without paying
  the full-32 shadow search tax.

## 2026-06-30 - `crt-wide32-r16p80x2-vanilla-prefilter-game-nonshadow20-v1`

Status: completed on `john0`; K16 is fast but loses score versus full-32
sampled search.

Purpose: run the first larger, non-shadow complete-game test of the dedicated
vanilla public-token Transformer K16 prefilter. The previous 4-seed shadow run
proved the bridge and retained-set safety telemetry, but it intentionally paid
the full-32 CPU cost. This run disables shadow full search so timing is real,
then compares the K16 prefilter trajectory against a paired full-32 sampled
search baseline on the same 20 seeds.

Implementation:

- Base runner:
  `scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_pilot.sh`.
- New non-shadow wrapper:
  `scripts/run_crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.sh`.
- The base runner now supports distinct `JOB_SLUG` values so long-running
  pilot reports/logs/PIDs do not overwrite each other.
- The controller now accepts `--full-baseline-workers`; full-search baseline
  games run in parallel because they do not use the Torch ensemble.
- The Markdown summary writer now distinguishes shadow safety runs from
  non-shadow timing runs.

Config:

- Seeds: 2026171000 through 2026171019.
- Game shape: 4-player Card A/no-bonus `research_aaaaa`, 80 decisions/game.
- Candidate surface: 32 greedy-ranked actions/root.
- Retained actions: K16 for prefilter-search.
- Downstream search: 16 rollout samples/action, top-4 sampled greedy
  continuations.
- Shadow full-search: disabled.
- Paired full-search baseline: enabled on the same 20 seeds with all 32
  candidates retained.
- Full-search baseline workers: 4.
- Model ensemble:
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260650_pilot.pt`,
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260651_pilot.pt`,
  and
  `checkpoints/crt_wide32_r16p80x2_semantic_vanilla_public_token_seed20260652_pilot.pt`.

Evidence:

- Report:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20.json`.
- Per-decision rows:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20_decisions.jsonl`.
- Human summary:
  `reports/crt_wide32_r16p80x2_vanilla_prefilter_game_nonshadow20_summary.md`.
- Detached remote log:
  `logs/r16p80x2_vanilla_prefilter_game_nonshadow20_job.log`.

Result:

- Prefilter-search: 20 games, 1600 decisions, mean seat score 95.4625,
  p50 96.0, p90 99.0.
- Full-search baseline: 20 games, 1600 decisions, mean seat score 96.3500,
  p50 97.0, p90 100.0.
- Mean paired delta, prefilter minus full search: -0.8875.
- Median paired delta: -1.25.
- Paired delta min/max: -4.50 / +2.75.
- Paired delta standard error: 0.4220.
- Per-seed paired deltas:
  -1.75, -2.00, 0.00, -2.25, -1.50, +0.75, -3.25, +0.50,
  -1.25, -4.50, 0.00, +2.75, -2.50, +1.25, +0.25, -2.50,
  -1.25, -2.75, +2.25, 0.00.
- Mean model scoring latency: 0.01237 seconds/decision.
- Mean total decision seconds: 2.3558 for K16 prefilter-search versus 4.4617
  for full-32 search.
- Measured speedup: 1.8939x.
- Measured time reduction: 47.20%.
- Per-decision rows fetched locally: 3200.

Remote verification:

- Rust exporter tests passed.
- Release build succeeded.
- RTX 5090 visible to Torch: `2.11.0+cu128`, CUDA 12.8, driver 591.86.
- Full Torch-enabled unit/schema suite passed: 22 tests.
- Artifacts fetched locally after completion.
- GPU was idle after the run and no simulator/controller process remained.

Decision:

- K16 vanilla-ensemble prefiltering should not be promoted as a strength path:
  the real non-shadow 20-seed run lost 0.8875 mean seat score against the same
  full-32 sampled search baseline.
- The speed result is real and useful: roughly half the candidates produced a
  47.20% mean per-decision time reduction.
- The scientific lesson is not "transformers failed"; it is that the current
  K16 retention point is too aggressive for this teacher/search regime despite
  passing the offline sampled-teacher gate.
- Next credible branch: test K24 in the same non-shadow complete-game setup, or
  train a stronger search-aware/retention-aware model whose gate is calibrated
  directly against gameplay loss, not only sampled-teacher root recall.
