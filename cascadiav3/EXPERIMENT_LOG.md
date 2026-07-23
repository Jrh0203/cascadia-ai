# Cascadia v3 Transformer Experiment Log

This log records v3 transformer architecture experiments as they run. Entries
distinguish implementation health from model merit; dry-run experiments are not
promotion evidence.

## 2026-07-02 - `gumbel-cycle1-budget-sweep` + `gumbel-selfplay-cycle3-v1` (EI-4)

Budget sweep on the cycle-1 champion (25 matched seeds, paired offline
against the honest control 95.40): n=64 `94.53` (-0.87), n=128 `95.11`
(-0.55), **n=256 `95.62` (-0.04 — statistical parity with the 10.9s/dec
rollout control at 3.2s/dec)**. Verdict: budget-bound; the real-outcome
value head converts simulations into strength monotonically. The Gumbel
stack now matches the legacy search at ~3.4x less compute with headroom.

Cycle 3 launched (~19:45): full scale 1,250+125 seeds on the optimized
stack, teacher upgraded to n=128 labels, blend back at w=0.5, replay window
cycles 2+1 at weights 0.5/0.25, warm start from the cycle-1 champion.
ETA gates ~06:00. Next gates should also test the cycle-2 checkpoint at
n=256 (its 4x-better regret may scale better with budget than n=64 showed).

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

Status: completed with positive no-search improvement but no K56 search
breakthrough; terminated 2026-07-02 in favor of the Gumbel self-play campaign
(see `gumbel-selfplay-stack-implementation-v1` above); partial artifacts
fetched and retained as teacher-comparison evidence.

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

Training result:

- Runbook status: `pass`.
- Generated train/validation roots: `20,000` / `4,000`.
- Generation throughput: `10.5309` roots/s and `1,347.9596` rollout evals/s.
- Training throughput: `0.0978` seconds/step.
- Selected checkpoint: `best_locked_val` at step `15,000`.
- Best locked validation final-Q regret: `1.909125`.
- Final step locked validation final-Q regret: `2.110875`, so the final
  checkpoint regressed and should not be used for gameplay.

100-game no-search result:

- Report:
  `reports/cascadiaformer_ei1_model_state_k32_r4_no_search_game100.json`.
- Manifest:
  `checkpoints/full_v3_ei1_model_state_k32_r4/best_locked_val.manifest.json`.
- CascadiaFormer-q mean seat score: `90.7600`.
- Greedy mean seat score on matched seeds: `87.5450`.
- Mean paired delta versus greedy: `+3.2150`.
- EI-1 q also exceeds EI-0 q's prior `89.6175` 100-game no-search mean.

K56 search result:

- Attempted report:
  `reports/cascadiaformer_ei1_model_state_k32_r4_k56_search_game20.json`.
- The long search benchmark exited before writing its final JSON report. No OOM
  or disk-pressure evidence was found; the old harness only journaled decisions,
  so the run was recoverable only as a partial/provisional result.
- Recovered summary:
  `reports/cascadiaformer_ei1_model_state_k32_r4_k56_search_game20_recovered_from_decisions.json`.
- Recovery method: for each complete 80-ply game, use the final four decisions'
  `selected_active_score` values as final seat scores. This is useful forensic
  evidence, not a substitute for the normal benchmark `done` payload.
- Recovered CascadiaFormer-search K56 mean: `96.4250` over all `20` candidate
  games.
- Recovered full-search K64 control mean: `96.765625` over the `16` completed
  control games.
- Recovered paired delta on the `16` completed pairs: `-0.453125`.

Decision:

- EI-1 did improve the no-search q policy, which means model-state expert
  iteration has merit.
- EI-1 did not move search-integrated play out of the existing `96-97` score
  band. Do not scale this exact K32/R4 objective as the path to 100.
- The benchmark harness now journals completed game rows to `*_games.jsonl`
  files in addition to per-decision rows, so future long search runs preserve
  score-grade completed-game evidence even if a final tail fails.
- For future 16-worker CPU search probes, prefer game counts that are multiples
  of 16, such as `32`, to reduce underutilized tail waves unless continuity
  with a 20-game historical seed set is required.

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

## 2026-07-03 — Cycle-3 export crash (zip64) and relaunch on optimized stack

Failure:

- Cycle 3 (EI-4) generation completed all 1,250 train seeds / 100,000 records
  in 56,085 s (~15.6 h, 0.022 seeds/s), then the train-tensor npz write died
  with the zip crate error "Large file option has not been set": one array
  exceeded the 4 GiB zip entry limit and `large_file(true)` (zip64) was never
  set. The partial 8.5 GB archive has no central directory; the mode writes
  no JSONL sidecar, so the search data is unrecoverable. Validation
  generation and training never started.
- Root cause is scale-triggered: the 100k-record corpus is ~10x cycle 1;
  no earlier corpus crossed 4 GiB per entry.

Fixes and changes deployed with the relaunch:

- `npz_writer.rs` now sets `.large_file(true)` on every entry (zip64;
  numpy/zipfile read these transparently). Commit `5e84d7b`.
- Optimization pass 2 (merged earlier today, commits `bb9d00c`/`a087f53`):
  eval-row dedup + per-chunk cache (43.7% of model eval rows eliminated at
  production search shape under the mock) and packed f64 responses (7.7x
  encode / 2.9x decode), both parity-gated. Dedup counters now appear in
  generation progress lines.

Relaunch:

- Same configuration and seeds as the failed run (train 2026730000 x 1250,
  val 2026830000 x 125, n=128, w=0.5, shared bridge, 16 sessions, replay
  tail cycles 2+1 at weights 1.0/0.5/0.25, warm start from cycle-1 champion).
- Labels will differ from the lost run at float precision (GPU batch
  composition changes with dedup); this is expected and acceptable.
- The rerun doubles as the production measurement of optimization pass 2:
  prior stack measured 0.022 seeds/s (~80 games/h) at n=128.

## 2026-07-03 — Cycle-3 (EI-4) gate battery: flat at all budgets -> model-class bound

Battery ran 16:36-17:15 (39 min for 250 games — first battery on the batched
shared-bridge harness `--batch-runner`; the cycle-2 battery took ~3 h).

| Metric | Cycle 1 | Cycle 2 | Cycle 3 |
|---|---:|---:|---:|
| No-search q (100g, seeds 2026994000) | 91.71 | 91.85 | 91.805 |
| Gumbel n=64 (100g, seeds 2026995000) | 94.53 | 94.4725 | 94.6475 |
| Gumbel n=256 (25g) | 95.62 | 95.79 | 95.67 |
| locked_val_final_q_regret | 0.79 | 0.21 | 0.152 |

Paired stats (cascadiav3.torch_benchmark_stats.paired_delta_stats):

- c3 vs c2, n=64, n=100 pairs: mean +0.175, t-CI95 [-0.1325, +0.4825] — ns.
- c3 vs c2, n=256, 25 pairs: mean -0.12, CI [-1.1775, +0.9375] — ns.
- c3 vs c1, n=256, 25 pairs: mean +0.05, CI [-0.6668, +0.7668] — ns.
- c2 vs c1, n=256, 25 pairs: mean +0.17, CI [-0.8361, +1.1761] — ns.

Interpretation:

- Three EI cycles with 10x data growth, stronger labels (n=128), a replay
  window, and a 5x value-regret improvement produced NO gameplay movement at
  any search budget. Cycle-2's regret gain did not convert at n=256 either.
- Conclusion: the campaign is **model-class bound at CascadiaFormer-S**, the
  explicit branch-3 outcome in CAMPAIGN_STATE.md's decision tree. Value-head
  quality and data scale are no longer the levers.
- Methodology gap found: the honest control's per-seed scores were never
  persisted (only aggregate 95.40 + paired deltas vs the EI-1 candidate), so
  control comparisons for c2/c3 are mean-only. Persist control per-seed on
  the next control re-run.

Decision: train CascadiaFormer-M from scratch on the existing cycle-3 corpus
(no new generation needed), same objective and selection; then a battery of
no-search / n=64 / n=256 plus first n=512 and depth_rounds=2 probes.

## 2026-07-04 — CascadiaFormer-M battery: first significant gameplay win; S saturation confirmed

M trained from scratch on the cycle-3 corpus (25k steps, batch 192, ~16.6h on
the pre-optimization trainer). Selection slip: run selected on
locked_val_total (step 5500); regret-optimal saved checkpoint is step 10000
(regret 0.1559 ~= S's 0.152). Both heads benchmarked.

| Benchmark | S (c3) | M total (s5500) | M regret (s10000) |
|---|---:|---:|---:|
| No-search q 100g | 91.805 | 86.9325 | 90.8775 |
| Gumbel n=64 100g | 94.6475 | 95.54 | 95.2375 |
| Gumbel n=256 25g | 95.67 | 96.63 | **97.11** |

Paired (same seeds, paired_delta_stats): M-total vs S at n=64 +0.8925
CI95 [0.5294, 1.2556]; M-regret vs S at n=64 +0.59 [0.2293, 0.9507];
M-regret vs S at n=256 +1.44 [0.4969, 2.3831] — ALL CI-excluding-zero, the
first significant gameplay improvements of the campaign. M n=64 (95.54 @
~1.9 s/dec) also exceeds the honest rollout control (95.40 @ 10.9 s/dec) on
mean. S-champion probes: n=512 vs n=256 -0.01 [-0.8576, 0.8376] (budget
saturated); depth2/n=64 -0.65 [-1.5020, 0.2020] (no depth win at S).

Conclusions: capacity was the binding constraint, exactly as the cycle-3
flat battery indicated. M's strength expresses through search (its no-search
q play is weaker than S — and total-loss selection is unusable no-search at
86.93: regret selection vindicated). New campaign best 97.11 (M-regret,
n=256, 25g). NEW CHAMPION: cycle3_m step_0010000 (regret-selected).

Next: deploy trainer+serving optimization stack; GPU A/B of fused CGAB;
power the 97-gate properly (n=256 at 100+ games); then cycle 4 = EI on M
(M teacher labels now affordable with fused forward + optimized stack).

## 2026-07-04 — 97-gate passed at power: M formally promoted; cycle 4 launched

Powered gate (n=256, 100 paired games, seeds 2026995000, both sides on the
fused-CGAB serving stack after the 25-game A/B showed EXACT score parity —
paired delta identically 0.0 across 25 games):

- M champion (cycle3_m step_0010000): mean 96.9125, p50 97.0, p90 100.0;
  2/100 games at mean seat >= 100.
- S incumbent (cycle3 best_locked_val): mean 95.7175, p50 96.0, p90 99.0.
- Paired M-S: +1.1950, CI95 [0.8306, 1.5594], n=100 — CI lower bound 3.3x
  the +0.25 promotion bar. PROMOTED.

Also merged today: engine hot-path pass 2 (habitat delta queries, sum-only
greedy ranking, scratch reuse): rollouts +76% throughput, 1.70x end-to-end
export CPU, byte-identical corpora; deployed to john0 (28/28 remote).

Cycle 4 (EI-5) launched on the full optimized stack: M teacher
(step_0010000), n=256 labels, w=0.75 (trust ramp), 1,250+125 seeds
(2026740000 / 2026840000), replay tails cycles 3+2 (1.0/0.5/0.25),
MODEL_SIZE=M warm-started from the champion, regret selection (wrapper
default), trainer knobs: --data-workers 4 --prefetch-factor 4 --tf32
--fused-optimizer --cgab-fused; bridge served with CASCADIA_CGAB_FUSED=1
and 8x cell budget via MODEL_SERVICE env prefix.

## 2026-07-05 — Cycle-4 battery: EI-on-M works at low budget; ceiling unmoved. PROMOTED.

Cycle 4 end-to-end: 8.1h (gen 26,743s at n=256 labels + M teacher; training
1,180s at 0.0472 s/step — the optimized trainer's first production run, ~50x
the first M run's step time; example-pass clamp correctly limited to 4,833
steps / 4 passes over 232k records). Generation dedup at n=256: 75.0%
(9.93M rows -> 2.48M sent).

| Benchmark (100g each) | M champ (c3m s10000) | Cycle-4 M | Paired |
|---|---:|---:|---|
| No-search q | 90.8775 | 91.1775 | — |
| Gumbel n=64 | 95.2375 | 95.77 | +0.5325 CI [0.1524, 0.9126] EXCL0 |
| Gumbel n=256 | 96.9125 | 96.95 | +0.0375 CI [-0.2996, 0.3746] ns |

PROMOTED (dominates: CI+ at n=64, parity at n=256, better no-search).
NEW CHAMPION: cycle4 best_locked_val. Locked regret 0.2576 (not comparable
across cycles — val distribution changed with n=256/w=0.75 labels).

Pattern: EI compresses the teacher's search strength into the prior
(low-budget play rises); the high-budget ceiling stays ~96.9-97.0. M's own
budget/depth scaling never probed (earlier probes were on saturated S; M
scaled +1.67 from n=64->256). Probes launched on cycle-4 champion:
n=512 50g and depth_rounds=2 at n=256 50g (seeds 2026995000).
Decision: probes CI+ -> budget/depth push toward 98+, plan the 1,000-game
100-gate; probes flat -> CascadiaFormer-L (repeat the capacity jump) and/or
w=1.0 cycle 5.

## 2026-07-05 — Cycle-4 champion probes: budget lever alive on M; depth dead

- n=512 (50g): mean 97.47, p50 97.0, p90 101.0 — paired vs n=256 +0.575
  CI95 [0.0301, 1.1199], EXCL0. M keeps scaling with budget where S was flat.
- depth_rounds=2 at n=256 (50g): 96.87, paired -0.025 [-0.4756, 0.4256] —
  ns at 1.8x cost. Depth lever closed (matches the S result).
- Budget curve on cycle-4 M (n=64/256/512): 95.77 / 96.95 / 97.47 —
  ~+0.5 per doubling, decelerating. Budget alone will not reach 100; the
  playbook is capacity jump (shift the curve) + EI (compress it downward).

Next: CascadiaFormer-L added (d1024/16L/16H, 207.0M params vs M 88.2M,
S 15.0M; suite green). L from-scratch training on the cycle-4 corpus
launches after the GPU knob-tuning sweep completes.

## 2026-07-05 — Fleet incorporation (john1-4) + serving env finalized

GPU knob sweep on john0 (M champion, 20g n=64 each, timing signal only):
base p50 1.533s / TF32 1.124s (1.36x, ADOPTED for generation; batteries
stay TF32-off for historical comparability) / bucketing 1.586s (no win
without compile; skipped) / compile FAILED (triton needs a C compiler WSL
lacks; fixable via CC=zig-cc, deferred) / bigger gather+row-cap neutral.
Trainer probe (M, 300 steps, full knobs): 1.69 s/step wall (backward 1.15,
forward 0.52, data 0.003 — worker fix confirmed), 17.9GB, SDPA verdict:
flash structurally rejected (float mask), mem_efficient USABLE and
selected — no forcing needed. Note: runbook train_step_seconds (0.047 for
cycle 4) measures a narrower quantity than wall step time.

Fleet (john1-4, Apple M4 10-core 16GB each): provisioned (rust 1.96,
python 3.12 venv via uv, torch 2.12.1 cpu+mps, source + cycle-4 champion
weights). MPS serves the fused-CGAB M forward cleanly on all hosts.
Calibration (4 seeds each, n=128, w=0.75, 3 sessions, MPS): ~19 seeds/h
per mini, +/-1% across hosts; fleet ~76 seeds/h ~= 1,800 seeds/day
continuous. Policy: fleet generates TRAINING DATA ONLY (MPS numeric drift
acceptable for self-play labels); all paired evaluation stays on john0.

Fleet production run launched: 1,000 supplementary seeds (2026750000 x250
per host, n=128, w=0.75, cycle-4 champion teacher) -> fleet_shard_johnN.npz
on each host, ~13h. Destined for the next training cycle's replay mix.

In flight on john0: CascadiaFormer-L (207M, d1024/16L) from scratch on the
cycle-4 corpus, full trainer knobs + grad-checkpoint on, ETA ~5-7h.

## 2026-07-06 — L (4-pass) battery: flat vs cycle-4 M; relaunched with 16 passes

L-v1 (from scratch, 4-pass clamp = 4,833 steps, 46 min train): no-search
90.7775; n=64 95.42 (-0.35 [-0.7249, +0.0249] vs c4-M, ns-negative);
n=256 96.79 (-0.16 [-0.5092, +0.1892], ns). NOT promoted.

Interpretation caveat before concluding capacity-exhausted: the 4-pass
example clamp was designed for warm-started EI cycles; the successful
from-scratch M run (cycle 3) effectively trained ~34 passes. From-scratch
L at 4 passes is optimization-starved, not a clean capacity test.
L-v2 relaunched with MAX_EXAMPLE_PASSES=16 (~19.3k steps, ~3h); v1 runbook
archived as *_runbook_v1_4pass.json. If v2 is also flat, the capacity
lever is genuinely closed at current data scale and the plan shifts to
data volume (fleet corpus, in flight: 1,000 n=128 seeds, ~06:30 ETA) and
teacher-quality levers (w=1.0, n=512 labels).

Ops: overnight monitors died with the session — L-v1 battery finished
23:19 but nothing woke the shepherd; box idled ~7h. Mitigations: single
consolidated watchdog monitor per work-wave + on any session resume,
IMMEDIATELY check all in-flight job logs before anything else (this is
now standing procedure in CAMPAIGN_STATE.md).

## 2026-07-06 03:00 — L-v2 (16-pass) training complete; battery launched

L-v2 trained to completion (19,333 steps, MAX_EXAMPLE_PASSES=16, grad
checkpointing, 0.418 s/step; runbook marker pass at 03:00:47).
best_locked_val = **step 7000** (regret-selected,
locked_val_final_q_regret) — regret bottomed at ~36% of the run and never
improved after; the 16-pass budget was more than enough optimization, so
a flat verdict now WOULD be a clean capacity result (v1's starvation
confound removed).

Battery launched 03:08 (`logs/l2_gates_job.{sh,log,pid}`, pid 894830):
no-search 100g seed 2026994000; Gumbel n=64 + n=256 100g seed 2026995000,
batch runner, fused CGAB + 8x cell budget, TF32 off; reports
`gumbel_l2_no_search_game100.json`, `gumbel_l2_gate_n{64,256}.json`
(v2-specific names, v1 reports preserved). Compare paired vs cycle-4 M
(`gumbel_cycle4_gate_n{64,256}.json`, `gumbel_cycle4_no_search_game100.json`).

Fleet meanwhile at 175-200/250 seeds per host (~0.42 rec/s each),
ETA ~05:30-06:30 for all four shards.

## 2026-07-06 04:40 — L-v2 verdict: FLAT at all budgets. Capacity closed at this data scale

L-v2 (16-pass, best step 7000 regret-selected) vs cycle-4 M champion,
100 paired games each leg:

| leg | L-v2 | c4-M | paired delta | CI95 | verdict |
|---|---:|---:|---:|---|---|
| no-search | 90.96 | 91.18 | -0.215 | [-0.68, +0.25] | ns |
| n=64 | 95.55 | 95.77 | -0.223 | [-0.62, +0.18] | ns |
| n=256 | 96.93 | 96.95 | -0.020 | [-0.33, +0.29] | ns |

With the optimization-starvation confound removed (regret bottomed at
step 7000 of 19,333 and never recovered — the model had all the
optimization it could use), 207M matches 88M exactly on the same corpus.
**Conclusion: model capacity is NOT the binding constraint at ~100k-root
data scale. The lever is data volume + teacher label quality.** This
mirrors the standard scaling result: model size only pays when data
scales with it. NOT promoted; c4-M remains champion.

## 2026-07-06 04:55 — Cycle 5 launched: M-taught EI, n=512 labels, w=1.0, doubled data

Per the documented branch: `logs/gumbel_selfplay_cycle5_job.*`, pid 904648.
- Teacher/incumbent: c4-M champion (best_locked_val, warm start).
- Labels: n=512 sims (the strongest teacher we've measured: 97.47 CI+
  probe), top_m 16, determinizations 4, k_interior 16.
- **w=1.0** — first rollout-free generation cycle (leaf = pure value
  bootstrap; verified code short-circuits rollouts at w>=1.0). Value head
  has had two cycles of real-outcome targets; probes show search trusts
  it. ~2x CPU savings offsets the 2x sim budget.
- Seeds: train 2026770000 x1250, val 2026870000 x125 (fresh blocks).
- Replay mix: cycle5 (1.0) + fleet shards john1-4 (0.75 each) + cycle4
  (0.5) + cycle3 (0.25) -> ~2.4x the fresh+replay data of cycle 4.
  Fleet tails must be filtered/materialized on john0 BEFORE the trainer
  stage reaches its test -s check (fleet ETA ~1-2h, generation ~8-12h;
  ample margin).
- Trainer: M warm start, 25k steps, b192, MAX_EXAMPLE_PASSES=4, regret
  selection, full perf knobs. Bridge: fused CGAB + 8x cell budget + TF32.

When done: standard battery vs c4-M (no-search/n64/n256, 100g paired).
If flat again with 2.4x data, data-scale hypothesis weakens too and the
next levers are search-side (n=512 serving budget is already CI+) and a
bigger fleet corpus regime.

## 2026-07-06 06:45 — Fleet wave-1 folded in; wave-2 launched

Wave-1 shards (4x 20,000 records, 80k roots total) fetched to john0,
filtered top-64 + relation tails materialized
(`fixtures/fleet_shard_johnN_top64_relation_tail.npz`, invariants PASS,
~97.4KB/record). Cycle-5's EXTRA_TRAIN_TAIL_TENSORS dependency satisfied
well before its trainer stage.

Fleet wave-2 launched on john1-4: seeds 2026780000 x250/host (fresh
block), same measured config (n=128, top_m 16, w=0.75 — rollout anchor
kept on MPS deliberately; value-only leaves would inherit MPS numeric
drift), 3 shared sessions, fused CGAB. Outputs `fleet2_shard_johnN.npz`,
ETA ~13h. Destined for cycle-6.

Cycle-5 generation pace: 425/1250 seeds at 6,664s (0.064 seeds/s) —
**n=512 w=1.0 generates FASTER than cycle-4's n=256 w=0.75** (rollout
elimination + eval dedup at high budget more than pay for 2x sims).
Train-gen ETA ~10:30, battery ~14:00.

## 2026-07-06 ~12:30 — Cycle-5 trainer OOM-killed; relaunched with 1 data worker

Generation, filtering, and materialization all completed (train corpus
100k roots at n=512/w=1.0 in ~5.4h, val 125 seeds). The TRAINER then
died: kernel OOM-killed a pt_data_worker (anon-rss ~35GB each). Root
cause: the 7-source mix (~380k records, ~37GB of relation-tail tensors)
is fully materialized PER DataLoader worker; 4 workers + main = ~175GB >
121GB RAM. This never bit before because prior mixes were ~2.5x smaller.

Fix: relaunched with REGENERATE_ROOTS=0 (reuses all tensors; only
training reruns) and --data-workers 1 --prefetch-factor 2 (~70GB peak).
Slight step-time risk vs 4 workers; acceptable (data time was ~0.003
s/step). FUTURE: teach the dataset to mmap/share tensors across workers
before scaling the mix further (cycle-6 with fleet2 will be ~460k
records — 1 worker still fits, ~74GB, but headroom shrinks).

## 2026-07-06 ~13:15 — Shard mmap fix landed (the OOM root-cause fix)

`ExpertTensorShard` now memory-maps ZIP_STORED npz members
(`_MmapNpz`; default ON, `CASCADIA_SHARD_MMAP=0` reverts, automatic
np.load fallback for compressed/object shards). All processes share one
page-cache copy of the corpus; anonymous RSS per worker drops from
~O(corpus) (~35GB at cycle-5 scale) to ~0. Bit-equality tests green
(4 new tests; full suite 98 OK). Deploys to john0 with the next
pipeline rsync — cycle-6 can go back to --data-workers 4 and scale the
mix without RAM ceiling concerns.

## 2026-07-06 12:16 — Cycle-5 battery: NOT promoted (n64 CI-); ablation launched

Cycle-5 (M warm start, n=512 w=1.0 labels, fleet n=128 mix, best step
7250/7916) vs cycle-4 M champion, 100g paired:

| leg | c5 | c4-M | delta | CI95 | verdict |
|---|---:|---:|---:|---|---|
| no-search | 91.46 | 91.18 | +0.283 | [-0.20, +0.76] | ns |
| n=64 | 95.39 | 95.77 | **-0.383** | [-0.75, -0.02] | **CI-** |
| n=256 | 97.08 | 96.95 | +0.130 | [-0.24, +0.50] | ns |

Pattern: q/value slightly better (no-search up), search guidance worse
at low budget (n64 relies most on prior quality via Gumbel top-m), flat
at n256 (97.08 is nominally the best 100g n256 mean ever recorded, ns).
Two confounded changes vs cycle 4: (a) w=1.0 value-only labels, (b)
fleet n=128/MPS data at weight 0.75. One of them poisoned the prior.

**Ablation (launched now, cheap since tensors exist): retrain identical
except NO fleet shards** (c5 fresh 1.0 / c4 0.5 / c3 0.25 — cycle-4's
mix shape). REGENERATE_ROOTS=0, profile gumbel_selfplay_cycle5_nofleet,
~1.5h train + 40min battery.
- nofleet beats c4-M or at least kills the n64 CI- -> fleet n=128 data
  was the poison -> fleet regime must change (higher-budget labels or
  value-only usage), w=1.0 exonerated.
- nofleet still CI- at n64 -> w=1.0 labels are the poison -> cycle 6
  reverts to w=0.75 (or 0.9) labels; fleet data gets a separate trial.
First training run with the shard-mmap fix deployed (workers back to 4).

## 2026-07-06 12:45 — Nofleet ablation trained; mmap fix = 5.5x trainer speedup

Nofleet retrain: 6,250 steps in 1,426s = **0.228 s/step wall** (M, b192,
4 workers + mmap). This morning's cycle-5 run did 7,916 steps at ~1.26
s/step with --data-workers 1 — i.e. the single-worker fallback was
DATA-BOUND, and the "M wall step 1.69s" probe from 07-05 evidently was
too. With shard-mmap + 4 workers the M trainer is now GPU-bound at
~0.23 s/step: full 25k-step budgets now cost ~1.6h, and the 4-pass
clamped cycles train in ~25 min. Battery for the ablation launched
(reports gumbel_c5nf_*).

## 2026-07-06 13:42 — Ablation verdict: fleet n=128 data was the poison; w=1.0 exonerated

Nofleet retrain (identical to cycle-5 minus the 4 fleet shards) vs c4-M,
100g paired: no-search +0.4475 [-0.0107, +0.9057] ns (a hair from CI+;
91.63 = best M-line no-search yet); n64 -0.0925 [-0.48, +0.29] ns (the
with-fleet CI- is GONE); n256 +0.035 ns.

Conclusions:
1. Fleet n=128/MPS labels at weight 0.75 caused cycle-5's n64 CI-.
   Fleet regime must change before its data re-enters training (options:
   much lower weight, higher label budget, or value-only usage). Wave-2
   left running (minis otherwise idle; data may still serve low-weight
   or value-only trials).
2. w=1.0 (rollout-free) labels are fine — generation keeps the ~2x CPU
   saving.
3. Still NOT promotable (no CI+ leg). The M-class ~97 plateau at n256
   now stands across: data 1x->3x, labels n256->n512, capacity M->L.
   EI compresses the teacher into the prior but the search ceiling
   itself is the binding constraint.

Next lever (unexplored): serving-side search shape. Probe sweep
launched on c4-M champion, 25g paired vs default-config control at the
same budget: determinizations 8 and 16 (vs 4) at n256, top_m 32 (vs 16)
at n512, k_interior 32 (vs 16) at n256, and an n=1024 ceiling probe.
Any CI+/promising delta gets a 100g confirm.

## 2026-07-06 15:02 — SERVING BREAKTHROUGH: determinizations + k_interior are live levers

Probe sweep on c4-M champion (25g paired vs default d4/k16 controls,
same seed block):

| config | mean | delta vs ctrl | CI95 | p50 s/dec | verdict |
|---|---:|---:|---|---:|---|
| n256 d8 | 97.60 | +0.93 | [+0.20, +1.66] | 3.2 | **CI+** |
| n256 d16 | 97.60 | +0.93 | [+0.37, +1.49] | 4.7 | **CI+** |
| n256 k32 | 97.28 | +0.61 | [+0.21, +1.01] | 2.5 | **CI+** |
| n512 m32 | 98.01 | +0.18 vs n512 | [-0.45, +0.81] | 4.7 | ns |
| n1024 | 97.49 | +0.82 vs n256 | [-0.01, +1.65] | 4.7 | ns |

Search was starved of DETERMINIZATIONS (hidden-world samples), not
simulations: d16@n256 costs the same wall-clock as n1024@d4 and beats
it. d8 saturates the gain at 2/3 the cost. k_interior 32 adds +0.61
nearly free. First CI+ moves at the ~97 plateau after capacity, data,
and label-budget all came up flat.

Combined probes launched (n256_d8_k32, n256_d16_k32, n512_d16_k32,
25g each). If gains stack, n512_d16_k32 could land ~98.5; then a 100g
confirm and the 100-gate math changes materially. Also note n512_m32's
98.01 = highest 25g mean yet recorded.

## 2026-07-06 16:04 — Combined probes: gains do NOT stack; d8/d16 is the lever

n256_d8_k32 +0.54 ns (97.21); n256_d16_k32 +0.86 CI+ (97.53) = same as
d16 alone; n512_d16_k32 97.52, -0.31 ns vs n512-d4 control (97.83).
k_interior adds nothing on top of determinizations; 25g noise (~±0.7)
explains the combined-config scatter. The replicated, isolated signal is
determinizations 4 -> 8 (saturates by 8; d16 = d8 at 2/3 the cost... 
inverse: d8 = d16 at 2/3 the cost).

100g confirmations launched vs the promoted n256-d4 champion config
(same 2026995000 seed block): n256_d8 and n512_d8
(gumbel_confirm_n{256,512}_d8.json). If n512_d8 lands ~98+ CI+, it
becomes the new serving config, cycle-6 teacher labels adopt d8, and
the 100-gate plan gets drafted against the new curve.

## 2026-07-06 17:58 — n512_d8 CONFIRMED at 100g: new serving config, 97.845

100g paired vs promoted n256-d4 champion config (seed block 2026995000):
- n256_d8: 97.25, +0.30 [-0.07, +0.67] ns (25g's +0.93 shrank at power)
- **n512_d8: 97.845, +0.895 [+0.5636, +1.2264] CI+** — 4/100 games at
  >=100 mean seat, p50 5.51 s/dec (2.4x the old config's cost).

SERVING CONFIG PROMOTED: champion = c4-M at n512/top_m16/w0.5/d8.
Best-ever 100g mean: 97.845 (previous 96.95). Gap to the 100-gate:
-2.16. Determinizations only pay at high sim budget — at n256 the
per-world sim count gets too thin (256/8=32 sims/world vs 512/8=64).

Cycle-6 launched overnight: teacher c4-M with **n=512 d8 w=1.0 labels**
(the confirmed-stronger search now generates the training targets),
seeds 2026790000x1250 / 2026890000x125, replay c6(1.0)/c5(0.5)/c4(0.25),
NO fleet data, 4 workers + mmap. Gen ETA ~8-10h, train ~25 min. Battery
vs champion (incl. an n512_d8-leg) in the morning. Queued after: n1024_d8
and d16-at-n1024 probes if cycle-6 compresses the prior further.

## 2026-07-06 19:00 — Generation knob tuning: NEGATIVE result; bf16 labels unsafe

Cycle-6 gen showed john0 underutilized (load ~11/32 cores, GPU 55-60%,
exporter ~4 cores, bridge ~4 cores) -> hypothesized eval-batching
starvation. Swept CASCADIA_SHARED_GATHER_US {2ms,4ms,8ms} x ROW_CAP
{192,768,1536} x bf16 autocast, at 8-seed and saturated 32-seed scale
(n512/d8/w1.0 production config):
- ALL configs within noise (104s/104s/101s at saturation; bf16 only
  ~3%). GPU forward is NOT the bottleneck; the loop is bound by per-ply
  lockstep cadence + search-side work that env knobs cannot reach.
- bf16 label-drift check (identical seeds, record-aligned): selected
  actions match fp32 only ~26%, max|dQ| ~4 — bf16 materially changes
  search decisions. NOT adopted for teacher labels (fleet lesson).
  fp32 configs agree 100% on actions across batching settings.

Cost of the detour: ~1.5h (killed cycle-6 at 55/1250 seeds + tuning).
Cycle-6 relaunched stock (pid 961106), gen ETA ~07:30-08:00, verdict
~12:00-12:30. ENGINEERING QUEUE: structural generation throughput —
partition game workers across 2-3 bridge processes (GPU and CPU both
half-idle; two bridges should approach ~2x), and/or intra-ply
pipelining so workers don't block on the gather window.

## 2026-07-06 19:30 — Fleet wave-2 archived; wave-3 launched at n256/d8

Wave-2 (4x 20k records, n=128 d4 w=0.75, seeds 2026780000) complete and
fetched to john0 fixtures/fleet2_shard_johnN.npz — ARCHIVED, not folded
into training (wave-1's n=128 labels caused the cycle-5 n64 CI-).
Reserved for low-weight/value-only trials.

Wave-3 REVISED after a sizing review (first launch at n256/d8 x250
was mis-sized: d8 does not pay at n=256 per the 100g confirm, and a
safety trial does not need 1,000 seeds): relaunched as **n=256, d4,
w=0.75, 100 seeds/host** (seeds 2026781000/100-spaced), ~11h. Purpose:
test whether higher-grade fleet labels (2x the eval budget of waves
1-2) can enter the training mix at low weight without the n64
regression. Scale to production volume only if the trial passes.
Hypothesis unchanged: weak LABELS (n=128 d4) poisoned cycle-5, not MPS
numerics per se.

## 2026-07-07 08:26 — Cycle-6 battery: FLAT on all four legs. EI at M is saturated

Cycle-6 (d8-taught labels, no fleet, best step 5000) vs c4-M champion,
100g paired: no-search +0.03 ns; n64 -0.05 ns; n256 +0.07 ns;
**n512d8 97.62 vs 97.845, -0.225 [-0.54, +0.09] ns** (5/100 games
>=100). The d8 teacher gain did NOT compress into the prior.

EI-saturation picture complete: at M capacity the prior has absorbed
all it can — better labels (n256->n512->n512d8), more data (1x->3x),
more capacity (L), all flat. c4-M REMAINS champion. Road to 100 =
serving-side search + structural throughput.

Launched (chained on one GPU): fleet-trial retrain (c6 corpus + fleet3
n256/d4 shards at weight 0.25) -> probe wave: n1024_d8, n1024_d16,
and the never-swept serving blend weight w=0.75/1.0 at n512d8 (gates
have always used w=0.5) -> fleet-trial n64+n256 100g battery (checks
for the wave-1-style regression with higher-grade fleet labels).

## 2026-07-07 11:19 — Probe wave 3: the WORLDS axis scales; blend answered; fleet safe

- **n1024_d16: 98.51, +0.92 CI+ [+0.43, +1.41] over n512_d8** (25g,
  8.1 s/dec). n1024_d8 flat (-0.07 ns). The pattern: hold ~64 sims per
  world and scale the NUMBER of determinized worlds — 8->16 worlds gave
  +0.9, same as 4->8 did. The worlds axis decays far slower than the
  sims axis.
- Serving blend: w=0.5 optimal. w=0.75 -0.75 ns; **w=1.0 -2.94 CI-**
  (serving rollouts carry real signal even though w=1.0 is fine for
  training labels).
- Fleet trial: SAFE. c6-corpus + fleet3 (n256/d4) at weight 0.25 vs
  the nofleet c6: n64 -0.07 ns (no wave-1-style regression), n256
  +0.03 ns. Upgraded fleet labels don't poison — but with EI saturated
  they have no customer yet; minis idle pending a data consumer.

Launched (probe4, chained): n1024_d16 100g CONFIRM, n2048_d32 25g
(32 worlds — worlds-axis extrapolation ~99.2+), and the ORACLE PEEK
run (new --gumbel-peek mode, commit 777ed98): n512_d8 100g with true
hidden state = information ceiling; decides whether 100 is reachable
by any honest agent.

## 2026-07-07 15:04 — n1024_d16 CONFIRMED (98.28); worlds axis peaks; ORACLE SHOCK

- **n1024_d16 at 100g: 98.28, +0.435 CI+ [+0.113, +0.757] vs n512_d8.
  NEW SERVING CONFIG. 11/100 games >=100 mean seat. 10.6 s/dec.**
  Gap to the 100-gate: -1.72.
- n2048_d32 (25g): 97.76, **-0.75 CI- vs n1024_d16** — the worlds axis
  saturates/reverses past ~16 worlds. No more free doublings.
- **ORACLE PEEK (n512_d8 + true hidden state, 100g): 97.50, -0.35 CI-
  vs HONEST n512_d8.** Perfect foresight LOSES to honest 8-world
  averaging. With peek, every determinization is the same (true) world,
  so d8 collapses to d1-truth — and d1-truth < d8-sampled. Conclusion:
  determinization gains were NEVER about approximating the hidden-state
  posterior; they are ENSEMBLE VARIANCE-REDUCTION over noisy value/
  rollout estimates. Hidden information is not the binding constraint;
  EVALUATION NOISE is. (Corollary: the intended "information ceiling"
  measurement is unmeasurable by peek — peek inherently destroys world
  diversity — and also unnecessary, since info was not the deficit.)

Strategy update: push ensemble quality, not belief modeling. Launched
probe5 peak-refinement (25g each vs n1024_d16): n1024_d12, n1024_d24,
n1536_d16, n1024_d16_m32. Also still open: the user question on
table-total (gate-aligned) objective.

## 2026-07-07 17:45 — Probe5: n1024_d16 IS the peak; search-shape tuning exhausted

vs n1024_d16 (25g paired): d12 -0.68 ns, d24 -0.24 ns, n1536_d16
-0.49 ns, m32 -0.09 ns. Nothing beats the confirmed config. Serving
optimum stands at **n1024/d16/m16/w0.5 = 98.28 (100g)**.

Next actionable prediction of the eval-noise theory: ensemble the VALUE
ESTIMATOR (checkpoint ensembling at the leaves), same mechanism as
world-ensembling. Implementing bridge-side multi-manifest ensembling
(opt-in), probes: champion+swa and champion+c6-best at n512_d8 25g.

## 2026-07-07 20:25 — Checkpoint-ensemble probes: correlated members don't pay

25g vs solo controls: champion+swa n512d8 +0.38 ns; champion+c6 n512d8
-0.12 ns; champion+c6 n1024d16 -0.10 ns (98.41 vs 98.28 solo, ns).
Interpretation: c6 is EI-taught BY c4 (and swa is the same run) — their
errors are correlated, and ensembles only cancel uncorrelated noise.
The world-ensemble mechanism needs member DIVERSITY.

Overnight chain launched (probe7): (1) champion+c3-M (different data
era, from-scratch) and champion+L-v2 (different architecture+init)
25g probes at n512d8; (2) train two fresh from-scratch M's on the
pooled c6-era corpus with different trainer seeds (16-pass, ~2h each
at the new 0.23 s/step); (3) 4-way diverse ensemble probe (champion +
c3m + freshA + freshB) at n512d8, then n1024d16. Morning readout
decides whether diversity-ensembling is the next confirmed lever.

## 2026-07-08 03:08 — Ensemble lever CLOSED

Full sweep (25g paired vs solo controls): swa +0.38 ns; c6 -0.12 ns;
c6@n1024d16 -0.10 ns; c3m -0.19 ns; L-v2 +0.30 ns; 4-way(c3m+freshA+
freshB) n512d8 +0.18 ns; **4-way n1024d16 -0.78 CI-**. Verdict:
checkpoint output-averaging does not pay. Diverse members trend
mildly positive but never significant; weak/lineage members actively
hurt at the peak config. The world-ensemble mechanism works because
per-world evaluation noise is large and world-independent; model noise
is dominated by SHARED bias that averaging cannot cancel.

Fresh from-scratch M's (3x pooled corpus, 16-pass, seeds 777001/2)
trained as ensemble members; solo n256 battery of seed_a queued — an
M-scale replication of the capacity/data-scale saturation result.

Campaign position: 98.28 (n1024_d16) stands as the measured honest
optimum of this architecture+search. Remaining moves: (1) user ruling
on table-total objective; (2) 1,000-game certification of 98.28
(~24h at 10.6 s/dec); (3) new-research directions (distributional
value head, market-refill chance nodes) with uncertain payoff.

## 2026-07-08 03:41 — Fresh-M solo replicates saturation; overnight program complete

ens_m_seed_a solo (from scratch, 3x pooled corpus, 16-pass, regret-
selected step 18500) at n256, 100g paired vs c4-M: 96.83, -0.125
[-0.50, +0.25] ns. Third independent replication of the training-side
plateau (L-v2, cycle-6, now fresh-M): at this architecture and
objective, ~97.0 at n256 is where trained M's land regardless of data
scale, initialization, or teacher strength.

CAMPAIGN POSITION (morning 07-08): honest measured optimum =
**98.28 mean seat, 100g, c4-M at n1024/d16/m16/w0.5** (11% of games
>=100). All tuning levers measured and closed: capacity, data, labels,
EI, search shape, worlds axis, blend, ensembles. The -1.72 gap needs
either the gate-aligned table-total objective (user decision pending),
or new research (distributional value head, market-refill chance
nodes), or a redefinition of done. John0 and fleet idle pending
direction.

## 2026-07-08 12:15 — Research program launch: table-total, leaf softmix, distributional q (chain on john0)

User approved the post-saturation research agenda (RESEARCH_LOG.md is the
deliverable doc). Implemented and launched, strictly sequential on john0:

1. **--gumbel-table-total** (51e049e): search values = table sum (terminals,
   rollouts); leaf bootstrap = own exact-grounded Q + Σ others' value-head
   finals; unvisited fallbacks shifted onto table scale. Probe: 100g n256/d4
   w0.5 seeds 2026995000+ vs gumbel_cycle4_gate_n256 (96.95).
2. **--gumbel-leaf-softmix τ** (a8e9c32): leaf bootstrap softmax(q/τ)-weighted
   mean instead of max-Q (max of noisy estimates is upward-biased; eval noise
   is the binding constraint). Probes at τ=2, τ=4, same seeds/baseline.
3. **--q-quantiles 8** distributional score-to-go head (pinball loss; serving
   q = quantile mean, bridge unchanged) + **--init-skip-mismatched** warm
   start. Training run full_v3_distq_k8 clones the cycle-6 recipe/data
   exactly (a known-flat control) so the head is the only variable; then a
   100g n256/d4 battery.

Ops: local builds need RUSTC pin (Homebrew rustc 1.85 shadows rustup 1.96);
john0 builds need zig-cc linker env; new pair_verdict.py on john0:/tmp.

## 2026-07-08 10:15 — Table-total v1 probe: CI− (-1.65). Value-head noise suspected; v2 (constant root shift) queued

table_total_n256: cand=95.3025 vs own-seat base=96.9500, delta=-1.6475
CI95=[-1.9959,-1.2991] n=100 CI−. The gate-aligned cooperative objective
LOST at n256/d4 w0.5. Suspected mechanism: v1 computes the other-seats
bootstrap shift from the value head AT EVERY LEAF; the value head has
never been load-bearing (own-seat search reads only q/score-to-go), so
its per-leaf variation injects unvalidated noise directly into the
across-action Q comparison — swamping any cooperation signal. The
rollout half (honest table sums) is unaffected. v2 = compute the shift
once at the root (constant across leaves, zero added variance; within a
depth-1 search the other seats' expected finals barely move). Fleet4
table generation and cycle-7 stay HELD pending the v2 verdict.

## 2026-07-08 11:20 — Leaf softmix: flat at both temperatures (closed)

softmix_t2_n256: 96.9225 vs 96.95, delta=-0.0275 CI95=[-0.379,+0.324] ns
softmix_t4_n256: 96.9375 vs 96.95, delta=-0.0125 CI95=[-0.360,+0.335] ns

Softening the leaf max-Q bootstrap changes nothing at either τ. Reading:
the upward max-bias is common-mode across root actions (each action's
leaf takes a max over a similar interior menu), and common-mode bias
cancels in the argmax comparison; the per-leaf variance reduction is
small next to determinization/rollout noise. Lever closed. Probes now
run ~33 min/100g on warm GPU (pace note for planning).

## 2026-07-08 12:15 — DISTRIBUTIONAL Q HEAD IS CI+ (+0.43): first training-side win since saturation

distq_k8_n256: cand=97.3775 vs scalar-head control 96.9500,
delta=+0.4275 CI95=[+0.0863,+0.7687] n=100 CI+.

full_v3_distq_k8: M, --q-quantiles 8 (pinball loss, serving q = quantile
mean), warm start from champion with --init-skip-mismatched (fresh q
head), otherwise the exact cycle-6 recipe (same corpora, guard-clamped
6250 steps over 300k examples) — the known-flat cycle-6 is the control,
so the quantile head is the only variable. EI at M was saturated for
the SCALAR head; the distributional head un-sticks the training side.
Follow-up armed: n1024/d16 100g confirm vs champion 98.28 (chained after
tta probe). If CI+ there → new champion line + distq EI cycle overnight.

Ops cost note: tablev2+tta chain died silently at 12:14 (eval_request_for_row
was #[cfg(test)]-gated; production TTA path used it; cargo check
--workspace does NOT cover the exporter workspace, and job scripts sent
build output to /dev/null). Fixed, relaunched 13:59. Lesson: always
`cargo build --release --manifest-path cascadiav3/real-root-exporter/...`
as preflight, never silence its output in job scripts.

## 2026-07-08 14:35 — Table-total v2: CI− (−1.05). Serving-side table objective CLOSED

table_v2_n256: cand=95.9000 vs 96.9500, delta=-1.0500 CI95=[-1.408,-0.692] n=100 CI−.

v2 (constant root shift) removed the value-head noise and recovered 0.6
of v1's −1.65, but the objective still loses ~1 point.Residual mechanism:
with a constant bootstrap shift the ranking is bootstrap-identical to
own-seat search; the only live difference is rollout/terminal leaves
scoring the WHOLE table — ~4× outcome variance per leaf at unchanged
per-action signal. Table scoring at serving leaves is a noise multiplier
that outweighs any denial-avoidance signal at this scale. CLOSED at
serving. Training-side table-native labels (cycle-7, staged) remain
theoretically distinct (training averages away label noise) but are
outcompeted for GPU by the CI+ distq line; parked with reasoning.
Overnight slot goes to distq EI-1.

## 2026-07-08 16:10 — Symmetry TTA: flat at 3× cost (closed), with the sharpest negative lesson of the day

tta3_n256: cand=96.9075 vs 96.9500, delta=-0.0425 CI95=[-0.400,+0.315] ns
(8.22 s/dec vs 2.79 baseline — 3× eval cost for nothing).
Cost-matched: tta3 vs n256_d8 (97.25, cheaper): delta=-0.34 ns, trending
worse. CLOSED.

Lesson: rotation barely decorrelates this model's eval error — CGAB
relation-bias attention is built on relative geometry (distances,
pairwise relations), which is largely rotation-invariant, so rotated
frames return nearly the same eval (and the same error). Determinized
worlds pay because they change the EVALUATION PROBLEM (different hidden
futures → genuinely independent value estimates), not the input frame.
Future variance-reduction levers must perturb the problem, not the
representation. The rotation machinery (game-crate transforms +
--gumbel-tta) stays in-tree: correctness-tested and reusable for
training-time augmentation if data diversity is ever the constraint.

## 2026-07-08 18:15 — Distq at champion config: 98.40, +0.12 ns vs 98.28. EI-1 launched overnight

distq_n1024_d16_vs_champion: cand=98.3975 base=98.2800 delta=+0.1175
CI95=[-0.215,+0.450] n=100 ns. 9/100 games >=100.

The +0.43 CI+ at n256/d4 compresses to +0.12 ns at n1024/d16: the
quantile head and the 16-world ensemble are overlapping variance
reducers — where search already denoises, the better head is partly
redundant. distq_k8 is champion-equal at high budget, strictly better
at low budget (and cheaper to serve well: 97.38 at 2.8 s/dec).

**distq EI-1 launched** (pid file logs/gumbel_selfplay_distq_ei1_job.pid):
generation with the distq model n512/d8 w1.0, seeds 2026810000x1250 +
2026910000x125, then --q-quantiles 8 training on new+c6+c5 at
1.0/0.5/0.25, init from distq_k8 (same shape). The open question it
answers: does better-search-from-a-better-head yield better LABELS
(compounding), now that the scalar-head saturation is broken?
Fleet5 (distq model, n256/d4, seeds 2026815000+, 150/host) generating
supplementary shards on john1-4 — stored for a safety-tested low-weight
fold-in, NOT auto-folded (cycle-5 lesson).

## 2026-07-08 22:15 — Official rules correction: wildlife return order pinned; free three-of-a-kind is now a policy action

Audited the engine against the official AEG Cascadia rulebook and corrected
the policy-space contract across the full stack.

1. A drafted wildlife token that is not placed now returns to the cloth bag
   before the end-of-turn market refill. The regression test empties the bag
   and proves that the returned token can supply that same refill.
2. A market with exactly three matching wildlife now exposes two legal
   branches, decline then accept. The engine never chooses for the caller.
   Random, greedy, pattern, MLX, rollout, terminal-improvement, oracle/beam,
   public-tree, API, exporter, and Gumbel policies all value both branches.
3. Gumbel uses separate declined/accepted model rows at the root and interior
   plies, searches both roots, and records the chosen branch, branch count,
   total simulations, and corrected ruleset identity in output artifacts.
4. Rules semantics are now identified as
   `cascadia-base-official-2026-07-08`; exporter config identity is
   `cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_08`.

Scientific ruling: every pre-correction score, paired control, corpus, and
checkpoint used forced acceptance. Preserve those artifacts as historical
architecture evidence, but they are not valid promotion controls under the
corrected action space. EI/fleet jobs already running old code are legacy-only;
greedy, no-search, and Gumbel baselines must be regenerated before research
strength claims resume. Canonical contract: `docs/v3/RULES_CONTRACT.md`.

Verification: Rust workspace 217 passed (one timing harness ignored);
real-root-exporter 39 passed; Python unittest 104 run (69 environment skips);
cluster pytest 109 passed; web lint clean, 7 tests passed, and production build
succeeded. The corrected deterministic rollout golden hash is
`bd39c8b6e42af15ec20837dbc76ba7025889de2cedda5f58c0ab45d3f5d43760`.

## 2026-07-09 01:03 — Chance boundary corrected; legacy john0 EI-1 stopped

The 07-08 rules patch made free three-of-a-kind accept/decline explicit, but
its first search implementation still evaluated the accepted branch after
drawing from the real hidden bag. That let the policy decide whether to
accept after observing the replacement, which is not information available
to a player. This is a second compatibility break, not a cosmetic refactor.

The permanent correction is decision → chance → draft throughout the stack:

1. Greedy, pattern, rollout, lookahead, oracle, and public-tree policies value
   accept/decline over public-hash-derived hidden-order samples, commit the
   branch, and only then rank drafts in the real revealed market.
2. Gumbel searches decline once, estimates accept over an independent
   `market_decision_samples` stream (default `8`), tie-breaks to decline, and
   runs a separate downstream search after a real accepted draw. Interior
   plies enforce the same boundary.
3. The hidden-order regression constructs two identical public
   three-of-a-kind roots whose actual accepted replacements differ and proves
   the accept/decline result is identical. The downstream action may differ;
   the pre-draw decision may not.
4. Rules identity advanced to `cascadia-base-official-2026-07-09` and
   `cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09`.
   Benchmark reports now reject mismatched runtime identities and record the
   exact deployed Git revision.

At user authorization, stopped the old forced-refresh john0 process groups:
EI-1 generation PGID `1225249` (825/1,250 games, 66k roots, no completed
artifact) and its queued battery PGID `1228689`. Verification at 01:03 EDT
found no self-play, Gumbel benchmark, or exporter process remaining on john0.
Those partial roots are legacy-only and will not enter training or promotion.

Validation completed before this entry: 61 search tests, 33 simulation tests,
and all 40 exporter tests passed, including both the hidden replacement trap
and forced accept/decline policy tests. The new deterministic rollout golden
hash is `24ca921ec767b442acbc5495c9fbacd8790beb0346c94b795625aaf8194e2b7a`.
The final pre-deploy gate also passed the complete 217-test Rust workspace,
104 Python tests (45 fixture-dependent skips), 109 cluster tests, web lint,
all 7 web tests, and the production web build.

## 2026-07-09 01:23 — Corrected-rules rebaseline launched on john0

Committed and pushed the rules correction, report provenance gates, and
idempotent rebaseline launcher. Deployed source revision
`863c696dd41e5b4c7e26385851201072a38c22f4` to john0 and verified the rules
constants in place.

john0 had Rust but no system C compiler or libc development package, and no
passwordless sudo. The first launch therefore failed loudly during the release
rebuild before creating game data. The permanent target-build fix pins Zig
`0.13.0` in the user account, verifies the official tarball SHA-256
`d45312e61ebcc48032b77bc4cf7fd6915c11fa16e4aad116b66c9468211230ea`, and
uses the checked-in `zig-cc-linker.sh` adapter. A clean release rebuild then
produced a fresh x86-64 Linux binary from the deployed revision.

The corrected job is live as PID `1262885` / PGID `1262878`. Its n16/d2
one-game smoke passed with ruleset
`cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09`, exact
source revision, and `market_decision_samples=8` recorded in the report. The
ordered 100-game program on fresh seeds `2027070900..2027070999` is:

1. cycle4 no-search policy/q plus greedy baseline;
2. cycle4 n256/d4;
3. distq_k8 n256/d4;
4. cycle4 n1024/d16;
5. distq_k8 n1024/d16.

The Gumbel legs share identical seeds and policy settings so the scalar versus
distributional-Q differences remain pairable. No strength or promotion claim
is valid until the corrected reports finish.

## 2026-07-09 01:38 — Rebaseline restarted with permanent refresh-decision telemetry

The first corrected no-search floor completed, but a pre-Gumbel report audit
found that the harness discarded Gumbel's temporary per-ply JSONL after
reducing it to scores. That preserved final strength but lost the evidence
needed to audit how often the newly corrected accept/decline policy actually
fired. Stopped the still-young cycle4 n256 leg before it produced a report.

Revision `d20daf44dc6aa4aad3d03c6ccb7d3a21c3013135` permanently adds:

- persistent Gumbel decision JSONL for every benchmark;
- report-level accept, decline, opportunity, and acceptance-rate counts;
- chance-sample and total simulation-overhead accounting;
- the same accept/decline telemetry for greedy/no-search games;
- paired t/bootstrap confidence statistics in the no-search report.

The replacement smoke passed with 80/80 decision rows retained. It encountered
7 legal optional refreshes and chose accept 5 times / decline 2 times, direct
runtime evidence that the policy is exercising both branches. All opportunities
used the registered 8 chance samples. The corrected market decision added 976
simulations above 1,280 chosen-branch simulations in this n16 smoke, a `76.25%`
whole-game search overhead that must be included in serving-cost comparisons.

The auditable battery is live as PID `1265148` / PGID `1265141` and is
regenerating the same no-search floor before the four Gumbel legs.

The regenerated no-search floor completed bit-for-bit on scores with 24,000
decision rows retained. Over 100 paired seeds: greedy `87.5450`; cycle4 policy
head `91.8425`, delta `+4.2975`, t-CI `[+3.8705,+4.7245]`; cycle4 Q head
`90.8925`, delta `+3.3475`, t-CI `[+2.8507,+3.8443]`. Refresh telemetry was
policy accept/decline `594/352`, Q `636/364`, greedy `1005/398`. Caveat: this
interactive no-search harness delegates the pre-draw refresh decision to
greedy-v1, then applies the model head to the revealed draft menu. The Gumbel
legs test the search policy's own refresh choice. cycle4 n256/d4 is now live.

No-search mechanism decomposition (paired by seed) is CI-positive in every
category. Policy minus greedy: wildlife `+0.8100` CI
`[+0.4458,+1.1742]`, habitat `+2.3900` CI `[+2.1592,+2.6208]`, retained
Nature Tokens `+1.0975` CI `[+0.8988,+1.2962]`. Q minus greedy: wildlife
`+0.6075`, habitat `+1.8775`, Nature Tokens `+0.8625`, all CI-positive.
Interpretation: most of the model's direct-policy advantage comes from
longer-horizon habitat structure and resource economy rather than immediate
wildlife-score greed.

## 2026-07-09 — Exact final-personal-turn K1 implemented; causal smoke flat, frontier 8.86x faster

Implemented the first rigorous slice of `docs/v3/RADICAL_DIRECTIONS.md` #1.
`--gumbel-exact-endgame-turns 1` recognizes a seat with one personal turn
remaining, enumerates the complete legal menu (ignoring the normal serving
cap), and selects by exact engine score with no model or simulations. Optional
three-of-a-kind refresh still uses hidden replacement samples before the real
draw is revealed. K>1 and table-total objectives fail loudly because neither
has the final-own-score identity.

The mode is wired through exporter CLI/help, policy games, self-play rows,
suggestions, benchmark orchestration, reports, and persistent decision
telemetry. Exact rows carry a one-hot improved policy, zero variance, one
chosen visit, and `exact_endgame=true`. The purpose-built comparator refuses
mismatched rules/source/checkpoint/seeds/search, requires 80 decisions per
seed, identical action/refresh traces through ply 75, four zero-simulation K1
decisions per game, and seat-0 non-regression before it will report a result.

Three MPS attempts were intentionally discarded rather than laundered into a
score:

1. john4 control vs john2 K1 diverged at ply 5 (cross-host MPS variance).
2. Same-host, two-worker K1 diverged at ply 24 despite identical seeds and
   settings (cross-game request batching changed numerical choices).
3. Same-host, four-worker K1 emitted
   `Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`;
   the entire process group was stopped and no report was used.

The valid engineering smoke ran serially on john4 (one worker/bridge), seeds
`2027071360..2027071361`, cycle4 checkpoint, n16/top8/d2, w1.0, four market
samples. Comparator verdict:

- plies 0–75 identical for both seeds;
- baseline/K1 mean seat score `92.25 / 92.25`, paired deltas `0 / 0`;
- 8/8 exact decisions, zero simulations; seat-0 deltas `0 / 0`;
- 6/8 final actions changed, all to equal-scoring alternatives;
- exact-frontier time `4.212s → 0.476s` (`8.86x` faster);
- whole-arm mean decision `1.584s → 1.565s` (`1.2%` faster), wall
  `254.5s → 251.4s` (`1.3%` faster); P95 did not improve.

Verdict: **score-inconclusive engineering pass, not promotion evidence.** The
permanent `run_exact_k1_gate.sh` regenerates both arms from one new revision at
100 fresh corrected-rules seeds and n256/d4, then applies the same trace and
solver invariants. A checksum-verified waiter is armed on john0 and will not
install the revision-marked `main` snapshot or start the gate until both the
current rebaseline and its verdict watcher finish. K2 remains gated on that
result.

Fleet audit during this work: john2–john4 were still running pre-correction
Fleet5 binaries after roughly nine hours. All three process trees were killed
and verified absent; no Fleet5 shard artifacts existed. john1's Fleet5 pid was
stale and no process/artifact existed. Minis remain generation/engineering
hosts, never gate hosts.

## 2026-07-09 04:00 — Optional-refresh search is the serving bottleneck; sample-4 promoted to a CUDA gate, not to production

Profiled the first 65 complete games of the live corrected-rules cycle4
n256/d4 arm on john0. Across 5,200 decisions, mean/P50/P95/P99 latency was
`11.782 / 6.325 / 60.112 / 74.551` seconds. The 611 decisions with an optional
refresh averaged `55.452s`; the 4,589 ordinary decisions averaged `5.968s`.
The refresh policy added 1,343,744 simulations above 1,331,200 chosen-branch
simulations. Action count was uncorrelated with latency (`r ~= -0.003`). A
live snapshot had the RTX 5090 at about 83% compute utilization but only 17%
memory utilization, with the exporter using roughly 7.4 CPU cores and the
bridge 4.1. Optional-refresh chance work, not legal-menu width, is the current
serving bottleneck.

Added `compare_market_samples`, a purpose-built causal and provenance gate.
It requires identical corrected rules, source revision, manifest path, seeds,
control, and all search settings except sample count; validates recorded sample
telemetry; requires all 80 decisions per seed; and rejects action divergence
before the first point where sample count could affect computation. It reports
per-opportunity and whole-game cost separately because a changed policy can
alter the number of later three-of-a-kind markets. Promotion scale is at least
100 paired games. The preregistered sample-4 gate requires score t-CI lower
bound `>= -0.25` and whole-decision speedup `>= 1.15x`.

The serial MPS screen itself caught two more invalidation modes:

1. john2 sample-8/sample-4 diverged at seed `2027071503`, ply 3, before any
   optional refresh. Its entire frontier was discarded.
2. john3's reduced-sample arms diverged from sample-8 at seed `2027071504`,
   ply 16, also before any optional refresh. All john3 scores were discarded.

The only causally valid screen was the serial two-seed john4 set (cycle4,
n16/top8/d2, w1.0, seeds `2027071500..1501`):

| Samples | Mean | Delta vs 8 | Mean decision | Speedup | Opportunities | Refresh sims/opportunity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 93.875 | - | 1.866s | 1.000x | 11 | 138.18 |
| 6 | 92.750 | -1.125 | 1.883s | 0.991x | 16 | 104.00 |
| 4 | 93.500 | -0.375 | 1.476s | 1.264x | 10 | 76.80 |
| 2 | 93.375 | -0.500 | 1.526s | 1.223x | 28 | 42.86 |

Sample-4 is the sole non-dominated reduced point in this tiny screen. It is
not strength evidence and is not adopted. `run_market_samples_gate.sh` runs a
fresh 100-seed CUDA candidate at n256/d4 and reuses the exact-K1 sample-8 arm
only after full contract validation. It is queued after exact K1 on john0;
failure preserves sample-8. Only a passing same-budget gate would justify a
separate cost-matched higher-n test.

## 2026-07-09 04:25 — Corrected cycle4 n256 baseline complete: 97.0675; refresh choice doubles search work

The first corrected-policy Gumbel baseline completed on john0 from source
`d20daf44dc6aa4aad3d03c6ccb7d3a21c3013135`: cycle4 M, seeds
`2027070900..2027070999`, n256/top16/d4, blend 0.5, K16 interior, eight market
samples. Report and all 8,000 decision rows passed rules/source/seed coverage
validation. Copied local/remote SHA-256 values matched: report
`928bdc78955523d79c35038d76c7ad55e48d2dd5dc4dbcfe647e36daf37a1711`,
decisions
`63528a5154c1a4bdeb4aa04226133ae6d794d153450e1127b87c44893a1cda56`,
and Markdown
`414c68aabe9d69d3a1940fadf2e2431109de05529c8e815842ed24bc89669321`.

Result: mean seat `97.0675`, P50 `97.0`, P90 `100.1`, and 2/100 game means
at least 100. Search encountered 952 optional refreshes, accepted 565 and
declined 387 (`59.35%`). Mean decision time was `11.729s`; refresh decisions
averaged `54.908s`, ordinary decisions `5.896s`. The chosen branches consumed
2,048,000 simulations; evaluating the market choice added 2,094,336 more.
This arm is a corrected baseline, not promotion evidence; the paired distq
arm is in flight.

Audit limitation: the deployed reducer retained total scores but deleted its
temporary `gumbel_game_done` category rows. No wildlife/habitat/Nature
mechanism is claimed for this arm. A sidecar now preserves each in-flight
distq seed file. Permanently, every runner writes a complete, seed-ordered
`*_games.jsonl`; reports embed per-seat raw breakdowns and overall/by-seat
category means, and the writer refuses partial or duplicate seed coverage.

## 2026-07-09 04:25 — Raw-request throughput diagnostic: 2.40x tiny/M (SUPERSEDED)

**Superseded at 04:50. Do not use these ratios as serving evidence.** The
fixed roots were human-auditable raw records, so the timed loop executed
legacy Python feature extraction. Live Rust search advertises and sends
precomputed `packed_features`; the corrected production-path entry follows.

Added a deterministic end-to-end bridge throughput benchmark and shell
runner. It exercises root collation, padded feature/relation tensors,
CascadiaFormer execution, and packed-response construction; records host,
platform, relevant environment, roots/manifests/weights hashes; and requires
stable response digests across iterations. The fixed corrected-rules roots
have 90/225/216/414 actions and SHA-256
`534d35fe625b7c4ee248a58ffd1cb265be127cff93eafdc0fe48fbcddfbaa35f`.
XS (`d_model=256`, 6 layers, 8 heads, FFN 1024; 5,121,607 parameters) is now
a first-class model/trainer config. Synthetic shapes are engineering-only,
not trained strength evidence.

Seven measured iterations after two warmups on john2–john4 MPS produced
identical output digests and close host rates. Three-host mean roots/s:

| Model | Parameters | batch 1 | speedup | batch 8 | speedup |
|---|---:|---:|---:|---:|---:|
| trained cycle4 M | 88,169,543 | 37.754 | 1.00x | 99.736 | 1.00x |
| trained EI-0 S | 15,016,007 | 121.619 | 3.22x | 183.178 | 1.84x |
| synthetic XS | 5,121,607 | 183.158 | 4.85x | 204.743 | 2.05x |
| synthetic tiny | 67,847 | 234.785 | 6.22x | 239.585 | 2.40x |

Verdict: the original proposal that a 3–10M student would buy n8k–n16k is
not supported by the current batched MPS serving path. Even a roughly 1,300x
parameter reduction buys only `2.40x`; fixed collation/encoding/relation and
response work is the asymptote. Run the exact same probe on john0 CUDA before
distillation. If CUDA plateaus similarly, optimize request amortization and
rollout topology first; if CUDA has materially larger size leverage, distill
XS and measure an equal-wall-clock score frontier.

## 2026-07-09 04:50 — Production-packed throughput correction reopens smaller-model/larger-search

Root-cause audit of the surprising `2.40x` ceiling found that the benchmark
had passed audit roots directly to `_model_eval_batch`. That exercises raw
Python token/action/relation feature extraction, but production Rust detects
the bridge's `packed_features` capability, computes those arrays before the
request, and removes the raw dictionaries. The original measurement was
internally repeatable but answered the wrong serving question.

Source `543ba6e5` now makes `production-packed` the default, performs the
conversion before timing, records the canonical prepared-payload hash, and
retains `--root-format as-is` only as an explicit legacy diagnostic. Source
root SHA-256 is
`534d35fe625b7c4ee248a58ffd1cb265be127cff93eafdc0fe48fbcddfbaa35f`;
prepared payload SHA-256 is
`e4546f632ddde46e3a5e9ded40f04b3cf78c4be7f0497772ead31aefa688a1f5`.
The response digests were unchanged between raw and packed paths on the four
roots, so the correction changes the measured path, not model semantics.
MPS timing also now synchronizes MPS at phase boundaries; previously its
asynchronous forward work was mislabeled as device-to-host copy.

Uncontended seven-iteration runs after two warmups on john2–john4 agreed
closely. Three-host means for representative repeated four-root mixes:

| Model | Parameters | batch 8 roots/s | speedup | batch 32 roots/s | speedup |
|---|---:|---:|---:|---:|---:|
| trained cycle4 M | 88,169,543 | 144.996 | 1.00x | 153.743 | 1.00x |
| trained EI-0 S | 15,016,007 | 443.174 | 3.06x | 518.879 | 3.38x |
| synthetic XS | 5,121,607 | 700.524 | 4.83x | 866.592 | 5.64x |
| synthetic tiny | 67,847 | 1,427.867 | 9.85x | 2,100.427 | 13.66x |

Verdict: smaller-model/larger-search is reopened, but with bounded arithmetic.
A credible 5M XS shape buys roughly `5-6x` in-bridge throughput on MPS; the
`10-14x` ceiling belongs to a near-zero model with no plausible strength
claim. Bridge rates also omit Rust search and game-engine work. The immediate
kill-test is therefore an end-to-end trained checkpoint calibration, not an
XS training run: three minis are running same-host single-seed M n64/d4 versus
trained S n192/d12 at sample 4. CUDA and paired score remain mandatory before
distillation or promotion.

## 2026-07-09 05:05 — Trained S n192 does not preserve the 3x bridge multiplier end to end; equal-wall estimate n130

To separate in-bridge throughput from whole-search throughput, john2–john4
each ran one same-host serial pair on distinct seeds `2027071700..1702` under
the corrected rules. Both used one worker, top16, blend 0.5, K16 interior,
and four market samples. Control was trained cycle4 M at n64/d4; candidate was
trained EI-0 S at n192/d12, matching the approximately `3x` batch-8 bridge
ratio. This is a three-game engineering calibration, never a strength gate.

| Host | M score | S score | delta | M wall | S wall | S/M wall |
|---|---:|---:|---:|---:|---:|---:|
| john2 | 94.75 | 95.00 | +0.25 | 478.73s | 684.71s | 1.430x |
| john3 | 98.00 | 96.25 | -1.75 | 350.40s | 628.14s | 1.793x |
| john4 | 95.50 | 95.25 | -0.25 | 424.68s | 538.83s | 1.269x |

Aggregate M/S means were `96.083 / 95.500` (delta `-0.583`, far too small for
inference) and `417.94 / 617.23s` per game. The candidate used 81,408 total
simulations versus 24,256 (`3.356x`), partly because its trajectories saw more
optional-refresh work. The aggregate wall ratio was `1.477x`; linearly scaling
the S budget gives n130 at equal wall. A rounded S n128/d8 follow-up was
launched
on the same seeds. Aggregate validation/report-ledger SHA-256:
`393661416b7630f5e4e1e3d016dc4ed40f24653a8aa446f7904be87347fc54a9`.

The S n128/d8 follow-up completed with all three 80-decision traces and raw
category ledgers valid:

| Host | M n64 score | S n128 score | delta | M wall | S wall | S/M wall |
|---|---:|---:|---:|---:|---:|---:|
| john2 | 94.75 | 90.00 | -4.75 | 478.73s | 585.81s | 1.224x |
| john3 | 98.00 | 95.25 | -2.75 | 350.40s | 369.64s | 1.055x |
| john4 | 95.50 | 96.50 | +1.00 | 424.68s | 395.68s | 0.932x |

Aggregate S n128 wall was `450.38s` versus M's `417.94s` (`1.078x`), so the
rounded budget did hit the intended wall neighborhood. Total simulations were
56,320 versus 24,256 (`2.322x`). Mean score was `93.917` versus `96.083`, a
paired `-2.167` over three games. This is not statistically usable, but it is
a negative directional screen and does not authorize XS distillation. Hold
for the CUDA production-packed multiplier and corrected-distq verdict. The
validated nine-arm summary SHA-256 is
`44957b49d4fa5b2dd6df953f9419a210fc9e228aebbc4ff42b5556088f40921e`.

## 2026-07-09 06:05 — Quantile-risk serving changes trajectories but has no gate-scale signal

Purpose: test the cheapest remaining distributional-Q ablation before
spending training compute. The K8 checkpoint already emits eight per-action
score-to-go quantiles; established serving averages them. Source `ef5499b7`
adds explicit q25/q50/q75 selection, linear interpolation at the centered
quantile levels, per-action monotone rearrangement for crossed heads, mode
provenance in the bridge hello and benchmark report, scalar-checkpoint
rejection, and a fixed-root diagnostic. Default mean serving is unchanged.

The exact-revision fixed-root probe ran on john2 MPS against 160 deterministic
corrected-rules greedy-policy roots with full 256-action menus (40,776 total
actions). Root source SHA was
`39f2285a236a184b6f11de3233f108057d9bac680c356b1161b989c2e0c05ff8`;
selected packed payload SHA was
`1ef2b966f566462b4ef38dab63faf1bc6df104e1fe1fb357906626f22b31ec25`;
manifest SHA was
`02fa7ccab88e2313363882d5251d9b44ae364a05eb23f4045725803da9bd6533`
and weights SHA was
`8d0272c971bcaae407fd23f3f47daae6fa50d8326a4af76243046c038c041f40`.
All 285,432 adjacent quantile pairs were ordered already, so rearrangement was
a safety invariant rather than an active correction. Direct derived-Q argmax
flip rates versus mean were q25 `5/160 (3.125%)`, q50 `4/160 (2.500%)`, q75
`3/160 (1.875%)`. Average within-model mean-Q regret was
`0.000065/0.000096/0.000058`; the head's risk shift is mostly common-mode
rather than action-ranking information. Report SHA:
`0c57c8fa1b0f1def6c70a038325885da499e148631f3ec3fc0009b2fec1c0f9b`.

The end-to-end kill test used the same exact source, rebuilt exporter SHA
`05118990835d9517e60a85aa665eaff2559cdb0a4a4db784434585c4cf82a250`,
distq M, n64/top16/d4, blend 0.5, K16 interior, four market samples, one
worker, and fresh seeds `2027071900..1902`. Every report, 80-ply decision
ledger, and raw score-category game row passed source/rules/search/seed and
category-sum validation.

| Host/seed | Mode | Mean | Candidate | Delta | First divergence | Wall ratio |
|---|---|---:|---:|---:|---:|---:|
| john2/1900 | q25 | 93.75 | 96.00 | +2.25 | 2 | 0.918x |
| john3/1901 | q25 | 95.00 | 94.75 | -0.25 | 20 | 1.332x |
| john4/1902 | q25 | 96.25 | 95.00 | -1.25 | 2 | 0.877x |
| john3/1901 | q50 | 95.00 | 95.00 | 0.00 | 21 | 0.925x |
| john4/1902 | q75 | 96.25 | 95.00 | -1.25 | 2 | 1.230x |

q25 pooled only as a directional engineering screen: `95.25` versus `95.00`,
paired delta `+0.25`, n=3, 95% t-CI `[-4.228,+4.728]`; mean wall ratio
`1.042x`. The first seed's apparent win did not replicate. Verdict: do not
spend john0 time on a standalone q-risk gate and keep production at the mean.
The modes remain useful as essentially free, genuinely trajectory-diverse
league personalities if the corrected-rules distq/EI line survives. The
fail-closed 32-artifact aggregate SHA is
`5304b88265c7d698635be8ba4d08b2e85dcf22654b563b3782b60aa96e71f42b`.

## 2026-07-09 06:38 — Shared-batch scaling has a shallow knee; execution provenance hardened

Purpose: use the idle mini fleet to determine whether higher parallel-game
concurrency can fill the live john0 utilization gap without changing policy.
A 30-second sample of the corrected distq n256/d4 jobs12 CUDA workload found
mean SM utilization `65.57%` (range `1-89%`), mean power `353.5W` of 600W,
2,481 MiB framebuffer allocation, and a contemporaneous `55.6%` CPU-idle
snapshot. The box is neither memory-, power-, thermal-, nor CPU-saturated.

The first MPS attempt was stopped before publication because reports did not
record `--batch-runner` or jobs. Source `fbe3f2d2` fixes that permanently:
every Gumbel report now records execution runner, requested jobs,
parallel-game cap, shared-bridge topology, device, and SHA/size identities for
the exporter, manifest, and weights. Candidate-only runs no longer falsely
label themselves paired rollout comparisons. The full Python suite passed
`116` tests (`45` expected fixture skips).

The exact-source experiment then used four identical corrected-rules seeds
`2027072000..2003`, distq M, n16/top8/d2, blend 0.5, K16 interior, four market
samples, fused CGAB, and one shared MPS bridge. jobs1 and jobs2 ran serially on
john2; jobs2 was independently replicated on john3; jobs4 ran on john4. A
jobs4 default-2M-cell-budget arm isolated chunk size from the production-CUDA
16M override.

| Arm | Wall | Speedup | Mean decision | Latency ratio | Action diffs |
|---|---:|---:|---:|---:|---:|
| jobs1 / 16M | 573.83s | 1.000x | 1.790s | 1.000x | 0 |
| jobs2 / 16M same-host | 500.26s | 1.147x | 3.047s | 1.702x | 0 |
| jobs2 / 16M replica | 503.38s | 1.140x | 3.053s | 1.706x | 0 |
| jobs4 / 16M | 486.13s | 1.180x | 5.589s | 3.122x | 0 |
| jobs4 / 2M | 488.73s | 1.174x | 5.640s | 3.151x | 0 |

All five arms produced `93.875` mean and identical actions at all 320 plies;
maximum root-value drift was `1.6e-5`. jobs2 replication differed by `0.62%`
wall. Cell-budget reduction changed jobs4 wall only `+0.54%`: requests were
already below the default chunk threshold. Verdict: jobs2 is the MPS knee;
jobs4 adds only `2.9%` throughput above it while greatly increasing latency
and memory pressure. No CUDA extrapolation and no live-chain modification.
The equivalent john0 experiment is jobs12/16/24 during an engineering window,
with action parity mandatory. Validated 20-artifact summary SHA:
`7d4fb02d1432a8a83c85ee1b123b0a842ce139e92703c9d9932a579d7f163d02`.

## 2026-07-09 06:48 — Corrected-rules n256 is inconclusive; fixed-chunk long tail removed

The first promotion-scale corrected-rules comparison completed on john0:
identical seeds `2027070900..2027070999`, n256/top16/d4, depth 1, blend 0.5,
K16 interior, eight hidden replacement samples, jobs12, one shared CUDA
bridge. Scalar cycle4 scored `97.0675`; distq-k8 mean serving scored `97.3075`.
The same-seed game-mean delta is `+0.2400` (SE `0.1784`, 95% t-CI
`[-0.1139, +0.5939]`, bootstrap CI `[-0.1000, +0.5950]`; 52 positive, 7 tied,
41 negative). Verdict: retain cycle4. The gain is not significant and the
100-point target remains unmet. These are candidate-only arms; d20 incorrectly
labels them as paired rollout-search comparisons, a provenance bug fixed by
`fbe3f2d2` for future reports.

Artifacts were fetched byte-for-byte and validated: all 100 seeds, exactly 80
decisions per seed, ruleset
`cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09`, source
`d20daf44`, and equal search settings. The interim comparison includes exact
input, remote exporter, manifest, and weights hashes. Its JSON SHA is
`287555fb6c233a4e7e14d7e362c7f796ebd35dd4f2b2558b1fd9e12c0b3dbdb8`.

The arm also isolated the utilization tail. The old exporter assigned 12
fixed contiguous seed chunks. At 95/100 only five games remained active and
GPU utilization fell to 18%; completed workers could not claim remaining
seeds. Observed per-seed decision times reconstruct the reported 9,016.7s
wall within 0.03% (9,014.5s). Dynamic list scheduling predicts 8,380.2s,
`1.076x` faster, before contention effects.

The permanent fix replaces fixed chunks with a bounded atomic seed queue in
all affected model-backed pipelines. Workers retain their bridge/session and
eval cache; `--model-sessions` remains the hard concurrency cap. A
deterministic test proves backfill while a long seed remains blocked, and the
full 44-test exporter suite including exact batch/single policy parity passes.

The exact-revision MPS check compared static `fbe3f2d2` to dynamic
`d7f6e487` on the same four seeds, jobs2, model, and search. All 320 actions
and scores matched; maximum root-value drift was `1.56e-5`. Wall was
`500.257s -> 503.113s` (`+0.57%`) while mean decision time was
`3.046829s -> 2.959602s` (`-2.86%`). This tiny static assignment was already
balanced, so the result establishes policy parity/nonregression rather than a
throughput gain. Dynamic source archive SHA:
`574a691df3f2ebbaa04d650dee5dce8f94f1f6ef064158f439cecbad32a34e4f`;
validation artifact SHA:
`e738e6a9948630ddc7a76a54fefc7d08bf0d9e417bda2ceb40aaa5a1c9958f0d`.

The broad workspace gate exposed and fixed a separate provenance defect.
`cascadia-provenance` recursively hashed all of `cascadiav3`, including 10 GiB
of checkpoints, 879 MiB of Rust build output, reports, and logs. A generated
file changed between its two passes, so the stability test failed after
`300.38s`. Source identity now hashes `git ls-files --cached --others
--exclude-standard` within the registered source roots; non-Git archives use
explicit generated-directory exclusions. Ignored-output invariance and
archive-fallback tests were added. All three provenance tests pass in `0.15s`.
This changes only source bookkeeping; checkpoint/report identities remain
separate SHA-addressed fields.

## 2026-07-09 07:48 — Pairwise labels are plentiful but confidence-limited; tensor provenance promoted to v3

Purpose: decide whether the proposed pairwise action comparator has a viable
supervision surface before changing the model, and close the artifact gaps
found while inspecting real Gumbel tensors.

Three idle Mac minis each generated one corrected-rules, all-seat Gumbel game
from exact source `752ba894`, cycle4 M, n64/top16/d4, depth 1, blend 0.5,
K16 interior, four market samples, exploration on, and exact final-personal-
turn solving enabled. Seeds were `2027073000..2027073002`; each shard contains
80 roots. Generation took `590.9 / 425.7 / 447.3s`; persistent-bridge eval
dedup saved `45.1% / 44.8% / 46.1%` of requested rows. Remote and fetched NPZ
SHA-256 values matched exactly:

- seed 2027073000: `092e76121a3530f5eb9ab1bd9ec5d4e90bd4bf85663284626b37395fe3eb2a36`;
- seed 2027073001: `87be982632071f73ac41fa2f79a792ca822b3ba359fcc5b0f7c4898902a130d7`;
- seed 2027073002: `024d8f1235ecec4272e05ce06121cae48205efe8620982d3bf6019945d3bcad5`.

Across 240 roots, 228 (95.0%) had at least two valid search actions. There
were 3,660 valid actions and 27,360 pair labels, or 114 pairs/root and a
projection of 11.4M pairs per 100k-root cycle. Absolute margins are abundant:
62.51% are at least 0.5 points and 36.95% at least 1.0. Confidence is the
constraint: only 6,384 pairs (23.33%) have at least two samples on both
actions; among those, 38.20% reach SNR 1.0 and 14.58% reach SNR 1.96. The
top-two margin median is 0.198, while top-two SNR median is 0.459. Verdict:
proceed with an antisymmetric comparator only with q-valid, count, margin,
and SNR filtering/weighting. Never train blindly on every pair or interpret a
one-sample zero variance as certainty. This audit is engineering feasibility,
not strength evidence. Combined audit JSON SHA-256:
`a22b8b6156b45e376cc4a7f877c9bcc788ef18170c950364e6ae5f342e80aba1`.

Inspection also found the old v2 NPZ contract omitted exact-endgame flags,
ruleset/source identity, several search switches, execution topology, teacher
artifact hashes, and used a fixed fake creation date; its seed-domain mode
was also wrong. The three shards therefore remain audit-only even though the
immutable launch evidence is known. New Gumbel generation uses
`cascadiav3.expert_tensor_shard.v3`: required explicit `exact_endgame`, exact
source revision, ruleset, complete search/execution contract, exporter
SHA/size, teacher manifest/weights SHA/size, and real creation time. Filtering
and relation-tail materialization preserve v3; malformed v3 fails closed, and
fallback/unverified teachers cannot enter a training corpus. The audit CLI is
covered on Python 3.9+ despite the repository's 3.12 runtime contract.

## 2026-07-09 08:43 — Confidence-aware pairwise comparator learns pairs but fails the routing gate

Purpose: run the bounded kill test authorized by the 240-root label audit.
The hypothesis was that an antisymmetric action comparator could use reliable
completed-Q differences more efficiently than another scalar-value target.
This entry is offline routing evidence only; no gameplay was run and no
champion changed.

Fresh training-eligible v3 data came from exact source `0f107219`, corrected
rules, cycle4 M, n16/top8/d2, depth 1, blend 0.5, K8 interior, four optional-
refresh samples, exploration on, exact K1, root menu 256, and one shared MPS
bridge. John2/3/4 generated seeds `2027073100..09`, `2027073110..19`, and
`2027073120..29`; each block has 10 games / 800 roots and took
`1377.9 / 1433.8 / 1378.6s`. Bridge dedup saved `26.0% / 26.1% / 26.2%` of
requested rows. The raw NPZ hashes are:

- 3100: `ab5d829371d92775f6ead31e89d4426185d6d6cf469b16897b6481c608e08807`;
- 3110: `830566c82fafc19344a3ea84afb070f0267fee74da7afd284703e6f50c4a6012`;
- 3120: `38b43e00509fc036965b874864703c88d2c12571cbd1a8468d9c6ef36e0e68a4`.

Every shard independently reloaded as v3, training-eligible, source/rules
matched, and contained 40 explicit exact-endgame roots. Across 2,400 roots,
2,280 (95%) had at least two valid search actions. There were 18,360 valid
actions and 63,840 undirected pairs. Margins were plentiful (82.10% at least
0.25; 67.36% at least 0.5), but only 13,680 pairs (21.43%) had variance
estimates on both actions; 43.25% of that subset reached SNR 1 and 17.21%
reached SNR 1.96. Median top-two margin/SNR were `0.3192 / 0.6108`.

Top-64 filtering preserved the selected action with zero drops; relation-tail
materialization and the Q semantic invariant passed for all three blocks. The
resulting hashes were `db29becc...` (3100), `c630ae20...` (3110), and
`e942694c...` (3120). Blocks 3100+3110 were the fixed training split; 3120
was never sampled by the optimizer.

The head-only fit warm-started cycle4 M, froze all 88.17M established
parameters, and trained exactly 99,072 comparator parameters (rank 64) for
500 steps / five example passes, batch 16, AdamW LR 1e-3, on john2 MPS. Pairs
required q-valid, count at least 2/action, margin at least 0.25, SNR at least
1, and were capped at 32 undirected pairs/root with both orientations. The
first launch exposed a pre-step MPS metrics bug: the trainer attempted an
on-device float64 cast, unsupported by MPS. Commit `03296681` moves the single
stacked metrics transfer to CPU before exact float64 aggregation and adds an
actual-MPS regression test. No failed-run checkpoint or metric entered the
experiment.

Locked validation selected step 350: pairwise loss `0.564040` and accuracy
`65.972%`, versus initialized-head step 1 `0.688984 / 60.424%`. The selected
manifest/weights hashes are `b281f813...` and `4a9adb35...`; the complete
training report hash is `2415a4fc...`. This establishes that the head learned
the pair labels, not that it improved decisions.

The decisive probe used exact source `2c3997e4` (archive SHA
`2f66c633dae943710d57f136e6832f50b85fb0e52516fc495e14dd5a298df484`)
and all 800 untouched 3120 roots. All modes used the same incumbent-policy
top-16 mask, but that mask was computed inside the top-Q-with-selected 64-action
training tensor, not the unfiltered legal menu. This is therefore an
optimistic restricted-surface reranking gate, not exact serving evidence. On
that surface the mask covered the completed-Q best on 702/795 roots with any
valid candidate (`88.302%`). 206 roots passed the within-mask top-two
count/margin/SNR gate. Comparator pair accuracy was `66.960%` unweighted and
`69.466%` confidence-weighted over 3,862 directed pairs.

| Routing mode | Top-1 | Mean completed-Q regret | Delta vs logits |
|---|---:|---:|---:|
| incumbent logits | 30.583% | 1.1496 | reference |
| pairwise Borda | 31.553% | 1.2121 | top-1 +0.971 pp; regret +0.0625 |
| logits + pairwise | 30.583% | 1.1711 | top-1 +0.000 pp; regret +0.0215 |

For pure Borda, candidate-only/logits-only correct counts were 12/10. The
paired top-1 delta 95% bootstrap CI was `[-3.398,+5.340]` percentage points;
regret-delta CI was `[-0.0507,+0.1781]` (positive is worse). The sum mode
changed only four top-1 outcomes (2/2 discordance), with top-1 delta CI
`[-1.942,+1.942]` points and regret-delta CI `[-0.0524,+0.0896]`. Probe JSON
SHA: `92834d4e6e523857383132f0621d4f2d833ec29fc31c132c611fe57baa4e0d8c`.

**Verdict: kill this serving branch before gameplay.** It learned pairwise
labels, but its two extra top-1 hits are noise-sized and paid for by worse
average completed-Q regret; the additive route did nothing directionally.
Keep the schema, comparator, safe top-K serving, and probe as reusable
research infrastructure, but keep incumbent logits in production and do not
spend john0 time on this checkpoint. Commit after this run makes the pairwise
probe label filtered action surfaces explicitly; exact full-menu candidate
recall requires `torch_policy_candidate_probe`.

All fetched artifacts are checksum-verified under ignored path
`cascadiav3/reports/pairwise_v3_n16_20260709/fit/`: three relation-tail
tensors, filter/invariant reports, full metrics/log/report, selected
checkpoint, v3 probe, and exact training/probe source archives. The Python
gate is 130/130 passing with 45 expected fixture skips.

## 2026-07-09 09:34 — Exact full-menu probe closes the head-only candidate-recall branch before gameplay

Purpose: correct the pairwise pilot's filtered-surface recall measurement and
test the stronger upstream hypothesis directly: perhaps incumbent search is
limited because policy logits fail to place the completed-Q-best action in the
top-16 candidate set. This entry evaluates two bounded, policy-head-only fits.
It is offline routing evidence only; no gameplay was run and no champion
changed.

The permanent `torch_policy_candidate_probe` rejects filtered tensors and
scores every legal action in the raw 800-root seed-3120 v3 shard. It saw 800
valid-Q roots, including 40 exact K1 roots. `--action-chunk-size 16384` exceeds
the largest legal menu in this shard, so each root/model pair used one complete
forward pass rather than a chunk-boundary approximation. As a generator-versus-
probe reproducibility check on the 760 non-exact roots, recomputed cycle4
priors matched the stored top-16 set on 729/760 roots (`95.921%`), averaged
`99.7368%` action overlap, and agreed on completed-Q-best coverage on 760/760.
All mismatches were policy-boundary swaps; the mean-overlap >=99% and exact
best-coverage-parity gates passed.

The exact incumbent baseline covered the completed-Q best in 689/800 menus
(`86.125%`), or 654/760 (`86.053%`) after excluding exact K1 roots. On the 206
count/margin/SNR-qualified roots, coverage was 186/206 (`90.291%`). Mean
candidate-oracle regret was `0.075117`; full-menu top-1 accuracy was 194/800
(`24.250%`). These replace the pairwise probe's optimistic `88.3%`, which was
measured only inside a top-Q-filtered 64-action tensor.

### Soft improved-policy imitation

The first fit warm-started cycle4 M, froze all but the 769-parameter policy
projection, and trained for 500 steps on the 3100+3110 top-Q-with-selected
relation-tail shards, with 3120 fixed for validation. Locked validation chose
step 400: policy loss improved `2.675326 -> 2.590267` and teacher top-1
`25.75% -> 26.50%`. Selected manifest/weights hashes are `5ecc053a...` and
`844f6e46...`; training report/metrics hashes are `d9ed7161...` and
`f91fadff...`.

The exact full-menu result moved in the wrong direction for the intended
mechanism: best-action coverage fell to 685/800 (`85.625%`, paired delta
`-0.500` percentage points, bootstrap CI `[-1.750,+0.750]`) and qualified
coverage fell to 185/206 (`89.806%`, delta `-0.485` points, CI
`[-2.427,+1.456]`). Candidate-oracle regret was statistically flat
(`0.072796`, paired delta `-0.000779`, CI `[-0.014891,+0.012161]`). Top-1
accuracy gained four roots to `24.750%`, but its paired CI
`[-0.750,+1.750]` points includes zero and does not rescue candidate recall.
Probe JSON SHA-256: `ac2daed82bac9434ce1080402ddaa076549d9b912f776b1633ec48aa0154e6b7`.

### Direct confidence-gated recall objective

The second fit changed the retained training surface and the objective rather
than adding another model. `top-prior-with-q-valid` keeps a fixed 64-action
menu selected by incumbent policy while mandating every Q-valid and selected
action. The three v3 relation-tail hashes are `84bc8ea5...`, `78b20088...`,
and `a79484ec...`; all source/rules/Q invariants passed. The new
`gumbel-policy-recall` objective uses policy CE weight `0.25` plus a unit
hinge requiring the completed-Q best to beat the 16th policy logit by `0.25`.
Hinge examples must be exact K1 roots or have at least two Q-valid actions,
top-two margin >=0.25, and SNR >=1. Selection used the exactly aggregated
full validation set, not batch-ratio averages.

This 769-parameter fit selected step 225 at 222/246 trusted retained-menu hits
(`90.244%`), one above initialization's 221/246; later steps reduced hinge
loss but never exceeded that count. The selected manifest/weights hashes are
`b8e0dc69...` and `f0457c2a...`; training report/metrics hashes are
`7c38e2d2...` and `f11c9640...`. Exact source was `4738f1b3`; deterministic
source-archive SHA-256 is `9e935a918808ae7f08b0589849eae9142caf4119e5c4b438beb313a524852228`.

On the same fixed validation roots, the exact full-menu audit rescued only two
candidate sets with no losses: 691/800 (`86.375%`), delta `+0.250` points,
bootstrap CI `[+0.000,+0.625]`. Qualified coverage rescued one root to 187/206
(`90.777%`), delta `+0.485` points, CI `[+0.000,+1.456]`. Top-1 accuracy was
exactly flat at 194/800. Mean candidate-oracle regret was slightly worse at
`0.075954` (paired delta `+0.000837`, CI `[-0.009988,+0.010742]`). The
non-exact stratum lost one top-1 decision despite the two coverage rescues.
Because checkpoint selection and this audit share seed block 3120, even the
two-root gain is selection-optimistic rather than independent confirmation.
Probe JSON SHA-256: `5b5668bb8fa678f74e98969378a6212a70d5e516612154d8eeb568cbfea8ed75`.

**Verdict: kill this small-data, head-only candidate-recall branch before
gameplay.** Soft imitation reduced recall; the purpose-built objective found
only two validation-menu rescues, no top-1 gain, and no regret improvement.
Do not spend john0 time or add more loss variants to this checkpoint. The
exact full-menu probe, retained-prior filter, objective, and weighted metrics
remain reusable infrastructure. Reopen upstream candidate recall only with a
materially different supervision source or architecture and a new untouched
root block, not another tuning pass on these 2,400 roots.

All artifacts are checksum-verified under ignored paths
`cascadiav3/reports/pairwise_v3_n16_20260709/policy_candidate/` and
`cascadiav3/reports/pairwise_v3_n16_20260709/policy_recall/`, including raw
reports, metrics/logs, selected checkpoints, filter/invariant reports, three
v3 tensors, exact probes, and source archive. The final Python 3.12 gate is
135/135 passing with 45 expected fixture skips. After checksum verification,
all policy-branch scratch trees were removed from john2 and john3.

## 2026-07-09 10:12 — Parallel leaf rollouts cut single-game latency 6%; batch concurrency is flat

Purpose: exploit the measured host CPU headroom without changing the serving
objective. At blend 0.5, Gumbel search batches model evaluation across live
simulations, but after each batch it previously completed every independent
sampled-greedy terminal leaf serially. This is execution evidence only; no
strength comparison or champion change is claimed.

Exact source `2544183971befec94fd76dccd95105e664f5f8b1` adds opt-in
`--gumbel-parallel-leaf-rollouts`. Leaf tasks carry their simulation's cloned
ChaCha8 RNG state, execute on the global Rayon pool, and are committed in
original simulation order. The established serial path remains the default.
The option is plumbed through the Rust CLI and Python benchmark runner and is
recorded in decision, game, report, and v3 self-play provenance. A unit test
requires bit-identical completed-Q, variance, improved policy, visits, root
priors/value, action, and simulation count. The deterministic source archive
SHA-256 is `130bc95d985ebca100b4c46f0e080840e841362a485895322ce0bc7aeefdcb68`.

The permanent `compare_gumbel_execution` gate validates passing corrected-
rules candidate-only reports; source, search, execution topology, exporter,
manifest, and weights identity; complete 80-ply decision coverage; exact
action/refresh and score parity; simulation telemetry; category sums; and
root-value drift no greater than `2e-5`. Its performance threshold is at least
`1.05x` wall speedup. A mistyped source-revision argument on the first launch
was detected immediately; that process tree was killed before a game/report,
its directory was deleted, and both admitted arms were regenerated with the
exact revision above.

### Jobs1: positive interactive-latency frontier

John2 ran serial then parallel on the same two seeds `2027073300..01`, exact
same-host source/binary/checkpoint, distq M, n16/top8/d2, four market samples,
blend 0.5, K16 interior, and one shared MPS bridge:

| Metric | Serial | Parallel | Ratio |
|---|---:|---:|---:|
| whole-arm wall | 308.144s | 290.442s | **1.061x faster** |
| mean decision | 1.922396s | 1.811754s | **1.061x faster** |
| P50 decision | 1.061741s | 1.038425s | 1.022x faster |
| P95 decision | 7.056486s | 6.498408s | **1.086x faster** |

All 160 actions, refresh choices, scores, simulation telemetry, and root
values were identical; maximum root-value drift was exactly zero. An
independent john3 parallel execution took `290.677s` wall and
`1.805942s`/decision, closely reproducing candidate timing. The same-host
comparison JSON SHA-256 is
`c25f7aca4e9bd33128e03d16e3dd11b42133545e4a0c9fbcf9c4398b2163db89`.

### Jobs2: no batch-throughput gain

John4 then ran the same seeds/config with both games concurrent against one
shared bridge. Serial wall was `269.197s`; parallel was `271.043s`, a
`0.993x` speed ratio (0.69% slower). Mean decision regressed
`3.169206s -> 3.203971s`, and P95 regressed `11.334988s -> 11.864051s`.
All 160 actions and scores remained identical. Cross-game request timing
changed MPS reduction grouping slightly, but maximum root-value drift was only
`4.354e-7`, far inside the preregistered `2e-5` execution tolerance.
Comparison JSON SHA-256:
`3680556a4614a1385734311e3653a9c720f703fafedc3cd48d0a5b2ab5d928b5`.

**Verdict: keep the opt-in implementation for single-game/interactive
latency; do not enable it for multi-game generation, promotion batteries, or
the queued jobs12 CUDA sequence.** Existing game concurrency already consumes
the host parallelism, so nested Rayon work adds overhead instead of throughput.
This result does not close GPU-native rollout generation: eliminating the
engine/device lockstep is materially different from oversubscribing CPU
terminal rollouts. The complete exporter gate is 46/46 passing; the final
Python 3.12 gate is 136/136 passing with 45 expected fixture skips. All
checksum-verified artifacts are under ignored path
`cascadiav3/reports/parallel_leaf_rollouts_20260709/`.

## 2026-07-09 10:26 — Preregistered jobs12/16/24 CUDA concurrency calibration

Purpose: resolve the live RTX 5090 utilization gap with an exact matched
throughput/parity measurement, not by extrapolating the four-seed MPS result
or editing the running corrected-rules chain. This is engineering evidence
only; it cannot support a strength claim or mutate a serving default.

`run_cuda_concurrency_probe.sh` runs jobs12, jobs16, and jobs24 sequentially
against one shared CUDA bridge on 48 fixed seeds beginning at `2027073400`.
Every arm uses the distq-k8 locked checkpoint, n64/top16/d4, eight optional-
refresh replacement samples, blend 0.5, K16 interior, exact-endgame off, and
serial leaf rollouts. It records one-second GPU utilization, power, memory,
and temperature telemetry alongside a complete 3,840-row decision ledger and
48-row score/category ledger. Reuse is allowed only after the report's exact
rules/source/seeds/search/topology/device contract and ledger/profile lengths
match.

`compare_cuda_concurrency` then validates passing corrected-rules candidate-
only reports, exact source and artifact identity, one shared CUDA bridge,
dynamic seed scheduling, telemetry invariants, action/refresh/score parity,
and root-value drift at most `2e-5`. Jobs12 remains the recommendation unless
the fastest eligible arm is at least `1.05x` faster. If that threshold passes,
the selected knee is the smallest eligible jobs count within 2% of the fastest
wall time. Non-finite timing/GPU data and incomplete profiles fail closed.

The calibration is queued last in the checksum-pinned post-chain john0
waiter: corrected n1024 rebaseline and verdict, exact K1, CUDA model-size
throughput, market sample-4, then concurrency. At 10:26 EDT the live cycle4
n1024 arm had 35/100 raw games and remained healthy; no CUDA concurrency arm
has run yet, and no default has changed. The implementation gate is 137/137
Python tests passing with 45 expected fixture skips; `bash -n` and the diff
whitespace check also pass.

## 2026-07-09 10:52 — Action-conditioned structured-value representation gate passes

Purpose: decide whether the ranked per-category value direction has enough
representation signal to justify a new exact-grounded data/model contract.
The existing `score_head(root_h)` is root-level and cannot honestly replace
per-action Q. This probe is deliberately offline; no gameplay or strength
claim is attached.

`torch_structured_value_probe` rejects filtered menus, non-v3 or nontraining
shards, rules/source mismatches, a teacher manifest/weights mismatch, and
overlapping seed intervals. It excludes exact K1 roots and evaluates one
selected action using its complete outgoing full-menu relation row. The model
now exposes `encode_action_queries`, and a unit test pins that method to the
ordinary forward representation while preserving all default outputs.

The fixed corpus was the three corrected pairwise-generation shards. Seeds
`2027073100..09` supplied 760 non-exact fit roots; `2027073110..19` supplied
760 roots to select ridge lambda from a fixed grid; `2027073120..29` supplied
760 untouched validation roots. All three raw tensors were unfiltered and
matched cycle4 manifest SHA `b8886c24...`, weights SHA `33559aab...`, rules,
and source `0f107219`. Lambda `100` minimized category MSE on the middle block;
the head was then refit on the first two blocks before the single held-out
read.

| Predictor | Held-out RMSE | MAE | Bias | Error SD |
|---|---:|---:|---:|---:|
| action-conditioned category sum | **3.4889** | **2.6964** | -0.3010 | 3.4781 |
| root category sum | 4.2525 | 3.2905 | +0.9547 | 4.1467 |
| root value | 4.2438 | 3.2877 | +0.9254 | 4.1444 |
| selected model Q | 4.4570 | 3.4010 | -2.2045 | 3.8762 |
| selected completed-Q teacher | 4.1528 | 3.1085 | -2.1818 | 3.5357 |

The action-conditioned head improves RMSE `15.99%` relative to the best
incumbent comparison, clearing the preregistered `10%` representation gate.
Verdict: build the proper exact-grounded branch. Do **not** serve this ridge
head: it predicts direct final categories only for the chosen action and has
neither exact category afterstates nor counterfactual category labels. The
real implementation must export per-action exact wildlife/habitat/Nature
components, predict category score-to-go residuals, sum them on the exact
scale, and retain scalar/distq completed-Q supervision across all q-valid
actions. Report JSON SHA:
`5c06de5da762352765a26c233b8718af7e69bc9040d698ad0758c2b72e908c2a`.
An independent repeat on the same MPS host was byte-identical for both JSON
and Markdown. The complete Python gate is 140/140 passing with 45 expected
fixture skips; Python compilation and the diff whitespace check pass.

## 2026-07-09 — Exact-grounded structured-Q path implemented (no model verdict yet)

Purpose: turn the passed representation preflight into an honest trainable
architecture without assigning the selected trajectory's category outcome to
counterfactual actions. This is implementation and contract evidence only;
no gameplay, strength, or promotion claim is attached.

New Gumbel self-play generation writes
`cascadiav3.expert_tensor_shard.v4`. Each record carries `active_seat`; each
legal action carries exact afterstate wildlife, habitat (including any habitat
bonus), and Nature-token components. Those components must sum to
`exact_afterstate_score_active`, and each terminal three-component vector must
sum to `final_score_vector`. The Rust exporter, NPZ writer, Python mmap/NPZ
reader, action filter, fixed-capacity relation-tail materializer, corpus
collator, and schema registry all preserve and validate the contract. v1-v3
remain readable; only v4 is admitted to structured-Q training.

CascadiaFormer adds opt-in `q_decomposition`. The new projection emits three
action-conditioned score-to-go residuals per action (and per quantile when
distributional); ordinary `q` is their exact sum. Thus the unchanged serving
identity remains:

```text
derived_final_q = exact_afterstate_score_active + sum(predicted components)
```

The loss uses terminal active-seat components minus the selected action's
exact afterstate components. It is applied only to the selected real action.
Every q-valid action still receives scalar/distributional completed-Q
supervision on the component sum. The default objective weight is `0.5` under
`gumbel-selfplay-structured-q`. `q-decomposition-head-only` freezes the trunk
and all incumbent heads; for scalar Q it exposes exactly `3 * (d_model + 1)`
parameters. Checkpoint/benchmark/exporter provenance now records
`q_decomposition`, and the bridge reloads/serves the summed output without a
new wire protocol.

The implementation tests legacy parameter-contract preservation, scalar and
distributional sum identities, exact target arithmetic, malformed afterstate
and terminal component rejection, filter/tail/collation survival, v4-only
trainer admission, head-only freezing, checkpoint reload, and an end-to-end
two-step v4 training smoke. That smoke uncovered a pre-existing trainer bug:
packed examples selected by `--overfit-one-batch` were mislabeled as JSONL and
failed in the JSON collator. An in-memory tensor-corpus wrapper now preserves
the NPZ dispatch contract.

Final local gate: 150/150 Python 3.12 tests passed with 45 expected
fixture-dependent skips; 46/46 real-root-exporter tests passed; workspace
`cargo check` passed with only the pre-existing unused API-field warning; the
schema registry and changed-file formatting/whitespace checks passed.

Scientific next step: generate a small fresh corrected-rules v4 train/locked-
validation pair, warm-start the incumbent with `--init-skip-mismatched`, and
run only the frozen component-head kill test. Full fine-tuning and gameplay
remain gated on that result.

## 2026-07-09 — Three-way structured-Q v4 corpus generated and staged

Purpose: supply a disjoint fit / hyperparameter-selection / untouched-verdict
split for the preregistered head-only gate. The Mac fleet generated data only;
all training and verdict execution remains john0-owned.

An initial n16/top8/d2 attempt was stopped after roughly seven minutes because
none of the three hosts had published an NPZ/manifest and the shape was too
slow for a 2,400-root plumbing gate. No partial artifact was retained. The
final immutable shape kept the scientifically important contracts—corrected
rules, full root menus, eight-sample optional-refresh choice, real terminal
outcomes, exact K1 rows, and no model fallback—while reducing counterfactual
search-label cost to n8/top4/d1, one determinization, blend 0.5, and K8
interior. Each host used two concurrent games through one shared MPS bridge.

All blocks used exporter source
`6e89d9555f6126bdc29f65657d8431cab3d2c024` and cycle4 teacher manifest SHA
`b8886c24cd93e19299e8c4cca4dd7671fe16b685d54949de014d6f9d5aee616d` /
weights SHA
`33559aab05324e74998164d4e59e7adec9fa3c77da531dd4797c718cf4cfd354`:

| Block | Host | Seeds | Seconds | Roots | Actions | NPZ SHA-256 |
|---|---|---|---:|---:|---:|---|
| fit | john2 | 2027073500..09 | 984.3 | 800 | 358,975 | `06d550b4b70b32bab1e7bea4d994a26341d7b5a8d3dc58b8fc636d89c31e8519` |
| selection | john4 | 2027073510..19 | 908.1 | 800 | 389,735 | `5095d572b2167f81931d2a5ba7d8a339ffba285d04dd3ff55eeac99c647688cc` |
| verdict | john3 | 2027073520..29 | 1059.8 | 800 | 365,045 | `cdbd54b0c2aaa79fbc4c1a12c73eaae863d3a62ab8924a8173df12c9717eb6b4` |

Every shard is training-eligible `cascadiav3.expert_tensor_shard.v4`; Python
shape/component validation and Q-identity validation passed, with maximum
`|Q - exact_afterstate - score_to_go| = 3.8146973e-6` against a `1e-4`
limit. Raw max legal menus reached 6,900 / 8,424 / 8,064 actions, demonstrating
why the fit path must use a provenance-preserving top-64 transform while the
selected real-action category label remains exact.

`run_structured_q_head_pilot.sh` requires all three hashes, selects among
fixed learning rates `3e-4 / 1e-3 / 3e-3` using only the middle block, and
invokes `torch_structured_q_probe` once on the third block. Legacy warm starts
initialize the three projections as equal thirds of incumbent Q, preserving
their sum within floating-point tolerance at step zero. The runner records a
scientific failure without launching gameplay and stops on malformed output.
All three raw NPZs and manifests are copied to john0 with matching hashes.

The idle post-chain waiter was replaced without touching the live rebaseline
or verdict watcher. New waiter PID `2241595` verified all three NPZ hashes and
the source archive before sleeping. It pins source
`f35b0d0b209444f8c09e7e603c380f1d8edbc100` via archive SHA-256
`460857f26f7431727db623313f92df2e5be13a27033bd72d642eb6d650fc7a81`.
Order remains corrected rebaseline/verdict, exact K1, structured-Q head pilot,
model throughput, market sample-4, then CUDA concurrency. The older
`11f254d9` waiter/archive was removed only after the new waiter passed its
preflight and was observed alive.

## 2026-07-09 — Structured-Q target and held-out baseline audit

Purpose: quantify the v4 target before john0 training, turn the preregistered
gate into absolute thresholds, and detect split drift or impossible component
semantics without inspecting any candidate prediction.

Each block contains 800 roots: 40 exact K1 rows and 760 non-exact rows. Every
non-exact row has four q-valid searched actions; the exact rows have one, hence
3.85 q-valid actions per root overall. Exact terminal component sums matched
the active-seat final score with zero observed error. The non-exact selected-
action score-to-go means were stable across fit / selection / verdict:

| Component | Fit | Selection | Verdict |
|---|---:|---:|---:|
| Wildlife | 31.8250 | 32.8474 | 32.5618 |
| Habitat | 11.7197 | 11.5434 | 11.8079 |
| Nature tokens | 1.7579 | 1.6803 | 1.7711 |
| Total | 45.3026 | 46.0711 | 46.1408 |

Nature-token residuals were negative on `6.32% / 8.68% / 7.24%` of non-exact
rows, as they must be when the real trajectory spends tokens after the chosen
action. Wildlife, habitat, and total residuals were never negative. The
teacher's selected-final RMSE against the real trajectory was `2.9992 / 3.4929
/ 3.5520`; the harder later blocks reinforce the need for disjoint selection
and verdict data rather than weakening the gate.

The candidate-blind verdict baseline used the exact top-64 + fixed relation-
tail surface that the queued probe will use. On 760 non-exact roots, incumbent
selected-final RMSE / MAE / bias were `3.7476 / 2.8394 / -1.5379`; teacher
values were `3.5520 / 2.6386 / -1.6946`. Therefore the better baseline is the
teacher and the 10% gate requires candidate RMSE `<= 3.1968`, plus a paired
absolute-error t-CI wholly below zero. Across 3,040 q-valid actions, incumbent
RMSE against teacher completed-Q was `1.7499`, mean teacher-Q regret was
`0.7515`, and top-1 agreement was `36.45%`. The retention ceilings are thus
candidate all-q RMSE `<= 1.8374` and mean regret `<= 0.8015`. No candidate was
evaluated and this read did not influence the fixed LR grid or selection rule.

## 2026-07-09 12:09 — Quarantined structured-Q fit expansion launched

Purpose: use otherwise idle data-generation Macs to remove the next corpus
latency if, and only if, the fixed head-only gate passes. This launch does not
change the john0 queue or the registered pilot data.

john2, john3, and john4 each reverified source marker
`6e89d9555f6126bdc29f65657d8431cab3d2c024`, teacher manifest SHA
`b8886c24cd93e19299e8c4cca4dd7671fe16b685d54949de014d6f9d5aee616d`,
and weights SHA
`33559aab05324e74998164d4e59e7adec9fa3c77da531dd4797c718cf4cfd354`
before launch. The shape exactly matches the pilot generator: n8/top4/d1,
one determinization, eight market-decision samples, blend 0.5, K8 interior,
exact K1, full root menus, exploration on, two concurrent games through one
shared MPS bridge, and no fallback.

| Host | PID | Seeds | Planned roots | Output |
|---|---:|---|---:|---|
| john2 | 90485 | 2027073600..49 | 4,000 | `structured_q_v4_expansion_20260709/expansion_a.npz` |
| john3 | 58489 | 2027073650..99 | 4,000 | `structured_q_v4_expansion_20260709/expansion_b.npz` |
| john4 | 26369 | 2027073700..49 | 4,000 | `structured_q_v4_expansion_20260709/expansion_c.npz` |

All three exporter/bridge process pairs were alive with empty startup logs
after launch. The artifacts are quarantined fit-capacity inventory: they must
complete with v4 provenance and invariant validation, remain disjoint from
the fixed selection/verdict blocks, and may not enter training unless the
preregistered head-only pilot passes. john1 was intentionally left to the
active UI/champion service; john0 remained untouched.

## 2026-07-09 12:13 — Live n1024 utilization sample confirms feed gaps

A read-only 30-s, one-Hz sample of the in-flight corrected cycle4 n1024/d16
jobs12 arm found GPU utilization mean / P50 / P90 `63.8% / 66% / 85%`, with a
wide `2%..88%` range. Mean power was `350.1W` (range `271.9..413.4W`), memory
was fixed at `2403 MiB`, and temperature averaged `62.1C`. Linux cumulative
process CPU readings were stable at `779%` for exporter PID `1739797` and
`407.5%` for bridge PID `1739800`.

Interpretation: the GPU still sees intermittent feed gaps while both Rust
search and Python bridge consume substantial CPU. This is diagnostic evidence
for the checksum-queued jobs12/16/24 concurrency calibration, not permission
to alter the 52/100 live scientific arm. No score field was read.

## 2026-07-09 12:23 — Cross-shard structured-Q admission audit implemented

Per-shard shape and Q checks were necessary but insufficient for the
quarantined expansion: they could not prove that three independently produced
files share one scientific contract or avoid the locked pilot's seeds.
`audit_structured_q_shards` now fails closed unless every raw v4 NPZ matches
its sidecar checksum, metadata, schema, eligibility, seed domain, record count,
and action count. It rechecks selected-action Q validity, completed-Q identity,
afterstate component sums, terminal component sums, exact-K1 row counts, and
the declared records-per-seed contract.

Across files it requires identical rules, source revision, search, execution,
and teacher manifest/weights identity while deliberately allowing host-local
generator binary identities. Primary and explicitly excluded locked shards
must have globally disjoint seed intervals. Optional expected source and
teacher hashes bind an operating invocation to preregistered artifacts.

The real fixed fit/selection/verdict corpus passed: 3 shards, 30 disjoint
seeds, 2,400 roots, 1,113,755 actions, 9,240 q-valid actions, 120 exact rows,
maximum Q-identity error `3.8146973e-6`, and zero afterstate/terminal component
sum error. Synthetic tests cover valid admission, strict seed-domain parsing,
overlap rejection, contract mismatch, and sidecar checksum tampering. Full
Python gate: 160/160 passing with 45 expected fixture skips. The expansion
remains quarantined until the same audit can run over all six raw shards.
Locked-corpus audit JSON SHA-256:
`720a2f84b9f02d28ceb4fb293274e78f6739394ff5b181e1272e269c61ba339b`.

## 2026-07-09 12:28 — Expansion harvest made fail-closed and repeatable

`fetch_structured_q_expansion.sh` is the sole harvest path for the three
quarantined fleet shards. It refuses while a producer or completion validator
is live; requires nonempty NPZ, sidecar, summary, invariant, producer-log, and
validator-log artifacts; requires both validation JSONs to pass; compares
remote and local NPZ/manifest SHA-256 after resumable `rsync`; and then runs
the cross-shard auditor against all three locked pilot files as exclusions.
It never addresses john0 and never emits a training command.

Shell syntax passed, remote-home `rsync` resolution was exercised against
john2, and a live-producer invocation failed before transfer with the expected
status 2. Four targeted unit tests pass, including pins for all hosts, source,
teacher hashes, exclusions, and the quarantine promise. At this checkpoint all
three producers had streamed 10/50 complete seeds; no fetch was attempted.

## 2026-07-09 12:32 — Corrected n1024 verdict gains category attribution

The d20 total-score comparator cannot explain a distq gain or regression, and
the old n256 reducer irretrievably discarded score components. The live
n1024 watcher is different: it preserves one complete engine game-done row
per seed with all four seats' wildlife-card, habitat-type, Nature-token, and
total scores. `compare_game_categories` now turns those two 100-row ledgers
into a same-seed distq-minus-cycle4 mechanism verdict.

The comparator requires passing candidate-only reports with identical rules,
source revision, seeds, and normalized search contracts; exactly 80 decisions
and four category-complete seats per ledger row; exact category-to-total sums;
and exact agreement between every ledger game mean and its aggregate report.
It reports overall and by-seat means, games/seat-scores at least 100, paired
t/bootstrapped intervals for all three categories and total, per-seed deltas,
and input SHA-256 identities. Per-seed and mean category deltas must sum to the
total delta.

Three adversarial unit tests cover missing seeds, search mismatch, and category
tampering. A real production-schema jobs2-versus-jobs1 parity replay passed on
four seeds with exact zero deltas in every category. Full Python gate:
164/164 passing with 45 expected fixture skips. The category verdict will run
only after both n1024 ledgers are complete; no partial score was inspected.

## 2026-07-09 12:40 — n1024 harvest reconciles mechanism and headline

`fetch_rules_n1024_verdict.sh` now makes the completed corrected-rules read a
single fail-closed operation. It refuses while either the main rebaseline PID
or the raw-ledger watcher is live, then requires both n1024 reports, summaries,
8,000-row decision ledgers, 100-row game ledgers, category summaries, and the
canonical total verdict. Every remote file is SHA-256 checked again after a
resumable local transfer.

The category comparator now optionally consumes the canonical total verdict.
It locates exactly one comparison by left/right experiment IDs, checks rules
and source provenance, checks both arm means against the ledgers, and requires
all paired-total statistic fields—including mean, dispersion, t interval,
bootstrap interval, confidence, and CI decision—to agree within `1e-12`.
Thus the category report cannot silently tell a different headline story.

The live invocation correctly refused before transfer with status 2 while
rebaseline PID `1265148` remained active. Five targeted tests and the full
Python gate pass: 166/166 with 45 expected fixture skips. No john0 artifact was
modified and no partial score was read.

## 2026-07-09 12:41 — Structured-Q admission audit now diagnoses target drift

Matching source/search/teacher provenance does not prove that a new seed block
has a comparable learning target. Audit schema
`raw_structured_q_cross_shard_v2` now records, per shard and outside exact K1
rows, final-score and wildlife/habitat/Nature/total score-to-go distributions,
selected-teacher-minus-real error distributions, and valid-Q actions per root.
It retains every v1 identity and disjointness check.

The regenerated locked-corpus report reproduces the independent target audit:
fit / selection / verdict final means `91.250 / 92.000 / 92.425`, total
score-to-go means `45.303 / 46.071 / 46.141`, Nature-negative fractions
`6.32% / 8.68% / 7.24%`, selected-teacher RMSE
`2.9992 / 3.4929 / 3.5520`, and exactly `3.85` q-valid actions per root.
The v2 JSON SHA is
`720a2f84b9f02d28ceb4fb293274e78f6739394ff5b181e1272e269c61ba339b`.

## 2026-07-09 12:48 — Candidate-blind structured-Q reserves preregistered

Purpose: keep the data-only fleet productive after the 50-seed expansion and
remove the next holdout-generation delay without allowing a future candidate
to choose its own evaluation seeds. No candidate checkpoint exists, and this
action does not fetch data, alter the john0 queue, or authorize training.

The roles and disjoint seed blocks were fixed as follows:

| Host | Role | Seeds | Planned roots | Sleeping chain PID |
|---|---|---|---:|---:|
| john2 | selection | 2027073750..69 | 1,600 | 97051 |
| john3 | verdict | 2027073770..89 | 1,600 | 64988 |
| john4 | replication | 2027073790..3809 | 1,600 | 30230 |

Every reserve uses the current raw-v4 n8/top4/d1 contract: one
determinization, eight optional-refresh samples, blend 0.5, K8 interior,
exact K1, 80 plies per seed, full root menus, two concurrent games through a
shared MPS bridge, and no model fallback. The launcher verifies source marker
`6e89d9555f6126bdc29f65657d8431cab3d2c024`, teacher manifest SHA
`b8886c24cd93e19299e8c4cca4dd7671fe16b685d54949de014d6f9d5aee616d`,
and weights SHA
`33559aab05324e74998164d4e59e7adec9fa3c77da531dd4797c718cf4cfd354`
both before arming and after the preceding validator exits. A reserve starts
only if its host's expansion summary and invariant reports both say `pass`.
It then generates, summarizes, validates the completed-Q identity, and hashes
the NPZ plus all sidecars locally. The script contains no fetch, admission,
training, or john0 path.

The first john2 arming attempt exposed a nested-shell quoting defect before it
could start generation. PID `96777` exited, no reserve NPZ or manifest was
created, and the stale chain PID/log were removed. The launcher now transfers
the remote runner as base64, decodes it into one Bash program, and passes all
eleven arguments positionally. Shell syntax and the targeted five-test suite
passed before rearming.

At the final live check, all three new chain PIDs, original exporter PIDs
`90485 / 58489 / 26369`, and validator PIDs `90916 / 58926 / 26606` were
alive. Each expansion had reached 30/50 seeds; every reserve chain log was
zero bytes and every reserve artifact was absent, proving that the chains
were waiting rather than generating early. Final local gate: 167/167 Python
tests passed with 45 expected fixture-dependent skips; `bash -n` and
`git diff --check` passed.

## 2026-07-09 12:56 — Reserve-holdout harvest made fail-closed

The reserve roles now have a canonical retrieval path before any of them
exists. `fetch_structured_q_reserve_holdouts.sh` refuses a live chain, requires
nonempty NPZ/manifest/summary/invariant/chain-log artifacts and passing JSON
reports, checks the exact completion sentinel, hash-matches remote and local
NPZ plus manifest files, and then runs one cross-shard audit. That audit treats
the locked fit/selection/verdict pilot and all three quarantined fit-expansion
blocks as exclusions. No john0 or training copy exists in the script.

The generic auditor now accepts exact expected seed domains for every primary
label. Expectations are all-or-none, labels must match the primary set, and
first seed, count, plies per seed, and mode must match exactly. The existing
expansion harvest now pins its three 50-seed domains; the reserve harvest pins
selection `2027073750..69`, verdict `2027073770..89`, and replication
`2027073790..3809`. This prevents a contract-valid and disjoint shard from
being accepted under the wrong experimental role.

Six targeted tests pass, including a wrong-domain rejection and static pins
for all hosts, roles, domains, six exclusions, and quarantine behavior. Both
harvest scripts pass `bash -n`. Live probes correctly refused with status 2:
the expansion harvester found producer `90485` active and the reserve
harvester found chain `97051` active. Neither transferred an artifact.
The new CLI path also passed against the real locked corpus with exact domains:
3 shards, 30 seeds, 2,400 records, 1,113,755 actions, 9,240 q-valid actions,
and 120 exact rows. Final Python gate: 168/168 passing with 45 expected
fixture-dependent skips; `git diff --check` passed.

## 2026-07-09 13:19 — Expansion audited; manifest-boundary failure repaired

All three expansion producers completed successfully:

| Host | Seeds | Seconds | Roots | Actions | NPZ SHA-256 |
|---|---|---:|---:|---:|---|
| john2 | 2027073600..49 | 3792.6 | 4,000 | 1,771,087 | `225aeff6dd73e0902b1786d09f8236e7e8c53301beda82441b70c05d0429e74f` |
| john3 | 2027073650..99 | 3813.3 | 4,000 | 1,791,934 | `0447d69bff7bef39261ef8cfd09bb37e6cbf2bf5545ac82eff3516615776b69a` |
| john4 | 2027073700..49 | 3806.3 | 4,000 | 1,736,266 | `5dc0860dca996a719d87f2aae33f88e50238bbb37e45536dce96380e3d041959` |

The completion boundary caught a launch defect rather than accepting a
partial artifact. The ad hoc expansion commands set `--out` but omitted
`--manifest`, so each exporter wrote its valid sidecar to the CLI default
`cascadiav3/fixtures/real_roots_manifest.json`. The expected sidecars were
absent; all validator shells exited before summary generation, and all three
reserve chains then exited on the missing summary. No reserve NPZ or manifest
was created.

For each host, the generated default manifest checksum exactly matched the
681–683 MiB expansion NPZ and declared 4,000 v4 records, the exact expected
seed domain, corrected rules, source `6e89d955...`, and the fixed teacher
manifest/weights. That generated sidecar was copied to the declared artifact
path, the summary and Q-identity validators were rerun, and both reports
passed. The failed chain logs and PID files were preserved under
`.failed_manifest_path` names. This is deterministic artifact recovery, not
data regeneration or relabeling.

Commit `4cd9c728` adds the missing role-specific `--manifest` argument to the
permanent reserve launcher and a regression assertion. Targeted six-test and
full 168-test Python gates passed with 45 expected fixture skips; shell syntax
and whitespace checks passed. Corrected reserve chains launched at 13:16 EDT:
selection PID `2465`, verdict `69950`, replication `33569`. Their exporter
children were observed with exact sidecar paths, no premature NPZs, and all
three completed their first seed by 13:19.

The canonical expansion harvest then hash-matched all three hosts and ran the
six-shard audit against the locked pilot. Result: pass; 150 seeds, 12,000
records, 5,299,287 actions, 46,200 q-valid actions, 600 exact rows, maximum
Q-identity error `3.8146973e-6`, and zero afterstate/terminal component-sum
error. Expansion final means are `91.485 / 91.885 / 91.490`; total score-to-go
means `45.846 / 46.001 / 45.701`; Nature-negative fractions
`10.00% / 6.82% / 8.79%`; selected-teacher RMSE
`3.169 / 3.375 / 3.287`; q-valid means are exactly `3.85`. Audit SHA-256:
`e1edbad3552abef2321808666948f299fbf3ba226b948d50a2314b696fb5eb14`.
The data remains local quarantined fit inventory; nothing was copied to john0
or added to a training command.

## 2026-07-09 13:44 — Candidate-blind structured-Q reserves complete per host

The three preregistered reserve roles completed after the repaired launcher
made each output manifest path explicit. They remain semantically distinct
holdouts, not additional fit data:

| Host | Role | Seeds | Seconds | Roots | Actions | NPZ SHA-256 |
|---|---|---|---:|---:|---:|---|
| john2 | selection | 2027073750..69 | 1530.0 | 1,600 | 711,027 | `48e48e74f0853c434d4ec157d188b092463676d40c1b389059f9a2dca86ad46d` |
| john3 | verdict | 2027073770..89 | 1624.4 | 1,600 | 667,699 | `99b85671881ae6bde5d49c1a07588b27b368fe3194450aea52e223ed54b668d8` |
| john4 | replication | 2027073790..3809 | 1506.5 | 1,600 | 680,007 | `41b5bd6098c96b05ab6a14e1a53042ae71104864e111e78bbc6f645b5caab5a7` |

All three manifests checksum-match the NPZs and declare the exact expected
seed domain, corrected rules, source
`6e89d9555f6126bdc29f65657d8431cab3d2c024`, teacher manifest
`b8886c24cd93e19299e8c4cca4dd7671fe16b685d54949de014d6f9d5aee616d`,
and teacher weights
`33559aab05324e74998164d4e59e7adec9fa3c77da531dd4797c718cf4cfd354`.
Per-host summary and completed-Q invariant reports pass. Manifest hashes are
`0a51fd1c... / 47b51bb9... / 19bd903d...`; summary hashes are
`577356ac... / 3bc44e46... / 8f97aff6...`; invariant hashes are
`96de886b... / 09bb7661... / 5cff79c0...`.

This is not yet a cross-shard admission verdict. The canonical orchestrator
must still run `fetch_structured_q_reserve_holdouts.sh`, which will reverify
the remote/local hashes and audit these three roles against the locked pilot
and all three fit-expansion shards. No reserve artifact has been copied to
john0 or a training command.

## 2026-07-09 20:57 — Corrected scalar n1024 complete; raw-ledger gap; distq live

The corrected-rules cycle4 scalar n1024/d16 arm completed at 16:50 EDT under
source `d20daf44dc6aa4aad3d03c6ccb7d3a21c3013135`, rules ID
`cascadia_research_aaaaa_4p_card_a_no_habitat_bonus_rules_2026_07_09`, and
seeds `2027070900..2027070999`. Its passing 100-game report records mean seat
`98.2975`, P50 `98.0`, P90 `102.0`, and mean search/decision time
`46.27328055s`. Report SHA-256 is
`8c164dc6c05f34633b24c36fb28e9fd4234271670f523c8bd6d4e6abab953761`;
the complete 8,000-row decision ledger SHA-256 is
`d42cf655aa26facd2107887250ccb5e6cefab9a0d32260eb27df3419a3ce9922`.
This is the first current-rules high-budget scalar baseline and lands almost
exactly on the legacy 98.28 result, but it remains below 100 and is not a
paired comparison against the old rules.

The raw-ledger watcher failed closed at arm finalization. It copied 99 valid
81-row scalar game files but missed seed `2027070908`; its log contains:

```text
missing raw game file: /home/john0/cascadia/cascadiav3/reports/rules_20260709_cycle4_n1024_d16_raw_games/gumbel_game_seed_2027070908.jsonl
```

The scalar benchmark's temporary directory is gone. The aggregate report and
decision ledger remain valid, and their candidate-per-seed record pins seed
`2027070908` to seat scores `97 / 98 / 97 / 100` (mean `98.0`) with all 80
decision rows present. However, the missing raw game-done row contains the
category arrays, so the scalar `*_games.jsonl` and category summary were not
published. A same-source, same-artifact, same-search one-seed replay must
match all 80 recorded actions/refresh decisions and the pinned seat totals
before its category row can repair the ledger.

Because watcher PID `1284321` exited before the distq arm began, it is not
mirroring the live distq raw files. At the 20:57 handoff snapshot, distq runner
PID `3556049`, exporter PID `3556050`, and bridge PID `3556053` were alive;
39/100 complete raw games existed only in
`/tmp/tmphk9xcpuk/gumbel_batch`. The last-10 completion rate was
`10.06 games/hour`, projecting completion near 02:59 EDT on July 10. The
distq aggregate report and final paired total verdict do not yet exist.

Main rebaseline PID `1265148`, total-verdict watcher PID `1268022`, and
checksum-pinned post-chain waiter PID `2241595` remain alive. The post-chain
waiter will automatically deploy source `f35b0d0b...` after the main and
verdict PIDs exit, then run exact K1, the structured-Q frozen-head pilot, CUDA
model throughput, market-sample-4, and jobs12/16/24 concurrency in that order.
The dead category watcher does not block that automatic sequence. See the
root `handoff-2026-07-09.md` for the urgent raw-file preservation step and
complete resume checklist.

## 2026-07-09 21:15 — Distq raw mirror installed; reserve holdouts harvested and globally audited

Session resume executed the handoff's two time-sensitive steps. No scientific
score was read and no live john0 job was touched.

1. **Distq raw-file preservation.** At 21:09 EDT all six john0 PIDs
   (`1265148`, `1268022`, `2241595`, `3556049`, `3556050`, `3556053`) were
   alive and the durable distq mirror directory did not exist. The handoff's
   copy command mirrored all 40 then-complete 81-row raw files into
   `cascadiav3/reports/rules_20260709_distq_k8_n1024_d16_raw_games/`. A
   replacement mirror loop now recopies every 120 seconds until runner
   `3556049` exits, then performs a final copy. Loop artifacts are
   `cascadiav3/logs/rules_20260709_distq_n1024_raw_mirror.{sh,log,pid}`
   (PID `3576186`, a fresh pid file — the failed watcher's pid files were not
   reused). Completion still requires all 100 files with exactly 81 rows and
   one game-done row each before any ledger claim.

2. **Reserve-holdout harvest.** `fetch_structured_q_reserve_holdouts.sh`
   passed end to end: remote/local NPZ and manifest hashes matched for all
   three roles, exact seed domains pinned, and the nine-shard audit against
   the locked pilot plus all three fit-expansion shards returned `pass` with
   3 shards, 60 seeds, 4,800 records, 2,058,733 actions, 18,480 q-valid
   actions, and 240 exact rows. Combined audit SHA-256 is
   `aab21d186955f7281fbc1fc0cce9b6ceb8e2b8ed9d9529aa0dc1b6071af5a3d2`.
   The holdouts remain quarantined; nothing was copied to john0 or any
   training input.

Verification context at resume: `HEAD == origin/main == 05b11a1f`, only the
known unrelated untracked operational files present; local web/API service
healthy (API PID `65336`, suggest exporter PID `68807`, HTTP 200); distq at
40/100 complete raw games and a last-10 rate near `10 games/hour`, consistent
with the projected ~02:59 EDT July 10 completion. The scalar seed-0908
category replay and the post-chain coordination decision remain open.

## 2026-07-09 22:25 — Rules-of-engagement adoption; john0 checkpoint retention pass

Operational transition, no scientific claim. Commit `b67b5163` adopts the
approved practices: durable-first raw-games evidence in the Gumbel benchmark
(`--raw-games-dir` default beside `--out`, stale-file refusal, report
provenance field, 8 new unit tests; full suite 176 passing), the waiter
HOLD/heartbeat pattern (`cascadiav3/scripts/lib_waiter.sh`, selftest passing),
the one-command status snapshot (`cascadiav3/scripts/campaign_status.sh`),
AGENTS.md rules-of-engagement amendments, the 2027 seed-registry catch-up,
and the `docs/handoffs/` convention. The post-chain waiter script and pinned
archive were copied from `/tmp` to `cascadiav3/logs/` (archive SHA verified
`460857f2...`); the running waiter was not touched.

Checkpoint retention (first application of the new line-close policy):
deleted 380 `step_*.{pt,weights.pt}` intermediates across 20 closed-line
checkpoint directories on john0 (EI-0/EI-1 era, phase0/smoke trees, gumbel
cycles 1-3/3_m/4_l/5/5_nofleet/6/6_fleettrial, ens_m_seed_a/b), freeing
150.3 GiB (checkpoints 250G -> 52G; john0 disk 50% -> 34%). Kept everywhere:
`best_locked_val.*`, `swa.*`, every manifest/metrics/report, the highest
step per line, and the record-referenced selected steps
(`cycle3_m/step_0005500+0010000`, `cycle4_l/step_0007000`,
`cycle6/step_0005000`, `ens_m_seed_a/step_0018500`). champion `cycle4`,
live `distq_k8`, and the loose crt_* pilot checkpoints were untouched.
Kept-artifact sanity listing passed; the live distq n1024 arm and all
watcher/waiter PIDs were verified alive after the pass.

## 2026-07-10 03:30 — Corrected n1024 verdict: distq ties scalar; one distq raw seed lost; post-chain marker crash fixed and resumed

**Corrected-rules distq-k8 n1024/d16 completed on schedule (100 seeds, source
d20, rules `..._2026_07_09`): mean seat `98.3850`.** The verdict watcher
published `rules_20260709_rebaseline_verdict.json` with all four preregistered
comparisons (n=100 paired):

- **distq minus cycle4 at n1024/d16: `+0.0875`, 95% t-CI `[-0.2411,+0.4161]`,
  ns — cycle4 scalar is RETAINED as champion at high budget.** This exactly
  reproduces the legacy tie (98.40 vs 98.28) under corrected rules.
- distq minus cycle4 at n256/d4: `+0.2400`, CI `[-0.1139,+0.5939]`, ns.
- cycle4 n1024/d16 minus n256/d4: `+1.2300`, CI `[+0.8564,+1.6036]`, CI+.
- distq n1024/d16 minus n256/d4: `+1.0775`, CI `[+0.7403,+1.4147]`, CI+.

Within-model high-budget scaling is strongly positive for both heads; the
head choice is score-neutral at n1024/d16 under corrected rules. Neither arm
reaches 100 (gap ~-1.6).

**Raw-ledger gap #2:** the replacement mirror loop (120s cadence) captured
99/100 distq raw files; seed `2027070962` completed in the final window and
its file was destroyed when the d20 harness deleted its `/tmp` directory at
process exit, before the post-exit copy ran. All 99 mirrored files have
exactly 81 rows. Seed 0962's seat totals are pinned in the aggregate report
and its 80 decision rows are durable, so it joins scalar seed `2027070908`
under the same one-seed d20 replay contract. The durable-first
`--raw-games-dir` change (commit `b67b5163`) eliminates this class for all
future arms; it was not in d20.

**Post-chain crash and resume:** the checksum-pinned waiter (PID 2241595)
correctly verified and deployed the f35 snapshot at 02:51 and wrote
`postchain_deployed_revision.txt`, but `run_exact_k1_gate.sh` requires the
marker at `exact_k1_deployed_revision.txt`; the gate failed closed
("source snapshot lacks the exact deployed revision marker") and the chain
stopped before any stage ran. The deployed tree was re-verified against the
durable archive copy (`tar -d`: content-identical, uid/gid only) and the
marker copied to the expected filename. The stage block was relaunched
verbatim as `cascadiav3/logs/postchain_resume_f35b0d0b.{sh,log,pid}` (PID
`3620337`) with heartbeats and `HOLD_postchain_resume` pause support per the
new waiter pattern. Stage 1 (exact-K1 gate, seeds `2027071400x100`) rebuilt
the exporter from f35 and is running. Remaining stages: structured-Q head
pilot, CUDA model throughput, market sample-4, jobs concurrency.

## 2026-07-10 08:35 — Exact-K1 100-seed gate: arms complete and flat; comparator failed closed on one concurrency-divergent seed

Both stage-1 arms of the f35 post-chain completed on seeds
`2027071400..2027071499` (corrected rules, n256/top16/d4, jobs12, one shared
CUDA bridge): baseline mean seat `97.2650`, exact-K1 mean seat `97.2350`,
both status pass, all four ledgers published (reports, decisions, games).
Descriptively the arms are flat (`-0.0300` overall); no paired verdict was
produced because `compare_exact_endgame` failed closed:
`pre-K1 action trace diverges at seed 2027071427 ply 18`.

Offline divergence forensics over both 8,000-row decision ledgers: 99/100
seeds have bit-identical action/refresh traces through ply 75 and first
diverge exactly at the K1 frontier (ply >= 76); exactly one seed
(`2027071427`) diverges at ply 18. This is the known shared-bridge
concurrency numerics mode (previously observed on MPS at two workers), now
measured at 1% frequency at CUDA jobs12 — not a K1 code defect, which would
shift many seeds. The preregistered comparator has no divergent-seed
tolerance, so the K1 verdict is BLOCKED pending an explicit methodology
ruling: (a) amend the comparator with a declared exclusion for pre-K1
divergent seeds and verdict on the 99 causally-valid pairs, (b) rerun both
arms at lower concurrency, or (c) record the gate as invalid-as-run. No
option was exercised without the user.

The crash stopped the chain before stage 2; stages 2-5 were relaunched as
`cascadiav3/logs/postchain_resume2_f35b0d0b.{sh,log,pid}` under the same f35
contract with heartbeats and HOLD support. The structured-Q frozen-head
pilot (stage 2) is training.

Also fixed while validating against the real ledgers: action identities are
`sha256:` content hashes in both d20 and f35 decision ledgers; the replay
validator no longer coerces `chosen_action_id` to int (tests updated to
hash-form IDs).

## 2026-07-10 10:20 — Structured-Q head pilot FAILS its preregistered kill test; CUDA throughput probe closes the small-model path; market sample-4 gate running

**Stage 2 (structured-Q frozen-head pilot): FAIL.** The resume2 chain ran the
pilot at 10:00:19-10:01:21 against the hash-pinned prebuilt v4 splits
(datasets built 07-09 under source `6e89d955`, fit/selection/verdict NPZ
`06d550b4/5095d572/cdbd54b0`; training and evaluation under `f35b0d0b`).
Three frozen-trunk head-only arms (lr 3e-4 / 1e-3 / 3e-3, 100 steps, batch 8)
trained in about a minute on the 5090; selection chose `lr3e3` (val metric
`1.1085`; `lr3e4` failed to converge at `4.5463`). One-shot held-out verdict
on the 760 untouched non-exact verdict roots
(`structured_q_head_pilot_20260709/heldout_verdict.json`, status **fail**):

- Selected-final RMSE `4.1573` vs best baseline (`selected_teacher_q`)
  `3.5520` — **-17.04%** against a required **>=+10%** (threshold `3.1968`).
  FAIL.
- Paired candidate-minus-baseline absolute error mean `+0.5302`, 95% t-CI
  `[+0.4461, +0.6143]` — wholly ABOVE zero (candidate significantly worse);
  the gate required wholly below. FAIL.
- All-q completed-Q RMSE `1.4162` vs incumbent `1.7482` (ratio `0.8101`,
  ceiling 1.05). PASS — and notably better: the decomposed head removes the
  incumbent's `+1.02` completed-Q bias (candidate bias `+0.05`).
- Mean q-regret `0.7500` vs incumbent `0.7515` (ceiling `0.8015`). PASS.

Component reads on the verdict block: wildlife is the weak axis (RMSE
`2.9018`, bias `-1.42`), habitat `1.8323`, Nature `1.7403`. Top-1 selected
agreement 35.7% vs incumbent 36.4%.

**Interpretation:** the exact-grounded decomposition trains a
better-calibrated completed-Q surface but is substantially worse than the
teacher's own selected-Q at predicting the real final outcome of the selected
action. The ridge preflight's `+15.99%` (closed-form fit on the frozen
latent) did not transfer to a trained frozen-trunk head at pilot scale.
**Preregistered consequence, executed as written: no full-model training, no
gameplay; the 12,000-root fit expansion and the three 1,600-root reserve
holdouts remain quarantined as unused evidence. The direction is closed
pending materially new evidence** (per the gate text in RESEARCH_LOG §4.9).

**Stage 3 (CUDA packed-throughput probe): pass, engineering-only**
(`model_throughput_20260709_cuda.json`, `scientific_eligibility:
engineering_throughput_only`). Production-packed features, batch sweep
1-32, TF32 off. Throughput speedups vs cycle4-M on the RTX 5090: S (5.87x
fewer params) only `1.89x` at batch 8 / `1.68x` at batch 32; synthetic XS
`1.98x / 2.01x`; synthetic tiny (1300x fewer params) `2.82x / 2.20x`. These
are far below the MPS ratios (S `3.06-3.38x`, tiny `9.85-13.66x`): the 5090
is dominated by fixed per-call overheads, so parameter reduction does not
convert to proportional serving throughput. Combined with the MPS equal-wall
result (S n128/d8 scored `-2.17` vs M n64/d4 at `1.078x` wall), the
smaller-model/larger-search direction is **closed on john0 CUDA serving** —
S buys at most ~1.9x search budget where >3x was already insufficient.

**Stage 4 (market sample-4 gate): running.** Smoke passed
(`market_samples_20260709_n256_d4_s4_smoke.json`). The gate reuses the
stage-1 baseline (`exact_k1_20260709_n256_d4_baseline.json`, samples=8,
seeds `2027071400x100`) as the incumbent arm and runs only the samples=4
candidate on the same seeds — a paired design at half the GPU cost. The
seed block registry entry now reflects that block `2027071400-1499` serves
the whole f35 post-chain, not stage 1 alone. Stage 5 (jobs12/16/24
concurrency calibration) follows.

## 2026-07-10 10:45 — K1 verdict finalized under John's ruling: ADOPTED for speed; exact K2 closed

John ruled: **keep K1, exclude the one concurrency-divergent seed by
declaration, leave K2 on model inference.** Implemented as a permanent
comparator amendment, not a one-off edit: `compare_exact_endgame` now takes
`--declared-divergent-seed` + `--exclusion-ruling`. The mechanism fails
closed — a declared seed must actually diverge pre-K1 (a causally clean
seed is refused), any undeclared divergence still aborts, the ruling text
and per-seed first-divergent-ply are embedded in the artifact, and the
eligibility string downgrades to
`promotion_scale_paired_gate_with_declared_exclusions`. Seven new contract
tests; suite 197/197.

Verdict on the 99 causally-valid pairs (inputs hash-verified against john0
before the run; artifact published back to
`cascadiav3/reports/exact_k1_20260709_n256_d4_verdict.{json,md}`, JSON SHA
`2ef285e3...`):

- Paired K1-minus-baseline: `-0.0379`, 95% t-CI `[-0.0859, +0.0101]` —
  inconclusive; **score-neutral**, as the MPS smoke predicted.
- **Seat-0 mean delta exactly `+0.0000` across all 99 games** while K1
  changed 332/400 final actions: on the only seat with a provably identical
  pre-decision state, the incumbent model already selects score-optimal
  final actions and exactness merely substitutes equal-scoring
  alternatives. K1 has no points to win at the final ply.
- Cost: exact frontier `1743.9s -> 60.2s` (**28.99x**, versus 8.86x in the
  MPS smoke); whole-arm `1.035x` mean-decision / `1.034x` wall at n256/d4.

**Adoption:** `--gumbel-exact-endgame-turns 1` is the serving/benchmark
default from here on (recorded in README standing contracts). **Exact K2 is
closed by ruling** — it would need a genuine max^n/chance tree for at most
the same zero points the model already captures at the frontier, so deeper
plies stay on model inference. The eval-noise thesis survives intact: the
final ply is simply the one place the model is already exact-equivalent.

## 2026-07-10 11:00 — Descriptive category profile of the champion: the gap to 100 is diffuse, not a failure mode

CPU-only analysis of the complete published raw game ledgers (99/100 seeds
per arm, pre-replay; descriptive, no paired claim). Champion cycle4
n1024/d16 decomposition: **98.301 = wildlife 61.03 + habitat 32.03 + nature
5.24** (per-species means: fox 15.85, salmon 12.89, hawk 11.87, elk 11.47,
bear 8.95; habitat 6.2-6.6 per terrain).

Three findings:

1. **No catastrophic seats.** Zero of 396 seats scored below 90; p10/p50/p90
   is 95/98/102 and 31.3% of seats already clear 100. There is no rescueable
   failure mode; the remaining ~1.7 points are diffuse marginal decisions.
   This is exactly what the decision-SNR thesis predicts.
2. **Bear zeroing is deliberate, near-cost-free triage — not a leak.** Bears
   score zero in 22.5% of seats (every other species <=1%), forfeiting an
   11.54 conditional mean. But zero-bear seats total 98.03 vs 98.38 for
   bear-scoring seats: the engine recovers ~11.2 of the ~11.5 forfeited
   points elsewhere. Residual cost ~0.08 mean points, uniform across seat
   positions (22/22/19/26). A "fix bears" direction is dead on arrival.
3. **High-budget scaling bought habitat, not wildlife.** n1024/d16 vs
   n256/d4: wildlife +0.25 while habitat +0.60 (31.43 -> 32.03). The budget
   win concentrates in long-horizon spatial planning — the noisiest, most
   plan-dependent component — which is direct empirical support for the
   worlds/determinization sweep (RESEARCH_LOG §5 item 7) as the open lever.

Distq n256/d4 shows the same shape one notch lower (97.311 = 60.78 + 31.43
+ 5.10). Full 100-seed category ledgers still await the two one-seed
replays; this note carries no verdict weight.

## 2026-07-10 11:20 — PREREGISTERED: worlds-allocation screen (search-shape re-sweep, RESEARCH_LOG §5 item 7)

First gate launched under the new AGENTS.md autonomy boundary. Motivation:
(a) determinizations are a free allocation knob, not a cost multiplier —
`gumbel.rs` cycles `det_index = visit_index % determinization_samples`
inside the fixed `n_simulations` budget, so more worlds trade per-world
visit depth for hidden-state coverage at identical wall cost; (b) the
n1024/d16-over-n256/d4 scaling win concentrated in habitat (long-horizon,
noise-dominated); (c) the decision-SNR thesis says averaging more worlds
attacks exactly that noise.

**Screen (launches when the GPU frees, after the two approved d20
replays):** champion cycle4 scalar (`best_locked_val`, manifest
`b8886c24...`), corrected rules, source `f35b0d0b`, n256/top16/depth-rounds
1, K1 on (new default), market samples 8, TF32 off, jobs12. Arm A
determinizations=4 (64 sims/world at the frontier of the budget cycle) vs
arm B determinizations=8 (32 sims/world) — the same 64->32 halving the
champion would see at d16->d32. Fresh disjoint seed block
`2027071500..1599`, both arms paired on all 100 seeds.

**Preregistered verdict:** paired per-seed mean-seat delta (B minus A),
95% t-CI, n=100. Decision rule, fixed now: CI+ -> preregister and launch
the high-budget confirmation (n1024/top16 det16 vs det32, both arms fresh
on reserved block `2027071600..1699`, ~20h). CI- or inconclusive -> the
worlds axis is closed at low budget; no high-budget spend without a new
mechanism argument; report either way. This is a paired score gate, not a
causal comparator — bit-identical traces are not required, so jobs12
concurrency is acceptable.

## 2026-07-10 12:05 — Market sample-4 gate: FAIL per preregistration, sample-8 stays; comparator's trace-frontier premise was invalid for this knob

**Verdict (preregistered rule from CAMPAIGN_STATE 07-09: pass requires
paired t-CI lower bound >= -0.25 AND whole-decision speedup >= 1.15x):**
candidate samples=4 scored `97.1075` vs the samples=8 baseline `97.2650`
(same seeds `2027071400..99`, paired n=100): delta `-0.1575`, 95% t-CI
`[-0.4684, +0.1534]`. The CI floor breaches the `-0.25` noninferiority
margin, so **the gate FAILS and sample-8 remains in place**, despite a
clean `1.575x` mean-decision speedup (`2.102x` on refresh-available
decisions). Artifact:
`market_samples_20260709_n256_d4_verdict.{json,md}` (SHA `aaeef35d...`),
published to john0.

**Instrument finding:** the gate script's comparator inherited the exact-K1
causal-frontier design and crashed ("action trace diverges before
market-sample exposure at seed 2027071406 ply 16"), killing the chain
before stage 5. Forensics over both 8,000-row ledgers: only 1/100 seeds
identical, 57 diverged at/after their first root refresh exposure, and
**42/100 diverged before it** (many at ply 0-3). That is 42x the measured
jobs12 concurrency rate and is mechanism, not noise: `market_decision_
samples` changes interior simulation values at any ply whose search horizon
reaches a refresh chance node, so the exposure-frontier premise is invalid
by design for this knob (the rejected MPS screens had shown the same
signature). The K1 declared-exclusion precedent was NOT applied — this is
not a contaminated-seed problem. `compare_market_samples` now classifies
trace divergence descriptively (identical / pre-exposure / post-exposure)
and verdicts purely on the preregistered score+speedup rule, which never
required trace identity. Contract tests added; the stale scaffold assertion
updated; suite 200/200.

**Queue:** GPU is on the approved d20 replays (cycle4 `2027070908` running,
distq `2027070962` next), then the preregistered worlds screen, then the
stage-5 jobs12/16/24 concurrency probe relaunch (it never started).

## 2026-07-10 12:05 — GPU queue made session-independent: gpu_autochain on john0

Per John's directive, the remaining pipeline no longer depends on the
orchestrator session. `cascadiav3/logs/gpu_autochain.{sh,log,pid}` on john0
(PID `3683183`, HOLD file `HOLD_gpu_autochain`, heartbeats) runs the whole
queue: (A) wait for the in-flight cycle4 seed-2027070908 replay, then
validate+install it fail-closed via a portable copy of
`validate_seed_replay` (`cascadiav3/logs/replay_tools/`); (B) generate,
validate, and install the distq seed-2027070962 replay under the same d20
contract; (C) if both installs pass, build both 100-row game ledgers and
publish the paired category mechanism verdict
(`rules_20260709_n1024_category_verdict.{json,md}`); (D) the preregistered
worlds screen (det4 then det8); (E) the stage-5 jobs12/16/24 concurrency
probe. A replay validation failure is logged loudly and the chain continues
to the screen rather than stranding the GPU (the superseded
`worlds_screen_waiter`, which required 100+100 files unconditionally, was
retired). Estimated pipeline: ~8 GPU-hours queued.

## 2026-07-10 12:35 — Cycle4 seed-2027070908 replay: validated bit-exact and installed; scalar raw ledger complete at 100/100

The d20-contract solo replay of scalar seed `2027070908` (n1024/d16, TF32
off, shared session) passed the fail-closed validator on john0: all 80
chosen-action IDs and refresh decisions match the durable decision ledger,
and the replayed seat totals `[97, 98, 97, 100]` equal the aggregate
report's pinned values exactly. Installed by the guarded 99->100 path;
`rules_20260709_cycle4_n1024_d16_raw_games/` is complete. This also
empirically validates the replay recovery contract itself: a solo rerun
reproduced a game originally generated under jobs12 shared-bridge
concurrency, so this seed was not concurrency-perturbed. The autochain is
now generating the distq seed-2027070962 replay (stage B).

## 2026-07-10 13:00 — Worlds confirmation fully automated behind the preregistered CI+ gate

`cascadiav3/logs/worlds_confirm_waiter.{sh,log,pid}` on john0 waits for
`gpu_autochain` to finish, computes the screen verdict on-box
(`worlds_screen_20260710_n256_verdict.{json,md}` via the stdlib-portable
`compare_search_shape`), and launches
`run_worlds_confirm.sh` (cycle4 n1024/top16, K1 on, det16 vs det32,
reserved block `2027071600..1699`, ~20h) ONLY when
`proceed_to_high_budget` is true — exactly the decision rule preregistered
at 11:20. A non-CI+ screen logs "worlds axis closed at low budget" and
spends nothing. Pause with `touch cascadiav3/logs/HOLD_worlds_confirm`.
With this, every remaining decision point in the preregistered plan
executes without an orchestrator session.

## 2026-07-10 13:15 — Distq seed-2027070962 replay bit-exact; category mechanism verdict published; rebaseline recovery CLOSED

The distq seed-`2027070962` solo replay also validated bit-exact (all 80
ledger actions, pinned totals) and installed: **both n1024/d16 raw ledgers
are complete at 100/100.** The autochain built both game ledgers +
category summaries; its `compare_game_categories` invocation failed closed
on reversed arm orientation (canonical comparison is left=distq,
right=cycle4) and was rerun correctly orchestrator-side on john0.

**Category attribution (distq minus cycle4, n1024/d16, n=100 paired,
cross-validated field-for-field against the canonical total verdict):**

| category | delta | 95% t-CI |
|---|---:|---|
| wildlife | `+0.1450` | `[-0.1458, +0.4358]` |
| habitat | `-0.0500` | `[-0.2353, +0.1353]` |
| nature | `-0.0075` | `[-0.1765, +0.1615]` |
| total | `+0.0875` | `[-0.2411, +0.4161]` |

The high-budget head tie is flat in EVERY category — no hidden mechanism
trade behind the aggregate tie; the heads are simply equivalent at n1024.
`fetch_rules_n1024_verdict.sh` harvested the complete artifact set
(category verdict SHA `92ea815b...`) into
`cascadiav3/reports/rules_20260709_rebaseline_complete/` on the
orchestrator. Every item from the 07-09/10 recovery contract is resolved.
The worlds screen det4 arm is on the GPU (stage D).

## 2026-07-10 19:20 — Worlds screen CI+ (det8 over det4); n1024 det16/det32 confirmation auto-launched; stage-E probe failure diagnosed (silent nvidia-smi PATH guard) and re-queued

**Worlds screen verdict (preregistered 07-10 11:20, computed on-box by
`worlds_confirm_waiter` at 19:15:31):** cycle4 scalar, corrected rules, source
`f35b0d0b`, n256/top16/d1, K1 on, jobs12, fresh block `2027071500..99`, 100
matched seeds per arm. det4 mean seat `97.1425`; det8 mean seat `97.5650`;
paired delta **`+0.4225`, 95% t-CI `[+0.1045, +0.7405]` — CI+**, the first
CI-positive search-shape result under corrected rules. Artifacts:
`worlds_screen_20260710_n256_{det4,det8}.{json,md}` + decision/game ledgers,
`worlds_screen_20260710_n256_verdict.{json,md}`.

**Cost caveat (do not quote this as a free knob):** the screen's premise was
that worlds cycle inside a fixed `n_simulations` budget at identical wall
cost. Measured: mean decision `12.2913s -> 18.3761s` (`1.495x`), whole-arm
wall `8513.7s -> 12778.2s`. More distinct worlds lower the eval-row dedup hit
rate, so det8 buys its `+0.42` at ~1.5x wall at n256. The preregistered
verdict rule (paired score CI, not wall-matched) still fires as written, but
any adoption decision after the confirmation must weigh the wall multiplier,
and a cost-matched det4-at-higher-n comparison remains the fair frontier
question.

**Confirmation launched automatically (19:15:31), exactly per the
preregistered CI+ rule:** `run_worlds_confirm.sh` — cycle4 n1024/top16, K1
on, det16 vs det32, reserved block `2027071600..1699`, det32 one-game smoke
first, ~20h total. Pause file `HOLD_worlds_confirm`. Note the prior related
evidence: legacy n2048/d32 was CI− vs n1024/d16 (that comparison held 64
sims/world in both arms); this confirmation instead halves per-world visits
(32 -> 16... i.e. 64/world at det16 vs 32/world at det32) at fixed n=1024 —
the same trade the screen just validated at n256.

**Stage E (jobs12/16/24 concurrency probe) failed instantly at 19:10:50 with
no output; root cause found:** `run_cuda_concurrency_probe.sh` preflights
with bare `test`/`grep`/`command -v` lines under `set -euo pipefail`;
`command -v nvidia-smi` fails in detached shells because `nvidia-smi` lives
only at `/usr/lib/wsl/lib/nvidia-smi` on john0 and `/usr/lib/wsl/lib` is not
on the non-interactive PATH. The guard exited silently (no message), so the
autochain logged only `CONCURRENCY-PROBE-FAILED`. Verified by rerunning each
guard individually over ssh: manifest OK, dynamic_seed_queue grep OK,
comparator OK, `command -v nvidia-smi` exit 1.

**Fixes:** (1) local `main` edit to
`cascadiav3/scripts/run_cuda_concurrency_probe.sh` (uncommitted): a loud
`preflight` helper for every guard, explicit `NVIDIA_SMI` resolution with an
`/usr/lib/wsl/lib` fallback, and `start_profile` now invokes `"$NVIDIA_SMI"`.
The pinned f35 tree on john0 was deliberately not modified. (2) The probe was
re-queued session-independently:
`cascadiav3/logs/concurrency_probe_waiter.{sh,log,pid}` on john0 (PID
`3731156`, pause file `HOLD_concurrency_probe`) waits for the worlds
confirmation and exporter/bridge idle, then relaunches the pinned probe with
`PATH="$PATH:/usr/lib/wsl/lib" SOURCE_REVISION=f35b0d0b...`; probe output
streams to `cascadiav3/logs/cuda_concurrency_probe_run.log`. This restores
the preregistered queue order (screen -> probe -> confirmation) as closely as
possible given the confirmation had already claimed the GPU when the failure
was diagnosed.

**Also this session (research planning, no score claim):**
`claude_max_research_ideas.md` written at the repo root — a tiered
break-100 research portfolio (root-decision variance engineering, opponent
marginalization, cooperative/table-native reframe, factored compound
actions, training-signal densification, velocity infrastructure), each
direction with a preregisterable kill test. Grounded in a code audit
(4 model evals/simulation at depth_rounds=1, 3 of them animating opponents;
hardcoded `c_visit=50/c_scale=1.0` over min-max-normalized Q; per-action
rollout RNG breaking CRN at leaves; greedy-ranked root-menu truncation at
256; root refresh = 9–10 full searches) and a salvaged literature sweep.

## 2026-07-10 21:10 — Worlds confirmation PAUSED by ruling; john0 reallocated to the break-100 portfolio; concurrency probe left to auto-fire

**Ruling (John, ~21:00):** the ~20h det16/det32 n1024 confirmation is the
lowest conviction-per-GPU-hour item on the board — it is not wall-matched
(the measured scaling anchor of ~+1.1/4x budget predicts ~+0.3 for its
~1.5x wall multiplier, so even a CI+ verdict is ambiguous between "worlds
are a special knob" and "1.5x compute helps"), and adoption would tax every
future gate ~1.5x on both arms. john0 is reallocated to the Tier-0
root-decision program + velocity items of `claude_max_research_ideas.md`.
The confirmation is **paused, not closed**: block `2027071600..1699` stays
reserved for it in the seed registry, and a future rerun repeats both arms
from scratch under the same preregistration (the benchmark fails closed on
stale raw dirs; no mixing is possible).

**Actions (21:06–21:07, explicit user permission):** set
`HOLD_worlds_confirm`; killed the confirm tree in runbook order —
`run_worlds_confirm.sh` (PID 3730663), benchmark python (3736837), exporter
batch (3736838), torch bridge (3736841). Verified zero surviving processes
and GPU idle (0% util, 1138 MiB residual).

**What was preserved / lost:** the det32 one-game smoke report
(`worlds_confirm_20260710_n1024_det32_smoke.json`, mean decision `42.32s`
at det32/n1024 — a wall datum consistent with the det8 dedup caveat) is
intact. The det16 arm (launched 20:11) had **zero completed games** — no
raw-games directory had been created yet (first games land ~56 min in), so
nothing durable was lost and no partial scores existed to be read.

**Queue:** `concurrency_probe_waiter` (PID 3731156) was deliberately left
untouched; with the confirm tree gone it detects the idle box and fires the
preregistered jobs12/16/24 probe with the PATH fix. Output →
`cascadiav3/logs/cuda_concurrency_probe_run.log`.

**Next (per portfolio, John's blanket go-ahead "john0 is all yours"):**
Tier-0 implementation (R0.1 sigma calibration flags, R0.2 paired-rollout
CRN), preregistered sigma sweep chained behind the probe, and the zero-GPU
audits (R1.1a contention, R1.3a greedy-256 coverage, R2.3 CUPED) on stored
ledgers.

## 2026-07-10 21:55 — PREREGISTERED: R0.1 sigma-calibration sweep (n256 screens + auto-gated confirm); R0.1/R0.2 knobs implemented; comparator generalized; probe cargo-PATH failure diagnosed

**Implementation landed (commit `5815718f` + follow-up):**
`--gumbel-c-visit`, `--gumbel-c-scale`, `--gumbel-sigma-norm
(minmax|zscore|fixed:<scale>|topk:<k>)`, and `--gumbel-paired-rollouts`
(CRN leaf rollouts: stream keyed to (determinization stream, visit index)
instead of (search seed, action index, visit index)). Defaults are
bit-identical to the incumbent — the legacy unpaired seed formula is
regression-pinned by test; 51/51 exporter tests pass (5 new).
`cascadiav3.compare_search_shape` generalized with repeatable
`--varied-key` (default `determinizations`, back-compatible; 7/7 contract
tests pass). Benchmark plumbing records all four knobs in `search`
provenance; flags are emitted to the exporter only when non-default so
default invocations stay replayable against older pinned binaries.

**Concurrency probe attempt 2 FAILED at 21:08:47 — second missing-PATH
class:** the waiter's `/usr/lib/wsl/lib` fix exposed the next gap:
`cargo: command not found` (line 92 of the pinned f35 script; `$HOME/.cargo/bin`
is also absent in detached shells). The fixed script (committed 04381253)
makes this loud. Remedy: the relaunch chain (below) exports
`PATH="$HOME/.cargo/bin:$PATH:/usr/lib/wsl/lib"` before invoking the probe.
The one-shot waiter exited after its failure branch; superseded by the chain.

**PREREGISTRATION — R0.1 sigma-calibration sweep (before any candidate
output exists):**

- *Motivation.* `sigma(q) = (c_visit + max_visits) * c_scale * minmax(q)`
  with c_visit=50 / c_scale=1.0 hardcoded Go defaults; measured decision SNR
  ~= 1; the Gumbel paper's own mitigation for noisy Q is a smaller c_scale
  (0.1 for Atari); min-max normalization lets one terrible candidate
  compress contender gaps (topk:<k> windowing is immune by construction).
- *Config (all arms).* cycle4 scalar `best_locked_val`, corrected rules,
  n256/top16/d4/depth1, K1 on, market samples 8, blend 0.5, k-interior 16,
  jobs12, batch runner, control none, TF32 off, CUDA on john0.
- *Screen arms (8).* c_scale in {0.05, 0.1, 0.25, 1.0} x sigma-norm in
  {minmax, topk:8} on 25 paired seeds `2027072100..24` (selection block;
  registry updated). The (1.0, minmax) arm IS the incumbent and is the
  paired baseline for the other 7. One-game smoke of the most exotic arm
  (0.05, topk:8) runs first.
- *Screen rule.* Best candidate arm by paired mean delta vs incumbent
  proceeds to the confirm iff mean >= +0.25 (7-arm selection is expected to
  inflate the best mean under the null; the confirm on a disjoint block is
  the arbiter). No CI requirement at screen scale (25 seeds is selection,
  not verdict). All-arms-below-floor => R0.1 CLOSED with the sweep table as
  the artifact.
- *Confirm rule (the R0.1 kill test).* Winner vs incumbent, 100 paired seeds
  `2027072200..99` (touched once), campaign-standard 95% t-CI
  (`compare_search_shape --varied-key c_scale --varied-key sigma_norm`).
  CI+ => preregister an n1024 confirmation on a fresh block before any
  adoption; inconclusive or CI- => R0.1 CLOSED.
- *No partial reads.* Verdicts are computed on-box by
  `cascadiav3/scripts/run_sigma_sweep.sh` (checked in; smoke -> 8 arms ->
  7 verdicts -> selection -> conditional confirm -> complete marker
  `sigma_sweep_20260710_n256_complete.json`). Pause: `HOLD_sigma_sweep`.
- *Cost estimate.* ~36 min/screen arm (worlds-screen anchor: 85 s/game at
  n256/d4 jobs12) => ~4.8 h + smoke; conditional confirm ~2 x 2.4 h.

**Chain:** deployed as `gpu_chain_20260710_sigma.sh` on john0 (session-
independent, pause `HOLD_gpu_chain_sigma`): stage A = concurrency probe
attempt 3 at the new deployed revision (engineering-only; failure tolerated,
logged, does not block stage B), stage B = the sigma sweep. New source
snapshot deployed to john0 with the revision marker updated; exporter
rebuilt with zig-cc and flag presence preflighted by the sweep script.

## 2026-07-10 22:36 — PREREGISTERED: R0.2 offline stability probe + R1.1a contention audit (ledger-replay modes implemented); chained behind the sigma sweep

**Implementation:** two new exporter modes over a shared no-search ledger
replayer (reconstructs every root row exactly as serving saw it — same
greedy menu cap, same market prelude — and advances via the ledger's
`chosen_action_id`; fails closed on seat/menu divergence, and skips
concurrency-divergent seeds loudly with a `replay_skipped_seeds` summary
field, 2027071427 precedent):

- `--search-stability-probe` (R0.2 offline kill test): samples ledger roots
  (stride 79 spans all 100 games; exact-K1 frontier roots excluded — they
  serve exactly), re-runs each root search 6x unpaired vs 6x paired (CRN)
  rollouts at n256/top16/d4, blend 0.5. Equal repeat index = equal search
  seed across variants, so worlds match and only the rollout-noise
  structure differs. Analyzer: `cascadiav3.analyze_search_stability`.
- `--table-contention-audit` (R1.1a): for every ledger decision, table
  value (value-head per-seat sum; exact at terminals) of the chosen
  afterstate vs the best model-Q alternative. Analyzer:
  `cascadiav3.analyze_table_contention` — flip rates + recoverable
  table/gate points per game, overall and conditioned on own-Q sacrifice
  <= {0.1, 0.25, 0.5, 1.0}. Measurement only; caveats recorded in the
  analyzer output (model-Q runner, value-head estimates, first-order).

Tests: 54/54 exporter (3 new: replay round-trip against a played game's
terminal scores; audit end-to-end on the mock bridge; probe seed-pairing
contract) + 5/5 analyzer contract tests.

**PREREGISTRATION (R0.2 offline rule, before any probe output exists):**
proceed to a preregistered n256 paired-rollouts gate iff the probe shows
pooled top1-top2 completed-Q gap variance (visited actions) at least **20%
lower** under paired rollouts (`proceed_to_gate` in
`search_stability_probe_20260710_analysis.json`). Flip-rate deltas are the
secondary read. All-null => R0.2 closes without a gate. The contention
audit has no decision rule — it sizes the R1.1 prize (>= 0.3 gate
points/game at sacrifice <= 0.25 is the portfolio's bar for prioritizing
R1.1b/c).

**Scheduling:** both run on john0 **after** `gpu_chain_20260710_sigma.sh`
exits (no CPU contention with the probe/sweep timing): a second chain
waiter deploys the new revision (tarball staged now; marker + rebuild only
after chain 1 exits, so the sweep's preflights stay valid), then runs the
stability probe (~1h GPU) and the contention audit (~8,000 root evals +
~16,000 afterstate evals, minutes on CUDA). Input ledger:
`rules_20260709_cycle4_n1024_d16_decisions.jsonl` (champion trajectory at
champion budget).

## 2026-07-10 23:37 — Concurrency probe attempt 3: ALL THREE ARMS COMPLETE; comparator crashed on a divergence invariant — root-caused and fixed; verdict recompute queued

**The probe's data collection succeeded** (first time through all arms):
jobs12/16/24 reports + decision/game ledgers + GPU profiles all landed at
source `83ffe12a` (~53 min total). The chain then failed only in
`compare_cuda_concurrency`: `decision invariant mismatch at seed 2027073423
ply 71` — the jobs24 arm flipped one argmax (batch-order float
nondeterminism, the measured 2027071427 class), and the comparator
tolerated the action difference but hard-enforced per-ply state invariants
*downstream of the fork*, where the two arms are legitimately playing
different games. Design bug, same premise-failure family as the
market-samples comparator (EXPERIMENT_LOG 07-10 10:20).

**Fix (this commit):** `_compare_arm` now walks each seed to its
divergence frontier — invariants are enforced strictly up to and including
the first chosen-action flip (root states there are provably identical, so
a mismatch is a real bug and still raises), while downstream plies are
excluded from comparison and the seed is classified in
`divergent_seeds`/`divergent_seed_count`. Verdict adds paired per-seed
score-delta stats (t-CI); knee eligibility is now pre-divergence numeric
parity (full-trajectory parity is unattainable under measured jobs
nondeterminism — with the old rule the recommendation could only ever
collapse to jobs12). Markdown surfaces divergent-seed counts and score
deltas. 5/5 new contract tests; recommendation semantics unchanged
otherwise (throughput knee, advisory only, never auto-adopted).

**Recompute:** chain 2 re-armed at the new revision; its first stage runs
the fixed comparator over the existing three arm artifact sets (no GPU,
reports reused byte-identical, `--source-revision 83ffe12a` matching the
arms) and writes the standard verdict + complete marker, then proceeds to
the R0.2 stability probe and R1.1a contention audit as preregistered.

## 2026-07-11 04:55 — Sigma sweep screens COMPLETE: all 7 candidate arms positive, c_scale=0.25 tops both families; confirm auto-launched (cs025_tk8)

Screen results (25 paired seeds `2027072100..24` vs the incumbent
cs10_mm arm; per-arm 95% t-CIs all straddle zero, as expected at screen
scale): `cs025_tk8 +0.70` > `cs025_mm +0.65` > `cs005_tk8 +0.36` >
`cs01_tk8 +0.34` > `cs01_mm +0.32` > `cs10_tk8 +0.21` > `cs005_mm +0.00`.
Two reads worth recording: (1) within BOTH normalization families the
ordering is a dose-response with an interior optimum at c_scale 0.25 —
consistent with a genuinely miscalibrated sigma rather than noise; (2)
topk:8 ≈ minmax at equal c_scale (small consistent edge to topk:8), so the
c_scale shrink is doing most of the work. Caveat: every delta shares the
single incumbent arm, so incumbent bad luck would shift all seven
positive together — the disjoint-block confirm is the arbiter.

**Preregistered rule fired:** best mean `+0.70 >= +0.25` floor →
`proceed_to_confirm=true`; the 100-seed confirm (incumbent vs cs025_tk8,
block `2027072200..99`, touched once) auto-launched 04:49:53. CI+ there →
preregister an n1024 confirmation; else R0.1 closes. Artifacts:
`sigma_sweep_20260710_n256_*.{json,md}` + `_selection.json` on john0.

## 2026-07-11 09:45 — R0.1 CLOSED (confirm null on the disjoint block); concurrency verdict: retain jobs12 (throughput flat, bridge-bound); chain 2 live

**R0.1 sigma calibration: CLOSED per preregistration.** The 100-seed
confirm (block `2027072200..99`, touched once) of screen winner cs025_tk8
(c_scale 0.25, topk:8) vs the incumbent came back `-0.2325`, 95% t-CI
`[-0.5440, +0.0790]` — inconclusive, mean on the wrong side. The screen's
+0.70 (and the 7/7-positive arm pattern) did not survive the disjoint
block: it was the shared-baseline artifact called out at screen time — one
lucky/unlucky incumbent arm shifts every candidate delta together.
**Methodological note for future sweeps:** a shared-baseline screen floor
does not protect against baseline luck; prefer two independent baseline
replicates (floor on the candidate-vs-worse-baseline delta) or an
ordering-only screen. Wall was also not free: `11.98s -> 12.20s` mean
decision. Verdict artifact: `sigma_confirm_20260710_n256_verdict.{json,md}`.
The noise-wall program moves to R0.2 (paired rollouts — offline probe
running now) and R0.3 (unvisited-Q bias correction).

**Stage-5 concurrency probe: RESOLVED — retain jobs12.** Fixed comparator
over the three completed arms: jobs16 `1.033x`, jobs24 `1.051x` vs jobs12,
GPU util ~66% mean at ALL three settings — the shared bridge, not the job
count, is the throughput bound at n64/d4; R2.4 gains must come from
bridge-side work (torch.compile, CUDA graphs). Divergence classification:
1 divergent seed at jobs16, 0 at jobs24 (plus the jobs24 fork seen in the
crashed run — forks are rare and real). Caveat recorded: knee eligibility
used the replay-grade root-value drift tolerance (2e-05) while cross-jobs
drift is inherently ~0.1-0.3, so jobs16/24 were excluded from knee
selection; the recommendation is unchanged because throughput is flat —
future cross-jobs runs should pass `--max-root-value-drift` ~0.5.
Artifacts: `cuda_concurrency_20260709_n64_d4_verdict.{json,md}` + complete
marker.

**Chain 2 now on stage A** (R0.2 search-stability probe, ~1h), then stage
B (R1.1a contention audit).

## 2026-07-11 11:00 — R1.1a contention audit MEASURED: no cheap cooperative points at the root; R1.1b/c deprioritized per the preregistered bar

Audit ran on the full champion cycle4 n1024/d16 ledger (100 games, 8,000
decisions, 0 replay-skipped seeds) in ~3 min on CUDA. Findings, in order
of load-bearing-ness:

1. **The naive positive-part bound is noise-dominated and must not be
   quoted:** +10.2 "recoverable" gate pts/game with a 51% flip rate is
   what summing `max(0, delta)` over ~80 value-head-noise deltas per game
   manufactures (98.8% of table pairs are value-head estimates,
   per-decision delta spread p10/p90 = -1.14/+1.30).
2. **Sanity check passed:** on the 100 exact-pair decisions (final
   personal turns, noise-free) flip rate is exactly 0% — mechanically
   correct: at the last own turn, own-optimal IS table-optimal because the
   other seats are fixed.
3. **The signal tracks own-Q, not contention.** Binned by own-Q sacrifice
   (chosen minus runner), the mean table delta is monotone from `+1.07`
   (runner own-Q much better, n=153) through `-0.03` at own-Q parity
   (n=2,962, flip 47.8%) to `-0.67` (runner much worse, n=28): the
   value-head table sum simply moves with the runner's own seat. The
   directional per-game sum (+7.1 table pts, CI+ [+3.4, +10.8]) therefore
   measures search-vs-model own-Q disagreement, not harvestable
   cooperative headroom.
4. **Cheap contention ≈ 0.** At |sacrifice| <= 0.1 the mean table delta is
   `-0.034`/decision — the preregistered bar (>= 0.3 gate pts/game at
   sacrifice <= 0.25 to prioritize R1.1b/c) is decisively not met.

**Ruling per preregistration: R1.1b (persona-table probe) and R1.1c
(table-native training) are DEPRIORITIZED.** Caveats that keep R1.1/R3.1
alive as a *diagnosis* question, not a next-action: the audit only bounds
root-level action swaps (interior-ply max^n behavior inside simulations
and trajectory-level strategy shifts are unmeasured), the alternative
ranking is model-Q (search completed-Q runners are not recoverable without
re-searching), and a value head trained on own-trajectory outcomes may be
blind to cross-seat effects. The sharper instrument for the equilibrium
question remains R3.6 (ceiling measurement). Artifacts:
`table_contention_audit_20260710{.jsonl,_analysis.json,_analysis.md}` on
john0; binned analysis in this entry (orchestrator-side).

## 2026-07-11 11:50 — R0.2 CLOSED at the preregistered rule (secondary CI+ noted); PREREGISTERED: R3.6 mega-budget ceiling probe (n4096/d16, launching now)

**R0.2 valid rerun (94 roots, serving-matched rollout params):** primary
pooled top1-top2 gap variance `0.020538 -> 0.021438` = **-4.4%** against
the preregistered >=20% floor — `proceed_to_gate=false`, **R0.2 CLOSED**.
Secondary (registered as secondary): chosen-action flip rate across
repeats dropped `0.4663 -> 0.4238`, per-root delta `+0.0426`, 95% t-CI
`[+0.0051, +0.0800]` — CI+. Reading: pairing measurably stabilizes the
final selection but the across-repeat gap variance is dominated by world
resampling, which pairing does not touch (a metric-design lesson recorded
for future offline probes: operationalize decision stability, not gap
dispersion). Standing disposition: paired rollouts ride along in a future
composed serving-v2 candidate gate (Tier-0 composition), no standalone
gate. Artifacts: `search_stability_probe_20260710{.jsonl,_analysis.*}`;
the invalid top-k-1 run is archived beside them as `*_invalid_topk1*`.

**PREREGISTRATION — R3.6 mega-budget ceiling probe (before any candidate
output exists):**

- *Question.* Does budget scaling continue at 4x the champion's simulation
  count, i.e., is the selfish-policy asymptote plausibly >= 100? This is
  the portfolio-allocation question (noise-reduction lane vs objective
  lane).
- *Design.* One arm: cycle4 scalar, n4096/top16/d16/depth1, market samples
  8, blend 0.5, k-interior 16, **K1 off** (exact-endgame-turns 0) to match
  the stored baseline exactly, jobs12 batch runner, TF32 off, 25 games on
  seeds `2027070900..24` — the rebaseline battery block, added-arm
  pattern; paired per-seed against the stored champion n1024/d16 scores
  from `rules_20260709_cycle4_n1024_d16.json` (25-seed baseline mean
  98.32). Revision note: arms run at `a48fc7d3` vs baseline `f35b0d0b`;
  default-path serving is bit-identical between them (regression-pinned),
  recorded here because compare_search_shape would refuse the revision
  mismatch — the verdict is computed by a pinned inline analysis instead.
- *Preregistered reads (informative probe, NOT promotion evidence; 25g CI
  ~ +/-0.6):* log-linear budget extrapolation from the measured +1.23/16x
  anchor predicts **+0.615** for this 4x step. Bands: paired mean >= +0.45
  => scaling lane OPEN; +0.15..+0.45 => decelerating, asymptote likely
  just under 100, objective work co-prioritized; <= +0.15 => scaling lane
  effectively closed for this policy family at feasible budgets. Wall and
  per-decision stats recorded as R0.6 inputs.
- *Cost.* ~8h at jobs12 (42s/decision anchor x ~4).

## 2026-07-11 18:50 — R3.6 ceiling probe: band = DECELERATING (+0.21 observed vs +0.615 log-linear); portfolio reweights toward structural programs

Mega arm (cycle4 n4096/top16/d16, K1 off, jobs12, 25 games on
`2027070900..24`, 7.0h wall, mean decision `131.42s` = 3.1x the n1024
anchor for 4x sims — dedup improves with n): mean seat `98.5300` vs the
stored champion n1024/d16 baseline `98.3200` on the same seeds. Paired
delta **`+0.2100`**, 95% t-CI `[-0.5925, +1.0125]`. Preregistered band:
**decelerating** (+0.15..+0.45): the point estimate is ~1/3 of the
log-linear prediction, though the 25g CI does not exclude it.

**Portfolio consequence (taken together with day one):** the three cheap
noise-wall bets (R0.1, R0.2, sigma/CRN family) returned null at n256, root
contention is ~zero at own-Q parity (R1.1a), and raw budget scaling looks
sub-log-linear at 4x. Under the portfolio's own falsification framework,
weight shifts to the structural programs: **R1.2 ghost opponents** (reclaim
the 3-of-4 opponent-eval tax = ~4x effective budget at equal wall),
**R1.4 training-signal densification** (un-saturating EI is the only lane
that raises the policy family itself), **R1.3a coverage audit**, with the
**R2.x velocity stack** (puzzle bank, bridge throughput) as multipliers.
A 100-seed completion of the mega arm (~21h) could tighten the scaling
coefficient but would not change the next actions; not scheduled.
Artifacts: `ceiling_probe_n4096_20260711{.json,_verdict.json,_verdict.md}`
+ ledgers on john0.

## 2026-07-11 18:54 — PREREGISTERED: R2.1 puzzle bank (mega-budget root resolution, launching tonight)

**Implementation:** `--puzzle-bank` exporter mode — ledger replay +
stride-selected root resolution, worker-pooled across seeds against one
shared bridge (the saturation pattern; first tool built under the 07-11
rule). One JSONL shard per seed + a bank manifest; the same mode at
candidate flags with repeats=1 produces a screen run, scored against the
bank by `cascadiav3.analyze_puzzle_screen` (bank-regret: bank-best mean
completed-Q minus bank value of the candidate's chosen action). 56/56
exporter tests (1 new end-to-end on the mock bridge) + 9/9 analyzer tests
(3 new).

**Bank design (preregistered):** champion cycle4 n1024/d16 ledger, stride
11 (~727 roots spanning all 100 games and phases), resolution =
n4096/top16/d16, repeats 2 averaged (repeat-agreement recorded per root as
a quality signal), serving rollout params (64/4), K1 setting irrelevant
(exact-frontier roots excluded), jobs12 shared bridge. Est. ~4.5h.
Artifacts → `cascadiav3/reports/puzzle_bank_20260711_n4096/`.

**Preregistered acceptance check (before first use as a screen):** run two
screens against the bank — the incumbent config and the R0.1 confirm loser
cs025_tk8 (both n256/d4, repeats 1, ~35 min each). The bank is accepted as
a screening instrument iff (a) incumbent mean bank-regret is materially
positive (searches at 1/16 budget must show regret vs n4096 truth), and
(b) cs025_tk8 does NOT show materially lower regret than the incumbent
(the 100-seed confirm measured it -0.23 +/- 0.31 — a screen that ranks it
clearly better contradicts gate truth and fails validation). Quantified:
(a) incumbent mean regret >= +0.05; (b) regret(cs025_tk8) - 
regret(incumbent) >= -0.02. Fail either => the bank is not used for
go/no-go decisions until the discrepancy is understood.

**Use policy:** screens rank candidates and allocate gates; they are never
promotion evidence and never overrule a paired gate.

## 2026-07-12 01:02 — PREREGISTERED: Tier-0.5 screen wave (ghost opponents, bias correction, LCB, combo) + R1.3a coverage audit; autonomous pipeline complete

**Code complete (commit cf3528e9 + queue orchestrator):** four new serving
flags with bit-identical defaults and full provenance —
`--gumbel-ghost-opponents` (R1.2A: interior non-root plies via CPU greedy,
zero model evals), `--gumbel-q-bias-correction` (R0.3), `--gumbel-lcb-c`
(R0.4), `--gumbel-refresh-sample-divisor` (R0.6i). 61/61 exporter tests
(5 new). Plus: `run_bank_screen.sh` (generic EXTRA_FLAGS candidate screen
vs the frozen n4096 bank), `run_menu_coverage_audit.sh` +
`analyze_menu_coverage` (R1.3a), and `run_experiment_queue.sh` +
`experiment_queue.py` (JSONL-config sequential queue runner: HOLD pause,
done-marker resume, per-stage logs, failure-tolerant). The full research
loop is now CLI+config operable: queue file -> screens -> verdicts.

**PREREGISTRATION — screen wave (queue_20260712_screen_wave.jsonl), gated
on bank acceptance passing:** all screens at the candidate tier (n256/d4,
repeats 1) on the bank's 700 roots; metric = mean bank-regret vs the
incumbent screen (puzzle_screen_20260711_incumbent). Proceed-to-gate rules
(screens select, gates decide):
- `ghost_opponents`: proceed to a preregistered WALL-MATCHED n256 gate iff
  its equal-budget regret penalty vs incumbent is <= +0.020 (the screen
  measures ghost bias only; the gate buys the reclaimed ~3x evals back as
  n/d).
- `q_bias_correction`: proceed to an n256 gate iff regret delta <= -0.010.
- `lcb_c1`: proceed iff <= -0.010.
- `qbias_lcb_combo`: proceed iff <= -0.015.
- `refresh_sample_divisor` is NOT bank-screenable (the bank mode never
  exercises the refresh machinery); its test is a wall-matched 25g probe
  (speed reallocation), to be preregistered separately.
- R1.3a coverage audit is measurement (drop rate + regret of the greedy-256
  cap vs full menus at n1024/d8); the portfolio's R1.3 program stands or
  falls on it (drop rate <1% and regret <0.01/root => close R1.3a-c).

Queue launches autonomously after the bank acceptance verdict; results in
`cascadiav3/reports/puzzle_screen_<name>_analysis.{json,md}` and
`menu_coverage_20260712_analysis.{json,md}`.

## 2026-07-12 01:20 — PREREGISTERED: generic paired-gate runner + refresh-divisor gate; conditional gate templates registered

**`run_paired_gate.sh` (new):** every future gate is now a queue entry —
two arms on one registered block, env-parameterized flags/budgets, verdict
via `compare_search_shape --varied-key ...`. Seed blocks registered:
`2027072300..99` (refresh gate), `2027072400..99` (ghost, conditional),
`2027072500..99` (serving-v2 combo, conditional).

**PREREGISTRATION — refresh-divisor gate (unconditional, may launch in any
free GPU window):** baseline = champion n256/d4 K1-on vs candidate = same
+ `--gumbel-refresh-sample-divisor 4`, 100 paired seeds `2027072300..99`.
Rule (mirrors the market-samples pattern): candidate is adopted for the
serving default iff the paired score CI floor is above the `-0.25`
noninferiority margin AND the verdict timing shows a mean-decision-seconds
saving (refresh plies are ~1/3 of wall; expect ~15-25% overall). CI floor
below margin => R0.6(i) closes.

**Ghost + combo gates are templates only** (`queue_20260712_gates_template
.jsonl`): their launch conditions and (for ghost) the wall-parity CAND_N
come from the screen-wave results; each requires its own completed
preregistration entry before launch.

## 2026-07-12 01:15 — R2.1 puzzle bank ACCEPTED; screen wave running

Acceptance verdict (preregistered 07-11 18:54): incumbent n256/d4 mean
bank-regret `0.2351` (check a: material, >= 0.05) and cs025_tk8 `0.2290`
(check b: delta `-0.006`, not materially better — matching the 100-seed
gate truth of `-0.23 +/- 0.31` ns). **The screen instrument agrees with
gate truth on the one cross-checkable case; the bank is accepted for
candidate ranking.** Screens complete in ~6 min each (worker-pooled).
The five-stage screen wave launched at rev `e252d68e` (01:14); the
refresh-divisor gate waiter is armed behind it. Task R2.1 complete.

## 2026-07-12 01:25 — Ghost screen PASS (+0.0074 vs +0.020 bar); PREREGISTERED + ARMED: ghost wall-matched gate

Screen: ghost mean bank-regret `0.2425` vs incumbent `0.2351` on the same
700 roots — bias penalty `+0.0074`, well under the preregistered `+0.020`.
Removing ALL opponent model evals costs ~nothing in root-decision quality
at equal n. Wall: ghost `1.65x` faster at n256 (230.5s vs 379.4s per
screen).

**PREREGISTRATION — ghost wall-matched gate (armed, runs after the refresh
gate):** baseline = champion n256/d4 K1-on; candidate = ghost + n512/d4
(reinvesting the reclaimed evals; expected wall ~1.2x baseline). 100
paired seeds `2027072400..99`. Rules: verdict CI+ (paired score, 95% t-CI
above zero) AND candidate mean decision seconds <= 1.25x baseline =>
R1.2A graduates to an n1024-tier confirmation (fresh preregistration);
CI- => ghost bias dominates, cap R1.2 at Stage A and revisit; ns => the
reclaimed budget buys nothing at this tier — retest at n1024 pricing
before closing. Varied keys: ghost_opponents + n_simulations.

## 2026-07-12 01:35 — Screen wave verdicts: ghost PASS (gate armed); qbias structurally null at serving; LCB + combo flat

Final screen table (mean bank-regret vs incumbent `0.2351`, bar in
parens): **ghost `+0.0074` (≤+0.020, PASS — wall-matched gate armed)**;
q_bias_correction `0.0000` (≤-0.010, no) — structurally null for the root
CHOICE at n256 because every top-m action gets visited, so the unvisited
correction never touches the chosen action; its value moves to
improved-policy training targets (R1.4 program); lcb_c1 `-0.0002` (≤-0.010,
no); combo `-0.0002` (≤-0.015, no — as implied by its components). The
serving-v2 composition idea is closed at n256 serving; paired rollouts'
CI+ selection stability remains available to future compositions. Coverage
audit runs next, then the refresh and ghost gates. Morning digest:
`morning_report.sh`; handoff: `docs/handoffs/handoff-2026-07-12.md`.

## 2026-07-12 05:50 — R0.6(i) refresh-divisor gate: ADOPTED (noninferior, 1.24x decision speedup); ghost gate launched

**Verdict (preregistered 01:20, applied literally):** candidate
(champion n256/d4 K1 + `--gumbel-refresh-sample-divisor 4`) vs baseline on
100 paired seeds `2027072300..99` at rev `e252d68e`:

- Paired score delta `+0.0375`, 95% t-CI `[-0.1611, +0.2361]` — CI floor
  is above the preregistered `-0.25` noninferiority margin. ✓
- Mean decision seconds `11.4195 -> 9.1872` (`1.243x`); whole-arm wall
  `7902.4s -> 6391.1s` (`1.236x`) — inside the preregistered 15-25%
  expectation. ✓

Both conditions hold => **`--gumbel-refresh-sample-divisor 4` is adopted
as the serving/benchmark default** (mirrors the exact-K1 adoption
pattern: score-neutral, pure speed; the market sample-4 knob it replaces
failed this same rule on 07-10). Refresh-decision *sample count* stays 8;
only the per-sample search budget is divided. Adoption applies to future
invocations — the ghost gate already running (preregistered 01:25,
launched 05:45) keeps its preregistered divisor-1 arms untouched.
Artifact: `gate_refresh_div4_20260712_verdict.{md,json}` on john0.

Chain state after this verdict: ghost wall-matched gate running (n512/d4
ghost vs n256/d4 champion, seeds `2027072400..99`, ~5-6h); coverage-audit
rerun waiter armed behind it (deploys rev `1c9211a5`, then reruns
`run_menu_coverage_audit.sh`).

## 2026-07-12 10:35 — R1.2A ghost wall-matched gate: CI+ (+0.545 at 1.05x wall) — GRADUATES to n1024-tier confirmation

**Verdict (preregistered 01:25, applied literally):** ghost opponents at
n512/d4 vs champion n256/d4 K1, 100 paired seeds `2027072400..99`, rev
`e252d68e`:

- Paired score delta **`+0.5450`**, 95% t-CI **`[+0.1823, +0.9077]`** —
  `ci_positive`. ✓
- Mean decision seconds `11.9383 -> 12.5289` = **`1.049x`** wall, inside
  the preregistered `<=1.25x` bound. ✓

**First CI+ wall-matched search improvement of the campaign.** Replacing
opponent-ply model evals with top-1 policy fast-forward (zero evals) and
reinvesting the reclaimed budget as 2x simulations buys ~half a point at
essentially equal wall. Both rule conditions hold => R1.2A graduates.
Artifact: `gate_ghost_wallmatched_20260712_verdict.{md,json}` on john0.

Per-sim cost from the timing block: baseline `11.9383/256 = 0.04663 s`,
ghost `12.5289/512 = 0.02447 s` — ghost fits `1.906x` sims at wall
parity.

## 2026-07-12 10:45 — PREREGISTERED: ghost n1024-tier confirmation (block 2027072600..99) — armed behind the coverage rerun

**Arms:** baseline = champion serving config n1024/d16 K1; candidate =
`--gumbel-ghost-opponents` n2048/d16 K1. CAND_N derivation (from the
gate's timing, per the 01:25 graduation clause): parity n = `1024 x
1.906 ≈ 1952`, rounded to the power-of-two `2048` — predicted wall
`~1.05x`, the same premium the completed gate measured. Both arms carry
`--gumbel-refresh-sample-divisor 4` (BASE_FLAGS — the serving default
adopted 05:50; the paired delta measures ghost under the config we would
actually serve). 100 paired seeds `2027072600..99` (registered
INFRASTRUCTURE §5). Rev `1c9211a5` (deployed 10:30 by the coverage
rerun; gameplay path unchanged from `e252d68e` — the diff is
replay-cap-only). VARIED_KEYS `ghost_opponents n_simulations`. Expected
wall ~13-17h total (both arms, jobs12).

**Rule:** paired 95% t-CI above zero AND candidate mean decision seconds
`<= 1.25x` baseline => ghost n2048/d16 becomes **champion-designate**:
promotion evidence is presented to John, who alone rules on champion
adoption (standing contract). CI ns => the ghost gain does not survive
high-budget pricing — R1.2A closes as a low-budget-only win (still
valuable for data generation and cheap serving); revisit via R1.2B/C.
CI- => ghost bias dominates at depth; cap R1.2 at Stage A.

**Launch:** `ghost_confirm_20260712.sh` waiter on john0 — waits for the
coverage-rerun chain (PID 3938586) to exit, then runs
`run_paired_gate.sh` with the env above. One scientific job at a time is
preserved by construction.

## 2026-07-12 10:55 — R1.3a coverage audit (valid rerun at rev 1c9211a5): drop rate 1.5% — R1.3 program stays OPEN

The replay-cap fix worked: **200/200 roots joined, 0 skipped** (the
07-12 first run at `e252d68e` skipped every seed and was logged
INVALID). Measurements (`menu_coverage_20260712_analysis.{json,md}`,
n1024/d8, median menu 256 capped vs 1258 full):

- Full-menu best action dropped by the greedy-256 cap: `3/200` =
  **`1.5%`** — ABOVE the preregistered `<1%` close bar.
- Mean regret overall: **`+0.0045`**/root (95% t-CI
  `[-0.0024, +0.0114]`), P95 `0.0000` — under the `<0.01` bar.
- Mean regret when dropped: `+0.3013` — misses are rare but material.

**Preregistered rule (01:02) applied literally: close required BOTH
bars; the drop-rate bar fails => R1.3a-c does NOT close.** Reading: the
greedy-256 cap is safe on average (expected ~0.0045/decision on the Q
scale) but has a thin tail — ~1 in 67 decisions silently loses the true
best action at ~0.30 Q each. Aggregated over ~83 decisions/game that
tail bounds a ~0.37 Q/game recoverable pool — same order as the ghost
gate's +0.545. R1.3b/c (smarter menu selection / cap raise where cheap)
stays a live modest-upside lane, priced by this measurement. The audit
is measurement only, never promotion evidence. Task R1.3a itself is
complete.

Chain: ghost n1024-tier confirmation launched 10:38 (waiter fired on
schedule); exporter PID 3980015; verdict expected ~13-17h.

## 2026-07-12 — PREREGISTERED METHODOLOGY (ruled by John): group-sequential paired gates

John ruled to adopt group-sequential early stopping for paired gates
("yes please incorporate that"). Design, fixed here before any gate uses
it:

- **Standard schedule:** interim looks at 40/60/80 pairs, final at 100
  (information fractions 0.4/0.6/0.8/1.0). Two-sided alpha `0.05`,
  Lan-DeMets **O'Brien-Fleming-like spending**; z boundaries for this
  schedule: `3.0992 / 2.5533 / 2.2538 / 2.0635` (computed by
  `sequential_boundaries.py`; final-look t-critical ~2.09 at df 99 vs
  fixed-N 1.98 — the whole cost of four looks).
- **Decision = repeated confidence interval** (mean ± t_k·SE, t_k from
  the look's boundary): superiority gates stop when the RCI excludes
  zero either side; noninferiority gates stop when the RCI is entirely
  above/below the margin. A straddling RCI ALWAYS continues; only the
  final look can return inconclusive. The final look spends all
  remaining alpha, so overall type-I error is exactly 0.05 (fixed-seed
  Monte Carlo check in tests: realized 0.05 ± 0.003).
- **Evidence rule:** for a preregistered sequential gate (planned final
  ≥100 pairs, looks fixed in the preregistration), the RCI at a stop IS
  promotion evidence; the naive 95% CI is reported for reference only.
  Interim looks executed by the runner at planned boundaries are not
  "reading a live arm" (AGENTS.md amended).
- **Machinery** (all tested, 38 new tests green; deploy AFTER the live
  ghost confirmation finishes): `sequential_boundaries.py` (spending +
  recursive boundary solver, validated against analytic identities and
  Monte Carlo), `merge_benchmark_reports.py` (chunk reports merged by
  re-running the real summarizers on concatenated rows — single-chunk
  merge is bit-exact), `sequential_gate.py` (RCI verdict wrapping
  `compare_search_shape`'s arm validation), `run_paired_gate.sh`
  `LOOKS`/`SEQ_RULE`/`SEQ_MARGIN` mode (alternating per-look chunks,
  resume-safe, HOLD-aware; default fixed-N path byte-identical in
  behavior).
- **Scope:** applies to gates preregistered AFTER this entry. The live
  ghost n1024-tier confirmation keeps its fixed-N 100-pair design.
  Retro-check on today's completed gates: both the ghost wall-matched
  gate and the refresh gate would have stopped at the 80-pair look
  (~20% GPU saved each); expected savings 20-60% per gate depending on
  effect strength.

## 2026-07-12 — R2.4 bridge throughput: investigation complete (memo + opt-in knobs + staged probe); PREREGISTERED probe plan

**Memo:** `docs/v3/BRIDGE_THROUGHPUT.md` — full request-lifecycle map with
line references. **Central structural finding: the serving pipeline is
strictly serial end-to-end** — the Rust aggregator blocks on each merged
192-row request and the Python serve loop is single-threaded, so the GPU
idles through every host phase (decode/collate/encode). That alone
explains the 63.8% mean util with dips to 2%. Also corrected stale
assumptions: COMPILE/TF32/bf16/bucketing/timing knobs and pinned
non-blocking H2D already existed; bf16 serving already ruled label-unsafe.

**Ranked levers:** (1) request pipelining/double-buffering — bit-identical,
realistic 1.2-1.4x, next build (R2.4b); (2) `CASCADIA_EVAL_CHUNK_ROWS=192`
(new knob) — today one merged 192-row request becomes >=6 serial 32-row
forwards; 1.1-1.3x, padding-drift class; (3) `CASCADIA_BRIDGE_COMPILE=1` +
`BUCKET=1` (reduce-overhead CUDA graphs) — 1.1-1.5x of forward time.

**Code (default-off, provenance-stamped):** chunk-rows knob, hardened
compile fallback (compile exceptions AND CUDA warmup failures fall back to
eager loudly), `bridge_env` block in the bridge hello payload, compile-mode
selection. 13 new tests; suite green (remaining local errors are this
Mac's missing-torch set, unchanged).

## 2026-07-12 — R2.4 lever #1 LANDED: request pipelining (both halves, default off); PREREGISTERED serial-vs-pipelined A/B

John directed implementing the ranked performance levers. Levers #2/#3
were already flag-complete; #1 (the serial-pipeline bound) is now built:

- **Rust (`CASCADIA_SHARED_INFLIGHT`, default 1 = bit-identical serial
  loop):** `SharedBridge` aggregator splits send/recv
  (`send_eval_batch_request` / `recv_eval_batch_response`), keeps up to
  K merged requests outstanding, gathers new work while the GPU
  computes, demuxes responses strictly FIFO, and on any response-stream
  failure fails ALL in-flight requests loudly (positional stream —
  never misattribute). Deadlock-free by construction: with responses
  outstanding it waits at most one gather window for new jobs before
  reaping. 63/63 exporter tests (3 new: pipelined demux under 8-worker
  load, single-worker no-deadlock, serial parity).
- **Python (`CASCADIA_BRIDGE_PIPELINE=1`, default off):** stdin reader
  thread + one-deep deferred finalize; request N+1 is decoded/collated
  while N's forward runs on the GPU; write(N) always precedes
  launch(N+1) so FIFO holds. Phase-split (`prepare/launch/finalize`)
  proven EXACTLY equal to `_model_eval_batch` on CPU torch (JSON +
  packed, multi-chunk, pairwise, quantile, ensemble — 14 new tests);
  serial path untouched. Provenance: `bridge_env.pipeline` in the hello
  payload; benchmark execution block records `shared_inflight` +
  `bridge_pipeline`.
- Measure-first micro-candidates (#6-#9) remain data-gated per the memo;
  #5 (second bridge process) is an alternative to #1, not additive —
  revisit only if the A/B disappoints.

**PREREGISTERED — pipelining A/B (engineering, armed behind the
throughput chain):** two 12-game runs at production topology
(n1024/d16, jobs12, div4, engineering seeds `1111120000..11`, identical
between arms): arm A serial, arm B `CASCADIA_SHARED_INFLIGHT=2` +
`CASCADIA_BRIDGE_PIPELINE=1`. Read: (a) per-seed mean scores compared
exactly; (b) wall + mean-decision ratio. Rules: **>=10% decision
throughput gain AND per-seed scores bit-identical => adopt pipelining
as the generation/serving default** (batteries/gates keep serial until
a follow-up battery-replay check); gain without bit-identity =>
classify the drift (expected class: gather regrouping, same as jobs
concurrency) and route through a sequential noninferiority gate before
any adoption; <10% gain => leave default-off, revisit after CHUNK_ROWS
lands (levers compose: pipelining hides host time that CHUNK_ROWS
shrinks). Never strength evidence.

**PREREGISTERED probe plan (engineering-only, never strength evidence):**
step 0 after the ghost confirmation ends — deploy, run one
`CASCADIA_BRIDGE_TIMING=1` production-shape sample for the phase split,
then `run_bridge_throughput_probe.sh` (refuses unless GPU idle; arms
eager / bucket / compile / compile+bucket at batches 8/32/96/192, TF32
off; reports rows/s + max-abs numerics diff vs eager). Adoption rules:
a knob goes to a score gate only if it delivers >=10% rows/s at batch
192; **compile is NOT bit-identical** (CPU smoke: ~2e-6 q drift), so any
adoption of compile/bucket/chunk-rows serving defaults requires a paired
noninferiority score gate — which is exactly what the new sequential
LOOKS mode makes cheap (expected ~3h instead of ~5h at n256/d4). Order:
CHUNK_ROWS first, then compile+bucket, then build R2.4b pipelining
(bit-identical: throughput A/B only, no score gate).

## 2026-07-13 00:15 — R1.2 ghost n1024-tier confirmation: INCONCLUSIVE — ghost gain does NOT survive high-budget pricing; R1.2A closes as a low-budget-only win

Verdict (`gate_ghost_n1024tier_20260712_verdict.{json,md}` on john0, 100
paired seeds `2027072600..99`, rev `1c9211a5`, fixed-N design per its
preregistration):

- Baseline (champion n1024/d16 K1 div4): mean seat score `98.2825`.
- Candidate (ghost n2048/d16 K1 div4): mean `98.2000`.
- **Paired delta `-0.0825`, 95% t-CI `[-0.3985, +0.2335]` —
  inconclusive.** Cost was fine: mean decision `35.15s -> 34.38s`
  (`0.978x`, well under the `1.25x` cap; the parity-n derivation was
  accurate — ghost n2048 prices almost exactly like champion n1024).

**Preregistered rule (10:45) applied literally: CI ns => the ghost gain
does not survive high-budget pricing — R1.2A closes as a low-budget-only
win.** No champion-designate; the champion remains cycle4 n1024/d16
(98.2975 under `..._rules_2026_07_09`).

Scientific reading: at n256-tier the reclaimed opponent budget bought
`+0.545` CI+; at n1024-tier doubling sims through ghosting bought
nothing (`-0.08` ns). This matches the R3.6 ceiling probe: the selfish
scaling curve is already decelerating at n1024→n4096 (+0.21 ns), so
converting opponent evals into MORE OWN SIMS reinvests into a saturated
axis. Ghost's value is real but budget-local — it moves points only
where sims are scarce. Consequences:

- **R1.2A standing use:** data generation and cheap/fast serving tiers
  (n256-class), where it is CI+ at ~1x wall. Not a champion lever.
- **R1.2B/C (revisit per the rule):** reinvesting reclaimed budget into
  something OTHER than sim count — deeper determinization (d16→d32),
  wider top-m, or R1.3b menu-cap relief — is the surviving hypothesis
  shape: spend reclaimed budget on axes that are NOT saturated. Any such
  arm gets its own preregistration and, now, a SEQUENTIAL gate design.
- Retro-check: a sequential design would NOT have saved GPU here — the
  RCI straddles zero at every look; inconclusive only resolves at the
  final look. (Sequential saves on decided gates, not null ones.)

Night queue continues automatically: the throughput-chain waiter (PID
4016261) fires next (deploy `d6cae30b` -> probe -> TIMING sample), then
the pipelining A/B (rev `c2e75cab`).

## 2026-07-13 00:18 — throughput chain: probe FAILED at init (no GPU work lost); fixed + re-armed behind the pipelining A/B

The bridge throughput probe died 9s after deploy: `run_bridge_throughput_probe.sh`
line 32 stamps provenance via `git rev-parse`, but john0 deploys are
tarball extracts with no `.git`, and the chain script did not pass
`SOURCE_REVISION` — `set -e` killed the script at variable init, before
root export or any model work. The chain correctly proceeded to the
`CASCADIA_BRIDGE_TIMING=1` production-topology sample (running, GPU
busy).

Fixes:
- `run_bridge_throughput_probe.sh` now falls back to
  `cascadiav3/logs/exact_k1_deployed_revision.txt` when git is absent
  (committed).
- `probe_rerun_20260713.sh` armed on john0 (waiter PID 4040310, monitor
  live): waits for the pipelining A/B chain to exit, then runs the probe
  with `SOURCE_REVISION` passed explicitly from the deploy marker — the
  tree will be at `c2e75cab` by then, which contains everything the
  probe needs. Night queue is now: TIMING sample (running) -> pipelining
  A/B -> throughput probe. Preregistered reads unchanged.

## 2026-07-13 00:35 — PREREGISTERED: R1.2B ghost+d32 sequential gate (block 2027072700..99) — the FIRST live group-sequential gate; armed behind the probe re-run

**Hypothesis (the ns branch's surviving shape, executed):** ghost's
reclaimed opponent-eval budget is real (~50% of champion-tier eval cost,
measured twice) but reinvesting it into MORE OWN SIMS is worthless (00:15
ns; sims axis saturated per R3.6). Reinvest it instead into **more
determinization worlds** — the knob that attacks evaluation noise, the
campaign's binding constraint, and the one knob with a CI+ screen
(det8>det4 at n256: `+0.4225`, CI `[+0.1045,+0.7405]`). The det16→det32
n1024 confirmation was paused by John's 07-10 21:10 ruling BECAUSE it
was not wall-matched (~1.5x dedup tax on every future gate). Ghost pays
that tax: det32/n1024 smoke measured `1.204x` champion wall (42.32s
07-10) and ghost's per-eval factor is ~0.49 (00:14 gate) — the composed
arm is predicted **≤0.8x champion wall**, i.e. wall-FAVORABLE where the
paused design was wall-adverse. This is a new design on a fresh block;
block `2027071600..99` stays reserved for the paused pure-worlds
confirmation, unchanged.

**Arms:** baseline = champion n1024/d16 K1 div4 (identical to the 00:14
gate's baseline). Candidate = **ghost n1024/d32** K1 div4
(`--gumbel-ghost-opponents`, `--gumbel-determinizations 32`, same
n_simulations — worlds cycle inside the fixed sim budget; per-world
visits halve 64→32, exactly the trade the n256 screen validated).
VARIED_KEYS `ghost_opponents determinizations`. Rev `c2e75cab` (will
already be deployed + built by the pipelining A/B chain).

**Design — first live GROUP-SEQUENTIAL gate (methodology entry 07-12):**
`LOOKS="40 60 80 100"`, superiority rule, alpha `0.05`, Lan-DeMets
O'Brien-Fleming spending (z boundaries `3.0992/2.5533/2.2538/2.0635`).
Seeds `2027072700..99` (registered). The RCI at a stop is the verdict.

**Rule (applied literally on completion):**
- **Stop/final POSITIVE and candidate mean decision seconds `<= 1.05x`
  baseline** => ghost n1024/d32 is **champion-designate**: evidence goes
  to John, who alone rules on promotion.
- Positive at `> 1.05x` wall => no designation; present to John with the
  wall tradeoff quantified (his call entirely).
- Final INCONCLUSIVE => **R1.2B closes**; the R1.2 program ends with
  ghost as a data-generation/cheap-serving tool only (any R1.2C
  reinvestment needs a fresh screened case).
- Stop/final NEGATIVE => same closure, plus: ghost bias composes badly
  with world-thinning — record as a constraint on future world knobs.

**Launch:** `ghost_d32_gate_20260713.sh` waiter on john0 — waits for the
probe re-run chain (PID 4040310) to exit, verifies the deploy marker is
`c2e75cab`, then runs `run_paired_gate.sh` in LOOKS mode. One scientific
job at a time holds by construction (the engineering queue is strictly
ahead of it). Expected wall: baseline chunks ~24.6ks + candidate chunks
~15-18ks if it runs to 100 pairs (~11-12h); an interim stop saves
20-60%.

## 2026-07-13 01:20 — R2.4 TIMING phase split measured (12-game production sample at d6cae30b): forward = 84% of bridge time but only ~55% of wall; chunks average ~30 rows

`bridge_timing_sample_20260713` (n1024/d16 div4 K1, jobs12, engineering
seeds `1111110000..11`, mean seat `98.42` — engineering datum only).
Bridge-side accumulator (`gumbel_batch_0.stderr.log`, final): chunks
`97645`, rows `2,971,626`, collate `153.6s` / h2d `144.0s` / forward
`1994.6s` / d2h `51.0s` / encode `18.9s`, total in-bridge `2362.0s`,
`1258 rows/s`.

Reads (descriptive; adoption rules unchanged):
- **Forward is 84.4% of bridge time but the bridge is busy only ~65% of
  the ~3621s wall — the GPU computes forwards ~55% of wall.** The other
  ~45%: ~10% bridge host phases (collate+h2d+d2h+encode — the
  PIPELINE=1 target) and ~35% outside the bridge (Rust search between
  requests, 2ms gather windows, JSONL transport — the INFLIGHT=2
  target). Both halves of lever #1 aim at exactly the measured gap.
- **Mean forward batch is ~30 rows** (2.97M rows / 97.6k chunks) at
  `1258 rows/s` — the eval-chunk fragmentation lever #2
  (`CASCADIA_EVAL_CHUNK_ROWS=192`) addresses; the probe re-run will put
  a rows/s number on batch 192 vs ~30.
- Ceiling arithmetic: if pipelining hid ALL non-forward time, wall would
  drop ~45% (1.8x); realistic one-deep pipelining bounds at the A/B.

## 2026-07-13 03:20 — R2.4 lever #1 A/B verdict: BIT-IDENTICAL but +4.2% — below the 10% bar; stays default-off per rule

`pipeline_ab_20260713_verdict.md` (12 games n1024/d16 div4 K1 jobs12,
identical engineering seeds `1111120000..11`, rev `c2e75cab`):

- **Per-seed seat scores bit-identical across all 12 seeds** — the
  FIFO demux + one-deep deferred finalize is exactness-safe in
  production on CUDA, not just in the CPU-torch proof.
- Mean decision seconds `37.47 -> 35.98` (**1.042x**); whole-arm wall
  `3672.5s -> 3548.2s` (**1.035x**).

**Preregistered rule applied: <10% gain => pipelining stays default-off;
revisit after CHUNK_ROWS.** Why the 45% non-forward gap yielded only 4%:
one-deep pipelining can only overlap the bridge-host phases (~10% of
wall) with forwards, and only when a second merged request is actually
waiting — but each Rust worker blocks on its own result (data
dependency: the search cannot proceed without the evals), so the
aggregator rarely has a second full batch ready inside the gather
window. The ~35% outside-bridge time is Rust search compute, which no
bridge-side pipelining can hide. Consequence for the memo's ranking:
the realistic lever order is CHUNK_ROWS (kills per-chunk overhead on
~30-row chunks) and compile (shrinks forward itself); pipelining's
value, if any, returns only after those shift the balance — exactly
the revisit clause. Positive side-product: the pipelining machinery is
validated bit-exact end-to-end, so a future revisit is a pure config
flip, no new risk.

## 2026-07-13 03:25 — R2.4 CLOSES: probe measured (all arms); every micro-lever below its preregistered bar; serving stack is within ~5% of architectural ceiling

`bridge_throughput_probe.{json,md}` (rev `c2e75cab`, TF32 off,
serving-realistic menus, batches 8/32/96/192):

- **Eager forward saturates by batch 32**: `250.8 -> 346.5 -> 358.5 ->
  359.9 rows/s` (b8/32/96/192). Production runs at ~30-row mean chunks
  (01:20 sample), i.e. already ~96% of the b192 rate — **CHUNK_ROWS
  upper bound is +3.9%, below its >=10% bar. Not advanced to a gate.**
- **compile (reduce-overhead)**: `1.005x` at b192 — nothing to buy.
  (Datum: bit-identical vs eager on this stack, contradicting the CPU
  smoke's ~2e-6 drift — CUDA-graph capture of the same kernels. Recorded
  for any future revisit; irrelevant now given no gain.)
- **bucket / compile_bucket**: NEGATIVE (`0.93-0.96x`) and NOT
  bit-identical (~3e-5 q drift) — dead on both counts.

**R2.4 verdict across the whole program:** pipelining +4.2%
(bit-identical, below bar), CHUNK_ROWS +3.9% bound, compile +0.5%,
bucket negative. The 01:20 phase split showed forwards at ~55% of wall;
the probe shows the forward itself is batch-saturated and the bridge
host phases are already thin. The remaining ~45% is Rust-side search
compute between eval requests — outside the bridge program's scope, and
the earlier jobs concurrency calibration already set jobs12 as the
operating point. **R2.4 closes: serving throughput is within ~5% of
what this architecture yields.** All knobs stay landed and default-off
(zero-risk revisit if the model or topology changes: bigger model =>
forward share rises => pipelining/CHUNK_ROWS re-price). GPU hours go to
scientific gates, starting with the R1.2B ghost+d32 sequential gate
(fires next in the queue).

## 2026-07-13 15:55 — R1.2B ghost+d32 sequential gate: FINAL_INCONCLUSIVE — R1.2B closes per rule; BUT candidate is 1.68x faster at floor -0.09 (speed-default candidate)

First live group-sequential gate completed
(`gate_ghost_d32_seq_20260713_verdict.md`, 100 pairs `2027072700..99`,
rev `c2e75cab`, looks 40/60/80/100 all executed by the runner —
continue/continue/continue/final):

- Champion n1024/d16: mean `98.2975` (identical to its canonical
  battery mean — a clean replication on a fresh block). Ghost n1024/d32:
  `98.4750`. Paired delta **`+0.1775`**, repeated CI (final boundary z
  `2.0635`) **`[-0.0935, +0.4485]`** — straddles zero.
  **FINAL_INCONCLUSIVE. Preregistered rule applied: R1.2B CLOSES; the
  R1.2 program ends** with ghost as a data-generation/cheap-serving
  tool. No champion-designate.
- **Cost: candidate mean decision `21.09s` vs baseline `35.46s` =
  `0.595x` wall (1.68x faster)** — the ≤0.8x prediction held with room;
  whole-arm `16.7ks` vs `28.2ks`.

**Honest post-hoc observation (NOT evidence, motivates a new
preregistration):** under the speed-knob noninferiority template
(margin `-0.25`, the refresh-div4 standard) this result would have been
a clear pass — RCI floor `-0.0935` is far above the margin, with a
1.68x speedup that would cut EVERY future gate's arm cost ~40% and
re-price all data generation. The superiority rule asked "is d32+ghost
BETTER"; the answer is "not provably — but it is provably-as-good at
0.6x price" is exactly the adoption class of exact-K1 and refresh-div4.
Per methodology, that claim needs its own preregistered noninferiority
gate on a fresh block (hypothesis from this data, tested on new data) —
preregistered next entry. Retro-note on sequential value: a
noninferiority-rule gate on this data would have stopped at the 80-pair
look (floor -0.16 > -0.25 at z 2.2538); the superiority question could
never stop early because the truth sits near zero — both behaviors are
the design working as intended.

## 2026-07-13 16:00 — PREREGISTERED: ghost+d32 SPEED-DEFAULT noninferiority gate (block 2027072800..99) — launching now (GPU idle)

**Question (new, motivated by the 15:55 closure's cost column, tested on
fresh data):** is ghost n1024/d32 score-noninferior to the champion
serving config, so that its measured `~1.68x` speedup can be adopted as
the serving/gate-arm speed default — the exact-K1 / refresh-div4
adoption class? This is a SPEED claim, not a strength claim; champion
promotion remains John's alone and is not at stake here.

**Arms (identical configs to the closed 07-13 gate, fresh block):**
baseline = champion n1024/d16 K1 div4; candidate = ghost n1024/d32 K1
div4 (`--gumbel-ghost-opponents`, CAND_DET=32). Seeds `2027072800..99`
(registered). Rev `c2e75cab`. VARIED_KEYS
`ghost_opponents determinizations`.

**Design:** group-sequential NONINFERIORITY, margin `-0.25` (the
standing speed-knob standard), looks `40/60/80/100`, alpha `0.05`, OBF
spending. The RCI at a stop is the evidence.

**Rule (applied literally):**
- Stop/final NONINFERIOR (RCI floor above `-0.25`) AND candidate mean
  decision seconds `<= 0.8x` baseline => **ghost+d32 is ADOPTED as the
  serving/benchmark/gate-arm default** (third adopted speed default
  after exact-K1 and refresh-div4). Consequences on adoption: future
  gate arms carry `--gumbel-ghost-opponents --gumbel-determinizations
  32` in BASE_FLAGS/CAND defaults (~40% cheaper arms); data generation
  re-prices; the champion's canonical score reference stays the
  existing battery until a fresh canonical battery is run under the new
  default.
- INFERIOR (RCI entirely below `-0.25`) or final inconclusive-below =>
  no adoption; serving default unchanged; ghost stays data-gen only.
- Wall condition failing (>0.8x — not expected; two independent
  measurements say ~0.6x) => no adoption, investigate the discrepancy.

**Expected cost/behavior:** if the true delta is ~+0.18 (the closed
gate's point estimate), the RCI floor clears the margin at the 60- or
80-pair look — expected stop ~7-10h. A stop_noninferior at look ≤3 is
also the first live demonstration of sequential early stopping.

**Launch:** direct (GPU idle, no waiter): `ghost_d32_noninf_20260713.sh`
on john0, nohup + pid + monitor.

## 2026-07-13 16:30 — RULED BY JOHN: research queue realigned to maximize break-100 probability

John ruled ("i trust your judgement, align the research queue as you see
fit to maximize our chance of breaking past 100"). The aligned queue,
with rationale grounded in the 13 verdicts to date (all serving-side
estimation ideas null => the noise wall must be attacked at its sources:
the value function's training signal, and unseen own-turn depth; plus
velocity multipliers that compound):

1. **R3.2 deep own-turn planning (kill test now):** depth-2 was closed
   flat at 1.8x cost when opponents ate 3/4 evals; ghost+d32 repriced it
   (~0.6x wall base). Screen behind the live noninferiority gate
   (~15 min GPU, no registered seeds): candidate n256/d4 + ghost +
   depth2 vs incumbent bank regret, proceed bar `<= +0.020` (bias-only
   read, budget buyback is the gate's job — the ghost-screen template).
   Screen pass => preregistered sequential gate on fresh block
   `2027072900..99` (designed after tonight's adoption verdict fixes
   the baseline config).
2. **R1.4 training densification (top expected value, build starts
   now):** design doc being drafted from the actual trainer/exporter
   code; staged kill tests before any full retraining cycle.
3. **R2.3 CUPED verdicts (velocity, no GPU, building now):** opt-in
   covariate adjustment (covariate fixed = baseline per-seed seat
   score) in the sequential-gate machinery; expected 10-30% CI
   shrinkage on top of sequential stopping and ghost-priced arms.
4. **R0.5/R3.4 adaptive budgets** (puzzle bank supervises for free) and
   **R1.3b menu relief** (priced ~0.37/game) — queued behind 1-3.
5. **R1.1c/R3.1 cooperative table values** — the only single idea whose
   ceiling covers the whole remaining gap; sequenced after R1.4's
   training infrastructure exists.

Deprioritized: R3.5 smarter worlds (family returns small effects; d32
ns), R0.7/R0.8 (the root-estimation class is 0-for-4). Scope note:
CUPED applies to gates preregistered after its own methodology entry
(to be written when the code lands + tests green).

## 2026-07-13 17:10 — PREREGISTERED METHODOLOGY: CUPED variance reduction for sequential gates (R2.3); code landed + tested

Landed (29/29 sequential tests green, 7 new; full suite at the known
missing-torch baseline): `sequential_gate.py --cuped` /
`run_paired_gate.sh SEQ_CUPED=1`.

- **Covariate is FIXED by this preregistration** — the baseline arm's
  per-seed seat score (now carried in `paired_score_deltas` rows). No
  per-gate covariate shopping, ever. theta = cov(delta, x)/var(x),
  re-estimated at each look on the accumulated pairs (standard
  group-sequential ANCOVA practice); residual variance uses df = n-2, so
  a useless covariate costs exactly one degree of freedom; a constant
  covariate falls back to the unadjusted estimator, flagged.
- The point estimate is UNTOUCHED (the correction is mean-centered) —
  CUPED only narrows the interval. The RCI (and thus the stopping
  decision) uses the adjusted SE; the naive CI stays raw,
  reference-only. Verdict JSON/md carry the full cuped block (theta, r,
  raw vs adjusted SE, variance-reduction fraction).
- **Scope: gates preregistered AFTER this entry** may set SEQ_CUPED=1 in
  their preregistration. The live ghost+d32 noninferiority gate keeps
  its no-CUPED design. Expected effect: paired seat-score deltas
  correlate with baseline seed difficulty; even r=0.3-0.5 buys 10-25%
  variance reduction, compounding with sequential stopping and
  ghost-priced arms.

## 2026-07-13 17:20 — PREREGISTERED: R3.2 depth-2 own-turn screen (bank, no registered seeds) — armed behind the live noninferiority gate

**Hypothesis:** at `depth_rounds=1` the search never sees our second
move; commitment decisions are where lookahead should pay. Depth-2 was
closed flat at 1.8x cost in the 4-evals/ply era; ghost+d32 repriced
own-turn depth (~0.6x wall base), so the frontier question reopens
exactly as the portfolio anticipated ("retest after R1.2").

**Design (screen; measurement only, never promotion evidence):**
`run_bank_screen.sh` with `SCREEN_NAME=ghost_depth2`,
`EXTRA_FLAGS="--gumbel-ghost-opponents --gumbel-depth-rounds 2"` (the
exporter's arg parser is last-wins, so the depth override composes with
the script's fixed n256/d4 candidate shape), rev `c2e75cab`, vs the
frozen n4096 bank (700 roots).

**Rule:** proceed to a preregistered champion-tier sequential gate iff
the screen's bank-regret penalty vs the incumbent screen
(`puzzle_screen_20260711_incumbent`, regret `0.2351`) is `<= +0.020` —
the ghost-screen template: the screen reads bias/allocation sanity at
equal n; the gate buys the depth cost back with ghost's reclaimed
budget. Fail => R3.2 closes cheaply (the kill test's purpose). The gate
(if earned) gets its own preregistration on block `2027072900..99`
AFTER tonight's speed-default verdict fixes the baseline config, and
will be the first SEQ_CUPED=1 gate.

**Launch:** `ghost_depth2_screen_20260713.sh` waiter on john0, armed on
the noninferiority gate's pid (4110505); ~10-15 min GPU when it fires.

## 2026-07-13 17:50 — R1.4 design complete (docs/v3/R1_4_DENSIFICATION_DESIGN.md); hypothesis revised by code audit; Stage 0 build starts

Design memo landed. Central corrections to the R1.4 brief (code-cited):
policy targets are ALREADY full-menu soft distributions and Q regression
ALREADY covers all visited actions with SE-confidence weighting — the
KataGo-dense share assumed available is smaller than the portfolio
priced. What genuinely remains sparse: (2.1) value/score/rank labels =
one Monte-Carlo outcome per position; (2.2) Q labels carry
generation-grade (n256-512) search noise vs n1024/d16 serving; (2.3)
improved-policy information covers only top_m≈16 of ≤256 retained
(unvisited mass is self-distillation); (2.4) zero spatial targets;
(2.5) packed v4 shards cannot express trajectory linkage. **Cheapest
lever found: `search_root_value` is exported, contract-required,
collated into every batch (torch_train_cascadiaformer.py:319-321), and
consumed by NO loss** — a low-noise value target sitting unread (V1).

Ranked menu: V1 (search-value target, S, zero regen) > P1 (label-side
q-bias correction at generation) > V2 (distributional value head) > D1
(targeted hard-root reanalyze) > T1 (v5 trajectory fields) > O1
(per-hex ownership, L). Structured-Q stays CLOSED (failed pilot 07-10;
the design forbids relitigating it). Staged kill plan: Stage 0 zero-GPU
label-noise audit with the preregistered V1 continuation bar
(search_root_value must cut value-target RMSE >=20% vs raw outcome at
|bias| <= 0.5) -> data-free retrains -> gates; whole qualification
bundle <= ~2 GPU-days before touching any corpus. Portfolio kill rule:
if Stages 1-3 move nothing, EI saturation survives its strongest
challenge and training-side work stops being resourced.

Stage 0 analyzer build starts now (CPU-side; runs parallel to gates).

## 2026-07-13 23:25 — ghost+d32 speed-default gate: STOP_NONINFERIOR at 60 pairs — ADOPTED (third speed default); first live sequential early stop

`gate_ghost_d32_noninf_20260713_verdict.md` (fresh block 2027072800..59
used, 60 of 100 planned pairs, rev `c2e75cab`):

- **STOP_NONINFERIOR at look 2/4**: paired delta `+0.3333`, RCI at z
  `2.5533` = `[-0.2122, +0.8788]` — floor above the `-0.25` margin.
  Point estimate positive for the third consecutive fresh block
  (+0.545 n256, +0.178 n1024 superiority, +0.333 here) — ghost+d32 is
  plausibly slightly better, provably not meaningfully worse.
- **Wall `0.688x`** (23.73s vs 34.48s/decision) — under the `<= 0.8x`
  adoption condition.

**Preregistered rule applied: ghost n1024/d32 (`--gumbel-ghost-opponents
--gumbel-determinizations 32`, with K1 + div4) is ADOPTED as the
serving/benchmark/gate-arm speed default** — the third adopted speed
default after exact-K1 (07-10) and refresh-div4 (07-12). NOT a strength
claim; the champion's canonical reference remains the cycle4 n1024/d16
battery (98.2975) until a fresh canonical battery runs under the new
default; champion promotion remains John's alone.

Consequences, effective for work preregistered after this entry:
- Gate arms default to the ghost+d32 config on BOTH sides (a candidate
  varies its own knobs on top). ~31% cheaper arms; with sequential
  stopping (this gate: stopped at 60/100, ~40% saved — the FIRST live
  early stop) and CUPED, gate cost is now ~3-4x cheaper than the
  fixed-N 07-12 baseline.
- Data generation re-prices at ~0.69x per decision at n1024-tier.
- Ghost-generated labels as TRAINING teachers remain UNVALIDATED
  (R1_4 design §8) — generation for corpora keeps the non-ghost config
  until the safety fold clears.
- Seeds 2027072860..99 of the block were never touched (early stop) —
  they stay burned with the block per registry discipline (blocks are
  touched once).

Next in queue (automatic): depth-2 screen (running 23:21) -> deploy
6cc01ab5 (CUPED + Stage 0 analyzer) -> Stage 0 label audit (CPU) ->
R3.2 gate if screen passes, on the NEW baseline (ghost n1024/d32 vs
same + depth2, VARIED_KEYS=depth_rounds, SEQ_CUPED=1, block
2027072900..99).

## 2026-07-13 23:30 — R3.2 depth-2 own-turn: screen FAILS the bar (+0.0586 vs <=+0.020) — R3.2 CLOSES at kill-test cost

`puzzle_screen_ghost_depth2_analysis.md` (700/700 roots joined, rev
`6cc01ab5`-era tree at c2e75cab, screen ran 23:21-23:26): mean bank
regret **`+0.2937`** (CI `[+0.2664, +0.3210]`) vs incumbent `0.2351` —
penalty **`+0.0586`**, ~3x over the preregistered `<= +0.020` proceed
bar. Chose-bank-best `29.1%`.

**Rule applied: R3.2 closes.** At equal simulation count, diverting
sims to animate our own second turn costs the ROOT decision more than
the lookahead returns — the same shape as the legacy depth2-flat
result, now confirmed under ghost pricing. The gate never launches;
registered block `2027072900..99` is RELEASED (never touched).

Honest caveat for any future revisit: the screen runs at n256/d4 where
per-candidate visits are scarce; a depth-2 case at n1024-tier budgets
would need a fresh preregistered screen making that argument
explicitly. Nothing in tonight's data motivates it — the campaign's
own-turn-depth lane is done. Portfolio consequence: Tier-3 search
reformulations narrow to R3.3 exactness expansion; the queue's center
of gravity moves fully to R1.4 (Stage 0 running, PID 4149025: cycle4
raw + top64 passes).

## 2026-07-13 23:45 — R1.4 Stage 0 verdict: V1 bar FAILS (closes) but late-game mechanism confirmed => V1b preregistered; adjacency CONFIRMED (T1(i) drops to S-cost); canonical battery launched

**Stage 0 results** (`r1_4_stage0_cycle4_raw.{json,md}`, 100,000 records
= the full cycle4 corpus, phase = tile-count proxy, 0 unknown-phase):

- **V1 preregistered bar: FAIL — V1 closes without a retrain.** Overall
  RMSE(outcome, search_root_value) `4.745` vs within-phase baseline
  `2.711` (-75% "reduction"); bias `-2.861`. BUT the preregistered
  phase-stratified read shows the mechanism is REAL and phase-gated:
  bias is monotone (opening `-7.29` -> endgame `+0.51`) and srv BEATS
  the noise baseline in the late game (late_mid RMSE `2.45` < `2.71`;
  endgame `1.46` = **46% better than the 1-sample noise floor**).
  Search values are total-score-calibrated but ~7 points pessimistic at
  the opening, converging by endgame — a generation-model calibration
  drift, not noise.
- **Density census:** q-valid fraction locked at `0.0626` (=16/256
  always); improved-policy mass on unvisited actions mean `0.3295`
  (median 0 — strongly bimodal: most roots ~0, a heavy tail near-total).
  P1 (label-side bias correction) strengthens.
- **Hard-root census:** `54.6%` of corpus roots are noise-flippable
  (top1-top2 gap < pairwise SE) — worse than the 46% serving estimate;
  even endgame is `49%`. D1's targeting pool is enormous.
- **Trajectory adjacency: CONFIRMED** — 1,249 identical-final-score
  runs of mean length 80.06 (=1,250 games x 80 plies, contiguous).
  **Packed record order preserves whole games => path-consistency
  targets need NO v5 schema.** T1(i) reprices from M to S.

**PREREGISTERED — Stage 1 arms** (data-free retrains on the champion
recipe, one variable each; offline bar per design §5: locked-val value
RMSE -10% without q-regret degradation >0.05; then bank screen; then
n256-tier sequential CUPED gate for arms clearing both):
- **V1b (new, from Stage 0 evidence):** value target mixes
  search_root_value ONLY where it beats the noise floor — lambda by
  phase: 0 for tile_count<13, 0.5 for >=13 (late_mid+endgame). Trainer
  flag, default-off, bit-identical when unset.
- **V2:** K-quantile distributional value head (pinball loss, distq
  template).
- **C1:** aux weight sweep (score/rank/uncertainty x4).
- **T0 (new, unlocked by adjacency):** path-consistency prototype —
  L2 between value(t) and stop-gradient value at the same seat's next
  root, adjacency-derived, weight 0.1. Zero schema work.
Trainer changes build now (flags default-off, R0.x pattern);
retrains chain behind the canonical battery.

**GPU (23:45): canonical battery launched** (PID 4150464): adopted
default ghost n1024/d32 K1 div4 on the rebaseline battery block
`2027070900..0999` (same seeds as the champion's canonical 98.2975;
battery blocks host all reference arms by precedent — gate blocks stay
touch-once). Purpose: the new canonical score reference + fresh
decision ledgers at the serving default (feeds R0.5/R1.3/D1). ~4.5h.
Descriptive only, never promotion evidence.

## 2026-07-14 00:05 — PREREGISTERED: the next-24h GPU slate (primary chain + three independent fillers; no idle windows)

Ordered queue (each item independently preregistered so the GPU never
waits on code or a verdict):

1. **Canonical battery** (running, PID 4150464, ~04:15 done).
2. **R1.3b menu-widening screen** (immediate post-battery filler, ~6-10
   min): bank screen `SCREEN_NAME=menu512`,
   `EXTRA_FLAGS="--max-actions 512"` (menu-construction cap; screens
   run at 64, serving at 256 — the screen reads whether widening buys
   regret at all). **Bar: regret delta vs incumbent `<= -0.010`**
   (strength-knob template) => preregister a champion-tier sequential
   CUPED gate (ghost+d32 baseline vs same + `--max-actions 512`) on a
   fresh block; else R1.3b closes (R1.3c cap-raise economics may still
   be revisited under D1 evidence).
3. **R1.4 Stage 1 retrains x4** (V1b, V2, C1, T0; ~3h each) as soon as
   the trainer flags land + verify — each followed by locked-val eval
   and a bank screen (~35 min). Offline bar per the 23:45 entry.
4. **D1 label-movement pilot** (fits any gap; ~4-5h): resolve a
   stride-sampled ~700-root subset of TONIGHT'S canonical ghost+d32
   ledger at n2048/d16x2 via the puzzle-bank machinery (fresh bank dir,
   never touching the frozen 20260711 bank), then a label-movement
   analysis vs the ledger's serving-time choices. **Bar for D1
   continuation: mega-budget argmax differs from the serving choice on
   >= 20% of measured-hard roots** (if mega labels agree with cheap
   ones, D1's premise dies and 30-60h is saved). Analyzer is CPU, built
   in parallel.
5. **Survivor gates** (any Stage-1 arm clearing offline bar + screen):
   n256-tier sequential CUPED gates on fresh registered blocks —
   preregistered individually at launch time.
6. **Ghost-label safety-fold corpus** (backstop filler, ~2h): generate a
   ~20k-root ghosted corpus (adopted default shape) for the R1_4 §8
   teacher-safety question; the fold retrain + locked-val read is a
   later Stage-1-class item. Generation only; labels quarantined until
   the fold clears.

Estimated GPU occupancy: 4.5h (battery) + 0.2h (menu512) + 12h
(retrains) + 2h (screens) + 4.5h (D1 pilot) + gates ≈ 25-28h of queued
work; fillers 4/6 are independent of the Stage-1 code path, so a slip
anywhere cannot idle the GPU.

## 2026-07-14 00:20 — Stage 1 trainer flags LANDED (55e8d4c1, bitwise-identical defaults); retrain chain ARMED; RESEARCH_AGENDA.md created

- **Trainer flags committed** (`55e8d4c1`): V1b
  (`--value-target-search-mix`, phase-gated at tile_count>=13, active
  seat only), V2 (`--value-quantiles K`, pinball, quantile-mean scalar
  keeps every downstream consumer contract), C1 (weight aliases of the
  pre-existing `--*-loss-weight` flags), T0
  (`--path-consistency-weight`; target-side variant: value(t) vs
  stop-grad collated `search_root_value` at t+4 same-game/same-seat —
  exactly the design memo's T1(i) formulation; pairing precomputed per
  shard in numpy, epoch shuffle safe). Defaults proven BITWISE
  identical end-to-end (pristine-HEAD vs new source: identical
  checkpoints, all 45 tensors; golden loss regression; torch suite
  353/OK, torch-free suite at the 15-error baseline).
- **Bar note (recorded):** a V1b arm's `locked_val_value` is measured
  against MIXED targets by construction — the preregistered "value RMSE
  -10%" comparison will be computed via an unmixed re-eval of the V1b
  checkpoint, not read off its training metrics.
- **Retrain chain armed** (`stage1_retrains_20260714.sh`, waiter PID
  4153213, behind the menu512 screen): deploys `55e8d4c1` (python-only,
  binary unchanged from c2e75cab), then per arm (v1b, v2q8, c1x4, t0pc)
  trains the manifest-recovered champion recipe (3 relation_tail
  corpora at `--train-source-weights 4,2,1`, 2500 steps, batch 192, lr
  1e-4, wd 0.05, warmup 0.02, seed 20260630, eval-every 250,
  val-max-batches 8, selection locked_val_final_q_regret min, SWA 0.20,
  warm start incumbent) + a puzzle-bank screen of its best_locked_val
  checkpoint. ~3.5h/arm; chain ~14h.
- **RESEARCH_AGENDA.md** (976e7b0b): living prioritized
  queue/scoreboard at docs/v3/, linked from the root README and the v3
  source-of-truth README. Stale session monitors cleaned (8 stopped; 3
  live: battery, menu512, stage1).

## 2026-07-14 04:05 — Canonical battery COMPLETE: adopted ghost+d32 default reads 98.3925 on the rebaseline block (descriptive)

- `rules_20260713_cycle4_ghost_d32_canonical` (100 games, seeds
  `2027070900..0999`, rev `6cc01ab5`, adopted serving default: ghost
  opponents, d32, exact-K1, refresh-div4, n1024): **mean seat
  `98.3925`**, P90 `102.0`, seats >=100: `129/400`, games with mean
  >=100: `10/100`. Per-seat means 98.34/98.48/98.66/98.09 — flat, no
  seat artifact. Decision seconds p50 `17.39` / p95 `77.84`; battery
  wall 4h13m (23:45->03:58).
- Same block under the pre-ghost champion config read `98.2975`
  (07-13 replication). Descriptive delta `+0.095` — consistent with
  the noninferiority gate's point estimate (+0.333, RCI straddling
  zero). **Reference number only, never evidence**; the canonical
  CHAMPION score remains 98.2975 at the champion config, and champion
  identity is unchanged (promotion is John's ruling).
- Fresh serving-default ledgers now exist for downstream supervision:
  `..._decisions.jsonl` (8,000 decisions, 4.2 MB) and
  `..._games.jsonl` — these are the D1 pilot's substrate (queue #4)
  and future R0.5/R3.4 adaptive-budget supervision.
- Chain advanced automatically: menu512 bank screen started 03:58
  (queue #2, ~10 min), Stage 1 retrain chain armed behind it.

## 2026-07-14 04:30 — menu512 screen VOID (instrument no-op, wrong flag); R1.3b goes to a direct gate; Stage 1 chain lost to a fatal env var, root-caused, relaunched; D1 pilot preregistered and chained

- **menu512 screen verdict: VOID, not a close.** Mean regret `+0.235148`
  / chose-best `0.3486` — BIT-identical to the incumbent screen. Two
  independent reasons the instrument measured nothing: (1) bank screens
  replay the ledger's STORED action menus (`analyze_puzzle_screen`
  hard-errors on menu drift, none occurred), so no flag can widen them;
  (2) the serving menu cap is `--gumbel-root-menu` (main.rs:417, default
  256) — `--max-actions` (the flag the screen varied) only governs
  greedy/rollout ranking. The preregistered close rule cannot be applied
  to a measurement that never engaged the candidate. Recorded
  instrument limitation: **bank screens cannot rank menu-widening
  candidates**; the same bit-identity retroactively explains the R0.3
  q-bias serving screen (also `0.235148`).
- **R1.3b straight to a gate** (per "screens rank, gates decide": the
  R1.3a audit is the ranking evidence — priced tail +0.37/game ceiling,
  1.5% of decisions drop the true best at +0.30 each). PREREGISTERED:
  champion-tier sequential CUPED SUPERIORITY gate on fresh registered
  block `2027073000..3099` (INFRASTRUCTURE.md), baseline = adopted
  ghost+d32 default, candidate = same + `--gumbel-root-menu 512`, looks
  40/60/80/100, alpha 0.05 OBF, SEQ_CUPED=1. Decision rule: RCI
  excludes 0 upward at any look => menu widening is real, adopt-or-gate
  further per wall cost (timing block recorded); RCI excludes +0.10
  downward or final look ns => R1.3b CLOSES (the 0.37 ceiling does not
  survive contact with play). Harness change landed for this
  (`df8e024b`): `--gumbel-root-menu` passthrough + `root_menu` in
  search provenance (always recorded), plus removal of a duplicated
  behavior-neutral argv block. Chained behind the D1 pilot
  (`r13b_gate_20260714.sh`, waiter PID 4173895).
- **Stage 1 chain incident + root cause (2 attempts lost, ~30 min):**
  all four arms died at the first forward with `CUDA driver error:
  unknown error`. Controlled bisect on john0: identical tiny job
  SUCCEEDS without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  and FAILS with it — deterministic env-var kill on WSL2 + RTX 5090 +
  torch 2.11.0+cu128 trainer workloads (the inference bridge tolerates
  the same var; every screen/gate ran fine with it). The var had been
  copied into the trainer chain from the bridge-side script pattern.
  Fix: never set it for trainer processes (v3 chain script carries the
  warning comment; INFRASTRUCTURE john0 notes updated). Relaunched
  04:18 (`stage1_retrains_v3_20260714.sh`, PID via waiter 4173058) with
  a CUDA warm-up gate + one retry per arm; v1b confirmed training (GPU
  100%, 31.6 GiB). GPU idle time from the incident: ~13 min.
- **D1 label-movement pilot PREREGISTERED (before any pilot data
  exists).** The 00:05 entry left "measured-hard" undefined; fixed now,
  strictly: benchmark ledgers carry no per-action serving data, so
  hardness is measured from the pilot's own repeat structure.
  (a) *Stratifier:* mega repeat agreement at n2048/d16 x2 —
  repeat-UNSTABLE fraction is reported as the noise-flippable census
  (frozen n4096 bank precedent: 39% unstable, 57.6% raw movement,
  median top-2 gap 0.049 — computed on the frozen instrument, not
  pilot data). (b) *Primary bar (STRICTER than the loose reading, and
  non-circular):* movement (mega argmax != serving choice) on
  repeat-STABLE roots >= 20% => D1 stays funded; below => D1 dies.
  (c) *Guard (recorded now):* if the bar passes but mean mega-regret
  per moved stable root < 0.05 points, the movement is near-tie churn
  and the verdict says so regardless of the rate. (d) *Spec:* stride-11
  ~727 roots of `rules_20260713_cycle4_ghost_d32_canonical_decisions
  .jsonl`, n2048/d16, repeats 2, FRESH dir `puzzle_bank_20260714_d1_
  n2048`, NO ghost (D1 labels are candidate teachers; ghost labels not
  safety-cleared). Analyzer `analyze_label_movement.py` (7 tests
  green) reports movement/regret overall, by stratum, and by phase
  (tile-count proxy, late >= 13 matching the V1b gate). Chained behind
  Stage 1 (`d1_pilot_20260714.sh`, waiter PID 4173887; deploys
  `df8e024b` python-only, zero exporter diff).
- **Night queue (GPU saturated ~24h):** Stage 1 v3 retrains+screens
  (~14h, to ~18:30) -> D1 bank+analysis (~1.2h) -> R1.3b gate
  (~5-8.5h, sequential stop possible). Survivor gates and the
  ghost-label safety fold remain the on-deck fillers.

## 2026-07-14 06:30 — v1b: selection is blind to value-target work (step-1 selected); measurement channel fixed by preregistration; control arm queued

- **v1b TRAIN COMPLETE (06:11, 1.9h) — and the champion selection rule
  picked STEP 1** (locked_val_final_q_regret 0.25492 at the warm start
  vs incumbent's recorded 0.25756; no training step beat it; trajectory
  0.259-0.293 across 2500 steps). Structural reading, not a null yet:
  the recipe warm-starts the incumbent on the exact corpora it already
  converged on, so q-regret (the selection metric) has no fresh signal
  to improve, and value-target work is INVISIBLE to selection — the
  arm's selected product is ~the incumbent (bank screen confirms:
  +0.2398 vs incumbent +0.2351, one 4e-6 step of drift, no signal).
  Meanwhile the q-head loss improved late (locked_val_q 0.2539 ->
  0.2217 at steps 2000-2500) — cause unknown (flag vs continued
  training), which motivates the control below.
- **Measurement-channel fix, PREREGISTERED NOW** (before v2q8/c1x4/t0pc
  finish and before any re-eval data exists): the Stage 1 offline bar
  is evaluated by ONE shared unmixed instrument —
  `_evaluate_records` with no Stage1TargetOptions on the trainer's
  exact locked-val slice (first 8x192 of cycle4 val), CPU — applied to
  incumbent + each arm's {selected, step-2500, SWA} checkpoints. Bar
  UNCHANGED in thresholds: value RMSE <= 0.90x incumbent AND q-regret
  <= incumbent + 0.05, both from the same instrument; ANY passing
  checkpoint qualifies its arm (that checkpoint is what proceeds to
  screen/gate). This is a channel fix (selection can't see value
  targets), not a bar move. Driver
  `stage1_unmixed_reeval_20260714.py` + chain armed (waiter PID
  4182622, fires with the D1 chain; CPU nice'd, no GPU contention).
  `torch_cascadiaformer_checkpoint_eval.py` gained `--max-batches` for
  future locked-slice re-evals (versioned).
- **Pure-control arm PREREGISTERED and chained** (`stage1_ctrl_
  20260714.sh`, waiter PID 4182962, behind the R1.3b gate): champion
  recipe rerun with ZERO Stage 1 flags, warm-started from the
  incumbent + the same unmixed re-eval. Role: causal control only —
  if ctrl reproduces the late-step q improvement, that movement is
  continued-training, not any flag; if ctrl's value RMSE matches
  incumbent while an arm's beats the bar, the flag is causal. Not a
  champion candidate.
- Design lesson recorded for Stage-1-class experiments: when the
  candidate mechanism targets a head the selection metric does not
  read, preregister the off-metric evaluation channel WITH the arm —
  selection-on-q + warm-start-on-converged-data otherwise degenerates
  to "return the incumbent".

## 2026-07-14 12:40 — Stage 1 chain done (all four arms select step 1); D1 bank died on K1 ledger rows, root-caused + refixed; R1.3b gate LIVE

- **Stage 1 v3 chain COMPLETE (12:18).** All four arms trained clean
  (~1.9h each) and ALL FOUR selected step 1: q-regret at the warm start
  (0.2549-0.2565) never beaten across any 2500-step trajectory. Bank
  screens of the selected checkpoints read incumbent-noise
  (v1b +0.2398). Two structural observations, both preregistered into
  the 06:30 measurement-channel fix before the later arms landed:
  (1) warm start + converged corpora + q-regret selection degenerates
  to "return the incumbent"; (2) every arm's late-step locked_val_q
  converges to ~0.221-0.224 (v1b 0.2217, v2q8 0.2228, c1x4 0.2223,
  t0pc 0.2240) — a shared attractor pointing at continued-training,
  not any flag; the ctrl arm decides. Also noted: v2q8's fresh
  quantile head trained 47.9 -> 0.73 pinball by step 1250 (real
  convergence, invisible to selection); t0pc's path-consistency loss
  14.74 -> ~5.5. Offline-bar verdicts wait on the unmixed re-eval
  (RUNNING on CPU since 12:21, parallel with the gate).
- **D1 pilot attempt 1 FAILED (12:18, zero roots), root-caused:** every
  seed's replay died at ply 76 — exact-K1 rows record the FULL legal
  enumeration (2,037-7,038 actions) while replay reconstructs the
  greedy-256 menu, and `replay_ledger_seed` bails the whole seed on the
  count mismatch. The existing `full_menu_fallback` path only covers a
  chosen action OUTSIDE the capped menu; when the exact solver's pick
  also sits inside greedy-256, the count check still fired. Two fixes:
  (a) TONIGHT'S RUN: exact-filtered ledger (`..._noexact.jsonl`, 7,600
  rows kept / 400 exact rows dropped) — scientifically the right
  substrate anyway, exact decisions carry zero label noise so D1 has
  nothing to relabel there (the bank's own sampler already excludes
  exact-frontier roots); (b) ROOT FIX (committed): replay tolerates a
  ledger action_count above the replay cap (the K1 wider-menu
  signature; seat + chosen-action checks still validate), so future
  K1-era ledgers replay without pre-filtering. cargo test 63/63.
- **R1.3b gate LIVE (12:20)** on block 2027073000 (ghost+d32 vs
  +root-menu-512, sequential CUPED superiority). Chain order now:
  gate (~6-8.5h) -> ctrl arm (~2h + CPU re-eval) -> D1 pilot v2
  (~1.2h, waiter PID 17200). GPU gap from the D1 failure: ~2 min.

## 2026-07-14 13:40 — Stage 1 offline bars: V1b/C1/T0 FAIL the value bar (scalar arms −2% to −5% vs −10% required); V2's number was pinball, honest metric re-running

- **Unmixed re-eval COMPLETE (13:25, CPU).** Instrument cross-check
  passed: incumbent q-regret 0.25763 on the re-eval vs 0.25756 at
  training time. Incumbent unmixed value RMSE `2.9793`; bar `2.6813`
  (−10%); q-regret allowance `0.30763`.
- **Scalar arms (comparable, MSE path) — all FAIL the value bar:**
  v1b step-2500 `2.8202` (−5.3%), SWA `2.8205`; c1x4 `2.8686` (−3.7%);
  t0pc `2.9181` (−2.1%). q-regret within allowance everywhere
  (0.262-0.269). Per the preregistered bar, **V1b and T0 do not
  advance to gates** unless the ctrl arm shows continued-training
  alone yields ~0% (it will more likely absorb most of the −2..−5%).
  C1's own bar was non-degradation: met (q loss improved, policy flat)
  — but with no positive signal, C1 stays a comparator, not a gate
  candidate.
- **V2's `pass=True` was an instrument artifact, caught before any
  verdict:** `_loss_components` selects pinball whenever the MODEL
  emits quantiles (branches on outputs, not on the eval flag), so
  v2q8's "RMSE 0.9648" is sqrt(pinball) — wrong units (its
  training-time pinball at step 2500 was 0.936, matching). Honest
  head-shape-independent metric now running: direct
  MSE(`value_vector`, `target_value`) on the same locked slice for all
  13 checkpoints (`stage1_value_mse_reeval_20260714.py`); the
  incumbent's direct MSE must reproduce ~8.88 as a self-check. V2's
  offline verdict WAITS on this number; bar unchanged.
- **Shared q-loss improvement across ALL arms at SWA** (locked_val_q
  ~0.2165-0.2179 vs incumbent 0.2533, ~-14%): flag-independent →
  provisionally continued-training+SWA. If the ctrl arm reproduces it
  with q-regret intact, ctrl's own SWA checkpoint becomes a legitimate
  bank-screen candidate (champion recipe, zero new flags) — the
  cheapest possible training-side candidate this program has produced.

## 2026-07-14 14:10 — Stage 1 OFFLINE VERDICT: all four arms FAIL the value bar; no arm advances to a gate

- **Direct value-MSE instrument COMPLETE** (self-checks passed:
  incumbent 2.9793 = sqrt of its locked_val_value; scalar arms
  reproduce the 13:25 numbers exactly). V2's honest numbers:
  step-2500 RMSE `2.8846` (−3.2%), SWA `2.8775` (−3.4%) — the
  quantile-mean is no better calibrated against outcomes than the
  scalar head. (Its step-1 selected checkpoint reads 95.3 — the
  random-init head, confirming selection never saw the trained head.)
- **Preregistered bar (value RMSE −10%, q-regret +0.05): V1b −5.3%
  FAIL · V2 −3.4% FAIL · T0 −2.1% FAIL · C1 −3.7% (comparator; its
  non-degradation bar met, no positive signal). No Stage 1 arm
  advances to an n256 gate.** Ranking within the nulls: v1b (search
  -value mixing) moved value RMSE the most — the PCZero mechanism is
  real but a factor ~2 too small at this corpus/recipe, and the ctrl
  arm may yet absorb part of it.
- Program read (R1_4 §5 kill ladder): trainer-only value densification
  (V1b/V2) and target-side path consistency (T0) are measured NULLs at
  this scale — consistent with §2.1 (value head not load-bearing at
  serving). R1.4 survivors: **D1** (pilot tonight), **P1** (label-side
  q-bias correction, needs a generation run), ctrl-SWA q-improvement
  lead (if ctrl reproduces q −14%, screen ctrl SWA). Stage 2/3 bundle
  retrains are NOT funded by these results.
- Still pending to interpret Stage 1 fully: ctrl arm (running after
  the R1.3b gate) — attributes the shared −2..−5% RMSE and −14% q
  drift to continued-training vs flags.

## 2026-07-14 22:15 — R1.3b CLOSED: root-menu 512 is a measured null at champion tier (final look ns)

- Gate `r13b_rootmenu512` (block 2027073000x100, ghost+d32 vs same +
  `--gumbel-root-menu 512`, sequential CUPED superiority): ran all 4
  looks, **FINAL_INCONCLUSIVE** — paired delta `-0.0275`, RCI
  `[-0.2664, +0.2114]`. Per the preregistered rule, **R1.3b CLOSES**:
  the R1.3a-priced recoverable tail (+0.37/game ceiling, 1.5% of
  decisions) is NOT captured by widening the greedy menu 256->512;
  the true effect is small-to-zero (upper RCI +0.21). R1.3c residual
  menu ideas deprioritized with it — the family needs a fundamentally
  different retrieval mechanism (R3.3's exact top-k bounds remain the
  live route to menu coverage).
- Wall: candidate was NOT more expensive (21.17 vs 22.37 s/decision) —
  menu width is not the serving cost driver (forwards batch by 256
  chunks; ranking is Rust-cheap). Whole gate ~9.7h.
- Methodology: first champion-tier CUPED gate to a FINAL look —
  variance reduction 20.1% (theta -0.507, r -0.457), on top of ghost
  pricing. Block 2027073000..3099 burned with the verdict.
- Chain advanced: stage1 ctrl arm now training (~2h + CPU re-eval),
  then D1 pilot v2 (~1.2h).

## 2026-07-15 00:30 — Ctrl arm CLOSES Stage 1: every flag effect was continued training; ctrl-SWA earns the preregistered screen

- **Ctrl (flagless champion recipe, warm-started) on the unmixed
  instrument:** step-2500 value RMSE `2.7953` (−6.2% vs incumbent),
  SWA `2.8056` (−5.8%) — MORE improvement than any flag arm (v1b's
  −5.3% was the best of them). Late-step q loss hits the same ~0.223
  attractor; SWA q `0.2198` (−13.2%) with q-regret `0.26311`
  (+0.0055, within allowance). Selection degenerated to step 1 again,
  as expected.
- **Attribution final: V1b, V2, T0 CLOSE OUTRIGHT** — the shared
  −2..−6% value drift and the −13-14% SWA q improvement are
  continued-training(+SWA) effects, present without any flag. No
  Stage 1 mechanism produced signal beyond the control. C1 closes as
  a flat comparator. R1.4's training-side survivors: D1 (bank building
  now) and P1 (generation-side).
- **Ctrl-SWA screen launched** (preregistered rule met): the −13.2%
  q-loss with intact regret is real and costs nothing new — bank
  screen behind the D1 chain (`ctrl_swa_screen_20260715.sh`, waiter
  PID 71185). Rule at launch: screen regret <= `0.2251` (incumbent
  0.2351 − 0.010) => preregister an n256 sequential CUPED gate; else
  the lead dies with the screen (locked-val q-loss improvements that
  don't move bank regret are calmer numbers decisions never consult —
  the exact null shape §5 predicted).

## 2026-07-15 01:45 — D1 PILOT PASSES ITS BAR DECISIVELY: 43.2% stable-label movement at 0.40 pts mean stake — D1 is FUNDED

- **Verdict (preregistered bar applied):** on the 380 repeat-STABLE
  roots (of 700 resolved), the n2048x2 mega argmax differs from the
  serving choice on **43.2%** (bar 20%) — and it is NOT near-tie
  churn: mean mega-regret of moved roots `0.397` (median `0.265`, p95
  `1.22`; 43 roots >= 0.5, 19 >= 1.0). **D1's premise is confirmed:
  serving-time labels are materially wrong exactly where decisions
  are contested.**
- Cross-checks: repeat-unstable fraction `45.7%` independently
  reproduces the ~46% noise-flippable census (R0.5). Phase gradient
  matches Stage 0's calibration drift: movement opening 57.1% > mid
  44.7% > late 33.9% — the OPENING is where cheap labels are worst.
- **Design-doc precondition honored:** §4-D1's caution — run D1's
  retrain only with a non-saturated head — now BINDS, because Stage 1
  proved the scalar heads can't absorb better targets (all flag arms
  <= control). D1 Stage A therefore pairs relabeled data with the
  **distq (q-quantiles 8) head** (the only CI+ training-side result),
  not the scalar champion head. Full Stage A design (substrate:
  training-corpus hard roots; harvest via Stage 0 criteria; fold
  weights; retrain recipe; gate) to be preregistered in daylight —
  it is a 30-60h GPU commitment and the queue's new top item.
- **Overnight filler launched** (`d1_full_relabel_20260715.sh`,
  waiter PID 79802, behind the ctrl-SWA screen): stride-1 relabel of
  ALL 7,600 non-exact canonical-ledger roots (~13h) => 10x-precision
  movement read + the complete mega-label benchmark for D1 Stage A
  eval and R0.5/R3.4 adaptive-budget supervision. Data generation
  only.

## 2026-07-15 01:50 — Ctrl-SWA lead DIES on the bank screen; R1.4 Stage 1 now fully closed

- Ctrl-SWA bank screen: mean regret `+0.2470` (CI `[0.2208, 0.2732]`),
  chose-best `33.0%` — WORSE than the incumbent's `+0.2351` and far
  from the preregistered `<= 0.2251` continuation rule. **The
  continued-training q-loss improvement (−13.2% locked-val) does not
  translate to decisions** — the exact §5 null shape ("calmer numbers
  decisions never consult"), now measured twice (values in Stage 1,
  q here).
- **R1.4 Stage 1 is fully closed: V1b, V2, C1, T0, and the ctrl-SWA
  lead — all nulls.** Central lesson strengthened: at this corpus and
  recipe, locked-val loss improvements of order 5-15% carry ZERO
  decision-level signal; only bank regret and paired gates count as
  screens for training-side candidates. R1.4 lives exclusively through
  D1 (funded, Stage A design pending) and P1 (generation-side).
- GPU: full-ledger relabel running (~13h, to ~14:45).

## 2026-07-15 15:10 — Full-ledger relabel REPLICATES the pilot at 10x; Stage A must be generation-first (raw records gone); ghost safety fold LAUNCHED

- **Full relabel COMPLETE (13.0h, 7,600 roots):** stable movement
  **43.6%** (pilot 43.2%), unstable fraction 42.6% (pilot 45.7%), mean
  moved regret 0.361 (pilot 0.397), phase gradient identical (opening
  50.3% > mid 49.5% > late 36.9%). The D1 label-noise measurement is
  now precise: `puzzle_bank_20260715_d1_full_n2048` (7,600 mega-labeled
  roots) is the standing benchmark for D1 Stage A eval and R0.5/R3.4
  adaptive-budget supervision.
- **Stage A design constraint discovered:** cycle4's raw per-root
  generation records are GONE from john0 (only packed npz shards
  remain, and packing discards seed/ply + per-action arrays — the
  known v4 limitation). Hard-root harvesting from the existing corpus
  is therefore impossible; **D1 Stage A becomes generation-first**:
  regenerate a corpus with raw retention -> harvest hard roots (Stage
  0 criterion) -> mega-relabel 10-20k (needs a small exporter feature:
  probe selection by (seed,ply) mask, since stride can't target) ->
  fold as weighted shards -> retrain with the distq head -> screen ->
  gate. Estimated 40-50h GPU end to end. This is the queue's top item
  but is deliberately NOT launched yet — John should see the plan
  first (it locks the GPU for ~2 days).
- **Ghost-label safety fold LAUNCHED** (queue #6; PID 148805;
  preregistered here BEFORE any read): 250-seed corpus at EXACT
  cycle-4 generation grade (n256/top16/d4/blend0.5, exact-endgame 0,
  TF32 on, M teacher) + `--gumbel-ghost-opponents` as the only
  variable, seeds 2026793000..3249 (registered), then top64+tail
  filtering, then a fold retrain (champion recipe + ghost shard at
  weight 1 on the 4,2,1 scale = 0.25 old units, the fleet-fold
  precedent), then reads vs the ctrl arm (same recipe, no fold — a
  perfect paired control). **Safety bar (preregistered): fold SWA bank
  regret within +-0.015 of ctrl SWA's +0.2470 AND unmixed q-regret
  within +0.01 of ctrl => ghost labels SAFE at 0.25-fold** (unlocking
  ~0.69x generation pricing for Stage A); any excess keeps the
  quarantine. Chain: gen (~1.5h) -> shard pipeline -> retrain (~2h) ->
  CPU re-eval -> SWA bank screen (~6 min). Note the step-1 selection
  degeneracy is expected again; all reads use step-2500/SWA vs ctrl's
  matching checkpoints.

## 2026-07-15 18:00 — D1 Stage A PREREGISTERED (design + tooling landed); generation launches tonight after the fold verdict; relabel tranche awaits John

**Stage A tooling (all committed, 65/65 exporter tests):**
- `--probe-roots` (95adb44f): puzzle-bank root selection by (seed,ply)
  JSONL mask — replaces stride, skips unlisted seeds without replay.
- `--decisions-out` + `--hard-roots-out` (a34b269a): selfplay-corpus
  generation sidecars — a replayable gumbel_decision ledger (proven to
  round-trip through the puzzle-bank replay in-test) and a per-root
  Stage 0 hardness census (top1-top2 completed-Q gap vs pairwise SE)
  computed in Rust at generation time. Together these remove the
  blocker found at 15:10 (packed shards discard provenance).

**Stage A preregistered plan:**
1. **Generation** (LAUNCHES TONIGHT after the ghost-fold verdict; a
   fresh corpus is reusable for any future cycle regardless of D1's
   fate): 1,250 seeds x 80 plies at cycle grade (n256/top16/d4/blend
   0.5, M incumbent teacher, TF32 on), seeds `2026794000..5249`
   (registered below), with BOTH sidecars. Ghost opponents ON iff the
   fold verdict is SAFE (0.69x pricing), else OFF. Requires a john0
   exporter rebuild (Rust changed) — done in the gap after the fold
   chain exits. ~7-10h.
2. **Harvest** (CPU): (seed,ply) mask = census rows with hard==true,
   stratified sample capped at 15k roots weighted toward opening/mid
   (movement 50/49% vs late 37%).
3. **Relabel tranche 1** (**AWAITS JOHN — ~26h GPU at 15k roots x2 at
   n2048/d16**): `--puzzle-bank --probe-roots` on the generation
   ledger, no ghost (teacher labels).
4. **Fold + retrain**: needs one more exporter feature (bank-mode
   training-record emission — bank rows are analysis records, not v4
   training records); build tomorrow. Retrain = champion recipe +
   relabeled shard, **distq head (--q-quantiles 8)** per the §4-D1
   precondition (Stage 1 proved scalar heads can't absorb better
   labels). Offline read vs ctrl; bank screen; n256 sequential CUPED
   gate on a fresh block, preregistered at launch.
5. **Kill rules**: if the retrain's bank regret does not improve
   >= 0.010 vs ctrl-SWA's +0.2470 at screen, D1 dies before any gate;
   if the gate is ns at final look, D1 closes and with it R1.4's
   training side (EI saturation confirmed at the label margin).

## 2026-07-15 18:20 — Fold retrain ran 11x faster and exposed a recipe-fidelity gap: CGAB_FUSED env accelerates the trainer

- The ghost-fold retrain completed 2,500 genuine steps in ~10 min
  (0.24 s/step) because `CASCADIA_CGAB_FUSED=1` — exported in the fold
  script for the generation bridge — also accelerates the TRAINER's
  forward (the fused relation-bias kernel; validated EXACT-parity on
  the serving stack, 07-04 25-game A/B). The Stage 1 arms and ctrl ran
  UNFUSED (2.7 s/step, ~1.9h each — ~9h of avoidable GPU across the
  slate).
- **Validity notes:** (a) Stage 1's verdicts are UNAFFECTED — every
  arm-vs-ctrl comparison ran on the same (unfused, fp32) stack, and
  the re-eval instrument scored all checkpoints identically. (b) The
  fold-vs-ctrl comparison mixes fused (fold) with unfused (ctrl)
  training; under the byte-identical parity precedent this is a
  speed-only difference — recorded as a caveat, and the fold verdict
  bar (±0.015 regret) is wide relative to any conceivable kernel
  epsilon. (c) RECIPE-FIDELITY GAP recorded: cycle4's champion trained
  with `--data-workers 4 --prefetch-factor 4 --tf32 --fused-optimizer
  --cgab-fused`; my manifest-recovered recipe carried none of these
  runtime knobs (tf32 in particular changes training numerics). Future
  champion-recipe retrains (incl. the Stage A distq retrain) must use
  the full optimized invocation — both for fidelity and ~10-40x cost.

## 2026-07-15 18:35 — GHOST LABELS CLEARED AS TEACHERS at 0.25-fold (safety bar passed on both legs)

- **Fold verdict (preregistered bar, both legs PASS):** fold SWA bank
  regret `+0.2516` vs ctrl SWA `+0.2470` (delta 0.0046 vs ±0.015
  allowance); fold SWA unmixed q-regret `0.25804` vs bar 0.27311
  (actually better than ctrl's 0.26311). A corpus with 20k ghosted
  roots folded at weight 1-on-the-4,2,1-scale trains a model
  indistinguishable from the clean-fold control on every instrument.
- **Consequence: R1_4 §8 quarantine LIFTS for generation.** Ghost
  opponents are now cleared for corpus generation at cycle grade —
  the Stage A corpus generates ghost-priced tonight (mechanical bar in
  the chain), and future cycles may use ghost generation with this
  fold entry as the precedent. Caveats recorded: cleared at n256/d4
  generation grade and 0.25-fold weight; higher fold weights need
  their own trial (fleet precedent pattern); fold trained fused vs
  ctrl unfused (byte-parity precedent, 18:20 entry).
- Fold-vs-ctrl descriptives (same-instrument): value RMSE 2.8697 vs
  2.8056, q 0.2298 vs 0.2198 — the ghost fold sits inside the
  continued-training band on every axis.

## 2026-07-15 18:45 — Pricing correction: ghost generation is ~2x SLOWER in all-seats selfplay (safety unaffected); Stage A gen ETA revised

- The fold corpus measured it: 250 ghosted seeds at 46.4 s/seed vs
  cycle4's 21.4 s/seed non-ghost (same n256 grade; session configs
  differ — 12 shared vs cycle4's setting — so treat as approximate).
  Mechanism: at SERVING, ghosting replaces opponent search and is a
  0.688x win; in all-seats SELFPLAY every seat ghosts its opponents'
  interior nodes via model policy evals, ADDING bridge traffic
  instead. **Ghost pricing is a serving-side win only; do not assume
  it for generation.** The safety clearance (18:35) stands — labels
  are teacher-safe — but the economic motive for ghost generation is
  gone pending a direct A/B.
- Stage A generation (running, ghost ON per the mechanical bar): ETA
  widens to ~7-16h (ends between ~02:00 and ~10:30). Letting it run:
  the corpus is scientifically valid either way, a restart would burn
  finished work, and the ghost-vs-clean generation A/B becomes free
  descriptive data (this corpus vs cycle4's timing at matched grade).
- Stage A relabel/retrain plans unchanged; the relabel tranche still
  awaits John's go.

## 2026-07-15 23:40 — Stage A generation attempt 1 DIED on a wildlife-bag edge (5h lost); exporter hardened; attempt 2 running ghost-OFF

- **Failure:** seed `2026794359` (ghost trajectory) reached "the
  wildlife bag is unexpectedly empty" at ~375/1250 seeds; the hard
  error aborted the run and the packed tensor work was lost (per-seed
  shards live in memory until the final pack; the sidecars survived —
  they stream). **Open rules-contract question for John (not mine to
  rule):** should an empty wildlife bag be reachable under
  `research_aaaaa`, and what is the correct rule when it happens? The
  sim currently treats it as an invariant violation.
- **Hardening (78d5e10b, 65/65 tests):** per-seed generation failures
  now SKIP with a recorded (seed, reason) list in the corpus manifest
  (`generation_skipped_seeds`); a >2% skip rate still fails the run
  (systemic-failure guard).
- **Attempt 2 launched (23:26, PID 196649):** rev 78d5e10b deployed +
  rebuilt; **ghost OFF by ruling** — pricing is serving-only (ghost
  generation measured ~2x slower) and a teacher-clean corpus removes a
  variable from Stage A; the 18:35 safety clearance stands for future
  use. Same registered seeds `2026794000..5249`, both sidecars. ETA
  ~7.5h (~07:00) at cycle4's non-ghost rate.

## 2026-07-16 00:30 — Wildlife-bag rules question RESOLVED by John: engine bug, found, replicated, fixed

- **John's ruling:** the bag must never be empty. His conservation
  argument: 100 tokens, at most 79-80 on boards plus 4 in the market
  on the last turn leaves >=16 in the bag; a three-of-a-kind wipe
  returns its tokens immediately, so 0 is unreachable under correct
  rules. Any empty bag is an engine failure.
- **Root cause (`crates/cascadia-game/src/game.rs`,
  `replace_wildlife`):** wiped tokens were set aside and returned to
  the bag only AFTER the automatic four-of-a-kind wipe loop finished,
  accumulating across iterations — a transient drain of 4 tokens per
  consecutive overpopulation. The official rule completes each
  resolution (remove -> refill -> return) before observing the next
  market, so the physical bag never shrinks across consecutive wipes.
  Late game (bag ~16, composition skewed toward one species — exactly
  when consecutive four-of-a-kinds are likely) the drain reached 0 and
  the refill hard-errored. Conservation was never violated; the leak
  was transient, which is why it took a rare deep line (ghost
  trajectory, seed 2026794359) to surface it.
- **Replicated** with a deterministic unit test
  (`consecutive_overpopulation_wipes_near_exhaustion_do_not_drain_the_bag`):
  market forced to four Bears, live bag shrunk to 7 tokens with four
  Elk on top (conservation preserved via `discarded_wildlife`); the
  Bear wipe refills with the four Elk, the second wipe needs 4 with 3
  left -> `WildlifeBagEmpty` on the pre-fix code. No 5h run needed.
- **Fix:** `replace_wildlife` now returns each resolution's set-aside
  tokens right after its refill (voluntary wipe and each automatic
  wipe iteration). Trajectories are bit-identical for all games whose
  wipe loop runs at most once (same return order, same hash counter,
  same bag state at each insert); only consecutive-four-of-a-kind
  games — the buggy paths — change. Full suite green: cascadia-game
  59/59 (incl. new repro), workspace all green, exporter 65/65
  (replay + golden fixtures unaffected).
- **Deployment:** NOT deployed to john0 — Stage A generation v2 is
  running (one job at a time; never deploy mid-job). It runs ghost-off
  on rev 78d5e10b where the edge is skip-and-record hardened (0 skips
  so far), so the corpus is safe. The fix rides the next deploy.
  Note for the record: corpora generated pre/post fix differ only on
  consecutive-wipe games — negligible label impact, now logged.

## 2026-07-16 00:56 — Stage A generation attempt 3: v2 was 3x slow (wrong bridge topology); John-approved restart on the bag-fix engine

- **Diagnosis of v2's pace** (63 s/seed vs cycle4's 21.4): v2 ran the
  SERVING bridge topology — `--model-sessions 12 --shared-model-session`
  (12 concurrent games through ONE shared python/CUDA process) — where
  cycle4's generation ran no `--model-sessions` at all, so workers =
  rayon threads = **24 concurrent games, each with an OWNED bridge
  session** (`main.rs`: `target_workers = model_sessions.unwrap_or(rayon
  threads)`). Observed under v2: GPU 35% utilized, the single shared
  bridge pegged ~486% CPU — half the parallelism plus a serialization
  choke. Recipe-fidelity failure, same class as the trainer-knobs gap
  (07-14): I copied the battery/fold invocation instead of cycle4's
  generation invocation. Rule now explicit: **shared-session topology is
  for serving jobs; generation replicates cycle4 (owned sessions,
  workers = rayon).**
- **John approved kill+restart** (~00:50): v2 killed at 50/1250 seeds
  (~1.6h work discarded vs ~11h saved; v2 would have finished ~21:30,
  v3 ETA ~08:30). Attempt 3 (PID 204702, launched 00:56) deploys rev
  `45fb5072` so the corpus generates on the **corrected rules engine**
  (wildlife-bag per-resolution return; skip-and-record retained as a
  backstop), ghost OFF per the 07-15 ruling, same registered seeds
  `2026794000..5249`, both sidecars, `--rayon-threads 32` exactly as
  cycle4's pipeline passed it (box caps at 24).
- Startup verified: build ok, 24 owned bridge sessions live, GPU 64%
  and warming (v2: 35%), VRAM 31.1/32.7 GB (cycle4-proven footprint).
  Monitor bzbzx6t57 on both chain and run logs. One operational scar
  for the notebook: a `pkill -f` pattern in the kill sequence matched
  the ssh session's own command string and killed it mid-sequence —
  kill by PID, never by `-f` pattern that appears in your own argv.

## 2026-07-16 02:08 — Stage A generation attempt 3 FAILED: john0/WSL reboot; zero usable output; no restart

- **Purpose/config:** continuation of the preregistered D1 Stage A corpus
  generation recorded at 00:56: source
  `45fb5072ec330103a45e80fc3f9e22d571f3f908`, incumbent cycle4
  `best_locked_val`, seeds `2026794000..5249`, 80 plies/seed, n256/top16/d4,
  blend 0.5, ghost off, exact-endgame 0, TF32 on, `--rayon-threads 32`, owned
  per-worker bridge sessions, durable decision/hard-root sidecars. The process
  still declared the exporter’s existing July-9 rules ID; the newly found
  rules-ID mismatch is recorded below as a blocker.
- **Observed failure:** `campaign_status.sh` at 02:08 found PID `204702` dead,
  john0 GPU idle (`0%`, `5.26 W`), and no Stage A helper live. The wrapper log
  contains deploy, successful build, and the 00:56:58 “generating” heartbeat,
  with no `COMPLETE` or `FAILED` line. Read-only host evidence gives boot time
  `2026-07-16 01:32:13` (`who -b`: 01:33); the WSL reboot terminated the run.
  No partial score or scientific result was read.
- **Durable evidence and SHA-256 (john0):**
  - `cascadiav3/logs/stage_a_generation_v3_20260716.sh` (3,383 bytes):
    `927bd0379452030be5aba53f38b8068b5f619ef4dd73ee672d2f1ba1674b712e`;
  - `cascadiav3/logs/stage_a_generation_v3_20260716.log` (372 bytes):
    `6f17b879bf3af6e90842514f783d0dde7d6ddcf0e2fdca94c4c0da2671a92311`;
  - `cascadiav3/logs/stage_a_gen_v3_20260716_build.log` (319 bytes):
    `7e0f304d2be2b556c555bb1d403ce24cb859cacfd96d925c9443717add0ef779`;
  - `stage_a_gen_v3_20260716_run.log`,
    `stage_a_20260715_decisions.jsonl`, and
    `stage_a_20260715_hard_roots.jsonl` are all 0 bytes, each SHA-256
    `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
  - `stage_a_20260715_tensor.npz` and
    `stage_a_20260715_manifest.json` do not exist.
- **Verdict:** operationally failed/invalid; no corpus, label, or strength
  evidence exists. The registered seed block remains allocated to Stage A.
- **Decision:** do **not** restart without John’s explicit permission. Before a
  rerun can produce admissible evidence, reconcile `RULES_CONTRACT.md` with
  the per-resolution return semantics in `45fb5072`, assign a new rules/config
  identity, and fail closed on July-9/July-16 mixtures. john0 remains idle.

## 2026-07-16 03:50 — Rules identity repaired: `..._rules_2026_07_16` / `cascadia-base-official-2026-07-16`

- Stage A attempt 3 confirmed dead: john0/WSL rebooted 01:32 (~35 min
  in); zero-byte run log and sidecars, no tensor/manifest; box idle,
  deployed revision still 45fb5072. The reboot is a host-side event
  (likely Windows Update), not a job failure.
- **Repair executed** (prerequisite per the 07-16 agenda ruling before
  any new scientific run): `RULES_SEMANTICS_ID` →
  `cascadia-base-official-2026-07-16` (cascadia-game), `RULESET_ID` /
  `EXPECTED_RULESET_ID` → `..._rules_2026_07_16` (exporter + all fresh-
  run benchmark/compare modules + their tests). Historical analyzers
  under `cascadiav3/reports/` deliberately keep the 07-09 stamp — they
  belong to their artifacts. RULES_CONTRACT.md rewritten: overpopulation
  is per-resolution return (John's conservation ruling), regression test
  named, and a second compatibility-break section added (07-09 identity
  = closed historical evidence boundary; the 98.2975 champion number
  needs a fresh canonical battery under 07-16 before use as a paired
  control; never mix identities in one CI).
- Tests: workspace green (cascadia-game 59, search 61, exporter 65);
  the 32 local python errors are the pre-existing no-numpy/torch Mac
  environment, not the bump.
- **Awaiting John:** explicit restart permission for Stage A generation
  (per the 07-16 agenda), now to be stamped under the 07-16 identity.

## 2026-07-16 09:00 — FULL AUTHORIZATION (John): execute the complete D1 pipeline to verdict. Stage A attempt 4 launched. Frozen preregistration.

John: "Full authorization to proceed with the full research agenda. Please
do all necessary engineering work, run and cue the experiments, and reach
a verdict on this. This is the most important task." Both prior decision
gates (Stage A restart, 15k relabel tranche) are hereby authorized;
champion PROMOTION, if the gate is positive, remains a separate John
ruling per standing methodology.

**Stage A attempt 4 LAUNCHED 08:54 (PID 15396, monitor bekcae52q):** rev
`6f40f010` (bag-fix engine + repaired `..._rules_2026_07_16` identity),
cycle4 topology (owned sessions, workers=rayon, `--rayon-threads 32`),
ghost OFF, seeds `2026794000..5249`, sidecars
`stage_a_20260716_{decisions,hard_roots}.jsonl`. ETA ~17:00.

**FROZEN PREREGISTRATION (research_answers_7_16.md §15 as amended below;
frozen before any Stage A data is read):**

1. **Harvest (before teacher output):** 15,000 hard roots (`hard==true`,
   exact-K1 rows excluded) split 6,000 opening / 6,000 mid / 3,000 late
   (phase = tile-count proxy: late >=13, mid >=8, else opening);
   within-phase stratification over fixed gap/SE deciles computed once
   from the full hard census, uniform within cells via deterministic
   salted hash order; <=12 roots/game first pass, deterministic
   phase-stratified top-up to <=16 on shortage.
2. **Sentinel: INCLUDED.** 1,500 phase-matched non-hard roots, same
   teacher methodology, descriptive only, never in any training arm.
3. **Teacher:** pinned champion (cycle4 best_locked_val) at n2048/d16,
   ghost OFF, TWO independent repeats; repeat search seeds REGISTERED as
   9000001 / 9000002; never uses the real hidden order.
4. **Aggregation:** visit-weighted Q over valid repeats
   (`Q_D1 = sum(n_i Q_i)/N`, `valid_Q = N>0`); policy = mean of repeat
   improved policies, renormalized, argmax tie-break by lowest action id;
   pooled population variance `sum n_i (v_i + (Q_i - Q_D1)^2) / N`;
   root value = mean; per-repeat arrays retained.
5. **Masked training view (versioned; raw corpus immutable):** base view
   masks cheap per-action Q / improved policy / search-root value at
   selected roots; D1 view carries the aggregated teacher fields and
   masks value/score/rank outcome losses; legacy rows default all-valid.
6. **Training mix:** Stage A base : cycle4 : cycle3 : D1 at raw weights
   4:2:1:1 (12.5% D1 draws, 4 expected passes/root at 2,500 steps x
   b192). K=8 distq head, warm start incumbent, full optimized invocation
   (CGAB_FUSED etc.), `--init-skip-mismatched`, frozen training seed
   20260630. Fail-closed per-source exposure audit required.
7. **Matched control (frozen rationale):** identical init/seed/steps/
   flags/corpora WITHOUT the D1 shard and WITHOUT masks — i.e. the
   status-quo deployable recipe. The comparison therefore measures the
   deployable D1 intervention as a package (mask stale + add fresh).
8. **Dose arms: RUN as secondary descriptives** (never gate): nested
   deterministic 5k (D1 weight 7/23) and 10k (7/11) retrains holding 4
   passes/root.
9. **Screen (locked bank):** D1 mean regret must be (a) <= control
   - 0.010 AND (b) <= 0.2370 absolute. Fail either => STOP before gate.
10. **Gate:** fresh registered block `2027079000..2027079099` (100
    seeds, touch-once), paired vs pinned champion, BOTH arms at n256/d4
    identical serving config, Lan-DeMets OBF looks 40/60/80/100, CUPED
    with the frozen baseline-score covariate. Positive stop or final
    CI+ => champion-tier confirmation run; **final-look ns => D1 and the
    R1.4 label-margin hypothesis CLOSE** (existing kill rule).
11. **Standing-stage rule:** reanalysis becomes pipeline infrastructure
    only after a positive gate AND one independent fresh-cycle
    replication.

Engineering scope now under construction (in dependency order): exporter
bank-mode training-record emission with per-repeat fields; harvest tool;
repeat-aggregation + masked-view builder; trainer masked-loss support;
matched-control invocation; relabel/retrain/screen/gate chain scripts.

## 2026-07-16 09:55 — D1 engineering COMPLETE; full pipeline chain armed (PID 25570)

All preregistered engineering built, tested, committed (c5455642 →
689f9d69) in ~1h of wall time while generation v4 runs:

1. **Exporter bank-mode training-record emission** (`--training-records-
   out`): repeat-aggregated v4 shards at relabeled roots (visit-weighted
   Q, pooled variance, mean renormalized policy, lowest-action-id
   tie-break), realized behavior outcomes, per-repeat audit sidecar,
   full v4 metadata contract (new `puzzle_bank_d1_relabel` mode).
   68/68 exporter tests.
2. **Harvest tool** (`harvest_d1_tranche.py`): 15k hard roots at
   6k/6k/3k phases, decile-stratified, interleaved phase filling under
   the shared per-game cap (sequential filling provably starves later
   phases), deterministic salted-hash order; 1.5k phase-matched
   sentinel. 7 tests.
3. **Masked training views**: optional `policy_valid`/`outcome_valid`
   per-record arrays through loader/collate/writer/filter/tail;
   `_loss_components` gates policy and outcome losses via masked means
   (absent/all-true masks are bit-identical to the pre-D1 trainer);
   `build_d1_training_views.py` (base view masks stale search at
   tranche roots; D1 view masks behavior outcomes; fail-closed
   ledger/audit/mask reconciliation; nested hash-ordered dose subsets).
   13 tests.
4. **Fail-closed exposure audit** (`audit_source_exposure`): replays the
   exact production sampler preflight. 4 tests.
5. **Model-vs-model gates**: `manifest` varied key in
   compare_search_shape (search must be identical, manifests must
   differ). +3 tests. `EXTRA_BANK_FLAGS` passthrough in
   run_puzzle_bank.sh.
6. Also fixed: flaky pairwise probe test (unseeded random-init Borda
   gate).

**Pipeline chain launched** (d1_pipeline_20260716.sh, PID 25570, rev
689f9d69, monitor bmvhuwykk): waits for generation v4 → deploy →
harvest → relabel tranche (pilot-exact teacher: jobs12 shared, TF32=0,
n2048/d16 x2, ~26h) → sentinel (~2.6h) → top64+tail prep + invariants →
masked views + dose subsets → exposure preflight (fail closed) →
retrains d1_15k / ctrl / d1_5k / d1_10k (full champion invocation,
K=8 distq, warm start, frozen seed) → bank screens → MECHANICAL
preregistered screen verdict (d1 <= ctrl−0.010 AND <= 0.2370; fail =>
stop) → on pass auto-launches the sequential CUPED n256 gate vs the
pinned champion on registered block 2027079000..99 (looks 40/60/80/100).
One caught-in-time bug: the first chain launch pinned rev be754db8,
which predates the EXTRA_BANK_FLAGS passthrough — the relabel would
have silently run the full ledger without the tranche mask or record
emission. Fixed before generation finished (waiter relaunched, PID
25570).

ETA: relabel ends ~07-17 late evening; retrains+screens ~2h; gate
overnight; **verdict expected ~07-18.**

## 2026-07-16 10:10 — Attempt 4 NEVER COMPLETED A SEED (24-owned CUDA contexts thrash); attempt 5 on the v2-proven 12-shared topology

- **Symptom:** 70 minutes at GPU "100% util" with zero completed seeds,
  zero-byte run log and sidecars. The tell: 149W power draw at 100%
  utilization (v2's healthy 12-shared config drew 253W at 35% util) —
  24 per-worker CUDA contexts time-slice on one 5090 and burn the GPU
  on context switching, not work. Retrospective: attempt 3 (same
  24-owned topology) also produced zero bytes in its 36 minutes — the
  "cycle4 topology" 21.4 s/seed precedent does NOT reproduce on this
  box today, and I promoted it after seeing only startup health, never
  a completed seed. Lesson recorded: **a generation attempt is not
  healthy until seeds complete; startup GPU% is not evidence.**
- **Engine exonerated:** 500 full random 4p games in 0.95s at HEAD (new
  permanent regression test `random_full_games_terminate_quickly`), and
  one full selfplay seed (80 plies, n256/d4, uniform fallback) in 248s
  locally — the wildlife-bag fix and search machinery are healthy.
- **Attempt 5 LAUNCHED 10:02 (PID 26197, monitor bn34wrswc):** rev
  689f9d69, 12 shared sessions / rayon 16 (v2's measured 63 s/seed),
  TF32=1, ghost OFF, same seeds/sidecars. ETA ~08:00 07-17. Pipeline
  waiter repointed at v5 and relaunched. Revised pipeline ETAs: relabel
  ends ~07-18 morning; screens midday; gate overnight; **verdict
  ~07-18/19.**

## 2026-07-16 21:15 — BUDGET RULING (John): "a few GPU days"; 125-day-class programs killed

John: "125 gpu days is outrageously expensive, i have a few gpu days to
allocate, i want a model that is faster than 125 days."

Consequences, recorded as a campaign ruling:

1. **Full Rival build (P4 GPU backend, P5 RivalNet, MF calibration) is
   DEAD at this budget.** The P2a/P2b GPU merit probes (~2–4 GPU-days)
   are consequently MOOT and canceled — no measuring the merit of an
   unfundable build. NX's GPU-resident build dies with the same ruling.
2. **Surviving portfolio** (fits "a few GPU-days"):
   - D1 to verdict (running; ~1.5–2.5 GPU-days remaining);
   - Gate 0 fresh 07-16 baseline (~0.3–0.5 day; campaign-required);
   - if D1 positive: D1 cycle 2 (~2.5 days/cycle; dose arms 5k/10k/15k
     decide whether the coverage curve justifies it);
   - **Rival-Lite** (late-game-only terminal relabeling, final 2–5
     turns, no RivalNet/MF/backend; reuses the complete D1 pipeline;
     v2 precedent +0.42/+0.52 CI+): ~1.5–2 days, the natural post-D1
     candidate under the budget.
3. **Free tier retained:** M1 selfish ceiling tomography (CPU-only)
   still runs — it gates whether ANY further GPU spend is rational.
   The pre-merit CPU scope shrinks to: WI-1 (CPU-1 battery), WI-2
   (tomography), and the WI-3 golden-trace prep only insofar as
   Rival-Lite's late-game continuations need the extracted incumbent.
   WI-4a/b/c (probe harnesses, chance-coupling machinery) are held
   unless Rival-Lite's design needs coupling proofs for its paired
   late-game panels — decide at design time, not by default.

## 2026-07-16 23:55 — WI-3 prep (golden-trace machinery) landed and independently verified; runnable CPU list COMPLETE

Agent commits `b0f36b71..9055e6fb` on `feat/rival-cpu-machinery`:
`golden_trace.rs` (1,967 lines), lib.rs registration, doc section 21.
Independent verification (this session, worktree at branch tip 9055e6fb):

- **Scope**: diff touches exactly 3 files. Zero exporter / cascadia-game /
  trainer / bridge / torch files; no `cascadia-v3-policy` crate created
  (hold ledgered in the doc). D1 wall respected.
- **Tests**: `cargo test -p cascadia-rival` green — 112 lib + all
  integration suites (1 pre-existing `#[ignore]` release battery).
  `cargo check --workspace` green (pre-existing cascadia-api warning only).
- **Contents**: `GoldenDecisionTrace` v1 (deny_unknown_fields, canonical
  JSON SHA-256, immutable publisher), `compare_traces` first-divergence
  comparator (32 variants, causal order), `GoldenTraceManifest` v1
  (seed-sorted, foreign-identity/duplicate-seed refusal), `CanonicalF64`.
  Menu + bridge exchanges stored as ordered digests (SHA-256), scalar
  decision facts verbatim.

**CPU scoreboard: WI-1 done (CPU-1 claimed), WI-2 done (tomography
built+verified), WI-3 prep done. All pre-D1-wall CPU work is complete.**
Remaining WI-3 half (production trace capture via extracted incumbent)
stays held at the D1 durable boundary per the budget ruling.

## 2026-07-17 11:05 — Stage A corpus COMPLETE; harvest clean; relabel running

Generation v5 finished: 1250/1250 seeds, 100,000 decision records, 25.0h
wall (~72 s/seed, zero skips), rules 07-16, rev 689f9d69. Artifacts:
`stage_a_20260716_{tensor.npz (2.97 GB), decisions.jsonl (100k),
hard_roots.jsonl (100k), manifest.json}`.

Pipeline waiter verified artifacts and chained autonomously at 11:03:19.
Harvest (d1_harvest_20260716): census 100k rows, hard pool 61,283;
tranche 15,000 selected, ZERO shortfall in all phase cells (6k opening /
6k mid / 3k late, decile-stratified, caps 12/16); sentinel 1,500, zero
shortfall. Masks: tranche bdcc1387…, sentinel 662d568b…, salt
`cascadia-d1-tranche-2026-07-16` as preregistered.

Relabel launched 11:03 (PID 45634): pilot-exact n2048/d16 ×2 repeats,
12-shared sessions, TF32=0 path via run_puzzle_bank.sh, records out
`d1_records_20260716.npz`. ETA ~26h → views/retrains/screens/verdict
~07-18 afternoon.

## 2026-07-18 17:05 — RELABEL DONE; records verified; sentinel relabel running

Tranche relabel finished 16:59 (25.9h, right on the ~26h estimate).
Output `d1_records_20260716.npz` (246 MB) verified this session:
- 15,000 roots (CSR action_offsets 15001), 3,835,162 candidate actions,
  74.7M relation edges, 1.74M token rows.
- Full v4 target set present: improved_policy, target_q,
  target_score_to_go, q_valid/q_count/q_variance, visits, priors,
  exact_afterstate_score_active(+decomp), final_score_vector,
  rank_vector, search_root_value.
- Metadata: mode/source = puzzle_bank_d1_relabel, outcome_provenance =
  behavior_trajectory_realized, rev 689f9d69, ruleset ..._2026_07_16,
  schema expert_tensor_shard.v4. All contract fields correct.

Chain proceeded autonomously to sentinel relabel (1,500 roots, ~2.6h).
Then: training views → preflight → 4 retrains → screens → mechanical
verdict → CUPED gate. Verdict ETA tonight ~22:00–23:00.

## 2026-07-18 21:10 — D1 SCREEN VERDICT: KILL. Program stops before the gate per preregistration

The chain completed autonomously at 20:51:57 (`D1 PIPELINE DONE (screen
kill)`). All four retrains (~10 min each, warm-start) and five screens
ran clean. The preregistered mechanical rule (d1_15k must beat ctrl by
>= 0.010 mean bank regret AND land <= 0.2370 absolute) fired KILL:

| arm            | mean_regret | vs ctrl   | p95    | zero-regret |
|----------------|-------------|-----------|--------|-------------|
| ctrl           | 0.2631      | —         | 1.0558 | 0.3059      |
| d1_5k          | 0.2504      | −0.0127   | 0.9630 | 0.3447      |
| d1_10k         | 0.2609      | −0.0022   | 0.9703 | 0.3152      |
| d1_15k (best)  | 0.2721      | +0.0090   | 1.0731 | 0.3401      |
| d1_15k (swa)   | 0.2650      | +0.0019   | 1.0411 | 0.3230      |

Verdict JSON: `reports/d1_20260716_screen_verdict.json`
(d1_vs_ctrl −0.00899 vs bar 0.010 → fail; absolute 0.2721 vs 0.2370 →
fail). **D1 is closed. No CUPED gate was run; seed block 2027079000–99
remains untouched.**

Honest observations recorded for the post-mortem, none of which
override the mechanical verdict:

1. **The primary arm went the wrong direction**: the full 15k deep-label
   dose made bank regret WORSE than the control fine-tune (+0.009).
   This is not "no effect"; it is mild harm at full dose.
2. **The dose curve is inverted-U**: 5k improved on ctrl by 0.0127
   (nominally larger than the 0.010 margin bar), 10k roughly flat,
   15k negative. Post-hoc, no CI, screens-rank-gates-decide —
   hypothesis-generating only. Consistent with the deep-label tranche
   (hard roots only, distributionally skewed) acting as a mild
   regularizer at low dose and a distribution-shift penalty at high
   dose.
3. **Every arm, including ctrl, failed the absolute continuity bar**
   (0.2370, the incumbent's pilot regret on this bank). The warm-start
   fine-tune recipe itself regresses the incumbent's bank behavior
   before any deep labels enter. Any future retrain-based program must
   first fix this continuity leak or the absolute bar will kill it
   regardless of label quality.
4. GPU cost of the full D1 cycle as run: ~2.2 GPU-days (25.0h gen +
   25.9h relabel + 2.7h sentinel + ~1.7h trains/screens).

Portfolio consequence (per the 07-16 budget ruling decision tree): the
"D1 positive" branches are dead. Remaining: Gate 0 (campaign-required
fresh baseline), M1 selfish-ceiling tomography on Gate 0's incumbent
games (CPU, free), then John rules on whether Rival-Lite's 1.5–2
GPU-days are justified by the M1 ceiling number — or the GPU campaign
closes at ~2.6 days total spend.

## 2026-07-18 21:15 — PREREGISTRATION: Gate 0 fresh champion baseline under rules 2026-07-16

Purpose: measurement, not hypothesis test. Mints the canonical champion
baseline under the post-bag-fix rules identity; its decision ledgers
double as the incumbent-measured game set for M1 tomography replay.
No decision rule attached; no promotion implications by itself.

- Script: `cascadiav3/scripts/run_rules_20260716_gate0.sh` (versioned;
  07-09 rebaseline structure, distq arms omitted per budget ruling).
- Config: champion identity only — cycle4 scalar CascadiaFormer-M,
  `full_v3_gumbel_selfplay_cycle4/best_locked_val`. Arms: 1-game smoke,
  no-search floor, n256/d4, n1024/d16 (canonical champion grade).
  All champion serving params pinned (top-m 16, depth-rounds 1,
  market-samples 8, blend 0.5, k-interior 16, jobs 12).
- Seeds: `2027160000..2027160099` (100 games), fresh block, zero prior
  references in any log or report. Touch-once.
- Rules: `..._rules_2026_07_16`, rev 689f9d69 (already deployed and
  built on john0 by the D1 chain; script re-verifies the ruleset grep
  and revision match before reusing anything).
- Budget: ~0.4 GPU-day (n1024/d16 dominates at ~46 s/dec × 8,000
  decisions / 12 jobs ≈ 8.6h; n256/d4 ~0.7h; floor ~minutes).
- Reference points (07-09, closed identity, NOT comparable pairwise):
  champion n1024/d16 98.2975; n256/d4 and no-search floors in the
  07-09 rebaseline reports.

## 2026-07-19 06:40 — GATE 0 COMPLETE: fresh champion baseline under rules 2026-07-16 = 98.19

Battery ran clean overnight (PID 174918, ~9.5h, seeds 2027160000x100,
rev 689f9d69, all reports status=pass, completion marker written):

| arm                        | mean seat | P50  | P90   |
|----------------------------|-----------|------|-------|
| no-search policy-head      | 92.055    | 92.0 | 96.0  |
| no-search q-head           | 90.8975   | 91.0 | 95.0  |
| no-search greedy heuristic | 87.7675   | 88.0 | 92.0  |
| cycle4 n256/d4             | 97.145    | —    | —     |
| **cycle4 n1024/d16 (canonical)** | **98.19** | 98.0 | 101.0 |

**98.19 is now the canonical champion baseline under
`..._rules_2026_07_16`.** It sits 0.1075 below the closed 07-09 number
(98.2975, P50 98.0, P90 102.0) on a DIFFERENT fresh seed block — the
delta is within seed noise for 100 games and is NOT evidence of a
rules-fix effect (the bag fix only alters trajectories on consecutive
four-of-a-kind wipes). The 100-goal gap under current rules: 1.81 pts.

Campaign spend to date: ~2.6 GPU-days (D1 ~2.2 + Gate 0 ~0.4).
Next per decision tree: M1 selfish-ceiling tomography (CPU, free) on
this battery's champion games; then John rules Rival-Lite (1.5-2 days)
vs campaign close.

## 2026-07-19 09:40 — PREREGISTRATION: CBDDB smoke test (John's request)

John: "can you try again on this CBDDB cards_alt rule set but using the
techniques we know now? don't do a multi day run, i'm looking for a
smoke test that can tell me if this has promise."

CBDDB = Bear C, Elk B, Salmon D, Hawk D, Fox B — the April-2026
cards_alt research set. Historical (v1/v2 NNUE/MCE era) anchors, base
scores (no habitat bonus): greedy-MCE-750 ~96.5, NNUE-MCE-750 ~97.2;
value-net RMSE plateaued 6.04–6.14 vs 4.81 on Card A (Hawk-D variance).

Discovery: v3 cascadia-game already implements ScoringVariant B/C/D for
all five animals (incl. Salmon-D/Hawk-D rescore invalidation); only
selection plumbing + verification is missing. New ruleset identity:
`cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19`.
AAAAA paths must stay bit-identical (flag default aaaaa).

Design (~0.5 GPU-day cap, stages gated in order):
- Stage 0 (CPU): plumbing (ScoringCards::CBDDB, GameConfig::
  research_cbddb, exporter --scoring-cards, python harness passthrough)
  + port legacy alt-card tests from archive branch as
  crates/cascadia-game/tests/alt_card_scoring.rs. Any legacy-vs-v3
  scorer disagreement BLOCKS GPU stages until resolved and is reported
  to John (scorer semantics = rules design = his authority).
- Stage 1 (GPU ~1h): zero-shot transfer — champion cycle4 weights +
  Gumbel search under CBDDB, no retraining. Arms: no-search floors,
  n256/d4 x 100 games. Seeds 2027190000..99 (fresh block, touch-once).
- Stage 2 (GPU ~9h): one adaptation cycle — CBDDB self-play corpus
  ~400 seeds x 80 plies at n256/d4 (seeds 2027191000..1399), one
  warm-start retrain (D1 recipe, K=8 distq init-skip-mismatched, seed
  20260630), re-eval n256/d4 x 100 on the Stage-1 block + n1024/d16
  x 30 spot (seeds 2027190000..29 subset).
- Promise verdict (informal, this is a smoke not a gate): PROMISING if
  zero-shot n256/d4 lands >= ~96.5 (old-tech ceiling zone) AND the one
  cheap fine-tune adds >= +1.5 on the paired block; NOT PROMISING if
  zero-shot craters (<94) and fine-tune moves < +0.5; anything between
  = report facts to John, no auto-continuation. No champion/promotion
  implications; CBDDB scores are NOT comparable to AAAAA numbers.

GPU sequencing: john0 idle (Gate 0 done; Rival-Lite awaiting John's
ruling; M1 is Mac-CPU). Stage 1 launches only after Stage 0 tests are
fully green.

## 2026-07-19 11:05 — CBDDB Stage 0 COMPLETE; Elk-B ruling; GPU stages unblocked

Stage 0 landed and independently verified (all suites run by this
session): engine 61+49+1 green, exporter 72 green including the pinned
AAAAA golden-hash test (bit-identity holds), python 383 tests with the
pre-existing 32 environmental errors unchanged.

- Plumbing: `ScoringCards::CBDDB`, `GameConfig::research_cbddb`,
  exporter `--scoring-cards aaaaa|cbddb` (12 call sites switched, 22
  emission sites resolved, fail-closed input validation), benchmark
  harness passthrough incl. the search-benchmark control-arm path.
  New identity: `cascadia_research_cbddb_4p_no_habitat_bonus_rules_2026_07_19`.
- Verification: 49 legacy tests ported from the archive branch
  (tests/alt_card_scoring.rs). 48/49 agreed exactly (Bear C table+bonus,
  Salmon D runs, Hawk D LOS/matching-DP, Fox B pairs).
- **RULING (John, 11:00): Elk B uses v3 strict-diamond semantics** (4-elk
  shape must be a true rhombus, per the official card). The legacy
  engine's looser any-member rule is historical; the April-2026 anchors
  (~96.5/97.2) are therefore slightly generous vs this rule. The
  disagreement test now pins 11 (triangle+single) as ruled.

Stage 1 (zero-shot) launches next on john0 (idle).

## 2026-07-19 13:40 — M1 SELFISH-CEILING TOMOGRAPHY COMPLETE: ~zero provable endgame headroom

Pipeline: Gate 0 champion battery (100 games, n1024/d16, rules 07-16) →
rival-ingest-exporter (all 100 games ingested; every one of 8,000
action hashes resolved uniquely; replayed finals exactly reproduced all
recorded score breakdowns; 100 sealed rival_trajectory_ledger.v1 files)
→ rival-tomography (evidence domain incumbent_measured, witness
semantics lower_bound_only). Two runs, 400 seat-trajectories each:

| config                        | T0 repack sum | T3 hindsight sum | mean/seat  | p90 | max |
|-------------------------------|---------------|------------------|------------|-----|-----|
| default (4k iters, beam 4)    | 0             | +18              | 0 / 0.045  | 0   | 2   |
| strong (40k iters, beam 16)   | +48           | +49              | 0.12/0.122 | 0/1 | 3-4 |

Reading: with the FULL chance tape known in hindsight, bounded oracle
search finds ~0.12 pts/seat of provable improvement over the champion's
realized play; final-board wildlife repacking finds ~0.12 pts/seat.
Median seat improvement is ZERO in every configuration. Against the
1.81-pt gap to 100, witnesses fund at most ~7-13% of the needed
distance — and these witness classes cover exactly the late-game
window Rival-Lite targets.

Preregistered asymmetry honored: witnesses fund, never kill — these
lower bounds do NOT prove the ceiling is low (bounded beam, chronology-
preserving pruning is conservative). But the funding test comes back
EMPTY: M1 provides zero evidentiary support for spending 1.5-2 GPU-days
on late-game terminal relabeling. Rival-Lite's case now rests solely on
the v2-era precedent, which D1 taught us transfers poorly.

Artifacts pinned: reports/m1_tomography_gate0_{default,strong}_20260719
.json (summary sha256 in-file); ingest bridge commits 93cbf603..9f0ab642
on feat/rival-cpu-machinery (151 tests green, verified this session).
Recommendation to John: do NOT fund Rival-Lite; ruling is his.

## 2026-07-19 14:05 — RULING (John): Rival-Lite KILLED; AAAAA GPU campaign CLOSED; CBDDB is the active line

John: "kill rival lite / continue with the alt rule set experiments /
this rule set should comfortably score above 105"

1. **Rival-Lite: KILLED.** With D1's screen kill, the fine-tune
   continuity leak, and M1's empty witness table, no relabeling program
   survives. The AAAAA (Card-A) GPU campaign closes at ~2.6 GPU-days
   total (D1 ~2.2 + Gate 0 ~0.4). Final AAAAA state: champion cycle4
   scalar at 98.19 (rules 07-16 canonical baseline), goal gap 1.81 pts,
   unresolved levers exhausted at this budget.
2. **CBDDB is the active experimental line.** John's calibration: this
   ruleset "should comfortably score above 105" — that number replaces
   100 as the aspirational bar for CBDDB work. Historical old-tech base
   anchors (~96.5/97.2) were nowhere near it; the smoke test measures
   how much of that distance v3 techniques close for free.
3. CPU machinery (rival crate: battery, tomography, golden traces,
   ingest bridge) remains merged on its feature branch as reusable
   measurement infrastructure — the tomography harness applies to any
   future CBDDB incumbent unchanged.

## 2026-07-19 14:45 — CBDDB STAGE 1 ZERO-SHOT: 99.4675 — promise bar CLEARED

Zero-shot battery complete (report pass, ruleset ..._2026_07_19, rev
9e71ff1d, seeds 2027190000x100, 8000 decisions):

| arm                     | mean seat | P50   | P90   |
|-------------------------|-----------|-------|-------|
| no-search policy-head   | 88.58     | —     | —     |
| no-search q-head        | 88.53     | —     | —     |
| greedy heuristic        | 80.89     | —     | —     |
| **zero-shot n256/d4**   | **99.4675** | 100.0 | 105.0 |

The AAAAA champion, with NO retraining, already beats the entire
April-2026 old-tech line under CBDDB (~96.5 greedy-MCE / ~97.2
NNUE-MCE, both under the more generous legacy Elk-B) by +2.3, and
scores +2.3 above its own AAAAA n256/d4 number (97.145) — consistent
with John's calibration that CBDDB "should comfortably score above
105": half the games already reach 100 zero-shot at generation-grade
search. Preregistered promise condition #1 (zero-shot >= ~96.5) is met
decisively. Search + true-scored rollouts recover ~11 points over the
miscalibrated no-search floor (88.6 -> 99.5).

Stage 2 launching: 360+40-seed CBDDB corpus (n256/d4, seeds
2027191000..1399), warm-start fine-tune (D1 recipe; selection on the
CBDDB val split, NOT the AAAAA val set), paired re-eval on the Stage-1
block (n256/d4 x100, n1024/d16 x30). Script:
cascadiav3/scripts/run_cbddb_smoke_stage2.sh. ETA ~12h.

## 2026-07-19 15:20 — AUTHORIZATION (John): "continue trying to break past 105"

Standing authorization for the CBDDB campaign: proceed through the
Stage-2 decision branches autonomously (iterated warm-start self-play
cycles while the dose-response supports them), inside the standing
few-GPU-day envelope, with per-cycle logging and kill criteria. Any
from-scratch CBDDB training run (week-scale) still requires a separate
explicit ruling. Certification protocol: any claim of ">105 achieved"
must come from a fresh touch-once seed block (2027195000+), not the
paired screening block 2027190000-99.

## 2026-07-20 09:45 — CBDDB Stage 2 fine-tune REGRESSED; trust-region anchor fix built + launched

Stage 2 naive warm-start fine-tune: n256/d4 = **98.75** (P50 99, P90 105)
vs zero-shot **99.4675** — paired **-0.72 regression**. Training metrics
showed the tell: value loss improved (val Q 8.74->5.61) while policy loss
stayed flat (~2.79) and PLAY got worse — the value head drifted in a way
that hurt search-time blending (blend-weight 0.5). Same continuity-leak
signature as D1's screen kill. Two rulesets, same failure => RECIPE
problem, not rules.

Killed the in-flight ft n1024/d16 eval (John's call): it measured the
abandoned model at champion grade with no paired zero-shot control, so
~2h of GPU for an uninterpretable number. Freed the GPU for the fix.

FIX (built + verified this session): trust-region anchor in
torch_train_cascadiaformer.py — a frozen incumbent supplies forward
KL(anchor||current) on the policy (action_mask-aligned, mirrors the
policy loss exactly) and L2 on the value_vector+score_decomposition
(outcome_valid, mirrors the value loss). Default-off bit-identical
(13/13 anchor tests pass on john0 incl. bit-identity). Loader tolerates
the incumbent's scalar q-head under --init-skip-mismatched (no anchor
term uses the q-head). Flags: --anchor-manifest,
--anchor-policy-kl-weight, --anchor-value-l2-weight.

Fix sweep running (rev d27949df, existing Stage-2 corpus reused — NO
regeneration): arm A "vonly" (l2=2.0, kl=0 — tests the value-drift
diagnosis), arm B "both" (kl=2.0, l2=2.0 — full trust region), each
eval'd n256/d4 x100 paired vs the 99.4675 floor; plus the missing
zero-shot n1024/d16 x30 control. Anchor confirmed live (val
anchor_value_l2=0.93 being penalized). Promote only if an arm beats
99.4675. ETA ~evening.

## 2026-07-20 15:40 — From-scratch CBDDB campaign PREREGISTERED (re-scoped to ~3 GPU-days)

John ruled: TRUE random-init from-scratch CBDDB model (tests whether
Card-A priors are a ceiling vs a floor). Then re-scoped: "12 GPU days is
too high, we can reduce the number of evaluations and use cheaper search
budgets." Revised plan (levers applied):

- Bootstrap: greedy + EI-0 greedy-state search bootstrap under
  --scoring-cards cbddb (gen largely model-free) -> random-init
  supervised train (model-S, ~25k steps, LR 2e-4 from-scratch fallback).
  ~0.4 GPU-day.
- Self-play cycles at CHEAP search n128/d2, ~800 seeds x 80 plies
  (~0.3 GPU-day/cycle vs 1.7 at n256/d4 1250 seeds). Ramp to n256/d4 for
  the final 1-2 sharpening cycles only.
- Evals at MILESTONES only, n256/d4 x100 (the paired bar vs zero-shot
  99.4675); single n1024/d16 x30 at the very end.
- MILESTONE GATE at bootstrap + 2 cheap cycles (~1.0 GPU-day cumulative):
  read climb rate toward 99.4675; continue only if the slope projects to
  reach/beat it. Full campaign ~2.5-3 GPU-days; worst-case exposure ~1
  day at the gate.
- Honest caveat on record: cheaper generation search = weaker teacher
  targets, may need more cycles / cap the ceiling; the cheap-early/
  expensive-late ramp mitigates; first knob to turn if the climb stalls
  is generation search budget.
- Interpretation: if from-scratch plateaus below 99.4675, that PROVES
  the Card-A transfer priors help (pivot back to warm-start + stronger-
  teacher). Not-comparable to AAAAA numbers; CBDDB target >105 (John).

FEATURES (John asked to re-evaluate; audit found no Card-A scoring bug,
all scoring-derived features auto-adapt via the variant-aware engine):
building CARD-AWARE encoder changes (Card-A output byte-identical =
golden-hash gate; only CBDDB paths change): (1) Hawk-D line-of-sight
relation edge (the one genuine structural gap — Hawk D scores LOS-pair
matching, absent from the adjacency-only relation graph); (2) card-aware
recompute of 6 stale Card-A hint dims (bear group-size, elk shape,
hawk-LOS-typed, fox pair-type). Seeds: fresh blocks 2027193000+ (gen),
2027195000+ reserved for certification.

## 2026-07-20 17:55 — Anchor-fix CONCLUSION: warm-start fine-tune cannot beat zero-shot on CBDDB (98.75 across all arms)

All three warm-start fine-tune variants on the n256/d4 CBDDB corpus land
at IDENTICAL 98.75 mean / 99.0 p50 / 105.0 p90 (n256/d4 x100, block
2027190000-99), below zero-shot 99.4675:
- naive ft (no anchor): 98.75
- vonly (value L2 anchor, w=2): 98.75
- both (policy KL + value L2, w=2): 98.75
Verified genuinely distinct checkpoints (games-file SHAs 8ac27e56 /
4237334d / 5b484940 all differ) that independently converge to the same
aggregate. Anchor at w=2 was too weak to differentiate (value L2 still
drifted to 2.38), but the deeper cause is the TEACHER-STUDENT GAP: the
n256/d4 self-play targets are generated at the SAME budget the student
plays, so imitating them caps the student at ~its own level while the
fine-tune step slightly degrades the 99.47 warm-start point. Anchoring
can at best recover 99.47; it cannot manufacture headroom from
same-budget targets.

Conclusion: warm-start fine-tuning on same-budget self-play is a
dead-end for beating zero-shot on CBDDB. This validates BOTH surviving
levers: (a) from-scratch (escape the transferred priors), (b) stronger-
teacher corpus (targets from deeper search than the student plays). The
from-scratch campaign is the active path. Anchor machinery retained
(default-off bit-identical) for any future stronger-teacher warm-start.

Zero-shot n1024/d16 x30 control still running (~20:30) — champion-grade
CBDDB baseline, banked regardless.

## 2026-07-20 17:55 — Path B feature (action-sourced Hawk-LOS edge) VERIFIED

Independently verified this session: action_relation_tail parity tests
pass for AAAAA AND CBDDB (fixture + real-state), golden hash unchanged,
80/80 exporter tests. Byte-exact train/serve parity for the new
action-source LOS edges (ids 13-16) confirmed. From-scratch feature set
is now complete and locked (commit 319e373b).

## 2026-07-20 22:15 — Champion-grade CBDDB baseline: zero-shot n1024/d16 = 101.2

The AAAAA champion, zero-shot under CBDDB at champion-grade search
(n1024/d16 x30, block 2027190000-99): mean 101.2, p50 102, p90 106.1.
+1.73 over its own n256/d4 zero-shot (99.4675) — search budget alone
buys ~1.7 pts under CBDDB, with NO retraining. Only 30 games (SE ~0.8)
but clearly above 100 and within ~3.8 of the >105 target.

Strategic read: under CBDDB, deep search on the transferred Card-A
policy already clears 100. The path to 105 is (better model) + (deep
search). The from-scratch model's bar at champion grade is therefore
101.2, not 99.47 — the from-scratch bet is that CBDDB-native priors
beat transferred Card-A priors by enough to matter on top of search.
Milestone gate reads the cheap-search climb; final comparison is vs
101.2 at n1024/d16. NOTE for John: if from-scratch is slow, the cheaper
route to 105 may be stronger-teacher warm-start + deep search rather
than from-scratch — surfacing for a possible redirect.

Anchor-fix chain fully complete; GPU now FREE for from-scratch bootstrap.

## 2026-07-21 01:22 — From-scratch CBDDB bootstrap LAUNCHED (smoke-verified chain)

Smoke (2 seeds / 50 steps, profile cbddb_fs_smoke) completed the full
chain end-to-end: CBDDB greedy-state EI-0 generation -> filter top-K32
-> relation-tail -> random-init training -> checkpoint + SWA -> pipeline
"completed" marker. Confirmed correct: scoring_cards=cbddb, init_manifest
empty (true random init), model-S, LR 2e-4.

Real bootstrap launched (pid 413745, rev 2b8eea28, profile
cbddb_from_scratch_bootstrap): 300 train + 50 val CBDDB seeds
(2027193000+/2027193500+), greedy_search_bootstrap, objective
search-improved-greedy-retention, model-S, LR 2e-4, 15k steps,
filter top-K32 greedy-prefix-strict, INIT_MANIFEST=none. This is the
random-init foundation the self-play cycles warm-start from.
Next: bootstrap-level eval (n256/d4) to establish the from-scratch
starting point, then cheap n128/d2 self-play cycles toward the
milestone gate (~1 GPU-day cumulative) vs zero-shot 99.4675 (n256/d4)
and ultimately champion-grade 101.2 (n1024/d16). q-quantiles=1 (scalar)
for the greedy bootstrap; the distributional q-quantiles-8 head (suited
to Hawk-D variance) switches on for the gumbel self-play cycles.

## 2026-07-21 11:05 — From-scratch bootstrap TRAINED; ~8h GPU-idle gap (dropped monitor event); eval launched

Bootstrap training completed 02:56 (15k steps, selection_guard_passed
True, locked_val_greedy_top1 0.313, locked_val_total 8.58 from random
init). Checkpoint: full_v3_cbddb_from_scratch_bootstrap/best_locked_val.

PROCESS NOTE (honest): the monitor's "[full-v3] completed" event did not
surface (ssh tail stream dropped over the ~1.5h training phase), so the
GPU sat idle ~02:56→10:58 before I ran the bootstrap eval. ~8 GPU-hours
lost. Lesson: for long single-phase jobs, poll the terminal state at
status checks rather than relying solely on a streaming monitor that can
silently drop. No scientific impact (deterministic artifacts intact).

Bootstrap-level eval launched (n256/d4 x100, block 2027190000-99,
scoring-cards cbddb): the from-scratch STARTING POINT vs zero-shot
99.4675 (n256/d4) and champion-grade 101.2 (n1024/d16). Cheap-cycle
runner prepped (run_cbddb_cycle.sh now defaults GEN n128/d2; cycle 1
warm-starts from the bootstrap, q-quantiles 8 distributional head via
init-skip-mismatched). Cycle 1 fires when the eval frees the GPU.

## 2026-07-21 13:10 — Raw-bootstrap eval abandoned (weak-model OOM/slowness); cycle 1 launched

Raw from-scratch bootstrap eval at n256/d4 OOM'd then ground at 14% GPU
util (~15 min/game, 1/30 seeds) — the near-uniform bootstrap policy
makes Gumbel search fan out over the benchmark's wide --max-actions 64
menus, ballooning batched eval requests. Root cause is the weak model,
not a leak (GPU clean, only Xwayland). John ruled: kill it, characterize
bootstrap as "confirmed weak (expected for EI-0)", go straight to cycle 1.

Cycle 1 (fs_c1) launched: warm-start from the bootstrap, CBDDB self-play
400 train + 40 val seeds (2027194000+/2027194800+) at CHEAP n128/d2,
--max-actions 8 (generation caps menus, so tractable unlike the eval).
Then warm-start train (D1 recipe, q-quantiles 8 distributional head via
init-skip-mismatched) + eval n256/d4 x100 (EVAL_JOBS lowered to 6 to
avoid soft-policy OOM) + n1024/d16 x30. This is the first MEANINGFUL
from-scratch number vs 99.4675 / 101.2.

## 2026-07-22 00:40 — fs_c1 first from-scratch number: 77.85 (n256/d4 x100) — far below bar

Cycle fs_c1 completed gen (400+40 seeds, ~6.6h) + train (666 steps —
the --max-example-passes 4 cap bound the requested 2500 on a ~32k-row
corpus; status pass, best locked_val_final_q_regret 1.565) + screening
eval. Report verified (status pass, ruleset ..._2026_07_19, manifest
cbddb_fs_c1_ft/best_locked_val, rev 48966a16, 100 games).

RESULT: mean seat 77.85 (P50 78, P90 87) vs zero-shot bar 99.4675 and
CBDDB greedy-heuristic floor 80.89. Bootstrap+1 cycle is still BELOW
greedy. Additionally mean_search_seconds 12.1 s/decision — policy still
near-soft, so the n256/d4 x100 eval alone took 4.6h wall.

No preregistered kill fires here (milestone gate is slope-based after
cycle 2), but two flags for John:
1) BUDGET: the in-flight n1024/d16 x30 eval extrapolates to ~20h wall
   (~16x per-decision cost at 12.1 s/dec baseline) — ~0.9 GPU-day for a
   number that is NOT the gate metric (gate slope is measured at
   n256/d4). Killing it requires John's permission (standing rule);
   push notification sent with recommendation to kill and either launch
   cycle 2 directly or rule the gate early.
2) SLOPE REALITY CHECK: to hit 99.47 by cycle 2 the jump would need to
   be +21.6 pts in one cycle. From-scratch curves are steepest early,
   but that magnitude would be unprecedented in this campaign.

GPU now: fs_c1 n1024/d16 x30 eval running (started 00:35). Default
(absent John's ruling): let it run — no process killed without
permission. Cycle 2 blocked behind it either way (one job at a time).

## 2026-07-22 00:55 — John's ruling: n1024/d16 eval on fs_c1 killed (~18 min in); cycle runner fixed

John (on the 77.85 result + eval-cost flag): "why run this expensive
eval on something we know to be low" — treated as the kill ruling. The
eval tree (cycle script 483306, benchmark 566416, exporter 566417 +
bridge) was SIGTERM'd by explicit PID ~00:53, GPU verified clean (no
compute apps). ~18 min spent of a ~20h extrapolated run.

Root cause was mine: run_cbddb_cycle.sh ran the n1024/d16 x30 eval
unconditionally every cycle, which is only meaningful near the
milestone/ultimate bars. Fixed: EVAL_N1024_GAMES now defaults to 0
(skipped, logged) and is opt-in for milestone/pre-certification cycles.

fs_c1 is otherwise COMPLETE for gate purposes: the n256/d4 x100 screen
(77.85) is the gate metric. GPU idle pending John's direction on
cycle 2 vs pivot to the stronger-teacher warm-start fallback.

## 2026-07-22 01:10 — fs_c1 77.85 INVALIDATED as a from-scratch measurement: model-size mismatch bug

Investigating John's "why did this do so much worse": the cycle runner
hardcoded --model-size M (copied from the D1 recipe) while the
from-scratch bootstrap is model S (15.0M params vs 88.2M). Under
--init-skip-mismatched the trainer skipped EVERY tensor (confirmed in
cbddb_fs_c1_train.log: the skip list covers the whole network). fs_c1
therefore trained a RANDOM-INIT 88M model for 666 steps on 32k rows —
the bootstrap contributed nothing. 77.85 measures "random M + 666 steps
+ n256/d4 search", not the from-scratch arc. The n128/d2 GENERATION was
correct (bootstrap model generated it); the corpus remains valid.

Fix: run_cbddb_cycle.sh now REQUIRES MODEL_SIZE (no cross-lineage
default) and threads it into the trainer. Lesson (recurring): 
--init-skip-mismatched converts architecture mismatches into silent
science bugs; any warm-start run must assert the skip list is small
(ideally just the q head) before trusting downstream numbers.

True cycle-1 datapoint (if wanted) is cheap: retrain at MODEL_SIZE=S on
the existing fs_c1 corpus (~15 min) + n256/d4 x100 eval (~4-5h at
soft-policy speed) ≈ ~0.2 GPU-day. GPU idle pending John's direction.

## 2026-07-22 01:06 — fs_c1s launched: corrected model-S cycle 1 (John: "try model S overnight")

Relaunched cycle 1 with the fixed runner under tag fs_c1s: MODEL_SIZE=S
(matches the 15M bootstrap), corpus REUSED via symlinks (no regen —
same 400+40 seed n128/d2 corpus, still valid), warm-start verified this
time: trainer skip list is exactly ['q_head.bias','q_head.weight'] (the
intended scalar->8-quantile swap); every other tensor loaded from the
bootstrap. Eval: n256/d4 x100 on block 2027190000-99; n1024/d16 skipped
(now opt-in). Provenance rev 01000bce (script-only delta vs deployed
48966a16; no Rust/py changes). ETA: train ~15 min, eval ~4-5h at
soft-policy speed -> true from-scratch cycle-1 number by ~05:30-06:00.
This is the honest datapoint 77.85 failed to be; gate math unchanged
(bar 99.4675 at n256/d4).

## 2026-07-22 01:30 — PREREGISTERED decision rule for fs_c1s + X1 draft (autonomy grant)

John: "agreed, i trust your judgement to assess the next experiment
after this autonomously." Locking the decision rule BEFORE the fs_c1s
number exists, to keep the autonomous call honest:

- fs_c1s screen (n256/d4 x100, block 2027190000-99) >= 95.0 -> the
  from-scratch line has a plausible one-more-cycle path; run cycle 2 at
  MODEL_SIZE=S, fresh seeds 2027196000x400 + 2027196400x40, same cheap
  n128/d2 gen. (Rationale: needs < +4.5 to reach the 99.4675 bar.)
- fs_c1s < 95.0 -> from-scratch line CLOSED (weak-teacher math
  confirmed; reaching the bar would need an unprecedented jump), and X1
  launches instead. No middle ground, no re-litigation.

X1 DRAFT — stronger-teacher distillation (the one un-killed warm-start
shape): teacher = AAAAA champion cycle4 under CBDDB (zero-shot 101.2 at
n1024/d16); generate a fresh CBDDB corpus with DEEPER search than the
n256/d4 student will use; fine-tune the champion (model M, q-quantiles
8) on those labels WITH the trust-region anchor (policy-KL + value-L2,
weights per test suite defaults); screen n256/d4 x100 vs 99.4675;
escalate to n1024/d16 x30 vs 101.2 only if screen >= 100.5.
Certification (>105 claim) only on fresh block 2027195000+, John-visible.
Teacher-budget options (choose at launch on remaining-envelope math;
champion CBDDB gen measured ~120 s/seed at n256/d4):
  (a) n512/d8, 400+40 seeds ~ 2.3 GPU-days (strongest labels)
  (b) n512/d8, 260+26 seeds ~ 1.5 GPU-days
  (c) n512/d4, 400+40 seeds ~ 1.2 GPU-days (weakest acceptable teacher)
Optional add-on: --gumbel-exact-endgame-turns 2 for exact late-ply
labels (the long-queued exact-late-labels idea) if CPU cost stays <10%
of gen wall. Seeds: X1 takes 2027196000-6439 if the from-scratch line
closes (cycle 2 then never runs); a later experiment would take
2027197000+. Fresh-block discipline unchanged.

## 2026-07-22 06:15 — fs_c1s RESULT: 98.3 (n256/d4 x100) — rule fires >=95 branch; cycle 2 LAUNCHED

True from-scratch cycle-1 number (corrected model-S warm start):
mean seat 98.3, P50 99, P90 104 on block 2027190000-99. Report
verified (status pass, ruleset ..._2026_07_19, manifest
cbddb_fs_c1s_ft/best_locked_val, rev 01000bce, 100 games,
mean_search_seconds 12.7). Context: 15.0M-param model S, greedy+EI-0
bootstrap + ONE cheap n128/d2 cycle, ~0.7 GPU-days cumulative — vs
88M champion zero-shot 99.4675 on the same block. The S->M bug was
masking a 20.45-point difference (77.85 invalid vs 98.3 real).
Implication vs the weak-teacher hypothesis: search amplification at
n128/d2 + distributional q head extracts far more than the greedy
label floor predicted. P90 already 104 at screen budget.

PREREGISTERED RULE (2026-07-22 01:30): 98.3 >= 95.0 -> cycle 2 at
model S. LAUNCHED 06:13 under the autonomy grant: CYCLE_TAG=fs_c2,
INCUMBENT=cbddb_fs_c1s_ft/best_locked_val, fresh seeds 2027196000x400
+ 2027196400x40 (as preregistered), n128/d2 gen, n256/d4 x100 screen,
n1024 skipped. Skip-list check due at TRAIN start: incumbent now has
the 8-quantile head, so the expected skip list is EMPTY. ETA: gen ~6h,
train ~15 min, eval ~5h -> fs_c2 number ~17:30. MILESTONE GATE then:
slope (98.3 -> fs_c2) decides continue-vs-pivot; bar remains 99.4675
screen / 101.2 ultimate; >105 cert only on 2027195000+, John-visible.
X1 stronger-teacher draft stays on the shelf unless the slope dies.

## 2026-07-22 09:15 — Capacity: john0 gen concurrency doubled for cycle 3+ (John: "definitely optimize (1)")

Measured during fs_c2 gen at old defaults (model-sessions 12, rayon
16): GPU 12% util (RTX 5090, 3.9/32.6 GB), CPU 37% of 32 cores —
concurrency-limited, not compute-limited. run_cbddb_cycle.sh defaults
now JOBS=24, GEN_RAYON_THREADS=28 (new env). EVAL_JOBS stays 6 on
purpose: screen-series comparability — batch composition varies with
job count and per-seed float invariance is unproven; queued micro-test
(same manifest, 10 seeds, jobs 6 vs 12, assert identical finals) gates
any change. NOT deployed to john0 yet — fs_c2 is executing the
deployed script and bash reads scripts incrementally; deploy happens
at cycle-3 launch after CYCLE fs_c2 COMPLETE. Cluster fan-out review
(john1-4) running separately.

## 2026-07-22 11:05 — Fleet fan-out: john2-4 provisioned + parameterized fleet tooling (John: reuse john1-4 infra)

Infra review findings (full report relayed to John): distributed
self-play across the minis was already done in the AAAAA era (fleet3/
4/5) via hand-edited scripts + native MPS bridge; the Bacalhau fabric
is real but CPU-only linux containers — wrong tool for this workload;
trainer natively takes comma-separated multi-shard --train, so no
merge tool is needed.

Done today (all CPU-side, zero contact with john0's running fs_c2):
- john2/3/4: current source (rev dae9fe14) rsynced, exporter rebuilt
  natively (Apple M4, 10 cores each; torch 2.12.1 MPS verified). Old
  binaries were Jul 9 pre-CBDDB.
- NEW scripts (parameterized successors of fleet5_gen.sh):
  fleet_cbddb_gen.sh (per-host MPS shard gen, same gen args as
  run_cbddb_cycle.sh), fleet_cbddb_launch.sh (seed-range allocation,
  incumbent rsync, detached launch + pid files, fleet ledger json,
  double-launch guard), fleet_cbddb_collect.sh (status/collect,
  ledger-vs-manifest seed-range verification, push to john0).
- Smoke in flight: 2 seeds @ SCRATCH range 2027300000 (discard-only,
  recorded here as burned) on john2 with the fs_c1s incumbent —
  validates build/bridge/artifacts and measures MPS s/seed.
- john1: reachable via Tailscale (user john1, john0_codex key; ssh
  config entry added) but now in sshd per-source auth-penalty cooldown
  after my key-probe burst; retry later. Optional host (web-UI
  contention, INFRASTRUCTURE.md).

## 2026-07-22 11:20 — Fleet smoke PASSED end-to-end; 36-seed calibration running

fleetsmoke (john2, 2 seeds @ 2027300000, fs_c1s incumbent, MPS):
GEN 11:01:44 -> 11:11:36, 160 records, zero skipped seeds; collect
path verified seed-domain-vs-ledger, ruleset id, and npz sha256 (a
manifest-schema fix landed in fleet_cbddb_collect.sh: seed range is
parsed from metadata.seed_domain, and the manifest checksum is now
enforced). Shard pushed to john0 fixtures (namespaced fleetsmoke_*,
delete at cycle-3 deploy).

2-seed timing (296 s/seed) is NOT representative — only 2 of 6
sessions occupied + bridge warmup. Calibration launched: fleetcal,
12 seeds/host on john2/3/4 simultaneously (scratch seeds
2027300100-2027300135, burned), measures steady-state MPS throughput
per host for cycle-3 corpus sizing. Scratch ranges burned so far:
2027300000-01 (smoke), 2027300100-135 (cal).

## 2026-07-22 12:25 — Fleet calibration: MPS is bridge-latency bound (~312 s/seed); session tuning in flight

fleetcal (12 seeds/host, SESSIONS=6, RAYON=8, all three minis
simultaneously): john2 62.4 min, john3 64.7, john4 62.3 -> ~312
s/seed/host, statistically identical to the 2-seed smoke (296) —
raising in-flight games 2->6 bought nothing. Diagnosis: the shared
single-process MPS bridge serializes model evals; games are
latency-bound on inference, not CPU-bound (M4 GPU small batches).
At this rate john2+3+4 combined = ~0.0096 seeds/s ≈ 25-30% of john0
after its concurrency doubling — usable but modest.

Tuning probe launched (one variable, two hosts in parallel):
fleetcal2a = john2 SESSIONS=12 (seeds 2027300200-211 scratch),
fleetcal2b = john3 SESSIONS=9 (seeds 2027300300-311 scratch). If
deeper request queues batch better on MPS, throughput should rise
measurably; otherwise fleet stays a ~25% top-up and cycle-3 sizing
uses john0-dominant math. Scratch ledger now: 2027300000-01,
2027300100-135, 2027300200-211, 2027300300-311 (all burn-only).

## 2026-07-22 13:40 — Session tuning NULL; speculative cycle-3 fleet shards launched (pipelining)

fleetcal2a (john2, SESSIONS=12): 294 s/seed; fleetcal2b (john3,
SESSIONS=9): 352 s/seed; baseline (SESSIONS=6): 312/323. Verdict:
session count is noise — ~300 s/seed/host is the MPS bridge floor.
Fleet = ~+30% of john0's post-doubling throughput. Default stays
SESSIONS=6.

PIPELINING DECISION (autonomy grant + "utilize our capacity"): the
minis would sit idle during fs_c2's 5h eval, which is exactly the time
to pre-generate cycle-3 fleet shards from the fs_c2 checkpoint (train
completed 12:45; eval does not alter it). Launched 13:40: CYCLE_TAG=
fs_c3, seeds 2027198000x60 john2 / 2027198060x60 john3 / 2027198120x60
john4 (FRESH block 2027198000-179, now spent), incumbent
cbddb_fs_c2_ft/best_locked_val, n128/d2. ETA ~18:40.
- Gate PASSES -> cycle 3 trains on john0's 400-seed shard (fresh block
  2027197000+400 / val 2027197600+40, reserved here) + these 180 fleet
  seeds: ~45% larger corpus, near-zero added wall clock.
- Gate FAILS -> shards discarded; cost = idle mini time + a burned
  block (logged, no reuse).
MPS-vs-CUDA distribution mix in a train corpus follows fleet3/4/5
precedent (AAAAA era). Evals remain john0-only, unchanged.

## 2026-07-22 15:31 — AAAAA pure-wildlife exact-solver development and proof preregistration

John asked for the maximum AAAAA wildlife-only score with exactly 20 animals,
no other game mechanics, arbitrary species choice, and a cap of six per
species. This is a local CPU combinatorial analysis; it does not touch john0,
the fleet, the active CBDDB campaign, rules identity, or any live process.

Development evidence before the exact run: a deterministic Rust simulated-
annealing search found a connected 68-point incumbent with counts B/E/S/H/F =
6/4/6/0/4 and breakdown 19/13/20/0/16. The production
`cascadia_game::score_board(..., ScoringCards::AAAAA)` scorer reproduced that
breakdown. Earlier cell-indexed HiGHS and CP-SAT formulations were discarded
as too symmetric; they produced no verdict and their unshipped source was
removed. A 20-labeled-token CP-SAT replacement independently found a
68-point witness for the same count allocation and rejected the highest
count-only relaxation allocation (6/1/6/1/6, bound 73) at threshold 69.

Preregistered exact decision rule before the final durable proof: enumerate
all 826 legal count vectors; eliminate only vectors whose standalone
count-only score upper bound is <69; solve every remaining vector at exact
score threshold >=69. Declare 68 optimal iff all 128 surviving allocations
return `INFEASIBLE`, the bundled 68 witness passes the independent Python
scorer and production Rust scorer, and all solver/test commands pass. Any
`UNKNOWN` allocation is retried unchanged with a longer per-allocation limit;
it is never treated as evidence. A feasible >=69 witness invalidates the
68 claim and must be production-scored before use.

Frozen final-run configuration: `tools/aaaaa_wildlife_exact.py`, labeled-token
CP-SAT model v1, OR-Tools 9.15.6755, 20 distinct connected axial coordinates,
global radius 19 around the lexicographically first fox, per-species token
ordering, exact AAAAA scoring witnesses, 8 workers, base seed 20260722 plus
allocation index, 120 seconds per allocation. Durable output:
`docs/v3/evidence/aaaaa_wildlife_optimum_2026-07-22.json`; source and artifact
SHA-256 values, wall result, and decision will be appended when complete.

**15:48 FINAL VERDICT — 68 PROVED OPTIMAL.** All 128/128 allocations with
count-only upper bound >=69 returned `INFEASIBLE`; there were zero feasible,
unknown, invalid, or timed-out allocations. Aggregate CP-SAT solver time was
462.741265 s; the slowest allocation took 56.684007 s, inside the frozen
120-second limit. The emitted optimum has counts 6/4/6/0/4 and wildlife
breakdown 19/13/20/0/16. Independent Python scoring and the production Rust
AAAAA scorer both return 68; the layout is connected and all counts are <=6.

Provenance: base revision `2ccc27c155b15040dde7a22a74d683e1cb7dc57f` plus
model source `tools/aaaaa_wildlife_exact.py` SHA-256
`39f1e503788df285615d27fa9dbeb9a0f49ba86c01aafcf79d374e0bb2130fd9` and
production verifier `crates/cascadia-game/src/bin/aaaaa_wildlife_solver.rs`
SHA-256
`39cdf1220e515b97bf72c023aaa902e49751a040c2466896e475c5efab13ddd2`.
Durable proof ledger SHA-256:
`163c338643fd7c14d45eb00fa1b833b4043260cef47effe14816787e124e3828`.
Decision: publish 68 as the exact optimum for the stated connected-20-token
model; this is a side analysis and does not change the active CBDDB campaign.

Validation after verdict: `cargo check --workspace` PASS; `cargo test
--workspace` PASS (280 tests, one timing harness ignored); exporter PASS
(80/80); v3 Python PASS under the project Python 3.12 environment with the
documented Torch 2.12.1 runtime (408 tests, 45 artifact-dependent skips);
cluster/tooling pytest PASS (109/109); exact-solver Python tests PASS (4/4);
Ruff format/check PASS. The bare Homebrew Python 3.14 attempt lacked NumPy and
Torch and was environment-invalid, so it contributed no code verdict.

## 2026-07-22 17:31 — AAAAA all-compositions solver performance preregistration

John requested one exact optimal board for every legal 20-animal count vector
under the six-per-species cap, and explicitly required a performance pass
aiming for 20x before launching the catalog search. This is a local CPU
engineering/analysis run. It will not touch, inspect, interrupt, or compete
with any live john0/fleet scientific process.

The fixed cap-six catalog contains 826 vectors. The pre-optimization
labeled-token formulation was benchmarked with OR-Tools 9.15.6755, eight
workers, random seed 20260722, and the same exact Python scorer used by the
published proof. Frozen exact benchmark panel and measured baselines:

- maximize `(6,6,6,2,0)`: `OPTIMAL`, score 62, 2.4168 s;
- maximize `(4,4,4,4,4)`: `OPTIMAL`, score 65, 8.4992 s;
- maximize `(6,1,6,1,6)`: `OPTIMAL`, score 67, 44.6858 s;
- feasibility threshold >=69 for `(4,2,6,2,6)`: `INFEASIBLE`, 56.684007 s
  in the hash-pinned v1 proof ledger.

The frozen aggregate baseline is 112.285807 wall-seconds. The aspirational
20x gate is <=5.614290 s on the same machine and worker count, with identical
scores/statuses and every emitted witness independently rescored. We will
report the achieved ratio honestly even if it misses 20x. Candidate changes
may reduce redundant SAT encodings, add sound symmetry breaking and bounds,
or parallelize independent count vectors; no approximation, timeout-as-proof,
or scorer substitution is allowed. Any panel member returning `UNKNOWN`, a
different optimum/status, or an independently invalid witness fails the
candidate regardless of speed.

Only after the selected formulation passes exact unit/regression tests will
the 826-vector production run begin. The production runner must be durable and
resumable, hash-pin its source/config, never publish a partial ledger as
complete, require `OPTIMAL` for every vector, verify every witness with an
independent scorer, and emit both a machine-readable ledger and a documented
Markdown catalog. The cap-eight comparison is analytic and fixed at 3,951
vectors, 4.783293x the cap-six catalog.

## 2026-07-22 17:40 — MILESTONE GATE FAILED: fs_c2 = 98.3375 (slope +0.04). From-scratch CLOSED; X1 LAUNCHED

fs_c2 screen (verified: status pass, ruleset ..._2026_07_19, manifest
cbddb_fs_c2_ft/best_locked_val, 100 games): mean 98.3375, P50 98,
P90 104. Slope vs fs_c1s (98.3) = +0.0375/cycle -> ~30 cycles to the
99.4675 bar: does NOT project. Preregistered gate FAILS. FROM-SCRATCH
LINE CLOSED at its plateau — the same-budget self-play ceiling, 4th
sighting this campaign (D1, s2 triple-98.75, and now fs at 98.3x2).
The 6.4x locked-val q-regret jump (1.44 -> 9.24) reads in hindsight
as the plateau's signature. Positive residue: a 15M CBDDB-native
model at 98.3 for ~1.4 GPU-days, and proof that Card-A priors are NOT
required to reach 98+ under CBDDB.

Speculative fs_c3 fleet shards (2027198000-179): DISCARDED per prereg;
mini jobs left to run out (finish ~19:05/~20:55, no contention, no
kill needed). Block logged as burned. Reserved-but-unused cycle-3
block 2027197000-7640 returns to the fresh pool.

X1 LAUNCHED 17:36 (autonomy grant; prereg draft 2026-07-22 01:30,
option (a)-sized): teacher = AAAAA champion cycle4 generating CBDDB
corpus at n512/d8 (deeper than the student's n256/d4 eval budget),
seeds 2027199000x300 + 2027199400x30 (fresh, now spent); then distill
INTO the champion (model M, q-quantiles 8 + init-skip-mismatched —
skip list must be q-head only) with trust-region anchor kl=2.0
l2=2.0 (the exact arm recipe from the anchor battery), max-example-
passes 8 (distillation on strong labels; locked-val selection + SWA
guard overfit; deliberate departure from the self-play 4-pass cap,
logged here). Eval: screen n256/d4 x100 on 2027190000-99 vs 99.4675,
AUTO-ESCALATE to n1024/d16 x30 iff screen >= 100.5. New concurrency
live (JOBS=24 RAYON=28, first use). Script: run_cbddb_x1.sh (deployed
with the updated run_cbddb_cycle.sh after fs_c2 completed). Rough ETA:
gen ~20-26h, train ~40 min, screen ~3-5h -> verdict tomorrow evening.
Success = screen > 99.4675 (beats zero-shot); real target = full
n1024/d16 > 101.2. GPU budget note: CBDDB spend ~3.1 GPU-days through
fs_c2; X1 adds ~1.2-1.4 — within the "few-GPU-day" pivot envelope
John acknowledged when approving the stronger-teacher direction.

## 2026-07-22 19:36 — AAAAA all-compositions optimization verdict and production preregistration

The performance pass completed before the 826-vector exact launch. Candidate
1 (compact reified adjacency tables + rooted arborescence) preserved the
frozen answers but reduced the four-case panel only from 112.285807 s to
107.762945 s = 1.042x. Candidate 2 (12-way dihedral anchor, exact depth steps,
coordinate-depth propagation, explicit degree bounds) regressed to 258.725970
s = 0.434x and was removed. Token-level common-neighbor cuts also regressed
the hard disconnected proof from 24.340998 s to 28.206209 s and were removed.
No rejected formulation remains in production source.

Selected exact changes: two compact reified tables make every pair adjacency
exact; elk line witnesses use coordinate-ordered combinations rather than up
to 24 equivalent permutations; aggregate fox overlap cuts expose the lattice
facts that two distinct target hexes have at most two common neighbors and
three or more have at most one; a disconnected relaxation is attempted before
the connected proof; and exact optimization is replaced with successive
incumbent+1 feasibility questions. The top `(6,1,6,1,6)` count vector's >=68
disconnected impossibility proof fell from 43.544487 s without fox overlap
cuts to 24.340998 s with them. End-to-end from a 65-point Rust incumbent, that
composition was improved and certified at 67 in 39.610 s. Ten one-point-gap
compositions certified in 38.70 s total (3.87 s/vector).

The requested aspirational 20x gate was not reached on the frozen worst-case
panel; the hard top case is 1.13x faster end-to-end than its old 44.6858 s
direct maximization. The catalog throughput design nevertheless removes exact
search entirely where possible: an 8-thread, 8-restart x 20,000-iteration
Rust calibration searched all 826 counts in 35.318636 s, independently
production-scored every connected board, and matched the elementary count
upper bound for 271 vectors. Those 271 receive immediate exact certificates;
the remaining queue is ordered by incumbent gap and uses one 8-worker solver,
which beat two contending 4-worker solver processes on this 10-core Mac. A
10-worker hard proof took 30.654988 s versus 24.340998 s at 8 workers, so 8 is
frozen.

Preregistered production run: release Rust candidate generator, 8 threads,
8 restarts/count, 20,000 iterations/restart, seed 20260722; exact catalog
runner jobs=1, CP-SAT workers=8, disconnected-relaxation limit=60 s,
connected limit=120 s, base seed=20260722. An `UNKNOWN` is recorded incomplete
and is never evidence; after the first pass, incomplete vectors are resumed
unchanged with longer limits. Completion requires all 826 results marked
proof-complete, independent Python rescoring, connected/count checks, and all
826 boards passing the production Rust AAAAA scorer. Durable artifacts:
`docs/v3/evidence/aaaaa_wildlife_candidates_2026-07-22.json`,
`docs/v3/evidence/aaaaa_wildlife_catalog_2026-07-22.json`, and
`docs/v3/AAAAA_WILDLIFE_CATALOG.md`. Frozen pre-launch source SHA-256:
exact model `594d52ec6c82f9aa644eb0aadbf35654c541ac8a5a5c5cab53cb89b90858688b`;
catalog runner `54d2eb2ba60e8fb0494d2b9d658fc533ec3a3dcb24efa30c098631163a688cc4`;
Rust candidate/production verifier
`c4991663cd34b9ba7fed34e04e9c2f62d3cadf1eae1e38c2af420d0065ea84d3`.

**19:38 production candidate stage complete.** All 826 connected boards were
generated in 34.643270 s and independently checked by both the Rust reference
scorer and production `score_board` path. Exactly 271 match the count-only
upper bound. Candidate artifact SHA-256:
`82340dcc952d187731650e146c9de9bf0d5b0bf9557d28ccee3ad84b9f6a6842`.
Decision: launch the preregistered exact catalog pass; no candidate result was
used to change its proof thresholds or configuration.

## 2026-07-23 04:20 — CBDDB all-compositions exact-solver development and bounded diagnostic

John extended the pure-wildlife catalog request: after the max-six AAAAA
catalog is complete, produce the same one-optimum-per-count catalog for CBDDB
(Bear C, Elk B, Salmon D, Hawk D, Fox B), retain all 826 configurations, and
report the holistic highest scoring board. This remains local CPU side
analysis and does not touch the live john0/fleet campaign.

Implemented an independent Python CBDDB scorer and labeled-token CP-SAT model,
plus a Rust annealing candidate generator/custom scorer and production
`score_board(..., ScoringCards::CBDDB)` verifier. The exact model represents
Bear-C full components and set bonus, Elk-B disjoint singles/pairs/triangles/
strict rhombi, Salmon-D valid runs and distinct adjacent non-salmon tokens,
Hawk-D line-of-sight maximum-weight matching, and Fox-B species appearing at
least twice around each fox. Python model-vs-oracle tests pass on the card edge
cases and varied fixed 20-token boards. Rust custom-vs-production checks pass
on varied connected boards and the count space is pinned at 826.

A bounded, explicitly non-verdict diagnostic ran the highest count-only
relaxation `(0,2,6,6,6)` for 10 seconds with two CP-SAT workers and seed
20260723. It returned `FEASIBLE`, model/independent score 39 with breakdown
0/5/16/12/6, while the deliberately loose count bound remained 100. No
optimality conclusion follows. Decision: the production sequence remains
AAAAA completion first; then generate strong Rust CBDDB incumbents before any
exact catalog launch. Timeout or `UNKNOWN` will never be accepted as proof.
Development source SHA-256: Python exact model
`362b5d7f82a156579e33c4b2c630c06bff3f45fa08f72a4dc70fe378eadca329`,
Python catalog runner
`7e93396d7e5d3efe551917fe5805b169b50b988f2e5f3ccd1a441414290c6cde`,
Rust candidate/verifier
`8181ddf434fdde8309bca619d51923b927408a6167614ab0cd9573a12094fe6d`,
and shared Rust support
`110dc92dcc95b5d0effbcfe17d6ac124ff2f216c6716ae860d496abe77956af9`.

## 2026-07-23 05:10 — AAAAA catalog first pass complete; deeper incumbent pass preregistered

The preregistered AAAAA all-compositions first pass exited normally with the
runner's incomplete status (exit 2). All 826 count vectors were attempted:
710 are formally certified and 116 are retained as incomplete, never as
proof. Certificate mix: 332 witnesses exactly matched the count relaxation,
344 disconnected relaxations proved no improvement, and 34 connected models
proved no improvement. Aggregate per-task solver wall time was 34,255.382497
seconds; slowest task 355.974106 seconds. Frozen first-pass ledger SHA-256
`9aeb528830e95fed89c7d5bcc26c9b87654f534daf2044f5a297f8b54480df1f`;
partial Markdown SHA-256
`f979561013c03279776bc8a3bf0f249ac0e9c3363fe8fa9e7f611ddf2cb356ed`;
durable log SHA-256
`ae5ef6639b498e1d36e3971320c83fd3aa5dea1c7477bb322ae0cea153e510c2`.
The frozen copies are
`docs/v3/evidence/aaaaa_wildlife_catalog_first_pass_2026-07-23.{json,md}`.

Performance diagnosis before retry: the v1 resume path would discard every
improved incumbent found inside an incomplete CP attempt and reload only the
shallower Rust board. That would repeat work. The next pass is therefore
strictly incumbent generation, not proof: release binary
`target/release/aaaaa_wildlife_solver` SHA-256
`fcf619f30ec21fa7d2b5a587d7ba44ca41976189f2074c501e91d4c3f223b207`,
8 threads, 32 restarts/count, 100,000 iterations/restart (20x the original
annealing effort), seed 20260723. Output:
`docs/v3/evidence/aaaaa_wildlife_candidates_deep_2026-07-23.json`; log:
`cascadiav3/logs/aaaaa_wildlife_candidates_deep_2026-07-23.log`.

Decision rule fixed before launch: for each count, retain the highest
independently rescored board across the original candidates, deep candidates,
and first-pass CP incumbent. Preserve the 710 completed certificates with
their original source/artifact provenance. Retry only the remaining counts,
using the retained board as a CP coordinate hint. A timeout remains
incomplete; no score or proof threshold is weakened.

The real 826-row v1→v2 migration smoke passed before production use: 710
completed proofs imported, all witnesses independently rescored, the frozen
ledger hash retained, and zero retry tasks submitted under `--limit 0`.
Frozen retry source SHA-256: exact model
`55619db79bd14c9f4935fbf3cad631ef78cf8b246fa770f4b02ddb8bdda309a8`;
catalog runner
`d291ff58b8a17af07bb47f83c67dbfdef34b68290ded7aa50d67cf064b3cbd6b`.

**05:25 deep incumbent result and exact retry launch.** The 20x candidate pass
completed all 826 vectors in 596.631632 seconds. Every board passed the
independent Python scorer after the Rust generator had already asserted its
custom and production scores. Relative to the original Rust pass, 550 scores
were unchanged, 252 improved by one, and 24 improved by two. On the 116
deferred vectors, the deep board beat the already-improved first-pass CP
incumbent in only six cases; none reached its effective exact upper bound.
Candidate artifact SHA-256
`d8c4f1ce9d9b7decac3156c6500b9e35407d70c74573877529666ba02ff496ae`;
log SHA-256
`57daace7c00f2299701197e18760918a2bc97e6a48c0dec1e3c9ec57dbfd0e24`.

Preregistered retry pass: import the frozen first-pass ledger, merge each
incumbent by independent score, preserve its 710 completed proofs with
per-result provenance, and submit only the 116 remaining vectors. One job,
eight CP-SAT workers, disconnected limit 60 seconds, connected limit 120
seconds, base seed 20260723, coordinate hints enabled, and the independently
proved global score ceiling 68 imposed as an additional upper bound. Durable
output remains `docs/v3/evidence/aaaaa_wildlife_catalog_2026-07-22.json` and
Markdown `docs/v3/AAAAA_WILDLIFE_CATALOG.md`; retry log
`cascadiav3/logs/aaaaa_wildlife_catalog_retry_2026-07-23.log`. Decision rule:
only a witness hitting the effective upper bound or an exact `INFEASIBLE`
answer closes a vector; every `UNKNOWN` remains incomplete for a later,
separately logged longer pass.

## 2026-07-23 05:50 — AAAAA deterministic-connectivity calibration preregistration

The first six hinted retry cases (all one-point gaps) returned `UNKNOWN` after
their fixed 60-second disconnected + 120-second connected attempts. This
confirms incumbent search is no longer the limiting phase. A separate exact
variant replaces the connected model's arbitrary rooted spanning tree with
shortest-path fixed-point equations. Every connected graph has exactly its
root-distance assignment; any disconnected component yields the contradiction
`minimum_depth = minimum_depth + 1`. Fixed connected/disconnected board tests
pass.

Before calibration output: use the already-completed `(3,6,6,0,5)` case at
threshold 62, the retained 61-point board as a coordinate hint, connected
model only, two otherwise-idle workers, 60-second limit, seed 20260723. The
active eight-worker retry remains untouched. Declare the variant promising
only if it returns exact `INFEASIBLE` or materially reduces the unresolved
bound within 60 seconds; `UNKNOWN` is no proof. Wrapper source SHA-256
`0c65743887a046c56c63d91428a4d86d6b5bc76a6878c9a15e2234ec803915d0`;
imported scoring/coordinate model SHA-256
`55619db79bd14c9f4935fbf3cad631ef78cf8b246fa770f4b02ddb8bdda309a8`.
Durable output:
`docs/v3/evidence/aaaaa_distance_connectivity_calibration_2026-07-23.json`.

**Calibration result — NOT SELECTED.** The deterministic-connectivity variant
returned `UNKNOWN` after 60.055958 seconds, with no witness, 271,183 branches,
and 55,246 conflicts. It therefore failed the preregistered promising gate;
the sound formulation remains tested code but is not used as proof evidence
or substituted into the active catalog. Artifact SHA-256
`fb171f2c486d1abd42c0503d0de908285f259ae5aeb29bea2841465a8f90039c`.
Decision: connectivity-tree symmetry alone is not the tail's root cause; the
next optimization must expose high-fox adjacency/coverage structure directly.

## 2026-07-23 06:00 — AAAAA eager-score channeling calibration preregistration

The high-fox feasibility model currently permits true deterministic score
features to remain false, because one-way implications are sufficient for
logical completeness. That creates many score-variable assignments for the
same coordinates. A separate wrapper adds the sound reverse implications:
Fox-A distinct-species flags equal the OR of relevant adjacency edges,
Hawk-A isolated flags equal the absence of hawk edges, and Bear-A pair flags
equal full two-bear components. The known 68 board remains exactly feasible;
wrapper source SHA-256
`86114e79a0d48fdd6261a5dd46be94a874f3f14bc5a668d85a7a585db6271d3a`.

Before output, repeat the fixed `(3,6,6,0,5)` threshold-62 connected case,
retained hint, two idle workers, 60 seconds, seed 20260723. Select only on
exact `INFEASIBLE` or a material search reduction versus the rejected
deterministic-connectivity calibration; `UNKNOWN` remains no evidence.
Durable output:
`docs/v3/evidence/aaaaa_eager_score_calibration_2026-07-23.json`.

**Calibration result — NOT SELECTED.** The eager-channeling model returned
`UNKNOWN` after 60.043956 seconds with no witness, 346,984 branches, and
69,402 conflicts. Although it processed more search nodes than the
deterministic-distance variant, it made no exact or bound progress and fails
the preregistered selection rule. Artifact SHA-256
`e71687f5888596866ee2326c6c95703f5e672ad3db24783f024224b04a164a97`.
Decision: preserve the tested sound wrapper, but do not use it as catalog
evidence; direct fox-layout structure remains necessary.

## 2026-07-23 06:10 — AAAAA realizable fox-graph calibration preregistration

Enumerated every connected polyhex adjacency graph through six foxes (counts
by size 2..6: 1, 2, 4, 8, 22 graph types). A labeled fox adjacency graph is
admitted iff every connected component is one of those exact unit-hex graph
types; components may remain arbitrarily separated, so the table excludes no
legal layout. The score threshold also gives a sound minimum number of
nonisolated foxes. For the fixed `(3,6,6,0,5)` threshold-62 case, all five
foxes must be nonisolated and the table admits 475 of 1,024 labeled graphs.
Known-board and count-bound tests pass. Source SHA-256
`2559cff7099ee2661d40def8f32a17ba1dfe17ece7a7d10597c158df32fd6ef7`.

Before output: repeat the same connected case/hint with two idle workers,
60 seconds, seed 20260723. Select only on exact `INFEASIBLE` or material
search progress versus both rejected variants; `UNKNOWN` is not proof.
Durable output:
`docs/v3/evidence/aaaaa_fox_graph_calibration_2026-07-23.json`.

**Calibration result — NOT SELECTED.** The realizable-graph model returned
`UNKNOWN` after 60.010195 seconds, with no witness, 337,376 branches, and
67,040 conflicts. The 475-row table is valid but did not make exact progress
on the fixed case, so it fails the selection gate. Artifact SHA-256
`e4daac01eede7f9365e3677b3e5f233c580ff969652e5b6a6880466c0b55a33d`.
Decision: retain the tested table implementation but do not mix it into
catalog evidence; fox adjacency must be coupled to per-species coverage and
non-fox scoring motifs to tighten this tail materially.

## 2026-07-23 06:03 — CBDDB catalog-candidate staging preregistration

Stage heuristic incumbents for all 826 legal 20-token count vectors while
the AAAAA exact retry owns eight workers. This is deliberately candidate
generation only: it cannot prove optimality, cannot close a CBDDB catalog
entry, and no CBDDB holistic conclusion will be drawn from it. The exact
CBDDB catalog pass remains sequenced after AAAAA completion as requested.

Run the release Rust solver with two otherwise-idle threads, eight restarts
per count vector, 20,000 mutation iterations per restart, and base seed
20260723. Every emitted board must pass both the solver's custom CBDDB scorer
and the production game scorer; the later Python exact runner will rescore
all imported candidates independently. Durable output:
`docs/v3/evidence/cbddb_wildlife_candidates_2026-07-23.json`; durable log:
`cascadiav3/logs/cbddb_wildlife_candidates_2026-07-23.log`. Release-binary
SHA-256 `37578ee5a379ec290fe04933e9ded9d1283bb9ba2e5743f309b93ab873ad7465`;
solver source SHA-256
`8181ddf434fdde8309bca619d51923b927408a6167614ab0cd9573a12094fe6d`;
shared-support source SHA-256
`110dc92dcc95b5d0effbcfe17d6ac124ff2f216c6716ae860d496abe77956af9`.
Decision rule: retain one highest verified candidate per vector (the seeded
search order resolves ties); completion of all 826 vectors and zero scorer
mismatches are required before the file may seed the exact pass. Any failure
or incomplete run is logged and not promoted to proof evidence.

**Result (06:08) — COMPLETE AS CANDIDATE STAGING, NOT EXACT EVIDENCE.** All
826 distinct legal count vectors completed in 224.244154 seconds. Independent
Python rescoring found zero score mismatches, zero overlaps/disconnections,
and 826 unique vectors. Candidate scores range from 59 to 84; the current
heuristic leader is counts `(6,0,3,6,5)`, breakdown `(18,0,12,27,27)`, total
84. This is an incumbent only, not a CBDDB optimum or upper-bound proof.
Candidate artifact SHA-256
`cd9d22bf0b7d0990c7b5d4daebcb750ce3f8283bdb42809133e8b0cecbb1084e`;
log SHA-256
`e4d0af708983f400173d67c815ab0fb1a6f1b4bb91ea7e0566d4d39679bc90c7`.
Decision: freeze the candidate file as the exact catalog's warm-start input;
do not launch the CBDDB exact catalog until the requested AAAAA exact catalog
is complete.

## 2026-07-23 06:08 — AAAAA retry first new exact closure

The active, untouched eight-worker retry proved counts `(4,4,2,4,6)` have
optimum 66. Its disconnected relaxation was `UNKNOWN` after 60.016976
seconds; the connectivity-required model proved threshold 67 `INFEASIBLE`
in 118.925412 seconds. Retained witness breakdown `(11,13,5,11,26)`.
The durable live ledger is now 711/826 exact, 725/826 boards stored, with 115
vectors still unproved. This is a progress transition inside the already
preregistered retry, not an adaptive run change; the process continues.

## 2026-07-23 06:12 — AAAAA occupied-center bound calibration preregistration

Every connected graph on twenty vertices has an occupied center with graph
eccentricity at most ten, and hex distance is no greater than occupied-graph
distance. A new exact wrapper therefore requires the solver to choose an
occupied token whose axial hex distance to every token is at most ten. The
choice is species-neutral and the extremal twenty-token path remains feasible,
so this removes no legal connected board. The known 68-point witness and the
diameter-19 path tests pass. Wrapper SHA-256
`32aa014a8749bde5a749139b606b71f96c6e205bdaaa94e90ebb940c35cf6310`;
test SHA-256
`94c60d79201d4e6b246970c896318d2ab86bfb26e92db4f41f64173bcde0db60`;
imported exact source SHA-256
`55619db79bd14c9f4935fbf3cad631ef78cf8b246fa770f4b02ddb8bdda309a8`.

Before output, calibrate the same fixed `(3,6,6,0,5)` threshold-62 connected
case and retained 61-point hint used by the three rejected variants, with two
idle workers, 60 seconds, seed 20260723. Durable output:
`docs/v3/evidence/aaaaa_occupied_center_calibration_2026-07-23.json`.
Selection rule: only an exact `INFEASIBLE` result within 60 seconds selects
this wrapper for catalog proof work; a feasible witness is retained if valid
but does not by itself select the proof formulation; `UNKNOWN` rejects it.

**Calibration result — NOT SELECTED.** The occupied-center model returned
`UNKNOWN` after 60.010080 seconds with no witness, 357,186 branches, and
58,124 conflicts. It therefore fails the preregistered exact-result gate.
Artifact SHA-256
`44af675ae4a53ec2c9d492d288da1918e9127905aa8a17c75a3ec7963c6bf3c6`.
Decision: retain the proven-safe constraint and tests, but do not use it as
catalog proof evidence or alter the active retry. The tail requires stronger
card-specific combinatorial bounds rather than another generic connectivity
restriction.

## 2026-07-23 06:16 — AAAAA dihedral lex-leader calibration preregistration

The base model removes translation and same-species label symmetry but still
admits rotations/reflections. A new exact wrapper retains only a representation
whose permutation-invariant species moment is minimal among every one of the
twelve hex-lattice dihedral transforms for which the same anchor remains the
lexicographically first token of its species. Identity is always eligible, so
the finite orbit always has a minimum and no physical board is excluded.
All twelve transforms are distinct and at least one transformed copy of the
known 68-point board passes the fixed-coordinate model. Wrapper SHA-256
`d2f5681852f58475a6a4a9963c8e229408b72dbcb5856923c5000dd6db4fa2ad`;
test SHA-256
`6e5a1ed88c202e811f542f485627c0f35a05b2b82ab3f5eb406c1dc777a64b4d`.

Before output, calibrate the same `(3,6,6,0,5)` threshold-62 connected case,
retained 61-point hint, two idle workers, 60 seconds, seed 20260723. Durable
output: `docs/v3/evidence/aaaaa_dihedral_calibration_2026-07-23.json`.
Selection rule: exact `INFEASIBLE` within 60 seconds selects the wrapper;
otherwise `UNKNOWN` rejects it. Any feasible independently valid witness is
retained but does not alone select a proof formulation.

**Calibration result — NOT SELECTED.** The dihedral lex-leader model returned
`UNKNOWN` after 60.057452 seconds with no witness, 350,219 branches, and
64,465 conflicts. Artifact SHA-256
`5f457f47d590b05513f9c99cb28e1fe6a5fb24267286b77c1bd316e062f351d0`.
It fails the preregistered exact-result gate. Decision: retain the sound
symmetry implementation and tests, but do not substitute it into the catalog;
generic orbit reduction is too small relative to the high-fox score search.

## 2026-07-23 06:20 — AAAAA motif incompatibility exploratory diagnostic

**UNPREREGISTERED DIAGNOSTIC — NOT PROOF EVIDENCE.** To identify the missing
card-specific cut after four generic variants failed, a disposable exhaustive
prototype examined the unique unresolved raw one-point-gap vector
`(3,6,6,0,5)`. A score of 62 equals its standalone count relaxation and hence
would require: one isolated Bear-A pair; an Elk-A 4+2 or 3+3 line packing; one
six-salmon unbranched component; every fox adjacent to another fox; and every
fox adjacent to bear, elk, and salmon.

The prototype enumerated 25 unbranched free six-hex salmon shapes and 4,623
five-fox boundary sets with no isolated fox. It then allowed bear/elk scoring
groups that do not cover a fox to live abstractly outside the local region—a
strict relaxation of the real board—and found 2,342 cases independently able
to attain both bear and elk coverage, but zero with non-overlapping local bear
and elk cells. Runtime was 10.7 seconds. Because this output was viewed before
a formal entry and came from disposable code, it is used only to choose the
next engineering direction and does not close the vector. Decision: build a
reviewable deterministic certificate generator with explicit superset proof,
unit tests, source hash, and durable output; preregister that run before
accepting its conclusion.

## 2026-07-23 06:38 — AAAAA motif certificate preregistration

The reviewed deterministic certificate generator is now frozen at source
SHA-256
`1d67cf1ee1b830b634f67ea8ba550e1ba305692a4ac3c2ea84c9799349ce078a`;
its test source SHA-256 is
`a404bb98b309d671c7276383cc0c7a2e16d1836e54ab89af31c837e5016d702b`.
Ruff passes and four tests pass, including the public free-polyhex counts
`1,1,3,7,22,82`, 25 valid six-salmon shapes, both optimal six-elk partitions,
and full relaxed-superset exhaustion.

Run once against the live AAAAA ledger's retained `(3,6,6,0,5)` incumbent and
write
`docs/v3/evidence/aaaaa_motif_certificate_3_6_6_0_5_2026-07-23.json`.
Accept the certificate only if the incumbent independently scores 61 on 20
distinct connected cells, the enumeration reproduces 82 free size-six
polyhexes / 25 valid salmon shapes / 4,623 fox sets / 2,355 bear-feasible fox
sets / 2,342 independently bear-and-elk-feasible fox sets, and finds exactly
zero non-overlapping relaxed realizations. Under those fixed conditions the
relaxed superset's infeasibility excludes 62, and the witness proves exact
optimum 61. Any mismatch or nonzero realization fails the certificate and
leaves the vector incomplete.

**Result — EXACT CERTIFICATE PASSED.** The frozen generator completed in
8.359806 seconds and reproduced every preregistered count: 82 free size-six
polyhexes, 25 valid salmon shapes, 4,623 fox sets, 2,355 relaxed bear-feasible
sets, 2,342 independently relaxed bear-and-elk-feasible sets, and zero
non-overlapping relaxed realizations. The retained 20-cell connected witness
independently scores `(4,18,20,0,19) = 61`. Therefore score 62 is impossible
and counts `(3,6,6,0,5)` have exact optimum 61. Artifact SHA-256
`236d429cc5f8a7edf29d3f5630523af413ced5f2615225da78bbd2238a01660b`.
Decision: accept this as the 712th AAAAA per-vector proof. Keep it separate
while the active catalog writer runs, then merge it with its distinct proof
method/provenance after that process exits naturally.

## 2026-07-23 06:52 — AAAAA zero-hawk local-packing diagnostic preregistration

Five unresolved zero-hawk vectors remain after the accepted motif certificate:
`(4,4,6,0,6)`, `(2,6,6,0,6)`, `(4,5,5,0,6)`, `(4,6,4,0,6)`, and
`(3,6,5,0,6)`. Their incumbents are respectively 65, 63, 64, 63, and 60.
Before diagnostic output, screen a generalized relaxed local-packing bound:
enumerate every maximum-score unbranched salmon shape; keep foxes that see
salmon locally and represent any permitted salmon-missing foxes abstractly;
enumerate required Bear-A pairs and Elk-A scoring lines; preserve forced local
cell non-overlap while awarding abstract pieces their maximum possible fox
coverage. This remains a superset of real boards.

This is an engineering diagnostic, not proof evidence. Continue to a frozen
audited certificate generator only if the relaxation excludes incumbent+1 for
at least one of the five vectors; otherwise record the loose witness/profile
and close this extension. Do not alter the active catalog from diagnostic
output.

**Diagnostic result — SELECTED FOR FORMAL CERTIFICATION.** The structured
set-packing model exactly optimized every eligible relaxed subcase and excluded
incumbent+1 for three vectors:

- `(3,6,5,0,6)`: 50 subcases, relaxed upper 60, 142.042763 s elapsed;
- `(4,6,4,0,6)`: 20 subcases, relaxed upper 63, 77.084830 s elapsed;
- `(4,5,5,0,6)`: 30 subcases, relaxed upper 64, 198.141487 s elapsed.

Each upper matches the retained incumbent. The other two registered vectors
were deliberately not run: their target permits a six-salmon 5+1 split worth
18, so the model's single maximum-component premise is not forced. They remain
open. Current bound source SHA-256
`ff542766b4248f88626c5bc47d2e08c483513aea4f26d98fa333e201ae13ef8e`;
tests SHA-256
`3724eb36b80c266a701e5877d8ba06e231a5bc28f55d2f3635848006e87874e2`.
Decision: add fail-closed certificate serialization and incumbent validation,
then preregister one frozen three-vector evidence run. Diagnostic output alone
does not yet promote the rows.

## 2026-07-23 07:16 — AAAAA zero-hawk certificates preregistration

Certificate source is frozen at SHA-256
`47d4f3970d0f4144188b4dc4ee7074c13a23254ee8df677ede5deb2cc622d499`;
tests remain
`3724eb36b80c266a701e5877d8ba06e231a5bc28f55d2f3635848006e87874e2`.
Ruff and all four unit tests pass. Run the three fixed cases with one CP-SAT
worker per subcase and a 30-second per-shape limit; durable output:
`docs/v3/evidence/aaaaa_zero_hawk_certificates_2026-07-23.json`.

Accept a row only if every enumerated submodel returns exact `OPTIMAL`, the
relaxed upper is respectively 60 / 63 / 64 for `(3,6,5,0,6)` /
`(4,6,4,0,6)` / `(4,5,5,0,6)`, and the independently rescored retained
connected incumbent matches that upper. Require 50 / 20 / 30 exhausted
subcases respectively. Any `UNKNOWN`, count mismatch, disconnected incumbent,
or larger relaxed upper fails closed and promotes nothing. This evidence is
kept separate until the active catalog writer exits naturally.

**Result — ALL THREE EXACT CERTIFICATES PASSED.** The frozen run completed in
417.013025 seconds. It exhausted 50 / 20 / 30 subcases and reproduced relaxed
upper bounds 60 / 63 / 64 for `(3,6,5,0,6)` / `(4,6,4,0,6)` /
`(4,5,5,0,6)`, exactly matching the independently rescored connected
incumbents. No submodel returned `UNKNOWN`. Artifact SHA-256
`e4d11f62b4d04118ffe059273be6bf2cc691ae48d912face0941a3ec0c348d76`.
Decision: accept these as the 713th through 715th AAAAA per-vector proofs;
merge them only after the active catalog writer exits naturally.

## 2026-07-23 07:23 — AAAAA Hawk-aware packing diagnostic preregistration

Six additional unresolved thresholds force Bear A, Salmon A, and Hawk A to
their standalone maxima: `(6,4,1,3,6)>=68`, `(6,5,1,2,6)>=68`,
`(4,4,1,5,6)>=66`, `(4,6,1,3,6)>=66`, `(4,6,4,2,4)>=65`, and
`(3,5,4,3,5)>=63`. Extend the selected local set-packing relaxation with
hawk singleton cells. Deliberately drop hawk-hawk isolation and let
noncovering hawks live abstractly; this strictly enlarges the legal-board set.
Preserve forced Bear/Elk/Hawk/Fox cell non-overlap around every maximum salmon
shape. Source SHA-256
`ef93ede65e967a0aa6d3223ec44d2b159271b9d6ef88f460bdde52ca73ed3682`;
tests SHA-256
`279761b2d6b962143839c799c58af4901c33bd4195347385416855568d90fac4`;
Ruff and both tests pass.

Run a one-worker, 30-second-per-shape diagnostic over the six cases only after
the active three-vector certificate run releases its core. Select this
extension for formal certification if at least one case exhausts with a
relaxed upper below its registered threshold. `UNKNOWN`, a feasible
relaxation, or an invalid premise promotes nothing for that case. Store no
diagnostic result in the live catalog.

**Initial diagnostic result — FAIL CLOSED / PARTIAL.** The four one-salmon
cases stopped before model construction because the premise checker treated
an impossible zero-point singleton-salmon state as a second-best alternative.
No bound was emitted for them. The two four-salmon cases completed but their
relaxations were feasible exactly at threshold: `(4,6,4,2,4)` upper 65 over
20 subcases in 12.516915 seconds; `(3,5,4,3,5)` upper 63 over 12 subcases in
18.491991 seconds. They are not selected and remain open.

The root cause is deterministic: a lone salmon is always a valid unbranched
component and necessarily scores two, so no second-best score exists. The
premise now represents that state as unattainable. Ruff and three tests pass;
fixed source SHA-256
`dc4c8794e78b5e29ea770af52e088fae805965ccd5b0c24586cc5ecc99c042c2`,
test SHA-256
`d69b716d6ff69d87d83a32a1a75d3aae2861198fc49b59ff88c6d992997fd70e`.
Preregister the corrected diagnostic over only the four affected one-salmon
cases, with the same one worker / 30 seconds per shape and selection rule.

**Corrected diagnostic result — NOT SELECTED.** All four one-salmon models
completed exactly but the relaxed upper equaled the threshold in every case:
68 over 11 subcases for `(6,4,1,3,6)`, 68 over 10 for `(6,5,1,2,6)`,
66 over 11 for `(4,4,1,5,6)`, and 66 over 15 for `(4,6,1,3,6)`.
Together with the two earlier feasible four-salmon results, the extension
certifies zero of six cases and fails its selection gate. Decision: retain the
tested sound diagnostic implementation, but do not build certificate
serialization or alter catalog evidence. Cross-component isolation—not local
cell capacity alone—is required to tighten the Hawk-A tail.

## 2026-07-23 07:36 — AAAAA explicit one-loss fox diagnostic preregistration

The two four-salmon Hawk cases sit exactly one point below their standalone
relaxations. Therefore a fox that misses salmon cannot also miss its fox
neighbor: with at most one salmon-missing fox, that fox must be adjacent to a
salmon-seeing fox. Extend the local model's fox domain from the salmon boundary
to the immediately adjacent outer ring, place the missing fox explicitly, and
score all of its bear/elk/hawk/fox observations through actual local cells.
This removes the prior free abstract coverage while still dropping Hawk
isolation, Bear isolation, whole-board connectivity, and noncovering remote
groups, so it remains a sound superset.

Frozen source SHA-256
`be00c8ede53917d3a8fdd2a0a3115b9f948ec1ed0636f573cfe633ac38af4b19`;
tests SHA-256
`28854a6533934a6534eb800eaf78b76df13aa5836be05b242d6b621599aa1366`;
Ruff and four tests pass. Screen only `(4,6,4,2,4)>=65` and
`(3,5,4,3,5)>=63`, one worker, 30 seconds per shape. Select for formal
certification only on exact exhaustion with a relaxed upper below threshold;
`UNKNOWN` or threshold feasibility remains no proof.

**Diagnostic result — BOTH SELECTED FOR FORMAL CERTIFICATION.** Exact
exhaustion returned `(4,6,4,2,4)` relaxed upper 64 over 20 subcases in
29.915816 seconds and `(3,5,4,3,5)` relaxed upper 62 over 12 subcases in
48.939456 seconds. Both are one below their registered thresholds and match
their retained incumbents. Decision: add fail-closed two-row serialization,
independent incumbent validation, and source hashes; diagnostic output alone
does not yet promote either row.

## 2026-07-23 07:41 — AAAAA Hawk one-loss certificates preregistration

Frozen certificate source SHA-256
`f907b4526b752d013a742465cc6387c5ccf03e145829cc2de0136ba04ce95697`;
tests SHA-256
`28854a6533934a6534eb800eaf78b76df13aa5836be05b242d6b621599aa1366`.
Ruff and four tests pass. Run both fixed cases with one worker and a 30-second
per-shape limit; durable output:
`docs/v3/evidence/aaaaa_hawk_one_loss_certificates_2026-07-23.json`.

Accept only if every submodel is exact, `(4,6,4,2,4)` exhausts 20 cases with
upper 64, `(3,5,4,3,5)` exhausts 12 with upper 62, and each independently
rescored incumbent is a connected 20-cell board at that score. Any `UNKNOWN`,
larger upper, invalid witness, or count mismatch fails closed for the entire
artifact. Keep passed evidence separate until the active catalog writer exits.

**Result — BOTH EXACT CERTIFICATES PASSED.** The frozen run completed in
80.140116 seconds. `(4,6,4,2,4)` exhausted 20 subcases at relaxed upper 64;
`(3,5,4,3,5)` exhausted 12 at upper 62. Both retained connected boards
independently rescore to those values and no submodel was `UNKNOWN`. Artifact
SHA-256
`d2f769f5a4e5bafc57485c57a90739a8368ce6ec244096f92eca9ae8f5cb3c04`.
Decision: accept these as the 716th and 717th AAAAA per-vector proofs, pending
safe post-writer merge.

## 2026-07-23 07:53 — AAAAA final raw-gap-one salmon diagnostic preregistration

The only raw two-point-gap vector not covered by accepted certificates is
`(3,6,3,3,5)`, incumbent 61, challenged at 62 versus standalone ceiling 63.
Score 62 has exactly three branch types: maximum non-fox scores with one Fox-A
observation missing; Elk-A 17 with every other category maximal; or Salmon-A
seven with every other category maximal. The first two use the explicit
one-loss model over all three maximum three-salmon shapes. Seven salmon points
requires a pair plus singleton; assign each fox to a salmon component, so by
pigeonhole one component covers at least three foxes. Model each possible
anchor component/local-fox count and award the other component/foxes free
coverage, a sound relaxation.

Source SHA-256
`4a73a72e87afdc8be4c776baa3aff0a5a1ff0e0ca5fb71da05a1d58eed3b5dac`;
test SHA-256
`492d0a5e9653d8a7144b6dea7c57080f46035759a1f886929ffc7b003ac3b46c`;
Ruff and the fixed-case test pass. Run one worker, 30 seconds per subcase.
Select for formal certification only if both salmon branches exhaust exactly
and their combined relaxed upper is at most 61. `UNKNOWN` or upper >=62
promotes nothing.

**Diagnostic result — NOT YET SELECTED.** The maximum-salmon branch exhausted
15 subcases at relaxed upper 61 in 40.848628 solver-seconds. The seven-point
pair-plus-singleton branch exhausted 12 anchor relaxations but remained
feasible at 62 because two foxes and their non-salmon observations were
awarded abstractly. Combined result: upper 62 in 42.089895 elapsed seconds;
no proof and no catalog change.

Decision: retain the successful maximum-salmon half and replace only the loose
split branch. Fix a salmon pair, enumerate the singleton at every separation
where a Bear pair/singleton, Elk line of length at most four, Hawk singleton,
or fox adjacency can span both clusters, and use one representative for all
farther separations where the local set-packing model factorizes. Model all
five foxes and all non-salmon coverage cells explicitly. Preregister that
refinement separately after its source and tests are frozen.

## 2026-07-23 08:01 — AAAAA joint split-salmon diagnostic preregistration

The refined branch enumerates 49 symmetry-reduced pair-plus-singleton
placements: every component separation two through seven, plus one separation-
eight representative. Possible fox cells around components farther than seven
are at least six apart; no Bear pair/singleton, Hawk singleton, fox edge, or
straight Elk group of length at most four can span them, so the local packing
problem factorizes and the representative covers every farther translation.
All five foxes and all Bear/Elk/Hawk coverage cells are explicit; Bear/Hawk
isolation and whole-board connectivity remain dropped, preserving a superset.

Source SHA-256
`d1e33dc2f541ae6a729621f91e698259324e51985c89825ce28473f0af3d1c86`;
tests SHA-256
`555caf7705a08194f749179667f7e00c23c767ddf1152ae2ccc1e23dfe189367`;
Ruff and both tests pass. Re-run the retained 15-case maximum-salmon branch
and the 98 joint split cases with one worker and 30 seconds per case. Select
only if all 113 submodels are exact and the combined relaxed upper is at most
61; otherwise no proof.

**Diagnostic result — SELECTED FOR FORMAL CERTIFICATION.** The 15
maximum-salmon submodels returned upper 61 in 42.619225 solver-seconds. All 98
joint pair-plus-singleton submodels returned upper 61 in 367.893904
solver-seconds. No case was `UNKNOWN`; combined elapsed 412.321246 seconds.
Decision: add fail-closed certificate serialization and incumbent validation,
then run one frozen reproduction. Diagnostic output alone does not promote the
row.

## 2026-07-23 08:12 — AAAAA joint salmon certificate preregistration

Frozen certificate source SHA-256
`43e4893520e45a80cb6f00c46546813158e31ef305ce466ad3eef1266bf897e5`;
tests SHA-256
`555caf7705a08194f749179667f7e00c23c767ddf1152ae2ccc1e23dfe189367`.
Ruff and both tests pass. Run one worker, 30 seconds per subcase; durable
output:
`docs/v3/evidence/aaaaa_gap_one_joint_salmon_certificate_2026-07-23.json`.
Accept only if all 15 maximum-salmon and 98 joint split-salmon submodels are
exact, both branch uppers and the combined upper equal 61, and the retained
`(3,6,3,3,5)` connected board independently scores 61. Any deviation fails
closed. Keep passed evidence separate until the active catalog writer exits.

**Result — EXACT CERTIFICATE PASSED.** The frozen run completed in 403.837969
seconds. All 15 maximum-salmon and 98 joint split-salmon submodels were exact;
both branch uppers and the combined upper were 61. The retained connected
board independently rescored to 61. Artifact SHA-256
`9903119c4f0cdf0d7293b48862b4eb7315787e0c783092ad7d0a9e65b90b6686`.
Decision: accept `(3,6,3,3,5)=61` as the 718th AAAAA per-vector proof,
pending safe post-writer merge. Every first-pass raw gap <=2 is now closed.

## 2026-07-23 08:29 — AAAAA gap-two/two-salmon diagnostic preregistration

Screen the four remaining raw-gap-three vectors with exactly two salmon:
`(4,5,2,3,6)>=67`, `(5,5,2,2,6)>=64`, `(3,5,2,4,6)>=63`, and
`(3,6,2,3,6)>=63`. Exhaust the maximum five-point salmon pair and every one
of 19 symmetry-reduced two-singleton separations (distance two through seven
plus a factorized far representative). Zero/one salmon-missing fox is placed
explicitly; the two-miss maximum-salmon branch retains optimistic abstract
coverage so remote fox pairs are not excluded. Bear/Hawk isolation and
whole-board connectivity remain dropped.

Source SHA-256
`eb4012f74802a12341365a167bdf3cb0b0884f7c8280ef732f966cf071c7a23a`;
tests SHA-256
`0a55d374a7aaa7b1f93014b5322a15070b54e799bb5b4331badb0f68886fffb7`;
Ruff and both tests pass. Use one worker, 30 seconds per subcase. Select each
case independently for formal certification only if both salmon branches are
exact and the combined relaxed upper is below its registered threshold.

**Diagnostic result — NOT SELECTED.** All four maximum-salmon branches were
exactly feasible at their registered thresholds, with the best case using the
optimistic two-missing-fox abstraction. The separated-singleton branches then
returned `FEASIBLE` rather than exact `OPTIMAL` at their 30-second boundary
(case 20 / 3 / 20 / 20 respectively), so each full result failed closed as
`UNKNOWN`. Elapsed by vector: 265.585994, 63.779285, 77.890045, and
180.804709 seconds. No certificate and no catalog change. Decision: the next
sound tightening must represent the two salmon-missing foxes as an explicit
self-adjacent second fox component; do not spend more time on the already
non-decisive separated-singleton branch first.

## 2026-07-23 08:41 — Exploratory brute-force diagnostic deprioritized

The unregistered-output process associated with the earlier zero-hawk packing
diagnostic remained alive after 70 minutes because its disposable prototype
recomputed Bear placements inside its Elk loop. Per the no-kill rule it was
not interrupted or restarted. Its Unix niceness was changed from 0 to 15 so
the hash-pinned eight-worker catalog retry receives CPU priority; the process
continues naturally and its eventual output remains diagnostic-only. No exact
solver configuration, deadline, ledger, or proof process was changed.

## 2026-07-23 08:45 — CBDDB deep incumbent staging preregistration

Use the two cores not consumed by the AAAAA eight-worker retry for a deeper
CBDDB heuristic pass; no CBDDB exact proof starts before AAAAA completion.
Run the already hash-pinned release solver over all 826 vectors with two
threads, 32 restarts per count, 100,000 iterations per restart (20x the first
staging effort), and seed 20260724. Durable output:
`docs/v3/evidence/cbddb_wildlife_candidates_deep_2026-07-23.json`; log:
`cascadiav3/logs/cbddb_wildlife_candidates_deep_2026-07-23.log`. Binary
SHA-256 `37578ee5a379ec290fe04933e9ded9d1283bb9ba2e5743f309b93ab873ad7465`;
source/support hashes remain
`8181ddf434fdde8309bca619d51923b927408a6167614ab0cd9573a12094fe6d` /
`110dc92dcc95b5d0effbcfe17d6ac124ff2f216c6716ae860d496abe77956af9`.

Require all 826 unique vectors, connected 20-cell boards, and zero independent
Python score mismatches. Compare each row to the frozen first staging file and
retain the higher independently scored incumbent (seed/order breaks exact
ties). Neither file is proof evidence. Report the holistic candidate change,
but do not call it optimal until exact exclusions are complete.

## 2026-07-23 07:50 — AAAAA motif-coordinate relaxation screen preregistration

The long hinted retry has shown that additional generic coordinate-model time
has very low marginal proof yield. Screen a new strict relaxation that retains
all 20 coordinates, exact non-overlap, forced Bear-A pairs, Elk-A scoring
lines, valid Salmon-A component shapes, and every positive Fox-A observation.
It deliberately drops whole-board connectivity, bear-pair isolation,
separation between scored salmon components, and hawk isolation. Every legal
board meeting the challenged score is therefore contained; only `INFEASIBLE`
is proof evidence.

Positive containment calibration is frozen before the screen: the fixed
holistic `(6,4,6,0,4)` 68-point board solves at exactly 52 non-fox plus 16 fox
points. Five unit tests and Ruff pass. Relaxation source SHA-256
`3b0ddd8e2e8015f41392f3a7810930eff1c02aeb58c6e9e71f4fb410c6a66065`;
batch source SHA-256
`0162dd5d83083ecb134a4ed71d79f604853ab012c02ee7154483878c9748ffef`;
tests SHA-256
`eb5013bbcb8caf7f1db01727e5aad403c1c29bc108a1a9df1901924fcaf4f0b7`.

Run eight fixed cases sequentially with one worker, 30 seconds per case, and
base seed 20260723. Three already-certified challenges calibrate strength:
`(3,6,6,0,5)>=62`, `(3,6,3,3,5)>=62`, and `(4,6,4,2,4)>=65`. Five unresolved
challenges sample distinct motif regimes: `(1,4,6,3,6)>=66`,
`(2,2,6,4,6)>=66`, `(4,2,4,4,6)>=66`, `(6,1,5,2,6)>=68`, and
`(4,5,2,3,6)>=67`. Durable output:
`docs/v3/evidence/aaaaa_motif_coordinate_relaxation_screen_2026-07-23.json`;
log: `cascadiav3/logs/aaaaa_motif_coordinate_relaxation_screen_2026-07-23.log`.
Accept each case independently only when CP-SAT returns exact `INFEASIBLE`.
`FEASIBLE` means the relaxation is too loose; `UNKNOWN` means the proof is
incomplete. Either outcome changes no catalog row. A passed unresolved case
still requires frozen incumbent validation and certificate serialization
before merge.

**Result — NO CASE SELECTED.** All eight cases exhausted their 30-second
limit as `UNKNOWN`; none returned `INFEASIBLE` or a relaxation witness.
Branches ranged from 283,731 to 390,101 and conflicts from 83,633 to 108,097.
The complete batch took 240.185064 seconds. Artifact SHA-256
`d666ab0d2915885e7d589baf76e90683ee9a695b605de0264812a06c9ae8bd89`;
log SHA-256
`233e7459621199c25efc56fb09cff2c4dcd2454a8fc0a576343d6b267a13a306`.
No proof and no catalog change. Decision: retain the tested formulation as a
sound diagnostic, but do not spend a longer batch on it. The next formulation
must eliminate free-coordinate symmetry by anchoring explicit fox components
and using finite local cell-set packing, matching the prior gap-two diagnosis.

## 2026-07-23 07:56 — AAAAA motif-coordinate v2 screen preregistration

Before abandoning the coordinate relaxation, apply four sound symmetry and
propagation changes exposed by the first screen: order indistinguishable fox
and hawk coordinates; order equal-sized Bear/Elk/Salmon scoring groups and
unscored leftovers conditionally; remove any card-score choice that cannot
reach the target even with every other standalone maximum; and make each
selected Fox-A species observation choose exactly one adjacent witness rather
than leaving many equivalent adjacency booleans. The fixed 68-point witness
still solves at 52+16. Ruff and all five tests pass. V2 source SHA-256
`ca46c1ad072e757589ee3687c0e0cbcbc9bf16c794f645f52cafaed714c5290a`;
unchanged batch/test hashes are
`0162dd5d83083ecb134a4ed71d79f604853ab012c02ee7154483878c9748ffef` /
`eb5013bbcb8caf7f1db01727e5aad403c1c29bc108a1a9df1901924fcaf4f0b7`.

Run three cases sequentially with one worker and the same 30-second boundary:
known exact calibrations `(3,6,6,0,5)>=62` and `(3,6,3,3,5)>=62`, then the
unresolved gap-one challenge `(6,1,5,2,6)>=68`; base seed 20260723. Durable
output:
`docs/v3/evidence/aaaaa_motif_coordinate_relaxation_v2_screen_2026-07-23.json`.
Accept only exact `INFEASIBLE` per case. If both calibrations remain `UNKNOWN`,
close this coordinate-relaxation direction regardless of branch-count change
and proceed to finite fox-component enumeration; no longer run more time.

**Result — DIRECTION CLOSED.** All three v2 cases remained `UNKNOWN` at the
30-second boundary. V2 reduced branches versus v1 from 290,449 to 215,029 on
`(3,6,6,0,5)` and from 283,731 to 243,547 on `(3,6,3,3,5)`, but did not prove
either known calibration; `(6,1,5,2,6)` likewise ended at 243,368 branches.
Elapsed 90.148777 seconds. Artifact SHA-256
`8f0229bc7a528ea1cd3362c1415a2aa9128c61edeacbbb92b3b0ee8295fd0133`;
log SHA-256
`7723dd55aad0e0f32092a8442f4c522175958c3e9129656f39fa5668b822b4bd`.
No proof and no catalog change. Per the registered rule, do not lengthen or
further tune the 20-coordinate relaxation. Continue only with finite local
fox-component enumeration and cell-set packing.

## 2026-07-23 08:05 — AAAAA explicit two-missing-fox screen preregistration

Tighten the four gap-three/two-salmon maximum-salmon branches that previously
remained feasible only because two salmon-missing foxes were abstract. At each
registered target, maximum Bear/Elk/Salmon/Hawk scores force exactly 28 Fox-A
points: the two missing salmon observations are the only permitted losses, so
every other observation—including fox self-observation—is forced.

Exhaust two cases. If either missing fox touches a salmon-adjacent fox, both
can be represented in the explicit second ring unless they form an adjacent
chain; if neither touches the local cluster, the two must be adjacent to each
other. The existing explicit-ring model covers the first geometry. A new
colored enumeration covers the adjacent-pair geometry with 144 dihedral-
reduced placements at salmon-component distance two through seven and one
distance-eight representative for all farther factorized translations. It
then packs maximum Bear pairs/singles, Hawk singletons, and each maximum Elk
line partition around all six foxes with exact cell non-overlap and exact
positive observations. Bear/Hawk isolation and board connectivity remain
dropped, preserving a strict superset.

Bound source SHA-256
`5050fae63465251dd95667d13cd8ad96e9ec3be76b91952e13eecddd0419ea3f`;
runner SHA-256
`2b72a9f53fa31e2446735f65299b68c1cb7c05a7d352b26ab248b024178e6b58`;
tests SHA-256
`c27adecfcfbf36dcf58ed278150afd7873727e2f287629a5556cbdebe7505c33`.
Ruff and three structural tests pass. Run the four fixed cases sequentially
with one worker and 30 seconds per submodel. Durable output:
`docs/v3/evidence/aaaaa_two_missing_fox_screen_2026-07-23.json`; log:
`cascadiav3/logs/aaaaa_two_missing_fox_screen_2026-07-23.log`.

Select each maximum-salmon branch independently only if every explicit-ring
and remote-pair submodel is `OPTIMAL` and the union upper is below its fixed
target. `UNKNOWN` or an upper at/above target proves nothing. Even four passed
branches do not yet certify rows: the split two-singleton salmon branches must
be closed separately before certificate serialization and catalog merge.

**Result — ALL FOUR MAXIMUM-SALMON BRANCHES PASSED.** Every submodel was
`OPTIMAL`; elapsed 1153.227437 seconds. Registered target / combined upper /
explicit-ring upper / remote-pair upper were respectively: `(4,5,2,3,6)`
67/66/66/65 over 1+145 submodels; `(5,5,2,2,6)` 64/63/63/62 over 1+145;
`(3,5,2,4,6)` 63/62/62/62 over 1+145; and `(3,6,2,3,6)` 63/61/61/61 over
2+290. Artifact SHA-256
`9aefbb263bdd121c05823ef60436005a542694c556328bc8f2735d0506675fa1`;
log SHA-256
`e0e3092421a7ab86a1a3be9b08e729a0dbd14295fe841b4129b1e312d00a2a7d`.
Decision: retain all four exact branch exclusions and proceed to the separately
registered split-salmon screen. No row is promoted yet.

## 2026-07-23 08:25 — AAAAA split-salmon feasibility screen preregistration

Close the complementary branch in which the two salmon are nonadjacent
singletons and score four. At each target, the maximum Elk-A partition needs
29 Fox-A points and permits at most one salmon-missing fox; the one-point-lower
Elk-A partition needs all 30 fox observations and permits none. Lower Elk-A
scores cannot reach the target. Exhaust all 19 symmetry-reduced singleton
separations (distance two through seven plus the proven factorized far case),
both eligible Elk scores, and zero/one missing-fox branches.

Use a threshold satisfiability model rather than asking CP-SAT to prove a
maximum: exact non-overlap and positive Bear/Elk/Hawk/self observations are
retained, along with the fixed singleton salmon cells and explicit outer-ring
missing fox. Maximum Bear/Hawk motifs are packed; their isolation and board
connectivity are dropped. Thus `INFEASIBLE` remains a sound upper-bound result,
while a relaxation witness rejects the branch immediately. Expected submodel
counts are 57 for each five-elk allocation and 95 for the six-elk allocation.

Feasibility source SHA-256
`fa212369498931a198b394fdaeb86800ca04f58fb28da62ed1d1b28d167b01cc`;
runner SHA-256
`2033b501c0b1dc162847f628d788614288bbfc4cdc51991efb55408e70a29ec1`;
tests SHA-256
`d8adaf3adbd15c0b66ab42322959f226b66862b9fb8914773ad6330d19ccfed4`.
Ruff and two structural tests pass. Run sequentially with one worker and 30
seconds per submodel. Durable output:
`docs/v3/evidence/aaaaa_split_salmon_feasibility_screen_2026-07-23.json`;
log: `cascadiav3/logs/aaaaa_split_salmon_feasibility_screen_2026-07-23.log`.
Select a row for formal certificate reproduction only if every split submodel
is exactly `INFEASIBLE` and its maximum-salmon branch passed above. Any
`UNKNOWN` or threshold witness proves nothing for that row.

## 2026-07-23 09:00 — Exact wildlife mini-fleet provisioning

John explicitly requested parallelizing the pure-wildlife computation over
the Mac mini fleet. `docs/v3/FLEET.md` was read before design. Read-only
preflight found john2, john3, and john4 reachable, idle, with 10 cores and
183/210/82 GiB free respectively. All have the existing Python 3.12.13 fleet
venv but no OR-Tools. john1 remains excluded because it hosts the web UI and
is not provisioned for CBDDB.

Revision `bdf33f47` adds deterministic count-vector tasksets, disjoint
round-robin sharding to both exact catalog runners, imported-ledger support for
CBDDB, source/dependency/configuration pinning, per-host durable output,
heartbeats, collision refusal, and a fail-closed collector that independently
validates every returned board and exact task coverage. Ruff, 10 unit tests,
three shell syntax checks, two single-vector end-to-end catalog smokes, and
`git diff --check` pass. The scripts never stop or replace a process.

Provision only `ortools==9.15.6755`, matching the orchestrator, into the
existing `~/cascadia/venv` on john2–john4. Run installations independently in
parallel, then require all three to import that exact version. Any host failure
blocks fleet launch; no fallback version is permitted.

**09:06 provisioning wrapper failure.** The first local parallel wrapper
exited before starting an installation because it assigned to zsh's reserved
read-only `status` parameter. Read-only follow-up confirmed that john2–john4
had no `~/cascadia-aaaaa-exact` directory, no OR-Tools import, and no pip
installation process. No remote state changed and no scientific computation
ran. Decision: correct only the local orchestration variable/shell, retain the
same pinned dependency and hosts, and retry provisioning.

**09:08 provisioning attempt failed on all hosts.** The corrected parallel
wrapper reached john2–john4, but each `~/cascadia/venv/bin/python -m pip`
invocation failed immediately with `No module named pip`. No package was
installed and no scientific computation ran. Durable per-host logs:
`cascadiav3/logs/fleet_wildlife_provision_{john2,john3,john4}_20260723.log`.
Decision: inspect the hosts for their already-supported installer and install
the same exact OR-Tools pin into the existing venv; do not bootstrap an
unreviewed dependency path.

**09:12 provisioning passed with dependency isolation.** Revision `089680b2`
changes the fleet worker to use a dedicated
`~/cascadia/wildlife-venv` (ledger-pinned as `wildlife_venv`) so installing
the CPU solver cannot perturb the Torch/MPS generation environment. Shell
syntax, all 10 catalog/sharding unit tests, and `git diff --check` pass. A
fresh Python 3.12 environment was created independently on john2–john4 and
`ortools==9.15.6755` installed successfully on every host. Sorted `pip freeze`
is identical on all three, SHA-256
`36c88ffd2e0d1210a4148f3d523e0f5b635dbfe7d71243d1d0be2bf55f4a3661`.
Provision logs for john2/john3/john4 have SHA-256
`dd44108f636767f7ab98ab79670d67e224d0a6a78c3913f95f5dabf2470d2179`,
`ba8e39f56d24fba831399d170702c99e4f93d89152f35662293b3b43f7f3328a`,
and `0f542814e6a188201626ba5c6381de590fb5303626e82ce7ce10cba5a894fdcf`
respectively. Decision: all three hosts are eligible for the hash-pinned AAAAA
exact-tail launch.

## 2026-07-23 09:15 — AAAAA exact-tail mini-fleet launch preregistration

Freeze a read-only snapshot of the active local AAAAA catalog; do not stop,
restart, or write through its PID `90993`. The validated snapshot contains
711 embedded complete proofs and has SHA-256
`2b74a5a3d10dba6225d191d5b094e17a4f0f945da3b854ef066f63b5c965541e`.
The deterministic `wildlife-catalog-taskset-v1` complement contains 115/826
canonical count vectors, SHA-256
`8cdec4eec69e84ebaf181419a01a1339bc1fa0443ad5554cd0a4360083ee54d3`.
Use the independently validated 826-row deep candidate file, SHA-256
`d8c4f1ce9d9b7decac3156c6500b9e35407d70c74573877529666ba02ff496ae`.

Launch tag `aaaaa_exact_tail_fleet1_20260723` from the commit containing this
preregistration (the exact revision is recorded in the launch ledger).
Round-robin shard the frozen taskset over john2/john3/john4 as 39/38/38
vectors. Each host runs two independent catalog processes with four CP-SAT
workers apiece, OR-Tools `9.15.6755`, relaxation limit 60 seconds, connected
limit 120 seconds, and base seed `20260725`. The imported snapshot, candidate
file, taskset, source files, dependency version, source revision, worker PIDs,
and every returned artifact are hash/configuration pinned. john1 and john0 are
excluded.

Decision rule: accept a count vector only when its catalog row is
`proof_complete` under the registered exact method and its witness passes the
collector's independent count/connectivity/scoring validation. `UNKNOWN`,
timeout, worker failure, missing coverage, duplicates, hash mismatch, or a
partial fleet remains incomplete and fails closed. Collection waits for all
three terminal shards and cannot mutate the active local catalog. The fleet
run is an exact finite puzzle computation, not a gameplay gate or strength
claim.

**09:17 launch failed before computation.** Tag
`aaaaa_exact_tail_fleet1_20260723`, revision `d180b0a2`, passed local input
validation and all-host preflight, then failed during john2 deployment because
the remote snapshot lacked the parent directory for
`crates/cascadia-game/src/bin/aaaaa_wildlife_solver.rs`. No worker PID or
output shard exists on any host. The planned launch ledger is retained with
SHA-256
`a55a205b35e48354bf6e9a722024ad09f3455fde99920b57d2fb2cc8d4c8d2b3`;
the single-use tag will not be reused or deleted. Decision: make deployment a
complete all-host phase that creates every source parent before any process
launch, test it, and preregister a fresh tag with otherwise identical inputs
and acceptance rule.

**09:19 retry preregistration.** Use fresh tag
`aaaaa_exact_tail_fleet2_20260723`. Inputs, 39/38/38 shard assignment,
hosts, solver configuration, and fail-closed decision rule are unchanged from
09:15. The resulting source revision must pass shell syntax, the 10
catalog/sharding unit tests, and `git diff --check`; its launcher must complete
deployment to all three hosts before starting the first worker. Any remaining
tag collision or deploy error blocks this retry without deleting prior state.

**09:23 retry failed before solving a vector.** Tag
`aaaaa_exact_tail_fleet2_20260723`, revision `136e1a1f`, deployed completely
and launched wrapper PIDs john2 `1005`, john3 `74154`, john4 `21079`.
The local post-launch ledger update then exposed macOS system Python 3.9's
lack of `zip(strict=...)`; the same error occurred immediately in all three
remote catalog processes. All wrappers terminated naturally with exit 1,
produced no shard ledger, and were not killed or restarted. The recovered
launch ledger records the PIDs and has SHA-256
`01f5158c517c0f5fda03bf26044a747e316087da946f99e00c9f61829258daa8`.

The 09:12 note's Python 3.12 description was incorrect: that environment was
created from system Python 3.9.6. The existing fleet generation venv is the
verified Python 3.12.13 source. Decision: retain both failed tags, require
Python 3.12.13 exactly in launcher and worker, use the repo venv for local
ledger transitions, and create a fresh isolated `wildlife-venv-py312` from
the known fleet interpreter. A third run may be preregistered only after all
hosts pass exact Python, OR-Tools, and identical-freeze checks.

**09:28 Python 3.12 provisioning passed; third launch preregistered.**
Revision `53003c52` makes Python `3.12.13` an exact launcher/worker/ledger
requirement, changes the isolated default to `wildlife-venv-py312`, and uses
the repo venv for local ledger transitions. Shell syntax, all 10
catalog/sharding unit tests, and `git diff --check` pass.

Each host's new environment was created from
`~/cascadia/venv/bin/python` and independently verified as Python `3.12.13`
plus OR-Tools `9.15.6755`. Sorted `pip freeze` is identical on john2–john4,
SHA-256
`feb90716f10168387d934d9639c586c0e8fcb8b24e948fd419de84296ad7ee53`.
Provision logs for john2/john3/john4 have SHA-256
`7e2fe9dd62f0225148e171f32ce473d92a72070504ffb118d6e457804f1c8ebc`,
`3193e1b9c4bf91ec451a292058e91e664b34927a2c6ab80cfb4be64891d6b053`,
and `ae676899296c74608f252828c62aafdde73da29880a8ab2052973bb8d78b0a21`
respectively.

Use fresh tag `aaaaa_exact_tail_fleet3_20260723` from the commit containing
this entry. Retain the frozen 115-vector taskset and 711-proof imported
snapshot, deep candidates, 39/38/38 assignment, three hosts, 2 jobs × 4
workers, 60/120-second limits, seed, and exact fail-closed decision rule from
09:15. Both prior failed tags remain immutable evidence. Launch only if the
new exact runtime preflight and collision checks pass on all hosts.

**Launch passed.** Tag `aaaaa_exact_tail_fleet3_20260723` launched from
revision `c726df878c6f49df7c4aee22dfd498c117c713df`. Wrapper PIDs are john2
`1862`, john3 `74966`, and john4 `21552`; initial heartbeats were fresh and
each log reported exactly 711 imported proofs plus its registered 39/38/38
assigned vectors. Launch-ledger SHA-256
`a8dd6c12f7418a2960417555162872984461d6872e115fa4b6ec757f7b02e2de`.
State: live; do not read a partial shard as evidence or restart a worker.

Durable collector waiter PID `52984` polls terminal markers only, emits
heartbeats to
`cascadiav3/logs/aaaaa_exact_tail_fleet3_collect.log`, and can be paused
between stages with
`cascadiav3/logs/HOLD_aaaaa_exact_tail_fleet3_collect`. After all three
workers terminate naturally it runs the fail-closed collector and accepts
completion only for 115/115 independently verified exact rows. It launches
no follow-on computation; an incomplete exact tail exits nonzero for explicit
analysis rather than silently advancing to CBDDB.

**Collector-waiter launch correction.** PID `52984` did not survive the
launching tool shell; it emitted neither a heartbeat nor an exit file and is
not live. The three remote solver wrappers are unaffected and retain their
own durable heartbeats, logs, outputs, and terminal markers. A terminal-only
foreground monitor is active for this session; do not double-launch another
collector while it remains active. If the session is interrupted, collection
is reboot-reconstructible with
`fleet_wildlife_exact_collect.sh aaaaa_exact_tail_fleet3_20260723 collect`,
which still refuses any partial or invalid fleet state.

## 2026-07-23 — john1 authorization for the CBDDB exact catalog

John explicitly authorized using john1 for the next ruleset computation.
Plan CBDDB as four deterministic shards over john1–john4: 207/207/206/206 of
the frozen 826-vector taskset, retaining two jobs × four CP-SAT workers,
Python `3.12.13`, OR-Tools `9.15.6755`, 60/120-second limits, independent
collection validation, and the rule that CBDDB does not launch before AAAAA's
exact tail is terminal and collected.

Read-only preflight found the john1 web UI reachable at its documented
Tailscale address. SSH rejected both the configured `john1` account and the
infrastructure document's `johnherrick` account with the documented
`~/.ssh/john0_codex` key; `FLEET.md` warns that repeated failed attempts can
trigger sshd source penalties. No UI or host process was stopped or restarted.
Decision: john1 is authorized but not eligible until a cooled-down retry
authenticates and exact runtime/source provisioning passes. The four-host
launch must fail closed rather than silently omit or trust an unverified host.

**Access follow-up:** a cooled-down direct retry and ProxyJump retries through
john2 both rejected the documented key for both `john1` and `johnherrick`.
Because the jump changes the connection source, this is not safely
attributable only to the direct-source sshd penalty; the account/key
authorization itself needs repair or confirmation. The UI remained reachable
and untouched. No provisioning or CBDDB computation started. Keep the
four-host shard plan frozen, but do not claim john1 capacity or launch it until
SSH access and the exact environment preflight actually pass.

**Topology correction from John:** this orchestrator workspace is john1
itself (`Johns-Mac-mini.local`, user `johnherrick`); it is not a fourth SSH
target. The failed SSH checks changed nothing. Local read-only preflight found
10 cores, Python `3.12.13` and OR-Tools `9.15.6755` already available in the
repo `.venv`, with the web UI and champion suggestion service live. Disk
headroom is 2.8 GiB, so do not create a duplicate environment; the exact run's
few-megabyte inputs/ledgers fit in place.

Decision: add explicit local-host execution to the exact launcher/collector.
For host label `john1`, preflight and deploy within the current repo and launch
the worker in a detached named `screen` session; john2–john4 remain SSH
targets. Record the distinct local venv path in the launch ledger. Do not stop
or restart the UI, champion server, or any active AAAAA process.

Implementation adds ledger-pinned `local_host` / `local_wildlife_venv`,
local preflight and input staging, detached-screen launch, and local status /
collection routing. The first two screen plumbing smokes used blocking
`screen -DmS`; their short children terminated naturally before the
post-command liveness check, revealing the option error without running a
solver. Switching to detached `screen -dmS` passed both PID liveness and
natural-exit checks. Shell syntax, the 10 catalog/sharding unit tests, and
`git diff --check` pass.

## 2026-07-23 10:15 — AAAAA exact-tail fleet pass terminal verdict

All three registered shards terminated naturally with catalog exit 2 and full
115/115 task coverage. The fail-closed collector independently validated every
returned board, source/input hash, shard assignment, and terminal marker.
Collection-manifest SHA-256:
`65826a579e395751097cc0eb2d99701bc66823d728352b9e22f7a349a63cff62`.

Result: 13/115 new exact coordinate-model certificates and 102 timeouts at the
registered 60-second disconnected plus 120-second connected limits. Proof
methods were 10 `connected_model_infeasible` and 3
`disconnected_relaxation_infeasible`. Median per-vector wall time was about
180 seconds; total solver wall was 20,163.2 seconds, completing in about one
hour over six concurrent jobs. Three of the 13 overlap the seven already
frozen specialized certificates, so the independently certified AAAAA union
is now **728/826**, leaving **98** unresolved.

Decision: AAAAA is not complete, and the user's requested ordering therefore
blocks CBDDB launch. The collected incomplete rows are valid incumbents but
not optimum claims. A stronger pass or additional finite specialized
certificates must close the 98-vector tail; do not represent a longer timeout
as guaranteed completion.

## 2026-07-23 — AAAAA remaining-gap confidence audit

Combine the frozen 711-proof snapshot, the independently collected fleet
shards, the seven specialized certificates, the deep 826-row candidate file,
the global 68 proof, and the per-count relaxation. For each of the 98 unique
unresolved vectors, take the best validated incumbent and the minimum sound
upper bound. Gap distribution:

```text
upper - incumbent: 1:4, 2:15, 3:30, 4:31, 5:15, 6:3
```

Thus 19/98 are already mathematically guaranteed within two; 79/98 are not
yet bounded that tightly. Empirical calibration is substantially stronger
than the raw relaxations:

- across all 728 certified vectors, the deep candidate is exact for 703 and
  one point low for 25; none is two or more low;
- among the 126 certified vectors whose original relaxation gap was at least
  three, it is exact for 117 and one point low for nine; none is two or more
  low;
- among the 10 non-overlapping new hard-tail coordinate proofs from the fleet,
  it is exact for nine and one point low for one.

The deeper annealing pass improved 60/98 remaining incumbents relative to the
initial pass (55 by one point, five by two), so the tail is appropriately
treated as selected hard cases rather than an exchangeable random sample.
Verdict: **high empirical confidence** that the remaining stored boards are
within two, but no defensible guarantee that all 98 are. The exact catalog
must continue to label 79 rows unproven until their sound gaps close.

## 2026-07-23 — AAAAA tail-difficulty profile

Join the same frozen exact/candidate inputs as the remaining-gap audit and
compare the 98 unresolved count vectors with the 728 certified vectors. The
first inline profile invocation had a bracket syntax error and produced no
result or state change; the corrected invocation completed.

The tail is not a random collection:

- all 98 have five or six foxes; 81 have six foxes;
- 96/98 have all five species present;
- mean unresolved counts B/E/S/H/F are
  `3.265/4.082/4.224/2.602/5.827`, versus
  `4.099/3.989/3.970/4.188/3.754` for certified rows;
- unresolved candidate score contribution averages are
  `7.44/12.09/12.71/6.87/24.82`, so fox supplies about 39% of the total and
  is near its count-only maximum.

Mechanism: the count relaxation gives each high-count fox one point for every
present wildlife type independently. With six foxes and all five types it
therefore assumes 30 fox points, but a legal board must give every fox a fox
neighbor plus adjacent Bear, Elk, Salmon, and Hawk observations while using
the same scarce non-fox tiles to form isolated Bear-A pairs, straight Elk-A
groups, unbranched Salmon-A runs, and isolated Hawk-A tiles. Those individual
maxima compete for the same hex adjacencies, and the additive relaxation does
not encode most of that global interference.

The proof task is also asymmetric: a strong board is already known, so the
solver must refute every board scoring one point higher. Each threshold model
places 20 labeled tokens in the complete radius-19 coordinate domain, reifies
all 190 pair adjacencies, and couples subset-based Elk/Salmon motifs, Fox
observations, non-overlap, and (in the exact pass) a rooted connectivity
arborescence. Across the 102 fleet timeouts, both the disconnected and
connected attempts remained `UNKNOWN` (204/204 attempts) after a median
1.85 million branches and 410 thousand conflicts per vector. Verdict: the
remaining cases are hard primarily because they require proving subtle
high-fox motif incompatibility across many symmetric near-solutions, not
because good incumbent boards are difficult to find.

## 2026-07-23 10:44 — AAAAA layered single-anchor calibration preregistration

Primary-source review of recent exact combinatorial search converges on a
layered alternative to another monolithic coordinate solve: arithmetic-profile
filtering, canonical local signatures, reduced SAT/CP realization, and an
independent decoding check. The new strict relaxation applies that pattern to
vectors with a unique non-Fox species token. It enumerates every subset of
foxes around that token modulo the ring's full dihedral symmetry, exhausts
every non-Fox score structure capable of reaching the challenged threshold,
and locally packs every scoring group or leftover token that covers an
explicit fox. Non-covering groups and foxes are awarded optimistic abstract
placements/coverage; Bear/Hawk isolation, Salmon-component separation, and
whole-board connectivity are dropped. Therefore local infeasibility remains
a sound exclusion.

Frozen source SHA-256
`0284fcac82dd453b187136ff0914a7ee7a169ff882d3e612ebe051d55b856eb4`;
tests SHA-256
`2f32972889070fa7b0be196b59223b4b972dd587e018ccb5153ce6948b8d756d`.
Ruff, five new structural tests, and all 17 retained specialized-bound tests
pass. Calibrate four already-proven hard-tail vectors chosen before output,
covering each possible anchor family and raw gaps five or six:
`(1,1,6,6,6)=64`, `(2,1,6,5,6)=64`, `(6,6,1,1,6)=66`, and
`(5,3,5,1,6)=62`.

First run a positive-containment challenge at the known incumbent score
(`--challenge-offset 0`), one worker and ten seconds per local submodel.
Every row must return `RELAXATION_FEASIBLE`, never `INFEASIBLE` or `UNKNOWN`,
with a returned upper at least the independently rescored incumbent. Durable
output:
`docs/v3/evidence/aaaaa_single_anchor_containment_calibration_2026-07-23.json`.
Then challenge incumbent plus one with the identical cases and limits; output:
`docs/v3/evidence/aaaaa_single_anchor_strength_calibration_2026-07-23.json`.
Select the formulation for a separately preregistered unresolved screen only
if containment passes, no strength case is `UNKNOWN`, and at least one of the
four plus-one challenges is exactly `INFEASIBLE`. Otherwise record no proof
and revise the representation rather than extending time.

**Result — CONTAINMENT PASSED; STRENGTH GATE FAILED.** All four incumbent-
score challenges returned `RELAXATION_FEASIBLE` after one exact local solve,
so the positive containment check passed. All four plus-one challenges also
returned `RELAXATION_FEASIBLE` after one solve; no known exclusion was
reproduced. The decisive witnesses used only one locally explicit anchor-
observing fox in three cases (two in the fourth), while the remaining foxes
received optimistic abstract coverage. This identifies the missing coupling:
the non-anchor fox clusters and the scoring motifs covering them must be
represented explicitly.

Containment artifact SHA-256
`6007e88dec30cf2eff3734501a3fe199bbbf6cff7ed3f6fa9e327b608a05588c`;
strength artifact SHA-256
`0d2a3dae9811b9c50256625bcc9f8332eebee96059c9e402a55bac02922f92dc`;
logs SHA-256
`e08bad0f17038e4400ac91aca4da333bb59bdb846fbdcd3404a0a6cf7de3e81d` /
`be4c861183d36305f3df2dd5b5bda4a1ca389006c24a7ceeb3a2213fa0d6b219`.
Elapsed was 0.143339 / 0.460073 seconds. Per the frozen decision rule, do not
screen unresolved rows and do not extend this abstraction's time. Retain it
as a tested first-stage filter and proceed only with explicit multi-cluster
coupling or a different exact representation.

## 2026-07-23 10:55 — AAAAA radius-two fox-neighborhood preregistration

The direct fox-adjacency table rejected at 06:10 does not expose when foxes
can share a non-Fox witness. Add a stronger redundant propagator inspired by
the recent specialized-propagator and canonical-local-signature literature:
classify every fox pair as hex distance one, distance two, or farther, and
allow only radius-two relation graphs realizable on the hex lattice. Connected
canonical coordinate shapes through five foxes number
`1,3,15,127,1338`; after graph isomorphism and arbitrary far separation of
components, the labeled allowed tables contain `1,3,24,437,14073` rows.
The table is globally exact through five foxes. Six-fox vectors receive all
six overlapping five-fox tables, a sound local-consistency relaxation.

This couples fox-fox adjacency and common-neighbor possibility before the
score model chooses per-species witnesses. The exact coordinate constraints
remain authoritative, so the tables are redundant and cannot remove a legal
board. The impossible four-fox unit-distance clique is rejected, actual
radius-two layouts are retained, and the fixed 68-point optimum remains
exactly feasible.

Frozen wrapper SHA-256
`5ac47104e9d38fa4629ceb9d9f31a7da6a783f6c61f606adc3b09ecb87107434`;
test SHA-256
`0f49e7ffaec389c1780ceb6918b85670080ee1ba341f9b3d83f62e8a14fcbb4f`.
Ruff and four focused tests pass. Calibrate the same already-certified
`(3,6,6,0,5)` threshold-62 connected case, retained 61-point hint, two
workers, 60 seconds, seed 20260723, and global ceiling 68. Durable output:
`docs/v3/evidence/aaaaa_fox_neighborhood_calibration_2026-07-23.json`;
log:
`cascadiav3/logs/aaaaa_fox_neighborhood_calibration_2026-07-23.log`.
Select only an exact `INFEASIBLE` result. `UNKNOWN` or a feasible relaxation
witness changes no catalog row and rejects this wrapper for unresolved work;
do not lengthen it.

**Calibration result — NOT SELECTED.** The radius-two table model returned
`UNKNOWN` after 60.011027 solver-seconds (69.860824 end-to-end including table
construction), with no witness, 287,656 branches, and 60,345 conflicts. This
is fewer branches than the direct fox-graph calibration's 337,376, but it did
not produce an exact result and therefore fails the registered gate. Artifact
SHA-256
`68e73bee7ab71f7859e2c50b03afc168ec16f4f2cc79ed1124dcc566a4701c8a`;
log SHA-256
`d73953611ee5287e679940c6f3fcdf10dbfe4671258bdfa90a580815bce11a39`.
Decision: do not run it over unresolved rows or lengthen it. The local
relation table must be coupled directly to the per-species Fox-A witness
choices before another calibration.

## 2026-07-23 10:59 — AAAAA canonical Fox-witness coupling preregistration

Couple the retained radius-two tables directly to Fox-A coverage. Each scored
fox/species observation is assigned to the lowest-index adjacent token, a
deterministic canonical witness that removes assignment symmetry. Foxes with
the same witness are constrained pairwise to distance at most two, and every
common-witness triple must realize the exact distance pattern of three
distinct cells in the witness token's six-cell ring. These are redundant
consequences of the exact coordinates and scoring rule, but expose the
per-species motif/coverage interaction requested by the failed 06:10 and
11:22 calibrations.

Frozen wrapper SHA-256
`78c90a138598f05fca3072cbe759bffc4cf85a652359b31a03a704c099939599`;
test SHA-256
`ca472af86b7fe86fda92c59223dfbf1cad0ab6f8638d974c4863c6778a35f2da`;
retained neighborhood wrapper SHA-256
`5ac47104e9d38fa4629ceb9d9f31a7da6a783f6c61f606adc3b09ecb87107434`.
Ruff and all six focused neighborhood/witness tests pass, including fixed
68-point containment.

Repeat the fixed already-certified `(3,6,6,0,5)` threshold-62 connected
calibration with its retained 61-point hint, two workers, 60 seconds, seed
20260723, and global ceiling 68. Durable output:
`docs/v3/evidence/aaaaa_fox_witness_calibration_2026-07-23.json`; log:
`cascadiav3/logs/aaaaa_fox_witness_calibration_2026-07-23.log`. Select only
exact `INFEASIBLE`; `UNKNOWN` or any witness rejects this formulation for the
unresolved tail, with no longer retry.

**Calibration result — NOT SELECTED.** The coupled model returned `UNKNOWN`
after 60.014552 seconds with no witness, 312,024 branches, and 65,484
conflicts. It is weaker in this calibration than the radius-two-only table
despite stronger propagation, consistent with the extra canonical-witness
Boolean structure costing more than its pruning saves. Artifact SHA-256
`6b82f03c4dd71fc73eb2e577197e005cd77dbec915b45a71634d45b59f9b8b08`;
log SHA-256
`f2814576742fb76dc69b2009749abfb6d3055341cebdb74d185813b7210050d7`.
Decision: reject it for the unresolved tail and do not lengthen. The online
review's general lesson survives, but the measured implementation verdict is
specific: static local tables inside the monolithic coordinate model are not
enough. Continue only with the already-successful layered finite
shape/profile/filter pipeline, where local geometry is enumerated outside the
20-coordinate solve.

## 2026-07-23 11:08 — AAAAA base exact retry exits naturally

The preregistered 05:25 retry completed without intervention. Its final
826-row v2 ledger contains 711 exact rows and 115 `incomplete_timeout` rows.
It added one exact result beyond the imported 710-row ledger:
`(4,4,2,4,6)=66`, proved by connected-model infeasibility at threshold 67
after the disconnected relaxation returned `UNKNOWN`. The final method counts
are 332 witness matches, 344 disconnected-relaxation infeasibilities, 35
connected-model infeasibilities, and 115 timeouts.

This does not change the already reported union: the new result was present in
the 711-row snapshot used by the terminal fleet pass. Unioning the final base
ledger, the 13 fleet proofs, and seven specialized certificates still yields
728/826 unique exact vectors, leaving 98 unresolved. Per John's requested
AAAAA-then-CBDDB order, the staged CBDDB exact taskset remains blocked.

Final base JSON SHA-256
`9694f41175799c1454da35c6f6b92f77e8fee79699fb6e805ca3c79e94ffd017`;
rendered Markdown SHA-256
`b309570fb1437dd7bec130767e30fc14d4ed68e3f47dff81c4e53470498a5c60`;
retry log SHA-256
`205635c72aa2319203d3cf7252738fbab311b1533d1a535ccad1e200ba868f81`.
The process was confirmed absent and every fleet host idle at the 11:08
read-only status check; no process was killed or restarted.

## 2026-07-23 11:35 — Cap-seven holistic wildlife-bound exploration

John asked how the all-board AAAAA and CBDDB upper bounds change when one
species may appear seven times among the same 20 tokens. This is deterministic
rules analysis, not an exact-search verdict or a promotion experiment. An
initial inline enumeration was used only to identify the formulas and candidate
maxima; it is not retained as evidence.

The durable calculation will be accepted only after a standalone enumerator
is frozen and passes three checks: (1) exhaustive equality with both existing
cap-six `count_relaxation` implementations over all 826 allocations, (2)
direct seventh-token parity with the production Rust scoring tables/formulas,
and (3) independent verification that the coefficient of
`(1+x+...+x^7)^5` at `x^20` is 2,226 and every reported maximizing allocation
sums to 20 with no count above seven. Report both the existing geometry-free
count bound and any strictly stronger elementary incidence bound; do not call
either the achievable optimum. The existing AAAAA 68 is a cap-six exact
holistic certificate and must not be compared as though it were the same
relaxation.

**11:38 result.** The frozen enumerator exhaustively reproduced both existing
cap-six relaxations on all 826 allocations, then enumerated all 2,226 cap-seven
allocations. AAAAA's direct geometry-free maximum changes 73→75. Adding the
universal hex-incidence fact that one token can neighbor at most six foxes
tightens the cap-seven result to **74**, versus the same 73 at cap six.
Cap-seven incidence-bound maximizers are `(2,2,7,2,7)`,
`(2,3,7,1,7)`, and `(2,4,7,1,6)`. CBDDB's geometry-free maximum changes
100→**102**, attained at `(0,3,6,4,7)` and `(0,4,3,6,7)`.

These are sound all-board upper bounds, not realizability claims. Therefore
the cap-seven AAAAA holistic optimum is currently in `[68,74]`: the existing
68 board remains legal, but its exact certificate covers only cap six. CBDDB
is currently in `[84,102]`, with 84 still only a validated heuristic witness.
No cap-seven coordinate solve was launched.

Enumerator SHA-256
`55f37219a5fe5f5c60514113beb216a42823ffd385cf06d6b6907d2ee3a6b5ed`;
test SHA-256
`ed4108417a2afe8c36da3cdf7a007b731c63c7b49b932c7da39071f0f2f45b15`;
evidence SHA-256
`fc27221050f55238d0aa11476825bae9ec4c1c1f9e084146347eb50146beae09`.
All four Python tests pass, including exhaustive cap-six parity and every
cap-seven maximizing allocation. Production Rust regression
`seventh_token_reference_patterns_match_a_and_cb_tables` passes for the
seventh-token AAAAA and Bear-C/Elk-B values; the complete
`cargo test -p cascadia-game` gate passes 124 tests. Durable interpretation:
`docs/v3/WILDLIFE_CAP7_UPPER_BOUNDS.md`.

## 2026-07-23 12:55 — PREREGISTRATION: all 1,024 wildlife-card rulesets

John requested one maximum-scoring pure-wildlife board for every ordered
combination of Bear/Elk/Salmon/Hawk/Fox A/B/C/D cards, still exactly 20 tokens
and at most six of each species. Scope is therefore all `4^5 = 1,024`
five-letter ruleset IDs in lexical product order. Habitats, tile compatibility,
drafting, Nature tokens, and every other game mechanic remain excluded.

Deliverable contract:

1. exactly one connected 20-token board per ruleset;
2. every species count in `[0,6]`;
3. score and five-part breakdown agree between an independent executable
   specification and the production Rust scorer under that exact card ID;
4. optimality requires either a witness matching a sound all-board upper bound
   or exact infeasibility of every strictly better count/profile branch;
5. `UNKNOWN`, timeout, heuristic search, an incomplete ledger, or a strong
   incumbent is never published as an optimum;
6. retain one canonical board under score ties and preserve per-result source,
   rules, solver, seed, and fleet provenance.

Engineering gate before production: implement and unit-test all twenty card
scorers and sound fixed-count bounds; reproduce AAAAA and CBDDB scorers and
known boards exactly; add exact encodings for the ten card variants absent
from the two current coordinate models; demonstrate independent model/oracle
agreement on adversarial fixtures; then calibrate throughput on a frozen,
stratified ruleset sample. Optimize before the 1,024-row launch. Production
will be sharded over john1–john4 only after that gate. The existing AAAAA
98-row tail remains an honest incomplete catalog and is not silently treated
as solved input for this broader task.

## 2026-07-23 13:00 — All-card scorer gate passes; candidate calibration

Source base `7a8ed38a4cb24f3ebcdcd99d7faaa259953feb5d` on john1. The independent
Python specification now covers all twenty wildlife cards and all 826 legal
cap-six count allocations. A separate Rust batch oracle constructs the
canonical production `Board` and invokes `score_board`. Four frozen connected
20-token boards (seeds `101,202,303,404`) crossed with all 1,024 ordered
rulesets produced **4,096/4,096 exact five-part score matches**. Canonical
oracle-response SHA-256:
`06ee8d41dbd14766291d70022259ac930d6ddbf4fc2d7592be7ce0a9cbbd1bc9`.
The verifier is permanent and rerunnable, not an inline-only assertion.

The first performance change constructs one production board per evaluated
layout and scores uniform A/B/C/D once each; a ruleset then selects its five
relevant components. It also searches species counts directly while preserving
the cap. Release calibration measured 10,000 evaluations of AAAAA in 0.11 CPU
seconds. The frozen 64-ruleset lexical pilot used eight threads, four restarts,
25,000 iterations/restart, seed `20260723`, and completed in 12.570 seconds.
Every emitted board passed the independent scorer. This deliberately shallow
quality pilot matched no global count relaxation: mean relaxation gap 16.8125;
the median ruleset still had 406 of 826 count branches above its incumbent
(mean 389.953, maximum 747). Decision: the candidate engine is fast enough,
but the present separable count relaxation is far too loose to serve as the
main proof architecture. Do not launch the 1,024-row production search until
the generalized exact encoding and a substantially stronger proof filter pass
their frozen calibration.

Hashes: independent scorer
`527988200686617997f7e7564b337aa62e2c23342f70972e8478422331c366d8`;
tests `7ca8d18eb6f0476fa751fe2874374f0d661c8814ee3d650dff1da784e6208733`;
oracle verifier
`031bed4ea5468ecd4350d58507b7199f02a8f9243b92968172814f7e27bb0e92`;
Rust oracle
`c2b6761562e8487d388e51637768cedb35b3845d0bb49f176241f295420024da`;
candidate engine
`a889a4ba34856c3d2605a06802da4509feb348df44d0bc21b01b5256927618da`;
shared search support
`e094671b33c4f525a189acce590416bfe1cc17856edbda0360770db540468ee0`;
pilot JSON
`8449b1d386ca83c061bfe7ece6881420c7d5865bfc5afa799e13187c75d96696`;
pilot log
`fce5f3657cd15d3045fdd30e08393d212870dbe20bd2b7c1d20ef3136da58711`.

## 2026-07-23 13:13 — Generalized exact model passes all-card fixed-board gate

The composable CP-SAT achievement-certificate model now implements all twenty
cards on the shared coordinate/connectivity core. It includes full-component
Bear/Salmon/Elk scoring, Elk A/B packing, an ordered remaining-token
realization of Elk D rings, Hawk A/B qualification and C visibility, Hawk D
maximum-weight matching, Fox A/B/C neighborhoods, and Fox D pair matching.
Every selected scoring object is a lower certificate for the represented
production board; maximizing the certificate is exact because the production
decomposition can select every scored object.

Fixed-board validation passed AAAAA 66, CBDDB 84, all A/B/C/D uniform sets,
mixed objectives, all 64 candidate-pilot boards, and finally **all 1,024
rulesets** on frozen connected seed-202 board `(4,4,4,4,4)`. The exhaustive
gate completed in 6.925 seconds on eight file-backed spawn workers with zero
mismatches; canonical row SHA-256
`005153bf58a77d32feca858fc225e04db3101d7008b295eabde9fedecb878f2f`.
The verifier is retained as `tools/verify_all_wildlife_exact.py`.

One first attempt to parallelize this gate from an inline stdin program was
invalid operationally: macOS spawn could not import `<stdin>` and its worker
pool repeatedly respawned failed children. No score was accepted from it.
John explicitly authorized termination; only that foreground process tree was
interrupted. A subsequent process audit found no residual local validation
worker and no wildlife job on john2–john4; their long-lived Bacalhau daemons
were untouched. The permanent file-backed verifier is the root fix.

Free-coordinate calibration then reproduced the known disconnected AAAAA
infeasibility for `(6,1,6,2,5)` at threshold 69 in 12.393 seconds
(145,251 branches, 6,480 conflicts). The difficult CBDDB `(6,0,3,6,5)`
threshold-85 disconnected branch remained `UNKNOWN` after 60.090 seconds
(bound 91, 1,029,719 branches, 167,953 conflicts). Decision: correctness gate
passes; tail-proof performance remains the launch gate.

Two sound count filters were added. Hawk C uses the tight cap-six visibility
edge maxima `0,0,1,3,5,7,9`, derived by summing consecutive-pair bounds over
the three axial projection families. Fox C partitions foxes by their selected
target species, then bounds each cross-adjacency score by the simple planar
bipartite/degree-six edge ceiling. On the frozen 64-row shallow pilot these
reduce mean global-bound gap `16.8125→11.1875`, median count branches above
the incumbent `406→275.5`, mean `389.953→288.016`, and maximum `747→689`.
They are proof filters only, not achieved scores.

Hashes: generalized exact model
`0fd423f6dce7ac4847ff15f16dff9ffe584b4d29f41b98b80821594875e06d5d`;
exhaustive verifier
`d76745ca90b2365f9270b1eb20e3e878300755dec8b42876fcb4fec2293a9dae`;
exact tests
`f870a8dfca6a277916ef863a84c82c3d7528c74db848f7ee6f62ef32f9cf088f`;
independent scorer/bounds
`de8817a123b4e6f688d92817faf8b5b03b2d1126dba2851da8db1fd00ead02f2`;
scorer tests
`d6e8d972d52a242758f792637f34fa97ac65020164592d4877133878fc7b698e`;
Rust candidate/bound engine
`df5888c5baf3e6ccc80ad5ec7b62d5182275b3fc92dc700c9b2377c3e71ecede`.

## 2026-07-23 13:16 — PREREGISTRATION: full all-card incumbent staging

The scorer and exact fixed-board gates have passed, so launch a heuristic-only
incumbent staging pass across john1–john4. This run can improve warm starts but
cannot prove or publish an optimum. Source revision
`9de711ec2dca221dd1a73ffbd9e1f1b494eb1a40`; candidate source SHA-256
`df5888c5baf3e6ccc80ad5ec7b62d5182275b3fc92dc700c9b2377c3e71ecede`;
durable worker SHA-256
`cd252332a8e3910afa315ab8121dfa28d6e523299068ac71ae4e05dabad80394`.

Frozen configuration: lexicographic ranges `[0,256)`, `[256,512)`,
`[512,768)`, `[768,1024)` assigned to john1, john2, john3, john4
respectively; eight threads per host; twelve restarts per ruleset; 100,000
iterations per restart; base seed `20260723`; release Rust build. Each shard
writes an atomic JSON catalog, PID, 30-second heartbeat, terminal exit file,
and log. Preflight must find no existing tag artifact and no wildlife worker.
Collection requires exact disjoint coverage of all 1,024 indices, schema and
configuration agreement, cap/connectivity validation, and independent plus
production rescoring of every board. No exact proof job is displaced.

**13:17 launch.** All four hash-pinned release builds passed. john1 launched
range `[0,256)` at `17:16:59Z`, wrapper PID 47397. The first remote launch
loop was invalid before starting a remote process because zsh did not split a
quoted host/range tuple; SSH rejected the combined string as an invalid
hostname. No remote worker or artifact existed after that attempt. The root
fix uses explicit function arguments. john2/3/4 then launched their planned
ranges at `17:17:12Z`, wrapper PIDs 37975/5482/32540. Startup lines on every
host report the pinned source, exact range, and frozen search settings.
Durable launch ledger:
`cascadiav3/fleet/all_cards_candidates_full_20260723_fleet.json`.

**13:28 terminal/collection verdict.** All four shards exited naturally with
code 0 and exact 256-row coverage. Elapsed times were 594.223, 609.068,
594.291, and 600.589 seconds on john1–john4. The collector validated all
1,024 direct boards, then cross-scored them together with 2,478 retained
AAAAA/CBDDB candidate and exact-catalog boards: 3,502 source boards total.
Cross-scoring strictly improved 444/1,024 ruleset incumbents. Every merged
winner passed cap/connectivity checks, the independent scorer, and the
production Rust oracle. Merged catalog SHA-256
`72c2d30839267ddc2a82134c832c0a504a9a7b82aa7d0b94d335b8fb2a75f46c`;
production-response SHA-256
`76bf1c58388a5e01f7f9dd7138f74ff6e20fc9ccdb33b10540261c2790f32683`.

Heuristic scores span 65–85. Eight rulesets currently tie at 85; these are
incumbents, not optimum claims. The strongest two layouts score
`0/21/10/27/27` at counts `(0,6,3,6,5)` and
`20/0/17/27/21` at `(6,0,5,6,3)`. No incumbent reaches its sound global
count bound. After the new Hawk-C and Fox-A/B/C filters, gap distribution is:
one ruleset at 1, three at 2, 23 at 3, and the remaining 997 at 4–17; mean
gap 8.3203. Median/mean/max count branches above an incumbent are
125/158.780/612. Decision: accept the merged boards as the frozen proof warm
starts; exact certification remains required for every row.

## 2026-07-23 13:29 — Stronger Fox proof filters and global runner

Two additional sound neighborhood relaxations landed after the pinned
candidate binary had launched (so they did not alter that run). Fox A now
enumerates per-fox observed-species masks subject to the universal capacities
that a target pair has at most two common hex neighbors and any target tuple
of size at least three has at most one. Fox B caps doubled-species
qualifications because one target-token pair can qualify at most two foxes.
Together with Hawk C and Fox C, the frozen 64-row pilot mean gap is now
10.9844, median/mean/max surviving count branches 268/278.75/689. The AAAAA
global count ceiling tightens 73→72 and its score-69 frontier 128→108;
CBDDB tightens 100→99 and its score-85 frontier 332→309.

The resumable all-card global proof runner uses fixed-threshold feasibility,
atomically checkpoints every count attempt, retains infeasibility
certificates across incumbent improvements, and fail-closes on every
`UNKNOWN`. Candidate merge and verification tools preserve shard/library
hashes and production-rescore all final rows. Unit tests cover score-matrix
composition, proof-completeness bookkeeping, and feasibility-mode witness
scores. These tools must pass a stratified runtime calibration before the
four-host exact launch.

Source hashes: merger
`fa6d6977704158a9f57bdf6b2e0a59274ae56009591a7f55200448ca00d60361`;
merged verifier
`45c6249c41c6e3da842346db77ae8e413f32c8f28aef0696c19b5f98dce979b9`;
global proof runner
`3c7ca3d9d840eed0505bfd8aa2c8a0d03d36c164058cb2d61db344c4d6ed63d7`;
exact model
`4c85c3c447879961e065cc400ba3b8195a0df8549dee10e3cbec7b3a2dc6855e`;
independent bounds
`48cfe51e750cdbc755a1770d6b161d2551c066c14ca0fd0e70126db4f022d2d8`;
future candidate engine
`8510a450caab1db02f2c4d1541716367e8e3cdeb185fbe12b98b79960dec99ef`.

## 2026-07-23 13:31 — PREREGISTRATION: stratified all-card proof calibration

Before launching 1,024 exact ruleset proofs, calibrate the committed resumable
runner on eight frozen objectives spanning the current proof frontier:

- john1: ADACA/index 200 (gap 1, 6 count branches), then AAAAA/index 0
  (known exact reference, 108 branches under the new bound);
- john2: ACACA/136 (gap 2, 8 branches), then CADDA/572 (median 125);
- john3: ADCCB/233 (85-point leader, 94 branches), then DDDDD/1023 (141);
- john4: CADAC/562 (maximum 612), then CBDDB/637 (309).

Each ruleset gets at most 300 seconds total, 30 seconds per fixed-count
feasibility attempt, eight CP-SAT workers, connected boards required, and the
merged catalog as its starting witness. One ruleset runs at a time per host.
Source revision `4f4e59c6dd1a5c5489483889c2b404a15ed49075`;
merged candidate SHA-256
`72c2d30839267ddc2a82134c832c0a504a9a7b82aa7d0b94d335b8fb2a75f46c`;
proof runner
`3c7ca3d9d840eed0505bfd8aa2c8a0d03d36c164058cb2d61db344c4d6ed63d7`;
exact model
`4c85c3c447879961e065cc400ba3b8195a0df8549dee10e3cbec7b3a2dc6855e`;
worker
`6a6bb7e6acfca8963b1c2f420aa251a392601bb0b4794f483a0032f3d9744a1b`.

Decision rule: 6–8 complete exact rulesets validates the five-minute
production shape; 3–5 requires one measured optimization/recalibration pass;
0–2 rejects the generic production shape in favor of additional specialized
filters. Regardless of the band, preserve every exact count exclusion and
connected witness; `UNKNOWN` remains unresolved. Do not read partial scores
while a host's two-rule calibration shard is live.

**13:32 launch.** All four candidate/source/runtime/hash preflights passed.
john1–john4 launched their registered two-index sequences at
`17:32:41–42Z`, wrapper PIDs 52972/39786/6866/32718. Each worker owns a
durable per-ruleset ledger, PID, 30-second heartbeat, and terminal marker.
Launch ledger:
`cascadiav3/fleet/all_cards_proof_calibration_20260723_fleet.json`.

**13:43 terminal verdict — generic shape rejected.** All four workers exited
naturally with code 0; john1/2 finished at `17:39:11Z` and john3/4 at
`17:43:42Z`. Collection waited for all four terminal markers. The final
collector then validated candidate/proof identities, cap six, connectivity,
independent scores, and unresolved-branch bookkeeping for all eight rows.

Exactly **2/8** rulesets completed: ACACA 76 (8/8 exclusions, 57.975 solver
seconds) and ADACA 77 (6/6, 43.776 seconds). This lands in the preregistered
0–2 band, so the generic five-minute production shape is rejected. No
1,024-rule production proof was launched.

The incomplete rows remain valid incumbents, not optima: AAAAA 68 retained
30 unresolved counts after 78 infeasibilities and three timeouts; ADCCB 85
retained 91; CADAC 66 retained 612; CADDA 78 retained 125; CBDDB 84 retained
309; DDDDD 78 retained 141. The latter five each exhausted approximately
300 seconds after only 10–12 attempts; hard attempts overwhelmingly consumed
the full 30-second per-count cap. All exclusions and witnesses are preserved
under
`cascadiav3/fleet/collected_all_cards_proof_calibration_20260723/`.
The eight proof SHA-256 values and exact per-row totals are frozen in the
completed launch ledger. Decision: build a sound specialized bound/filter
pass, measure it on these same frozen rows, and preregister a new calibration
before any full launch.

## 2026-07-23 13:55 — PREREGISTRATION: exact Fox-C cross-edge bound

The first specialized filter replaces Fox C's generic planar bipartite
edge ceiling with a cap-six lattice-exact table. For every left/right size
in `1..6`, CP-SAT maximizes cross-adjacency inside a connected cross-edge
component. Such a component has diameter at most `left+right-1`, making the
finite anchored hex disk complete. A deterministic integer DP then combines
connected components and isolated vertices, so the resulting global table
does not assume the optimum itself is connected.

Freeze 36 component proofs as four disjoint nine-pair shards:
john1 `[0,9)`, john2 `[9,18)`, john3 `[18,27)`, john4 `[27,36)`;
120 seconds/component, eight workers, seed `20260723`, OR-Tools `9.15.6755`.
Collection requires exact disjoint coverage, `OPTIMAL` status and equal
objective/bound for all 36 rows, plus left/right symmetry. Source
`tools/derive_hex_bipartite_edge_bounds.py` SHA-256
`3a6ea9eedfaecdae464b3016916a09a0ffe189ce47cf32c7fa266ca0cfe426aa`;
test SHA-256
`ed997ec0f55026489ddf518b6353cfb7a812d898781b326d365fa0d8c77911d8`.
The durable heartbeat/terminal worker is
`cascadiav3/scripts/fleet_hex_bipartite_bound_worker.sh`, SHA-256
`3be5b67e38ce0fc8d915e58bd08cd1ce366441f83c9a16283fd7d959412ba8af`.

Decision rule, fixed before table output: select the new Fox-C bound only if
all 36 proofs complete and it is everywhere no weaker than the current sound
bound, with either a ≥5% reduction in the frozen Fox-C count frontier or a
≥10% reduction for frozen hard case CADAC. Otherwise record it as a valid but
operationally immaterial negative and continue to the next specialized card
interaction. This derivation does not inspect or change any incumbent score.

**13:53 first launch failed before solving — no result.** All four nine-pair
shards exited 1 at the first component. The derivation had reused the
production 20-token adjacency helper, whose loop indexed tokens 2..19 beyond
the smaller component coordinate arrays. No shard JSON was written and no
bound value was observed or accepted. The manual detached-launch command also
mis-escaped its auxiliary wrapper-PID text; solver PID/heartbeat/terminal
files remained correct. The root fixes are a token-count-generic local
adjacency constructor, a one-by-one live solver regression test, and having
the durable worker atomically record its own `$$` rather than relying on the
outer launch shell.

**13:56 v2 preregistration.** Repeat the identical frozen ranges,
configuration, correctness gate, and selection rule under single-use tag
`fox_c_edge_bound_v2_20260723`. Corrected derivation SHA-256
`4f5ad319e8dc890337c07c2b95a206cdbec3308efd1af5762de8320f4f84ed38`;
three-test SHA-256
`c5a804719ef9ea41bc1f319aeb52e5dfaee2169a8fdec2c8ef89238da40f2ec3`;
self-recording worker SHA-256
`e8d24582e61df3e75dae6d35510ed77a02cceded17343c3f3e042b037f9e4753`.
The failed tag remains immutable.

**13:55 v2 launch.** All four idle/runtime/source/tag preflights passed.
john1–john4 launched `[0,9)`, `[9,18)`, `[18,27)`, `[27,36)` at
`17:55:33Z`, wrapper PIDs 60848/42565/9032/33082. All self-written PID files
are valid and all four heartbeats name their live solver PID and exact range.
Durable ledger:
`cascadiav3/fleet/fox_c_edge_bound_v2_20260723_fleet.json`.

**13:56 terminal/selection verdict.** All four shards exited 0 at
`17:56:03Z`. The collector accepted exact disjoint 36/36 coverage, every
component `OPTIMAL` with objective equal to best bound, and independently
solved left/right symmetry. Aggregate solver work was 13.525 seconds,
17,481 branches, and 17 conflicts. Collected derivation SHA-256
`f0825fe804f5a86bebce901e73cf91896180ca67a90581e09571feee88fa780b`;
shard hashes are frozen in the completed fleet ledger.

The exact global cross-edge table for side sizes 0..6 is:

```text
0 0 0 0 0 0 0
0 1 2 3 4 5 6
0 2 4 5 6 7 8
0 3 5 7 9 10 11
0 4 6 9 10 12 14
0 5 7 10 12 14 15
0 6 8 11 14 15 17
```

Against the frozen merged incumbents, the 256 Fox-C rulesets' total surviving
count frontier falls **51,854→15,553 (−70.01%)**. Hard calibration row CADAC
falls **612→361 (−41.01%)**. Both preregistered selection thresholds pass by
large margins, so the exact table replaces the planar relaxation. Across all
1,024 rulesets this alone removes 36,301 of the former 162,591 count
branches (−22.33%) without changing a score. Production bound source SHA-256
`c86c278c8e6f4b9df41c127be2d37ea293a94c19489a8dd6eec69d366cd20376`;
bound-test SHA-256
`4335d8faf490ce6c8398959178a4815a13131884de9f179c3b94679aa7a5c645`.
Because sound count filters now evolve independently of the coordinate model,
the proof identity and fleet preflight also pin the bound/scorer source hash;
the final collector records it. Updated proof runner SHA-256
`7c265ad2f1a44fd3e035f9fc156b389cfba4a0d54f66c8c02c11184a7ba61e30`;
collector
`c09ebc6e1a4893db599fc681b3dd71e4bb27bb0fdf63d6727e81a8daf03c0f4b`;
proof worker
`12e72974fe0dc08789bf57456e8ea6971acd37008b96183c7bee82d5e0d9d93a`.

## 2026-07-23 14:02 — PREREGISTRATION: exact Fox-B qualification bound

Reuse the now-validated finite-component theorem for Fox B. For every
fox/target-species size pair in `1..6`, maximize the number of fox vertices
having at least two adjacent target vertices inside one connected cross-edge
component; combine components and isolates by the same exact DP. In the Fox-B
count relaxation, sum these exact per-target-species qualification capacities
before the existing per-fox scoring DP. This is sound because dropping
cross-species geometric correlations only enlarges the relaxation.

Freeze metric `qualified_left` as four disjoint nine-pair shards on
john1–john4, ranges `[0,9)`, `[9,18)`, `[18,27)`, `[27,36)`;
120 seconds/component, eight workers, seed `20260723`, OR-Tools `9.15.6755`.
The same 36/36 exact-coverage, objective/bound, and independently solved
left/right table checks apply. Generic derivation SHA-256
`eff1c85db870715241f6918fb6ec612599f4811510308646794cde4fcc9bce83`;
four-test SHA-256
`989a6d42d14a59e9a29e16b43157cb78d0fae7205ec0f52c065263d20c84ea29`;
metric-aware worker SHA-256
`3560b1ec2d9445747537d588fd5efc7e855e4a2f860dfa0bc52f673ff880f0ea`.

Selection rule fixed before output: the table must be everywhere no weaker
than the current target-pair/common-neighbor capacity and must reduce either
the frozen 256-rule Fox-B count frontier by ≥5% or one of frozen hard rows
ADCCB/CBDDB by ≥10%. Otherwise retain it as a proved negative. No incumbent
score is inspected or changed by this derivation.

**14:01 launch.** All idle/tag/source/runtime preflights passed. The four
frozen ranges launched at `18:01:11–12Z`, wrapper PIDs
63519/43235/9538/33155, with metric `qualified_left` present in every worker
environment and fresh self-recorded PID/solver/heartbeat metadata. Ledger:
`cascadiav3/fleet/fox_b_qualification_bound_20260723_fleet.json`.

**14:02 collection assertion corrected before table publication.** All 36
ordered component solves were terminal and exact, but the collector stopped:
the preregistration had mechanically carried over left/right symmetry from
the edge-count metric. `qualified_left` is intentionally asymmetric under
side exchange, so that assertion is mathematically inapplicable. No combined
table or frontier statistic had been emitted. The root fix applies symmetry
only to metric `edges`; ordered-pair coverage plus objective-equals-bound
remain the qualification correctness gate. A new asymmetric-table regression
test passes. Corrected collector SHA-256
`3d77ead3c6a50a8544db06bfbe9d2ba25595e82310d33209ecf107fd1a8bec4c`;
five-test SHA-256
`4311246fec5461bb0301b987acc7e111946051fa0633aaa525553af60fdf8c07`.
Re-collect the same immutable four shards; do not rerun or alter a solve.

**14:03 terminal/selection verdict.** All four shards had exited 0 by
`18:01:42Z`. The corrected collector accepted exact ordered 36/36 coverage
and objective-equals-bound for every component. Aggregate solver work was
0.911 seconds, 19,366 branches, and 41 conflicts. Derivation SHA-256
`3609ce2218007982ea823994c8d5d00bcafe2a8699cfe3c151210c11f50ed79e`;
shard hashes are frozen in the completed fleet ledger.

The exact global maximum qualified-fox table (rows fox count, columns one
target-species count, 0..6) is:

```text
0 0 0 0 0 0 0
0 0 1 1 1 1 1
0 0 2 2 2 2 2
0 0 2 3 3 3 3
0 0 2 4 4 4 4
0 0 2 4 5 5 5
0 0 2 4 6 6 6
```

The frozen 256-rule Fox-B frontier falls **65,519→59,188 (−9.66%)**,
passing the ≥5% selection gate. ADCCB falls 94→71 (−24.47%), also passing;
CBDDB falls 309→283 (−8.41%). The exact table is selected. Combined with the
Fox-C table, specialized filters have now removed 42,632 of the original
162,591 all-rule count branches (−26.22%), leaving 119,959. Production bound
source SHA-256
`d8c78ea6ae37cee14c072b7a0fa55919a940e85854bdf66b990721b4d86fa820`;
test SHA-256
`d71f8eed9a73db8586906a2fcdb3594df0944fbc202367c34e2e5379f2df758e`.

## 2026-07-23 14:10 — PREREGISTRATION: exact Fox-A dual-observation bound

Strengthen Fox A's common-neighbor mask relaxation. For each cap-six triple
`(foxes, first-target count, second-target count)`, exactly maximize foxes
adjacent to at least one token of both target species inside one connected
cross-edge component. A complete three-dimensional component DP covers
disconnected support and isolates. The resulting capacity replaces the loose
`min(foxes, 2ab)` bound for every two-species subset; the existing sound
three/four-species tuple capacities remain.

There are 216 ordered triples. Freeze ranges `[0,54)`, `[54,108)`,
`[108,162)`, `[162,216)` on john1–john4; 120 seconds/component, eight
workers, seed `20260723`, OR-Tools `9.15.6755`. Collection requires exact
ordered coverage, `OPTIMAL` plus objective-equals-bound for all 216, and
independently solved symmetry under exchange of the two target classes.
Derivation SHA-256
`24d88c9c4dafacaf7a540960e2a2f396d9643b58c94f87be906a40dd8a74fe52`;
shared lattice-support SHA-256
`2e0485daecc81d8000efae94155155f4bfbeb8549aee1f80c1ee79cb7d85e3d7`;
three-test SHA-256
`08e50f6b29caae40224dc77f1a77471a4fef8557319e2bcf14b69c68c0bc9667`;
worker SHA-256
`8faceb2a35adeb71e217e460bd66fc55f2f67150da0bc181997a7bb26cf2a3d9`.

Selection rule fixed before output: everywhere no weaker than
`min(foxes,2ab)`, and either ≥5% reduction in the frozen 256-rule Fox-A
frontier or ≥10% reduction for AAAAA or CADDA. Otherwise retain as a proved
negative. No incumbent score changes.

**14:08 launch.** All idle/tag/source/support/runtime preflights passed.
The four 54-triple shards launched at `18:08:04–05Z`, wrapper PIDs
66899/44138/10298/33244, with fresh self-recorded solver heartbeats. Ledger:
`cascadiav3/fleet/fox_a_dual_observation_bound_20260723_fleet.json`.

**14:09 component-absence semantics corrected before collection.** All 216
models reached terminal exact statuses: 194 `OPTIMAL`, 22 `INFEASIBLE`, and
zero `UNKNOWN`. The two workers owning impossible large-class components
exited 2 because the shard wrapper had equated “component exists and is
optimal” with “component subproblem resolved.” For example, a connected
cross-edge support containing one fox and twelve target vertices cannot
exist. Exact `INFEASIBLE` therefore means that component type is absent; the
global DP still represents the requested counts using isolates or smaller
components. No table had yet been collected.

The corrected collector accepts only `OPTIMAL` with equal objective/bound or
exact `INFEASIBLE`, maps absent component types to no DP transition, and
continues to fail on `UNKNOWN`. A live one-fox/six/six infeasibility test and
an infeasible-component collector/DP regression now pass. Corrected source
SHA-256
`81a78064080846b03539ff97f0bd34e7bcd10014a0d227fe1a666592d39d08ce`;
five-test SHA-256
`8764b189a85437ae53bedc9aea6eb54b5f71efab39e823004bba94b71d571035`.
Re-collect the immutable shards; do not rerun a solver.

**14:10 terminal verdict — exact but not selected.** The corrected collector
accepted all 216 resolved component types (194 `OPTIMAL`, 22 exact
`INFEASIBLE`), target-class symmetry, and every available objective/bound.
Aggregate work was 12.269 seconds, 521,203 branches, and 15,112 conflicts.
Derivation SHA-256
`2205a311197563b3ece3c9487791df7d1c52a98a19ae95ddd5603a6f19256931`;
shard hashes are frozen in the completed ledger.

The global 7×7×7 result equals `min(foxes, 2ab)` in **all 343 entries**.
Thus the existing Fox-A pair common-neighbor capacity was already exact;
frontier reduction is zero and neither selection threshold passes. Do not
integrate a redundant table. This closes two-species Fox-A capacity as a
source of further improvement; any stronger Fox-A relaxation must couple
three or four target species or couple Fox A to another animal card.

## 2026-07-23 14:15 — PREREGISTRATION: specialized-bound proof recalibration

Re-run the six incomplete members of the frozen eight-row proof calibration
after selecting the exact Fox-C and Fox-B tables. ACACA 76 and ADACA 77
remain the two certified frozen rows; do not rerun or replace them. Fresh
proof identities are mandatory because the bound/scorer source is now
hash-pinned.

Assignments: john1 AAAAA/0 then CADAC/562; john2 ADCCB/233 then CADDA/572;
john3 CBDDB/637; john4 DDDDD/1023. Configuration remains exactly 30
seconds/count, 300 seconds/ruleset, eight workers, connected boards, merged
candidate SHA-256
`72c2d30839267ddc2a82134c832c0a504a9a7b82aa7d0b94d335b8fb2a75f46c`.
Source revision `12b4c8dc53d9618ce7478cf8bbafd9671713b88f`;
proof runner SHA-256
`7c265ad2f1a44fd3e035f9fc156b389cfba4a0d54f66c8c02c11184a7ba61e30`;
exact model
`4c85c3c447879961e065cc400ba3b8195a0df8549dee10e3cbec7b3a2dc6855e`;
selected bounds
`d8c78ea6ae37cee14c072b7a0fa55919a940e85854bdf66b990721b4d86fa820`;
worker
`12e72974fe0dc08789bf57456e8ea6971acd37008b96183c7bee82d5e0d9d93a`.

Apply the original band to the combined frozen eight rows: 6–8 complete
validates the five-minute production shape; 3–5 requires another measured
optimization; 0–2 rejects it. Preserve every exact exclusion and witness;
do not inspect partial scores. No full 1,024-row launch occurs before all six
rerun shards are terminal and the band is applied.

**Prelaunch provenance hardening.** The generalized exact source imports its
coordinate/component primitives from `tools/cbddb_wildlife_exact.py`.
Preflight review found that dependency was deployed but not separately
identity-pinned. No recalibration worker had launched. The proof identity,
fleet preflight, and final collector now pin it explicitly. Exact-support
SHA-256
`362b5d7f82a156579e33c4b2c630c06bff3f45fa08f72a4dc70fe378eadca329`;
updated proof runner
`9a6565e916157329b523fa2553ee9da63e7a5ac533f27e4c90e67b6e434f784c`;
worker
`47e5458ca99a5bc29f31e1f5ad85f48f50614a9de653a01f3ff83f233d694170`;
collector
`2b7298fa1a1e83c1fd320cd086eae0465af8bca3281cf283d4ee0239ee710cf9`.
The frozen cases, limits, and decision band are unchanged.
The proof worker now also atomically records its own wrapper PID, removing the
outer-shell escaping hazard seen in the first lattice-bound launch. Final
prelaunch worker SHA-256
`bcb8217d863184520c98759547a632039f27d9333727ca5564dd310f714fa1bf`.

**14:14 launch.** All candidate/source/support/runtime/tag/idle preflights
passed. john1 launched indices `0,562`, john2 `233,572`, john3 `637`, and
john4 `1023` at `18:14:30–31Z`; self-recorded wrapper PIDs are
71296/44974/10948/33385 and every heartbeat identifies the first live index.
Ledger:
`cascadiav3/fleet/all_cards_proof_recalibration_20260723_fleet.json`.

**14:25 terminal verdict — connected shape still rejected.** All four
workers exited naturally with code 0. Collection waited for terminal state
and validated all six fresh proof identities, incumbents, scores,
connectivity, and unresolved sets. None of the six completed, so with frozen
ACACA/ADACA the combined result remains **2/8**, the original rejection band.
No full launch follows.

The selected tables did materially reduce breadth: AAAAA has 23 unresolved
counts after 85 infeasibilities and two timeouts; ADCCB 64 after seven
infeasibilities/eight timeouts; CADAC 361; CADDA 125; CBDDB 283; DDDDD 141.
CADAC/CADDA/CBDDB/DDDDD each spent essentially all 300 seconds on ten
30-second unknowns. Proof SHA-256 values are frozen in the completed ledger;
all valid exclusions remain under
`cascadiav3/fleet/collected_all_cards_proof_recalibration_20260723/`.

Decision: keep the selected bounds, but reject direct connected solving as
the production architecture. Next measure a disconnected-relaxation
prescreen on these same six rows. Infeasibility in that strictly larger
layout space is a sound connected-board exclusion, and it removes the
380-choice connectivity arborescence that dominates the coordinate model.

## 2026-07-23 14:32 — PREREGISTRATION: disconnected proof prescreen

Measure the specialized technique suggested by both the literature pass and
the earlier AAAAA calibration: solve the same six frozen rows in the strictly
larger space of arbitrary distinct layouts, omitting connected-board
arborescence variables. Any threshold infeasibility in this relaxation is a
sound exclusion for connected boards. A feasible disconnected layout is not
a witness and never changes the connected incumbent.

Use the same host/index assignment, 30 seconds/count, 300 seconds/ruleset,
eight workers, seed, candidates, exact model, and selected bounds as the
connected recalibration; only `connectivity_required=false` changes. Proof
identity now includes that flag, preventing cross-mode resume. Candidate
SHA-256
`72c2d30839267ddc2a82134c832c0a504a9a7b82aa7d0b94d335b8fb2a75f46c`;
disconnected-aware runner
`fe71485120e078cf419cf1bdc482b898b39bdf3ac2bdb493be094a67b8a03a05`;
exact model
`4c85c3c447879961e065cc400ba3b8195a0df8549dee10e3cbec7b3a2dc6855e`;
exact support
`362b5d7f82a156579e33c4b2c630c06bff3f45fa08f72a4dc70fe378eadca329`;
bounds
`d8c78ea6ae37cee14c072b7a0fa55919a940e85854bdf66b990721b4d86fa820`;
mode-aware worker
`03480072ff67259679df9fb110f0eb0dc2c3d1551ea691bbb5644d8911950277`.
Connected/disconnected zero-time identity smokes both fail-closed as
incomplete and record `true/false` correctly.

After all shards are terminal, union only exact disconnected infeasibilities
with the already-frozen connected exclusions. Select a two-stage production
prescreen if this completes at least one additional calibration row or reduces
the six-row unresolved total by ≥20% from the connected baseline 997. A
disconnected `FEASIBLE`/`UNKNOWN` is no evidence. Do not read partial output.

**14:28 launch.** All mode-aware identity, candidate, source, support,
runtime, tag, and idle preflights passed. The frozen four shards launched at
`18:28:57Z`, wrapper PIDs 74738/46781/12276/33575. Every live proof identity
has `connectivity_required=false`; heartbeats are fresh. Ledger:
`cascadiav3/fleet/all_cards_disconnected_prescreen_20260723_fleet.json`.

While the prescreen runs, the final collector was generalized to accept
multiple ledgers for one ruleset, validate each identity/status/unresolved set
independently, choose only a connected validated incumbent, and union only
exact infeasibility thresholds across connectivity modes. A synthetic
connected/disconnected two-exclusion union test passes. Collector SHA-256
`65096a1e2be39b395efe522a64eb374a0f3a88d4d7388f7317dfb813dc6b4a60`;
four-test SHA-256
`6067392486d14f8596b2e8571c4e192981887be8860c212ad7b2c66e7453fdec`.
