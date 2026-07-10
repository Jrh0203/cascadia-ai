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
