# Gumbel Self-Play Campaign: The 100-Point Plan

This is the canonical plan for pushing CascadiaFormer past a 100 mean seat
score in 4-player Cascadia. It replaces the rollout-teacher expert-iteration
line (EI-0/EI-1) as the active strategy. Implementation landed 2026-07-02.

**Rules update, 2026-07-09:** the free three-of-a-kind wildlife wipe is an
optional policy action followed by a chance draw. Gumbel values accept over
public-hash-derived hidden samples, commits accept/decline before revealing the
real replacement market, and makes the same ordered decision at interior
plies. `--gumbel-market-decision-samples` controls the independent chance
estimate (default `8`); it is intentionally separate from the ordinary search
determinization count. All earlier measurements used forced acceptance and require
rebaselining; see [`RULES_CONTRACT.md`](RULES_CONTRACT.md).

## 1. Diagnosis: why the previous method plateaued

Measured trajectory before this campaign: greedy `87.56` → EI-0 q-head
no-search `89.62` → full K64/R16 rollout search `96.98`. Four structural
problems capped it there:

1. **The teacher was flat one-ply Monte Carlo with greedy playouts.**
   Per-action Q = mean of sampled-greedy rollouts to terminal. Its ceiling is
   the greedy rollout policy itself: any plan greedy will not complete
   (multi-turn elk lines, hawk spacing, held nature tokens) is misvalued by
   construction. The K64/R32 ceiling test proved it: doubling samples per
   action changed the 20-seed mean by `-0.14` while doubling cost. Variance
   was not the bottleneck; rollout-policy bias was.
2. **Hidden-information leak.** Search rollouts advanced clones of the live
   `GameState`, including the true hidden tile-stack/bag order, so
   serving-time search quietly peeked at future draws. All historical
   search-integrated numbers (95.8 / 96.4 / 96.98) carry this leak and are
   marked legacy-leaky; honest baselines get re-measured (Phase A).
3. **Greedy-ranked candidate menus.** K32/K64 menus were greedy's top
   actions, so the teacher could never label — and the model could never
   learn — actions greedy ranks poorly, biasing against exactly the
   non-greedy plans the transformer exists to find.
4. **Tiny, correlated, over-epoched data.** 20k roots from 250 games
   (80 consecutive plies each), trained ~240 passes/example; the guarded
   checkpoint landed at step 7,250 of 25,000 — an overfitting canary.

## 2. What was implemented (2026-07-02)

### Search: `cascadiav3/real-root-exporter/src/gumbel.rs`

Gumbel AlphaZero-style root search with neural leaf values:

- Root menu = **full legal action set** (optional model-prior-ranked cap via
  `--gumbel-max-root-actions`; never greedy-ranked).
- Gumbel top-m + sequential halving over model policy logits;
  `sigma(q) = (c_visit + max_visits) * c_scale * minmax(q)`;
  improved policy = `softmax(logits + sigma(completed_q))` over the whole
  menu; completed-Q for unvisited actions falls back to the model's own
  derived final Q.
- **No-peek by construction**: every simulation redeterminizes hidden
  stack/bag order (`redeterminize_hidden`) *before* applying the root action;
  determinizations cycle over a fixed stream (common random numbers across
  actions; pinnable via `determinization_seed` for paired evals).
- Interior plies advance every seat by argmax of its own derived final Q
  (`exact_afterstate + predicted score-to-go`) — max^n play under the model.
- Leaf value = `w * max-Q bootstrap + (1-w) * sampled greedy terminal
  rollout`. `w` (`--gumbel-blend-weight`) is the trust ramp: 0.5 while the
  value head is still rollout-trained, 1.0 once it has seen real outcomes.
- Simulations advance in lockstep; every ply of every live simulation lands
  in **one batched model evaluation** (`eval_batch` protocol), moving search
  cost from CPU rollouts to the GPU.

### Data: `--gumbel-selfplay-tensor-corpus` (schema v2)

All four seats play via Gumbel search with root exploration noise; **every
visited root is a training record** (the search that plays the game produces
the labels — no separate labeling pass):

- `per_action_Q` = completed-Q, `per_action_score_to_go` = completed-Q −
  exact afterstate, `q_valid` = visits > 0, `per_action_Q_variance` =
  per-action simulation variance;
- new `improved_policy` (soft policy target) and `search_root_value`;
- `final_score_vector` / `rank_vector` / `score_decomposition` are backfilled
  from the **actual game outcome** after terminal — real-outcome value
  targets replace rollout means.
- Schema `cascadiav3.expert_tensor_shard.v2`; v1 modes unchanged and byte
  identical (golden test); filter/relation-tail tools pass the new fields
  through (retained improved-policy slices are renormalized).

### Training: `--objective gumbel-selfplay`

- Policy loss = soft-target cross-entropy against `improved_policy`
  (≡ KL up to a constant), replacing the one-hot/softmax(Q/8) blend.
- Value loss up-weighted (0.5): the head now sees real outcomes and search
  bootstraps on it. No greedy-retention terms.
- `--max-example-passes` clamps steps so a corpus cannot be looped hundreds
  of times (EI-0 regression guard); cycles run at ≤4 passes.

### Evaluation

- `torch_benchmark_stats.paired_delta_stats`: mean, SE, 95% t-CI and
  bootstrap CI for paired per-seed deltas; the search benchmark reports it,
  and all promotion decisions require **CI excluding zero**, not point
  deltas (the 20-seed prefilter run had paired SE 0.42 — sub-point deltas at
  n=20 are noise).
- `torch_cascadiaformer_gumbel_benchmark`: paired harness, Gumbel candidate
  (Rust process per seed slice, own bridge session, `--jobs` parallel) vs
  the full rollout-search control with `--rollout-determinize` (honest, no
  peek) by default.
- 20-game runs are demoted to smoke tests; gates run ≥100 paired games.

### Performance fixes

- Afterstate reuse: one clone+apply per candidate serves exact scoring and
  the rollout/simulation base (previously re-cloned + re-applied per
  rollout).
- Batched model bridge (`eval_batch` + `protocol_features` detection, chunked
  at 32 rows/forward) and numpy relation-matrix construction (was pure-Python
  O(seq²) lists per request).
- Self-play exporter reuses bridge sessions per rayon chunk with a
  `--model-sessions` cap on concurrent GPU-resident bridges.

## 3. Campaign phases

All john0 commands run from `/home/john0/cascadia` inside the torch venv,
following the OPERATIONS.md launch/status/fetch pattern. EI-1 (rollout-teacher
model-state bootstrap) was terminated in favor of this campaign; its partial
artifacts are retained as teacher-comparison evidence only.

### Phase A — validate the search (gate first, scale later)

1. **Honest rebaseline.** Re-run the rollout-search control with
   `--rollout-determinize` on the 100-seed set:

   ```bash
   PYTHONPATH=cascadiav3/src python3 -m cascadiav3.torch_cascadiaformer_search_benchmark \
     --manifest cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json \
     --games 100 --retain-k 64 --max-actions 64 --rollouts-per-action 16 \
     --rollout-determinize --include-full-search-baseline \
     --experiment-id honest-rollout-rebaseline-v1
   ```

   Expect the honest mean at or below the leaky 96.98. All subsequent
   comparisons use the honest number.
2. **Gate.** `MANIFEST=<EI-0 guarded manifest> bash cascadiav3/scripts/run_gumbel_phase_a_gate.sh`
   — 100 paired games, Gumbel (n=64, m=16, w=0.5, depth 1) vs honest full
   K64/R16 rollout search.
   - **Win (mean > 0, CI excludes 0):** proceed to Phase B.
   - **Tie:** raise `GUMBEL_N_SIMULATIONS` to 128–256 and/or
     `GUMBEL_DEPTH_ROUNDS=2`; tune `GUMBEL_K_INTERIOR` (8 → 32). Re-gate.
   - **Lose:** per-decision forensics against shadow rollout labels; drop
     `GUMBEL_BLEND_WEIGHT` to 0.25 (rollout-heavier teacher) and proceed to
     Phase C anyway — the value head was trained on rollout means and is the
     expected weak link; Phase C exists to fix it.

### Phase B — ceiling probe (decides where the bottleneck is)

`MANIFEST=<manifest> bash cascadiav3/scripts/run_gumbel_ceiling_probe.sh`
— 512 simulations, top-m 32, w=1.0, 8 determinizations, 20 paired seeds, no
timing constraint.

- **Mean ≥ 100:** the model already supports 100-point play; the campaign
  becomes search-budget engineering + distillation (jump toward Phase D
  with intermediate EI cycles only as needed).
- **97–100:** model-bound but close; 2–3 self-play cycles projected.
- **< 97:** model-bound; Phase C prioritizes corpus scale and value-target
  quality over search budget.

### Phase C — self-play expert iteration (EI-2+)

Per cycle, on john0:

```bash
MODEL_MANIFEST=<incumbent manifest> bash cascadiav3/scripts/run_gumbel_selfplay_cycle.sh launch
```

- **Data:** 1,250 train seeds × 80 plies ≈ 100k roots/cycle (5× EI-0/1),
  full games, all-seat self-play, exploration on, distinct seed blocks per
  cycle. 125 validation seeds held out.
- **Replay window:** last 2–3 cycles via `EXTRA_TRAIN_TAIL_TENSORS` +
  `TRAIN_SOURCE_WEIGHTS` (e.g. `1.0,0.5,0.25`, newest first).
- **Training:** warm start from the incumbent (`INIT_MANIFEST`), LR 1e-4
  cosine, batch 192, `MAX_EXAMPLE_PASSES=4`, SWA final 20%, selection =
  min locked-val final-Q regret.
- **Blend ramp:** `GUMBEL_BLEND_WEIGHT` 0.5 → 0.75 → 1.0 across cycles as
  the value head is retrained on real outcomes (w=1.0 removes all CPU
  rollouts from data generation).
- **Promotion gates per cycle** (all at ≥100 paired games, CI-excluding-0):
  - no-search q-head vs incumbent q-head;
  - Gumbel-search vs incumbent Gumbel-search (same budget);
  - the existing 95/97/100 ladder from TRAINING_PIPELINE.md applies at these
    sample sizes; the 100 gate keeps its 1,000-game confirmation run.
- **Rejection is not a halt:** a rejected candidate stays in the opponent
  pool; the incumbent remains champion; the next cycle's data still comes
  from the incumbent.
- **Throughput fallback:** if per-cycle generation exceeds ~12h wall-clock,
  switch topology to one shared bridge server (single Python process, many
  Rust workers over a socket) before buying model-quality compromises; the
  `--model-sessions` cap is the interim knob.

### Phase D — distillation and speed (only after a ≥97 checkpoint exists)

- Retention/prefilter tuning and searchless q-head distillation (the
  searchless-chess route) with the Gumbel teacher as the distillation
  source. Explicitly out of scope until the 97 gate passes.

## 4. Decision log

| Date | Decision | Basis |
|---|---|---|
| 2026-07-01 | Stop scaling rollout count/width | K64/R32 `-0.14` vs R16; K56 forensics |
| 2026-07-02 | Kill EI-1; pivot to Gumbel self-play | Rollout teacher ceiling; EI-1 kept both binding constraints (greedy rollouts, greedy menus) |
| 2026-07-02 | Fix + rebaseline the hidden-order leak | Serving search must be public-information-legal; old numbers marked leaky |
| 2026-07-02 | Gates require ≥100 paired games with CI | Paired SE ~0.42 at n=20 makes sub-point deltas unreadable |

## 5. Run bookkeeping

Every run gets an `EXPERIMENT_LOG.md` entry (purpose, config, artifacts,
result, decision), same conventions as before. Phase gates append their JSON
reports under `cascadiav3/reports/` and the markdown summaries are the
human-readable record. PERFORMANCE.md collects measured throughput deltas
(afterstate reuse, batched bridge, search decision seconds) as they land on
john0 hardware.
