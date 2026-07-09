# Campaign Working State (updated 2026-07-09)

Live working notes for the Gumbel self-play campaign. Companion to
[GUMBEL_SELFPLAY_CAMPAIGN.md](GUMBEL_SELFPLAY_CAMPAIGN.md) (strategy) and
`cascadiav3/EXPERIMENT_LOG.md` (per-run records). Update this file whenever
the in-flight picture changes.

## RESUME HERE (07-09 rules correction — rebaseline before research resumes)

The official rules audit found a load-bearing policy-space error: the free
three-of-a-kind wildlife refresh is optional, but every automated policy had
forced it. The corrected engine exposes decline and accept; greedy, pattern,
lookahead, API suggestions, Gumbel root search, and Gumbel interior plies now
choose from public information, sample the replacement chance outcome, and
only then draft from the revealed market. Unplaced drafted wildlife is pinned
to return to the bag before the end-of-turn refill. See
`docs/v3/RULES_CONTRACT.md`.

**Compatibility ruling:** every existing score baseline, paired battery,
corpus, and checkpoint was generated under the forced-refresh policy. Preserve
them as historical architecture evidence, but do not use them as promotion
controls. Before resuming EI or a 100-point claim, regenerate greedy,
no-search, n256/d4, and n1024/d16 baselines with the new rules identity and
fresh promotion seeds.

**Live corrected rebaseline on john0 (auditable restart 07-09 01:37 EDT):** PID
`1265148`, PGID `1265141`, source revision
`d20daf44dc6aa4aad3d03c6ccb7d3a21c3013135`, seeds `2027070900..2027070999`,
and `market_decision_samples=8`. The one-game n16/d2 smoke passed and recorded
the corrected rules ID plus exact source revision, all 80 per-ply decision
rows, and refresh telemetry: 7 opportunities, 5 accepts, 2 declines. The job
completed the 100-game greedy/no-search floor and is now running cycle4
n256/d4, followed sequentially by distq_k8 n256/d4 and both models at
n1024/d16 on the same fresh seeds. Log/pid:
`cascadiav3/logs/rules_20260709_rebaseline.{log,pid}`. Canonical launcher:
`cascadiav3/scripts/run_rules_20260709_rebaseline.sh`; every completed report
is reused only when both rules ID and source revision match.
Verdict watcher PID `1268022` waits for the chain and then writes
`rules_20260709_rebaseline_verdict.{json,md}` with paired distq-minus-cycle4
intervals at both budgets plus within-model scaling deltas.

**Corrected no-search floor (100 paired seeds, complete):** greedy `87.5450`;
cycle4 policy head `91.8425`, delta `+4.2975`, 95% t-CI
`[+3.8705,+4.7245]`; cycle4 Q head `90.8925`, delta `+3.3475`, CI
`[+2.8507,+3.8443]`. All 24,000 decisions are retained. Optional refresh
accept/decline counts were policy `594/352`, Q `636/364`, and greedy
`1005/398`. The interactive no-search harness uses greedy-v1 for this
pre-draw market decision, then the named model head ranks the revealed draft;
the Gumbel legs are the model/search-driven refresh-decision evidence.

The old forced-refresh EI-1 generation and queued battery were stopped before
deployment (PGIDs `1225249` and `1228689`). Its 825 partial games/66k roots
are quarantined as legacy and are not inputs to this campaign.

**First exactness ablation (K1, implemented 07-09):**
`--gumbel-exact-endgame-turns 1` replaces model/search on each seat's final
personal turn with complete-menu engine scoring. It still decides an optional
three-of-a-kind refresh over hidden replacement samples before exposing the
real draw. Exact rows are explicit in telemetry, use zero simulations, and
ignore the normal root-menu cap. Unsupported K>1 and table-total combinations
fail loudly. Full local gates passed (43 exporter tests, 106 Python tests with
45 fixture-dependent skips, release build, and workspace check). MPS exposed
two useful invalidation modes: a cross-host pair diverged at ply 5, and even a
same-host two-worker pair diverged at ply 24; a four-worker exact arm also hit
a Metal command-buffer OOM. None is score evidence. The final one-worker,
two-seed john4 smoke passed the causal comparator: plies 0–75 were identical,
all 8 K1 decisions used zero simulations, seat 0 did not regress, and 6/8
final actions changed. Score was exactly flat (`92.25` both arms, per-seed
deltas `0/0`). K1's own eight decisions were `8.86x` faster (`4.212s` to
`0.476s` total), but whole-arm wall/mean-decision time improved only `1.3%` /
`1.2%`. This is engineering evidence only. A checksum-verified waiter is
armed on john0 (pid file
`cascadiav3/logs/exact_k1_waiter_main.pid`): only after the current rebaseline
and verdict watcher exit will it install the exact revision-marked `main`
snapshot, rebuild, and run a fresh same-revision 100-seed corrected n256/d4
baseline/K1 gate.

**Optional-refresh performance ablation (07-09):** a 65-game streamed profile
of the live corrected cycle4 n256/d4 arm found that 611 refresh-available
decisions averaged `55.452s`, versus `5.968s` for 4,589 ordinary decisions.
Refresh evaluation added 1,343,744 simulations above 1,331,200 chosen-branch
simulations; action count had essentially zero latency correlation. Serial
sample-count screens on john2 and john3 were rejected because MPS traces
diverged before their first refresh opportunity. The valid two-seed john4
frontier made sample-4 the only non-dominated reduced count: score
`93.875 -> 93.500`, mean decision `1.866s -> 1.476s` (`1.264x`), while
sample-6 and sample-2 were both slower and lower-scoring end to end because
their changed trajectories encountered more refresh opportunities. This is
engineering evidence only. A revision-audited `run_market_samples_gate.sh`
will follow exact K1 on john0, reuse its identical validated sample-8 arm, and
run a fresh 100-seed sample-4 candidate. Passing requires t-CI lower bound
`>= -0.25` and whole-decision speedup `>= 1.15x`; failure leaves sample-8 in
place.

**Mini-fleet audit (07-09):** john2–john4 were still running Fleet5 under the
pre-correction forced-refresh binary for roughly nine hours. Those process
trees were killed and verified absent; no Fleet5 shard artifact existed to
quarantine. john1's Fleet5 pid file was stale and no process/artifact existed.
The minis remain engineering/data-generation workers only, never promotion
gate hosts.

## RESUME HERE (07-08 evening — distq EI-1 + fleet5 running overnight)

**Day's verdicts (all 100g paired):** distq_k8 n256 **+0.43 CI+** (first
training-side win); distq n1024/d16 98.40 vs 98.28 **+0.12 ns**
(champion-equal; gains overlap with worlds ensemble); table-total v1
−1.65 CI− / v2 −1.05 CI− (closed: table-scoring leaves multiply noise);
softmix flat (closed: common-mode bias cancels); TTA×3 flat at 3× cost
(closed: rotation-invariant representations don't decorrelate). Decision
SNR: 46% of decisions noise-flippable (median SNR 1.06).

**Running overnight:** (1) john0 `gumbel_selfplay_distq_ei1_job` —
distq-model generation n512/d8 w1.0 seeds 2026810000x1250/2026910000x125,
then --q-quantiles 8 train (init distq_k8, mix new/c6/c5 1.0/0.5/0.25).
On completion: battery vs distq_k8 AND vs c4 champion (n256 + n1024/d16
legs, seeds 2026995000). Promote on CI+ with no CI−. (2) Fleet5 john1-4
— distq-labeled shards seeds 2026815000+ (150/host); fetch + process but
DO NOT auto-fold (safety trial first). Monitors armed on both.

**Deliverable doc for the user: docs/v3/RESEARCH_LOG.md** (complete:
architecture, all directions, verdicts, lessons).

## PREVIOUS RESUME (07-08 afternoon — DISTQ CI+, chain running)

**Scoreboard (all 100g paired vs 96.95 n256/d4 unless noted):**
- **distq_k8 (quantile q head): +0.43 CI+** — first training-side win
  since saturation; clean ablation vs cycle-6 recipe. Champion-config
  confirm (n1024/d16 vs 98.28) chained.
- table-total v1: −1.65 CI− (per-leaf value-head noise); v2 (constant
  root shift) in flight.
- leaf softmix τ2/τ4: flat ns (common-mode bias cancels) — closed.
- Decision SNR: median 1.06; 46% of decisions noise-flippable.

**Chain on john0:** tablev2 probe (running since 13:59) → tta3 probe
(--gumbel-tta 3, symmetry TTA) → distq n1024/d16 confirm. Monitor armed.
**Staged, launch-gated:** `logs/gumbel_selfplay_distq_ei1_job.sh` (distq
EI cycle, the overnight long-runner if the confirm holds); cycle7_table
job + fleet4 scripts (if tablev2 CI+).
**Ops lesson (cost: ~1.75h GPU):** exporter is NOT covered by `cargo
check --workspace`; a cfg(test)-gated fn broke the non-test build while
tests passed, and job scripts silenced build output. Preflight with
`cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml`
and never >/dev/null the build in job scripts.

## PREVIOUS RESUME (07-08 midday — research program launch)

**User ruling (07-08 ~09:30): research agenda approved** — table-total,
value-noise reduction, lean into what pays, kick off a long experiment at
the end. Deliverable: `docs/v3/RESEARCH_LOG.md` (keep updated).

**In flight on john0 (strictly sequential chain, each waits on the prior
pid file under `cascadiav3/logs/`):**
1. `table_probe_job` (pid file, started 09:38) — 100g n256/d4 w0.5
   **--gumbel-table-total** candidate arm, seeds 2026995000+, report
   `reports/gumbel_table_total_n256.json`. Verdict: `/tmp/pair_verdict.py
   <cand> <base> <label>` vs `reports/gumbel_cycle4_gate_n256.json`
   (96.95). ~2h.
2. `softmix_probe_job` — rebuilds exporter, then 100g each
   **--gumbel-leaf-softmix** τ=2 and τ=4 (n256/d4 w0.5), reports
   `gumbel_softmix_t{2,4}_n256.json`. ~4h.
3. `distq_job` — trains `full_v3_distq_k8` (M, **--q-quantiles 8**,
   warm start champion via --init-skip-mismatched, cycle-6 recipe/data =
   clean ablation vs known-flat control), then 100g battery
   `gumbel_distq_k8_n256.json`. Smoke-tested end-to-end.

**Decision tree on verdicts:** table-total CI+ → confirm at n1024/d16 +
launch fleet table-native corpus generation (RESEARCH_LOG §4.3); softmix
CI+ → stack with best config; distq CI+ → it becomes the new champion
line. Anything CI− → record and drop. New code this session: 51e049e
(table-total), a8e9c32 (softmix), distributional head + init-skip commits.

## PREVIOUS RESUME (07-08 morning — tuning program COMPLETE)

**Honest measured optimum: 98.28 mean seat (100g), c4-M champion at
n1024/top_m16/w0.5/d16, 10.6 s/dec, 11/100 games >=100.** Every tuning
lever is now measured and closed (see EXPERIMENT_LOG 07-06..07-08):
capacity (L flat, fresh-M flat), data (3x flat), labels/EI (saturated),
search shape (n1024_d16 is the peak; worlds axis reverses past 16),
serving blend (w0.5 optimal), ensembles (closed: shared bias).
Key science: determinization = ensemble variance-reduction, NOT
hidden-info approximation (oracle peek LOSES to honest search).

**Decision menu for the user (john0 + fleet idle):**
1. Table-total (gate-aligned cooperative) objective — potentially
   several points; changes what the benchmark means. PENDING RULING.
2. 1,000-game certification of 98.28 (~24h) — certifies the number,
   cannot pass the 100-gate.
3. New research: distributional value head; market-refill chance-node
   expectimax; multi-bridge structural throughput.

## PREVIOUS RESUME (07-07 afternoon)

**Champion: cycle-4 M** (`checkpoints/full_v3_gumbel_selfplay_cycle4/
best_locked_val.manifest.json`) at **SERVING CONFIG n1024/top_m16/
w0.5/d16: 98.28 on 100g** (+0.435 CI+ vs n512d8; 11/100 games >=100;
10.6 s/dec). Gap to 100-gate: -1.72.

**KEY INSIGHT (07-07, oracle experiment):** peeking at the true hidden
state LOSES to honest multi-world search (-0.35 CI-) — determinization
gains are ensemble variance-reduction over noisy value estimates, NOT
hidden-info approximation. Eval noise is the binding constraint. Push
ensembling/value-noise reduction, not belief modeling. Worlds axis
peaks ~16 (n2048_d32 CI-). EI at M fully saturated (cycle-6 flat incl.
n512d8 leg). Fleet n256/d4 data at 0.25 weight is SAFE but has no
customer while EI is saturated (minis idle). OPEN QUESTION for user:
gate-aligned table-total objective (all 4 seats are ours; denial moves
lower the gate metric) — potentially several points if allowed.

**07-06 findings (all 100g-paired unless noted, see EXPERIMENT_LOG):**
- L-v2 (207M, 16-pass): FLAT everywhere -> capacity closed at this
  data scale.
- Cycle-5 (n512 w1.0 labels + fleet mix): n64 CI-; nofleet ablation
  proved **fleet n=128/MPS data at 0.75 weight was the poison**; w=1.0
  exonerated (keeps ~2x CPU savings). Neither promotable; M-class ~97
  plateau held across capacity/data/label-budget levers.
- SERVING BREAKTHROUGH: **determinizations 4->8** is the live lever
  (probes CI+ at 25g; n512_d8 confirmed CI+ at 100g). k_interior gains
  don't stack; d16=d8; n256_d8 fades to ns at power (needs >=64 sims
  per world). Trainer now GPU-bound 0.23 s/step (mmap fix ca5c387 +
  4 workers): full cycles train in ~25 min.

**In flight right now:**
1. **Cycle 6** on john0 (`logs/gumbel_selfplay_cycle6_job.*`, pid
   956506, launched ~18:05, marker
   `reports/full_v3_gumbel_selfplay_cycle6_runbook.json`): teacher =
   c4-M champion with the CONFIRMED d8 search (n=512, top_m 16, d8,
   w=1.0), seeds 2026790000x1250 / 2026890000x125, replay
   c6(1.0)/c5(0.5)/c4(0.25), NO fleet data, trainer 4 workers + mmap.
   Gen ETA ~8-10h (d8 roughly doubles leaf evals; dedup offsets some),
   train ~25 min. When done: battery vs champion — no-search + n64 +
   n256 100g (standard spec) PLUS an n512-d8 100g leg (compare against
   gumbel_confirm_n512_d8.json, same seed block). Promote on any CI+
   with no CI-.
2. **Fleet wave-2** john1-4 (ETA ~19:45): fleet2_shard_johnN.npz,
   seeds 2026780000. FETCH + process (top64 + relation tail) but DO
   NOT fold into training — wave-1's n=128 labels at weight 0.75
   caused a n64 CI- (see 07-06 ablation). Store for low-weight or
   value-only trials. Fleet regime needs redesign before wave-3.

**Queued next:**
- n1024_d8 and n1024_d16 probes (25g) once cycle-6 gen is off the GPU
  (or interleave — probes are CPU-light but share the GPU; keep
  sequential per john0 policy).
- If cycle-6 promotes and/or n1024_d8 lands ~98.5: draft the
  1,000-game 100-gate confirmation plan (at 5.5 s/dec, 1000 games ~
  12-13h with batch runner).
Watchdog: bger035og (cycle-6 terminal/error/stall + fleet2). Verdict
scripts: /tmp/c5_verdict.py pattern on john0 (gumbel:
candidate_per_seed; no-search: paired_score_deltas["q"]).

**Fleet ops (john1-4, no -p flag, different usernames — john1=johnherrick,
john3=john3):** repo at ~/cascadia, venv at ~/cascadia/venv (python3.12 via
uv, torch 2.12.1 with MPS), release binary built, cycle-4 champion weights
present. ~19 seeds/h per mini at n=128/MPS/3 sessions. Launch pattern in
EXPERIMENT_LOG 07-05 entry.

**Serving env (generation):** CASCADIA_CGAB_FUSED=1
CASCADIA_EVAL_CELL_BUDGET=16777216 CASCADIA_BRIDGE_TF32=1. Batteries:
fused yes, TF32 NO. Trainer knobs: --data-workers 4 --prefetch-factor 4
--tf32 --fused-optimizer --cgab-fused (+ --grad-checkpoint on for L).
SDPA already mem_efficient; compile broken on WSL (no cc; try zig-cc).
M wall step 1.69s b192; L-ckpt ~0.57s b192 (46-min 4-pass run).

**100-gate context:** target = mean seat >=100 over 1,000 games.
Current best 97.11. A 1,000-game n256 confirmation ~20h on john0 —
plan only when a champion hits ~99+ at n256/n512 on 100g.

## CHAMPION (2026-07-04 evening): CascadiaFormer-M cycle3 step_0010000

`checkpoints/full_v3_gumbel_selfplay_cycle3_m/step_0010000.manifest.json`
(regret-selected, regret 0.1559). First CI-significant wins of the campaign:
n=64 95.24/95.54 (+0.59/+0.89 vs S, both excl-0), **n=256 97.11** (+1.44
excl-0). S is saturated (n512 flat, depth2 flat, 3 data cycles flat).
M no-search q is WEAKER than S (90.88) — M's strength expresses via search.
**97-GATE PASSED at power (07-04 ~20:30)**: M vs S at n=256, 100 pairs:
96.9125 vs 95.7175, paired +1.1950 CI [0.8306, 1.5594] — promoted. M p90 =
100.0; 2/100 games >= 100 mean seat. Fused-CGAB A/B: EXACT parity (paired
delta 0.0 on 25 games), serving ~6% faster at n=64 (CPU-bound there).
**IN FLIGHT (07-05 evening)**: (1) CascadiaFormer-L (207M) from-scratch on
cycle-4 corpus, john0, `logs/gumbel_selfplay_cycle4_l_job.*`, runbook
marker `full_v3_gumbel_selfplay_cycle4_l_runbook.json`, ETA ~5-7h — the
capacity-repeat experiment; battery vs cycle-4 champion decides the road
to 100. (2) Fleet john1-4 (M4 minis, provisioned, MPS serving, ~19
seeds/h each): 1,000 supplementary n=128 seeds (2026750000, 250/host,
w=0.75, c4 champion teacher) -> `~/cascadia/fleet_shard_johnN.npz`, ~13h;
fetch + fold into next cycle's replay mix. Fleet = training data only.
Serving env adopted for generation: fused CGAB + 8x cell budget + TF32
(batteries keep TF32 OFF). Cycle-4 M promoted champion earlier today
(n64 95.77 CI+; n256 96.95; probes: n512 97.47 CI+, depth2 dead).

**PREVIOUS: Cycle 4 (EI-5)** launched 07-04 ~21:00, pid 718264,
`logs/gumbel_selfplay_cycle4_job.*`, completion marker
`reports/full_v3_gumbel_selfplay_cycle4_runbook.json`: M teacher
(step_0010000), n=256 labels, w=0.75, seeds 2026740000x1250 /
2026840000x125, tails c3+c2 (1.0/0.5/0.25), MODEL_SIZE=M warm start,
regret selection, TRAINER_EXTRA_ARGS="--data-workers 4 --prefetch-factor 4
--tf32 --fused-optimizer --cgab-fused", bridge fused via MODEL_SERVICE env
prefix + 8x cell budget. Gen ETA ~7-10h (measure from progress lines).
When done: battery vs M champion (no-search, n=64, n=256 100g paired) —
promotion = CI+ vs M; then 100-point confirmation run planning (1,000
games) if means approach 100. Engine pass 2 deployed (rollouts +76%).
All optimization passes 1-5 + engine pass 2 now LIVE on john0.

## Scoreboard (all honest / no hidden-order peek)

| Agent | No-search q (100g) | Gumbel n=64 (100g) | Gumbel n=256 (25g) |
|---|---:|---:|---:|
| Greedy | 87.85 | — | — |
| EI-1 (rollout teacher) | 90.07 (500g) | 93.36 | — |
| **Cycle-1 champion** | 91.71 | 94.53 (−0.87 vs control) | **95.62 (−0.04 = parity)** |
| Cycle-2 (rejected) | 91.85 | 94.47 | untested — test at n=256 |
| Honest rollout control | — | 95.40 @ 10.9 s/dec | same |
| Legacy leaky control | — | (96.98 — invalid) | — |

Key facts: value-head regret 0.79→0.21 (cycle 2) did NOT move n=64 gameplay
but budget-scaling is monotone (64→128→256: 94.53→95.11→95.62) → the head
converts budget into strength; campaign is **budget+model-scaling bound,
not data-noise bound**. Gumbel at n=256 = 3.2 s/dec vs control 10.9 s/dec.

## In flight right now

- **CascadiaFormer-M from-scratch training** on john0 (launched ~22:40 07-03,
  pid 607806, `logs/gumbel_selfplay_cycle3_m_job.{log}`): MODEL_SIZE=M,
  scratch init (INIT_MANIFEST empty — S weights can't warm-start M; called
  run_full_v3_training_pipeline.sh directly because the cycle wrapper forces
  INIT_MANIFEST=$MODEL_MANIFEST), REGENERATE_ROOTS=0 reusing cycle-3 tensors
  via `fixtures/full_v3_gumbel_selfplay_cycle3_m_*` SYMLINKS to the cycle3
  files; same steps/batch/objective/selection as cycle 3 for comparability.
  Checkpoint dir `checkpoints/full_v3_gumbel_selfplay_cycle3_m/`. When done:
  battery = no-search 100g (2026994000), Gumbel n=64 100g + n=256 25g
  (2026995000, --batch-runner), PLUS first n=512 25g probe and a
  depth_rounds=2 n=64 25g probe (search-scaling ceiling questions).
- **DECIDED 07-03 evening (cycle-3 gates)**: flat at all budgets — no-search
  91.805 / n64 94.6475 (+0.175 vs c2, ns) / n256 95.67 (all c1/c2/c3 n256
  within noise). Regret 0.152 (best ever) did not convert. MODEL-CLASS BOUND
  at CascadiaFormer-S -> branch 3: model scaling before more data cycles.
  Methodology gap: honest control per-seed never persisted (mean 95.40 only)
  — persist per-seed on the next control re-run.

## Previous in-flight (done)

- **Cycle-3 gate battery** launched ~17:00 07-03 on john0
  (`logs/cycle3_gates_job.{sh,log,pid}`, done marker `ALL_GATES_DONE`):
  (1) no-search 100g seed 2026994000 -> `reports/gumbel_cycle3_no_search_game100.json`;
  (2) Gumbel n=64 100g candidate-only seeds 2026995000 (batch runner)
  -> `reports/gumbel_cycle3_gate_candidate.json`, pair offline vs stored
  control per-seed in `reports/gumbel_phase_a_gate.json`;
  (3) n=256 25g cycle-3 -> `reports/gumbel_cycle3_budget_n256.json`;
  (4) n=256 25g cycle-2 -> `reports/gumbel_cycle2_budget_n256.json`.
  First battery on the batch runner (one shared bridge, --jobs 12).
- **Cycle 3 rerun COMPLETED 16:30 07-03**: generation 17,402s (~4.8h, 3.2x
  pass-2 stack; production dedup 62.5% rows saved: 4.97M requested -> 1.87M
  sent), training 337s, checkpoint
  `checkpoints/full_v3_gumbel_selfplay_cycle3/best_locked_val.manifest.json`,
  locked_val_final_q_regret 0.152 (new best; c1 0.79, c2 0.21).

- **Cycle 3 (EI-4) RERUN** on john0: the first attempt completed all 1,250
  generation seeds (~15.6 h) then DIED writing the train tensor — a single
  npz array crossed the 4 GiB zip entry limit without zip64
  ("Large file option has not been set"); data unrecoverable, fixed by
  `.large_file(true)` in npz_writer.rs (`5e84d7b`). Relaunched 2026-07-03
  ~12:30 with identical seeds/config (1,250+125 seeds, n=128, w=0.5, replay
  cycles 2+1 at 1.0/0.5/0.25, warm start cycle-1 champion) on the
  optimization-pass-2 stack (eval dedup+cache, packed responses) — the rerun
  doubles as pass-2's production measurement (prior stack: 0.022 seeds/s).
  Job: `logs/gumbel_selfplay_cycle3_job.{pid,log}` (pid 555312); completion
  when `reports/full_v3_gumbel_selfplay_cycle3_runbook.json` exists.
  Champion manifest: `checkpoints/full_v3_gumbel_selfplay_cycle/best_locked_val.manifest.json`.
- **Pass-2 production measurement (rerun, john0): 0.069-0.072 seeds/s vs
  0.022 old stack = ~3.2x.** Generation ETA ~5h (start ~12:30 -> ~17:30),
  checkpoint ~18:00, gates after.
- Optimization pass 3 MERGED locally (not yet needed on john0 mid-run):
  (1) batched benchmark harness — `--gumbel-benchmark-batch` Rust mode +
  `--batch-runner` opt-in in torch_cascadiaformer_gumbel_benchmark.py; one
  process + shared bridge for all seeds; per-seed outputs field-identical to
  single-seed mode (test-enforced). USE THIS for the cycle-3 gate battery
  candidate games. (2) forward-path knobs (all default-off):
  `CASCADIA_BRIDGE_BUCKET=1` (shape bucketing; ~2e-7 drift class already
  admitted by chunk padding), `CASCADIA_BRIDGE_COMPILE=1` (torch.compile +
  CUDA warmup), `CASCADIA_BRIDGE_TIMING=1` (per-phase breakdown). Trunk
  factoring verdict: forward is ALREADY factored (trunk runs once per root;
  per-action cost only in cross-attn query + CGAB tail) — no exact win
  available there; next non-exact idea is replacing the CGAB [B,A,S+A,d]
  materialization with relation-count matmuls (reduction-order drift).
  Tune BUCKET/COMPILE/TF32/gather/row-cap on john0 during the gate battery.

## Gate battery to run when cycle-3 lands (sequential, one job script)

1. No-search 100g `--first-seed 2026994000` (compare 91.71 / greedy 87.85).
2. Gumbel n=64 100g candidate-only on seeds 2026995000 (pair offline vs
   stored control per-seed in `cascadiav3/reports/gumbel_phase_a_gate.json`
   using `cascadiav3.torch_benchmark_stats.paired_delta_stats`).
3. Gumbel n=256 25g for BOTH cycle-3 and cycle-2 checkpoints (cycle-2's 4x
   regret may convert at high budget even though n=64 was flat).

Promotion: CI-excluding-zero paired improvement. Rejected candidates join
the opponent pool; champion stays.

## Decision tree after cycle-3 gates

- **n=256 beats control (CI+)**: search has passed the honest baseline →
  push budget (n=512, depth_rounds=2) toward the 95/97 gate ladder
  (TRAINING_PIPELINE.md gates apply at >=100 paired games; the 97-gate
  needs +0.25 over incumbent at 250-500 pairs). Then cycle 4 with n=256
  labels.
- **Cycle-3 flat at n=64 but scales at n=256**: same as above; serving
  answer is budget; consider CascadiaFormer-M (config exists, model-size M)
  for the next training to raise the model ceiling — data pipeline
  unchanged, just MODEL_SIZE=M + more steps (grad-ckpt already configured).
- **Everything flat incl. n=256**: model-class bound → CascadiaFormer-M
  and/or depth_rounds=2 experiments before more data cycles.
- 100-point definition: mean seat score >=100, 1,000-game confirmation run
  (TRAINING_PIPELINE.md 100-gate).

## Throughput facts (optimized stack, deployed on john0)

- Generation: n=64 labels ~278 games/h; n=128 labels ~80 games/h (evals
  dominate post-optimization; budget costs ~3.5x not 2x).
- Optimization pass 2 (2026-07-03, merged locally, NOT yet on john0): eval
  dedup+cache (43.7% of eval rows eliminated at production shape) + packed
  responses (7.7x encode / 2.9x decode) + TF32/bf16/shared-bridge env knobs.
  See PERFORMANCE.md "Pass 2". Deploy to john0 AFTER cycle-3 job completes,
  BEFORE the gate battery; measure real throughput there.
- Optimizations landed (all bit-parity gated): packed-features protocol
  (8.4x collate), engine pass (2-3.6x rank, 2.2-2.6x rollouts), shared
  aggregated bridge (`SHARED_MODEL_SESSION=1`, MODEL_SESSIONS=16 = parallel
  games). Owned-bridge mode remains for benchmark harnesses.
- 12+ owned CUDA contexts thrash the box (near-stall). Shared bridge fixed
  this. Jobs on john0 run STRICTLY SEQUENTIALLY (concurrent jobs strangle
  each other through GPU round-trip queueing).

## Operational knowledge (john0 + local)

- ssh -p 2222 john0; repo /home/john0/cascadia; venv
  `source /home/john0/venvs/torch/bin/activate`; always
  `export PYTHONPATH=cascadiav3/src PYTHONDONTWRITEBYTECODE=1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Remote cargo needs: `. ~/.cargo/env` + `BLAKE3_NO_ASM=1` +
  `CC=/home/john0/.local/bin/zig-cc` +
  `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=` same (no system cc).
- Local cargo needs `PATH=$HOME/.cargo/bin:$PATH RUSTUP_TOOLCHAIN=1.96.0`
  (homebrew rustc 1.85 too old). Local python for tests: `.venv-v3/bin/python`
  (repo root venv, has torch); system python3.13 lacks numpy.
- Job pattern: write `logs/<name>_job.sh` heredoc on john0, nohup, pid file,
  poll via ssh loop in a Monitor (grep -a; logs can be "binary").
  Kill order: job pid, then `pkill -9 -f gumbel-selfplay-tensor-corpus`,
  then `pkill -9 -f torch_inference_bridge` (bridges via sh -c, ppid checks
  unreliable — pkill by name only when nothing else runs).
- Benchmark harnesses use owned bridges + `--model-manifest` required
  (no --allow-model-fallback: fail-loud by design).
- macOS rsync 2.6.9: ONE remote source per command (use scp for multiple).
- Seed-block allocations used so far: cycle1 train 2026710000/val 2026820000
  (60) — NOTE cycle1 actually used TRAIN_FIRST_SEED default 2026710000 with
  120 seeds, val 2026810000x30; cycle2 2026720000x400 / 2026820000x60;
  cycle3 2026730000x1250 / 2026830000x125; gates no-search 2026994000x100;
  gumbel gates + sweeps 2026995000 (Phase A control per-seed stored).
- Replay tensors: `fixtures/full_v3_gumbel_selfplay_cycle{,2,3}_train_tensor_top64_relation_tail.npz`
  via `EXTRA_TRAIN_TAIL_TENSORS` + `TRAIN_SOURCE_WEIGHTS` (newest first).
- Runner: `cascadiav3/scripts/run_gumbel_selfplay_cycle.sh launch` does
  rsync+preflight+detached run. Key env: MODEL_MANIFEST, PROFILE, JOB_SLUG,
  SHARED_MODEL_SESSION, MODEL_SESSIONS, GUMBEL_N_SIMULATIONS,
  GUMBEL_BLEND_WEIGHT, TRAIN_SEED_COUNT/VAL_SEED_COUNT, TRAIN_FIRST_SEED/
  VAL_FIRST_SEED, REGENERATE_ROOTS=1, MAX_EXAMPLE_PASSES (default 4).

## Deferred / queued work

- Full Phase B probe rerun (512 sims, w=1.0) on the current champion —
  budget sweep partially superseded it; still useful at n=512+.
- Test cycle-2 checkpoint at n=256 (queued in the gate battery).
- CascadiaFormer-M training run when model-class bound is confirmed.
- Distillation/retention (Phase D) only after a >=97 checkpoint.
- Benchmark-side shared-bridge support (gates still owned-bridge; fine).
- EI-1 corpus is v1 schema; only v2 shards join replay windows.

## Monitor discipline (learned 07-05/06, the hard way)

- NEVER end a monitor's remote ssh command with `pgrep`/`grep` whose
  nonzero exit (no match) makes the ssh look failed — a `|| retry` wrapper
  then skips the completion check forever. End remote pipelines with
  `| tail -1` (exit 0) or capture output without exit-code coupling.
- One consolidated watchdog per work-wave, not one monitor per job.
- Error patterns: case-matters (`error: ` for argparse), exclude benign
  matches (`*_invariant_error` metric keys, preflight BrokenPipeError).
- Monitors die with the session: on ANY session resume, first action is
  checking every in-flight job log listed here.
- pkill/pgrep -f self-match: quote patterns that appear in your own
  command line (kill by pid file, chain jobs on done-marker files).
