# Campaign Working State (2026-07-03 morning)

Live working notes for the Gumbel self-play campaign. Companion to
[GUMBEL_SELFPLAY_CAMPAIGN.md](GUMBEL_SELFPLAY_CAMPAIGN.md) (strategy) and
`cascadiav3/EXPERIMENT_LOG.md` (per-run records). Update this file whenever
the in-flight picture changes.

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
