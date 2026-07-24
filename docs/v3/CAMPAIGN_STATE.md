# Campaign Working State (updated 2026-07-16)

Live working notes for the Gumbel self-play campaign. Companion to
[GUMBEL_SELFPLAY_CAMPAIGN.md](GUMBEL_SELFPLAY_CAMPAIGN.md) (strategy) and
`cascadiav3/EXPERIMENT_LOG.md` (per-run records). Update this file whenever
the in-flight picture changes.

## RESUME HERE (07-16 13:17 — attempt 5 last durable state; john0 unreachable)

**Current reachability:** the read-only `campaign_status.sh` check at 13:17
EDT could not reach john0. Therefore current process/GPU/heartbeat liveness is
unknown. Do not infer failure, relaunch, or inspect scientific sidecars from
that network fact. No partial Stage A scientific output or score was read, and
no process was killed or restarted.

**Last durable transition (EXPERIMENT_LOG 10:10):** attempt 4 completed no
seed. The 24-owned-session topology held the GPU at nominal 100% while CUDA
contexts time-sliced; zero-byte run/sidecar output and low power exposed the
thrash. Attempt 5 launched at 10:02 as
`stage_a_generation_v5_20260716` on PID `26197`, monitor `bn34wrswc`, source
rev `689f9d69`, the v2-proven 12 shared sessions / Rayon 16, TF32 on, ghost
off, and the same registered seeds `2026794000..5249`. The authorized
`d1_pipeline_20260716.sh` waiter was repointed at v5. This is a launch record,
not a current liveness claim; a generation arm is healthy only after completed
seeds, not from startup GPU utilization.

**Rules/provenance repair COMPLETE (03:50).** Commit `31fc2c30` reconciled the
per-resolution wildlife return semantics, assigned the July-16 engine and
scientific IDs, propagated them through exporter/manifests/fresh-run
comparators, and added fail-closed mixed-identity tests. Workspace rules and
exporter suites passed. Every 98.x scoreboard result remains historical
July-9 evidence; no admissible July-16 canonical score exists yet.

**D1 preregistration frozen before Stage A output was read.** The 09:00
`cascadiav3/EXPERIMENT_LOG.md` entry is authoritative: 15k hard roots split
`6k/6k/3k` by phase plus a 1,500-root descriptive sentinel; n2048/d16,
ghost-off teacher with repeat seeds `9000001/9000002`; exact repeat
aggregation; masked training views; raw mix `4:2:1:1`; matched no-D1 control;
5k/10k dose descriptives; locked bank bars; and a fresh
`2027079000..2027079099` sequential-CUPED gate. John authorized the complete
relabel/retrain/screen/gate chain. Champion promotion remains separately
reserved to John.

The three predecessor proposals—
[Cascadia-NX](../../stochastic_board_game_ai_architecture_research_7_16.md),
[Cascadia-Anchor](../../incumbent_anchored_gpu_rollout_policy_improvement_7_16.md),
and [Cascadia Foundry](../../cascadia_foundry_original_architecture_proposal_7_16.md)—
are now reconciled by the
[Cascadia Rival final proposal](../../cascadia_rival_final_architecture_proposal_7_16.md).
John ruled on 07-16 that the policy class is explicitly non-cooperative;
Foundry-Commons, table utility, donation, joint four-board planning, shared
cross-seat prices/memory, and the associated `76%` conditional forecast are
withdrawn. Rival retains Anchor's incumbent/terminal-own-score spine, requires
an NX-style cheap continuation to earn a statistically controlled
multifidelity role, and uses only Foundry's seat-local contracts and unilateral
tomography. Its wrapper is an offline/shadow/one-seat labeler; an ordinary
distilled model is the sole v1 promotion/target candidate. Rival is post-D1,
has zero current-rules strength evidence, and
does not change this queue, displace john0, or authorize reading live-arm
output.

**Resume checklist:**

1. Restore/recheck john0 reachability through the ordinary read-only status
   path. Do not infer PID state from the 13:17 network failure.
2. Do not inspect partial Stage A scientific output. Verify only complete-seed
   progress plus process/heartbeat freshness through the durable
   monitor/status path.
3. Do not kill, restart, deploy over, or compete with last-recorded PID
   `26197` or its waiter without explicit user permission. Honor `HOLD_*`
   checks in the chain.
4. After the wrapper declares completion, verify the complete manifest,
   July-16 rules/source/config identity, registered seeds, sidecar/tensor
   counts, and SHA-256 evidence before harvesting anything.
5. Continue the already authorized harvest/sentinel, two-repeat relabel,
   aggregation/masked-view, retrain/control/dose, locked-bank, and conditional
   gate stages exactly as frozen in the 09:00 experiment-log entry.
6. If a gate is positive, stop at the reserved champion-promotion decision and
   present John the complete provenance/verdict; do not auto-promote.

Historical resume state follows; it is superseded by the block above.

**R3.6 ceiling probe RESOLVED (18:41):** n4096/d16 paired `+0.2100` vs the
stored champion arm, CI `[-0.5925, +1.0125]` — preregistered band
**decelerating** (~1/3 of the +0.615 log-linear prediction; 131.4s/decision
= 3.1x cost for 4x sims, dedup improves with n). Taken with the day-one
nulls, weight shifts to **R1.2 ghost opponents** and **R1.4 training
densification**, with the R2.x velocity stack as multipliers
(EXPERIMENT_LOG 18:50).

**Live now:** `puzzle_bank_20260711.sh` (PID `3888566`, rev `e78975a0`) —
R2.1 bank generation: ~727 champion-ledger roots resolved at n4096/d16 x2
repeats, worker-pooled jobs12 (saturation pattern), ~4.5h. Preregistered
acceptance check (incumbent + cs025_tk8 screens vs gate truth) runs before
first use (EXPERIMENT_LOG 18:54).

**Live now:** `ceiling_probe_20260711.sh` (PID `3843186`, rev `a48fc7d3`) —
the preregistered **R3.6 mega-budget probe**: cycle4 at n4096/d16 (K1 off,
matching the stored baseline), 25 games on `2027070900..24` paired against
the stored champion n1024/d16 per-seed scores. Bands: mean ≥ +0.45 scaling
lane OPEN; +0.15..+0.45 decelerating; ≤ +0.15 lane closed (log-linear
prediction +0.615). Informative probe, never promotion evidence.

**Day-one Tier-0 verdicts (EXPERIMENT_LOG 04:55 → 11:50):**
- **R0.1 sigma calibration: CLOSED** (screen 7/7-positive was a
  shared-baseline artifact; 100-seed confirm `-0.2325` CI
  `[-0.544,+0.079]`).
- **R0.2 paired rollouts: CLOSED at the preregistered floor** (gap-variance
  `-4.4%` vs required `-20%`); secondary CI+ (flip rate `0.466 → 0.424`,
  CI `[+0.005,+0.080]`) → rides along in a future composed serving-v2
  gate. First probe run was invalid (rollout top-k defaulted to 1 —
  vacuous); fixed at four layers, archived as `*_invalid_topk1*`.
- **R1.1a contention audit: no cheap cooperative points at the root**
  (table delta `-0.03`/decision at own-Q parity; naive +10/game bound is
  value-head noise-harvest, do not quote). R1.1b/c deprioritized.
- **Concurrency: retain jobs12** (throughput flat, bridge-bound ~66% util
  at all settings) — R2.4 bridge work is the standing throughput lever.
- **Saturate-the-GPU rule added to AGENTS.md (John's 07-11 ruling).**

**Overnight verdicts:** (1) **R0.1 sigma calibration CLOSED** — screens
were 7/7 positive (best cs025_tk8 +0.70) but the 100-seed disjoint-block
confirm returned `-0.2325`, CI `[-0.5440, +0.0790]`; the screen pattern
was a shared-baseline artifact (lesson recorded, RESEARCH_LOG §4.10).
(2) **Concurrency probe RESOLVED: retain jobs12** — throughput flat
(jobs24 `1.051x`), GPU ~66% util at all settings; the bridge is the
bound, R2.4 gains must come from bridge-side work. (3) Chain 2 (PID
`3764249`) deployed rev `927004fd` and is on the **R0.2 stability probe**
(preregistered: ≥20% pooled gap-variance reduction under paired rollouts →
n256 gate), then the **R1.1a contention audit**. EXPERIMENT_LOG 04:55,
09:45.

**Chain 2 armed (PID `3757640`, log/pid `gpu_chain2_20260710_audits.*`,
pause `HOLD_gpu_chain2_audits`):** waits for chain 1 to exit, then deploys
revision `a6d590b0`, rebuilds the exporter, and runs two preregistered
ledger-replay measurements over the champion cycle4 n1024/d16 decisions
ledger (EXPERIMENT_LOG 22:36): the **R0.2 search-stability probe** (100
roots × 6 unpaired vs 6 paired-CRN repeats at n256/d4, matched search
seeds; rule: ≥20% pooled gap-variance reduction → preregister an n256
gate) and the **R1.1a table-contention audit** (chosen vs best model-Q
alternative under table value; sizes the cooperative-play prize —
portfolio bar: ≥0.3 gate pts/game at own-Q sacrifice ≤0.25 prioritizes
R1.1b/c). New exporter modes + analyzers landed on `main` (54/54 + 5/5
tests).

**Live now:** `gpu_chain_20260710_sigma.sh` (PID `3747964`, log/pid in
`cascadiav3/logs/`, pause `HOLD_gpu_chain_sigma`) at deployed revision
`83ffe12a` — stage A: concurrency probe attempt 3 (attempt 2 failed 21:08 on
the second missing-PATH class, `cargo`; the chain provides
`$HOME/.cargo/bin` + `/usr/lib/wsl/lib`), stage B: the **preregistered R0.1
sigma-calibration sweep** (EXPERIMENT_LOG 21:55): smoke → 8 arms (c_scale
{0.05,0.1,0.25,1.0} × {minmax,topk:8}, 25 paired seeds `2027072100..24`,
incumbent = cs10_mm arm) → 7 paired verdicts (`compare_search_shape
--varied-key`) → screen floor +0.25 → conditional 100-seed confirm
`2027072200..99` (touched once). Pause `HOLD_sigma_sweep` between arms.
ETA: probe ~1h + sweep ~5h + conditional confirm ~5h.

**Implementation landed on `main` (83ffe12a):** `--gumbel-c-visit/c-scale`,
`--gumbel-sigma-norm minmax|zscore|fixed:<s>|topk:<k>`,
`--gumbel-paired-rollouts` (CRN leaf rollouts, R0.2 — implemented and
provenance-plumbed, NOT part of the sigma sweep; its offline variance check
is the next serving experiment). Defaults bit-identical (regression-pinned);
51/51 exporter + 7/7 comparator tests.

**Worlds det16/det32 n1024 confirmation: PAUSED (John's 21:00 ruling), not
closed.** Rationale: not wall-matched (scaling anchor predicts ~+0.3 for its
~1.5x wall, so even CI+ is ambiguous) and lowest conviction-per-GPU-hour on
the board next to the portfolio. Confirm tree killed cleanly 21:06 with
permission (`HOLD_worlds_confirm` set); det16 arm had **zero completed
games** — nothing durable lost; det32 smoke report intact. Block
`2027071600..1699` stays reserved; a future rerun repeats both arms under
the same preregistration. EXPERIMENT_LOG 21:10.

**GPU now:** `concurrency_probe_waiter` (PID 3731156) fires the jobs12/16/24
probe on idle detection (PATH fix applied; output →
`cuda_concurrency_probe_run.log`). Behind it: the break-100 Tier-0 program
(`claude_max_research_ideas.md`) — R0.1 sigma-calibration sweep
(preregistration pending), R0.2 paired-rollout offline variance check;
zero-GPU audits (R1.1a contention, R1.3a coverage, R2.3 CUPED) run
orchestrator-side in parallel.

## PREVIOUS RESUME (07-10 19:25 — worlds screen CI+; n1024 det16/det32 confirmation RUNNING (~20h); stage-5 probe re-queued behind it)

**Worlds screen verdict (n256, det4 vs det8, seeds `2027071500..99`): CI+.**
det4 `97.1425` vs det8 `97.5650`, paired `+0.4225`, 95% t-CI
`[+0.1045, +0.7405]` — first CI-positive search-shape result under corrected
rules. `worlds_confirm_waiter` fired the preregistered rule at 19:15:31 and
launched `run_worlds_confirm.sh`: **cycle4 n1024/top16, K1 on, det16 vs
det32, block `2027071600..1699`, ~20h** (det32 smoke first). Pause with
`HOLD_worlds_confirm`. Cost caveat recorded in EXPERIMENT_LOG 19:20: det8 at
n256 cost `1.495x` mean decision time (worlds reduce eval-dedup hits), so the
"free allocation knob" premise is wrong on wall — weigh this at adoption
time; a cost-matched det4-at-higher-n frontier question stays open.

**Stage E concurrency probe failed at 19:10:50 — root-caused and re-queued.**
Silent preflight: `command -v nvidia-smi` fails in detached shells
(`/usr/lib/wsl/lib` not on PATH) and `set -e` exits without a message. Local
`main` fix (uncommitted) makes every preflight loud and resolves
`NVIDIA_SMI` explicitly. On john0, `concurrency_probe_waiter.{sh,log,pid}`
(PID `3731156`, pause `HOLD_concurrency_probe`) waits for the confirmation +
idle exporter/bridge, then relaunches the pinned probe with
`PATH="$PATH:/usr/lib/wsl/lib"`; output →
`cascadiav3/logs/cuda_concurrency_probe_run.log`.

**Research planning:** `claude_max_research_ideas.md` (repo root) — tiered
break-100 portfolio with kill tests; see EXPERIMENT_LOG 19:20 for the code
audit facts it rests on.

## PREVIOUS RESUME (07-10 13:20 — recovery CLOSED; worlds screen on GPU; confirmation auto-gated)

**Rebaseline recovery is fully closed:** both one-seed d20 replays
validated bit-exact and installed (ledgers 100/100); category attribution
is flat in every category (see EXPERIMENT_LOG 13:15); canonical harvest
complete. The GPU is on the preregistered worlds screen (det4 then det8);
`worlds_confirm_waiter` will compute the verdict and launch the n1024
det16/det32 confirmation only on CI+. Earlier same-day context below.

## PREVIOUS RESUME (07-10 10:45 — K1 ADOPTED by ruling; structured-Q pilot FAILED; stage 4 live)

**Post-chain progress (resume2, PID in
`cascadiav3/logs/postchain_resume2_f35b0d0b.pid`):**

- **Stage 1 (exact-K1 gate): RESOLVED — K1 adopted (John's 07-10 ruling).**
  Seed `2027071427` excluded by declaration (jobs12 concurrency divergence
  at ply 18); verdict on 99 pairs `-0.0379`, CI `[-0.0859,+0.0101]`,
  score-neutral, `28.99x` exact-frontier speedup. K1 is the serving
  default; exact K2 is closed — deeper plies stay on model inference.
  Artifact: `exact_k1_20260709_n256_d4_verdict.{json,md}` on john0.
- **Stage 2 (structured-Q head pilot): FAIL, direction closed.** Verdict
  `structured_q_head_pilot_20260709/heldout_verdict.{json,md}`: candidate
  selected-final RMSE `4.1573` vs teacher `3.5520` (needed <=`3.1968`);
  paired CI `[+0.4461,+0.6143]` wrong side of zero. Retention gates passed.
  Per preregistration: no full-model training, no gameplay; expansion and
  reserve holdouts stay quarantined. EXPERIMENT_LOG 07-10 10:20.
- **Stage 3 (CUDA packed throughput): pass, engineering-only.** S only
  `1.9x` M on the 5090 — small-model serving closed on john0.
- **Stage 4 (market sample-4 gate): FAIL — sample-8 stays.** Paired
  `-0.1575`, CI `[-0.4684,+0.1534]` breaches the preregistered `-0.25`
  floor; speedup `1.575x` was not enough. The comparator's exposure-frontier
  check was invalid for this knob (42/100 pre-exposure divergences by
  mechanism) and now classifies traces descriptively. Artifact:
  `market_samples_20260709_n256_d4_verdict.{json,md}`.
- **Stage 5 (jobs12/16/24 concurrency): never ran** — the stage-4
  comparator crash killed the chain. Queued for relaunch after the replays
  and the worlds screen.

**GPU queue (session-independent):** `cascadiav3/logs/gpu_autochain.{sh,log,pid}`
on john0 owns the whole remaining pipeline — cycle4 seed-0908 replay
(validate+install, regenerating if the in-flight run dies), distq seed-0962
replay, both category ledgers + the paired category mechanism verdict
(`rules_20260709_n1024_category_verdict.{json,md}`), then the preregistered
worlds screen (det4/det8, block `2027071500..1599`), then the stage-5
jobs12/16/24 concurrency probe. Replay validation failures are logged
loudly and skipped rather than stranding the GPU. Pause with
`touch cascadiav3/logs/HOLD_gpu_autochain`. Behind it,
`worlds_confirm_waiter.{sh,log,pid}` computes the screen verdict on-box and
launches `run_worlds_confirm.sh` (n1024 det16 vs det32, block
`2027071600..1699`, ~20h) only if the preregistered CI+ rule fires; pause
with `HOLD_worlds_confirm`. The canonical harvest is
`fetch_rules_n1024_verdict.sh` once the category ledgers exist.

## PREVIOUS RESUME (07-10 03:30 — rebaseline COMPLETE; cycle4 retained; post-chain resumed)

**Final corrected-rules verdict (07-10 ~03:05):** distq-k8 n1024/d16 scored
`98.3850`; paired distq-minus-cycle4 at n1024/d16 is `+0.0875`, 95% t-CI
`[-0.2411,+0.4161]`, not significant. **Cycle4 scalar is retained as
champion.** Within-model n1024/d16-minus-n256/d4 scaling is CI+ for both
heads (`+1.2300` scalar, `+1.0775` distq). Verdict artifacts:
`rules_20260709_rebaseline_verdict.{json,md}` on john0.

**Second raw-seed loss:** the mirror captured 99/100 distq raw files; seed
`2027070962` was destroyed by the d20 harness temp-dir cleanup at process
exit (same class as scalar `2027070908`). Both seeds need the identical
one-seed d20 replay under their pinned decision ledgers and seat totals
before `compare_game_categories` can run. Durable-first raw-games output
(commit `b67b5163`) removes this failure class from all post-f35 runs.

**Post-chain state:** the original waiter deployed f35 at 02:51 but wrote its
revision marker to `postchain_deployed_revision.txt` while the gates require
`exact_k1_deployed_revision.txt`; every stage failed closed and the waiter
exited without running anything. After re-verifying the deployed tree against
the durable archive, the marker was copied to the expected name and the stage
block relaunched verbatim as
`cascadiav3/logs/postchain_resume_f35b0d0b.{sh,log,pid}` (PID `3620337`,
heartbeats + `HOLD_postchain_resume` pause file). Stage 1 exact-K1 gate is
running on seeds `2027071400x100`; then structured-Q head pilot, CUDA model
throughput, market sample-4, and jobs12/16/24 concurrency, strictly
sequential.

## PREVIOUS RESUME (07-09 rules correction — rebaseline before research resumes)

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
completed the 100-game greedy/no-search floor, both n256/d4 arms, and the
cycle4 n1024/d16 arm. The corrected-rules scalar n1024 report passed at 16:50
EDT: 100 seeds, mean seat `98.2975`, P50 `98.0`, P90 `102.0`, and mean
decision time `46.2733s`. Report SHA-256 is `8c164dc6...`; its complete
8,000-row decision ledger is `d42cf655...`. The distq-k8 n1024/d16 arm is now
running on the same seeds under runner/exporter PIDs `3556049 / 3556050`. At
20:57 EDT it had 39/100 complete 81-row raw games, a last-10 rate of
`10.06 games/hour`, and a projected completion around 02:59 EDT on July 10.

The n1024 raw-ledger watcher PID `1284321` is dead. It durably copied 99/100
scalar games but missed seed `2027070908`; the scalar temporary directory is
gone. Its log failed closed on that exact missing file, so neither the scalar
category ledger nor category summary was published. **Mitigated 21:15 EDT:**
the live distq raw files are now mirrored into
`cascadiav3/reports/rules_20260709_distq_k8_n1024_d16_raw_games/` by a
replacement copy loop
(`cascadiav3/logs/rules_20260709_distq_n1024_raw_mirror.{sh,log,pid}`, PID
`3576186`, 120s cadence, final copy after runner `3556049` exits). The
aggregate scalar report is valid; the scalar seed-0908 exact replay remains
the open step for category attribution. Watcher pid files are
`cascadiav3/logs/rules_20260709_distq_k8_n256_raw_watcher.pid` and
`cascadiav3/logs/rules_20260709_remaining_raw_watcher.pid`. Rebaseline log/pid:
`cascadiav3/logs/rules_20260709_rebaseline.{log,pid}`. Canonical launcher:
`cascadiav3/scripts/run_rules_20260709_rebaseline.sh`; every completed report
is reused only when both rules ID and source revision match.
Verdict watcher PID `1268022` waits for the chain and then writes
`rules_20260709_rebaseline_verdict.{json,md}` with paired distq-minus-cycle4
intervals at both budgets plus within-model scaling deltas.
The total-score verdict is no longer the end of the analysis. Once the raw
n1024 watcher publishes both complete 100-row game ledgers,
`compare_game_categories` will bind them back to their reports and emit paired
distq-minus-cycle4 wildlife, habitat, Nature-token, and total deltas with CIs.
It fails on missing seeds, search/rules/source mismatch, category-sum error, or
any ledger/report total disagreement.
Canonical orchestrator harvest is
`cascadiav3/scripts/fetch_rules_n1024_verdict.sh`. It refuses while the
rebaseline or raw-ledger watcher is live, hash-verifies every fetched report,
decision ledger, game ledger, and category summary, and requires the category
total statistics to reproduce the canonical total verdict field-for-field.

**Structured-Q implementation and v4 data (complete; john0 training not yet
started):** the representation gate's authorized path is now real rather
than a design note. New Gumbel generation emits exact-grounded schema v4:
`active_seat`, per-action wildlife/habitat/Nature afterstate components,
real terminal components, and fail-closed sum invariants. Filtering and
relation-tail materialization preserve the new tensors. CascadiaFormer has an
opt-in action-conditioned component head whose sum is the existing
score-to-go output; legacy checkpoints remain state-contract compatible when
disabled. The trainer exposes `gumbel-selfplay-structured-q` and
`q-decomposition-head-only`, requires v4 NPZ for both train and validation,
and supervises categories only on the selected real trajectory while keeping
completed-Q loss over every q-valid action. Unit coverage includes scalar and
distributional sums, malformed shards, transforms/collation, head freezing,
checkpoint reload, and an end-to-end two-step v4 train. This work has no score
claim yet. The Mac fleet may generate the three corrected-rules v4 blocks, but
the frozen-head training and held-out gate remain john0-only. john0's live
n1024 rebaseline remains untouched. The pilot is checksum-queued behind the
already-approved exact-K1 gate.

The hashes are now fixed. Three Mac hosts generated disjoint 800-root blocks
from exporter source `6e89d955`, cycle4 teacher manifest `b8886c24...` /
weights `33559aab...`, corrected rules, and seeds `2027073500..29`. The final
shape was n8/top4/d1, one determinization, eight optional-refresh samples,
blend 0.5, K8 interior, exact K1, full root menus, two games per shared MPS
bridge. Generation took `984.3 / 908.1 / 1059.8s`; all raw shards are
training-eligible v4 with 800 records and Q-identity max error `3.8147e-6`.
Immutable NPZ hashes are fit `06d550b4...`, selection `5095d572...`, and
untouched verdict `cdbd54b0...`. They are copied and hash-matched on john0.
The first n16/d2 launch was terminated before any NPZ/manifest was published
because it had not produced a first seed on any host after about seven
minutes; no partial artifact was admitted.

The replacement post-chain waiter is PID `2241595`, source revision
`f35b0d0b209444f8c09e7e603c380f1d8edbc100`, archive SHA-256
`460857f26f7431727db623313f92df2e5be13a27033bd72d642eb6d650fc7a81`.
It verified the archive and all three raw NPZ hashes before waiting. The live
rebaseline PIDs were not touched. Its strict sequence is corrected rebaseline
and verdict -> exact K1 -> structured-Q head pilot -> CUDA model throughput ->
market sample-4 -> jobs12/16/24 concurrency. A valid scientific structured-Q
failure returns control to the remaining queue; a crash or malformed verdict
stops it.

The verdict is preregistered in `torch_structured_q_probe`: exclude exact K1
rows; require at least 10% selected-final RMSE improvement over the better of
incumbent model Q and selected completed-Q teacher; require the paired
absolute-error t-CI below zero; cap all-q RMSE at 1.05x incumbent and mean
q-regret increase at 0.05. Learning-rate selection and the final verdict use
disjoint seed blocks.

The candidate-blind held-out baseline read is now fixed. On the 760 non-exact
verdict roots, selected incumbent final-Q RMSE is `3.7476` and selected teacher
RMSE is `3.5520`, so the teacher owns the primary baseline and a candidate must
reach at most `3.1968` RMSE before the paired-CI requirement is considered.
Incumbent all-q RMSE against the four searched actions per non-exact root is
`1.7499`, mean completed-Q regret is `0.7515`, and top-1 agreement is `36.45%`;
the corresponding hard ceilings are `1.8374` all-q RMSE and `0.8015` mean
regret. No candidate output was inspected and no hyperparameter was selected
on the verdict block.

**Quarantined v4 expansion (complete and audited 13:19 EDT):** john2–john4
generated three data-only 50-seed blocks at the validated n8/top4/d1 shape on
seeds `2027073600..49`, `2027073650..99`, and `2027073700..49`. Wall times
were `3792.6 / 3813.3 / 3806.3s`; each produced 4,000 roots. NPZ hashes are
`225aeff6... / 0447d69b... / 5dc0860d...`. The canonical harvest verified
remote/local NPZ and manifest hashes, exact seed domains, passing per-host
reports, one source/rules/search/execution/teacher contract, and disjointness
from the locked pilot. The combined audit passes with 150 seeds, 12,000 roots,
5,299,287 actions, 46,200 q-valid actions, 600 exact rows, maximum Q-identity
error `3.8147e-6`, and zero component-sum error. Audit SHA-256 is
`e1edbad3552abef2321808666948f299fbf3ba226b948d50a2314b696fb5eb14`.

Target distributions are stable across the expansion blocks: final means
`91.485 / 91.885 / 91.490`, total score-to-go means
`45.846 / 46.001 / 45.701`, selected-teacher RMSE
`3.169 / 3.375 / 3.287`, and q-valid actions per root exactly `3.85` in each.
These shards remain speculative fit-capacity inventory only: keep them out of
the fixed pilot and john0 queue, and admit them to training only if the frozen-
head pilot passes. john1 remains reserved for the UI/champion service.

The first completion boundary exposed an omitted exporter `--manifest` path:
all NPZs were valid, but their generated sidecars went to the CLI default, so
all three validators and reserve chains failed closed before reserve output.
The default manifests checksum-matched their NPZs and exact provenance, were
placed at the declared sidecar paths, and both validators were rerun to pass.
Failed chain evidence is preserved with `.failed_manifest_path` names. Commit
`4cd9c728` makes every reserve output sidecar explicit and tests that contract.

**Candidate-blind reserves (harvested and globally audited 21:15 EDT):** the
canonical `fetch_structured_q_reserve_holdouts.sh` run passed: remote/local
hashes matched for all three roles, exact seed domains pinned, and the
nine-shard audit against the locked pilot plus all three fit-expansion shards
returned `pass` (3 shards, 60 seeds, 4,800 records, 2,058,733 actions, 18,480
q-valid actions, 240 exact rows). Combined audit SHA-256
`aab21d186955f7281fbc1fc0cce9b6ceb8e2b8ed9d9529aa0dc1b6071af5a3d2`. The
holdouts remain quarantined from john0 and training. Original per-host record:
roles
were fixed before any candidate existed: john2 selection seeds
`2027073750..69`, john3 verdict `2027073770..89`, and john4 independent
replication `2027073790..3809`, each 20 seeds / 1,600 roots at the identical
raw-v4 contract. The corrected chains completed in
`1530.0 / 1624.4 / 1506.5s`. All three manifests checksum-match their NPZs;
summary and invariant reports pass. NPZ hashes are selection `48e48e74...`,
verdict `99b85671...`, and replication `41b5bd60...`; action counts are
`711,027 / 667,699 / 680,007`. These are not extra fit data and cannot
influence the existing pilot. The arming script cannot fetch, train, or
address john0. They still require the canonical local harvest and nine-shard
audit before any use.
Canonical reserve harvest is
`cascadiav3/scripts/fetch_structured_q_reserve_holdouts.sh`. It refuses any
live chain, requires passing reports plus the completion sentinel, verifies
remote/local NPZ and manifest hashes, pins every role's exact seed domain, and
audits the three holdouts against both the locked pilot and all three fit-
expansion shards. It therefore cannot run the final audit until the expansion
has itself been harvested. The tool has no john0 or training copy path.

**Corrected no-search floor (100 paired seeds, complete):** greedy `87.5450`;
cycle4 policy head `91.8425`, delta `+4.2975`, 95% t-CI
`[+3.8705,+4.7245]`; cycle4 Q head `90.8925`, delta `+3.3475`, CI
`[+2.8507,+3.8443]`. All 24,000 decisions are retained. Optional refresh
accept/decline counts were policy `594/352`, Q `636/364`, and greedy
`1005/398`. The interactive no-search harness uses greedy-v1 for this
pre-draw market decision, then the named model head ranks the revealed draft;
the Gumbel legs are the model/search-driven refresh-decision evidence.

**Corrected cycle4 n256/d4 baseline (100 games, complete):** mean seat
`97.0675`, P50 `97.0`, P90 `100.1`, with 2/100 game means at least 100.
Across all 8,000 decisions, 952 offered a refresh; search accepted 565 and
declined 387 (`59.35%` accept). Mean decision latency was `11.729s`; refresh
opportunities averaged `54.908s` versus `5.896s` ordinarily. Market choice
added 2,094,336 simulations above the 2,048,000 chosen-branch simulations.
Report/decision hashes were copied and matched locally. This is the first
current-rules search baseline, not a promotion comparison. The deployed
reducer discarded score categories with its temporary game files, so this
arm cannot support a category-level claim; all newly launched gates now write
a complete seed-ordered `*_games.jsonl`, embed per-seat score breakdowns, and
fail instead of publishing an incomplete game ledger.

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
armed on john0 from a checksum-pinned final-main snapshot (pid file
`cascadiav3/logs/exact_k1_waiter_main.pid`): only after the current rebaseline
and verdict watcher exit will it install the exact revision-marked `main`
snapshot, rebuild, and run a fresh same-revision 100-seed corrected n256/d4
baseline/K1 gate. It then runs the same-revision CUDA model-size throughput
probe, the sample-8 versus sample-4 gate, and finally the engineering-only
jobs12/16/24 CUDA concurrency calibration, strictly sequentially. The
concurrency arm never mutates a default: it records complete matched traces
and one-second GPU telemetry, then recommends the smallest parity-passing arm
within 2% of the fastest only if the best wall speedup over jobs12 is at least
1.05x.

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
will follow exact K1 and the CUDA throughput probe on john0, reuse the exact
gate's identical validated sample-8 arm, and run a fresh 100-seed sample-4
candidate. Passing requires t-CI lower bound
`>= -0.25` and whole-decision speedup `>= 1.15x`; failure leaves sample-8 in
place.

**Model/search inversion preflight (07-09):** a new fixed-root bridge
benchmark pins roots, production-packed payloads, model parameters, outputs,
environment, and reports. The first raw-input run was invalid as serving
evidence because it timed Python feature extraction that live Rust search
bypasses; its `2.40x` tiny/M ratio is superseded. With production-packed roots
on john2–john4 MPS, batch-8 means were `144.996 roots/s` for trained 88.17M M,
`443.174` (`3.06x`) for trained 15.02M S, `700.524` (`4.83x`) for synthetic
5.12M XS, and `1,427.867` (`9.85x`) for synthetic 67.8K tiny. At batch 32 the
ratios were `3.38x / 5.64x / 13.66x`. This reopens the smaller-model/larger-
search direction without pretending synthetic shapes are strong. Three
same-host, single-seed MPS calibrations found that trained S n192/d12 was
`1.477x` slower than M n64/d4 despite the `3.06x` bridge rate; mean scores were
`95.500` versus `96.083` (three-game smoke only). The implied equal-wall S
budget was about n130. The completed S n128/d8 follow-up was near equal wall
(`1.078x`) but scored `93.917` versus M's `96.083` (delta `-2.167`, three-game
smoke only). This is negative enough to withhold XS distillation, not a
strength verdict. A same-tool CUDA probe remains queued after exact K1 and
before the sample-count gate; only materially better whole-search leverage or
a stronger/distq student can reopen training.

**Distributional-Q risk-serving kill test (07-09):** source `ef5499b7` adds
explicit mean/q25/q50/q75 bridge modes, records the mode in bridge/report
provenance, rejects scalar checkpoints before launch, and monotonically
rearranges independent heads before interpolation (the existing mean is
unchanged). A fixed corrected-rules corpus of 160 full-menu roots / 40,776
actions found zero crossed adjacent heads. q25/q50/q75 changed the direct
derived-Q argmax on only `3.125% / 2.500% / 1.875%` of roots; their average
mean-policy-Q regret was below `0.0001`. In same-host n64/d4 gameplay, q50 was
score-flat on one seed and q75 was `-1.25` on one seed. q25 was extended to
all three precomputed mean-control seeds: deltas `+2.25/-0.25/-1.25`, mean
`95.25` versus `95.00` (`+0.25`, n=3, 95% t-CI
`[-4.23,+4.73]`), with `1.042x` mean wall ratio. Risk serving clearly changes
trajectories (first divergences at plies 2/20/2 for q25) but has too little
fixed-root ranking leverage and no directional gameplay case for a CUDA gate.
Keep mean as production default; retain q25/q75 only as cheap future league-
diversity personalities if corrected-rules EI survives. Fixed-root report SHA
`0c57c8fa1b0f1def6c70a038325885da499e148631f3ec3fc0009b2fec1c0f9b`;
validated 32-artifact gameplay summary SHA
`5304b88265c7d698635be8ba4d08b2e85dcf22654b563b3782b60aa96e71f42b`.

**Shared-batch utilization and concurrency (07-09):** the live john0 distq
n256 arm at jobs12 averaged only `65.6%` 5090 SM utilization over 30 seconds
(range `1-89%`), `353.5W` against 600W, and 2.48 GiB framebuffer use; a CPU
snapshot was `55.6%` idle. The gap is concurrency/lockstep, not capacity. A
provenance-complete four-seed MPS screen found exact action parity across
jobs1/2/4, but weak throughput scaling: jobs2 `1.147x`, jobs4 `1.180x`, while
mean decision latency rose `1.70x/3.12x`. A 2M versus 16M cell-budget control
was flat (`+0.54%` wall), so bridge chunk sizing is not responsible. Jobs2 is
the mini operational knee; do not extrapolate it to CUDA or modify the live
chain. A resumable jobs12/16/24 performance-and-parity calibration is now
queued at the end of the checksum-pinned john0 post-chain waiter; it uses 48
fixed seeds at n64/d4 and cannot alter a runtime default. Summary SHA
`7d4fb02d1432a8a83c85ee1b123b0a842ce139e92703c9d9932a579d7f163d02`.

**Dynamic seed-queue fix (07-09):** the utilization gap had a concrete
long-tail cause: fixed contiguous seed chunks could not backfill after a
worker exhausted its chunk. At 95/100 in the live distq arm, only five games
remained active and GPU utilization had fallen to 18%. Benchmark batch,
Gumbel self-play export, and model-state bootstrap now share a bounded dynamic
queue while retaining one bridge client/session and cache per worker. The
worker/session cap is unchanged. The Rust exporter suite passes 44/44,
including deterministic backfill and exact batch-versus-single record parity.
Replaying the completed arm's observed seed durations predicts 9,014.5s
static versus 8,380.2s dynamic (`1.076x`), explicitly an estimate pending a
post-chain john0 measurement. An exact-revision jobs2 MPS nonregression run
passed with 0/320 action changes, identical scores, `1.56e-5` maximum
root-value drift, flat wall (`+0.57%`), and `-2.86%` mean decision time. The
four-seed static arm was already balanced; validation SHA
`e738e6a9948630ddc7a76a54fefc7d08bf0d9e417bda2ceb40aaa5a1c9958f0d`.

**Parallel leaf-rollout execution frontier (07-09):** blended search's
independent terminal greedy rollouts can now run on the Rayon pool behind
`--gumbel-parallel-leaf-rollouts`, with stable per-simulation RNG streams,
commit order, CLI/report provenance, and a fail-closed trace comparator. On
two fixed distq-M n16/d2 MPS games, jobs1 improved wall/mean-decision time by
`1.061x / 1.061x` with 0/160 action changes, identical scores/telemetry, and
zero root-value drift. The required jobs2 concurrency control was slightly
worse (`269.197s -> 271.043s`, `0.993x`); action/score parity still passed and
maximum drift was `4.35e-7`. Keep the option for interactive single-game
latency only. Do not enable it for fleet generation, promotion batteries, or
the queued john0 jobs12 chain, and do not mistake this shallow CPU frontier
for the still-open GPU-native whole-rollout direction. Comparison SHAs:
jobs1 `c25f7aca...`; jobs2 `3680556a...`.

**Corrected-rules n256 interim result (07-09):** cycle4 scored `97.0675` and
distq-k8 mean serving scored `97.3075` on the same 100 seeds, a paired
`+0.2400` with 95% t-CI `[-0.1139, +0.5939]` and bootstrap CI
`[-0.1000, +0.5950]`. Retain cycle4: this is not significant and neither arm
reaches 100. Both are candidate-only search arms; source d20's eligibility
label is wrong and current source fixes it. The n1024 chain continues.
Fail-closed interim artifact SHA
`287555fb6c233a4e7e14d7e362c7f796ebd35dd4f2b2558b1fd9e12c0b3dbdb8`.

**Pairwise-comparator kill test (07-09):** a provenance-safe v3 campaign
generated 2,400 corrected-rules roots on john2–john4, then trained only the
99,072-parameter antisymmetric comparator head against two seed blocks and
held the third out. The head genuinely learned its labels (held-out pair
accuracy `60.4% -> 66.0%`; confidence-weighted full-probe accuracy `69.5%`),
but serving-aligned top-16 routing failed. On 206 confidence-qualified held-
out roots, Borda changed logits' top-1 accuracy `30.58% -> 31.55%` (only two
net hits; paired 95% bootstrap delta CI `[-3.40,+5.34]` points) while worsening
completed-Q regret `1.1496 -> 1.2121`. Adding logits and Borda was exactly flat
on top-1 and also worsened regret. No gameplay was launched; incumbent logits
remain live. The reported `88.3%` top-16 coverage was inside a top-Q-filtered
64-action tensor, not the full legal menu; it must not be quoted as serving
recall. The follow-up probe now rejects filtered tensors and chunk-scores the
exact full menu. Probe SHA `92834d4e...`; full record and correction are in
`cascadiav3/EXPERIMENT_LOG.md`.

**Exact full-menu candidate-recall kill test (07-09):** the corrected probe
establishes cycle4's actual top-16 coverage of the completed-Q best at
`689/800 = 86.125%` (`654/760 = 86.053%` outside exact K1 roots), with
`186/206 = 90.291%` coverage on confidence-qualified roots. Recomputed
full-menu priors averaged `99.737%` top-16 overlap with generator priors and
agreed on best-action coverage for all 760 non-exact roots, so this is a
validated serving measurement rather than the pairwise tensor's filtered
`88.3%`. A 769-parameter soft-policy fit reduced coverage by four roots. A
purpose-built confidence-gated recall hinge rescued only two menus
(`86.125% -> 86.375%`) and one qualified root, while top-1 stayed exactly flat
and candidate-oracle regret did not improve. The recall candidate was selected
on the same seed-3120 validation block, making even that tiny gain optimistic.
No gameplay was launched. Close this small-data/head-only route; do not add
more objective variants without materially new supervision or architecture
and a new untouched root block. Exact recall probe SHA `5b5668bb...`; full
record is in `cascadiav3/EXPERIMENT_LOG.md`.

**Structured-value representation gate (07-09): PASSED.** On three disjoint
corrected-rules v3 seed blocks, a frozen selected-action latent plus a linear
wildlife/habitat/Nature head used 760 non-exact roots for fitting, 760 for
regularization selection, and 760 untouched roots for the verdict. Held-out
sum RMSE was `3.4889`, versus the best incumbent comparison at `4.1528`, a
`15.99%` reduction that clears the preregistered `10%` gate. This is offline
representation evidence only. The direct-final ridge head cannot serve; the
authorized next branch is an exact per-action category-afterstate schema plus
an action-conditioned residual head whose sum retains scalar/distq Q
supervision. Probe SHA `5c06de5d...`.

**Live john0 high-budget chain (07-09 12:13 EDT):** cycle4 n1024/d16 remains
healthy under PID `1739796` / exporter `1739797`; the watcher has copied and
validated 52/100 complete 81-row seed files. Dynamic
scheduling is backfilling new seeds, and the exporter remains CPU-saturated.
Distq-k8 n1024/d16 follows on the same fresh seeds, then the verdict watcher
publishes the paired result. Do not use partial scores as a verdict and do not
disturb this chain.

A read-only 30-second sample at this exact n1024/d16 workload measured GPU
utilization mean / P50 / P90 `63.8% / 66% / 85%` (range `2%..88%`), power
`350.1W` mean, and fixed `2403 MiB` device memory. Exporter and bridge process
CPU readings were `779%` and `407.5%`. This confirms intermittent model-feed
gaps despite heavy CPU use; it supports the already-queued matched
jobs12/16/24 calibration but is not authority to mutate the live arm.

**Mini-fleet audit (07-09):** john2–john4 were still running Fleet5 under the
pre-correction forced-refresh binary for roughly nine hours. Those process
trees were killed and verified absent; no Fleet5 shard artifact existed to
quarantine. john1's Fleet5 pid file was stale and no process/artifact existed.
The minis remain engineering/data-generation workers only, never promotion
gate hosts.

## Historical resume sections (pruned 2026-07-09)

Superseded RESUME HERE sections from 07-03 through 07-08 (pre-rules-correction
scoreboards, champion history, cycle 1-6 in-flight state, fleet wave notes,
and the old operational-knowledge block) were pruned from this file during the
2026-07-09 doc cleanup. All of their scientific content lives in
`cascadiav3/EXPERIMENT_LOG.md` and `RESEARCH_LOG.md`; the still-true
operational knowledge was folded into `INFRASTRUCTURE.md`. Recover the full
text with:

```bash
git show archive/doc-prune-2026-07-09:docs/v3/CAMPAIGN_STATE.md
```

## ADDENDUM (07-12 01:10 — fully autonomous pipeline armed)

The complete research loop is now CLI+config: `run_experiment_queue.sh
<queue.jsonl>` runs preregistered stages sequentially (HOLD pause,
done-marker resume, failure-tolerant); `run_bank_screen.sh` screens any
serving flag combo against the frozen n4096 bank in ~35 min. Live chain on
john0 (`screen_wave_20260712.{sh,log,pid}`, PID 3931619): bank acceptance
(running) → acceptance-gated deploy of rev `e252d68e` → the preregistered
screen wave (ghost opponents R1.2A, q-bias R0.3, LCB R0.4, combo) + the
R1.3a menu-coverage audit (EXPERIMENT_LOG 01:02). New serving flags all
default-off with provenance. Results land in
`cascadiav3/reports/puzzle_screen_*_analysis.md` and
`menu_coverage_20260712_analysis.md`; proceed-to-gate rules are in the
preregistration. The worlds det16/32 confirmation resume and the refresh
divisor wall-matched probe are the next queue candidates (not scheduled).

## ADDENDUM (07-12 06:00 — refresh-div4 ADOPTED; ghost gate running)

Overnight chain results (all preregistered rules applied literally, see
EXPERIMENT_LOG 01:15 → 05:50):

- **Bank ACCEPTED**; screen wave verdicts: ghost PASS (`+0.0074` bias vs
  `+0.020` bar, 1.65x faster), q-bias structurally null at n256 serving,
  LCB flat, combo flat.
- **Refresh-divisor gate ADOPTED (05:50):** `+0.0375` paired, CI
  `[-0.1611, +0.2361]`, `1.243x` mean-decision speedup — CI floor above
  the `-0.25` margin + wall saving => `--gumbel-refresh-sample-divisor 4`
  is the new serving/benchmark default (second adoption after exact-K1).
- **Live now:** ghost wall-matched gate (launched 05:45, PID file
  `gate_ghost_wallmatched_20260712.pid` via `ghost_gate_20260712.sh`):
  n512/d4 `--gumbel-ghost-opponents` vs n256/d4 champion, seeds
  `2027072400..99`, ~5-6h. Preregistered rule (01:25): CI+ AND ≤1.25x
  wall => graduate to n1024-tier confirmation (block `2027072600..99`
  registered; template stage 4 in
  `cascadiav3/queues/queue_20260712_gates_template.jsonl`, needs fresh
  preregistration + CAND_N from the gate's timing); CI- => cap R1.2 at
  Stage A; ns => retest at n1024 pricing before closing.
- **Chained behind it:** coverage-audit rerun waiter
  (`coverage_rerun_20260712.sh`, PID 3938586) — deploys rev `1c9211a5`
  (replay-cap fix), reruns `run_menu_coverage_audit.sh`; preregistered
  read: drop rate <1% AND regret <0.01/root => close R1.3a-c.
- Morning digest: `ssh john0 'bash /home/john0/cascadia/cascadiav3/scripts/morning_report.sh'`.

## ADDENDUM (07-12 10:50 — ghost CI+; n1024-tier confirmation armed)

- **Ghost wall-matched gate: CI+ (10:30).** `+0.5450`, CI
  `[+0.1823, +0.9077]`, `1.049x` wall (12.53s vs 11.94s/decision) — both
  preregistered conditions hold; R1.2A graduates (EXPERIMENT_LOG 10:35).
  First CI+ wall-matched search improvement of the campaign.
- **R1.3a coverage audit COMPLETE (10:38, valid at rev `1c9211a5`):**
  200/200 roots; drop rate `1.5%` (close bar `<1%` FAILS), mean regret
  `+0.0045` (bar passes) => **R1.3 stays open** per the preregistered
  both-bars rule; tail bound ~0.37 Q/game (EXPERIMENT_LOG 10:55).
- **Launched behind it (10:38):** ghost n1024-tier confirmation
  (`ghost_confirm_20260712.{sh,log,pid}`, waiter PID 3979458;
  preregistered EXPERIMENT_LOG 10:45): baseline champion n1024/d16 vs
  candidate ghost n2048/d16 (parity n from gate timing: 1024x1.906≈1952
  → 2048, predicted ~1.05x wall), both arms
  `--gumbel-refresh-sample-divisor 4`, seeds `2027072600..99`, rev
  `1c9211a5`, ~13-17h. Rule: CI+ AND ≤1.25x wall => ghost n2048/d16 is
  **champion-designate; John alone rules on promotion**; ns => R1.2A is
  a low-budget-only win, revisit via R1.2B/C; CI- => cap R1.2 at Stage A.
- Refresh-div4 remains ADOPTED (05:50 entry); verdict artifacts for both
  completed gates are on john0 under `cascadiav3/reports/gate_*_verdict.md`.
- **Armed behind the confirmation (18:56):** R2.4 engineering chain
  (`throughput_chain_20260713.{sh,log,pid}`, waiter PID 4016261): deploys
  rev `d6cae30b` (sequential-gate machinery + bridge knobs), runs
  `run_bridge_throughput_probe.sh` (eager/bucket/compile arms + numerics
  diff), then a 12-game `CASCADIA_BRIDGE_TIMING=1` production-topology
  sample on engineering seeds `1111110000..11` (n1024/d16, div4).
  Engineering only — never strength evidence. Morning reads: ghost
  confirmation verdict (fixed-N rule, 10:45 entry) -> probe report ->
  timing phase split -> if probe clears >=10% rows/s at batch 192,
  preregister the first SEQUENTIAL noninferiority gate (LOOKS mode,
  cbd2d214) for CASCADIA_EVAL_CHUNK_ROWS on a fresh registered block.
- **Methodology since 07-12:** group-sequential gates are sanctioned
  (AGENTS.md amended; looks 40/60/80/100, Lan-DeMets OBF, RCIs); the
  serving pipeline's serial-bound analysis is in
  `docs/v3/BRIDGE_THROUGHPUT.md`.
- **Armed behind the throughput chain (19:37):** R2.4 lever #1 A/B
  (`pipeline_ab_20260713.{sh,log,pid}`, waiter PID 4019282): deploys
  rev `c2e75cab` (request pipelining, both halves, default-off), then
  12-game n1024/d16 arms serial vs `CASCADIA_SHARED_INFLIGHT=2` +
  `CASCADIA_BRIDGE_PIPELINE=1` on identical engineering seeds
  `1111120000..11`; verdict `pipeline_ab_20260713_verdict.md` (exact
  per-seed score identity + wall ratio). Preregistered adoption rule in
  EXPERIMENT_LOG. Full john0 night queue: ghost confirmation (fixed-N)
  -> throughput probe + TIMING sample (rev d6cae30b) -> pipelining A/B
  (rev c2e75cab). Three monitors armed, one per chain.

## ADDENDUM (07-13 00:20 — ghost confirmation ns; R1.2A closed low-budget-only; night queue reordered)

- **Ghost n1024-tier confirmation: INCONCLUSIVE (00:14).** Champion
  n1024/d16 `98.2825` vs ghost n2048/d16 `98.2000`; paired `-0.0825`,
  CI `[-0.3985, +0.2335]`, wall `0.978x`. Preregistered ns branch
  applied: **R1.2A closes as a low-budget-only win** (CI+ `+0.545` at
  n256-tier stands for data generation / cheap serving); no
  champion-designate; champion remains cycle4 n1024/d16. Surviving
  hypothesis: R1.2B/C — reinvest reclaimed opponent budget into
  NON-sim axes (d32 determinizations, wider top-m, menu-cap relief),
  since the own-sims axis is saturated (matches R3.6). EXPERIMENT_LOG
  07-13 00:15.
- **Throughput probe failed at init (00:16) — no GPU work lost;
  re-armed.** `git rev-parse` in a tarball tree; script fixed
  (marker-file fallback, committed) and `probe_rerun_20260713.{sh,log,pid}`
  waiter (PID 4040310) armed behind the pipelining A/B with
  `SOURCE_REVISION` passed explicitly. Night queue is now: TIMING
  sample (running at rev `d6cae30b`) -> pipelining A/B (rev `c2e75cab`)
  -> throughput probe. Monitors live on all three.
- Morning reads, in order: `bridge_timing_sample_20260713.md` phase
  split -> `pipeline_ab_20260713_verdict.md` (adopt if >=10% decision
  gain AND bit-identical) -> `bridge_throughput_probe.md` (knob to a
  sequential noninferiority gate only if >=10% rows/s at batch 192).
- **PREREGISTERED + ARMED (00:35): R1.2B ghost+d32 sequential gate** —
  the first live group-sequential gate (looks 40/60/80/100, OBF,
  superiority RCIs). Champion n1024/d16 vs **ghost n1024/d32** (reinvest
  ghost's ~50% reclaimed eval budget into doubled worlds = noise
  reduction, not saturated sims), fresh block `2027072700..99`
  (registered), rev `c2e75cab`. Predicted candidate wall ≤0.8x champion
  — answers the wall objection that paused the pure det16/32
  confirmation (block 2027071600..99 stays reserved for that, unchanged).
  Rule: positive stop at ≤1.05x wall => champion-designate (John alone
  rules); inconclusive => R1.2B closes. Waiter
  `ghost_d32_gate_20260713.{sh,log,pid}` PID 4041081, armed behind the
  probe re-run; monitor live. Full queue now: TIMING sample ->
  pipelining A/B -> probe re-run -> ghost+d32 sequential gate.

## ADDENDUM (07-13 03:30 — R2.4 CLOSED, all levers below bar; ghost+d32 sequential gate LIVE)

- **Engineering night queue complete, R2.4 closed (03:25).** TIMING
  phase split: forwards = 84% of bridge time but ~55% of wall, ~30-row
  mean chunks. Pipelining A/B: **bit-identical on CUDA**, +4.2% — below
  the 10% bar, stays default-off. Probe: eager forward saturates by
  batch 32 (CHUNK_ROWS bound +3.9%), compile +0.5% (bit-identical
  datum), bucket negative + drifting. **Serving is within ~5% of the
  architectural ceiling; remaining wall is Rust search compute.** All
  knobs stay landed/default-off for zero-risk revisit.
- **LIVE NOW: R1.2B ghost+d32 sequential gate** (launched 03:22, waiter
  PID 4041081 fired on schedule): champion n1024/d16 vs ghost n1024/d32,
  seeds `2027072700..99`, looks 40/60/80/100 (first live sequential
  gate), rev `c2e75cab`. Verdict rule in EXPERIMENT_LOG 07-13 00:35;
  artifacts `gate_ghost_d32_seq_20260713_*`. Expected: first look
  verdict after ~40 pairs both arms (~4.5-5h), full run ~11-12h if no
  early stop.
- Ghost n1024-tier confirmation verdict (00:14, ns) and its R1.2A
  closure are in the 00:20 addendum above.

## ADDENDUM (07-13 16:05 — R1.2B closed (final inconclusive, +0.18); ghost+d32 1.68x speed observation; noninferiority speed-default gate LIVE)

- **R1.2B ghost+d32 sequential gate: FINAL_INCONCLUSIVE (15:51).**
  `98.2975` (a clean champion replication) vs `98.4750`; delta
  `+0.1775`, RCI `[-0.0935, +0.4485]`. All four looks executed;
  **R1.2 program CLOSED per rule** — no champion-designate. Cost:
  candidate **`0.595x` wall (21.09s vs 35.46s/decision)** — the
  prediction held.
- **PREREGISTERED + LAUNCHED (16:00/15:54): ghost+d32 speed-default
  NONINFERIORITY gate** — fresh block `2027072800..99`, same arms,
  margin `-0.25`, looks 40/60/80/100, PID 4110505, monitor live.
  Noninferior stop at ≤0.8x wall => ADOPT as serving/gate-arm speed
  default (the K1/div4 class; NOT a strength claim; champion promotion
  stays with John). Expected stop at the 60/80-pair look (~7-10h).
- R2.4 remains closed (03:30 addendum). john0 queue: this gate only.

## ADDENDUM (07-13 17:30 — queue realigned by John's ruling; CUPED landed; R3.2 screen armed)

- **RULED BY JOHN (16:30 entry):** research queue realigned to maximize
  break-100 probability: (1) R3.2 depth-2 kill test, (2) R1.4
  densification build, (3) R2.3 CUPED [DONE], (4) R0.5/R3.4 + R1.3b,
  (5) R1.1c/R3.1 after R1.4 infra. Deprioritized: R3.5, R0.7, R0.8.
- **R2.3 CUPED LANDED (be7cddbc):** `SEQ_CUPED=1` on any sequential
  gate; covariate fixed = baseline per-seed seat score; RCI narrows
  10-25% expected; scope = gates preregistered after the 17:10
  methodology entry.
- **R3.2 screen ARMED (waiter PID 4135978, monitor live):**
  ghost+depth2 vs frozen bank behind the live noninferiority gate;
  proceed bar `<= +0.020` regret vs incumbent; pass => sequential
  SEQ_CUPED=1 gate on `2027072900..99` (preregister after the
  speed-default verdict).
- **R1.4 design doc in progress** (agent drafting
  `docs/v3/R1_4_DENSIFICATION_DESIGN.md` from the actual trainer/
  exporter code).
- john0 queue: ghost+d32 noninferiority gate (live, PID 4110505) ->
  ghost+depth2 screen (~15 min) -> GPU free for the R3.2 gate launch
  decision + R1.4 staging.

## ADDENDUM (07-13 23:30 — ghost+d32 ADOPTED at 60-pair early stop; depth-2 screen running)

- **ADOPTED (23:25, preregistered rule): ghost n1024/d32 is the
  serving/gate-arm speed default** (K1 + div4 + ghost + d32).
  STOP_NONINFERIOR at look 2 (60 pairs, seeds 2027072800..59): delta
  `+0.3333`, RCI `[-0.2122, +0.8788]` vs margin `-0.25`, wall `0.688x`.
  First live sequential early stop (~40% of the gate saved). Champion
  identity and canonical score reference unchanged; ghost labels NOT
  yet cleared for training corpora (needs the safety fold, R1_4 §8).
- Depth-2 bank screen running (fired 23:21, ~15 min). Next, in order:
  deploy `6cc01ab5` (CUPED + Stage 0 analyzer), run Stage 0 label audit
  (CPU), and if the screen clears `<= +0.020` regret vs incumbent,
  preregister + launch the R3.2 gate on block `2027072900..99` as the
  first SEQ_CUPED=1 gate, on the NEW default baseline (ghost n1024/d32
  vs same + `--gumbel-depth-rounds 2`, VARIED_KEYS=depth_rounds).

## ADDENDUM (07-13 23:50 — Stage 0 verdict; V1 closed, V1b/T0 unlocked; canonical battery running)

- **R1.4 Stage 0 COMPLETE** (7s per pass — vectorized; full cycle4
  corpus, 100k records): V1 bar FAILS (overall bias `-2.86`) => V1
  closed without a retrain; BUT srv beats the 1-sample noise floor in
  the late game (endgame RMSE `1.46` vs `2.71`) => **V1b preregistered**
  (phase-gated mixing, tile_count>=13). **Adjacency CONFIRMED** (1,249
  contiguous games) => T0 path-consistency needs no schema change.
  Hard-root fraction `54.6%`. Unvisited improved-policy mass mean
  `0.33` (bimodal) => P1 strengthens.
- **GPU: canonical battery running** (PID 4150464, launched 23:45,
  ~4.5h): adopted default ghost n1024/d32 on the rebaseline block
  `2027070900..0999` => new canonical reference + fresh serving-default
  decision ledgers.
- **Building now:** Stage 1 trainer flags (V1b, V2 quantile value head,
  C1 weight flags, T0) — default-off, bit-identical; retrains chain
  behind the battery.
- R3.2 CLOSED (23:30, screen +0.0586 vs +0.020 bar); block 2027072900
  RELEASED.

## ADDENDUM (07-14 04:05 — canonical battery done: 98.3925 descriptive; menu512 screen running; Stage 1 chain next)

- Canonical battery COMPLETE (03:58, 4h13m):
  `rules_20260713_cycle4_ghost_d32_canonical` = **98.3925** mean seat
  on `2027070900..0999` under the adopted ghost+d32 serving default
  (same block, pre-ghost champion config: 98.2975). Descriptive
  reference only; champion identity and canonical champion score
  unchanged. Fresh 8,000-decision serving-default ledger now exists
  (D1 pilot substrate).
- menu512 bank screen auto-started 03:58 (~10 min); Stage 1 retrain
  chain (waiter PID 4153213) deploys 55e8d4c1 and starts v1b when the
  screen exits.

## ADDENDUM (07-14 04:35 — Stage 1 relaunched after env-var incident; menu512 screen VOID; D1 + R1.3b gate chained)

- **Stage 1 chain v3 RUNNING** (PID via waiter 4173058, started 04:18):
  v1b training confirmed (GPU 100%). Incident: `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True` is FATAL for trainers on this box (bisected;
  bridge tolerates it) — burned two chain attempts, ~13 min GPU idle,
  fix is never setting it for trainer processes (INFRASTRUCTURE.md
  warning added).
- menu512 bank screen = VOID (bit-identical to incumbent: frozen menus
  can't widen + `--max-actions` was the wrong flag; real cap is
  `--gumbel-root-menu`, default 256). R1.3b goes straight to a
  preregistered champion-tier sequential CUPED superiority gate
  (block 2027073000..3099) — chained LAST (`r13b_gate_20260714.sh`,
  waiter PID 4173895).
- D1 pilot chained behind Stage 1 (`d1_pilot_20260714.sh`, waiter PID
  4173887): deploys df8e024b (python-only), builds
  `puzzle_bank_20260714_d1_n2048` (stride 11, x2 repeats, no ghost),
  runs `analyze_label_movement`. Hard-root definition + bar + churn
  guard preregistered 04:30 BEFORE data generation.
- Night queue: Stage 1 (~18:30) -> D1 (~1.2h) -> R1.3b gate (~5-8.5h).
  Monitors: b4icf50gw (Stage 1 v3), bqdgb1bp2 (D1 + gate).

## ADDENDUM (07-15 18:15 — Stage A tooling landed; night pipeline armed: fold verdict -> rebuild -> Stage A generation)

- **D1 FUNDED** (pilot 43.2% stable movement at 0.40 pts stake;
  full-ledger replication 43.6% on 7,600 roots). Stage A preregistered
  (EXPERIMENT_LOG 07-15 18:00): tooling complete (`--probe-roots`,
  `--decisions-out`, `--hard-roots-out`; 65/65 exporter tests).
- **Stage 1 fully closed** (all four flag arms + ctrl-SWA lead: nulls;
  everything was continued training). **R1.3b closed** (root-menu 512
  ns at final look).
- **Live on john0:** ghost-fold chain (PID 148805: corpus gen ~1.5h ->
  fold retrain ~2h -> CPU re-eval -> SWA screen). Behind it
  (waiter 150316): Stage A generation — deploys a449b162, REBUILDS the
  exporter, applies the fold safety bar mechanically (ghost ON only on
  a clean numeric pass; fallback OFF), generates 1,250 seeds
  (2026794000..5249) with both sidecars (~7-10h).
- **Superseded 07-16 09:00:** this snapshot was awaiting John for the Stage A
  relabel tranche; John has since fully authorized it and the remaining
  bank-mode/training-view engineering is in progress.
  Monitors: bsfmncz84 (fold), b6f1ousg5 (Stage A gen).

## SNAPSHOT 2026-07-21 14:00 — SUPERSEDES EVERYTHING ABOVE

Full narrative + numbers: docs/v3/RESEARCH_LOG.md section 13.
Blow-by-blow: cascadiav3/EXPERIMENT_LOG.md (2026-07-16 09:00 onward).

**Goal state:** AAAAA (Card-A) campaign CLOSED by John's ruling 07-19 at
~2.6 GPU-days (final baseline 98.19 n1024/d16, goal gap 1.81; D1 screen-
killed, Rival-Lite killed after M1 found ~zero provable endgame headroom).
**Active line: CBDDB** (Bear C, Elk B strict-diamond, Salmon D, Hawk D,
Fox B), target **>105** (John). Identity `..._cbddb_..._2026_07_19`.

**Banked CBDDB numbers (AAAAA champion, zero retraining, screening block
2027190000-99):** floors 88.6/88.5/80.9; n256/d4 x100 = 99.4675;
n1024/d16 x30 = 101.2. Warm-start fine-tune on same-budget self-play is
a PROVEN DEAD END (98.75 across naive/vonly/both anchor arms — teacher-
student gap). Trust-region anchor exists in the trainer (default-off).

**Running now:** from-scratch campaign (random-init; John 07-20; re-
scoped ~3 GPU-days). Bootstrap trained (EI-0 greedy, model-S, 15k steps,
checkpoints/full_v3_cbddb_from_scratch_bootstrap). Cycle fs_c1 live on
john0 (launched 13:09: 400+40 seeds n128/d2 from seeds 2027194000+,
warm-start train w/ q-quantiles 8, eval n256/d4 x100 + n1024/d16 x30;
first number ~21:00 07-21). Monitor bt3xnnmwf; log
cascadiav3/logs/cbddb_fs_c1.log; runner run_cbddb_cycle.sh.

**Next decision:** milestone gate after cycle 2 (seeds 2027196000+):
continue only if slope projects past 99.4675 (n256/d4); ultimate bar
101.2 (n1024/d16); >105 certification ONLY on fresh block 2027195000+.
Fallback if from-scratch stalls: stronger-teacher warm-start (deep-
search labels) + anchor — the one warm-start shape not killed.

**Standing rules:** one scientific job at a time on john0; never deploy
during a scientific run; preregister before peeking; kill only by
explicit PID; poll terminal state on long jobs (monitors can drop);
seeds ledger — spent: 2026794000-5249 (Stage A), 2027160000-99 (Gate 0),
2027190000-99 (CBDDB screening, reusable paired), 2027191000-1399 (s2),
2027193000-3550 (bootstrap), 2027194000-4840 (fs_c1); reserved:
2027195000+ (certification), 2027079000-99 (unused D1 gate block).

## LOCAL PUZZLE EXPLORATION 2026-07-23 11:08 — EXACT TAIL IN PROGRESS

This is independent local CPU work and does not displace or make claims about
the john0 campaign above. The requested 20-token/cap-6 pure-wildlife catalog
is sequenced AAAAA then CBDDB. The base AAAAA retry exited naturally at 11:07
with 711/826 exact: one result beyond the imported 710 proofs and 115
timeouts. AAAAA reached 732/826 exact after the terminal three-host fleet pass
added 13 coordinate-model certificates, three
overlapping the seven frozen specialized certificates. The collector
validated 115/115 returned rows; 102 timed out and 94 unique vectors remain
uncertified after the specialized-proof and split-Salmon bitset union. CBDDB's
two-spare-core heuristic staging completed all 826 vectors in 224.244154
seconds with zero independent scorer/connectivity failures. Its current
84-point leader at counts `(6,0,3,6,5)` is only an exact-solver warm start.
Its four-host exact taskset is staged but remains blocked on AAAAA completion.
Durable methodology/status: `docs/v3/WILDLIFE_OPTIMAL_CATALOGS.md`; detailed
provenance: `cascadiav3/EXPERIMENT_LOG.md`.

Latest exact-tail transition: a primary-literature pass implemented layered
single-anchor packing, exact radius-two fox-neighborhood tables, and canonical
Fox-A witness/ring coupling. All are sound and containment-tested; all failed
their preregistered strength gates on already-certified exclusions, so no
unresolved row was screened and no proof count changed. Static local tables
inside the monolithic coordinate model are now closed. The next ranked
architecture is external canonical enumeration of complete interacting
fox/motif components, arithmetic profile sharding, factorized far cases, and
local cell-set packing. Review and implementation contract:
`docs/v3/AAAAA_EXACT_TAIL_LITERATURE_REVIEW.md`.

Cap-seven side analysis (11:38) exhaustively enumerated all 2,226 count
allocations without launching a coordinate search. The comparable elementary
AAAAA bound changes 73→74 after the universal six-neighbor fox-incidence
constraint; CBDDB changes 100→102. These are bounds, not achieved scores.
Durable derivation: `docs/v3/WILDLIFE_CAP7_UPPER_BOUNDS.md`.

New local catalog scope (12:55): John requested one certified-optimal board
for each of all 1,024 ordered A/B/C/D wildlife-card combinations at cap six.
The exactness/provenance contract is preregistered in `EXPERIMENT_LOG.md`.
All 4,096 frozen independent/production scorer comparisons passed. The first
64-ruleset candidate calibration completed in 12.57 seconds, but its separable
count bounds left a median 406/826 count branches above the incumbent.
The generalized exact model has since matched all 1,024 objectives on a frozen
board. Tight Hawk-C visibility and Fox-C planar bounds reduce the pilot median
to 275.5 branches; a known AAAAA exclusion reproved in 12.4 seconds while a
hard CBDDB exclusion remained unknown at 60 seconds. Candidate staging and
tail-proof optimization are therefore active; no arbitrary-ruleset score has
yet been accepted as optimal and no production proof shard has launched.

Full all-card incumbent staging is preregistered at 13:16 over four disjoint
256-ruleset john1–john4 shards (revision `9de711ec`, 12×100k annealing states
per ruleset). All four shards completed naturally by 13:28. Cross-scoring
3,502 direct and retained boards improved 444/1,024 rows; every merged winner
passed independent and production scoring. Scores span 65–85, with eight
85-point leaders, but none reaches its sound bound. The merged artifact is now
the frozen exact warm start; stratified proof-runtime calibration is next.

The eight-rule stratified exact calibration is preregistered at 13:31:
30 seconds/count, five minutes/ruleset, one eight-worker ruleset at a time on
each mini, with easy/reference/median/leader/max-tail strata. No full exact
launch occurs until its frozen completion band is applied.

All four calibration shards launched at 13:32 and exited naturally by 13:43.
Only ACACA 76 and ADACA 77 completed: **2/8**, the preregistered rejection
band. AAAAA retained 30 unresolved count branches; ADCCB/CADAC/CADDA/CBDDB/
DDDDD retained 91/612/125/309/141. The generic five-minute production shape
is rejected and no 1,024-rule exact launch occurred. All valid exclusions and
witnesses are preserved in the collected calibration ledgers. A specialized
sound-filter pass and frozen-row recalibration are now required.

The first preregistered specialized pass derives an exact cap-six hex-lattice
cross-adjacency table for Fox C. Its 36 connected-component subproofs are
sharded 9/9/9/9 over john1–john4, then combined by a complete disconnected
component DP. It must pass exact status/bound, coverage, symmetry, and frozen
frontier-reduction gates before selection.

The 13:53 first launch failed before its first solve because a reused
adjacency helper assumed 20 tokens; no result artifact exists. The corrected
v2 uses a size-generic adjacency model, has a live one-by-one regression test,
and makes the worker self-record its PID. Its identical frozen rerun is
preregistered under a new single-use tag. All four corrected shards launched
at 13:55 with valid self-written PIDs and fresh heartbeats.

All four v2 shards exited 0 at 13:56. The collector proved all 36 lattice
component maxima with exact solver bounds and symmetry. The selected table
cuts the frozen 256-rule Fox-C frontier 51,854→15,553 (−70.01%) and CADAC
612→361 (−41.01%), passing both preregistered thresholds. Across all rulesets
it removes 36,301 count branches (−22.33%) without changing an incumbent.

The next preregistered specialized pass applies the same exact
connected-component-plus-DP derivation to Fox B: maximum foxes with at least
two neighbors of one target species for all 36 cap-six side-size pairs.
Selection requires complete proofs plus ≥5% Fox-B or ≥10% ADCCB/CBDDB
frontier reduction.
All four Fox-B derivation shards launched at 14:01 with fresh heartbeats.
They exited cleanly; collection then rejected an inapplicable left/right
symmetry assertion before publishing a table. Qualification is asymmetric by
definition. The fixed collector retains exact ordered coverage and
objective/bound equality, adds an asymmetric regression, and will re-collect
the same immutable proofs.

The corrected collector accepted all 36/36 exact Fox-B component proofs.
The selected table cuts the frozen Fox-B frontier 65,519→59,188 (−9.66%),
ADCCB 94→71, and CBDDB 309→283. Together the Fox-C/B tables remove 42,632
of the original 162,591 all-rule count branches (−26.22%).

The third preregistered specialized pass targets Fox A. It proves all 216
cap-six `(foxes, target-A, target-B)` maxima for foxes observing both species,
then combines components in a complete 3-D DP. It replaces the loose `2ab`
pair capacity only if exact coverage/symmetry and the frozen ≥5% Fox-A or
≥10% AAAAA/CADDA reduction gate pass.
All four 54-triple Fox-A derivation shards launched at 14:08 with fresh
heartbeats.
All 216 subproblems resolved exactly (194 optimal component types, 22
impossible connected component types, zero unknown). Collection paused before
publication to correct the wrapper's component-absence semantics; immutable
proofs will be re-collected with exact infeasibility represented as no DP
transition.

The corrected Fox-A collector accepted all 216 exact component resolutions,
but the 343-entry global table equals the existing `min(foxes,2ab)` capacity
everywhere. The preregistered improvement gate fails at 0%; the table is not
integrated. Two-species Fox-A capacity is now closed as an optimization
direction.

A specialized-bound recalibration is preregistered on the six formerly
incomplete frozen rows, retaining certified ACACA/ADACA as the original
eight-row baseline. It uses the same 30-second/count, five-minute/ruleset
shape and applies the original combined completion band before any full
launch.
All four specialized-bound recalibration shards launched at 14:14 with fully
pinned exact-support and bound identities and fresh self-recorded heartbeats.

The recalibration exited naturally by 14:25 and completed none of the six
hard rows; combined with frozen ACACA/ADACA it remains 2/8, the rejection
band. The new bounds still cut unresolved breadth (AAAAA 23, ADCCB 64, CADAC
361, CADDA 125, CBDDB 283, DDDDD 141), but four rows spent their full budget
on ten unknowns. No full launch occurred. The next preregistered optimization
is a disconnected-relaxation prescreen on the same frozen rows.

The disconnected prescreen is preregistered with identical budgets and
assignments. Its proof identity now includes connectivity mode. Selection
requires exact-exclusion union to complete another frozen row or reduce the
six-row unresolved total by at least 20% from 997.
All four disconnected prescreen shards launched at 14:28 with mode-pinned
identities and fresh self-recorded heartbeats.

The four shards exited cleanly by 14:40. No disconnected row completed.
Unioning only exact disconnected infeasibilities with the frozen connected
proofs reduced the six-row unresolved total just 997→994 (0.30%): AAAAA 23,
ADCCB 61, CADAC 361, CADDA 125, CBDDB 283, DDDDD 141. This fails both
preregistered gates, so the disconnected prescreen is rejected and no full
1,024-row run follows. All exclusions remain frozen evidence; the next
architecture must strengthen card interactions or replace the coordinate
proof formulation rather than merely remove connectivity.

Collector review then corrected an underclaim: a proof-less row whose board
meets the sound global count bound is already certified. DCAAC 69, DCCAC 69,
DDAAC 72, and DDCAC 72 pass that condition. With the frozen ACACA 76 and
ADACA 77 coordinate proofs, 6/1,024 rulesets were certified at that
transition. This
bookkeeping fix does not change any score, unresolved hard-row frontier, or
the disconnected-prescreen rejection.

A 40× deep incumbent calibration is now preregistered on the same six hard
rows (48 million states maximum each, john1–john4). It is a measured warm-
start intervention, not a proof: only independently rescored strict
improvements will be merged, and heuristic equality alone never certifies a
row. All four shards launched at 14:50 with pinned sources/binaries, fresh
self-recorded PIDs, and live first-index heartbeats.

At 15:02 john1's local wrapper and child were found dead without a terminal
marker; its log never advanced past launch. Root cause is orchestration:
`nohup` was used instead of the fleet contract's detached local `screen`.
john2 remains live and john3/john4 are terminal, but every result file remains
closed. No restart or reassignment is authorized; the exact-catalog work
continues independently.

john2 later reached terminal exit 0 as well. The calibration is now
`failed_sealed`: all remote outputs remain unread because the frozen
all-four-terminal-marker condition cannot be met. Nothing from this heuristic
run enters the catalog. A fresh john1 `screen` retry needs explicit permission.

In parallel, a coupled adjacency-resource bound is preregistered. It will
charge Bear/Elk/Salmon motifs and Fox observations against shared exact
lattice edge caps and degree-six budgets, including explicit sharing of
salmon–fox edges. Selection requires ≥5% total frontier reduction or ≥10% on
two hard rows after containment tests; otherwise it is a frozen negative.

The exhaustive 846,848-cell comparison passed every containment check but the
coupled bound was identically equal to the existing bound: zero cells reduced
and 119,959→119,959 frontier branches. It is rejected and not integrated.
Degree and pair-edge budgets without richer geometric/card interaction are
now closed as a source of improvement.

The next measured solver optimization is a sound twelve-way dihedral symmetry
break around an arbitrary ordered fox pair. It has a frozen four-case,
30-second paired gate and must either deliver ≥2× on the known exclusion or
resolve a formerly unknown case without regression before selection.

The symmetry break passed all correctness gates but regressed the known
AAAAA exclusion from 20.55 seconds/INFEASIBLE to 30 seconds/UNKNOWN and
resolved no hard case. Arbitrary anchor labels cost more than the twelve
geometric images saved. It is rejected and the original exact source is
restored byte-for-byte.

A narrower anchor-centroid symmetry experiment is preregistered default-off:
it exchanges twelve geometric images for at most six anchor labels while
preserving every other within-species order. It must pass the same correctness
and paired runtime gate before any production selection. The reproducible
default-off implementation and all correctness gates have now landed; the
paired runtime gate is next.

Anchor-centroid also regressed the known proof to `UNKNOWN` at 30 seconds and
resolved no hard case. It is rejected and removed; its source remains
recoverable at revision `eb1e6446`. Simple geometric value-symmetry breaks
that give back any current fox ordering are now closed.

The next default-off exact optimization is an arithmetic score-profile
propagator: enumerate only attainable per-card score tuples above threshold
while preserving the existing coordinate and token-symmetry model. It has the
same complete correctness and paired runtime gate.

The implementation is reproducibly landed default-off. All correctness gates
pass; the four frozen cases need only 2/6/29/886 score tuples and table
construction is under 0.004 seconds. The paired runtime verdict is next.

The in-model table misses selection: the known proof is 6.8% slower and no
hard case resolves. It remains default-off. CADAC and CBDDB branch counts do
fall 35% and 20%, which triggers the preregistered external profile-sharding
follow-up for the tractable AAAAA/CADAC 2/6/29-profile cases.

The external calibration is preregistered as 37 independent connected
single-worker branches over john1–john4, 30 seconds/profile. It selects only
on a 2× parallel known-proof win or complete exact resolution of all profiles
for one hard case; all shard outputs remain closed until four terminal markers.

The fixed-profile runner and deterministic 37-task set are now frozen and
smoke-validated, with round-robin 10/9/9/9 assignments. All four exact
runtimes and seven source/input hashes passed preflight. The calibration
launched at `2026-07-23T19:35:12Z` and all four shards exited zero by
`19:36:22Z`. Terminal collection verified exact coverage and identity:
5/37 profiles were infeasible and 32 remained unknown (AAAAA known 1/2,
AAAAA hard 4/6, CADAC 0/29). Both frozen selection conditions fail, so
external score-profile sharding is closed; its five exclusions remain reusable.

The first component-local replacement is implemented for the four AAAAA
split-Salmon branches: exact canonical fox-layout enumeration plus bitset
Bear/Elk/Hawk set packing. An unregistered first-case diagnostic closed all
57 prior timeouts in 92.606 seconds (6.50× serial); it is not proof evidence.
A hash-frozen four-host formal reproduction is next.

The formal split-Salmon proof is now preregistered as one frozen case per
john1–john4 host, with 57/57/57/95 exact submodels and sealed shard outputs.
Only four exact-infeasible results may combine with the already-passed
maximum-Salmon branches and validated incumbents.

All four hash/runtime/collision preflights passed and the formal bitset shards
launched at `2026-07-23T19:53:14Z`. All four exited zero by `20:03:35Z`;
the collector certified optima 66/63/62/62 for the four registered counts
after maximum-Salmon union and Rust rescoring. AAAAA is now 732/826 exact.
Fleet speedup was 4.82×, below 20×; the Bear-heavy quadratic dominance tail is
not the governing bottleneck. Its exact subset-index replacement passes all
property/certificate tests, but the frozen speed rerun reproduced exact
57/57/57/95 exclusions in 101.818/618.447/100.767/194.250 seconds. The
618.447-second critical path is only 4.84× and fails the ≤149.739-second gate.
The hard case materialized 14,648,710 cover configurations; fuse cover
generation with packing so infeasible assignments are pruned before
materialization, then remeasure before generalization. The first exact
implementation orders species by candidate volume and caches identical
per-layout deficit queries; 16 tests and Ruff pass. The frozen case-1 probe
retained all 57 exact exclusions and fell 618.447→12.279 seconds (50.36×).
A fresh four-case reproduction must now retain 57/57/57/95 exact exclusions
and a ≤149.739-second critical path before generalization. Its first wrapper
attempt produced no scientific output because of a zsh PID-scalar error; the
same frozen run was registered under a fresh retry tag with a native PID
array. The retry reproduced all exact 57/57/57/95 exclusions in
75.108/12.992/72.167/106.349 seconds. Its 106.349-second critical path is
28.16× versus the frozen sequential screen and clears the 20× gate. The
species-ordered cached bitset engine is selected for generalization.

Catalog-accounting transition: a fail-closed monotone augmenter now rebases
the existing exact exclusions onto the validated deep candidates and imports
the already-complete global AAAAA certificate. Its frozen production run must
validate every board through both scorers and all 128 AAAAA threshold-69
exclusions; if accepted it raises the integrated completed-ruleset count from
six to at least seven without rerunning solved proof work.

The first augmentation output is rejected despite passing every scoring check:
its base artifact omitted the earlier complete ACACA/ADACA shards and therefore
reported only five completed rows after adding AAAAA. A fresh correction now
recollects all three frozen proof directories and must recover exactly six
base rows before the unchanged augmentation may produce seven.

The first correction wrapper passed collection parents to a non-recursive
collector and found no proof files, so it is invalidated. Retry 2 enumerates
all 12 john1–john4 leaf directories explicitly and retains the same six-row
base/seven-row augmented acceptance rule.

Those leaves exposed an older identity schema in the complete ACACA/ADACA
shards; retry 2 failed before output. A strict adapter now reconstructs the
missing support/rules hashes from the complete fleet ledger's pinned git
revision and also requires its per-index output hashes. Retry 3 retains the
same acceptance rule with this legacy provenance made explicit.

Retry 3 then failed before output because current tighter bounds shortened an
incomplete legacy unresolved list. Retry 4 validates that historical list's
summary against the hash-matched fleet ledger, retains its exact exclusions,
and recomputes only the final union under current bounds.

Retry 4 passed. The integrated all-rules catalog is now 7/1,024 exact:
AAAAA 68, ACACA 76, ADACA 77, DCAAC/DCCAC 69, DDAAC/DDCAC 72. All
1,024 selected boards pass independent and production scoring; 17 deep
improvements are rebased; 119,139 count branches remain across 1,017
incomplete rulesets. The score-85 leader on eight rulesets remains an
incumbent, not a holistic optimum.

Next proof batch: seven rulesets have exactly one remaining count each.
Their frozen connected exact jobs are assigned 2/2/2/1 across john1–john4 at
300 seconds/count. Results stay sealed until four terminal markers; a clean
pass would advance the integrated catalog 7→14.

The first wrapper attempt failed before Python on all hosts because Bash 3
treated an empty connected-mode argument array as unbound under `set -u`.
There are zero result files. The fixed worker passes all-host shell checks and
the identical batch is frozen under a fresh retry tag.

The retry launched at `2026-07-23T21:14:45Z`; all four shards were terminal
by `21:15:25Z`. Every one-count threshold query returned exact
`INFEASIBLE`, with zero timeout/unknown. ADAAC 72, ADACC 80, BDAAC 73,
BDACC 80, BDBAC 73, CDACC 80, and DDACC 80 are now certified. The
independent and production scorer gates passed over all 1,024 selected
boards; the integrated catalog advances 7→14 exact with 119,132 count
branches across 1,010 incomplete rulesets. The score-85 leader on eight
rulesets remains an incumbent, not a holistic proof. Curated evidence:
`docs/v3/evidence/all_wildlife_near_complete7_2026-07-23.json`.

Next exact batch: the eight rulesets with exactly two remaining branches are
frozen under taskset SHA
`87885d0ac0325fe70e84a133d9d5eb11ffeab17b4eecd252ff339a2c416b9f86`
for a 2/2/2/2 john1–john4 pass. The unchanged 300 seconds/count and 330
seconds/ruleset limits give an approximately 11-minute hard wall cap per
host. Results remain sealed until all four terminal markers exist; a clean
pass would advance 14→22 exact.

The two-branch batch launched at `2026-07-23T21:22:54Z` under revision
`e964665b`; all four wrappers passed pinned source/candidate/runtime/collision
checks. All four were terminal zero by `21:23:27Z`. Every one of the 16
threshold queries returned exact `INFEASIBLE` in 0.77–3.39 seconds, with no
timeout/unknown. Independent and production scoring of all 1,024 boards
advances the catalog 14→22 exact, leaving 119,116 branches across 1,002
unresolved rows. The score-85 leader remains an incumbent. Curated evidence:
`docs/v3/evidence/all_wildlife_two_branch8_2026-07-23.json`.

Next tail tranche: a tested deterministic builder froze every row with 3–5
remaining branches from the 22-row catalog. The taskset has 23 rulesets / 95
queries at SHA
`5dfa433a83fba599322a100c045f04738b25c46ae65cb98721d681c731b92340`.
Assignments are balanced at 25/22/24/24 branches across john1–john4. The
unchanged 330-second ruleset cap gives a 33-minute per-host hard wall;
results will remain sealed until four terminal markers.

The 3–5-branch batch launched at `2026-07-23T21:29:48Z` under revision
`b1fb514d` and all four wrappers exited zero by `21:36:22Z`. Twenty-two of
23 rows closed exactly; 93/95 queries were `INFEASIBLE`. DCACC proved three
branches and retains two `UNKNOWN` branches after its 330-second row cap.
Independent and production rescoring advances the catalog 22→44 exact,
leaving 119,023 branches across 980 rows. Curated evidence:
`docs/v3/evidence/all_wildlife_three_to_five_branch23_2026-07-23.json`.

Next tail tranche: every row with 6–10 branches is frozen in taskset SHA
`d4c159c5f4b4990baf6696bc156c8ff1a8085bfd39b07e0c698d39c7da25736a`.
It covers 43 rulesets / 354 queries, balanced 90/90/90/84 across
john1–john4. The 330-second row cap gives a 60m30s hard wall per host.
DCACC's two residual branches remain separate; no unknown is discarded.

The 6–10-branch batch launched at `2026-07-23T21:40:31Z` under revision
`f349a8cf`; all four wrappers exited zero by `22:05:28Z`. The pass certified
36/43 rows, retained 301 exact exclusions and seven feasible witnesses, and
left 35 branches across seven partial rows. Five incumbents improved.
Independent and production rescoring advances the catalog 44→80 exact,
leaving 118,704 branches across 944 rows. Curated evidence:
`docs/v3/evidence/all_wildlife_six_to_ten_branch43_2026-07-23.json`.

Bounded-maximization sidecar: implementation and corrected engineering smoke
pass. The one-second DCACC case retained `[77,78]`, and the fail-closed
collector production-rescored all 1,024 boards without changing their
scores. This validates plumbing only; a preregistered representative fleet
pilot must show useful bound contraction or a better witness before scaling.

The bound pilot is frozen at eight cases / two per mini. Four target known
one-point timeout branches; four target CBDDB/DDDDB/DBDAB/DADAB high
ceilings. Broad scaling requires a strict high-ceiling contraction or a board
above 85; timeout-only success keeps the lane targeted. Results remain sealed
until four terminal markers.

The eight-case bound pilot launched at `2026-07-23T22:25:51Z` under revision
`7c7ce831`; all four wrappers were active after pinned preflight. The hard
terminal deadline is approximately `22:36Z`.

All four bound-pilot wrappers were terminal zero at `22:36:07Z`; only then
were the eight outputs opened. The four known one-point cases did not
contract. All four high-ceiling cases did: CBDDB/DDDDB 99→98 and
DBDAB/DADAB 94→92 at ruleset level. There were no new witnesses or exact
closures. The frozen verdict is `BROAD_SCALE`. Independent validation and
the all-1,024 production rescore preserve 80 certified rows and the holistic
`[85,99]` interval; the upper remains 99 because eight other rows were not
part of the pilot. Curated evidence:
`docs/v3/evidence/all_wildlife_bound_probe_pilot8_2026-07-23.json`.

The collector now persists aligned per-count certified uppers and inherited
probe provenance; a two-generation regression prevents recursive merges from
forgetting earlier contractions. The selected broad pass contains every
current top-frontier count tie for every row above 85: 435 tasks across 280
rulesets at taskset SHA
`623cb660909c8e0d86e46c8a5776f4e567d9a573c760252505733508ab844759`.
Round-robin fleet balance is 109/109/109/108 tasks, with a just-under-ten-hour
hard wall. Outputs remain sealed until four terminal markers.

The top-frontier pass is live. john1/john2 began at `22:44:57Z` and
john3/john4 at `22:46:21Z`; all four passed runtime, source, input-hash,
collision, and idle checks and have fresh five-second heartbeats. Wrapper
PIDs are 63770/78279/36045/36561. The hard terminal deadline is
`2026-07-24T08:46:21Z`. Ledger:
`cascadiav3/fleet/all_wildlife_bound_probe_topfrontier435_fleet_20260723.json`.
Do not inspect task outputs before all four exit files exist.

John extended the useful fleet window through approximately 09:00 EDT. A
non-adaptive continuation is frozen against the same sealed base: 224
second-distinct-frontier cases across 113 rulesets, exactly 56 per host,
taskset SHA
`5c7221f6fb63c8d4777a8d68d4f93bfe86c78c3c9ca63f1c39bef64ee0c2a8c0`.
It is disjoint from all 435 live cases and was selected without reading their
outputs. A durable HOLD-aware waiter may launch only after four zero exit
files and a complete hash/collision/runtime preflight. At the measured
five-minute cadence it should use the fleet through roughly 08:45 EDT.

That waiter is active in detached screen
`wildlife_bound_secondfrontier224_waiter`, PID 82731, with SHA
`c2471c8fe1282ca3e84f93d2bf758d3e17864d122617762482ef85a244e13a2a`.
Its first heartbeat at `23:42:04Z` reported 0/4 predecessor terminals. HOLD
path: `cascadiav3/logs/HOLD_all_wildlife_bound_secondfrontier224`. Queue
ledger:
`cascadiav3/fleet/all_wildlife_bound_probe_secondfrontier224_queue_20260723.json`.

The first 435-task pass is terminal zero on all hosts. Its safe preliminary
read shows 434 strict contractions, 137 exact count closures, 12 uniquely
improved rulesets, and a provisional global interval `[85,98]`; formal
collector/oracle publication is still pending.

The 224-task continuation launch partially failed. john1 completed its 56
tasks, but nested remote-screen quoting launched nothing on john2–4; the
waiter's receipt bug accepted three blank PIDs and exited zero. No remote
result exists and all continuation payloads stay sealed. An exact 168-case
recovery set is frozen at SHA
`516ecf0b7737a1d2f9cf2bd5e949d03bda2f1461a41492b1d70fe52b95748017`,
balanced 42/host. A new host-local nohup launcher requires a numeric live
worker PID plus heartbeat before success. Final merge coverage is 659 unique
probes.

John has now explicitly authorized the failed-sealed deep-incumbent recovery.
Only john1's missing AAAAA/CADAC indices `0,562` will rerun under fresh tag
`all_cards_candidate_deep_recovery_john1_20260723` in the required detached
screen. The terminal john2–john4 parent outputs and new local output remain
sealed until the recovery terminal marker exists.

The hash/collision/idle preflight passed and recovery launched at
`2026-07-23T19:42:18Z` in detached screen
`wildlife_candidate_deep_recovery_john1` (wrapper PID `28671`), initially on
AAAAA index 0. Do not inspect any candidate output before its terminal marker.

The candidate recovery exited zero at `2026-07-23T19:59:49Z`, satisfying the
all-six unseal condition. Six files are durably collected; board validation
and all-1,024 production cross-scoring passed. Seventeen rulesets improve;
CADAC rises 66→68 and DDDDD 78→79, reducing their frontiers 361→214 and
141→100. The holistic warm-start score remains 85 on eight rulesets. The
new merged candidate SHA is
`96a8ba0464fe30e294b293483d15d667808ab0c737c96968d362b22cde1881c9`;
it is heuristic evidence only.

---

# SNAPSHOT 2026-07-23 20:30 — CAMPAIGN CLOSED (supersedes everything above)

John's ruling 2026-07-23: Cascadia experiments CLOSED for now; he is
moving to a different game. NOTHING IS RUNNING (john0 verified idle;
minis handed back to John for his wildlife-catalog work).

- Best CBDDB result: AAAAA champion ZERO-SHOT — 99.4675 screen /
  101.2 full battery (P90 106.1). >105 target not reached.
- Every training intervention lost to zero-shot (fine-tune tax
  98.7-98.9 band x4; from-scratch plateau 98.3).
- Resume here: docs/v3/RESEARCH_LOG.md §14 (resumption contract:
  final table, negative results, options A/B/C with B recommended,
  seed ledger, scripts, checkpoints, standing rules).
