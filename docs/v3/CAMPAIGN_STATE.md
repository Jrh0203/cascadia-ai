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
completed the 100-game greedy/no-search floor plus both n256/d4 arms and is
now running cycle4 n1024/d16, with 51/100 raw games complete at 12:00 EDT;
distq_k8 n1024/d16 follows on the same fresh seeds. A live sidecar copied the
growing distq-n256 seed files with overwrite-on-poll semantics; a second
watcher does the same for both n1024 arms. Each publishes only after strict
100-seed validation. This preserves category scores even though the deployed
pre-fix reducer retains totals only. Watcher pid files are
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

**Quarantined v4 expansion generation (launched 12:09 EDT):** while john0
remains exclusively on the promotion chain, idle john2–john4 are generating
three data-only 50-seed blocks at the already validated n8/top4/d1 shape.
Seeds are `2027073600..49`, `2027073650..99`, and `2027073700..49`; PIDs are
`90485`, `58489`, and `26369`. Every launch reverified source marker
`6e89d955`, teacher manifest `b8886c24...`, and weights `33559aab...` before
starting. Expected output is 4,000 roots per host / 12,000 total. These shards
are speculative fit-capacity inventory only: keep them out of the fixed
fit/selection/verdict pilot, do not copy them to john0's queue, and do not
admit them to any training unless they complete, validate, and the frozen-head
pilot first passes. john1 remains reserved for the active UI/champion service.
Completion watchers PIDs `90916 / 58926 / 26606` will reopen each shard and
write summary/invariant reports only after its producer exits successfully.
Admission no longer relies on those per-host checks alone: the new
`audit_structured_q_shards` tool requires all expansion NPZs and sidecars on
one host, proves one exact contract plus seed disjointness from the locked
three-way pilot, and fails closed on overlap or sidecar tampering. It has
already passed against the real locked 2,400-root corpus. Audit schema v2 also
records per-shard final-score, component score-to-go, selected-teacher error,
Nature-spending frequency, and q-valid-menu distributions so expansion drift
is visible before any admission decision.
Canonical harvest is `cascadiav3/scripts/fetch_structured_q_expansion.sh`.
It refuses while any producer or validator is live, verifies passing remote
reports and remote/local NPZ plus manifest hashes, fetches all six artifact
types, then runs the six-shard audit. It has no john0 or training copy path.

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
