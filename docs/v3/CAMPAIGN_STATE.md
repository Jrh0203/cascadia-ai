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
