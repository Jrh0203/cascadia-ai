# Cascadia v3 Research Log — the road to 100

**Deliverable doc.** Every research direction tried in the Gumbel self-play
campaign: what it was, why we tried it, what we measured, and the verdict.
Updated continuously; the freshest entries are in the "Active program" section.

Goal (the gate): **mean seat score ≥ 100 over 1,000 games of 4-player
self-play.** Best legacy measurement as of 2026-07-08 evening: **98.40** (distq_k8
at n1024/d16, 100 games; +0.12 ns vs the 98.28 scalar champion — the two
are statistically tied at high search budget, distq strictly better at
low budget). This number predates the corrected optional three-of-a-kind
policy action and is not a current promotion baseline. See the rules
compatibility break below.

---

## Rules compatibility break (2026-07-08)

The engine previously forced every free three-of-a-kind wildlife refresh.
Officially, the active player may accept or decline. The corrected policy
stack evaluates both branches, and unused drafted wildlife is regression-
tested to return to the bag before the end-of-turn market refill. Contract:
[`RULES_CONTRACT.md`](RULES_CONTRACT.md).

All results in this log through the 98.40 distq measurement are legacy under
the forced-refresh policy space. They remain useful architecture evidence but
cannot promote a post-fix model or serve as paired controls. Greedy, no-search,
and Gumbel baselines must be regenerated under rules semantics
`cascadia-base-official-2026-07-09`.

### Corrected no-search floor (2026-07-09)

Fresh seeds `2027070900..2027070999`, 100 complete four-player games per arm,
ruleset and exact source revision enforced:

| Policy | Mean seat score | Paired delta vs greedy | 95% t-CI |
|---|---:|---:|---:|
| greedy-v1 | 87.5450 | — | — |
| cycle4 policy head | **91.8425** | **+4.2975** | **[+3.8705,+4.7245]** |
| cycle4 Q head | **90.8925** | **+3.3475** | **[+2.8507,+3.8443]** |

All 24,000 per-ply rows are retained. Optional refresh was a real decision:
the policy arm accepted/declined `594/352`, Q `636/364`, and greedy
`1005/398`. The no-search interactive harness deliberately uses greedy-v1 to
make the pre-draw market decision before the selected model head ranks the
revealed draft, so this result proves the direct draft policy remains strongly
above greedy under corrected rules; it is not evidence that the model head
itself learned refresh choice. The in-flight Gumbel baselines make that choice
through search and are the relevant evidence for that question.

The policy-head `+4.2975` mechanism is broad but dominated by planning terms:
wildlife `+0.8100` (CI `[+0.4458,+1.1742]`), habitat `+2.3900`
(CI `[+2.1592,+2.6208]`), and retained Nature Tokens `+1.0975`
(CI `[+0.8988,+1.2962]`). The Q-head decomposition is likewise CI-positive:
wildlife `+0.6075`, habitat `+1.8775`, Nature Tokens `+0.8625`. The direct
transformer policy's clearest edge is therefore habitat construction plus
resource restraint, not merely larger immediate wildlife patterns.

### Corrected cycle4 search baseline (2026-07-09)

The first complete corrected-policy search arm used the cycle4 M checkpoint,
fresh seeds `2027070900..2027070999`, n256/top16/d4, blend 0.5, 16 interior
actions, and eight hidden replacement samples per optional refresh. Exact
source revision and corrected rules identity were recorded.

- mean seat score `97.0675`; P50 `97.0`; P90 `100.1`;
- 2/100 game means at least 100;
- 952 refresh opportunities: 565 accept, 387 decline (`59.35%` accept);
- mean decision `11.729s`; refresh decisions `54.908s`, ordinary decisions
  `5.896s`;
- 2,048,000 chosen-branch simulations plus 2,094,336 market-decision
  simulations.

This is a baseline, not a paired promotion verdict; distq on the same seeds
is in flight. It establishes that the corrected search actually exercises
both accept and decline at scale and that the score remains near the legacy
n256 band despite a materially larger policy space. The deployed reducer
retained total seat scores and all 8,000 decision rows but deleted its
temporary per-game category rows, so wildlife/habitat/Nature decomposition
cannot be reconstructed honestly from this artifact. A sidecar is preserving
the in-flight distq raw game files, and the permanent benchmark now writes a
complete seed-ordered game ledger plus category aggregates; publication fails
if any seed is missing. Do not infer category mechanisms from this cycle4 arm.

### Exact final-personal-turn frontier (K1, 2026-07-09)

The first exact-endgame slice is implemented behind
`--gumbel-exact-endgame-turns 1`. When the active seat has one personal turn
left, every afterstate already contains that seat's final score. The policy
therefore enumerates the complete legal menu, ignores the serving root-menu
cap, and chooses the exact maximum without a model call or simulation. The
free-refresh branch remains decision → chance → draft: exact accepted-market
optima are averaged over hidden replacement samples before the real draw is
revealed. K>1 and table-total combinations are rejected because neither has
this one-ply terminal identity.

The permanent comparator validates rules, source, checkpoint name, seeds,
all non-K1 search settings, 80-row trace coverage, identical actions through
ply 75, exactly four zero-simulation K1 decisions per game, and seat-0 exact
score non-regression. This caught two invalid MPS comparisons that an ordinary
paired-score script would have misreported: cross-host execution diverged at
ply 5; same-host two-worker execution diverged at ply 24. A same-host
four-worker exact arm also hit a Metal command-buffer OOM. These are ops
findings, not K1 results.

The valid serial john4 smoke used two matched seeds at n16/d2, pure bootstrap,
and four market samples. Pre-K1 traces were identical. Baseline and K1 both
scored `92.25`; per-seed deltas were `0/0`, so the score verdict is
inconclusive engineering-only evidence. K1 changed 6/8 final actions but chose
equal-scoring alternatives; seat-0 deltas were `0/0`. The exact frontier was
`8.86x` faster (`4.212s` → `0.476s` across eight decisions), which became only
a `1.2%` mean-decision / `1.3%` wall-clock improvement over whole games.

**Verdict: proceed to the preregistered 100-seed corrected n256/d4 CUDA
gate, but do not claim strength.** The smoke establishes correctness and a
local cost win, not points. The CUDA gate regenerates both arms from the same
new revision and rejects any pre-K1 trace divergence. Extending to K2 remains
conditional on that score/cost verdict; K2 requires a genuine max^n/chance
tree, not another one-ply shortcut.

---

## 1. Architecture

### 1.1 Model: CascadiaFormer

Transformer over tokenized game state (M ≈ 100M params; L ≈ 207M tested).
Inputs: public-state tokens + per-action feature tokens with pairwise
relation-bias attention (CGAB). Heads:

- **policy logits** (per action)
- **q / score-to-go** (per action, scalar): predicted final own score minus
  exact afterstate score — serving always grounds Q as
  `exact_afterstate_score + score_to_go`
- **value_vector** (4 seats, absolute seat order): predicted final score of
  every seat
- **rank / score-decomposition / uncertainty / aux heads** (training-time
  regularizers)

### 1.2 Search: Gumbel AlphaZero over determinized worlds

Rust exporter (`real-root-exporter/src/gumbel.rs`):

- Root: batched model eval → Gumbel top-m over the full legal menu →
  sequential halving on `g + logits + sigma(completed_q)`.
- Hidden information: every simulation **redeterminizes** the hidden
  tile/bag order (strict no-peek); `d` distinct determinized worlds are
  cycled across each action's simulations (common random numbers across
  actions).
- Optional three-of-a-kind refresh: decide from public information, value
  accept over `8` independent hidden replacement samples by default, commit
  accept/decline, then search the real revealed market. The refresh chance
  sample count is recorded separately from `d`.
- Interior plies: every seat advances by argmax of its own derived final Q
  (max^n multiplayer), menus capped at k_interior.
- Leaf value: `w·(model bootstrap) + (1−w)·(sampled-greedy rollout)`;
  w=0.5 at serving, w=1.0 acceptable for training labels.
- Completed-Q: visited actions = mean simulation value; unvisited = model
  derived Q. Improved policy = softmax(logits + sigma(completed_q)).

**Champion serving config: n=1024 simulations, d=16 worlds, w=0.5 → 98.28**
(10.6 s/decision). Scores 11/100 individual games ≥ 100.

### 1.3 Training: expert iteration (EI)

Self-play games at search strength → every visited root exports
(improved_policy, completed_q, real final outcome vectors) as training
targets → objective `gumbel-selfplay` (soft-target policy CE + Q + value +
aux) → gate new checkpoint vs incumbent with paired-seed batteries, 95% CI
excluding zero required for promotion.

---

## 2. Scaling laws & levers measured (all closed)

| Lever | Result | Verdict |
|---|---|---|
| Simulations n (64→2048) | +gains flatten past 1024 | closed, peak n1024 |
| Worlds d (4→32) at ~64 sims/world | **+0.9/doubling** 4→8→16; REVERSES at 32 (CI−) | closed, peak d16 |
| Leaf blend w at serving | w=0.5 best; w=1.0 −2.94 CI− | closed |
| Oracle peek (true hidden state) | **LOSES** to honest multi-world search | key science, see §3 |
| Model capacity (M→L 207M) | flat | closed |
| More data (3×), fresh-from-scratch M | flat (3 independent replications) | closed — EI saturated |
| Better labels (n512/d8-taught cycle-6) | flat | closed |
| Checkpoint ensembles (SWA, lineage, cross-arch M+L) | never CI+; weak members actively hurt (−0.78 CI−) | closed — shared model bias |
| bf16 serving | 26% action agreement w/ fp32 (label-unsafe), ~3% faster | rejected |
| Generation env knobs (gather/row-cap) | flat even at saturation — per-ply lockstep is latency-bound | closed; structural fix queued |
| Trainer data path | **5.5× step-time win** (shard mmap; 1.26→0.23 s/step) | shipped |

## 3. Key scientific finding

**Determinization gains are ensemble variance-reduction, not
hidden-information reasoning.** An oracle that peeks at the true hidden
tile order (upper bound on the value of hidden info) *loses* to honest
search over 16 determinized worlds. Therefore the binding constraint on
playing strength is **noise in the value estimates**, not hidden
information, capacity, or data. Every direction in the active program
attacks value noise or sidesteps the competitive objective.

## 3b. Decision-SNR measurement (2026-07-08)

From the cycle-6 selfplay shard (n512/d8 w1.0 search, 20k sampled roots):
per-decision signal = top-2 completed-Q gap, noise = pairwise SE from the
exported per-action simulation variances.

- top1−top2 gap: median **0.049** points (p25 0.013, p75 0.146)
- pairwise SE: median **0.051** points
- decision SNR: median **1.06** — **46% of decisions are flippable by
  simulation noise (SNR < 1); 62% are marginal (SNR < 2).**

The argmax at the root is a coin flip for nearly half of all plies. This
is the per-decision mechanism behind the worlds scaling law (+0.9 per
doubling ≈ SE ÷ √2) and the headroom estimate for every variance lever
(softmix, TTA, budget shape). Caveat: observed gaps are inflated by the
same noise, so the true flippable fraction is, if anything, higher;
near-tie flips cost little individually but compound over ~80 plies.

---

## 4. Active program (2026-07-08, updated through the evening)

### 4.1 Table-total search objective — CLOSED at serving (v1 −1.65 CI−, v2 −1.05 CI−)

**Hypothesis.** The gate metric is the *table mean* (all four seats are
ours). Max^n competitive search spends points on denial moves that lower
the table mean. Retargeting search values from own-seat to
table-total converts destroyed value into measured points.

**Design.** `--gumbel-table-total`: terminal & rollout values become table
sums; leaf bootstrap = own exact-grounded Q + Σ other seats'
`value_vector` estimates; unvisited-action fallbacks shifted onto the same
scale (additive shift preserves ranking). Interior plies remain selfish
argmax (approximation; noted).

**Experiment.** 100-game candidate arm at n256/d4 w=0.5, paired seeds
2026995000+, verdict vs the existing own-seat n256/d4 baseline (96.95).

**Result (v1): CI− — 95.30 vs 96.95, delta −1.65, CI95 [−2.00, −1.30].**
Diagnosis: v1 recomputed the other-seats bootstrap shift from the value
head at every leaf. The value head was never load-bearing (own-seat
search reads only q/score-to-go), so its per-leaf variation injected
unvalidated eval noise straight into the across-action Q comparison —
exactly the quantity the campaign proved is the binding constraint.

**Result (v2): CI− — 95.90 vs 96.95, delta −1.05, CI95 [−1.41, −0.69].**
Constant root shift removed the value-head noise (recovering 0.6 of
v1's loss) yet the objective still loses ~1 point. Mechanism: the
remaining difference from own-seat search is rollout/terminal leaves
scoring the whole table — ~4× outcome variance per leaf at unchanged
per-action signal. Table scoring at serving leaves is a noise
multiplier; the cooperation signal is smaller than the noise it costs.
**CLOSED at serving** (two variants, both CI−). Training-side
table-native labels (§4.3) are theoretically distinct — training
averages away label noise — but parked: the CI+ distq line owns the GPU.

### 4.2 Softened leaf bootstrap (max-bias correction) — CLOSED (flat)

**Hypothesis.** The leaf value bootstrap takes the **max** over the leaf
menu's Q estimates. The max of N noisy estimates is upward-biased and
high-variance — and eval noise is the proven binding constraint. A
softmax(q/τ)-weighted mean lowers both bias and variance at the cost of a
slightly pessimistic policy value.

**Design.** `--gumbel-leaf-softmix <tau>`; interior advance stays argmax;
τ→0 recovers max. Implemented + unit-tested (monotone in τ, bounded by
max and mean, changes search values end-to-end).

**Result: flat at both temperatures (closed).** τ2 −0.03 ns, τ4 −0.01
ns (CIs ±0.35). Reading: leaf max-bias is common-mode across root
actions — every action's leaf takes a max over a similar interior menu —
and common-mode bias cancels in argmax comparisons, so correcting it
buys nothing. A useful negative: it sharpens where the real noise lives
(across determinized worlds and rollouts, not within-leaf aggregation).

### 4.3 Table-native q head (cycle-7 design, contingent on 4.1) — DESIGNED

If the table-total serving probe pays, the clean EI follow-up is: generate
selfplay labels **with** `--gumbel-table-total` — the exported
`score_to_go` (completed-Q − own exact afterstate) then natively embeds
the other seats' expected finals, so a cycle-7 q-head learns **table-Q**
directly. At serving, search runs table-mode without the value-vector
shift (dfq is already table scale), and even interior plies become
cooperative (argmax table-Q) instead of the selfish approximation. This
supersedes the earlier "reanalyze value targets" idea for the table path:
the q head does everything and the value head is no longer load-bearing.
Requires a `table_native_q` serving flag (table terminals/rollouts, no
shift). Fleet (john1-4, idle) can generate this corpus without touching
john0.

### 4.4 Distributional (quantile) score-to-go head — **CI+ at n256/d4; champion-equal at n1024/d16; EI-1 RUNNING**

**Hypothesis.** Reduce per-eval variance at the source. The head search
actually consumes is the **q / score-to-go head** (not the value head), so
that is where the distribution goes: K=8 quantiles trained with pinball
loss; serving "q" = quantile mean, so bridges and search need no changes.
Multi-quantile trunks regularize the conditional mean (the C51/QR-DQN
effect) and the head sees the target's spread instead of collapsing it.

**Design.** `--q-quantiles 8` + `--init-skip-mismatched` (warm start from
champion, fresh q-head). Recipe otherwise identical to cycle-6 (same data,
same steps/LR/selection) — so the run is a clean "same everything,
distributional head" ablation against a known-flat control.

**Result (n256/d4): CI+ — 97.38 vs 96.95, delta +0.43, CI95
[+0.09, +0.77].** The first training-side win of the campaign, against a
control (cycle-6 recipe, scalar head) that was measured flat three ways.

**Result (champion config n1024/d16): 98.40 vs 98.28, +0.12 ns.** The
gain compresses at high search budget: the quantile head and the
16-world ensemble are overlapping variance reducers, so where search
already denoises, the better head is partly redundant. Net position:
distq_k8 is champion-equal at high budget, strictly better at low
budget (97.38 at 2.8 s/dec — better play for a quarter of the serving
cost).

**EI-1 (running overnight):** generation with the distq model (n512/d8
w1.0, 1,375 fresh seeds), quantile-head training on new+cycle-6+cycle-5
at 1.0/0.5/0.25. Tests whether better-search-from-a-better-head yields
better *labels* — compounding — now that scalar-head saturation is
broken. Fleet (john1-4) concurrently generating a distq supplementary
corpus (held out for a safety-tested low-weight fold-in; never
auto-folded, per the cycle-5 poisoning lesson).

**Quantile-aware serving kill test (corrected rules, 2026-07-09): no CUDA
gate.** The bridge now exposes provenance-recorded q25/q50/q75 statistics
without changing default mean serving. On 160 deterministic full-menu roots
(40,776 actions), the trained K8 head had zero adjacent-head crossings.
q25/q50/q75 changed direct derived-Q argmax on only
`3.125%/2.500%/1.875%` of roots, with average mean-Q regret under `0.0001`.
Search amplifies those small changes into distinct trajectories, but the
same-host n64/d4 screen did not establish a useful direction: q50 was flat
(`95.00` vs `95.00`, one seed), q75 was `-1.25` (`95.00` vs `96.25`, one
seed), and the extended q25 pairs were `+2.25/-0.25/-1.25` (`95.25` vs
`95.00`, mean `+0.25`, n=3, CI `[-4.23,+4.73]`, wall ratio `1.042x`). This is
engineering-only evidence. Standalone risk serving is screened out; keep the
modes as cheap policy-diversity controls for a future league, not as a current
strength claim.

### 4.5 Market-refill chance-node expectimax — DEPRIORITIZED (evidence)

The oracle experiment already bounds this: an oracle on the true hidden
state has ZERO chance-sampling variance, and it still lost to honest
multi-world search. So chance variance is not the binding noise term —
model eval error (decorrelated by input perturbation) is. Explicit
chance nodes would attack a non-binding constraint. Killed before
implementation; reasoning recorded so it isn't re-proposed.

### 4.5b Symmetry test-time augmentation (TTA) — CLOSED (flat at 3× cost)

The one lever family that measurably paid (+0.9/doubling of worlds) is
input-space perturbation that decorrelates model eval error, and it caps
at d=16. Hex-board symmetries are an orthogonal perturbation axis:
evaluate the model on rotated board frames and average.

**Implemented (commits 5ff2303, e0a6e95):** `HexCoord::rotated`,
`Board::rotated` (tile rotation composes r→r−k), 
`GameState::with_rotated_boards`, `TurnAction::rotated`, with the
load-bearing invariant tested over full legal menus: apply∘rotate ==
rotate∘apply, and exact scoring rotation-invariant. Exporter:
`--gumbel-tta k` evaluates each unique row on k rotated frames and
elementwise-averages priors/score-to-go (cache stores the average);
end-to-end mock-bridge policy game passes. Cost: k× model evals.

**Result: flat — 96.91 vs 96.95, ns, at 3× eval cost (8.2 vs 2.8
s/dec); cost-matched it loses to simply doubling worlds (d8 = 97.25 for
less compute). CLOSED.**

**Lesson (load-bearing for future proposals):** rotation barely
decorrelates this model's eval error. CGAB relation-bias attention is
built on relative geometry, which is rotation-invariant, so rotated
frames return nearly the same eval and the same error. Determinized
worlds pay because they perturb the *evaluation problem* (different
hidden futures → independent estimates), not the input representation.
Variance-reduction levers must change the problem, not the frame.

### 4.6 Multi-bridge generation throughput — BACKLOG (enabler)

~2× generation wall-clock via worker partitioning across bridge processes.
No points directly; halves the cost of every probe and EI cycle.

### 4.7 Pairwise action comparator — CLOSED as a serving-strength branch

The complete bounded pilot used 2,400 fresh corrected-rules v3 roots, a fixed
1,600/800 train/validation seed split, confidence-filtered bidirectional
pairs, and a 99,072-parameter rank-64 antisymmetric head with the incumbent
fully frozen. It learned the labels: selected held-out pair accuracy rose from
60.4% to 66.0%, and the full probe was 69.5% confidence-weighted.

That did not translate into better routing. The serving-aligned probe first
fixed the incumbent logits' top-16 mask and evaluated all modes inside it. On
206 qualified roots, pure Borda gained only two net top-1 hits
(`30.58% -> 31.55%`, paired bootstrap delta CI `[-3.40,+5.34]` percentage
points) while increasing completed-Q regret (`1.1496 -> 1.2121`). Adding
logits and Borda was top-1 flat and also worse on regret. No gameplay was run.
Keep the implementation as infrastructure; do not spend promotion compute on
this checkpoint.

The initial `88.3%` candidate-coverage read was computed inside the
top-Q-with-selected 64-action training tensor. It is not exact full-menu
serving recall. The permanent policy-candidate probe now rejects filtered
tensors and chunk-scores all legal actions; use that result, not 88.3%, to
judge whether upstream candidate recall is a real bottleneck.

### 4.8 Exact full-menu policy candidate recall — MEASURED; HEAD-ONLY ROUTE CLOSED

The exact probe used the untouched raw 800-root validation shard rather than a
retained-action training tensor. It scored every legal action for both models,
failed closed on incomplete surfaces, and verified the recomputed incumbent
prior against generator priors. Across 760 non-exact roots, exact top-16 sets
matched 95.92%, mean action overlap was 99.737%, and completed-Q-best coverage
agreed on every root. The few set mismatches were near-zero policy-boundary
swaps and did not alter the measured baseline mechanism.

Cycle4's real top-16 completed-Q-best coverage is 689/800 (`86.125%`), or
654/760 (`86.053%`) outside exact K1 roots. It rises to 186/206 (`90.291%`)
on roots whose completed-Q comparison clears the count, margin, and SNR gate.
Mean candidate-oracle regret is only `0.0751`; this quantifies a real but
fairly small upstream ceiling.

Two 769-parameter policy-head-only attempts did not exploit it:

- Soft improved-policy imitation lowered exact coverage to 685/800
  (`-0.500` percentage points, paired bootstrap CI `[-1.750,+0.750]`) and
  confidence-qualified coverage to 185/206. Its four-root top-1 gain was
  uncertainty-sized and did not repair the target mechanism.
- A direct confidence-gated top-16 hinge selected 222/246 trusted retained-
  menu hits versus 221/246 at initialization. The exact audit rescued just two
  full-menu sets with no losses (691/800, `+0.250` points) and one qualified
  set, while top-1 was flat and oracle regret was slightly worse. Selection
  and audit shared seed block 3120, so this is optimistic validation evidence,
  not independent replication.

No gameplay was run. Keep the exact probe and objective machinery, but do not
iterate more losses on these 2,400 roots. Reopening candidate recall requires
materially different supervision or architecture plus a new untouched root
block. Probe SHAs: soft `ac2daed8...`; direct `5b5668bb...`.

### 4.9 Action-conditioned structured value — REPRESENTATION GATE PASSED

The existing category auxiliary is root-level and therefore cannot replace
per-action Q. Before changing the data/model contract, a provenance-safe
preflight tested the actual post-CGAB selected-action latent on three disjoint
corrected-rules v3 seed blocks. Exact K1 rows were excluded, leaving 760 roots
for ridge fitting, 760 for lambda selection, and 760 untouched roots for the
held-out verdict. Raw unfiltered menus, source revision, rules identity, and
the cycle4 teacher manifest/weights all had to match.

The linear action-conditioned wildlife/habitat/Nature head reduced held-out
real-final-score RMSE to `3.4889`, versus `4.1528` for the best incumbent
comparison (selected-action completed-Q), `4.2525` for the root category sum,
`4.2438` for root value, and `4.4570` for selected model Q. Its `15.99%`
relative reduction passes the preregistered `10%` representation threshold.
The result is not a serving or strength claim: it predicts direct categories
only for the selected action. It authorizes the proper next step—export exact
per-action afterstate categories and train an action-conditioned, exactly
grounded decomposed score-to-go head while retaining total-Q supervision over
all searched actions. Probe SHA `5c06de5d...`.

The authorized production branch is now implemented, but has not yet earned a
model verdict. New Gumbel generation writes
`cascadiav3.expert_tensor_shard.v4` with `active_seat` and an exact
action-aligned three-component afterstate vector. Rust and Python readers
reject component/scalar mismatches; filtering, relation-tail materialization,
and collation preserve the fields. The optional model head predicts three
action-conditioned score-to-go residuals and defines ordinary Q as their sum,
so the existing bridge continues to rank
`exact_afterstate_score_active + predicted_score_to_go`. Only the selected
real action receives category supervision; all q-valid actions retain the
completed-Q loss on the sum.

The preregistered first experiment is deliberately cheap:
`gumbel-selfplay-structured-q`, incumbent warm start,
`q-decomposition-head-only`, and untouched v4 validation. The legacy model
and state-dict contract are unchanged while the feature is disabled. Do not
open full-model training or gameplay unless this head-only branch improves the
locked component and total-Q read without a policy/value retention failure.
The fixed held-out gate requires at least 10% selected-final RMSE improvement
over the better incumbent baseline, a paired absolute-error CI wholly below
zero, all-q completed-Q RMSE within 1.05x incumbent, and mean q-regret increase
at most 0.05. Exact-endgame rows are excluded from the primary read.

The three-way data split is ready. Corrected-rules v4 generation used disjoint
10-seed blocks `2027073500..09` (fit), `..10..19` (learning-rate selection),
and `..20..29` (one-shot verdict), producing 800 roots each. Search was
n8/top4/d1 with one determinization, eight refresh samples, blend 0.5, K8
interior, exact K1, and full root menus. Raw NPZ hashes are `06d550b4...`,
`5095d572...`, and `cdbd54b0...`; all three pass v4 and Q-identity validation
and are staged on john0. The Macs supplied data only. Training and the verdict
remain behind the approved exact-K1 john0 gate.

---

## 5. Future research directions (ranked, as of 07-09)

1. **Finish the corrected distq rebaseline** — ACTIVE. Distq n256/d4 and
   both models at n1024/d16 must establish whether the legacy distributional
   gain survives the rules compatibility break. No EI corpus or promotion
   result may cross that boundary.
2. **Exact final-personal-turn K1** — IMPLEMENTED; 2-seed causal MPS smoke
   was score-flat and made the exact frontier 8.86x faster. A fresh 100-seed
   corrected n256/d4 CUDA baseline/K1 gate is queued. K2 is gated on that
   result.
3. **Calibrate smaller-model/larger-search end to end.** The first fixed-root
   result was wrong for production because it timed raw Python feature
   extraction; live Rust sends packed features. The corrected john2–john4
   batch-8 ratios are M/S/XS/tiny `1.00x / 3.06x / 4.83x / 9.85x`, rising to
   `1.00x / 3.38x / 5.64x / 13.66x` at batch 32. Synthetic XS/tiny have no
   strength claim. Three serial MPS calibrations of M n64/d4 versus trained S
   n192/d12 found only a `~2x` equal-wall search-budget multiplier.
   The rounded S n128/d8 follow-up was near equal wall (`1.078x`) but scored
   `93.917` versus M n64/d4's `96.083` (three-game delta `-2.167`, smoke only).
   Run the same packed probe on john0 CUDA, but do not distill XS unless better
   whole-search leverage or a stronger/distq student recovers this accuracy
   loss.
4. **Distributional-Q expert iteration** — PAUSED at the rules boundary.
   The legacy quantile head broke training-side saturation (+0.43 CI+ at
   n256), but the corrected paired verdict owns the next decision. If it
   survives, resume EI with corrected-policy data; next training knobs are
   K=16 and a distq + L capacity retry. Quantile-aware serving is implemented
   but its fixed-root/n=3 screen did not justify a standalone CUDA gate.
5. **Action-conditioned structured value** — representation gate passed and
   exact-grounded v4 implementation complete. The frozen selected-action
   category head cut held-out real-outcome RMSE by
   `15.99%` versus the best incumbent comparison on 760 untouched non-exact
   roots. Generate a corrected v4 pilot corpus and run the head-first kill
   test; do not serve the direct-final ridge preflight or skip directly to a
   full-model/gameplay run.
6. **Table-native q head (cycle-7)** — staged but parked: serving-side
   table objectives measured CI− twice (noise multiplier); the
   training-side variant is theoretically distinct (labels average away
   noise). Revisit only if the distq line stalls AND the gate's
   cooperative reading is confirmed acceptable.
7. **Search-shape re-sweep under distq** — the n1024/d16 peak was
   established with the scalar head; a better value function can shift
   the optimal sims/worlds trade (maybe fewer worlds needed → cheaper).
8. **Free-refresh as a search decision — IMPLEMENTED 2026-07-09.** The
   engine and every automated policy now expose
   and value decline and accept. Gumbel searches separate roots and makes the
   same choice at interior plies; the 100-game cycle4 baseline accepted 565
   and declined 387 opportunities. This is a rules correction, not an
   experiment to score against the forced-refresh baseline; all old numbers
   are compatibility-broken. See `RULES_CONTRACT.md`.
9. **1,000-game certification** — run when a corrected-rules champion plausibly clears
   ~99+ at 100g; currently premature.
10. **Closed (do not re-propose without new evidence):** oracle/belief
   modeling, checkpoint ensembles, leaf softmix, symmetry TTA,
   chance-node expectimax, serving-side table-total, pairwise comparator
   serving, small-data/head-only policy candidate recall, capacity/data scaling
   for the scalar head. See §2/§4 for the measurements.

## 6. Historical record (campaign to date, condensed)

- **Baselines:** greedy 87.6 → no-search q-head 89.6 → rollout search
  96.97 (later found hidden-info-leaky; honest rebaseline lower) →
  Gumbel n256/d4 96.95 → n512/d8 97.845 → **n1024/d16 98.28**.
- **EI cycles:** EI-0/EI-1 (rollout teacher era, superseded) → cycles 3–4
  (Gumbel selfplay, produced champion M) → cycle-5 CI− (fleet label
  poisoning at weight 0.75, n128/d4 MPS labels; nofleet ablation
  exonerated w=1.0 labels) → cycle-6 (d8-taught) flat → fresh-M solo flat.
  **EI is saturated at M capacity: replicated 3×.**
- **Fleet (john1-4 M4 minis, MPS):** training data only, never gates.
  n256/d4 labels at fold weight 0.25 verified safe (no regression);
  currently no customer while EI is saturated.
- **Ops lessons:** batteries TF32 OFF / generation TF32 ON; fp32 serving
  batch-invariant, bf16 not; paired verdicts via `paired_delta_stats`
  (t_ci_low/t_ci_high), promotion = CI excluding zero at ≥100 games;
  john0 jobs strictly sequential.

Full chronological detail: `cascadiav3/EXPERIMENT_LOG.md`.
Resume state / decision history: `docs/v3/CAMPAIGN_STATE.md`.
