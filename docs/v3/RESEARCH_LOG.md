# Cascadia v3 Research Log — the road to 100

**Deliverable doc.** Every research direction tried in the Gumbel self-play
campaign: what it was, why we tried it, what we measured, and the verdict.
Updated continuously; §7 carries the latest consolidated verdicts (07-16); the live queue is [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md).

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

**Final verdict (2026-07-10): K1 ADOPTED for speed; K2 closed to model
inference by ruling.** The 100-seed CUDA gate ran flat (baseline `97.2650`
vs K1 `97.2350`) but one seed (`2027071427`) diverged at ply 18 from jobs12
shared-CUDA-bridge concurrency numerics, so the strict comparator failed
closed. John ruled on 07-10: keep K1, exclude that seed by declaration, and
leave K2 on model inference. The comparator gained a fail-closed
declared-exclusion mechanism (the declared seed must actually diverge
pre-K1; a clean seed is refused; the ruling text is embedded in the
artifact). On the 99 causally-valid pairs: paired delta `-0.0379`, 95% t-CI
`[-0.0859, +0.0101]`, inconclusive — score-neutral as expected. Seat 0 (the
only seat with a provably identical pre-decision state) gained exactly
`+0.0000` across all 99 games while K1 changed 332/400 final actions: the
incumbent model already selects score-optimal final actions, and exactness
substitutes equal-scoring alternatives. The exact frontier ran `28.99x`
faster (`1743.9s` -> `60.2s`; `1.035x` mean-decision, `1.034x` wall at
n256/d4). **K1 is the serving/benchmark default going forward
(`--gumbel-exact-endgame-turns 1`); exact K2 is not pursued — deeper plies
stay on model inference.** Verdict artifact:
`exact_k1_20260709_n256_d4_verdict.{json,md}` (SHA `2ef285e3...`), on john0.

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

### 4.9 Action-conditioned structured value — CLOSED (head-only kill test FAILED 07-10)

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
are checksum-queued behind the approved exact-K1 john0 gate at source
`f35b0d0b`; no gameplay can launch automatically from this pilot.

A candidate-blind target audit found no split-level component pathology. Mean
non-exact wildlife / habitat / Nature score-to-go was respectively
`31.825/11.720/1.758`, `32.847/11.543/1.680`, and
`32.562/11.808/1.771` across fit, selection, and verdict. Nature residuals can
correctly be negative when later token spending exceeds earning (`6.3%` to
`8.7%` of rows); total residuals never were. The held-out baseline is now
numerically pinned: teacher selected-final RMSE `3.5520` beats incumbent
`3.7476`, so the primary 10% threshold is `3.1968`. The incumbent retention
baselines are `1.7499` all-q RMSE and `0.7515` mean completed-Q regret, making
the corresponding ceilings `1.8374` and `0.8015`. No candidate prediction or
hyperparameter touched the verdict block during this audit.

The strictly quarantined fit expansion is complete: 50 seeds each on
john2–john4 (`2027073600..3749`, three disjoint blocks) at the same source,
teacher, rules, and n8/top4/d1 contract. It contributes 12,000 roots and
5,299,287 actions. Cross-shard audit passes against the fixed pilot with exact
seed domains and audit SHA `e1edbad3...`. Final-score means
`91.485 / 91.885 / 91.490`, total score-to-go means
`45.846 / 46.001 / 45.701`, and teacher RMSE
`3.169 / 3.375 / 3.287` show no material target drift. This is not part of the
fixed pilot and cannot affect its selection or verdict. If the head-only gate
passes, it removes fit-generation latency; if the gate fails, it remains
unused evidence rather than licensing another objective search.

The next three blocks are also fixed before candidate creation, but as
holdouts rather than fit capacity: selection seeds `2027073750..69`, verdict
seeds `2027073770..89`, and independent replication seeds
`2027073790..3809`, each 1,600 roots at the same raw-v4 contract. They are
complete on john2–john4 with passing per-host summary/invariant reports,
checksum-matched manifests, 4,800 total roots, and 2,058,733 total actions.
NPZ hashes are `48e48e74... / 99b85671... / 41b5bd60...`. This removes future
data latency without permitting candidate-dependent seed choice. The arming
path cannot fetch, admit, train, or address john0, and the three semantic roles
must remain separate. A combined harvest and audit against all six earlier
pilot/expansion shards remains mandatory before use.

Raw-v4 admission is now a permanent cross-shard check rather than a manual
manifest comparison. `audit_structured_q_shards` reopens every NPZ, binds it
to its sidecar checksum/metadata/counts, verifies selected-Q and exact
component identities, requires one source/rules/search/execution/teacher
contract, and proves that candidate plus explicitly excluded locked seed
intervals do not overlap. The real fixed pilot passes as 30 seeds, 2,400
roots, 1,113,755 actions, 9,240 q-valid actions, and 120 exact rows.
Schema v2 of that audit carries the target distributions too; expansion
admission must compare final-score, component-residual, Nature-spending, and
teacher-error distributions against the locked blocks rather than relying on
matching provenance alone.
Admission can additionally require one exact seed-domain declaration for
every primary shard, closing the gap where a valid, disjoint, contract-matched
file could still occupy the wrong semantic role. Both the fit-expansion and
reserve-holdout harvesters use this mode. The reserve harvest also treats all
six locked/fit shards as exclusions, so selection, verdict, and replication
cannot silently overlap any earlier corpus.

Operational lesson: tensor generation requires `--out` and `--manifest` as an
explicit pair. The expansion launch omitted the latter, so valid NPZs wrote
their manifests to the CLI default; validators and reserve chains correctly
failed before admission or reserve generation. Checksums and provenance
proved the generated manifests belonged to those NPZs, validation was rerun,
and the permanent reserve launcher now pins and tests the sidecar path.

**Verdict (2026-07-10): the preregistered head-only kill test FAILED and the
direction is closed.** Three frozen-trunk LR arms (100 steps, batch 8)
trained on the pinned fit block; selection chose lr3e-3. The one-shot verdict
on the 760 untouched non-exact roots scored candidate selected-final RMSE
`4.1573` versus the `3.5520` teacher baseline — `-17.04%` against the
required `+10%` — and the paired absolute-error CI `[+0.4461,+0.6143]` sits
wholly on the wrong side of zero. The two retention gates passed, and the
decomposed head is actually the better completed-Q predictor (all-q RMSE
`1.4162` vs `1.7482`, removing the incumbent's `+1.02` bias), but that is
not the promotion criterion. The ridge preflight's `+15.99%` closed-form win
on the same latent did not survive being trained as a head. Per the
preregistration: no full-model training, no gameplay; the 12,000-root fit
expansion and the reserve holdouts stay quarantined as unused evidence.
Reopening requires materially different supervision or architecture plus
fresh untouched blocks. Verdict artifacts:
`structured_q_head_pilot_20260709/heldout_verdict.{json,md}` (candidate
manifest `c8c80c56...`, verdict shard `218ff1b5...`).

### 4.10 Gumbel sigma calibration (R0.1) — CLOSED (confirm null, 07-11)

The portfolio's cheapest bet: `sigma(q) = (c_visit + max_visits) * c_scale
* norm(q)` with hardcoded Go defaults (50 / 1.0 / min-max) under measured
decision SNR ≈ 1; the Gumbel paper's own noisy-Q mitigation is a smaller
c_scale. Knobs exposed (`--gumbel-c-visit/c-scale/sigma-norm`, four
normalization schemes, bit-identical defaults) and swept 8 arms
(c_scale {0.05, 0.1, 0.25, 1.0} × {minmax, topk:8}) at n256/d4 on 25
paired seeds: all 7 candidates beat the incumbent (best cs025_tk8 +0.70)
with a clean dose-response shape — but the preregistered 100-seed confirm
on the disjoint block came back `-0.2325`, CI `[-0.5440, +0.0790]`.
**Closed.** Two durable lessons: (1) the screen's 7/7-positive pattern was
a shared-baseline artifact (one unlucky incumbent arm lifts every delta) —
future sweep screens need independent baseline replicates or ordering-only
selection; (2) at n256 the sigma stack is not the binding miscalibration —
the noise wall lives elsewhere (rollout decorrelation R0.2, unvisited-Q
bias R0.3, final-selection variance R0.4). Reopening requires n1024-scale
evidence or a changed Q-noise regime (e.g., after R0.2/R0.3 land).
Artifacts: `sigma_sweep_20260710_n256_*` + `sigma_confirm_20260710_n256_*`.

---

## 5. Future research directions (ranked, as of 07-09)

1. **Corrected distq rebaseline — COMPLETE (07-10).** distq-k8 n1024/d16
   scored `98.3850` versus scalar `98.2975`: paired `+0.0875`, 95% t-CI
   `[-0.2411,+0.4161]`, not significant — **cycle4 scalar retained as
   champion**, exactly reproducing the legacy high-budget tie under corrected
   rules. distq stays the strictly better low-budget server (97.31 at n256).
   Within-model n1024/d16 scaling is CI+ for both heads (+1.23 / +1.08).
   Distq EI resumption is therefore not score-motivated at high budget;
   revisit only if a future direction needs the quantile head's low-budget
   or league-diversity properties. Category attribution completed 07-10
   after both one-seed d20 replays validated bit-exact: the tie is flat in
   every category (wildlife `+0.145` ns, habitat `-0.050` ns, nature
   `-0.008` ns) — the heads are equivalent at n1024, not trading
   mechanisms. `rules_20260709_n1024_category_verdict.{json,md}`.
2. **Exact final-personal-turn K1 — ADOPTED (07-10); exact K2 closed by
   ruling.** The 100-seed CUDA gate verdict on 99 causally-valid pairs
   (seed `2027071427` excluded by John's declared ruling — jobs12
   concurrency divergence at ply 18) is `-0.0379`, CI `[-0.0859,+0.0101]`,
   score-neutral, with a `28.99x` exact-frontier speedup. Seat-0 delta was
   exactly zero across all 99 games: the model already picks score-optimal
   final actions. K1 (`--gumbel-exact-endgame-turns 1`) is the
   serving/benchmark default; K2 and deeper stay on model inference.
   Verdict: `exact_k1_20260709_n256_d4_verdict.{json,md}`.
3. **Calibrate smaller-model/larger-search — CLOSED on john0 CUDA (07-10).**
   The first fixed-root result was wrong for production because it timed raw
   Python feature extraction; live Rust sends packed features. The corrected
   john2–john4 batch-8 ratios are M/S/XS/tiny `1.00x / 3.06x / 4.83x / 9.85x`,
   rising to `1.00x / 3.38x / 5.64x / 13.66x` at batch 32. Three serial MPS
   calibrations of M n64/d4 versus trained S n192/d12 found only a `~2x`
   equal-wall search-budget multiplier, and the rounded S n128/d8 follow-up
   scored `93.917` versus M n64/d4's `96.083` at near-equal wall. The queued
   CUDA packed probe completed 07-10
   (`model_throughput_20260709_cuda.json`, engineering-only): on the RTX
   5090, S is only `1.89x / 1.68x` (batch 8 / 32), XS `1.98x / 2.01x`, and
   tiny `2.82x / 2.20x` — fixed per-call overheads dominate, so parameter
   reduction converts even less to throughput than on MPS. S buys at most
   ~1.9x search where >3x was already insufficient to close the accuracy
   loss. Do not distill smaller students for john0 serving; revisit only
   with an architecture that changes the per-call overhead structure.
4. **Distributional-Q expert iteration** — PAUSED at the rules boundary.
   The legacy quantile head broke training-side saturation (+0.43 CI+ at
   n256), but the corrected paired verdict owns the next decision. If it
   survives, resume EI with corrected-policy data; next training knobs are
   K=16 and a distq + L capacity retry. Quantile-aware serving is implemented
   but its fixed-root/n=3 screen did not justify a standalone CUDA gate.
5. **Action-conditioned structured value — CLOSED (07-10).** The head-only
   kill test failed its preregistered held-out gate: candidate selected-final
   RMSE `4.1573` vs teacher `3.5520` (`-17.04%` against a required `+10%`),
   paired CI wholly on the wrong side of zero. Retention gates passed and the
   decomposed head is the better completed-Q predictor (ratio `0.8101`, bias
   `+1.02` -> `+0.05`), but the ridge preflight's `+15.99%` did not survive
   training. Per preregistration: no full-model run, no gameplay; the
   12,000-root expansion and reserve holdouts stay quarantined. See §4.9.
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

## 7. Campaign week 2026-07-10 → 07-16 (consolidated verdicts)

The densest verdict week of the campaign. Live queue and decision rules
now live in [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md); this section is
the permanent record of what closed and what it taught.

### Adopted (velocity/economics — score-noninferior, ~10x cheaper experiments combined)

- **Exact-K1 endgame** (07-10), **refresh-div4** (07-12, 1.24x),
  **ghost+d32 serving default** (07-13, 0.688x wall), **puzzle-bank
  screens** (~6 min candidate ranking, 07-12), **group-sequential gates**
  (Lan-DeMets OBF, first live early stop 07-13), **CUPED** (10-25%
  interval narrowing, 07-13).

### Failed / closed — strength programs (do not re-propose without new evidence)

- **R0.1 sigma calibration** confirm-null; **R0.2 CRN paired rollouts**
  −4.4% vs −20% floor; **R0.3 q-bias at serving** structurally null;
  **R0.4 LCB selection** flat — the root-estimation class is **0-for-4**
  (07-11..12).
- **R3.2 deep own-turn planning**: starves the root (07-13).
- **R1.2 ghosts as a strength lever**: CI+ at n256-tier only; ns at
  champion tier under both reinvestments (07-13). Survives as speed
  default + cleared teacher (0.25-fold, 07-15). Pricing is
  **serving-only** — ghost generation measured ~2x SLOWER (07-15).
- **R1.3b menu widening** (root-menu 512): final look ns, RCI
  [−0.27, +0.21] (07-14). Bank screens are VOID for menu candidates
  (frozen menus). Coverage survives only via exact top-k (R3.3).
- **R1.4 Stage 1 trainer arms V1b/V2/C1/T0**: ALL effects were
  continued-training in disguise — the flagless control beat every arm;
  the control's own SWA lead then died on the bank screen (07-14..15).
- **R2.4 bridge throughput**: every lever below bar; serving is within
  ~5% of the architectural ceiling (07-13).
- **Structured-Q**: failed its preregistered pilot −17% vs +10% bar
  (07-10).
- **CascadiaFormer-L (207M vs M 88.2M)**: flat at every budget with the
  optimization confound removed (07-06; context for the week's lesson).

### Meta-lessons (measured, some twice)

1. **Locked-val loss improvements of 5-15% carry ZERO decision-level
   signal.** Only bank regret and paired gates screen training
   candidates. Measured on Stage 1 and again on the ctrl-SWA lead.
2. **Recipe fidelity is a failure class**: the champion's trainer knobs
   (CGAB_FUSED etc., 11x step speed) and cycle4's generation topology
   (24 owned sessions vs 12 shared — 3x pace) were both silently lost by
   copying the wrong reference invocation. Always replicate the recorded
   champion invocation exactly.
3. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` kills trainers**
   on this WSL2+5090 box (bridge tolerates it). See INFRASTRUCTURE.md.
4. **Rules-contract bug found by deep search** (07-16): consecutive
   four-of-a-kind wipes transiently drained the wildlife bag to zero
   (official rule returns each resolution's tokens before the next);
   replicated by unit test, fixed at rev 45fb5072. John's conservation
   argument (>=16 tokens at game end) was the diagnosis.

### Alive and funded

- **R1.4-D1 label correction**: pilot PASSED 07-15 — mega-search
  (n2048/d16) moves the label on **43.2-43.6% of repeat-stable roots**
  (bar 20%), with mean moved-root regret `0.397` in the pilot and `0.361` in
  the full run. The pilot signal persisted in the repeat-stable subset of the
  full 7,600-root run (which contains the pilot, so this is not an independent
  replication).
  The only training-side idea with a measured mechanism. Stage A attempt 3
  was terminated by the 07-16 john0/WSL reboot before any usable corpus row;
  the rules-ID repair completed, John fully authorized the pipeline, and
  attempt 4 completed no seed because 24 owned CUDA contexts thrashed. Attempt
  5 launched at 10:02 on the v2-proven 12-shared/Rayon-16 topology; john0 was
  unreachable at the 11:10 read-only status check, so current liveness is
  unknown. No partial scientific output was read.
  MuZero Reanalyze supports the general current-teacher refresh mechanism but
  does not validate Cascadia's targeted fraction, budget, or fold weight.
- The July 16 research ordering supersedes the earlier “adaptive/table/
  exactness next” queue: live Stage A → already authorized D1
  harvest/relabel/retrain/screen/gate, with human calibration in parallel,
  then bounded adversarial diagnostics, the Cascadia-Anchor semantic/A-EXACT
  feasibility shadow preflight, and the Cascadia-NX representation preflight.
  Adaptive allocation, table-native values, reliability-sigma, stratified
  worlds, and exactness expansion are deferred behind direct offline evidence.
  See the reprioritized **L1-L10** in
  [`RESEARCH_AGENDA.md`](RESEARCH_AGENDA.md).

Full chronological detail: `cascadiav3/EXPERIMENT_LOG.md`.
Resume state / decision history: `docs/v3/CAMPAIGN_STATE.md`.

## 8. External research verdict — 2026-07-16

Frozen scope: [`research_questions_7_16.md`](../../research_questions_7_16.md).
Complete primary-source synthesis, exact measurements, caveats, source ledger,
and D1 prescription:
[`research_answers_7_16.md`](../../research_answers_7_16.md). These are
literature/repository decisions, not new promotion evidence.

### 8.1 External ceiling and target

- No production Board Game Arena Cascadia population existed at the
  2026-07-16 cutoff, and no public exact-rules expert/bot dataset or solver
  benchmark was located.
- WBC is the best public human proxy but uses random wildlife cards,
  habitat-majority scoring, tournament incentives, and a different opponent
  distribution. The 2025 final's recorded `107/101/99/99` minus its published
  `7/6/2/4` majority bonuses gives **bonus-stripped recorded**
  `100/95/97/95`, mean `96.75`. This is not counterfactual no-bonus play and
  cannot calibrate the exact-rules ceiling.
- The champion's `98.2975` is a 100-game estimate replicated on a fresh
  100-game block, not its known expectation or a passed 1,000-game gate.
- **Decision:** retain `100` as the frozen internal engineering gate. Do not
  call it a human-superhuman threshold until strong humans are measured
  directly under the exact identity with a blinded variance pilot and a
  precision-derived sample size. A model-only standard-rules bridge is
  sensitivity analysis, not a substitute.

### 8.2 Imperfect-information transfer

- Suphx GRP redistributes final rank credit across Mahjong rounds; it is not
  luck correction. Suphx oracle guiding anneals privileged hands/wall features
  away, and its raw-dropout evidence is weaker than VLOG's
  public-prior/privileged-posterior value learner or AlphaStar's training-only
  privileged critic.
- A realized hidden order is exogenous in Cascadia. A one-realization oracle
  label adds realization noise; only exact-conditional hidden-world averaging
  can estimate a public action value. Evaluation control variates must preserve
  conditional expectation and cannot redefine the score target.
- **Decision:** close GRP, raw oracle dropout, pMCPA, and a “luck-corrected”
  reward. A privileged posterior critic is at most a post-D1 falsifier.

### 8.3 D1 relabeling recipe

- MuZero/Reanalyze, EfficientZero, Expert Iteration, KataGo, ReZero, and PER
  support current-teacher refresh, soft target replacement, recent/on-policy
  coverage, and a uniform floor. They do **not** publish a transferable
  hard-state fraction, search budget, repeat count, or fold weight.
- The July 15 preregistration already fixed the 15k cap, opening/mid emphasis,
  n2048/d16x2 no-ghost teacher, K=8 distq, `>=0.010` bank bar, fresh
  sequential-CUPED n256 gate, and final-n256-null close rule. The July 16
  report proposes, but does not retroactively preregister, `6k/6k/3k` phase
  quotas, gap/SE deciles, deterministic per-game caps/top-up, exact repeat
  aggregation, per-head replacement masks, raw fold weights `[4,2,1,1]`,
  optional 5k/10k descriptives, exact gate looks, and a second-cycle standing
  rule. Freeze each amendment before its relevant data boundary.
- Repeat aggregation must average the two improved policies, pool Q only over
  visited estimates, average root values, and use deterministic action-ID tie
  breaking. For population variances, omit invalid repeats and use
  `Q=sum(n_i Q_i)/N` and
  `var=sum(n_i*(v_i+(Q_i-Q)^2))/N`, `N=sum(n_i)`. Summing visits and rerunning
  the Gumbel transform would spuriously sharpen the target.
- A versioned schema/loader/trainer change must mask policy and every duplicated
  behavior-outcome loss (value, score, rank, and outcome-derived Q) on D1
  duplicates. The existing global exposure guard is not source-weight aware;
  add a fail-closed actual-draw audit. `[4,2,1,1]` targets a 12.5% D1 share,
  60k expected draws, and four expected passes over 15k roots.
- **Decision:** D1 stays first because the n2048/d16 hard-root label movement
  persisted from the pilot (`43.2%`) through the full ledger (`43.6%`). The complete 15k masked
  fold, student improvement, and gameplay effect remain untested. Reanalysis
  becomes a standing pipeline only after a positive paired game gate and one
  fresh-cycle replication.

### 8.4 Search, targets, robustness, and serving

1. **Best-arm allocation:** retain sequential halving. If D1 produces a model
   that materially changes the Q regime, defer until then a frozen-root,
   measured-wall-matched comparison with successive rejects and one
   variance-adaptive halving method. Preserve CRN; freeze the high-budget
   action reference/ties independently.
2. **Gumbel constants:** retain `c_visit=50`, `c_scale=1.0`, and min-max. The
   selected `c_scale=.25` plus top-k:8 candidate failed fresh confirmation,
   closing that preregistered static sweep rather than every lower scale. Any
   future L3 needs a disjoint low/high-budget reliability curve and must apply
   reliability after normalization (`c_scale_eff=rho*c_scale`), because
   positive affine shrinkage before min-max cancels. Published 2048/EWN work
   found that fewer *training simulations* could outperform more in stochastic
   variants, but did not isolate `c_visit`/`c_scale` or provide a
   variance-to-scale law.
3. **Multiplayer targets:** Cascadia already has a four-seat state value head.
   R1.1c's genuine intervention is table-native **per-action Q and
   table-derived improved-policy labels**, with table utility at root and
   interior. This changes selfish play into cooperative table planning and
   remains reserved to John's methodology ruling.
4. **Determinizations:** basic root-world CRN is already implemented. R0.2
   paired only the remaining rollout RNG and increased the measured gap
   variance by 4.4% (`0.020538 -> 0.021438`) against the required 20%
   reduction. Close “add pairing”; defer any formally unbiased stratified
   world sampler.
5. **Symmetry:** close the tested three-rotation inference TTA configuration.
   Do not prioritize a standalone trainer-augmentation arm; random D6 training
   remains unmeasured and is eligible only as a cost-neutral shadow arm after
   reflection transforms/tests exist.
6. **Adversarial probes:** after D1 and the rules-ID repair, run only a bounded
   diverse diagnostic bank with high-budget confirmation, cross-checkpoint
   transfer, held-out generator families, and natural-frequency accounting.
   Do not start full adversarial training from the KataGo analogy alone.
7. **Distributional serving:** retain K=8 and serve the existing arithmetic
   mean projection plus exact afterstate score. The heads receive scalar
   search-Q labels and are not calibrated final-return quantiles; their spread
   is neither a validated risk statistic nor a model-error estimate. Close the
   tested q25/q50/q75 modes and do not queue untested CVaR under a mean-score
   objective.

### 8.5 Ordering and blocker

**Superseding operational update (11:10):** the July-16 rules identity repair
completed at 03:50, and John fully authorized the Stage A restart plus the 15k
D1 relabel/retrain/screen/gate chain. The full harvest, sentinel, teacher,
repeat aggregation, masking, training mix/control/dose arms, bank screen, and
fresh sequential-CUPED gate were frozen in `cascadiav3/EXPERIMENT_LOG.md`
before Stage A output was read. Attempt 4 completed no seed because 24 owned
CUDA contexts thrashed. Attempt 5 launched at 10:02 on the v2-proven
12-shared/Rayon-16 topology; the 11:10 read-only status check could not reach
john0, so current liveness is unknown. Champion promotion remains separately
reserved. The queue is now: let the authorized D1 chain reach its registered
boundary untouched; exact-rules human calibration may proceed in parallel;
then bounded adversarial diagnostics, the Cascadia-Anchor
semantic/A-EXACT-feasibility shadow preflight, and the bounded Cascadia-NX offline
preflight before lower-ranked allocator/table/reliability/world work.

No source in the review authorizes a promotion, rules-design change, or live
experiment by itself.

## 9. Structured stochastic-game architecture review — 2026-07-16

Complete literature synthesis, archived NNUE audit, architecture specification,
falsifiers, and primary-source ledger:
[`stochastic_board_game_ai_architecture_research_7_16.md`](../../stochastic_board_game_ai_architecture_research_7_16.md).
This section records the consolidated verdict; it is not Cascadia strength
evidence.

### 9.1 Strongest cross-game signal

- The closest located game-shape result is a 2025 Azul MSc thesis: a shallow
  NNUE/search agent beat the strongest handcrafted heuristic in `94.07%` of
  `10,218` automated games, and longer search consistently strengthened the
  same evaluator. It is two-player and not peer reviewed.
- In stochastic 2048, the strongest peer-reviewed learning-based result located
  in the review—a symmetry-shared n-tuple afterstate learner plus six-ply
  expectimax—reported `625,377` average and `72%` reaching 32768 over 100
  deep-search games. This does not claim superiority over hand/tablebase systems
  under other protocols. A Stochastic MuZero result in the same research line
  is about `510,000`; the comparison is cross-study, not controlled.
- TD-Gammon, Stockfish NNUE, DouZero, KataGo, and Pgx collectively support
  different components of the synthesis: afterstates/legal actions, cheap
  structured evaluation, search or Monte Carlo as teacher, global auxiliary
  context, and accelerator-native simulation.
- Exact rules make learned MuZero/Dreamer dynamics a poor first Cascadia bet.
  Retain the decision-afterstate-chance factorization and execute the real
  simulator.

### 9.2 Ranked architecture hypothesis

The leading system challenger identified by the July 16 review is
**Cascadia-NX**:

1. a versioned exact compiler for scoring-card-conditioned local motifs,
   habitat components, market/bag/phase summaries, and legal compound-action
   deltas;
2. D6-symmetry-tied sparse factor embeddings with NNUE-style accumulators;
3. a small global component/action graph used as a residual, optionally only
   for search survivors;
4. a scalar own-score-to-go action head served strictly as exact afterstate
   score plus predicted remaining score;
5. category, four-seat, distribution, and motif heads as auxiliary
   regularizers only; and
6. GPU-resident exact-rules `max^n` search with sampled conditional chance and
   a frozen world-sampling contract.

NX explicitly inherits v3's exact legal menu/action queries, afterstate-score
grounding, component relations/D6 identities, shared state encoding across
action chunks, scalar own-Q, `max^n`, Gumbel sequential halving, exact K1, and
root-world CRN. The claimed changes are dependency-complete sparse deltas,
hard symmetry sharing, a semantic component/action residual, jointly
calibrated fast/full paths, and accelerator-resident rules/search.

The report names the systems extension **Covariance-Audited GPU World Search**.
The incumbent already redeterminizes full hidden tile-stack/wildlife-bag
orders and cycles the same `d` root worlds across actions; that root-world CRN
is the control, not a new method. R0.2's additional rollout-policy RNG sharing
worsened gap variance by 4.4%. Marginal correctness must follow from sampler
construction and CPU/GPU parity, with mean agreement only an audit. Any
coupling/control-variate eligibility must be frozen from a disjoint covariance
block, with independent worlds used everywhere else. The calibration artifact
is keyed to rules, sampler, checkpoint, opponent/rollout policies,
depth/budget, candidate set, and allocator; any identity change invalidates it.
Positive covariance alone is insufficient: fixed-wall action-selection error
and pairwise-difference variance must both improve on untouched roots.

### 9.3 Why this does not reopen closed work

- The archived 11,231-feature, `512 -> 64 -> 1`, 5.78M-parameter NNUE had
  unqualified direct observations around `90.7` and qualified K32/R600-class
  results around `95.8–96.35` under old rules. Those are an observed strength
  class, not a universal architecture ceiling. A three-game rollout-scaling
  pilot was flat; deeper wildlife lookahead double-counted future value and
  hurt in archived observations. Its attempted accumulator was 2.5x faster but
  regressed about three points because feature dependencies were incomplete.
- Cascadia-NX is eligible only as a coupled system: correct full dependency
  tracking, current v3 reanalysis/TD targets, explicit global semantics and
  compound-action Q, and a GPU-native engine/search boundary. “Bring back
  NNUE,” a pure local n-tuple model, or an ordinary smaller transformer remains
  closed.
- The bakeoff includes the closest reproducible archived
  `11,231 -> 512 -> 64 -> 1` NNUE retrained on the same current targets. NX must
  beat that control on the registered quality/throughput frontier; novelty is
  a testable claim, not a diagram-level assertion.
- Before training, measure factor-table cardinality/memory, median/p95/max
  dependency invalidation, and full-menu delta latency versus full recompute;
  board-wide dependency closure kills the incremental thesis. Compare a
  full-menu graph residual against survivor-only routing with jointly
  calibrated heads and an explicit rescued-action miss rate.
- Terminal behavior returns, moving TD bootstraps, and search-improved Q are not
  one estimand. Shards carry behavior/opponent/teacher/sampler/search hashes;
  the first bakeoff freezes a target policy and keeps behavior returns as a
  separately weighted anchor or diagnostic rather than silently blending them.
- Archived geometry-only GNN ADR 0073 worsened value correlation and MAE. The
  proposed graph is a small residual over exact semantic components, not a
  geometry-only trunk.
- The failed structured-Q sum remains closed. Category heads are auxiliary.
  Pairwise-Borda, table-total serving, risk serving, symmetry TTA, blind CRN,
  generic menu widening, and chance-node leaf expectimax remain closed.

### 9.4 Decision and ordering

**Status: OPEN HYPOTHESIS; ZERO CASCADIA STRENGTH EVIDENCE.** Do not interrupt
or reorder the fully authorized D1 chain. After D1 reaches its frozen boundary,
a bounded current-rules offline bakeoff may compare the transformer with local
factor, global-summary, component-graph, and two-fidelity arms on identical
states and targets. Only several-fold end-to-end throughput with retained or
better high-budget teacher regret justifies the GPU planner. Only a fresh
paired gameplay gate can establish strength, and the final objective remains
mean seat score at least 100 over 1,000 games under one pinned identity.

## 10. Incumbent-anchored GPU rollout policy improvement — 2026-07-16

Complete proposal, architecture, inference contract, predecessor audit,
falsifiers, and primary-source ledger:
[`incumbent_anchored_gpu_rollout_policy_improvement_7_16.md`](../../incumbent_anchored_gpu_rollout_policy_improvement_7_16.md).
This section records the consolidated verdict; it is not current-rules
strength evidence.

### 10.1 Corrected proposal identity

The leading bounded serving challenger is **Cascadia-Anchor**:

1. let the exact frozen incumbent own the free-three accept/decline decision,
   commit it, reveal any replacement, and begin only at the resulting
   post-prelude public draft node;
2. preserve the incumbent draft as mandatory anchor and build a small frozen
   challenger set for that fixed visible market;
3. use cheap direct-transformer or distilled continuations only to screen;
4. compare one challenger with the anchor on fresh terminal worlds whose
   continuation is the complete serving incumbent;
5. override only when a multiplicity-valid lower bound on paired terminal
   own-score advantage exceeds a preregistered practical margin; and
6. fall back literally to the stored incumbent action on uncertainty, timeout,
   incomplete evidence, or any provenance/parity failure.

Three policies are versioned separately: `pi_I` is the complete serving
incumbent, `pi_R` is the rollout continuation, and `pi_W` is the wrapper. The
policy-improvement interpretation against the current system requires
`pi_R = pi_I`. A greedy, direct-transformer, or distilled continuation changes
the estimand and cannot certify improvement over transformer-plus-Gumbel.

A-EXACT confirmation has two state/RNG layers: an outer physical chance world and
the incumbent’s inner no-peek search determinizations at every future public
state. Letting the policy see the outer hidden order is oracle leakage.
Freezing an accept-branch draft before the replacement is public is also
invalid; v1 does not override the current turn’s refresh decision.

### 10.2 Direct Cascadia predecessor

The anchor/fallback mechanism is not novel. Current source retains
`LateConservativeBasePolicyImprovementStrategy` in
`crates/cascadia-search/src/policy_improvement.rs`. Under the historical v2
pattern-aware policy and old rules it:

- activated in the final five personal turns;
- evaluated the K8+H6+B8 frontier over eight shared canonical public worlds;
- completed each candidate under the frozen pattern policy;
- used acting-seat exact terminal base score;
- admitted only challengers with a positive one-sided paired t-LCB; and
- otherwise played the exact pattern anchor.

Archived ADR 0024 confirmed **+0.420**, 95% CI
**[+0.179,+0.661]**, over 50 games. ADR 0068’s canonical-redetermination
requalification retained **+0.520**, 95% CI
**[+0.260,+0.780]**, over 50 fresh games. The latter was demoted because the
frozen non-Bear wildlife guardrail was -0.375, not because total-score signal
failed. These are supportive mechanism results under a different policy and
rules identity, not evidence for v3 or July-16 rules.

The successor fixes the predecessor’s per-decision inference weakness: the old
rule selected among multiple challengers using the same eight per-challenger
90% bounds. Anchor uses a frozen selection stream and fresh confirmation
stream, a positive practical margin, familywise control, and either fixed
bounded-mean inference or a correctly implemented anytime/group-sequential
rule.

### 10.3 Architecture and central falsifier

The GPU batch is nested:

`roots x candidates x outer worlds`, then at each future decision
`live states x legal actions x incumbent internal worlds/simulations`.

Exact rules, legal masks, without-replacement chance, transitions, scoring,
terminal detection, and compact state remain device-resident; the Rust engine
is the bit-exact oracle. Wavefront queues compact states needing policy,
chance, or terminal work. Forced-anchor mode must be action/RNG/score
bit-identical to the incumbent. A-EXACT additionally requires complete nested
policy-trace parity with the production bridge—packed rows, numerical mode,
Gumbel/rollout traces, market branches, RNG consumption, actions, and scores.
Rules parity alone is insufficient; any action divergence defines a proxy.

The central risk is compute. Full incumbent continuation runs the expensive
serving search at every future simulated decision. No paper or repo result
shows this is affordable. Before the port, a root-specific score-range,
paired-variance, alpha/margin, and interval-family power calculation must also
show that a useful effect is statistically resolvable. Start in final-two
through final-five personal-turn strata; exclude final-personal-turn roots
from Anchor and leave the incumbent’s exact-K1 path unchanged. Treat completed
A-EXACT terminal pairs per wall-second as a kill test. If only a cheap proxy
is feasible, the wrapper loses its incumbent-improvement interpretation and
remains an offline/shadow arm.

### 10.4 Evidence and caveats

- Tesauro and Galperin’s 1996 backgammon work directly supports terminal
  action evaluation followed by a frozen base controller and reports large
  reductions in base-policy decision error; it does not prove finite-sample
  Cascadia safety.
- HPCI and SPIBB support held-out evaluation and literal baseline fallback
  under their own assumptions; their theorems do not transfer to this
  four-player online wrapper.
- Pgx and Mctx establish accelerator-native simulation/planning precedent, not
  a Cascadia speedup.
- Common random numbers help only through positive covariance. Current
  root-world coupling is the control; R0.2’s added rollout-policy coupling
  worsened gap variance by 4.4%. Any new depth-dependent coupling requires
  exact marginal proof plus disjoint variance/selection-error calibration,
  with independent fallback.
- A local one-deviation result followed by `pi_I` does not prove that all four
  seats repeatedly using `pi_W` improve symmetric self-play. Require a
  one-seat diagnostic and a fresh four-seat paired gameplay gate.

### 10.5 Decision and ordering

**Status: OPEN POST-D1 HYPOTHESIS; ZERO CURRENT-RULES STRENGTH EVIDENCE.**
Anchor is the lowest-downside bounded serving test and the preferred first
preflight; its probability of finding a positive gain is unknown.
Cascadia-NX retains higher clean-slate upside if evaluator economics are the
true ceiling. Neither reorders the authorized D1 chain.

After D1 reaches its frozen boundary, the bounded order is:

1. exact CPU/GPU rules and complete incumbent-policy trace parity, including
   post-prelude market-boundary golden traces;
2. bounded-inference power and A-EXACT final-two-to-final-five feasibility;
3. proxy-screen fidelity and fresh-confirmation shadow audit;
4. equal-wall comparison with more ordinary Gumbel compute;
5. unilateral diagnostic; and
6. only then the preregistered symmetric paired game gate.

The full policy promotion rule and 1,000-game ≥100 target remain unchanged.
R0.2 generic CRN, R0.4 completed-Q LCB, R3.2 ordinary deeper Gumbel, exact
K2, cooperative/risk serving, generic menu widening, and legacy NNUE revival
remain closed.
