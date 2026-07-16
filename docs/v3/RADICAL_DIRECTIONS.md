# Radical Directions — speculative architecture bets beyond the current stack

Written 2026-07-08, end of the day the scalar-head plateau broke. These are
NOT the incremental next steps (those are RESEARCH_LOG §5) — these are
architecture-level departures, each judged against the campaign's two
hardest-won facts: **(1) eval noise is the binding constraint** (median
decision SNR ≈ 1, 46% of argmaxes are coin flips; oracle info doesn't help;
input-frame perturbation doesn't help; problem-perturbation ensembling does)
and **(2) exactness beats estimation wherever it can reach** (exact
afterstate grounding is why the q head works at all).

Ranked by my judgment of payoff × plausibility.

## 1. Exact endgame solving (hybrid neural/exact search) — K1 implemented

**Idea.** On a player's final personal turn, every legal afterstate already
contains that player's final score. Enumerate the complete menu and choose the
engine-scored maximum instead of adding model score-to-go or search noise.
Neural search remains responsible for the mid-game; exactness takes over at
the first frontier where the value identity is rigorous.

**Why it fits the evidence.** It removes eval noise entirely from the
plies where SNR matters most — endgame placements are worth whole points
and the noise there is the same size as mid-game. Every point of the
campaign's +10 came from replacing estimates with exactness or averaging;
this is the terminal version of that move.

**Implemented K1 (2026-07-09).** `--gumbel-exact-endgame-turns 1` bypasses
the evaluator and Gumbel simulations when the active seat has one personal
turn left, ignores the ordinary root-menu cap, and chooses the maximum exact
afterstate score with deterministic tie-breaking. The optional refresh keeps
the correct information boundary: compare decline with exact accepted-market
optima averaged over public-derived hidden samples, commit, then solve the real
revealed market. Telemetry marks exact decisions and zero simulations. The
mode rejects table-total objectives because the table's score is not final
when an individual seat finishes.

**Correction to the original sketch.** K=2 is not “free.” It crosses opponent
turns and future public draws, so a rigorous K2 needs a real max^n/chance tree
with common-random-number determinizations (and a precise table-vs-own-score
objective), not repeated one-ply exact ranking. The trace-validated 2-seed
n16/d2 MPS smoke was score-flat, changed 6/8 final actions, and made the exact
frontier 8.86x faster (only 1.2% over whole-game mean decision time). A fresh
same-revision 100-game corrected n256/d4 CUDA gate is next. Only a CI-positive
or material gate-scale cost win justifies building K2.

## 2. Invert the AlphaZero ratio: smaller model, larger search — production preflight reopens the frontier

**Idea.** Distill M into a 3–10M student and spend measured serving savings
on simulations. The original sketch assumed the same wall-clock might buy
n8k–n16k with d64+ worlds; the production-shaped preflight supports a real
multiplier, though not that whole range for a credible 5M model.

**Why it fits.** The SNR analysis says decisions flip on *sampling* noise;
simulations divide sampling noise by √n regardless of model bias. Nobody
has measured where the (model bias) × (search averaging) iso-wall-clock
frontier actually peaks — the campaign only ever moved along the
fixed-model axis. AlphaZero-family results repeatedly show small-net/big-
search dominating at fixed compute in exact-scoring games.

**Measured preflight (2026-07-09, corrected).** The first pass mistakenly fed
raw audit roots to Python and measured feature extraction that live search
already performs in Rust. It reported only `2.40x` tiny/M and is superseded.
The corrected tool pre-packs outside the timed loop to the production
`packed_features` wire shape and records both source and prepared hashes.
Across john2–john4 at batch 8, trained 88.17M M delivered `144.996 roots/s`;
trained 15.02M S `443.174` (`3.06x`); synthetic 5.12M XS `700.524`
(`4.83x`); and synthetic 67.8K tiny `1,427.867` (`9.85x`). At batch 32 the
ratios rose to `3.38x / 5.64x / 13.66x`. Response digests matched on all three
hosts. Synthetic shapes say nothing about strength, but the serving headroom
is real enough to test.

**Revised sketch.** First calibrate the trained S checkpoint end-to-end at
roughly equal model-eval compute. Three one-game MPS pairs showed that S
n192/d12 was actually `1.477x` slower than M n64/d4; changed trajectories
consumed `3.356x` simulations, and non-model Rust/game work caps the isolated
bridge gain. The implied equal-wall point is S n130, so n128/d8 is the rounded
follow-up. Then run the identical hashed probe on john0 CUDA. If
the equal-wall S arm preserves score, distill policy+q+quantiles into the
first-class XS config and repeat the empirical frontier. It did not in the
first screen: S n128/d8 was close to equal wall (`1.078x`) but averaged
`93.917` versus M n64/d4's `96.083` (delta `-2.167` over only three games).
That is insufficient for a strength claim but negative enough not to spend a
training day yet. Hold for the CUDA multiplier and corrected-distq verdict.
Budgets must come from measured whole-search throughput, not parameter counts.

## 3. Pairwise comparator head — train the decision, not the value

**Idea.** The argmax doesn't need calibrated values; it needs correct
*comparisons*. Add a head trained on pairs: P(action i ends better than
action j | state), supervised from search completed-Q pairs and real
outcomes. Serve by round-robin voting among the top-m actions.

**Why it fits.** 46% of decisions are within-noise ties of the top-2 —
exactly the regime where ranking losses beat regression losses (the whole
learning-to-rank literature exists because of this). The scalar/quantile q
head spends capacity on absolute calibration the argmax never uses.

**Implemented pilot (2026-07-09).** CGAB's action embeddings now feed a
strictly antisymmetric low-rank head: merit difference plus a skew bilinear
interaction. The trainer emits both orientations, requires q-valid actions,
at least two samples/action, margin ≥0.25 and SNR ≥1.0, caps each root at 32
hard reliable pairs, and supports head-only warm-start training. The bridge
serves a permutation-equivariant all-opponent Borda logit behind an explicit
provenance-recorded policy mode; Q/value serving is unchanged.

The data premise needed correction. In a 240-root corrected-rules audit,
27,360 pairs existed (11.4M projected/100k roots), but only 23.33% had enough
samples to estimate variance and only 14.58% of those reached SNR 1.96. Pair
volume is free; trustworthy labels are not. The inspected v2 shards also lost
exact-endgame and generation provenance, so they are audit-only. Fresh v3
corrected-rules generation is required before the kill test. After an offline
held-out accuracy gate, compare established logits, pure pairwise Borda, and
their sum on identical n256/d4 seeds; only gameplay can establish value.

**Result: CLOSED before gameplay.** A larger 2,400-root v3 corpus supplied a
fixed 1,600/800 train/validation split. The head learned its pair labels
(selected held-out accuracy `60.4% -> 66.0%`; full confidence-weighted probe
`69.5%`), but the serving-aligned top-16 gate did not improve decisions. On
206 qualified held-out roots, Borda gained only two net top-1 hits
(`30.58% -> 31.55%`, paired bootstrap CI for the delta
`[-3.40,+5.34]` percentage points) and worsened completed-Q regret
`1.1496 -> 1.2121`. Logits plus Borda was top-1 flat and also worse on regret.
No john0 gameplay was launched. Keep the implementation, but do not re-open
this serving direction without a materially different data/decision model.
The initially reported `88.3%` candidate recall was inside a top-Q-filtered
64-action tensor, not the full legal menu, and is not serving evidence. The
new full-menu policy probe fails closed on such filtered inputs.

**Upstream recall follow-up: CLOSED for head-only tuning.** The exact probe
puts incumbent top-16 completed-Q-best coverage at `86.125%` over 800 roots
and `90.291%` over 206 confidence-qualified roots. A 769-parameter soft-policy
fit lost four covered menus. A direct confidence-gated recall hinge rescued
only two menus and one qualified root, left top-1 flat, and did not improve
candidate-oracle regret; moreover, checkpoint selection and audit used the
same validation block. No gameplay was justified. Keep exact full-menu
measurement infrastructure, but do not substitute more loss tuning on this
small corpus for the materially different data/decision model required to
reopen this direction.

## 4. Whole-rollout generation on GPU (kill the lockstep wall)

**Idea.** The serving/generation bottleneck is per-ply lockstep: every
simulated ply is a bridge round-trip. Replace CPU greedy rollouts with a
*learned autoregressive rollout head* that generates entire game
continuations on-GPU in one batched pass per rollout (or train a policy
that runs the whole rollout as a single sequence generation).

**Why it fits.** Generation knobs are measured dead — the wall is
latency, not compute; the GPU idles between plies. Moving the inner loop
onto the GPU is worth ~10× simulations at fixed wall-clock, which per the
worlds/sims scaling laws is worth real points, and it makes EI cycles
~10× cheaper — compounding every future direction.

**Sketch.** Two flavors: (a) cheap — batch rollout *policies* on GPU while
the exact engine applies moves (the Rust engine is fast; the round-trip is
the cost, so amortize by running 64 rollouts per request); (b) radical —
a JAX/CUDA port of the scoring-relevant game subset so rollouts never
leave the device. (a) is a week; (b) is a month and only pays if (a)
saturates.

**CPU-parallel precursor (2026-07-09): bounded, not the solution.** The
existing blended terminal rollouts are now independently executable on the
Rayon pool with exact RNG/order provenance. A same-host jobs1 MPS screen was
bit-identical over 160 decisions and improved wall by `1.061x`, but jobs2 was
`0.993x` (slightly slower) despite action/score parity. Host parallelism is
already consumed by concurrent games; nesting more CPU work does not repair
the production lockstep wall. Keep the opt-in mode for interactive latency,
leave all batch paths serial, and keep the GPU-native proposal open: its value
is eliminating device/engine synchronization, not merely spreading terminal
rollouts across host cores.

## 5. Structured value: per-scoring-card decomposition heads

**Idea.** Replace the monolithic score-to-go with per-card heads: elk
score-to-go, salmon score-to-go, per-terrain corridor-growth-to-go —
each grounded in its exact current partial score, summed for the total.

**Why it fits.** The campaign's most reliable trick is
decompose-and-ground (exact afterstate + learned residual). Each card's
remaining potential is a *simpler function* than their sum — simpler
functions ⇒ lower per-head error, and errors across heads decorrelate
where a single head's error doesn't (this is ensemble variance reduction
*inside* the architecture, the only ensemble mechanism that ever worked
for us). The score_decomposition head already exists as an aux — promote
it to load-bearing.

**Sketch.** Trainer: use the selected real trajectory for category residual
targets, retain completed-Q supervision on the head sum for all searched
actions, and never copy a selected category outcome onto counterfactual
actions. Serving sums the heads instead of reading monolithic q. Kill-test:
head-only locked validation first; only then a cycle-style decomposed-head
versus incumbent comparison.

**Implementation audit (2026-07-09).** The existing `score_decomposition`
auxiliary is root-level (`score_head(root_h)`), not action-conditioned. It
cannot honestly replace per-action Q at serving. This direction therefore
requires a new action-conditioned decomposition head plus retraining; it is
not a free serving ablation of the current checkpoint.

**Action-representation preflight: PASSED (2026-07-09).** A fail-closed probe
used three disjoint corrected-rules v3 seed blocks: 760 non-exact roots to fit
a ridge head on the frozen selected-action latent, 760 to choose regularization,
and 760 untouched roots for the verdict. Every tensor retained its exact full
menu and matched the probed cycle4 teacher manifest/weights. On held-out real
terminal outcomes, the action-conditioned wildlife/habitat/Nature sum reached
`3.4889` RMSE and `2.6964` MAE. The best incumbent comparison was selected-
action completed-Q at `4.1528` RMSE; root decomposition/value and model Q were
`4.2525 / 4.2438 / 4.4570`. The `15.99%` RMSE reduction clears the
preregistered `10%` representation gate.

This is a go for the real schema, not for serving the ridge head. The probe
predicts direct final categories only for the chosen action; it has neither
per-action exact category grounding nor counterfactual category labels. The
next implementation must export each action's exact afterstate category vector,
train action-conditioned category score-to-go heads whose sum stays grounded,
and retain scalar/distq supervision for all searched actions. Probe JSON SHA:
`5c06de5da762352765a26c233b8718af7e69bc9040d698ad0758c2b72e908c2a`.

**Historical pre-verdict state — exact-grounded implementation COMPLETE.** New
Gumbel exports are schema v4 with active seat and exact per-action
wildlife/habitat/Nature afterstate components. The optional component head
defines ordinary score-to-go as its exact sum, supports scalar or quantile
training, and is reloadable by the unchanged bridge. The v4-only structured
objective supervises categories on the selected action and completed Q on all
q-valid sums. `q-decomposition-head-only` freezes every incumbent parameter
for the preregistered cheap gate. This closes the architecture/plumbing task;
it does not yet establish that the trained head improves validation or play.

**Training verdict: CLOSED (2026-07-10).** The preregistered head-only gate
failed: selected-final RMSE was `4.1573` versus teacher `3.5520`, a `-17.04%`
change against the required `+10%`. Retention diagnostics passed, but the
ridge preflight did not survive the actual training contract. No full-model
run or gameplay followed. Category heads may remain auxiliary regularizers in
a materially different trunk; the load-bearing additive serving design is
closed.

## 6. League self-play (break the self-play attractor)

**Idea.** Fresh-M from scratch converged to the *same point* as champion
lineage — self-play has one attractor and EI orbits it. Train against a
league: past checkpoints, quantile-risk-shifted variants (serve q10/q90
instead of the mean — free with the distq head), and rule-restricted
agents. Label diversity where checkpoint ensembles failed at serving.

**Why it fits (loosely).** The saturation evidence says more of the same
distribution teaches nothing; the poisoning evidence says *degraded*
diversity hurts. Untested middle: *strong-but-different* opponents. The
distq quantile head makes personality variants free, which is new.

**Sketch.** Generation seats draw manifests/serving-configs from a pool;
one EI cycle; standard ablation battery. Cheap to try once EI-1's verdict
says whether the flywheel is even turning again.

**Risk-personality preflight (2026-07-09).** q25/q50/q75 serving is now
implemented with explicit provenance and coherent quantile interpolation.
The modes do change trajectories, but q25's three-seed n64/d4 result was only
`+0.25` (`95.25` vs `95.00`, CI spanning roughly -4.2 to +4.7); q50 was flat
and q75 negative in their one-seed screens. That kills risk shifting as a
standalone strength gate, not as a league-diversity mechanism. League work
remains gated on the corrected-rules distq/EI verdict.

## 7. Cascadia-NX: structured afterstate factors + paired GPU exact search

Full clean-sheet review and source ledger:
[`stochastic_board_game_ai_architecture_research_7_16.md`](../../stochastic_board_game_ai_architecture_research_7_16.md).

**Idea.** Replace repeated full-state transformer inference with a correctly
incremental, D6-symmetry-tied factor evaluator over scoring-card motifs,
habitat components, market/bag state, and explicit legal compound-action
deltas. Add a small semantic component/action graph only as a global residual.
Run the cheap evaluator inside a GPU-resident exact decision/afterstate/chance
planner. At the root, compare candidates over the same complete physical
tile/wildlife future tapes and keep pairing only where measured covariance is
positive.

**Why it fits.** The best located stochastic score-game results—2048 n-tuple
afterstate TD + expectimax, Azul NNUE + search, TD-Gammon, and DouZero—favor
cheap structured afterstate/action evaluation. Pgx and variance-reduced MCTS
show that accelerator-native exact simulation and correctly coupled worlds can
change the search frontier. This attacks both measured campaign constraints:
current per-call economics and variance of action differences.

**Why it is materially new.** The archived 5.78M sparse Cascadia NNUE was not
correctly incremental, lacked the proposed global/action semantics and current
v3 reanalysis targets, and plateaued around 90.7 direct / 95.8–95.9 searched
under old rules. Ordinary small-model/larger-search, geometry-only GNN,
load-bearing structured-Q, generic CRN, pairwise Borda, and chance-node leaf
expectimax remain closed. Category/four-seat heads are auxiliary only; serving
stays exact afterstate score plus scalar expected own score-to-go.

**Falsifier.** After the authorized D1 chain reaches its frozen boundary, run
only a bounded current-rules offline bakeoff. Require exact feature/delta/D6
parity, several-fold end-to-end throughput with retained or improved
high-budget teacher regret, exact paired-world marginals, positive covariance,
and material action-difference variance reduction. Only then build the full
GPU planner. Only a fresh paired gameplay gate establishes strength.

**Status:** open post-D1 hypothesis; zero Cascadia strength evidence. It does
not displace or reorder D1.

## Explicitly not on this list

Belief modeling / bag inference (oracle LOST — information is not the
constraint), checkpoint output-ensembles (shared-bias, measured 4-ways),
input-symmetry tricks (representation is already invariant), serving-side
cooperative objectives (noise multiplier, measured twice), bigger
monolithic models on the same labels (measured three ways), learned dynamics
for rules the exact engine already owns, ordinary legacy-NNUE revival, and
blind common-random-number pairing without a positive covariance audit.

*Current sequencing: do not disturb the fully authorized D1 chain. After D1
reaches its frozen boundary, #7 may begin with the bounded offline
representation/covariance bakeoff. The full GPU port is conditional on that
preflight. #4 remains a structural systems enabler; #5 is closed; #6 remains a
later diversity mechanism rather than a serving ensemble.*
