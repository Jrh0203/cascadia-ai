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

## 1. Exact endgame solving (hybrid neural/exact search)

**Idea.** The last 2–3 plies per seat have small menus and zero remaining
hidden-draw depth that matters; solve them *exactly* (max^n enumeration over
determinized bags, or full expectimax over the residual bag) instead of
estimating with the model. Neural search mid-game, tablebase-style exactness
where points actually crystallize.

**Why it fits the evidence.** It removes eval noise entirely from the
plies where SNR matters most — endgame placements are worth whole points
and the noise there is the same size as mid-game. Every point of the
campaign's +10 came from replacing estimates with exactness or averaging;
this is the terminal version of that move.

**Sketch.** In `gumbel.rs`, when `turns_remaining(root_seat) <= K`, bypass
Gumbel: enumerate the full action tree per determinized world, score
terminal states exactly, average over worlds. K=2 is likely free
(menus ~30² per seat); K=3 needs pruning. Kill-test: 100g at n256/d4 +
exact-K2 vs baseline — a week of evenings, all serving-side.

## 2. Invert the AlphaZero ratio: tiny model, enormous search

**Idea.** Distill M (100M params, ~17 ms/eval) into a 3–10M student and
spend the savings on simulations: same wall-clock buys n8k–n16k with d64+
worlds. We measured +0.9/doubling of worlds *before* the reversal and the
reversal may have been budget-shape, not worlds, at fault.

**Why it fits.** The SNR analysis says decisions flip on *sampling* noise;
simulations divide sampling noise by √n regardless of model bias. Nobody
has measured where the (model bias) × (search averaging) iso-wall-clock
frontier actually peaks — the campaign only ever moved along the
fixed-model axis. AlphaZero-family results repeatedly show small-net/big-
search dominating at fixed compute in exact-scoring games.

**Sketch.** Distill policy+q+quantiles from the distq corpus into an S/XS
config (both heads exist already; `--model-size S` + `--init-manifest`
distillation run is a day). Serve at n4096/d32 and sweep. Kill-test: if
S-at-n4096 can't beat M-at-n1024 at equal wall-clock, the frontier bends
the other way and we know M's capacity is load-bearing at serving, which
is itself new information (it currently is NOT load-bearing at n256).

## 3. Pairwise comparator head — train the decision, not the value

**Idea.** The argmax doesn't need calibrated values; it needs correct
*comparisons*. Add a head trained on pairs: P(action i ends better than
action j | state), supervised from search completed-Q pairs and real
outcomes. Serve by round-robin voting among the top-m actions.

**Why it fits.** 46% of decisions are within-noise ties of the top-2 —
exactly the regime where ranking losses beat regression losses (the whole
learning-to-rank literature exists because of this). The scalar/quantile q
head spends capacity on absolute calibration the argmax never uses.

**Sketch.** CGAB already produces per-action embeddings; a bilinear
comparator over action-pair embeddings is a small head. Train on the
existing corpus (pairs are free — no new generation). Serving: comparator
reranks the top-m after sequential halving, or replaces sigma(q) in the
final halving round. Kill-test: does comparator-reranked n256/d4 beat
97.38? One training run + one battery.

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

**Sketch.** Trainer: score-to-go target already decomposes (the exporter
emits score_decomposition per seat). Serving: sum the heads instead of
reading q. Kill-test: one cycle-6-style ablation (identical data/recipe,
decomposed head vs distq) — the distq playbook, re-run.

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

## Explicitly not on this list

Belief modeling / bag inference (oracle LOST — information is not the
constraint), checkpoint output-ensembles (shared-bias, measured 4-ways),
input-symmetry tricks (representation is already invariant), serving-side
cooperative objectives (noise multiplier, measured twice), bigger
monolithic models on the same labels (measured three ways).

*Sequencing suggestion: #1 and #3 are serving-side and cheap — they can
run during any training-side experiment. #2 gates on a distillation run,
#5 on the EI-1 verdict, #4 unlocks scale for everything and can start as
engineering whenever GPU-planning allows. #6 last — it needs the
flywheel confirmed.*
