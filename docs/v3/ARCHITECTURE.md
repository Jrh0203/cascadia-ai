# CascadiaFormer Architecture

## Recommendation

Build CascadiaFormer-Zero: a sparse public-state transformer with legal-action
queries, Cascadia-specific relation bias, score-to-go/value heads, policy heads,
opponent/market auxiliary heads, and search-supervised expert iteration.

The core bet is narrow and testable: the remaining score gap is not raw local
pattern recognition alone. It is cross-turn pattern realization, market timing,
stochastic refill handling, and opponent-conditioned access. CascadiaFormer
should model those public entities jointly, then use search to improve and
verify action choice.

The sole finalized post-D1 challenger is
[Cascadia Rival](../../cascadia_rival_final_architecture_proposal_7_16.md):
preserve this transformer-plus-Gumbel system as the frozen incumbent and
literal fallback, use a much cheaper structured policy only after it proves
terminal-difference correlation and trajectory economics, correct it with
paired full-incumbent terminal continuations, and distill confirmed selfish
improvements through one gated policy iteration at a time. Rival is research,
not the current serving default, and has zero current-rules strength evidence.

## Literature Basis

The architecture is intentionally no-frills:

- AlphaZero supplies the self-play plus search-improved policy/value recipe.
- ZeusAI for 7 Wonders Duel is the closest published board/card-game analogue:
  heterogeneous component tokens, transformer value/policy model, stochastic
  afterstates, and MCTS supervision.
- Chessformer and Leela transformer work show that domain-aligned tokenization,
  geometric bias, and action heads matter more than generic scale.
- Searchless chess transformer results argue for per-action value targets, but
  Cascadia lacks a Stockfish-grade oracle, so searchless distillation comes only
  after a strong Cascadia search teacher exists.
- Set Transformer, Graphormer/GraphGPS, Pointer Networks, Perceiver IO, Gumbel
  AlphaZero, and multiplayer AlphaZero inform the variable entity set,
  graph/geometry bias, dynamic action query, limited-budget search, and
  multi-seat value design.

## Public Input Schema

All model inputs must be public-state legal. Hidden future stack order is never
encoded.

Token groups:

- `game`: turn number, active seat, scoring card, cleanup state, phase bucket.
- `players`: seat-relative player summaries, nature tokens, current scores,
  remaining turns, and optional policy identity during self-play.
- `board_cells`: placed tiles and wildlife on each board with q/r, terrain
  edges, allowed wildlife, placed wildlife, keystone, rotation, owner, and age.
- `frontier_cells`: legal adjacent empty cells, local geometry, and prospective
  connectivity.
- `market`: four tile slots, four wildlife slots, three-of-kind state, and wipe
  affordances.
- `supply`: observable tile/wildlife counts and uncertainty summaries allowed by
  the engine boundary.
- `score`: current and potential category signals per seat.
- `history`: recent public actions only.
- `actions`: one token per legal compound action enumerated by Rust.

The action vocabulary is dynamic. Rust enumerates legal actions exactly; the
model scores those action queries instead of emitting a fixed global action id.

## Geometry And Relation Bias

CascadiaFormer uses a Cascadia geometric attention bias over public entities:

- axial hex distance and direction;
- same board, same component, same frontier, and adjacency;
- D6 orbit and transform identity;
- action uses market tile slot or wildlife slot;
- action touches a tile/wildlife/component already represented in state tokens;
- seat relation and turns until each opponent can affect the market;
- species, terrain, habitat, and scoring-category compatibility.

The default board fast path is a radius-6 hex disk with 127 stable cells. Legal
states outside the disk must remain exact through overflow entities. Overflow is
not clipped or projected.

## Model Shape

The current model family is an entity transformer plus action-query decoder:

```text
public tokens -> state transformer -> state latents
legal action tokens -> action encoder -> action queries
action queries x state latents -> policy, score-to-go Q, value, auxiliary heads
```

Primary heads:

- legal-action policy logits;
- per-action score-to-go Q;
- root value and score distribution;
- rank and score-differential summaries;
- category score decomposition.

Auxiliary heads:

- uncertainty for Q weighting and diagnostics;
- opponent next-draft and market-survival signals;
- pattern portfolio signals for Bear, Elk, Salmon, Hawk, and Fox;
- greedy-retention diagnostics while bootstrapping.

Initial model sizes:

| Model | Layers | Width | Heads | Use |
|---|---:|---:|---:|---|
| CascadiaFormer-tiny | 1 | 64 | 4 | plumbing and fixed-overhead floor only |
| CascadiaFormer-XS | 6 | 256 | 8 | 5M-parameter distillation/search-ratio probe |
| CascadiaFormer-S | 8 | 384 | 8 | bootstrap, EI-0, ablations |
| CascadiaFormer-M | 12 | 768 | 12 | current RTX 5090 champion family |
| CascadiaFormer-L | 16 | 1024 | 16 | only after data and gates justify it |

Use bf16 mixed precision on CUDA, gradient checkpointing when needed, and packed
relation-tail batches rather than Python-built dense relation matrices.

## Serving Semantics

The model's raw Q output is predicted score-to-go, not final score. Any Q-based
serving path must rank actions by:

```text
derived_final_q = exact_afterstate_score_active + predicted_score_to_go
```

This is mandatory because exact immediate score can differ across legal
afterstates. A synthetic rank-flip test must fail if serving ranks by raw
score-to-go alone.

Distributional-Q checkpoints expose eight independently trained quantile
heads at centered levels `(k + 0.5) / K`. Established serving uses their
arithmetic mean. The bridge also supports explicit `q25`, `q50`, and `q75`
research modes: it first sorts each action's heads (monotone rearrangement),
then linearly interpolates the requested probability. Rearrangement is needed
because independent quantile heads can cross; it preserves the arithmetic mean
exactly, so default `mean` behavior is unchanged. Non-mean modes are
distributional-checkpoint-only, provenance-recorded ablations, not promotions
by themselves.

### Structured value frontier

The legacy `score_decomposition` output is a root-level auxiliary and must not
be mistaken for an action-value head. A corrected-rules held-out preflight now
shows that the post-CGAB selected-action latent contains materially better
terminal wildlife/habitat/Nature signal than the incumbent scalar/value
comparisons (`3.4889` versus best-baseline `4.1528` RMSE, 760 untouched
non-exact roots). This authorizes an action-conditioned decomposition design,
but not the direct-final ridge used by the probe.

The exact-grounded design is now implemented behind
`q_decomposition=false` by default. Schema v4 exports the active seat's exact
afterstate wildlife/habitat/Nature vector for every action. The optional head
predicts three category score-to-go residuals and defines ordinary Q as their
exact sum:

```text
predicted_score_to_go = sum(predicted_category_score_to_go)
derived_final_q = exact_afterstate_score_active + predicted_score_to_go
```

The bridge's scalar exact-afterstate addition is algebraically identical to
summing the three exact components and three residuals. Selected real outcomes
provide category supervision as `terminal_components - selected_afterstate_components`;
counterfactual actions do not receive invented category labels. All q-valid
actions retain scalar/distributional completed-Q supervision on the component
sum. Distributional mode emits category quantiles and sums corresponding
quantile levels before existing risk selection.

When disabled, `q_component_head` is absent and the legacy state-dict/output
contract is unchanged. Schema loading, filtering, relation-tail materialization,
collation, terminal/afterstate sum invariants, head-only freezing, checkpoint
reload, and derived-Q serving are tested. This makes the implementation safe
to load and reproduce, but the preregistered head-first validation gate has now
failed: selected-final RMSE `4.1573` versus teacher `3.5520` (`-17.04%`
against a required `+10%`). The load-bearing structured-Q direction is closed;
no full-model run or gameplay occurred. Category heads remain eligible only as
auxiliary regularizers in a materially different architecture.

When a structured run warm-starts from a legacy checkpoint, each category
projection is initialized to one third of the loaded Q projection (per
quantile). Their sum therefore reproduces the incumbent Q projection at step
zero within floating-point reassociation tolerance; the head-first test
measures learned decomposition rather than recovery
from random total-Q initialization. The initialization mode is recorded in the
training report.

### Pairwise comparator (experimental)

The optional comparator consumes the same post-cross-attention, post-CGAB
action embeddings as the policy/Q heads. Its logit is a scalar merit
difference plus a low-rank skew interaction:

```text
C(i,j) = m(h_i) - m(h_j)
       + (L(h_i)·R(h_j) - L(h_j)·R(h_i)) / sqrt(rank)
```

Therefore `C(i,j) = -C(j,i)` and `C(i,i) = 0` by construction, while the skew
term can represent preferences that a single scalar action value cannot. The
legacy model contract is unchanged when `pairwise_comparator=false`.

Training emits both pair orientations and uses only actions with `q_valid`, at
least two samples/action, absolute completed-Q margin at least 0.25, and pair
SNR at least 1.0. A root contributes at most 32 undirected pairs, chosen from
the hardest pairs above the confidence gate; loss weights scale with SNR and
are clamped. `--pairwise-head-only` freezes the incumbent trunk and all legacy
heads for a cheap first kill test.

Serving mode `pairwise-borda` first retains the established policy's top-K
candidate mask (default 16), then averages antisymmetric log-odds against every
other retained action to produce a permutation-equivariant prior score. This
keeps unseen long-tail actions out of the comparator's support. It changes
policy priors only; derived final Q/value semantics stay untouched. The bridge
and benchmark record `policy_mode` plus `pairwise_policy_top_k`, and non-default
modes fail before launch unless the checkpoint manifest declares the comparator.

The completed held-out pilot learned reliable pair labels but failed the
serving-aligned decision gate: pairwise Borda improved top-one by only two net
roots, worsened completed-Q regret, and its bootstrap interval spanned zero;
logits plus Borda was top-one flat. The serving direction is closed and these
interfaces are retained only as research plumbing. A new architecture may use
reliable pair margins as an auxiliary loss, not silently reopen Borda serving.

## Search Semantics

The serving-strength search is Gumbel top-m + sequential halving
(`real-root-exporter/src/gumbel.rs`) with the model at both ends: policy
priors select root candidates from the full legal set, and leaf values are
derived final Q from batched model evaluations. Interior plies advance every
seat by its own argmax derived final Q (max^n). A blend weight `w` mixes the
value bootstrap with sampled greedy terminal rollouts while the value head
earns trust; `w = 1.0` removes CPU rollouts entirely.

At blended serving weights, each simulation owns an independent terminal
rollout RNG stream. `--gumbel-parallel-leaf-rollouts` may resolve those tasks
on the Rayon pool after a batched model step and then commits results in stable
simulation order. It is therefore an execution-only option, not a policy
change. The measured 2026-07-09 frontier improved one-game MPS latency by
about 6% but was 0.7% slower with two concurrent shared-bridge games; batch
batteries leave it disabled.

**No-peek contract:** search may never observe the true hidden tile-stack or
bag order. Every simulation redeterminizes hidden state before the root
action is applied; the legacy rollout path honors the same contract behind
`--rollout-determinize`. Benchmarks that violate this contract are marked
legacy-leaky and are not promotion evidence.

**Optional market-policy contract:** a free three-of-a-kind wildlife wipe is
an accept/decline decision followed by a chance draw and then the ordinary
draft decision. Policy and search value accept from public-hash-derived hidden
samples; they never condition acceptance on the real replacement order.
CascadiaFormer uses separate model rows for sampled accepted markets and, only
after acceptance, for the real revealed market. See
[RULES_CONTRACT.md](RULES_CONTRACT.md).

**Exact final-personal-turn frontier:** with
`--gumbel-exact-endgame-turns 1`, search recognizes that an active seat with
one personal turn remaining has no score-to-go after its action. It enumerates
the complete legal menu (ignoring serving pre-filters), scores every afterstate
with the engine, and deterministically chooses the maximum without invoking
the model or running simulations. When a free three-of-a-kind refresh is
available, decline is compared with the mean exact accepted-market optimum
over public-derived hidden samples; only after acceptance wins is the real
replacement revealed and solved exactly. This frontier is intentionally
own-score-only and rejects table-total objectives. K>1 is not implemented:
it requires genuine multi-seat/chance-tree solving rather than this terminal
identity.

### Open serving-wrapper challenger: Cascadia-Anchor

Full specification:
[`incumbent_anchored_gpu_rollout_policy_improvement_7_16.md`](../../incumbent_anchored_gpu_rollout_policy_improvement_7_16.md).

Cascadia-Anchor is a research proposal, not a serving default. It leaves
CascadiaFormer and current Gumbel search intact as the frozen incumbent
`pi_I`. Anchor v1 leaves the free-three refresh accept/decline decision to the
incumbent, commits it, reveals any accepted replacement, and starts only at
the resulting post-prelude public draft node. There it stores the exact
incumbent draft `a0`, screens alternatives for that fixed visible market, and
may replace `a0` only after fresh terminal continuations show a challenger’s
paired expected own-score advantage clears a preregistered practical margin.
Unresolved or invalid evidence executes `a0` literally. It never freezes a
post-accept draft before chance reveals the replacement.

The confirmation continuation `pi_R` must equal the complete serving
incumbent `pi_I` to support a local improvement claim over the current system.
The existing sampled-greedy leaf rollout, a direct transformer action, or a
distilled policy may screen but changes the estimand and cannot authorize an
incumbent-relative override in the first design.

A-EXACT confirmation has two independent simulation layers: the outer physical
hidden world being completed and the incumbent’s internal public-information
determinizations/search RNG at every future decision. The incumbent may not
observe the outer hidden order. A proposed GPU implementation batches
`roots x candidates x outer worlds` and uses compacted wavefront queues for
the nested incumbent searches, with exact Rust rules/scoring as the oracle.
Full-incumbent cost is a central feasibility kill test. Rules parity is not
enough: packed rows, numerical mode, Gumbel/rollout traces, RNG consumption,
market decisions, and every nested action must match the production bridge.
Any action divergence is an approximate continuation, not A-EXACT. A
pre-port bounded-inference power calculation must also show that the required
terminal-pair count is affordable.

The direct historical predecessor remains in
`crates/cascadia-search/src/policy_improvement.rs`. Its old-rules pattern-policy
confirmations were CI-positive on total score, but it used the same eight
samples to compare multiple candidates with unadjusted per-challenger c90
bounds. The successor requires a frozen selection stream plus fresh
confirmation, multiplicity-valid inference, forced-anchor bit identity, and a
fresh symmetric four-seat gameplay gate. No current-rules strength claim
exists.

### Final post-D1 challenger: Cascadia Rival

Full specification:
[`cascadia_rival_final_architecture_proposal_7_16.md`](../../cascadia_rival_final_architecture_proposal_7_16.md).

Rival makes Anchor the rollout-estimand spine and high-fidelity control.
The incumbent owns the exact root action and every unresolved or invalid case.
An NX-style exact semantic compiler and small D6-tied RivalNet may propose a
challenger and run many cheap, exact-rule continuations, but they do not
inherit incumbent fidelity.

For one frozen challenger, Rival measures terminal active-seat differences at
two fidelities. A paired panel runs both RivalNet and full-incumbent
continuations; an independent extra panel runs RivalNet only. A fixed
multifidelity control variate uses measured low/high correlation to estimate
the full-incumbent difference more efficiently. Candidate selection,
coefficient calibration, paired confirmation, and extra-low samples are
disjoint. Any random coupling must preserve the exact dynamic-urn marginal or
fall back to independent worlds.

The fixed finite-panel coefficient includes its `n_L/(n_H+n_L)` allocation
factor. The first lower bound is an analytic two-independent-sample bounded
Hoeffding construction with a deterministic per-game error ledger. Stable
correlation of either sign may help; empirical coverage tests alone cannot
define the interval.

Four isolated seat-relative instances each maximize their own expected raw
terminal score. Opponent state predicts resource pressure, not utility. Table
mean, donation, shared cross-seat memory/prices/plans, coordinated four-board
genomes, and seat sacrifice are forbidden. Own-coordinate `max^n` remains the
incumbent search heuristic consistent with that contract; paranoid minimax
against a fictitious three-seat coalition is also excluded.

`B_k` is the ordinary base/high-fidelity continuation, `W_k` is an offline
shadow/one-seat labeling instrument, and `M_(k+1)` is the sole ordinary v1
promotion/target candidate. Confirmed categorical preferences may form one new
relabel tranche. A retrained candidate is a new identity and requires a fresh
paired complete-game gate;
local confidence does not prove whole-policy safety. Rival does not begin
until D1 reaches its frozen boundary, a fresh baseline exists, unilateral
selfish tomography shows headroom, and exact compiler/simulator economics
pass. Its present target-reaching forecast is 25--35% within at most two
iterations and 3,000 post-D1 john0 GPU-hours, not strength evidence.

### Historical clean-sheet exploration: Cascadia Foundry

Full specification:
[`cascadia_foundry_original_architecture_proposal_7_16.md`](../../cascadia_foundry_original_architecture_proposal_7_16.md).

Foundry is a historical research proposal, not a serving default. It replaces
learned scalar value and ordinary move-tree search with exact terminal score contracts,
reverse completion lattices, persistent four-board plans, tiny public-state
controller programs, and atomic shared-resource scarcity prices. Independent
quality-diversity archives are canonicalized by semantic cell and lineage; the
result is frozen into a complete reactive policy capsule.

The deployment identity is the capsule meta-policy, not an individual
high-scoring genome. After commitment, the capsule may deterministically
condition/repair its plans from public facts, but it does not mutate, resample,
or run new hidden scenarios. The complete capsule is evaluated on fresh
dynamic-urn scenario streams and followed through terminal. Receding-horizon
resynthesis is a different proxy unless its full population-update policy is
simulated inside every continuation.

Foundry-Sovereign instantiates the same seat-local capsule four times with
isolated memory and cyclic seat equivariance. Foundry-Commons is explicitly a
single centralized policy with shared public memory/prices and possible
resource donation. John ruled on 07-16 that the allowed policy class is
explicitly non-cooperative. Commons, table utility, donation, joint four-board
planning, shared cross-seat state/prices, and the associated conditional 76%
forecast are therefore withdrawn rather than queued. Learned table-total
serving also remains scientifically closed.

Every hidden-world cohort pins controller, blueprint, memory, and mutation
hashes across lanes. Scenario RNG drives exact state-dependent urn returns and
wipes; it is independent of the physical seed. Identical public histories must
produce identical memory/actions. Static terminal contracts prove spatial
score witnesses, not acquisition history; chronology claims require complete
`GameState` replay.

The proposal has zero current-rules strength evidence. Only Sovereign's
single-seat score contracts, chronology/nonanticipativity audits, and
commitment-collapse diagnostics survive as optional Rival modules, where they
must earn admission through natural-frequency confirmed-challenger and
gameplay gates.

## Promotion Philosophy

Validation loss, imitation accuracy, and greedy top-1 retention are diagnostics.
Promotion requires paired gameplay evidence with:

- score mean and confidence intervals;
- category and wildlife breakdowns;
- search-retention/regret metrics;
- timing/resource ratios;
- clean provenance and dataset manifests.

The expected first useful mode is search-integrated serving: CascadiaFormer as a
K24/K32/K64 prefilter or value model inside sampled search. A direct no-search
policy is useful only after it proves nonregression.
