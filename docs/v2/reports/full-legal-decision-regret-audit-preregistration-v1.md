# Full-Legal Decision Regret Audit Preregistration

Status: **active**

Date: 2026-06-15

## Purpose

Phase 0 is complete. Before training another model, this audit determines
whether the missing score is primarily caused by:

1. legal actions absent from the champion's K32 frontier;
2. legal actions present but misvalued;
3. insufficient continuation search;
4. opponent and market-survival effects;
5. Nature Token draft or paid-wipe policy.

All prior research conclusions are ignored. Existing code may be reused only
as tested infrastructure.

## Frozen Domains

The diagnostic corpus uses 13 untouched raw-u64 game seeds, `61000-61012`.
All four seats follow the accepted exact MLX K32/R600 champion, producing
1,040 pre-action decision states balanced across seats and personal turns.

The work is sharded without overlap:

- john1: seeds `61000-61004`;
- john2: seeds `61005-61008`;
- john3: seeds `61009-61012`.

Performance-smoke seed `60999`, implementation-test seeds, and later online
oracle qualification seeds are disjoint from this corpus.

## Canonical Action Boundary

At each decision, the free three-of-a-kind replacement is applied when
feasible. The audit then enumerates every legal paired and independent draft,
tile coordinate, legal rotation, wildlife placement, and legal no-placement
choice for that observed market. Strictly dominated same-slot independent
drafts remain present and are labeled; they are not silently removed.

A paid wildlife wipe is a chance decision whose replacement wildlife is not
known when the token is spent. It is therefore not treated as an ordinary
deterministic placement action. The audit records no-wipe plus all 15 legal
single-wipe choices at token-bearing states and evaluates them as separate
public-information chance options. Multiple wipes are represented
sequentially: after each observed replacement, the policy may stop or choose
another legal wipe. No bundled action may use the real hidden bag to choose a
post-wipe draft.

## Evaluation Ladder

### Cheap complete screen

Every deterministic post-prelude action receives:

- exact current and resulting v2 score breakdowns;
- exact action metadata and canonical hash;
- the accepted MLX NNUE remaining-value estimate;
- a screen value equal to the model's immediate plus remaining estimate.

The cheap screen is a proposal mechanism only. It cannot establish regret,
headroom, or a ceiling.

### Substantial public evaluation

The following union is evaluated with full-terminal public-information
rollouts:

- the strongest 64 actions under the cheap screen;
- every action in the champion's current frontier;
- the champion-selected action;
- 16 deterministic sentinels stratified across the remaining screen ranks.

Duplicate actions are evaluated once and retain every source label. The
substantial budget is R1200 with common random numbers within each sequential
halving round.

The sentinels test whether top-64 truncation is credible. If a sentinel is
substantially best often enough to invalidate the screen, the top-64 design is
rejected and widened before any ceiling claim.

### High-confidence public evaluation

The strongest eight substantial actions are joined with the champion action
and the best action in the champion frontier. The deduplicated set is
re-evaluated with full-terminal public-information R4800 and common random
numbers within each round.

For every decision the report distinguishes:

- champion selected action;
- best high-confidence action from the champion frontier;
- best high-confidence action from the complete legal screen;
- whether each action was in top-64, the champion frontier, or the sentinel
  set;
- mean, standard deviation, sample count, and uncertainty-aware regret.

The public-information evaluator receives only the observed state and public
supply. Hidden tile order and wildlife order are redetermined from a
domain-separated seed.

## Perfect-Future Diagnostic

One decision per early, middle, and late phase in each game receives an
additional realized-hidden-future diagnostic over the high-confidence
finalists. Each candidate is applied to the real hidden game state and
continued by the accepted champion policy to terminal.

This is explicitly hindsight information. It is reported as
`realized-hidden-future` and is never used to choose the public oracle action,
train a deployable model, or satisfy the public ceiling gate.

## Paid-Wipe Diagnostic

At token-bearing audited states, no wipe and every legal single-wipe subset
are evaluated before replacement information is revealed. Each option uses
the same public redeterminations and a contingent post-replacement action
search. Reports include:

- expected value of each first wipe option;
- value of stopping;
- probability that a paid wipe is preferred;
- expected token return;
- slot count and wildlife composition wiped;
- phase and current token count.

The first implementation may smoke this diagnostic on a preregistered subset,
but Phase 1 cannot close until token/prelude results are included.

## Decomposition

Decision regret and action changes are summarized by:

- early, middle, and late personal-turn phase;
- paired versus independent draft;
- drafted wildlife species;
- Bear, Elk, Salmon, Hawk, and Fox score deltas;
- habitat score delta;
- Nature Token spend and paid-wipe choice;
- opponent demand for the drafted wildlife;
- visible duplicate count, public supply, and market-survival proxy;
- champion-frontier membership and cheap-screen rank;
- substantial and high-confidence uncertainty.

Game-block confidence intervals are used so the 80 decisions from one game are
not treated as independent observations.

## Correctness Gates

Before substantive collection:

1. complete action enumeration matches `GameState::legal_turn_actions` exactly
   in count, identity, order-independent set equality, and legality;
2. every action round-trips through the v2-to-legacy bridge and preserves its
   canonical identity;
3. arbitrary-root rollout evaluation agrees with the existing champion path
   when given the same frontier, budget, seed coupling, and RNG seed;
4. batching, chunking, and distributed sharding produce identical per-action
   estimates and merged reports;
5. public evaluation is invariant to hidden-state redetermination of the input
   game;
6. the realized-hidden-future diagnostic changes when hidden order changes and
   is never read by public selection;
7. all Rust workspace, feature-gated legacy AI, and focused MLX service tests
   pass;
8. model, weights, binary, source, seed domain, budgets, and worker provenance
   are recorded.

Any illegal action, missing legal action, duplicate identity, ordering drift,
hidden-information leak, non-finite value, fallback, or incomplete game
rejects the implementation.

## Performance Gate

Before the 13-game collection, profile one complete early, middle, and late
audit decision on seed `60999`, including enumeration, score construction,
MLX screening, substantial evaluation, high-confidence evaluation, reporting,
IPC, and service lifecycle.

If at least 80% of end-to-end time remains inside the already accepted exact
full-terminal rollout pipeline, the workload is classified as the same
dominant Phase 0 loop and must stay within its measured throughput envelope.
Otherwise this audit receives a new frozen reference and must achieve a 10x
single-Mac end-to-end speedup with exact action, value, and report parity
before substantive collection.

No speed result may reduce actions, budgets, horizon, samples, numerical
fidelity, model quality, or diagnostics.

### Owner-Directed Closure Amendment

On 2026-06-15, before substantive seeds were opened, the project owner
declared the strongest exact performance result sufficient and directed the
project to proceed with model-strength experiments.

The accepted full-legal teacher measures 143.775461 seconds, a 1.686192x
exact end-to-end speedup over the 242.433050-second frozen reference. The
former 24.243305-second target is retired as a prerequisite for this audit.
All action, budget, horizon, numerical, diagnostic, and exactness requirements
remain unchanged.

## Substantive Gates

The diagnostic corpus advances only if:

- all 1,040 decisions and all 13 games complete with clean shutdown;
- every canonical post-prelude action is screened;
- top-64 screen recall is at least 98% against the high-confidence winner, or
  the screen is widened and rerun;
- mean regret of the retained top-64 set is at most 0.15 points;
- the high-confidence result is stable enough that its game-block confidence
  interval supports the reported dominant error sources;
- paid-wipe and realized-hidden-future diagnostics are complete and separately
  labeled.

## Reachable-Ceiling Gate

After the diagnostic identifies a credible public operator, run it online on
a disjoint preregistered seed suite against the accepted champion.

Phase 1 exits only when:

- the public-information oracle scores at least `102.000` mean;
- its paired improvement over the accepted champion is at least six points in
  mean with a positive 95% game-block confidence lower bound;
- the dominant proposal, value, continuation, opponent, and token error
  sources are quantified with confidence intervals;
- the result reproduces across john1, john2, and john3.

If these gates fail, Phase 1 remains active. The audit, oracle, or action
coverage is improved before any large learning loop begins.
