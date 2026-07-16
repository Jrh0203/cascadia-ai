# Cascadia-Anchor: incumbent-anchored GPU rollout policy improvement

**Date:** 2026-07-16
**Literature cutoff:** 2026-07-16
**Status:** research proposal; post-D1 challenger; zero current-rules strength
evidence
**Objective:** exceed a mean seat score of 100 over 1,000 four-player games,
before habitat bonus, under one pinned rules and serving identity

**Final disposition:** Cascadia-Anchor is the policy-improvement spine and
high-fidelity control of
[Cascadia Rival](cascadia_rival_final_architecture_proposal_7_16.md). Rival
adds a statistically controlled low/high-fidelity continuation bridge,
unilateral headroom gates, and frozen rollout-policy iteration; Anchor remains
the clean full-incumbent estimand wherever its nested cost is affordable.

## Executive verdict

The architecture I have the most confidence can produce a **positive
improvement over the current system** is not a replacement for
CascadiaFormer. It is a conservative online policy-improvement layer around
the current transformer-plus-Gumbel policy:

> Always compute the incumbent action. Generate a small set of alternatives.
> Use exact terminal GPU rollouts to compare one challenger with the incumbent
> on fresh stochastic worlds. Override only when a statistically valid lower
> bound clears a preregistered practical margin. Otherwise execute the
> incumbent action exactly.

This report calls that system **Cascadia-Anchor**.

That recommendation is narrower than “this will reach 100.” It is the
highest-confidence route to **not throwing away the strength already earned**
while looking for missed points. It has four unusually strong reasons behind
it:

1. Classical Monte Carlo rollout policy improvement directly supports
   comparing root actions by taking each action and then following a frozen
   base policy. In backgammon, Tesauro and Galperin reported substantial
   reductions in decision error for base controllers ranging from weak linear
   policies to TD-Gammon, and emphasized that the trials are naturally
   parallel ([NeurIPS 1996](https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html)).
2. Cascadia already had a direct predecessor. The v2
   `LateConservativeBasePolicyImprovementStrategy` used an incumbent anchor,
   paired terminal continuations, a one-sided confidence guard, and literal
   fallback. It scored **+0.420**, 95% CI **[+0.179,+0.661]**, in its original
   50-game confirmation and **+0.520**, 95% CI
   **[+0.260,+0.780]**, after canonical redetermination. It was demoted for a
   preregistered non-Bear wildlife guardrail, not because the score gain
   disappeared. Those are historical old-policy/old-rules results, not v3
   strength evidence.
3. The current campaign has measured a noisy action-comparison problem:
   median decision SNR is near one and roughly 46% of decisions are
   noise-flippable. Terminal score is the exact objective. An incumbent-
   anchored terminal comparison asks a more direct question than another
   learned head: “does this deviation actually finish with more points?”
4. Accelerator-native board-game simulators are a demonstrated systems
   pattern. Pgx reports 10–100x throughput over compared Python environments
   and thousands of parallel simulations on GPU/TPU
   ([NeurIPS 2023 paper](https://papers.nips.cc/paper/2023/file/8f153093758af93861a74a1305dfdc18-Paper-Datasets_and_Benchmarks.pdf)).
   This establishes plausibility, not a promised Cascadia speedup.

The proposal has one central feasibility risk: if the rollout continuation is
truly the current serving policy, every future simulated decision must itself
run CascadiaFormer plus its current Gumbel search. That creates a nested search
whose cost is approximately:

    candidates
    × outer physical worlds
    × remaining decisions
    × incumbent search cost

No paper and no repository result proves that this is affordable. GPU batching
may change the economics, but affordability is a **kill test**, not an
assumption.

There is also no “cannot lose” theorem here. A finite-sample guard can make a
precise local claim about one deviation followed by the incumbent. It cannot
prove that repeatedly wrapping every later decision for all four seats
improves symmetric self-play. Only a fresh paired four-seat gameplay gate can
establish that.

This proposal is therefore:

- **more conservative than Cascadia-NX** because it retains the current model
  and action as the mandatory fallback;
- **more direct than another value architecture** because final confirmation
  uses terminal score, not a learned bootstrap;
- **not NNUE** because it is a planning and routing architecture, not a
  particular evaluator; and
- **not a rebranding of current Gumbel leaf rollouts** because those use a
  sampled-greedy continuation and a value blend, not the complete serving
  incumbent.

The previous clean-slate architecture proposal remains the higher-risk
structured representation and systems predecessor:
[`stochastic_board_game_ai_architecture_research_7_16.md`](stochastic_board_game_ai_architecture_research_7_16.md).

The historical third proposal was the most radical:
[`cascadia_foundry_original_architecture_proposal_7_16.md`](cascadia_foundry_original_architecture_proposal_7_16.md)
replaces scalar evaluation and ordinary search with exact terminal score
contracts and a committed reactive program capsule. John rejected its
cooperative Commons class and withdrew the associated 76% forecast. Anchor is
now the high-fidelity spine of
[Cascadia Rival](cascadia_rival_final_architecture_proposal_7_16.md); no branch
has current-rules strength evidence.

## 1. Evidence labels and claim boundary

This report uses four labels.

- **Published-direct:** the cited primary source directly measured or proved
  the stated result under its own assumptions.
- **Repo-direct:** the current repository or its archived, hash-addressable
  history directly contains the result. Historical scores retain their
  original rules and policy identities.
- **Transfer:** a mechanism worked elsewhere, but its Cascadia effect is
  unknown.
- **Synthesis:** a design proposed here. It is neither a published algorithm
  under this name nor Cascadia evidence.

The following boundaries are load-bearing:

- No located paper studies this Cascadia rules identity, current transformer,
  four-player self-play distribution, or 100-point endpoint.
- The historical 98.x v3 scoreboard is evidence under the July-9 rules
  identity. There is no admissible July-16 canonical baseline score at this
  report’s cutoff.
- The v2 conservative-rollout results used a pattern-aware base policy,
  historical rules, final-five activation, eight determinizations, and a
  different candidate frontier. They support the mechanism; they do not
  estimate the gain over CascadiaFormer.
- High-confidence policy-improvement and baseline-bootstrapping papers prove
  results under their own batch-MDP or off-policy-evaluation assumptions.
  Their baseline-fallback principle is useful here, but their theorems do not
  automatically transfer to approximate, online, four-player stochastic
  self-play.
- “Highest confidence” is a research-priority judgment. It is not promotion
  evidence, a probability estimate, or a guarantee of reaching 100.

## 2. The exact object being proposed

Three policies must never be conflated.

- \(\pi_I\): the **serving incumbent**, meaning the frozen CascadiaFormer
  checkpoint, current legal-action construction, current Gumbel configuration,
  exact-K1 behavior, market handling, and every other serving parameter.
- \(\pi_R\): the **rollout continuation policy** used after the candidate root
  action inside counterfactual simulations.
- \(\pi_W\): the resulting **wrapper policy**, which either executes the
  incumbent action or a confirmed challenger.

### 2.1 Root boundary: post-prelude public draft only

A Cascadia turn can contain a free three-of-a-kind refresh decision:
accept/decline first, a chance replacement second, and a draft/placement
decision only after the resulting market is public. A fixed action cannot
legally choose “accept” and also name a draft from a replacement that has not
been revealed.

Anchor v1 therefore does **not** override the current refresh decision:

1. the frozen incumbent chooses accept or decline from the ordinary public
   prelude state;
2. the engine commits that decision;
3. if accepted, the real replacement is revealed under exact rules; and
4. Anchor begins only at the resulting **post-prelude public draft state**
   \(s_d\), with one fixed visible market.

If no refresh is available, the ordinary public state is already \(s_d\).
Candidate actions in the first proposal are draft/placement actions for that
fixed market. The committed prelude is preserved when the complete
`TurnAction` is logged/executed. Future decisions inside terminal
continuations still use the complete incumbent, including its own legal
prelude decision followed by its downstream draft.

This boundary is deliberately conservative. A later wrapper could compare
branch-contingent refresh policies—decline plus its draft policy versus accept
plus a draft policy for every sampled replacement—but that is a separate
chance-node design and identity. It must never freeze a post-accept draft
before the replacement is public.

At post-prelude state \(s_d\), for active seat \(i\), let
\(a_0=\pi_I^{\text{draft}}(s_d)\) be the exact incumbent draft action. For a
candidate draft action \(a\), define:

\[
\mu_a(s_d;\pi_R)=\mathbb E\left[
G_i
\mid
S=s_d,\ A_i=a,\ \text{all subsequent decisions use }\pi_R
\right],
\]

where \(G_i\) is the active seat’s exact terminal raw score before habitat
bonus. The expectation includes the correct conditional distribution of all
remaining tiles and wildlife, all later market events, all opponent actions,
and any frozen policy randomness.

The local paired advantage is:

\[
\Delta_a(s_d;\pi_R)=\mu_a(s_d;\pi_R)-\mu_{a_0}(s_d;\pi_R).
\]

The wrapper’s local question is not “which action has the largest noisy
estimate?” It is:

> Is there enough fresh evidence that one frozen challenger beats the action
> the incumbent would actually play by more than a practical margin
> \(\epsilon\)?

If not, \(\pi_W^{\text{draft}}(s_d)=a_0\).

### 2.2 When this is policy improvement over the incumbent

The strongest interpretation requires:

\[
\pi_R=\pi_I.
\]

That means every later decision for every seat in every outer rollout is made
by the complete frozen transformer-plus-Gumbel policy. A direct transformer
argmax, sampled-greedy completion, distilled policy, or old pattern heuristic
is cheaper, but it changes the estimand. A challenger that is good when future
play is greedy may be bad when future play uses the current search policy.

Accordingly, Cascadia-Anchor defines three rollout modes:

| Mode | Continuation | Role | Claim permitted |
| --- | --- | --- | --- |
| A-EXACT | exact serving \(\pi_I\) | final confirmation | local improvement estimate against the true incumbent continuation |
| A-DIRECT | frozen direct transformer policy, no search | cheap screen and ablation | improvement only under A-DIRECT continuation |
| A-DISTILL | frozen distilled approximation to \(\pi_I\) | cheap screen if fidelity clears | improvement only under A-DISTILL; no incumbent guarantee |

A-DIRECT and A-DISTILL may discard candidates to save compute. They may not
authorize a production override. Only a fresh A-EXACT confirmation can do
that in the first eligible design.

### 2.3 Local claim versus policy claim

An A-EXACT confirmation can support:

> Take \(a\) now, then return all four seats to the frozen incumbent: expected
> active-seat terminal score exceeds taking \(a_0\) now and then returning to
> the incumbent.

It does not establish:

> Repeatedly run the wrapper at all eligible decisions for all four seats and
> mean self-play score will rise.

Repeated overrides change later behavior, visited states, market competition,
and every opponent’s response. The complete wrapper is a different
multi-agent policy. That is why the local guard is routing evidence and the
paired full-game gate is strength evidence.

## 3. This has a direct Cascadia predecessor

The anchor/fallback idea must not be presented as novel. The current source
tree still contains the historical implementation in
[`crates/cascadia-search/src/policy_improvement.rs`](crates/cascadia-search/src/policy_improvement.rs).

The v2 `LateConservativeBasePolicyImprovementStrategy`:

1. used the pattern-aware policy’s action as the anchor;
2. activated only in the final five personal turns by default;
3. enumerated the frozen K8+H6+B8 candidate frontier;
4. sampled eight shared public-information redeterminizations;
5. applied each candidate, completed the game under the frozen pattern policy,
   and scored the acting seat’s exact terminal base total;
6. computed paired candidate-minus-anchor returns;
7. admitted a challenger only when a one-sided 90% Student-\(t\) lower bound
   exceeded zero; and
8. played the anchor exactly when none qualified.

Its archived results are recoverable from tag
`archive/pre-v3-repo-cleanup-2026-07-01`:

| Historical evaluation | Treatment | Baseline | Paired delta | 95% CI | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| ADR 0024 original confirmation, 50 games | 91.915 | 91.495 | +0.420 | [+0.179,+0.661] | all frozen score/category/runtime gates passed |
| ADR 0068 canonical-redetermination requalification, 50 games | 92.100 | 91.580 | +0.520 | [+0.260,+0.780] | total-score gain retained; demoted because non-Bear wildlife delta was -0.375 |

Those figures are **historical mechanism evidence only**. The base policy,
candidate generator, rules, score class, and strength tier all differ from v3.

### 3.1 What the predecessor got right

- literal incumbent fallback;
- paired physical worlds;
- exact terminal scoring;
- frozen continuation;
- public-information redetermination;
- deterministic, legal, replayable matches;
- a late-game scope that made full completion affordable; and
- disjoint pilot and gameplay-confirmation seed blocks.

### 3.2 What the successor must fix

The old per-decision rule tested many challengers on the same eight samples,
then selected the largest lower bound. A 90% one-sided interval per challenger
does not control the familywise false-override rate across the whole menu, and
the same samples both identified and certified the winner. Repeated decisions
compound the issue.

Cascadia-Anchor therefore adds:

- a selection/confirmation sample split;
- either fixed-sample simultaneous bounds or an anytime-valid,
  multiplicity-controlled sequential rule;
- a positive practical margin \(\epsilon\), not merely zero;
- complete identity hashing;
- forced-anchor parity against the current serving policy;
- explicit separation between outer physical chance and the incumbent’s
  internal search randomness;
- GPU-resident exact simulation;
- equal-wall controls against spending the same compute on more ordinary
  Gumbel search; and
- unilateral and symmetric full-policy gates.

The new contribution is the **current-policy-fidelity, GPU, and inference
generalization**, not the anchor/fallback concept.

## 4. Why current Gumbel search is not already this proposal

The current v3 search in
[`cascadiav3/real-root-exporter/src/gumbel.rs`](cascadiav3/real-root-exporter/src/gumbel.rs)
does several things worth retaining:

- exact legal compound-action enumeration;
- model policy priors;
- Gumbel top-\(m\) candidate selection—sampling without replacement during
  exploration, deterministic top-logit selection in evaluation;
- sequential halving;
- shared root determinizations across actions;
- exact afterstate-score grounding;
- \(max^n\) interior choices;
- exact-K1;
- explicit market refresh handling; and
- completed-Q values from visited simulations.

Anchor v1 preserves that refresh handling exactly and starts only after its
accept/decline branch and any real replacement reveal have resolved. It
compares draft actions against one fixed public market; it does not collapse
the prelude chance node into a clairvoyant compound action.

Its leaf value, however, is:

\[
w\cdot\text{model bootstrap}
{}+(1-w)\cdot\text{sampled-greedy terminal rollout}.
\]

At the serving default \(w=0.5\), neither branch is an exact terminal
continuation of the complete serving policy. Current Gumbel search is a
bounded planning algorithm guided by a learned evaluator and a cheap rollout.
Cascadia-Anchor is an outer one-deviation test whose confirmation endpoint is
terminal score under a frozen continuation.

The distinction is practical:

- current Gumbel asks which action looks best within a fixed search budget;
- Anchor asks whether a particular challenger has enough independent terminal
  evidence to justify departing from the exact incumbent action.

Anchor should reuse Gumbel’s candidate information. It should not pretend the
two estimators are identical.

### 4.1 Why the campaign’s negative search results do not settle Anchor

The proposal must absorb, not ignore, the current campaign’s strongest
counterevidence:

- increasing the historical champion from n1024/d16 to n4096/d16 produced
  only +0.21 with CI [-0.59,+1.01] at roughly 3.1x decision cost;
- the R3.2 deeper-own-turn screen missed its regret bar;
- the R0.4 completed-Q LCB selector was flat;
- R0.2’s extra rollout-policy RNG sharing worsened gap variance by 4.4%;
- static sigma and Q-bias corrections did not survive confirmation; and
- exact-K1 was a 28.99x frontier speed win but score-neutral because the model
  already chose score-optimal final actions.

Those results make an ordinary “more search,” “add an LCB,” or “share more
randomness” story unconvincing. Anchor differs in one causal respect: it keeps
the incumbent decision as an explicit control and asks for fresh paired
**terminal** evidence about one deviation under a frozen continuation. The
historical v2 implementation is positive evidence that this distinction once
mattered in Cascadia.

It may not matter at the current strength tier. The negative evidence is why
A-EXACT affordability, independent terminal audit, and the equal-wall
extra-Gumbel control are mandatory falsifiers rather than optional ablations.

## 5. System architecture

### 5.1 Decision flow

    public turn state
        |
        +--> incumbent refresh accept/decline
        |             |
        |       commit; reveal replacement if accepted
        |             |
        |       post-prelude public draft state s_d
        |             |
        +-------------+--> run frozen incumbent draft -----+
        |                                                   |
        |                                             anchor action a0
        |
        +--> frozen candidate factory
                 |
                 +--> a0, incumbent top-m, survivors,
                      and limited diversity slots
                                |
                                v
                       cheap A-DIRECT/A-DISTILL screen
                                |
                         one challenger a*
                                |
                                v
                    fresh A-EXACT paired confirmation
                    a* versus a0 to terminal score
                                |
                   +------------+------------+
                   |                         |
             LCB(Delta) > epsilon       unresolved / fail
                   |                         |
                   v                         v
                play a*                  play a0 exactly

The first causal bakeoff should keep the candidate factory deliberately
narrow: \(a_0\) plus draft actions already exposed by the incumbent
post-prelude root evaluation or Gumbel survivor set. Introducing a new
proposer and a new terminal adjudicator simultaneously would make a positive
result hard to attribute.

After the mechanism wins, optional proposal sources may include:

- the direct transformer top-\(m\);
- high-prior actions eliminated early by Gumbel;
- high completed-Q alternatives;
- one or two full-menu diversity slots selected without terminal outcomes;
- a later D1 checkpoint;
- a later Cascadia-NX specialist; or
- policy-population proposals.

The incumbent action is mandatory even if it falls outside another proposer’s
menu. Candidate construction is frozen before selection worlds exist.

### 5.2 Two independent state and randomness layers

Full-incumbent continuation creates a subtle but essential nesting.

**Outer physical world.** This is the counterfactual game being completed. It
contains one conditional hidden tile/wildlife order and consumes real chance
events exactly as a physical game would.

**Inner incumbent search worlds.** At each future public state, \(\pi_I\) must
perform its own no-peek determinizations, Gumbel sampling, rollout randomness,
and market evaluation. Those inner worlds guide the policy action but are not
the outer world.

The incumbent must never see the outer hidden order when choosing an action.
Doing so would turn confirmation into an oracle-information evaluation and
change both the policy and the estimand.

At the wrapper root, the current turn’s prelude has already been committed.
At every later outer turn, the incumbent’s inner policy first chooses the
prelude from public information; only the outer engine then exposes the real
replacement and requests the incumbent’s downstream draft. Golden traces must
cover no-refresh, decline, and accept/reveal/draft paths.

The random identity should be counter-based or otherwise domain-separated by:

    rules identity
    wrapper identity
    source root
    candidate
    outer-world index
    future decision index
    policy-search role
    internal determinization
    internal simulation
    stochastic subevent

Selection, confirmation, audits, and full-game gates use distinct domains.
Forced-anchor mode must not perturb the incumbent’s game RNG stream.

### 5.3 Exact endpoint

A-EXACT confirmation runs to game over and records the active seat’s exact
score before habitat bonus. In the pinned no-habitat-bonus rules identity,
`ScoreBreakdown.total == ScoreBreakdown.base_total`; the ledger records both
fields and refuses an identity where that equality is not guaranteed unless a
new endpoint is explicitly registered. It does not use:

- a value bootstrap;
- a quantile or risk objective;
- table total;
- rank utility;
- a learned dynamics model; or
- a category-weighted surrogate.

The full four-score vector and category decomposition remain diagnostics for
mechanism and regression analysis. They do not silently alter the objective.

### 5.4 Activation horizon

Start in the final **two to five personal turns**, with the exact cutoff chosen
on inspectable feasibility/calibration roots and then frozen.

Reasons:

- the v2 positive control was final-five;
- fewer remaining decisions sharply reduce nested A-EXACT cost;
- terminal score is closer and less variable;
- final-personal-turn roots are excluded from Anchor while the incumbent’s
  adopted exact-K1 path remains unchanged; K1 already showed the model selects
  score-optimal final actions and yielded no score gain; and
- this is a fixed-policy terminal completion, not a reopening of the failed
  ordinary depth-2 Gumbel program or exact K2.

If late-game A-EXACT is not affordable after GPU residence, the exact-incumbent
version is infeasible. Weakening the continuation after reading results is not
an acceptable rescue.

## 6. How search runs on the GPU

GPU execution should be a resident wavefront simulator, not a CPU loop that
calls a GPU model one state at a time.

### 6.1 Outer batch

The outer tensorized work unit is:

    roots × candidates × physical worlds

Each lane holds a compact struct-of-arrays game state:

- public board occupancy and token species;
- exact habitat components and scoring caches;
- market tiles and wildlife;
- remaining bag/stack priority arrays or permutation indices;
- per-seat turns and Nature Tokens;
- scoring-card identity;
- active seat and phase;
- legality metadata; and
- deterministic RNG counters.

The engine applies actions, resolves market cleanup/chance, updates exact
integer scores/features, and advances the active seat without a host round
trip.

### 6.2 Nested incumbent batch

At a future outer decision, every live lane needs an incumbent action. That
expands into:

    live outer states
    × legal actions
    × incumbent internal worlds
    × incumbent simulations/frontier

The implementation should use queue compaction and fixed kernels:

1. compact live outer states needing a decision;
2. build exact legal afterstates;
3. batch the transformer evaluation;
4. initialize incumbent Gumbel state;
5. advance all internal simulations one wavefront;
6. compact terminals and new model frontiers;
7. repeat until each incumbent decision resolves;
8. apply the selected action only to its corresponding outer world; and
9. return the lane to the outer completion queue.

This is conceptually similar to accelerator-native batched planning in
[Mctx](https://github.com/google-deepmind/mctx) and simulator vectorization in
[Pgx](https://arxiv.org/abs/2303.17503), but Cascadia’s exact rules and
without-replacement physical chance require a dedicated implementation.

Dynamic recursion, per-lane heap allocation, and host synchronization at every
ply should be avoided. Candidate menus, frontier nodes, and rollout lanes use
bounded arenas plus masks/offsets.

### 6.3 What remains exact

- legal moves and compound action application;
- market refresh and cleanup;
- without-replacement tile/wildlife chance;
- wildlife and habitat scoring;
- Nature Token accounting;
- terminal detection;
- public-information redetermination;
- active-seat score endpoint; and
- replay identity.

The transformer remains floating point under the incumbent’s frozen precision
contract. Integer game evolution and scoring must match the Rust oracle
exactly.

### 6.4 CPU oracle and parity

The current Rust engine remains the reference oracle. GPU implementation is
not accepted on “close mean score.” Two separate parity gates are mandatory.

**Rules/state parity:**

- action-by-action legal-menu parity;
- exact transition parity;
- exact terminal and category-score parity;
- canonical public-state/redetermination invariance;
- replay-hash parity;
- without-replacement marginal construction checks; and
- deterministic reruns under a pinned identity.

**Complete incumbent-policy trace parity:** GPU batching can change floating-
point reductions, TF32 behavior, tie order, sampled-greedy ordering, and RNG
consumption even when integer rules are perfect. On a large pinned corpus,
the nested implementation must reproduce the production bridge’s:

- packed evaluator rows and masks;
- evaluator outputs bit-exactly where the serving contract requires it, or
  within a frozen tolerance that is separately proved action-stable;
- root candidate order, exploration/evaluation mode, Gumbel/search seeds,
  visit counts, completed-Q values, and selected actions;
- market accept/decline and revealed-market downstream draft decisions;
- sampled-greedy continuation traces and terminal rollout ordering;
- every RNG-domain counter/consumption trace; and
- final action, replay, and terminal-score identity.

The corpus must include no-refresh, decline, accept/reveal/draft, tie, exact-K1,
and truncation edge cases. Any policy-action divergence means the port is an
approximation \(\hat\pi_I\), not \(\pi_I\). It may be renamed and tested as a
proxy, but it loses the A-EXACT incumbent-relative claim.

A proposed engineering bar is zero mismatches over at least one million
randomized reachable transitions plus exhaustive curated edge fixtures. That
number applies to rules/state parity and is a synthesis to preregister, not a
literature constant. The policy-trace corpus receives its own frozen size and
strata before evidence is read.

### 6.5 Central throughput kill test

Before building the full wrapper, benchmark these completed outer trajectories
per second:

1. exact rules with a trivial frozen policy;
2. A-DIRECT direct-transformer continuation;
3. A-DISTILL distilled continuation;
4. A-EXACT at current serving settings; and
5. A-EXACT with every lane starting in final-two, final-three, final-four, and
   final-five personal-turn strata.

Measure complete **decisions and terminal comparisons per wall-second**, not
raw kernel calls or model rows. Include candidate-menu construction,
compaction, nested search, score extraction, and synchronization.

If A-EXACT cannot confirm a useful fraction of held-out late-game decisions
within the allowed serving budget, the system either:

- remains a shadow/offline teacher;
- uses A-DIRECT/A-DISTILL as an explicitly proxy-only research arm; or
- is closed.

It does not inherit an incumbent-improvement claim from a cheaper
continuation.

## 7. Statistically valid override rule

### 7.1 Screen, then confirm

For each eligible root:

1. compute and freeze \(a_0\);
2. verify the incumbent prelude is already committed and \(s_d\) is public;
3. build the frozen draft-candidate set without terminal outcomes;
4. on a **selection stream**, spend a fixed A-DIRECT/A-DISTILL or small
   A-EXACT budget across candidates and choose one challenger \(a^\star\);
5. discard those outcomes for inferential purposes;
6. on a fresh **confirmation stream**, compare only \(a^\star\) and \(a_0\)
   under A-EXACT;
7. override only when the registered lower bound on
   \(\Delta_{a^\star}(s_d;\pi_I)\) exceeds \(\epsilon\); and
8. otherwise execute \(a_0\).

Fresh confirmation removes the winner’s-curse bias caused by choosing and
certifying the maximum on the same worlds. High Confidence Policy Improvement
similarly separates policy search from held-out safety evaluation under its
own setting ([Thomas et al., ICML 2015](https://proceedings.mlr.press/v37/thomas15.html)).

If more than one challenger survives into confirmation, use simultaneous
one-sided bounds with familywise error control. Do not apply an unadjusted
interval to the selected maximum.

### 7.2 Statistical-power kill test before the GPU port

Nonasymptotic validity can be computationally useless if the score range is
wide. For a paired difference \(D\in[-R,R]\), a one-sided Hoeffding half-width
is:

\[
h_N=R\sqrt{2\log(1/\alpha)/N}.
\]

As an illustration—not a proposed configuration—\(R=50\),
\(\alpha=0.0025\), and \(h_N=0.5\) require about 120,000 terminal pairs for
one decision. \(R=100\) requires about 480,000. Physical-world pairing may
sharply reduce empirical variance, but it does not narrow a worst-case
Hoeffding range.

Before a broad GPU implementation:

1. prove the tightest valid global and root-specific score-difference ranges;
2. estimate paired-difference variance by late-game stratum on inspectable,
   frozen CPU/current-policy roots;
3. choose the exact fixed or sequential interval family;
4. freeze candidate practical margins, alpha allocation, desired power, and
   maximum serving budget;
5. plot required \(N\) over the full \((R,\sigma_D,\epsilon,\alpha)\) grid;
   and
6. close the serving design before the port if no statistically valid rule can
   resolve useful effects within the wall budget.

This calculation is separate from nested-search throughput. Both statistical
sample count and per-pair execution cost must clear. A tested
variance-adaptive empirical-Bernstein rule may be necessary in the first
practical design even though the conceptually simplest reference remains
fixed-sample.

### 7.3 Fixed-sample first implementation

The first implementation should be simple and hard to misuse:

- fixed selection budget;
- exactly one frozen challenger;
- fresh fixed confirmation size \(N_C\);
- no intermediate reads;
- paired terminal differences \(D_j\);
- one nonasymptotic bounded-mean lower confidence bound; and
- override only if \(L_{N_C}>\epsilon\).

Terminal Cascadia score is finite. The rules implementation should certify a
valid bound \(R_s\) for each root or a valid global bound such that:

\[
D_j\in[-R_s,R_s].
\]

Then normalize to \([0,1]\), construct the bounded-mean interval, and transform
it back to score points. A paired \(t\)-interval may be logged for continuity
with historical reports but should not be the only safety decision unless its
assumptions and multiplicity handling are explicitly accepted.

The practical margin \(\epsilon\) accounts for:

- latency/opportunity cost;
- the chance that tiny local gains fail to compose; and
- the project’s desire for decisions large enough to matter.

It does not compensate for simulator or policy-port bias. Any semantic or
trace-parity failure is categorical and fail-closed.

Its value is selected on calibration data and frozen before the untouched
root and gameplay gates.

### 7.4 Sequential optimization later

After the fixed design is validated, a sequential version may stop when:

- \(L_t>\epsilon\): override;
- \(U_t\le\epsilon\): fall back early;
- maximum budget reached unresolved: fall back; or
- any failure/provenance issue occurs: fall back.

Ordinary intervals repeatedly inspected are invalid. Use either:

- a time-uniform confidence sequence for bounded means, such as the
  nonparametric confidence-sequence framework of
  [Howard et al.](https://doi.org/10.1214/20-AOS1991); or
- a separately derived and tested bounded-difference group-sequential
  procedure with frozen looks and alpha spending.

The repository’s sanctioned Lan–DeMets runner governs paired **gameplay**
gates with planned final looks of at least 100 pairs. It is not automatically
valid for terminal samples within one decision. Reuse is allowed only after
the per-root input process, boundaries, multiplicity, and empirical coverage
have been independently shown applicable; otherwise implement a distinct
registered procedure.

Empirical Bernstein methods motivate variance-adaptive stopping
([Mnih, Szepesvári, and Audibert, ICML 2008](https://icml.cc/Conferences/2008/papers/523.pdf)),
but an implementation must still be valid for the bounded paired-difference
process and the chosen stopping rule.

Best-arm literature distinguishes fixed-budget selection from
fixed-confidence identification
([Kaufmann, Cappé, and Garivier, JMLR 2016](https://www.jmlr.org/papers/v17/kaufman16a.html)).
Sequential Halving is useful for the screen; it is not, by itself, an
incumbent-beating certificate.

### 7.5 Alpha ledger

If the design wants a formal bound on any false local override claim within
one game, let \(H_{\max}\) be the maximum eligible wrapper decisions across
all four seats in one complete game and allocate:

\[
\alpha_{\text{decision}}=\alpha_{\text{game}}/H_{\max}.
\]

Any simultaneous challengers and sequential looks spend from that decision
budget. This controls the statistical claim under the frozen simulator and
estimand. It does not prove full-policy improvement or supply a finite
lifetime guarantee over indefinite deployment.

### 7.6 Fail-closed behavior

The wrapper plays \(a_0\) on:

- no cleared challenger;
- unresolved confirmation;
- timeout;
- incomplete worlds;
- NaN or numerical failure;
- GPU/CPU parity error;
- rule, checkpoint, policy, sampler, or budget hash mismatch;
- replay failure;
- chance-marginal audit failure; or
- unavailable accelerator capacity.

Fallback means the exact stored incumbent action, not the challenger with the
highest current sample mean.

## 8. Physical-world coupling and variance reduction

For paired confirmation, \(\omega_j\) denotes the complete registered joint
random object for pair \(j\): the outer physical chance tape plus whatever
coupling of the two continuations’ inner incumbent-policy randomness the
sampler contract permits. Different \(j\) bundles remain independent.

\[
D_j=G_i(s,a^\star,\omega_j;\pi_I)
{}-G_i(s,a_0,\omega_j;\pi_I).
\]

The paired variance is:

\[
\operatorname{Var}(D)=\operatorname{Var}(G_{a^\star})
{}+\operatorname{Var}(G_{a_0})
{}-2\operatorname{Cov}(G_{a^\star},G_{a_0}).
\]

Common random numbers help only when they induce positive covariance.
Veness, Lanctot, and Bowling showed meaningful simulation-equivalent gains in
several stochastic games, but also make the covariance requirement explicit
([NeurIPS 2011](https://proceedings.neurips.cc/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html)).

The incumbent already shares root physical determinizations across actions.
That is the first paired control. R0.2’s additional sharing of rollout-policy
RNG worsened gap variance by 4.4%, so “more coupling” is not a default.

A very recent May 2026 paper specifically studies common random numbers for
rollout planning and proves a depth-dependent coupling under stated
conditions while also constructing cases where naive full coupling is worse
than independent sampling
([Yadav et al.](https://arxiv.org/abs/2605.04732)). It is currently available
as arXiv v1 with an RLJ 2026 journal reference. Its one-step \(d=1\)
construction is worth translating, but the existing Cascadia hidden tape
cannot be assumed to satisfy its theorem without a rules-level proof.

The registered coupling bakeoff should compare:

1. independent outer worlds;
2. current physical root-world coupling;
3. the translated depth-dependent coupling; and
4. any continuation-policy coupling only after separate evidence.

Every method must preserve each action’s exact conditional marginal. Coupling
eligibility is frozen on a disjoint calibration block and requires both:

- lower paired-difference variance; and
- lower fixed-wall action-selection error on untouched roots.

If not, use independent worlds. Control variates are eligible only when their
expectation is exactly known or an unbiased, independently calibrated
construction is proved. An online coefficient fit on the same samples can
introduce bias.

## 9. Identity and reproducibility contract

One wrapper identity hashes:

- rules and score objective;
- incumbent checkpoint and numerical precision;
- incumbent legal-menu and serving configuration;
- exact-K1 and market behavior;
- wrapper root boundary and incumbent ownership of refresh/prelude;
- candidate-factory rule;
- A-DIRECT/A-DISTILL checkpoint or distillation identity;
- A-EXACT continuation definition;
- activation horizon;
- outer-world sampler and coupling;
- inner incumbent RNG domains;
- selection budget;
- confirmation budget and inference method;
- familywise alpha allocation;
- practical margin \(\epsilon\);
- timeouts and fallback behavior;
- CPU/GPU simulator revisions; and
- source revision.

Changing any item creates a new wrapper. Old root audits and gameplay gates do
not silently transfer.

Every decision ledger records:

- public-state hash;
- committed prelude and post-prelude public-market hash;
- \(a_0\) and its incumbent trace identity;
- candidate set and proposal sources;
- selection worlds and selected challenger;
- fresh confirmation worlds;
- paired terminal returns;
- bound, margin, and decision;
- override/fallback reason;
- latency by stage;
- simulator parity/version evidence; and
- complete policy/sampler hashes.

Durable ledgers are written before summary reports. Incomplete decisions fail
closed.

## 10. Multiplayer deployment

### 10.1 Unilateral diagnostic

Run one wrapped seat against three frozen incumbent seats, balanced across all
four positions. This measures whether local deviations are exploitable or
create unusual market interactions. It is descriptive/diagnostic, not the
campaign endpoint.

### 10.2 Symmetric primary gate

Compare:

- four seats using the frozen wrapper; and
- four seats using the frozen incumbent,

in separate games paired by game seed and rules identity.

Use the game as the independent unit:

\[
D_g=\frac14\sum_{i=1}^{4}G^{W}_{g,i}
{}-\frac14\sum_{i=1}^{4}G^{I}_{g,i}.
\]

The four seats within a game share a market and trajectory and are not four
independent observations.

An optional conservative deployment mixture:

\[
\pi_\lambda=(1-\lambda)\pi_I+\lambda\pi_W
\]

may test whether gradual adoption reduces interaction shift. \(\lambda\) must
be frozen before a gate, and a positive mixture result does not promote the
full wrapper.

## 11. Why this differs from NNUE

NNUE is a cheap, incrementally updated neural evaluator. It maps a state or
afterstate to a value quickly enough that search can call it many times.

Cascadia-Anchor is an **online control architecture**:

- the incumbent proposes an action;
- a counterfactual simulator estimates terminal consequences;
- a statistical guard decides whether to override; and
- the incumbent remains the fallback.

Anchor can use CascadiaFormer, NNUE, Cascadia-NX, a distilled policy, or a
handcrafted controller inside different roles. Its defining feature is not the
evaluator. It is the incumbent-relative terminal comparison and routing rule.

| Question | NNUE | Cascadia-Anchor |
| --- | --- | --- |
| Primary object | fast learned evaluator | conservative policy-improvement wrapper |
| Output | value/Q estimate | incumbent action or confirmed override |
| Exact terminal simulation required | no | yes for A-EXACT confirmation |
| Baseline fallback | not inherent | mandatory |
| Statistical action guard | not inherent | load-bearing |
| Can wrap current transformer | no; it would replace/augment evaluation | yes |
| Main risk | representation bias/accuracy | nested rollout cost and finite-sample inference |

The previous
[Cascadia-NX proposal](stochastic_board_game_ai_architecture_research_7_16.md)
is NNUE-inspired in its incremental factor core, although it adds a semantic
global residual and GPU planner. Anchor and NX are complementary:

- Anchor is the lowest-downside bounded serving test and my preferred first
  preflight; its probability of finding a positive gain is unknown.
- NX has greater architectural upside if the current evaluator/bridge is the
  true ceiling.
- A successful NX policy could later become the incumbent, the screen policy,
  or a challenger proposer inside Anchor.

## 12. Comparison with the current system and Cascadia-NX

| Dimension | Current CascadiaFormer + Gumbel | Cascadia-Anchor | Cascadia-NX |
| --- | --- | --- | --- |
| Incumbent retained | n/a | yes, literal fallback | no; challenger evaluator/planner |
| Candidate scoring | learned Q + bounded search + greedy/value leaf | terminal A-EXACT challenger-vs-anchor confirmation | structured fast/full evaluator + GPU world search |
| Architecture risk | lowest | medium systems/statistical risk | high representation + systems risk |
| Compute risk | known expensive | potentially extreme nested search | large engine port, intended cheaper evaluator |
| Formal local estimand | completed-Q under search | one deviation then frozen continuation | search value under NX |
| Main upside | current baseline | recover missed actions without replacing baseline | change the quality/throughput frontier |
| Main falsifier | existing plateau | A-EXACT infeasible or overrides fail fresh gate | factors invalidate broadly or fail teacher-regret/equal-wall gates |

The honest priority statement is:

- for a **bounded first serving test with literal incumbent fallback**, test
  Anchor first; its probability of a positive score gain is unknown;
- for **maximum plausible upside toward 100**, Anchor and NX remain uncertain,
  and NX may have the larger ceiling;
- neither should interrupt the already authorized D1 chain.

The old v2 gain of about half a point would not, by itself, close the
historical 1.6–1.7 point gap. Anchor’s value is that it can operate at more
decisions and use a stronger proposer/continuation—but those differences can
also erase the old effect. The route to 100 is empirical, not arithmetic
extrapolation from v2.

## 13. Fair bakeoff and falsifiers

### 13.1 Required controls

| Arm | Purpose |
| --- | --- |
| C0 | frozen current-rules incumbent \(\pi_I\), once a valid July-16 baseline exists |
| C1 | forced-anchor wrapper; must be action/RNG/score bit-identical to C0 |
| C2 | shadow Anchor computes decisions but never overrides |
| C3 | proxy screen followed by A-EXACT confirmation |
| C4 | confidence-gated selection versus mean-argmax at equal samples and wall |
| C5 | Anchor versus additional ordinary Gumbel simulations/worlds at equal wall |
| C6 | CPU/production-bridge versus GPU exact state and complete incumbent-policy traces |
| C7 | independent worlds versus current root coupling versus any new coupling |
| C8 | one wrapped seat versus three incumbents |
| C9 | four wrapped seats versus four incumbent seats; primary strength gate |

The historical v2 policy is a positive mechanism control and code reference,
not a current gameplay comparator.

### 13.2 Gate 0 — semantic validity

Before strength claims:

- zero CPU/GPU rule/scoring mismatches at the registered test scale;
- full legal-menu parity;
- forced-anchor bit identity;
- complete production-bridge parity for packed rows, numerical mode,
  Gumbel/rollout traces, market branches, RNG consumption, and nested actions;
- public-information/no-peek invariance;
- golden parity for no-refresh, decline, and accept/reveal/downstream-draft;
- exact sampler marginals by construction and audit;
- confidence-method coverage on synthetic bounded distributions;
- constructed equal/worse-action fixtures with measured false-override rate;
- deterministic replay and complete identity ledger.

Any failure blocks later gates.

### 13.3 Gate 1 — A-EXACT feasibility

On inspectable calibration roots, measure:

- required terminal pairs over the registered range/variance/margin/alpha
  grid;
- A-EXACT terminal pairs per second by remaining-turn stratum;
- GPU utilization and memory;
- batch occupancy/compaction loss;
- per-decision p50/p90/p99 latency;
- fraction resolved at each budget;
- fraction falling back for timeout;
- incremental cost per accepted override; and
- equal-wall quality versus more incumbent Gumbel compute.

The budget, horizon, and minimum useful override coverage are synthesized
engineering thresholds to freeze before the untouched block. If the budget
cannot resolve the smallest worthwhile effect, the correct result is frequent
fallback or closure—not a post hoc weaker guard.

### 13.4 Gate 2 — proxy fidelity

For A-DIRECT and A-DISTILL, measure against A-EXACT on disjoint roots:

- candidate recall;
- top-action agreement;
- selected-challenger agreement;
- completed terminal-advantage regret;
- calibration by phase and score card;
- error correlation with the incumbent;
- screen compute; and
- rate at which a proxy discards the eventual A-EXACT winner.

Proxy misses reduce upside. Proxy-only false positives do not matter if fresh
A-EXACT confirmation remains mandatory.

### 13.5 Gate 3 — untouched root audit

Freeze one wrapper and apply it in shadow mode to held-out roots sampled from
complete incumbent games. Use a third independent fixed-size audit stream for
every proposed override. Report:

- eligibility and override coverage;
- independently estimated mean advantage;
- fraction of overrides with positive audit advantage;
- harmful-override rate and upper confidence bound;
- gain contributed per encountered root;
- calibration of declared lower bounds;
- phase/card/score strata;
- timeout/fallback reasons; and
- compute per net expected point.

Cluster uncertainty by source game because roots from the same trajectory are
dependent. This is mechanism validation, not promotion evidence.

### 13.6 Gate 4 — full policy

After the wrapper is fixed, run the repository’s sanctioned fresh paired gate:

- at least 100 planned game pairs;
- current corrected rules identity;
- one frozen incumbent and one frozen wrapper;
- game-level mean-seat difference;
- 95% repeated CI excluding zero for promotion evidence;
- fixed final look or preregistered Lan–DeMets looks;
- no manual partial reads;
- complete raw per-game ledgers; and
- latency, override rate, one-seat results, and categories as secondary
  diagnostics.

Only John decides promotion.

### 13.7 Gate 5 — campaign target

After any promotion ruling, freeze the complete wrapper identity and run 1,000
fresh four-player games. Success remains:

> mean seat score at least 100 before habitat bonus.

Do not redefine the target around override accuracy, root regret, or a smaller
score gain.

## 14. Failure modes and explicit responses

| Failure | Consequence | Response |
| --- | --- | --- |
| A-EXACT nested search is unaffordable | exact-incumbent confirmation rarely resolves | late-game restriction, better batching, or close production wrapper |
| A-DIRECT/A-DISTILL policy mismatch | proxy ranks the wrong deviation | use proxy only to screen; quantify recall; A-EXACT confirms |
| candidate censoring | true improvement never reaches confirmation | incumbent survivor menu plus limited frozen diversity; measure A-EXACT reference regret |
| selection and confirmation reuse data | winner’s curse and false overrides | domain-separated fresh confirmation |
| unadjusted many-action bounds | familywise error inflation | select one challenger or simultaneous bounds |
| ordinary CI under repeated peeking | invalid coverage | fixed N, confidence sequence, or registered alpha spending |
| naive common worlds create negative covariance | higher gap variance | disjoint covariance calibration and independent fallback |
| outer hidden order leaks into \(\pi_I\) | oracle-biased policy | separate outer physical and inner search state/RNG |
| wrapper freezes a pre-reveal draft | illegal or oracle-conditioned action | v1 starts post-prelude; incumbent owns refresh; golden branch traces |
| GPU simulator drift | exact endpoint is false | Rust oracle, bit parity, fail closed |
| GPU policy trace diverges | continuation is \(\hat\pi_I\), not \(\pi_I\) | classify as proxy; no incumbent-relative override |
| forced-anchor changes RNG | fallback no longer equals incumbent | store \(a_0\); isolate domains; bit-identical parity test |
| all-seat adoption changes dynamics | local gains fail to compose | unilateral diagnostic plus symmetric game gate |
| repeated overrides shift states | shadow-root estimates become stale | full-policy gate; optional frozen mixture; re-audit new identity |
| guard is too conservative | policy ties incumbent but adds latency | coverage/compute gate; shadow-only or close |
| tiny positive local effects | cannot cover systems and interaction risk | practical margin \(\epsilon>0\) |
| stale checkpoint/rules/sampler | invalid historical inference | full identity hash and refusal |

## 15. Closed directions this proposal does not reopen

- **Generic rollout-RNG CRN (R0.2):** current physical root coupling is the
  control; additional coupling requires a new disjoint covariance proof.
- **Completed-Q LCB (R0.4):** Anchor’s fresh terminal-return confirmation is a
  different estimator. The failed plug-in completed-Q rule stays closed.
- **Ordinary deeper Gumbel (R3.2) and exact K2:** Anchor follows a frozen
  policy to terminal; it does not claim another own-turn depth knob is exact.
- **Chance-node leaf expectimax:** no learned leaf correction is being
  reintroduced.
- **Cooperative/table-total/risk serving:** terminal own raw score remains the
  objective.
- **Generic menu widening, pairwise Borda, structured-Q, symmetry TTA, static
  sigma/Q-bias tuning, ghost as a strength lever, smaller ordinary
  transformer plus more search, or another larger transformer:** none is
  smuggled into the first causal test.
- **Generic NNUE revival:** Anchor can use the current transformer unchanged.
  Cascadia-NX remains the separate structured-evaluator proposal.

## 16. Implementation sequence

### Phase A — freeze the contract

Write the exact \(\pi_I,\pi_R,\pi_W\) identities, post-prelude root boundary,
terminal estimand, candidate factory, horizon, RNG domains, inference method,
fallbacks, and evidence ladder. Add golden traces proving no-refresh,
decline, accept/reveal/draft, and later outer-chance/inner-search behavior.

### Phase B — requalify the historical mechanism as a control

Port the v2 anchor/terminal/fallback harness to the current rules identity
without claiming strength. Reproduce:

- exact anchor presence;
- terminal continuation semantics;
- paired differences;
- fallback;
- canonical redetermination;
- replay; and
- old archived fixtures where compatible.

This creates a trustworthy CPU reference for later GPU work.

### Phase C — minimal GPU exact engine

Port only the operations needed for late-game terminal completion:

- legal action application;
- committed-root draft application plus later market/prelude/chance
  transitions;
- scoring;
- terminal detection;
- policy interface; and
- deterministic RNG.

Do not begin with a general replacement for every engine path. Prove exact
late-game parity and completed-trajectory throughput first.

### Phase D — proxy continuations and batching

Implement A-DIRECT, then optionally A-DISTILL. Establish the outer wavefront,
transformer batching, compaction, and ledger path. Run shadow-only screens and
fidelity audits.

### Phase E — nested A-EXACT

Embed the frozen serving incumbent with separate internal search worlds.
Benchmark final-two through final-five strata. This is the decisive systems
falsifier.

### Phase F — fixed screen/confirm wrapper

Implement the fresh confirmation stream, bounded-mean interval, practical
margin, literal fallback, and forced-anchor mode. Validate false-override and
coverage fixtures.

### Phase G — untouched evidence ladder

Run semantic gate, feasibility gate, proxy-fidelity block, held-out shadow
root audit, equal-wall control, unilateral diagnostic, and then the symmetric
paired gameplay gate.

### Phase H — optional amortization

If the wrapper wins, its confirmed decisions may become an offline teacher for
distillation. A distilled student is a new policy and needs its own gate.
Production Anchor should retain the exact fallback until the student proves
otherwise.

## 17. Recommendation

Build **one bounded, late-game Cascadia-Anchor preflight after the authorized
D1 chain reaches its frozen boundary**, without competing with or inspecting
that live line.

The first preflight should answer only:

1. Can a GPU exact simulator reproduce current Rust late-game trajectories
   with zero registered mismatches?
2. Can full serving-incumbent A-EXACT continuations be batched cheaply enough to
   confirm final-two through final-five deviations?
3. Does a cheap A-DIRECT/A-DISTILL screen retain A-EXACT’s useful challenger often enough?
4. On untouched roots, do fresh terminal bounds admit a nontrivial set of
   overrides whose independent audit advantage is positive?
5. At equal wall, is the wrapper better than simply giving current Gumbel more
   simulations/worlds?
6. Do those local changes survive a fresh symmetric four-seat game gate?

If the answers are yes, Anchor is the most defensible next serving
architecture: it converts the current model from an all-or-nothing policy into
a strong incumbent with an exact terminal appeals court.

If A-EXACT is unaffordable, the proposal has still found the true boundary. A-DIRECT/A-DISTILL
may remain useful as offline teachers, shadow diagnostics, or components of
Cascadia-NX, but they must not inherit an improvement guarantee over the full
incumbent.

If the symmetric gate is null or negative, close the wrapper even if its local
confidence accounting is perfect. The project goal is score above 100, not an
elegant per-root certificate.

## 18. Primary-source ledger

### Rollout policy improvement

| Source | Evidence used |
| --- | --- |
| [Tesauro and Galperin, *On-line Policy Improvement using Monte-Carlo Search*](https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html) | Root-action terminal evaluation under a frozen base policy, backgammon decision-error reductions, and natural parallelism. |
| [Bertsekas and Castanon, *Rollout Algorithms for Stochastic Scheduling Problems*](https://doi.org/10.1023/A:1009634810396) | Rollout as one-step policy-iteration approximation and conditional “no worse than base heuristic” theory; assumptions do not transfer automatically. |
| [Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO) | Action sampling without replacement, sequential halving, and correctly-evaluated-Q policy-improvement context; already part of the incumbent. |

### Statistical guard and baseline fallback

| Source | Evidence used |
| --- | --- |
| [Thomas, Theocharous, and Ghavamzadeh, *High Confidence Policy Improvement*](https://proceedings.mlr.press/v37/thomas15.html) | Held-out safety evaluation and user-selected lower-bound/confidence principle under batch-RL assumptions. |
| [Laroche, Trichelair, and Tachet des Combes, *SPIBB*](https://proceedings.mlr.press/v97/laroche19a.html) | Reproduce the baseline in insufficiently supported regions; formal finite-MDP/batch assumptions do not cover Cascadia. |
| [Kaufmann, Cappé, and Garivier, *Best-Arm Identification*](https://www.jmlr.org/papers/v17/kaufman16a.html) | Fixed-budget versus fixed-confidence distinction and sequential stopping context. |
| [Karnin, Koren, and Somekh, *Almost Optimal Exploration in Multi-Armed Bandits*](https://proceedings.mlr.press/v28/karnin13.html) | Sequential Halving for fixed-budget screening, not a standalone safety certificate. |
| [Mnih, Szepesvári, and Audibert, *Empirical Bernstein Stopping*](https://icml.cc/Conferences/2008/papers/523.pdf) | Variance-adaptive stopping with probabilistic guarantees under the paper’s construction. |
| [Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*](https://doi.org/10.1214/20-AOS1991) | Anytime-valid bounded-mean inference for later sequential confirmation. |
| [Wu et al., *Conservative Bandits*](https://proceedings.mlr.press/v48/wu16.html) | Maintaining performance relative to a baseline while exploring; design principle rather than a transferred Cascadia theorem. |

### Variance and GPU execution

| Source | Evidence used |
| --- | --- |
| [Veness, Lanctot, and Bowling, *Variance Reduction in Monte Carlo Tree Search*](https://proceedings.neurips.cc/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html) | CRN/control-variate conditions and measured simulation-equivalent gains in stochastic games. |
| [Yadav et al., *Using Common Random Numbers for Simulation-based Planning with Rollouts*](https://arxiv.org/abs/2605.04732) | Recent depth-dependent rollout coupling theory and counterexamples to naive full coupling; arXiv-v1/RLJ-2026-reference and transfer caveats. |
| [Koyamada et al., *Pgx*](https://papers.nips.cc/paper/2023/file/8f153093758af93861a74a1305dfdc18-Paper-Datasets_and_Benchmarks.pdf) | Accelerator-resident board-game simulation and reported 10–100x throughput over compared Python environments. |
| [DeepMind Mctx](https://github.com/google-deepmind/mctx) | Reference accelerator-native batched tree-search implementation. |

## 19. Repository context

- [v3 source of truth](docs/v3/README.md)
- [live campaign state](docs/v3/CAMPAIGN_STATE.md)
- [current architecture](docs/v3/ARCHITECTURE.md)
- [research log and closed directions](docs/v3/RESEARCH_LOG.md)
- [living research agenda](docs/v3/RESEARCH_AGENDA.md)
- [radical directions](docs/v3/RADICAL_DIRECTIONS.md)
- [current Gumbel implementation](cascadiav3/real-root-exporter/src/gumbel.rs)
- [historical conservative policy-improvement implementation](crates/cascadia-search/src/policy_improvement.rs)
- [previous stochastic-game architecture proposal](stochastic_board_game_ai_architecture_research_7_16.md)
- [Cascadia Foundry original architecture proposal](cascadia_foundry_original_architecture_proposal_7_16.md)
- [July 16 research question brief](research_questions_7_16.md)
- [July 16 detailed answers](research_answers_7_16.md)
- archived v2 decisions:
  `docs/archive/v2/decisions/0023-confidence-gated-terminal-policy-improvement.md`,
  `0024-confidence-gate-original-terminal-frontier.md`, and
  `0068-canonical-redetermination-strong-requalification.md` at tag
  `archive/pre-v3-repo-cleanup-2026-07-01`

Recovery examples:

    git show archive/pre-v3-repo-cleanup-2026-07-01:docs/archive/v2/decisions/0024-confidence-gate-original-terminal-frontier.md
    git show archive/pre-v3-repo-cleanup-2026-07-01:docs/archive/v2/decisions/0068-canonical-redetermination-strong-requalification.md
