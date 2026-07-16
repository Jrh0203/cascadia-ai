# Cascadia Rival: finalized adversarial architecture for breaking 100

**Date:** 2026-07-16

**Literature cutoff:** 2026-07-16

**Status:** FINAL RESEARCH PROPOSAL; post-D1; zero current-rules strength
evidence; no experiment launch, rules change, or promotion authorization

**Objective:** mean seat score at least 100 over 1,000 complete four-player
games, before habitat bonus, under one pinned rules and serving identity

**Policy class:** four isolated, seat-relative, explicitly non-cooperative
agents; every acting seat maximizes its own expected raw terminal score

**Present target-reaching forecast:** approximately 25--35%, conditional only
on information available at this report's cutoff; this is a subjective design
forecast, not a confidence interval or strength result

## Executive verdict

The final proposal should not average the three July 16 proposals together.
That would multiply their failure modes. It should assign each surviving idea
one narrow role, preserve the strongest policy already built, and make exact
terminal endpoints estimate the expected own-score action difference.

The recommended system is **Cascadia Rival: incumbent-anchored adversarial
multifidelity terminal rollout iteration**.

At each eligible research root, the `W_k` appeals instrument does five things:

1. computes and stores the action of the frozen transformer-plus-Gumbel
   incumbent;
2. uses a much cheaper, structured afterstate policy to search broadly and
   propose one challenger;
3. uses exact-rules GPU continuations to compare challenger and incumbent in
   active-seat terminal score;
4. corrects abundant cheap-continuation evidence with a smaller, paired set of
   full-incumbent continuations through a preregistered multifidelity control
   variate; and
5. plays the incumbent exactly whenever the corrected lower bound does not
   clear a practical margin, any identity check fails, or the computation does
   not resolve.

Rival v1 is offline-first: `W_k` is a shadow/one-seat research and labeling
instrument, not the promotion candidate. Confirmed corrections become a new,
hash-pinned tranche for **frozen rollout-informed terminal relabeling**:

```text
ordinary base B_k
    -> terminally confirm one-deviation preferences under B_k
    -> train one ordinary candidate M_(k+1)
    -> fresh paired full-game gate against B_k
    -> promote only by John's ruling
    -> repeat only after a real positive gate
```

This is the combined design I judge most credible for **literal fallback and
bounded activation exposure now**, with whole-policy downside determined only
by gameplay, and for a path beyond the roughly half-point historical Anchor
effect later. It uses:

- [Cascadia-Anchor](incumbent_anchored_gpu_rollout_policy_improvement_7_16.md)
  for the incumbent action, terminal one-deviation estimand, fresh
  confirmation, and literal fallback;
- [Cascadia-NX](stochastic_board_game_ai_architecture_research_7_16.md) for an
  exact semantic compiler, a cheap D6-tied afterstate policy, and an
  accelerator-resident rules/search boundary; and
- [Cascadia Foundry](cascadia_foundry_original_architecture_proposal_7_16.md)
  only for single-seat score contracts, chronology-aware tomography,
  nonanticipativity tests, and commitment-failure diagnostics.

It categorically rejects Foundry-Commons. There is no table-total reward,
market donation, four-board genome, cross-seat resource exchange, shared
memory, shared lineage, coordinated capsule, or sacrifice of one seat for
another. The earlier `76%` Foundry forecast was explicitly conditional on the
cooperative Commons policy class and is withdrawn from this final proposal.

The new ingredient is **Rival-MF**, an application of multifidelity Monte
Carlo to incumbent-relative terminal action differences. A cheap continuation
is not renamed as the incumbent. Instead, its measured correlation with the
full-incumbent return is used as a control variate, while occasional true
full-incumbent continuations preserve the high-fidelity estimand. Published
multifidelity Monte Carlo can produce an unbiased estimate of high-fidelity
statistics using many cheap and occasional expensive evaluations; published
multifidelity RL applies the same correlation principle to state-action
returns. Neither source proves the Cascadia construction, its coupling, or its
confidence procedure. Those are explicit falsifiers here.

The [independent critique](architecture_proposal_critiques_7_16.md) ranked
Anchor above NX above Foundry and recommended extracting tomography. This
proposal accepts the ranking, rejects the critique's cooperative carve-out at
John's direction, and adds one tighter synthesis:

> Anchor is the policy-improvement spine. NX must earn the role of a cheap,
> correlated continuation rather than immediately replace the transformer.
> Foundry is reduced from a controller to a seat-local proposal and diagnostic
> language. Multifidelity correction is the bridge between cheap breadth and
> incumbent-fidelity terminal evidence.

The honest forecast is not 75%. Before a valid July-16 baseline, selfish
headroom, low/high-fidelity correlation, exact GPU parity, throughput, and a
fresh gameplay gate exist, I assign roughly:

- **45--55%** that an ordinary `M` candidate produces some CI-positive gain
  over its valid `B` base;
- **25--35%** that an ordinary frozen `M` policy has true current-rules mean at
  least 100 within the finite scope defined in §20.1; and
- **55--65%** only after the major premise gates pass and independent shadow
  evidence projects enough gain to cover the measured gap.

Anything above 75% becomes defensible only after direct gameplay evidence
places the frozen policy's latent mean safely above 100. At that point the
confidence comes from the result, not the architecture diagram.

This proposal does not reorder the authorized D1 chain. At the report cutoff,
john0 remained unreachable, attempt 5's liveness was unknown, no partial D1
scientific output had been read, and no admissible July-16 canonical baseline
existed. Rival begins only after D1 reaches its frozen boundary.

## 1. Decision boundary and evidence labels

### 1.1 What “final” means

“Final” means this is the recommended synthesis of the three architecture
proposals and their critique. It does not mean the architecture is proven,
funded ahead of D1, implemented, or promoted.

The document fixes:

- the non-cooperative policy class;
- which ideas survive and which are discarded;
- the proposed load-bearing architecture;
- the sequence of cheap falsifiers;
- the exact evidence needed to raise confidence; and
- the point at which the proposal must be closed.

Any material change to the objective, high-fidelity continuation, low-fidelity
policy, world coupling, inference rule, or deployment composition creates a
new proposal identity.

### 1.2 Evidence labels

This report uses six labels.

- **Identity:** follows from the pinned policy/rules definition.
- **Published-direct:** a primary paper or official project report measured or
  proved the stated result under its own assumptions.
- **Repo-direct:** the repository directly contains the result under the named
  historical identity.
- **Transfer:** a demonstrated mechanism whose Cascadia effect is unknown.
- **Original synthesis:** a construction proposed here. It is not a published
  Cascadia algorithm or current strength evidence.
- **Forecast:** an explicit subjective probability range, never promotion
  evidence.

The report keeps literature, repository results, and synthesis separate. No
located paper studies this exact game, rules identity, opponent distribution,
hardware, or score gate.

## 2. The adversarial contract

### 2.1 Competitive, general-sum, and selfish

Cascadia under this research identity is competitive but not constant-sum.
One seat's score can rise without another seat's score falling, yet seats
compete for the same public market and finite supply.

For active seat `i`, frozen continuation policy `pi`, public draft state `s`,
and isolated public-history-derived memory snapshots `m_0..m_3`, define the
complete rollout state:

\[
x=(s,m_0,m_1,m_2,m_3).
\]

For a stateless policy the corresponding `m_j` is empty. Rival estimates:

\[
Q_i^\pi(x,a)=
\mathbb E\left[
G_i
\mid X=x, A_i=a,
\text{every later seat }j\text{ uses its own instance of }\pi
\right].
\]

`G_i` is seat `i`'s exact terminal raw score before habitat bonus. Every future
seat executes its complete frozen seat-relative serving rule, whose utility
coordinate is its own terminal score. `max^n` describes the vector-backup
semantics inside applicable search components; it does not imply that an
approximate or stochastic serving rule literally takes an exact argmax at
every node. This is not a centralized team controller and not paranoid
minimax against a three-seat coalition.

Symmetric evaluation still reports:

\[
J_{\mathrm{sym}}(\pi)=
\mathbb E\left[\frac14\sum_{i=0}^{3}G_i\right].
\]

That arithmetic identity is an evaluation aggregation. It does not enlarge
the permitted policy class. Four independently acting copies of one policy
are not equivalent to one controller that can coordinate their memories and
actions.

### 2.2 Categorical exclusions

The following mechanisms are removed, not deferred:

- table-total or table-mean action utility;
- explicit cooperation or resource donation;
- a joint four-board completion genome;
- cross-seat scarcity prices used to allocate items between controlled seats;
- shared plans, lineages, archives, or persistent memory across seats;
- seat sacrifice to improve average table score;
- a centrally elected four-seat capsule;
- opponent score minimization as a surrogate for own score; and
- the cooperative `76%` Foundry forecast.

Every policy instance has isolated memory. The same weights may be instantiated
for each seat, but each instance receives a seat-relative public state and may
store only its own public-history-derived plan state.

### 2.3 What opponent modeling is allowed to do

Opponent boards and public histories remain important. They predict:

- which market pair is likely to disappear before the active seat acts again;
- which species, terrain, and tile roles face high demand;
- how a draft changes the active seat's later option set;
- how an opponent's selfish action changes the public market; and
- whether a denial action improves the active seat's own expected terminal
  score.

They may not contribute utility merely because an opponent loses points. A
denial move is valid only through its causal effect on the acting seat's own
future score.

### 2.4 Information boundary

All policies and proposal generators are public-information policies.

- A future hidden order is private to the simulator.
- A real or simulated policy never receives a scenario ID, hidden suffix,
  future wipe/return schedule, or physical seed.
- Identical public histories and policy identities must produce identical
  policy memory and action distributions.
- A counterfactual action creates its own subsequent public history and
  branch-specific memory update for each isolated seat; memory is never copied
  from the incumbent branch after histories diverge.
- The free three-of-a-kind branch remains public decision, chance reveal, then
  public draft.
- A plan may react after a reveal; it may not precommit a draft that depends
  on an unrevealed replacement.

Foundry's scenario-braid action-hash test survives because it is a strong
no-peek audit. Foundry's centralized plan population does not.

## 3. Measured constraints the architecture must explain

The design is governed by repository evidence, not by analogy alone.

| Repo-direct fact | Architectural consequence |
| --- | --- |
| Historical July-9 scalar mean was 98.2975; no admissible July-16 canonical mean existed at cutoff. | Do not size the remaining gap or forecast from 98.2975 as though it were current truth. |
| Historical decision audit: median SNR about 1.06; roughly 46% of decisions noise-flippable. | Estimate action differences and terminal consequences; absolute value aesthetics are secondary. |
| Historical July-9 n4096/d16: only +0.21, nonsignificant, at about 3.1x decision cost. | More ordinary simulations on the same evaluator/search axis cannot be the main payoff story. |
| Historical root-estimation class: 0-for-4. | Do not revive static sigma, generic CRN, Q-bias, or completed-Q LCB under new names. |
| Pre-July-16 D1 pilot/full-ledger mechanism audit: the teacher moved 43.2--43.6% of repeat-stable hard labels with about 0.36--0.40 moved-root regret. The two percentages are stages of one program, not independent replications, and label movement is not correctness or strength. | Hard-root target correction was the strongest funded training-mechanism signal at cutoff; D1 remains first, with no current-rules gameplay evidence. |
| Historical v2 incumbent-anchored terminal rollout gained +0.420 and +0.520 with positive intervals. | Terminal one-deviation policy improvement has the only direct Cascadia gameplay precedent among the proposals. |
| Historical July-9 exact K1: about 29x faster and score-neutral. | Preserve exactness, but do not assume exact deeper horizons automatically add strength. |
| Archived pre-v3 partial accumulator: about 2.5x faster and about 3 points weaker. | Incremental dependencies require full recomputation parity and an economics kill test. |
| Historical capacity/data/ensemble probes: a 207M transformer, more ordinary data, fresh training, and checkpoint ensembles were flat. | Do not answer with another larger transformer or output ensemble. |
| Historical john0 CUDA probe: a smaller ordinary transformer was only about 1.9--2.0x faster. | The low-fidelity path must change the computation boundary and deliver several-fold trajectory throughput. |
| Historical bridge audit: current bridge within roughly 5% of its architectural ceiling. | A real speed breakthrough requires a resident simulator/search process, not another bridge micro-optimization. |
| July-16 operational incident: 24 CUDA contexts showed 100% utilization while completing zero seeds. | Completed terminal comparisons per wall-second, not utilization, is the systems metric. |
| Historical table-total serving: losses of about 1.65 and 1.05 points in two variants. | Cooperative serving remains scientifically closed and is now outside the allowed policy class. |

Except for the July-16 operational incident, these are mechanism evidence
under older rules, policy, or source identities. None is July-16 strength
evidence. Together they make the proposal narrower than a general “GPU NNUE
plus more search” story. Rival must create a new estimator and training signal:
full-policy terminal one-deviation value, made affordable through a correlated
low-fidelity continuation.

## 4. What survives from the three proposals

| Source | Retained | Discarded or demoted |
| --- | --- | --- |
| Current v3/D1 | exact legal menus, exact afterstate grounding, Gumbel candidates, `max^n`, K1, current checkpoint, D1 correction | treating current completed-Q as terminal truth; assuming more of the same search crosses 100 |
| Anchor | incumbent action, one-deviation estimand, post-prelude boundary, terminal own score, split screen/confirm, literal fallback, paired full-game gate | full-incumbent continuation assumed affordable at every root; proxy continuation mislabeled as incumbent; local safety treated as whole-policy safety |
| NX | exact semantic compiler, D6-tied factor path, legal-action hyperedges, small global correction, GPU-resident exact rules, afterstate/chance separation | immediate wholesale replacement of the transformer; pure “saved compute becomes more ordinary sims”; blind sparse incrementality |
| Foundry | single-seat score witnesses, completion slack, chronology-preserving replay, public-history audit, commitment-collapse diagnostics | Commons, Futures Exchange, donation, joint genome, DSL as production policy, archive-support vote, persistent central capsule, 76% forecast |
| July-16 critique | Anchor-first ordering, NX economics preflight, tomography before large build, conjunction-risk warning | the cooperative tomography/Commons route, rejected by the user |

## 5. System overview

```text
                   W_k research root / public state
                                  |
                   +--------------+--------------+
                   |                             |
            frozen incumbent pi_I       exact semantic compiler
                   |                             |
             incumbent action a0          RivalNet low fidelity
             incumbent survivors        + seat-local intent features
                   |                             |
                   +----------- candidate union-+
                                  |
                         cheap terminal screen
                                  |
                      freeze one challenger a*
                                  |
             +--------------------+--------------------+
             |                                         |
      many low-fidelity                         smaller paired set
      exact-rule returns                       of low + full-incumbent
             |                                         |
             +----------- Rival-MF correction --------+
                                  |
                     fresh lower bound on Delta_H
                                  |
                 +----------------+----------------+
                 |                                 |
          LCB > epsilon                    unresolved / invalid
                 |                                 |
       W_k trial plays a*              W_k trial plays a0
                 |
    fixed-weight categorical label
                 |
         train ordinary M_(k+1)
                 |
       M_(k+1) paired full-game gate
```

The architecture has six load-bearing layers:

1. a frozen high-fidelity incumbent;
2. an exact semantic state/action compiler;
3. a cheap, correlated RivalNet continuation policy;
4. an exact accelerator-resident simulator;
5. a multifidelity terminal appeals court; and
6. a gated terminal-relabel loop producing an ordinary policy.

Single-seat score contracts and adversarial probes are candidate/diagnostic
plugins. The architecture must function without them. They enter only after an
ablation proves they add confirmed challengers at useful natural frequency.

## 6. Layer 1: the high-fidelity incumbent

Let `pi_I` be one frozen, complete serving identity:

- checkpoint and numerical precision;
- exact legal-action construction;
- Gumbel configuration and candidate rules;
- exact-K1 and refresh behavior;
- root-world sampler and RNG domains;
- opponent/interior policy;
- source and rules revision; and
- all fallback and timeout behavior.

If D1 wins its registered gate and John promotes it, the promoted D1 policy
becomes the initial Rival incumbent. If D1 fails or is not promoted, Rival uses
the then-current valid incumbent. It never assumes D1 success.

At every eligible real root, Rival computes and stores:

```text
a0 = pi_I(public_state)
```

Forced-anchor mode must reproduce the incumbent's complete action, RNG,
refresh, replay, and score trace. A fallback is `a0`, not the highest currently
sampled challenger.

The current transformer is therefore not discarded. It serves as:

- the mandatory action anchor;
- the high-fidelity continuation in terminal correction;
- a root candidate and global-context expert;
- a teacher for the cheap policy; and
- the policy to which unresolved states revert.

This is a systems architecture around the transformer, not another claim that
an 88M model should be invoked at every simulated leaf forever.

### 6.1 Base, wrapper, and distilled identities

Three complete policies must never be conflated:

- `B_k`: the ordinary frozen transformer-plus-Gumbel base at iteration `k`;
- `W_k`: the online Rival appeals wrapper around `B_k`; and
- `M_(k+1)`: the ordinary model trained from iteration `k` corrections.

The high-fidelity continuation in iteration `k` is always `B_k`. It is not
`W_k`. Recursively simulating `W_k` would nest a new appeals court at every
future decision, changing both cost and estimand.

Rival v1 chooses the offline-first contract: `W_k` is a shadow, root-labeling,
and one-seat mechanism instrument; `M_(k+1)` is the sole symmetric promotion
and target candidate. If `M_(k+1)` wins and John promotes it, it may become
`B_(k+1)`. Online wrapper deployment is a separate future proposal requiring
its own latency, multiplicity, symmetric gameplay, and recursive-continuation
contract. Promotion of a wrapper could never silently turn it into a base
policy.

## 7. Layer 2: exact semantic compiler

The rules engine compiles public state and legal actions into stable integer
features. Each feature declares its complete dependency footprint.

### 7.1 Own-board factors

- exact current wildlife, habitat, and Nature score;
- Bear pair/group status and invalid third-Bear risks;
- Elk line candidates, axes, lengths, and conflicts;
- Salmon endpoints, degree constraints, branches, and cycles;
- Hawk isolation zones;
- Fox neighbor diversity and missing-type sets;
- habitat components, open ports, merge opportunities, boundary capacity, and
  articulation-like bottlenecks; and
- turns remaining, phase, exact breakpoints, and reachable obligation slack.

### 7.2 Public resource and opponent-pressure factors

- visible tile/wildlife market pairs;
- observable remaining supply;
- free-refresh, automatic-wipe, paid-wipe, and Nature affordances;
- opponent public boards, current patterns, turns to next access, and likely
  demand;
- predicted pair survival until the active seat's next turn; and
- action-conditioned replacement and depletion summaries.

These features estimate pressure on the active seat's future resources. They
do not add opponent score to or subtract it from utility.

### 7.3 Legal-action hyperedges

Each exact legal action records:

- market pair and refresh/prelude identity;
- tile coordinate and rotation;
- wildlife coordinate or decline;
- Nature expenditure;
- exact afterstate score and category deltas;
- changed feature IDs and affected components; and
- intent obligations advanced, preserved, repaired, or destroyed.

### 7.4 Correctness and economics gate

The compiler must show zero mismatch against slow full recomputation under
registered property/exhaustive tests, backed by by-construction invariants,
over:

- randomized complete legal games;
- every A-card rule and breakpoint;
- all D6 rotations and reflections;
- long Salmon components and cycles;
- Bear regrouping, Fox neighborhoods, Elk alternatives, and Hawk exclusions;
- large habitat merges;
- no-refresh, decline, accept/reveal/draft, paid wipe, chained automatic wipe,
  wildlife return, and Nature transactions; and
- radius-6 fast-path and exact overflow states.

Before training, measure factor cardinality, memory, and median/p95/max
invalidation fanout. Benchmark three implementations at complete-menu and
complete-trajectory level:

1. fixed-width dense recomputation;
2. component-local delta recomputation; and
3. fully incremental accumulators.

Use whichever is fastest under exact parity. “NNUE-inspired” is not permission
to force sparse updates when dense GPU recomputation wins.

## 8. Layer 3: RivalNet, the low-fidelity policy

### 8.1 Role, not parameter count, defines RivalNet

RivalNet is a deliberately rolloutable, seat-relative policy/value model. Its
first job is not to replace CascadiaFormer. Its jobs are to:

- propose challengers the incumbent did not retain;
- execute millions of cheap, public-information continuation decisions;
- preserve useful correlation with full-incumbent action differences;
- expose phase/commitment failures; and
- provide a low-fidelity control variate.

The most important metric is therefore not validation loss alone. It is the
frontier of:

```text
complete trajectories per second
    versus
correlation with full-incumbent paired terminal action differences
```

An imperfect cheap policy can be valuable if it is strongly correlated with
the high-fidelity return. A superficially accurate model can be useless if its
errors reverse the exact challenger-versus-incumbent differences.

### 8.2 Proposed shape

The first bounded family is:

```text
D6-tied motif/component embeddings
        |
8 x 32-or-64 dimensional sparse/dense accumulators
        |
exact global summaries + market/supply/opponent-pressure summaries
        |
256 -> 128 -> 64 phase-conditioned trunk
        |
legal-action query / hyperedge head
        |
policy + own score-to-go + four-seat diagnostics + auxiliaries
```

Start in the 1--8M parameter range and select by measured economics. The exact
width is not a claim.

A small component/action graph is an ablation, not a mandatory layer. Add it
only if a fixed-width summary model misses global topology and the graph's
terminal-correlation gain exceeds its trajectory cost. The archived
geometry-only GNN result is a direct warning against making a graph the whole
representation.

### 8.3 Outputs

Load-bearing outputs:

- legal-action policy;
- active-seat own score-to-go mean; and
- rollout action under the frozen RivalNet identity.

Serving identity remains:

```text
exact_afterstate_score_active + predicted_own_score_to_go
```

Auxiliary-only outputs:

- final four-seat score vector;
- category score components;
- opponent next-draft and market-pair survival;
- score-contract reachability/slack;
- phase and breakpoint completion;
- rollout/high-fidelity disagreement; and
- predicted value of additional computation.

Category outputs do not become the failed additive structured-Q serving head.
Four-seat outputs do not become table utility. Distribution heads do not
become risk-sensitive serving under a mean-score objective.

### 8.4 Complete low-fidelity policy identity

`pi_L` is a complete serving rule, not merely a value network:

- the exact engine performs forced automatic wipes and every mandatory chance
  transition;
- for a free three-of-a-kind accept/decline decision, `pi_L` estimates the two
  public actions with a frozen, low-cost set of internally redetermined
  replacement samples, commits accept or decline without seeing the physical
  replacement, and drafts only after any accepted replacement is public;
- on every visible market it scores the complete exact legal compound-action
  menu, including Nature spending, paid wipes where legal, placement,
  rotation, wildlife placement, and decline choices;
- it uses the incumbent's adopted exact-K1 rule on a final personal turn;
- its sample counts, temperature or argmax rule, canonical tie break, policy
  RNG domains, timeout, and fallback are part of the hash-pinned identity; and
- it never calls the incumbent inside a nominal low-fidelity continuation.

If a future fast policy delegates refresh or any other branch to `B_k`, it is
a new hybrid fidelity with its own measured cost, estimand, and calibration.
The complete policy, not only its neural weights, is what defines `D_L`.

### 8.5 Fast/full variants

RivalNet may have:

- a fast local/summary path for every simulated state; and
- a small global correction for roots or survivors.

Whenever fidelity changes, every surviving action is rescored at the same
fidelity before comparison. Never compare a stale fast value for one action
with a corrected value for another.

The first bakeoff includes full-correction-on-full-menu as the retention
oracle. Survivor routing remains disabled unless the fast path almost never
drops an action the full path would rescue.

### 8.6 Graduation bars

Freeze exact bars before held-out outputs are inspected. A pre-port power and
cost study derives the maximum affordable cost per resolved root, required
roots/hour, total local GPU-hours, and the low/high sample allocation
`n_H*c_H + n_L*c_L`. Relative speedups cannot fund a system that still takes
minutes per root.

Proposed starting bars, explicitly engineering syntheses rather than
literature constants, are:

- complete low-fidelity trajectory throughput of at least the greater of 5x
  the full-incumbent rate and the power-derived absolute rate;
- a preregistered lower confidence bound near 99% on the fraction of untouched
  roots for which the top-8 candidate union contains at least one independently
  terminal-confirmed challenger with high-fidelity advantage above `epsilon`;
- teacher-regret degradation no worse than a small preregistered bound such as
  0.02;
- stable, materially nonzero low/high paired terminal-difference correlation
  in every enabled stratum; and
- a measured multifidelity effective-sample gain of at least 3x at equal wall.

The 99% recall target is an intentionally severe aspiration, not a literature-
or repo-supported expectation; the existing completed-Q candidate-coverage
audit was much lower and used a different reference. “Effective-sample gain”
means the equal-wall ratio of high-fidelity-only variance or squared
confidence width to Rival-MF variance or squared confidence width, with the
same selected-challenger distribution and valid coverage.

Stable negative correlation is usable by a negative control coefficient;
variance reduction depends on squared correlation. RivalNet's separate
candidate-proposal role still requires high useful-challenger recall. A sign
that changes between calibration and held-out data is instability, not useful
negative correlation.

If the global correction erases speed, dependency closure becomes board-wide,
or terminal-difference correlation is weak, RivalNet does not graduate as the
control variate. A direct transformer policy, a simpler distilled policy, or
closure of the multifidelity route becomes the honest result.

## 9. Seat-local score contracts: oracle and proposer, not controller

Foundry's strongest surviving observation is that Cascadia boards are partial
constraint objects. Its weakest bet was making a population of those objects
the deployed policy.

Rival retains a much smaller object:

```rust
struct SeatContract {
    rules_id: Hash,
    own_terminal_board: PackedBoard,
    exact_own_score: ScoreBreakdown,
    placement_dag: BitMatrix,
    remaining_resources: ResourceVector,
    substitutions: Vec<ResourceDomain>,
    obligations: Vec<OwnScoreObligation>,
    deadline_slack: Vec<i8>,
    repair_operators: Vec<SeatLocalRepair>,
    d6_hash: Hash,
}
```

### 9.1 Allowed uses

Single-seat contracts may:

- prove that an own-board spatial construction scores a stated amount;
- expose exact Bear, Elk, Salmon, Hawk, Fox, habitat, and Nature obligations;
- measure completion slack, substitution count, and fragility;
- propose an action that preserves several high-scoring own-board futures;
- produce auxiliary targets and deterministic features for RivalNet;
- diagnose where an incumbent action collapses feasible own-score plans; and
- support static and chronology-preserving single-seat tomography.

### 9.2 Forbidden uses

Contracts may not:

- allocate one physical resource between controlled seats;
- ask another seat to leave a pair for the active seat;
- share memory or prices across policy instances;
- vote directly for the real action by archive population;
- replace stochastic acquisition with a spatial feasibility claim;
- treat a known future as public; or
- certify policy value from terminal-board score alone.

### 9.3 Own-resource shadow features

For active seat `i` and resource `r`, an optional diagnostic is:

\[
\lambda_{i,r}=
\Phi_i(\mathcal P_i)
-
\Phi_i(\operatorname{Repair}(\mathcal P_i,r^-)).
\]

`Phi_i` is a frozen, seat-local portfolio score. `lambda` says how much the
active seat's represented plans depend on that resource. It is an input or
candidate feature, never a certified point value and never a cross-seat bid.

Opponent demand can be estimated separately as the probability that resource
`r` disappears before seat `i` acts again. Combining own dependence with
public opponent demand helps forecast scarcity while leaving utility strictly
seat-local.

### 9.4 Admission rule

The contract module remains off by default. On held-out natural roots, it must
add challengers that:

1. are absent from both incumbent and RivalNet candidate sets;
2. survive exact legality and chronology checks;
3. receive positive terminal confirmation under the frozen policy identity;
4. occur often enough to matter at natural frequency; and
5. improve equal-wall candidate recall or gameplay.

If it only generates beautiful but acquisition-impossible boards, if its
repair cost dominates, or if its proposals never survive terminal
confirmation, remove it. Rival does not protect Foundry novelty.

## 10. Rival-MF: multifidelity terminal action differences

### 10.1 High- and low-fidelity estimands

At a frozen post-prelude public rollout state `x=(s,m_0..m_3)`, let:

- `a0` be the exact incumbent action;
- `a` be one frozen challenger;
- `pi_H = B_k = pi_I` be the complete ordinary serving-base continuation; and
- `pi_L` be RivalNet or another frozen cheap continuation.

For a valid physical scenario and policy-randomness bundle `omega_j`, define:

\[
D_{H,j}(a)=
G_i(x,a,\omega_j;\pi_H)
-
G_i(x,a_0,\omega_j;\pi_H),
\]

and

\[
D_{L,j}(a)=
G_i(x,a,\omega_j;\pi_L)
-
G_i(x,a_0,\omega_j;\pi_L).
\]

The target remains:

\[
\Delta_H(a)=\mathbb E[D_H(a)],
\]

the one-deviation advantage followed by the full incumbent. `Delta_L` is not
substituted for it.

The shorthand `omega_j` contains two strictly separated layers:

- `omega_phys,j`: the outer physical scenario used to complete the game; and
- `omega_policy,j,t`: fresh internal policy/search randomness at future public
  decision `t`.

At every future high-fidelity decision, `B_k` receives only the public state
and its isolated public-memory snapshot. It redeterminizes hidden information
and samples its Gumbel/search worlds from the inner policy domain; it never
observes the outer scenario's hidden inventory, order, or random keys. Inner
policy randomness is independent across candidate actions and fidelities in
v1. Coupling it is a separate, disjointly calibrated intervention and defaults
off because the repository's added rollout-policy coupling worsened variance.

### 10.2 Fixed multifidelity estimator

Use two fresh confirmation panels after the challenger is frozen:

- `H`: `n_H` paired high/low evaluations; and
- `L`: `n_L` additional low-fidelity evaluations, independent of `H`.

With control-variate coefficient `beta_cv` fixed on a disjoint calibration
block:

\[
\widehat\Delta_{MF}
=
\overline D_H^{H}
+
\beta_{cv}
\left(
\overline D_L^{L}
-
\overline D_L^{H}
\right).
\]

Both low-fidelity means have the same expectation, so their expected
difference is zero under the frozen sampler and policy identities. The
high-fidelity target is retained while correlation can reduce variance.

For exactly the independent-panel estimator above, its variance is:

\[
\operatorname{Var}(\widehat\Delta_{MF})=
\frac{\sigma_H^2}{n_H}
+\beta_{cv}^2\sigma_L^2
\left(\frac{1}{n_H}+\frac{1}{n_L}\right)
-\frac{2\beta_{cv}\sigma_{HL}}{n_H}.
\]

The corresponding population-optimal coefficient is:

\[
\beta_{cv}^*=\frac{n_L}{n_H+n_L}
\frac{\operatorname{Cov}(D_H,D_L)}
     {\operatorname{Var}(D_L)}.
\]

Under the same equal-low-variance assumptions, the minimum is:

\[
\operatorname{Var}(\widehat\Delta_{MF};\beta_{cv}^*)
=\frac{\sigma_H^2}{n_H}
\left(1-\rho^2\frac{n_L}{n_H+n_L}\right).
\]

The familiar `1 - rho^2` limit is approached only when the extra low-fidelity
mean is effectively known (`n_L` much larger than `n_H`). If paired and extra
low panels have different variances or sampling laws, use the general variance
optimization instead of this simplified coefficient.

Published multifidelity RL derives an ideal high-fidelity sample reduction
factor involving `1 - rho^2`, where `rho` is low/high return correlation.
That is mechanism evidence, not a Cascadia promise. Finite-sample error,
action pairing, dynamic trajectories, bounded inference, and cost ratios all
matter here.

### 10.3 Why this is better than calling a proxy “exact”

A-DIRECT and A-DISTILL answer different policy questions from A-EXACT. Rival
does not hide that difference.

The cheap policy contributes only through a zero-mean correction whose
coefficient and allocation were frozen independently. Every enabled
Rival-MF stratum still contains true full-incumbent continuations. If
correlation is weak, the coefficient goes to zero and the method becomes
ordinary high-fidelity Monte Carlo. If the remaining high-fidelity sample
count is unaffordable, the local incumbent-relative serving claim closes.

### 10.4 Dynamic-urn physical coupling

Low and high continuations must each have the exact conditional marginal game
distribution. A complete fixed suffix is insufficient when policies create
different wipe, return, and draw histories.

The proposed simulator-private coupling assigns a stable unique physical ID to
every tile and wildlife token. Counter-based continuous uniforms are keyed by
registered chance-event domain, event index, and physical item ID. At a draw,
the currently eligible physical ID with smallest fresh priority wins.
Sequential draws use new event indices; returned wildlife becomes eligible at
a later fresh index and never reuses an exposed key. Separate domains cover
tile draws, wildlife draws, wipes/returns, physical scenarios, policy actions,
and search randomness. A future event index is selected from public transition
semantics without inspecting any uniforms assigned to that index.

This construction is intended to make the argmin uniform conditional on every
reachable inventory and fresh indices produce the correct sequential
without-replacement law. It may induce useful cross-policy covariance, but it
does **not** earn an exact-marginal claim until that conditional argument is
written down for every chance event and checked against the oracle.

This construction is a proposal until proven. Required tests are:

- a proof that argmin over currently eligible unique physical IDs is uniform
  conditional on every reachable history;
- a proof that fresh, unobserved event keys give the correct sequential
  without-replacement law and returned tokens never reuse exposed keys;
- no event-key reuse and identical low-fidelity marginals in paired `H` and
  extra `L`;
- exhaustive small-bag enumeration and independent-world replication;
- equality with the Rust distribution on randomized reachable states;
- correct behavior under declined wildlife, voluntary wipes, chained
  overpopulation, exclusions, and Nature actions;
- no policy access to scenario uniforms or hidden inventories, with inner
  policy/search randomness independent by default; and
- disjoint calibration showing lower fixed-wall action-difference error.

If a shared dynamic-urn construction cannot be proved marginal-exact, use
independent panels. If independent low/high returns have inadequate
correlation, Rival-MF closes rather than weakening the estimand.

### 10.5 Coefficient, allocation, and coverage contract

For every enabled stratum, freeze before confirmation:

- rules, sampler, incumbent, low policy, horizon, and candidate identities;
- `beta_cv`, `n_H`, and `n_L`;
- paired-versus-independent action and fidelity coupling;
- score-difference bounds;
- one-sided inference family and practical margin;
- statistical error allocation across decisions and any planned looks; and
- timeout and fallback rules.

Estimate coefficients and cost-optimal allocations on calibration data only.
A changed policy or sampler invalidates the calibration.

Calibration must reproduce the deployed selected-challenger distribution, not
sample arbitrary legal actions. Freeze the candidate-source mixture,
candidate-count caps, `S` allocation, winner-selection rule, activation
stratum, action family, and remaining-turn band; run that exact pipeline on
calibration roots. Test the frozen result on independently repeated instances
of the same pipeline.

Before scientific use, test coverage on:

- synthetic bounded correlated distributions;
- constructed zero-advantage and harmful-advantage cases;
- exact small game subproblems;
- held-out late-game Cascadia roots; and
- deliberately weak, uncorrelated, and negatively correlated proxies.

The first implementation uses fixed sample counts and no intermediate reads.
Sequential stopping is eligible only after a time-uniform or separately
error-spent procedure is derived and coverage-tested for this estimator.

### 10.6 Fixed first-generation lower bound

Coverage tests supplement an inference theorem; they do not create one. Rival
v1 uses a conservative, fixed-sample two-panel Hoeffding construction.

For `h` in paired panel `H` and `l` in independent panel `L`, define:

\[
X_h=D_{H,h}-\beta_{cv}D_{L,h},
\qquad
Y_l=\beta_{cv}D'_{L,l}.
\]

Then:

\[
\widehat\Delta_{MF}=\overline X+\overline Y.
\]

The pinned rules/scoring identity must provide a certified finite difference
range `D_H,D_L in [d_min,d_max]` with width `R_D`. A conservative range width
for `X` is `R_X=(1+|beta_cv|)R_D`; for `Y` it is
`R_Y=|beta_cv|R_D`. With fixed error budgets `delta_H` and `delta_L`:

\[
\operatorname{LCB}_{MF}=
\overline X+
\overline Y-
R_X\sqrt{\frac{\log(1/\delta_H)}{2n_H}}-
R_Y\sqrt{\frac{\log(1/\delta_L)}{2n_L}}.
\]

Conditional on the independent `S` result and disjoint frozen calibration,
the bound has one-sided error at most `delta_H + delta_L` under the stated iid,
bounded, and panel-independence contract.

Before launch, compute the certified maximum number of eligible appeals in a
complete four-player game. Assign deterministic root budgets whose sum over
all possible appeals is no more than `delta_game`; within each root,
`delta_H + delta_L <= delta_root`. Training-corpus scientific claims receive a
separate family budget. A positive practical margin `epsilon` controls effect
size and distribution-shift tolerance; it does not control false-activation
probability. The statistical error ledger does that.

This bound may be too conservative to resolve any action. That is a legitimate
power kill. An empirical-Bernstein or confidence-sequence replacement requires
its own proof, implementation tests, and preregistration. No ordinary 95% root
interval supplies lifetime protection over indefinite deployment; Rival v1 is
an offline bounded research program and complete-game evidence remains
decisive.

### 10.7 Central kill test

On untouched roots, compare at equal wall:

1. high-fidelity-only terminal differences;
2. low-fidelity-only differences;
3. Rival-MF;
4. current root-world coupling versus independent worlds; and
5. extra ordinary Gumbel compute.

Report:

- low/high `rho` by phase, remaining turns, action family, and candidate
  source;
- variance and confidence-width ratio;
- action sign and selected-action error against a much larger high-fidelity
  reference;
- high- and low-fidelity terminal pairs per second;
- total resolved roots per wall-hour; and
- mean agreement as an audit, never as proof of unbiasedness.

Rival-MF advances only if it gives at least the preregistered absolute
resolved-root rate and a material equal-wall high-fidelity-only variance or
squared valid-confidence-width reduction, suggested at least 3x, with valid
coverage and lower selection error. Stable negative covariance is eligible;
weak or calibration-to-test unstable covariance kills the central bridge
between NX and Anchor.

## 11. Terminal appeals court

### 11.1 Root boundary

Rival v1 leaves the optional free-three refresh decision to the incumbent:

1. `pi_I` decides accept or decline from public information;
2. the engine commits the decision;
3. an accepted replacement is revealed under exact rules; and
4. Rival begins at the resulting public draft state.

The wrapper never chooses a draft for an unrevealed accepted market. Future
turns inside continuations still execute the complete frozen policy's own
decision/chance/draft semantics.

### 11.2 Candidate factory

The candidate set is frozen before terminal outcomes exist and always
contains `a0`. The initial causal arm uses:

- `a0`;
- incumbent Gumbel survivors;
- a small fixed number of RivalNet top actions; and
- at most one or two semantically distinct contract proposals after the
  contract module passes admission.

Candidate construction is full-menu legal. It records proposal source but
does not give any source an outcome-dependent quota.

The first test must not simultaneously add a large new proposer population,
new inference, and new continuation. Build the union in registered stages so
a positive result is attributable.

### 11.3 Select, discard, confirm

For each eligible root:

1. compute and store `a0`;
2. build the fixed candidate set;
3. use a selection stream and cheap exact-rule continuations to nominate one
   challenger `a*`;
4. discard selection outcomes for inference;
5. use fresh `H` and `L` confirmation panels for `a*` versus `a0`;
6. compute a registered one-sided lower bound for `Delta_H`;
7. emit one fixed-weight categorical preference only if the bound clears
   `epsilon`; in an actual one-seat `W_k` mechanism trial, play `a*`; and
8. otherwise emit no correction and execute `a0` exactly in the trial.

If more than one challenger enters confirmation, use simultaneous bounds. The
preferred first design confirms exactly one.

### 11.4 Activation and value of computation

Start with final-two through final-five personal-turn strata, excluding exact
K1 roots. The exact horizon is selected from inspectable power/throughput data
and frozen before untouched roots.

Rival v1 uses fixed, simple activation strata. A learned **value-of-
computation router** is a later original-synthesis extension. Its target is not
which action wins; it predicts whether spending the next compute block is
expected to reduce final decision regret enough to justify its wall cost.

Potential router inputs are:

- current top-two gap and effective standard error;
- disagreement between incumbent and RivalNet;
- low/high correlation stratum;
- candidate-source diversity;
- contract collapse/fragility;
- phase, remaining turns, and menu size;
- expected terminal-pair cost; and
- historical probability that more compute changed the action beneficially.

Published value-of-computation work supports explicitly asking how a
computation changes final action quality. It does not validate this router.
The router must beat fixed activation at equal wall on a disjoint root bank
before serving.

### 11.5 Fail-closed behavior

`W_k` plays `a0` on:

- no eligible challenger;
- ineffective or calibration-to-test unstable low/high correlation;
- unresolved confidence bound;
- timeout or incomplete sample panel;
- NaN, overflow, or numerical failure;
- rule, model, sampler, coefficient, allocation, or source hash mismatch;
- GPU/Rust transition or score mismatch;
- incumbent-policy trace mismatch;
- replay failure;
- strategy-fusion or hidden-information violation; or
- unavailable accelerator capacity.

This bounds activation exposure and makes failures observable. It does not
bound composed-policy score downside and is not a theorem that recursive
wrapper-versus-wrapper self-play improves. Rival v1 does not promote the
wrapper; only the ordinary `M_(k+1)` complete-policy game gate answers the
deployment question.

## 12. Layer 4: exact accelerator-resident simulation

### 12.1 One resident process, not many CUDA owners

The device architecture is one long-lived process and one model/rules service
with internal work queues. Do not repeat the 24-owned-context topology that
showed nominal utilization while completing no seed.

The work tensor is:

```text
root x candidate x physical scenario x fidelity x policy frontier
```

Each lane contains compact structure-of-arrays state for:

- four boards and exact overflow;
- market pairs and public supply;
- dynamic wildlife bag and tile inventory;
- habitat components and scoring caches;
- turns, active seat, Nature Tokens, and phase;
- scoring-card/rules identity;
- public policy memory; and
- counter-based simulator RNG.

The implementation exposes two non-interchangeable types and buffers:

```text
PrivateSimState  # hidden tile/wildlife identities, physical RNG, full rules
PublicPolicyObs  # public board/market/supply/history + one seat's memory only
```

Only a typed projection from `PrivateSimState` may construct
`PublicPolicyObs`; model kernels cannot accept the private type. Metamorphic
tests vary hidden suffixes, exclusions, physical seeds, and future return
schedules while holding public history and policy RNG fixed and require
identical policy observations, memory transitions, and action distributions.

### 12.2 Wavefront execution

Use bounded queues rather than recursive host calls:

1. compact lanes needing a legal menu;
2. enumerate/apply exact actions;
3. compact lanes needing RivalNet inference;
4. compact the smaller lanes needing direct-transformer or full-incumbent
   search;
5. advance chance transactions through the dynamic-urn kernel;
6. compact terminal and active lanes;
7. update exact scores and difference statistics; and
8. repeat until every requested continuation terminates.

Variable action counts, phase divergence, and long-tail overflow are measured,
not hidden behind kernel-call throughput.

### 12.3 Exactness contract

The Rust engine remains the oracle for:

- legal actions;
- placement and rotation;
- wildlife legality and return;
- free and paid wipes;
- chained overpopulation;
- without-replacement draws;
- habitat, wildlife, and Nature scoring;
- terminal detection; and
- canonical replay.

Require zero registered mismatches over at least one million randomized
reachable transitions plus exhaustive curated edge fixtures. Require zero
terminal trace/score mismatches over at least 10,000 complete games.

Finite tests do not prove universal equality. The claim is zero mismatch under
the registered property/exhaustive suite plus by-construction type, legality,
and marginal invariants.

If Rival claims full-incumbent continuation, rules parity is not enough. It
must separately reproduce:

- packed model rows and masks;
- numerical precision and action-stable outputs;
- root candidates, Gumbel values, visits, and selected actions;
- refresh decisions and downstream revealed-market drafts;
- rollout ordering and RNG consumption; and
- every future incumbent action.

Any action divergence changes `pi_H`. The port can be evaluated as a new
high-fidelity policy, but it cannot retain the incumbent-relative claim.

### 12.4 Throughput contract

Measure:

- complete low- and high-fidelity trajectories per second;
- resolved challenger-versus-incumbent roots per hour at registered power;
- p50/p90/p99 real-decision latency;
- compaction occupancy and branch divergence;
- bytes per active lane, maximum concurrent lanes, model/activation workspace,
  legal-menu and compaction-buffer maxima, p99 queue occupancy, and fail-closed
  OOM behavior;
- model rows, exact transitions, and terminal plies per real action;
- memory and power; and
- equal-wall action-selection error.

GPU utilization, raw evaluator rows, or isolated rules plies are diagnostics.
The decision metric is precision-required terminal comparisons per wall-second
and total local GPU-hours under the Gate-1.5 power envelope.

Pgx and Mctx establish accelerator-native simulation/search as a real pattern.
They do not predict performance for Cascadia's dynamic menus, rules, WSL2
stack, or RTX 5090.

## 13. Rival iteration: terminal evidence becomes training signal

The wrapper alone is unlikely to cover the whole historical gap. The route to
a larger gain is rollout-informed terminal relabel iteration. “Policy
iteration” is descriptive here, not a monotonic-improvement guarantee under
simultaneous four-seat change, candidate truncation, function approximation,
or statistical error.

### 13.1 Frozen iteration

For iteration `k`:

1. freeze complete ordinary base `B_k`;
2. generate natural games and collect public roots under `B_k`;
3. mine repeat-stable hard roots, incumbent/RivalNet disagreements, bounded
   adversarial diagnostics, and admitted contract proposals;
4. freeze candidate sets without terminal outcomes;
5. use disjoint `S/H/L` panels to confirm categorical Rival-MF preferences
   under `B_k`;
6. if quantitative advantages are needed, estimate them on an additional
   independent audit/value panel `A` after admission;
7. construct a hash-pinned correction corpus;
8. train exactly one ordinary candidate `M_(k+1)` with a uniform base-data
   floor;
9. screen on a locked root bank;
10. run one fresh paired full-game gate of `M_(k+1)` against `B_k`; and
11. iterate only if `M_(k+1)` is positive and John promotes it to `B_(k+1)`.

This is a target-changing loop, not continuous self-confirmation. Every
iteration gets new calibration, training, screen, and verdict partitions.

### 13.2 Target separation

Do not average incompatible targets into one value.

| Target | Estimand | Allowed use |
| --- | --- | --- |
| D1/high-budget completed-Q | current search teacher under its exact search identity | policy/Q reanalysis and candidate ranking |
| Terminal behavior return | `V` under recorded behavior/opponents | behavior-value anchor |
| Afterstate TD(lambda) | versioned bootstrap under one frozen policy | dense credit assignment |
| Rival-low terminal return | return under `pi_L` | cheap policy training and control variate only |
| Rival-MF confirmed preference | one deviation then full `B_k` continuation, with `LCB > epsilon` | fixed-weight pairwise policy preference only; not an absolute Q target |
| Independent `A`-panel advantage | fresh one-deviation difference under `B_k`, sampled only after admission | optional separate advantage head or capped preference weight; never mixed into score-to-go |
| Score contracts | spatial/reachability obligation | auxiliary and proposal only |
| Opponent next action/market survival | public conditional prediction | auxiliary resource-pressure modeling |

Each shard records rules, behavior, opponents, teacher, high/low policies,
sampler, search, coefficient, allocation, and source hashes. A changed policy
invalidates moving TD targets and multifidelity calibration.

### 13.3 Losses and exposure

A candidate may train with separate heads or masks for:

\[
\mathcal L=
\mathcal L_{\mathrm{policy}}
+\lambda_Q\mathcal L_{Q_{\mathrm{search}}}
+\lambda_{pref}\mathcal L_{\mathrm{confirmed\ preference}}
+\lambda_{adv}\mathcal L_{\mathrm{independent\ advantage}}^{A}
+\lambda_{MC}\mathcal L_{V^{B_k}}
+\lambda_{TD}\mathcal L_{TD(\lambda)}
+\lambda_{opp}\mathcal L_{\mathrm{opponent}}
+\lambda_{intent}\mathcal L_{\mathrm{intent}}.
\]

This equation names data roles, not a frozen recipe. Loss weights are selected
on development data and frozen before the candidate screen.

Rival v1 sets `lambda_adv = 0` unless the independent `A` panel exists. The
same `H/L` estimate that admits a correction cannot supply a regression
magnitude. Unconfirmed challengers receive no invented negative or zero
terminal label; those roots keep their original behavior/search targets and
target masks.

The repository's earlier antisymmetric pairwise comparator learned its labels
but failed serving-aligned routing and worsened regret. Rival's preference
target differs because it is terminally confirmed and trains the full policy,
not a Borda serving head. That distinction is a hypothesis, not an exemption:
full-menu retention and complete gameplay remain mandatory.

At least half of effective draws should remain broad uniform/on-policy data in
the first iteration. Hard and adversarial roots receive capped weights. The
actual-draw audit must prove source exposure, not infer it from nominal dataset
sizes.

### 13.4 Why this can exceed the teacher

D1 asks a stronger version of the current search to correct labels. Rival asks
a different question: after this action, then following the frozen policy to
terminal, which action actually produces more own score?

The exact endpoint can correct both model and search bootstrap bias. The
multifidelity estimator attacks its cost/variance rather than replacing the
endpoint. A gated iteration can therefore create policy labels not bounded by
the current completed-Q ranking.

It can still fail. Terminal returns are noisy, candidate sets can censor the
winner, and a one-deviation improvement can disappear after retraining changes
the state distribution. The full-game gate is the only composition evidence.

## 14. Adversarial robustness without changing the objective

### 14.1 Bounded probe bank first

Self-play policies can contain systematic blind spots. The Go adversarial-
policy literature shows that even superhuman systems may be exploitable. It
does not imply that broad adversarial training improves Cascadia mean score.

After D1, generate a bounded diagnostic bank from:

- incumbent/RivalNet disagreement search;
- wildlife-demand and market-pressure specialists trained on their own score;
- selfish resource-competition specialists whose denial behavior emerges only
  when it improves their own score;
- contract-collapse states;
- high-SNR repeated policy failures;
- D6/geometry and overflow stressors; and
- refresh/wipe/Nature boundary cases.

Every proposed failure needs high-budget or terminal confirmation,
cross-checkpoint transfer, held-out generator families, and a natural-frequency
estimate.

A coalition or target-attacking policy that sacrifices its own score may be
used only as a clearly labeled invalid stress diagnostic. It cannot enter the
training/continuation population or support a strength claim under the allowed
policy class.

### 14.2 Opponent population as an ablation

A small frozen training/evaluation pool may contain:

- the current incumbent;
- prior promoted policies;
- one or two strong structurally different policies; and
- confirmed adversarial specialists.

Every actor still maximizes its own score. The pool changes the distribution,
not the reward.

This is not automatically part of Rival. Population training can improve
robustness while lowering homogeneous self-play mean. It enters only if an
ablation improves natural opponent cross-play without regressing the primary
symmetric gate.

### 14.3 Required deployment diagnostics

For the ordinary distilled candidate `M_(k+1)`, report in addition to symmetric
self-play:

- one candidate seat versus three frozen incumbents, balanced by seat;
- one incumbent versus three candidate seats;
- the candidate against each frozen opponent-pool member;
- absolute-seat and cyclic-permutation results;
- category and resource-pressure changes; and
- exploit/adversarial-policy transfer.

These are diagnostics. The campaign target remains the pinned 1,000-game
symmetric mean-seat score.

## 15. Selfish ceiling tomography before a large build

The critique was right that ceiling tomography should precede an expensive
architecture build. Its four-board route is outside the revised adversarial
policy class. Rival uses only **unilateral, seat-local tomography**.

The experiment begins with a fresh, hash-pinned baseline `b` under the exact
rules and serving identity that Rival would improve. Roots are sampled before
their scores are inspected and retain natural-frequency weights. Hard-root
strata may be oversampled for mechanism discovery, but they must be reweighted
before any claim about game-level headroom.

Roots from one game are dependent because they share a market and trajectory.
Uncertainty uses source-game clusters: either sample one registered root per
game or use a game-level paired analysis/bootstrap. Natural-frequency weights
do not turn correlated roots into independent observations.

Define the buffered target gap:

\[
\delta_b = 100.10 - b.
\]

The extra `0.10` is a proposed design margin, not a change to the campaign's
100-point target. A system designed to land exactly at 100.00 has too little
room for evaluation error and identity drift.

### 15.1 Five measurements with different meanings

| Measurement | Construction | What it establishes | What it does not establish |
| --- | --- | --- | --- |
| T0: own-board repack | Optimize the target seat's terminal score using only the pieces it actually acquired; call it exact only with an optimality certificate. | Conditional placement and arrangement slack or a best-found lower bound for that resource multiset. | Acquisition headroom, executable policy value, or a full-game upper bound. |
| T1: public one-seat witness | T1a uses a bounded CPU high-fidelity Anchor controller on preregistered tractable late roots before RivalNet exists; T1b later uses the complete ordinary distilled candidate. Three selfish `B_k` opponents and exact chance remain. | An executable lower bound on unilateral headroom against the incumbent population. | Symmetric four-copy composition or optimality. |
| T2: late-game best response | At tractable late roots, integrate chance and frozen-policy randomness while optimizing the target seat's remaining public-information decisions. Call the expected gap exact only when exhaustive integration or certified bounds close. | An exact local gap when certified; otherwise a feasible witness and proof interval for the stated root/opponent policy. | Early-game value or a different opponent population. |
| T3: known-world one-seat oracle | Reveal a fixed exogenous chance tape only to an offline target-seat solver and optimize its legal actions; opponents remain selfish incumbents. | An optimistic information-relaxed ceiling and the price of hidden future uncertainty. | A deployable policy. It is an upper bound only when the search is exhaustive and the relaxation is proved to contain every public policy. |
| T4: resource-relaxed bound | Give the target seat a provably optimistic superset of legal future resources, then exactly bound its best score. | A certified local upper bound when the relaxation and branch-and-bound proof close. | A useful action or symmetric game strength. |

The report must call a number a **witness**, **heuristic best found**, or
**certified upper bound**. A high-scoring board found by search is a witness,
not an upper bound. A truncated known-world search is a heuristic best found,
not a proof.

### 15.2 Chronology is part of the problem

Counterfactual play must preserve the causal game:

- alternative drafts remove different public pairs;
- replacements and automatic wipes are recomputed by the exact rules;
- opponents observe the changed public history and choose selfishly;
- returned wildlife and Nature spending alter later supplies; and
- the target policy never sees a future random outcome before the real rules
  would reveal it.

Holding the realized future market fixed after changing an earlier draft is
not chronology-preserving replay. It is an inconsistent counterfactual and is
excluded.

Use domain-separated exogenous random tapes so that a counterfactual consumes
the random event associated with its semantic event, not merely “the next RNG
word.” If that construction cannot be shown marginally correct, compare
independent futures and accept the higher variance.

### 15.3 Tomography decision rule

Tomography is a mechanism gate, not strength evidence. Before outputs exist,
freeze:

- the root cohort and natural-frequency weights;
- target-seat rotations;
- public and known-world information boundaries;
- solver limits and proof-gap conventions;
- the incumbent and opponent identities;
- the target-gap baseline; and
- the conditions below.

The target-reaching route uses three asymmetric rules:

1. a valid T4 upper bound below `delta_b` closes Rival's frozen-base terminal-
   improvement path immediately for that opponent/distribution identity;
2. T1a and T2 must independently clear a preregistered legal public-policy
   witness threshold `h_min` to fund the large compiler/simulator build; and
3. T0 and T3 diagnose where headroom may lie but can never fund Rival alone.

If T4 cannot be certified, its looseness does not count as positive evidence.
If no honest T1a/T2 witness clears `h_min`, a beautiful repack or clairvoyant
oracle does not keep the program alive. `h_min` is set from the fresh target
gap, measured variance, and implementation cost before outputs exist.

For confidence to rise into the 55--65% conditional range later in this
document, the fresh baseline must be at least roughly 98.2 and an honest
public-information witness plus independent shadow evaluation must expose
enough recoverable value to cover `delta_b` with margin. A 2.5-point
unilateral witness would be compelling; it is a premise gate to discover, not
a number assumed by the architecture.

### 15.4 Why unilateral tomography is still incomplete

One improved seat against three incumbents and four isolated improved seats are
different distributions. If every seat drafts better, market pressure and
resource availability change. The change can attenuate or amplify an
individual improvement.

Therefore:

```text
one-seat witness != symmetric score gain
sum of root gaps != game gain
known-world gain != public-policy gain
conditional repack gap != acquisition-policy gain
```

Only a complete symmetric game experiment can establish composition.

## 16. Exact serving and labeling protocol

### 16.1 Frozen panels

Each eligible root has three disjoint random panels:

1. **proposal panel `S`:** cheap worlds used to rank a fixed candidate union;
2. **paired confirmation panel `H`:** worlds on which both low- and
   high-fidelity continuations are evaluated for the incumbent and one frozen
   challenger; and
3. **extra low panel `L`:** independent low-fidelity worlds used by the
   multifidelity control variate.

The challenger selected on `S` is frozen before `H` or `L` is touched.
`beta_cv`, sample counts, strata, coupling, practical margin, and confidence
construction are frozen on a disjoint calibration cohort.

Selection may use sequential halving inside `S`. Confirmation uses one fixed
look unless a valid confidence-sequence design is preregistered. An ordinary
fixed-sample interval may not be inspected repeatedly and stopped when it
crosses zero.

The `H/L` outcome decides only whether a fixed challenger earns a categorical
preference label. Conditioning on `LCB > epsilon` biases the retained point
estimates upward. Rival v1 therefore gives every confirmed preference a fixed,
preregistered policy-loss weight and does not regress its observed magnitude.
Any quantitative advantage target requires a fourth, independent post-
selection audit/value panel `A` that is not used for admission.

### 16.2 Root algorithm

This is the `W_k` research/labeling instrument used in shadow and one-seat
mechanism trials. The ordinary distilled `M_(k+1)` policy is the v1 promotion
candidate.

```text
function RIVAL_DECIDE(public_state s, seat i, identity I):
    verify rules, source, model, simulator, RNG and compiler hashes
    a0 = exact_incumbent_action(s, i, I)

    if forced_action(s) or not eligible_stratum(s):
        return a0

    C = exact_legal_union(
            incumbent_survivors(s),
            RivalNet_proposals(s),
            optional_contract_proposals(s))
    verify a0 in C

    a_star = cheap_select_one_challenger(C - {a0}, proposal_panel_S)
    freeze(a_star)

    paired = run_low_and_high(a0, a_star, paired_panel_H)
    extra  = run_low(a0, a_star, extra_low_panel_L)
    estimate, lower_bound = fixed_Rival_MF(paired, extra, calibration)

    if any identity, coverage, marginal, parity or completeness check fails:
        return a0
    if lower_bound <= practical_margin_epsilon:
        return a0
    return a_star
```

The actual implementation is allowed to batch roots and worlds aggressively.
It is not allowed to change the statistical unit, omit slow worlds, compare
mixed-fidelity actions, or return a substitute fallback.

### 16.3 Practical margin

The activation threshold is not merely `Delta > 0`. It must clear a frozen
practical margin for:

- numerical tolerance;
- policy-distribution shift after an override;
- the possibility that local gains fail to compose; and
- the serving cost itself if latency is part of the product constraint.

The margin does not control false-activation probability. The bounded lower
confidence construction and its per-game statistical error ledger do that.

The margin is selected on design seeds and frozen. Sweeping it on the verdict
block creates multiple candidates and invalidates the claimed selection
procedure.

### 16.4 Statistical scope

A valid lower bound for one root says only that, under the frozen continuation
and sampling construction, the selected action's expected terminal own-score
difference cleared the threshold. It does **not** prove:

- the full policy is safe;
- the low-fidelity model is globally accurate;
- later overrides preserve the same continuation distribution;
- a retrained policy inherits the local gain; or
- the final mean exceeds 100.

The literal fallback creates an asymmetric design, but it is not a theorem of
non-regression for the composed policy. That phrase is reserved for a fresh
paired full-game result.

### 16.5 Required identity ledger

Every root record contains at least:

- public state, all four isolated policy-memory snapshots, rules, source,
  compiler, simulator, and policy hashes;
- active seat, phase, root stratum, and natural-frequency weight;
- exact legal menu and canonical action hashes;
- incumbent, candidate-union, selected-challenger, and final action IDs;
- `S`, `H`, `L`, and any optional `A` cohort identities and domain-separated
  random keys;
- low/high terminal own scores for both actions;
- completion, timeout, overflow, and fallback flags;
- fixed coefficient, allocation, variance estimate, interval, and margin;
- device, precision, kernel, batch-shape, and wall-time fields; and
- a parent manifest tying the record to the full policy identity.

Raw per-world ledgers are durable beside the report. A summary without its
complete raw ledger is not publishable.

## 17. Fair bakeoff: isolate the load-bearing claims

The architecture is not tested as one giant bundle. It earns complexity one
claim at a time.

### 17.1 Frozen arms

| Arm | Purpose | Eligible conclusion |
| --- | --- | --- |
| A: current valid incumbent | Fresh canonical control and target gap. | Baseline only. |
| B: promoted D1, if any | Tests whether target correction already captured the available signal. | New incumbent only after John's promotion ruling. |
| C: high-fidelity Anchor | Full-incumbent terminal comparison without Rival-MF. | Mechanism and gold-standard cost/quality control. |
| D: RivalNet direct | Cheap policy used by itself. | Measures quality/speed frontier; not presumed replacement. |
| E: low-fidelity override | Terminal decisions using only cheap continuation. | Quantifies surrogate bias; never the final safety claim. |
| F: Rival-MF override | Cheap breadth plus fixed high-fidelity correction. | Tests the central synthesis. |
| G: F plus score-contract proposals | Tests whether contracts add confirmed natural-frequency challengers. | Contract admission only. |
| H: F plus bounded opponent pool | Tests robustness distribution separately. | Population admission only if primary mean does not regress. |
| I: one frozen `M_(k+1)` | Ordinary distilled policy trained from the single selected correction recipe. | Sole v1 promotion and target candidate. |

Arms G and H are not run unless F first works. This prevents a null result
from being uninterpretable. Arms C--H are mechanism/labeler studies, not
promotion candidates. Exactly one Arm-I identity is chosen on design evidence
before promotion seeds are touched.

### 17.2 Equal-wall and equal-evidence views

Report both:

- quality at equal wall-clock cost; and
- cost to reach equal effective precision or confirmed-challenger recall.

The throughput denominator is a complete, valid terminal action pair or a
complete game. GPU utilization, positions per kernel, issued worlds, and
inference calls are secondary diagnostics. Timed-out and invalid units remain
in the denominator.

### 17.3 Controls that prevent a flattering comparison

- Same pinned rules, source, scorer, action menu, root cohort, target seat,
  precision, and machine.
- Separate compile/warmup timing from steady-state timing, but report both.
- Include transfer, compaction, overflow, synchronization, and fallback cost.
- Rotate absolute seats and use cyclic permutations.
- Treat a complete game as the independent unit; use game-clustered
  uncertainty for root cohorts or sample one root per source game.
- Use disjoint design, selection, shadow, promotion, and target seed blocks.
- Hash-pin every checkpoint and simulator binary.
- Evaluate forced-anchor parity before any challenger result.
- Freeze every arm before its paired output is read.
- Report failures and incomplete units, not only successful batches.

No result on a hard-root-enriched cohort is converted into expected game
points without natural-frequency weighting and a complete-game experiment.

## 18. Evidence ladder and kill criteria

Rival is deliberately staged so the cheapest decisive evidence comes first.

### Gate 0: D1 boundary and canonical control

Do not displace, inspect, or adapt the current D1 chain. After it reaches its
frozen boundary:

1. resolve the valid incumbent identity;
2. run a fresh current-rules canonical baseline;
3. define `delta_b`; and
4. allocate fresh seed blocks.

**Kill or rescope:** identity cannot be reproduced, or the target/baseline is
not measured under the same rules.

### Gate 1: selfish tomography

Run T0--T4 on a preregistered root cohort, beginning with tractable late-game
states and the CPU high-fidelity T1a public one-seat witness.

**Fund:** T1a and T2 independently clear the frozen legal public-policy witness
threshold, while any certified T4 ceiling comfortably covers `delta_b`.

**Kill or narrow:** a valid T4 bound is below the gap, or neither honest T1a nor
T2 clears the witness threshold. T0/T3 cannot fund the program, and a
cooperative escape hatch is not available.

### Gate 1.5: cheap correlation and power falsifier

Before a GPU port or new model, run a small CPU/current-bridge panel using an
existing cheap direct, distilled, or pattern policy as `pi_L` and the complete
base as `B_k` on tractable late roots. This cannot qualify production
Rival-MF; it cheaply asks whether paired terminal action differences have any
stable exploitable covariance.

Use measured score-difference variance, the fixed bounded lower confidence
rule, target margin, activation frequency, and current high/low costs to derive
required `n_H`, `n_L`, resolved roots/hour, games/hour, and total local
GPU-hours. Compute the best-case envelope as well.

**Kill or narrow:** even optimistic correlation/speed cannot make the required
high-fidelity samples fit the absolute work budget.

### Gate 2: compiler and exact-engine parity

Show zero mismatch under registered semantic-feature, legal-menu, transition,
score, random-marginal, D6, overflow, and forced-action tests, backed by exact
by-construction invariants. Benchmark dense, component-local, and incremental
designs on complete trajectories.

**Fund:** bit-identical output and a credible path to complete-trajectory
speedup.

**Kill:** any irreducible correctness mismatch, or exact feature maintenance
is slower than the existing path at the required batch shape.

### Gate 2b: complete high-fidelity policy adapter

Implement the full `B_k` continuation boundary: packed rows, transformer
numerics, legal menus, Gumbel candidates/backups, refresh decisions, exact-K1,
inner redeterminizations, rollout order, policy RNG, tie/fallback behavior, and
all future actions. The outer physical scenario remains simulator-private.

**Fund:** forced-base complete policy traces are action-stable against the
production bridge and absolute high-fidelity terminal-pair throughput fits the
Gate-1.5 envelope.

**Kill or offline-narrow:** rules parity passes but policy traces diverge, p99
queue/memory behavior is unsafe, or complete base continuations remain
unaffordable.

### Gate 3: RivalNet economics

Train only after Gate 2. Freeze a small architecture bakeoff and evaluate
complete-trajectory throughput, terminal-difference correlation, and
confirmed-challenger retention.

**Fund:** at least the greater of roughly 5x full-incumbent trajectory speed
and the Gate-1.5 absolute rate; near-lossless useful-challenger retention for
the proposal role; and stable nonzero correlation of either sign with measured
equal-wall benefit for the control-variate role.

**Kill or simplify:** global corrections erase speed, the representation
misses systematic topology, or correlations are weak or unstable between
calibration and held-out selected-challenger distributions.

### Gate 4: Rival-MF calibration

On a disjoint root bank, estimate low/high correlations and costs; freeze the
control-variate coefficient and high/low allocation; then test on untouched
roots.

**Fund:** at least roughly 3x equal-wall variance or squared valid-confidence-
width reduction, absolute resolved-root throughput above the power bar,
correct coverage, and no material stratum failures.

**Kill:** low fidelity is too weakly correlated, coupling changes marginals,
coverage fails, or the required high-fidelity fraction removes the economic
advantage.

### Gate 5: shadow policy

Run `W_k` in shadow on full games: log what it would change without altering
the incumbent's trajectory. Then run actual one-seat mechanism games, because
shadow counterfactuals do not compose. `W_k` remains a labeler/instrument, not
the v1 promotion candidate.

**Fund:** changes occur at natural frequency, confirmed margins survive
fresh worlds, and one-seat own score improves without rules/identity drift.

**Kill:** appeals almost never activate, local gains disappear after the first
changed trajectory, or runtime is operationally infeasible.

### Gate 6: one frozen relabel iteration

Build one balanced terminal-confirmed categorical-preference tranche, train
one ordinary `M_(k+1)` from one frozen `B_k`, and select exactly one checkpoint
using design/selection data. Quantitative advantage regression remains off
unless its independent `A` panel exists.

**Fund:** exposure audits pass; retention, broad-state quality, and frozen
root corrections survive; the candidate identity is complete.

**Kill:** labels are washed out, hard-root weighting harms broad policy, or no
single candidate survives the preregistered selection bar.

### Gate 7: paired gameplay promotion evidence

Run exactly the preregistered `M_(k+1)` versus `B_k` contrast for at least 100
paired complete games under the campaign's current promotion contract. A
positive repeated-confidence result is valid only under a
preregistered sanctioned group-sequential design; otherwise read the result
once at completion.

**Fund:** the 95% paired interval excludes zero, provenance is clean, and John
chooses to promote.

**Kill or revise:** CI includes zero, any guardrail fails, or provenance is
incomplete. Validation loss, shadow score, terminal-label regret, and GPU
throughput cannot substitute for gameplay.

### Gate 8: target battery

Freeze the ordinary `M_(k+1)` policy once. Run 1,000 complete four-player
symmetric games
under one rules/serving identity. Report mean seat score and uncertainty.

**Success:** mean seat score is at least 100 under the campaign contract.

Anything else is not the target, even if the policy wins a paired promotion
gate.

## 19. Red-team report: strongest reasons Rival may fail

This section is intentionally hostile. It was synthesized from independent
architecture, feasibility, and devil's-advocate reviews, then applied to the
combined design.

### 19.1 The baseline may make the target much harder than the document looks

There was no admissible July-16 canonical mean at cutoff. The historical
98.2975 is useful context, not the current denominator. If corrected rules or
the resolved D1 incumbent sit below about 97.8, Rival likely needs several
independent points, not a refined half-point override.

**Detection:** Gate 0.

**Response:** recompute every gap and confidence range from the fresh
baseline. Do not preserve the proposal's ranking by pretending the old number
is current.

### 19.2 Selfish recoverable headroom may not cover the gap

The game may simply offer little exploitable unilateral improvement under the
shared market and current strong policy. The critique's cooperative route
would have changed the policy class; that route is now prohibited.

**Detection:** legal T1/T2 tomography plus T4 bounds.

**Response:** close Rival if the headroom is absent. No architecture can
manufacture points above the feasible policy ceiling.

### 19.3 Full-incumbent terminal continuation may be computationally absurd

A rough illustrative root with 64 paired worlds, two actions, and 20 **total
future seat decisions across the table** requires about 2,560 incumbent
decisions before retries. At roughly 21 seconds per current decision, a serial
implementation would be on the order of 15 hours for one root. These are not
a workload estimate; they expose the nested multiplier. Even a nominal 100x
systems speedup can leave minutes per root.

**Detection:** Gates 1.5 and 2b report complete paired terminals per wall-hour
with all overheads.

**Response:** Rival-MF must reduce high-fidelity calls by measured correlation,
and terminal relabeling can run offline. If the remaining high-fidelity cost is
still infeasible, close the online wrapper and retain only a bounded offline
labeling experiment or close the direction entirely.

### 19.4 Exact terminal score removes bootstrap bias, not chance variance

Two exact continuations can finish far apart because later markets differ.
“Terminal” is not synonymous with “precise.” In fact, a long terminal horizon
may have more variance than a short biased evaluator.

**Detection:** paired difference variance by phase and remaining horizon.

**Response:** compare mean-squared decision error and effective precision per
wall, not endpoint purity. Disable early-game appeals if the terminal signal
is uneconomic there.

### 19.5 D1 and Rival may harvest the same errors

D1 corrects hard labels; Rival corrects one-deviation decisions. If D1 works,
the remaining wrong-argmax pool may shrink. Their historical or projected
gains cannot be added.

**Detection:** post-D1 disagreement frequency and confirmed residual regret.

**Response:** always re-anchor on the resolved incumbent and rerun tomography.
Treat D1 success as a stronger start, not as a point credit in Rival's effect
budget.

### 19.6 RivalNet may be fast for the wrong reasons

A tiny network can agree with easy incumbent actions while reversing the rare
differences that matter. Top-1 teacher agreement and validation RMSE can hide
this.

**Detection:** correlation with paired terminal action differences,
independently confirmed useful-challenger recall, and phase-stratified errors.

**Response:** optimize the cost/correlation frontier. If no small model retains
the relevant ordering, use it only as a proposal generator or abandon it.

### 19.7 The control variate can be unbiased yet useless

With weak correlation, the variance reduction tends toward zero while every
low-fidelity rollout still costs time. Stable negative correlation is usable
with a negative coefficient, but an aggregate correlation can hide a sign or
magnitude that changes in the exact late-game or contract-collapse strata
where Rival activates.

**Detection:** untouched, stratum-specific effective-sample gain and coverage.

**Response:** enable Rival-MF only in strata whose fixed calibration passes.
Use high fidelity alone elsewhere. Close the central novelty if the aggregate
equal-wall gain stays below the preregistered bar.

### 19.8 Coupling may silently change the game distribution

Dynamic urns, wildlife returns, wipes, and conditional reveals make naive
common random numbers dangerous. Consuming a shared RNG stream after actions
diverge can create action-dependent random marginals.

**Detection:** property tests and distributional tests for every semantic
chance event, plus independent-world replication.

**Response:** use semantic-event random domains with a proof of marginal
equality, or independent futures. Variance reduction never outranks a correct
estimand.

### 19.9 Candidate selection can erase the estimator's guarantees

Selecting the largest noisy low-fidelity winner and evaluating it on the same
worlds creates winner's-curse bias. Estimating the control coefficient on the
same confirmation panel can do the same. Conditioning on a positive `H/L`
bound also biases admitted point-estimate magnitudes upward.

**Detection:** immutable cohort manifests and audit of every random key.

**Response:** disjoint `S`, `H`, `L`, and calibration cohorts; one frozen
challenger; one frozen coefficient; fixed-weight categorical labels. Any
quantitative advantage target comes from an additional independent `A` panel.
Any reuse invalidates the root claim.

### 19.10 One-deviation improvement is not policy improvement in practice

Classical rollout guarantees require assumptions that a truncated,
candidate-limited, stochastic multiplayer implementation may violate. After
the first override, later states differ; after retraining, the continuation is
no longer the policy used to label the action.

**Detection:** actual one-seat games, then symmetric full games, not summed
root deltas.

**Response:** one frozen iteration at a time. Do not recurse until a complete
paired gameplay gate is positive.

### 19.11 GPU residence may increase utilization while reducing science

Branch divergence, variable legal menus, compaction, overflow, host-device
synchronization, and many CUDA owners can turn a plausible kernel into zero
completed trajectories.

**Detection:** completion-weighted throughput, p95 latency, batch occupancy,
fallback rate, and power/utilization shown together.

**Response:** one resident process, one queue, wavefront compaction, bounded
buffers, exact overflow, and CPU controls. Never report utilization alone.

### 19.12 NNUE-style incrementality may not fit Cascadia

One placement can merge habitats, reconnect Salmon, change Fox neighborhoods,
alter several future action affordances, and invalidate global summaries. A
dependency-complete “incremental” update may touch most of the board.

**Detection:** measured invalidation fanout and full-trajectory bakeoff among
dense, local-delta, and incremental paths.

**Response:** select the fastest exact path. RivalNet is not required to be an
NNUE accumulator.

### 19.13 Score contracts may be sophisticated decoration

Terminal board witnesses ignore stochastic acquisition. Repair operators can
consume more search than they save, and the best proposal may already be in
the incumbent union.

**Detection:** incremental confirmed-challenger yield at natural frequency
and equal wall.

**Response:** keep contracts out of the load-bearing path and delete the
module if Arm G does not beat Arm F.

### 19.14 Adversarial training may optimize rare pathologies

The Go exploit literature proves vulnerability can persist; it does not prove
that Cascadia exploit probes improve mean score. A specialist population can
pull capacity away from the homogeneous target distribution.

**Detection:** natural-frequency estimate, transfer across checkpoints, and a
primary-distribution regression gate.

**Response:** diagnostic bank first, tiny frozen population second, automatic
removal on primary-mean regression.

### 19.15 Multiplayer semantics may still be wrong

Own-coordinate `max^n` is the incumbent search heuristic consistent with the
selfish non-constant-sum contract, but approximate opponent values and tie-
breaking can produce strategy fusion or unrealistic responses. Paranoid
minimax would be wrong in the opposite direction by treating three opponents
as a coalition.

**Detection:** public-history action-equality tests, exact late-game solutions,
opponent-policy calibration, and tie-break audits.

**Response:** each simulated seat invokes its own frozen public policy and own
value coordinate. Do not invent coalitions, table utility, or hidden-state
contingent policies.

### 19.16 Training can wash out rare correct labels

Even good terminal corrections may be sparse relative to the broad corpus.
Overweighting them can destroy general policy calibration; underweighting them
can make the iteration a no-op.

**Detection:** actual draw-source exposure, fixed-root retention, broad
validation, and complete gameplay.

**Response:** balanced capped sampling, explicit source manifests, one
selection rule, and no conclusion from validation loss alone.

### 19.17 Research multiplicity can manufacture a winner

Architecture widths, strata, margins, coefficients, candidate sources,
opponent pools, and checkpoints create a large garden of forking paths.

**Detection:** preregistration and a ledger of every attempted identity,
including failures.

**Response:** small bounded bakeoffs, one selected candidate per stage,
disjoint verdict seeds, and no retroactive “family” redefinition.

### 19.18 A positive paired gate may still miss 100

Promotion evidence is relative. A statistically positive +0.4 candidate from
a 98.3 baseline is still below target.

**Detection:** fresh absolute target battery.

**Response:** state both claims separately: “better than incumbent” and
“mean at least 100.” Never substitute one for the other.

## 20. Realistic confidence and effect budget

### 20.1 What the probability means

The 25--35% present forecast is my subjective probability that at least one
ordinary `M` policy produced by this **finite** Rival program has a true
current-rules symmetric mean seat score of at least 100.

The forecasted program is capped at:

- two Rival relabel iterations after the resolved D1 boundary;
- at most three RivalNet shapes and one selected `M` candidate per iteration;
- one contract-proposal ablation and one opponent-population ablation;
- no wrapper candidate touching promotion seeds;
- no more than 3,000 net john0 scientific GPU-hours after D1;
- existing local hardware and no paid compute; and
- one final target battery for the selected ordinary policy.

If Gate 1.5 shows the program cannot fit that work budget, it closes or a new
forecast must be issued before outputs are inspected. D1 reaching 100 by
itself would achieve the campaign goal, but it is the starting boundary and is
not counted as a Rival-produced success. The forecast is not:

- a confidence interval;
- a probability that the implementation passes parity;
- a probability that Rival beats the incumbent by any amount;
- a sum of the prior proposals' forecasts; or
- a substitute for the 1,000-game target battery.

The range conditions on repository and literature evidence available at the
cutoff. It is deliberately below 50% because the current absolute baseline,
selfish headroom, low/high correlation, exact accelerator economics, and
game-level composition are all unknown.

### 20.2 Effects cannot be added

The following arithmetic is forbidden:

```text
D1 projected gain
+ historical Anchor gain
+ NX throughput gain
+ Foundry contract gain
= points above 100
```

D1 and Rival target overlapping decision errors. Throughput is not score.
Multifidelity correction is an estimator, not a point source. Contracts and
adversarial probes are unproven optional proposal sources. Historical Anchor
results were under an old policy and rules identity.

The only honest planning statement is that the direct historical Cascadia
precedent suggests an **approximately half-point-scale** local mechanism,
while reaching 100 from a baseline near the old 98.3 would require a much
larger composed result. Rival's credible upside comes from at most two frozen
relabel iterations and qualitatively better terminal labels, not from adding
independent credits. No numerical per-iteration gain prior is evidence-backed;
the forecast includes substantial null and negative-result probability.

### 20.3 Conditional confidence ladder

| Evidence state | Subjective target-reaching probability | Why |
| --- | ---: | --- |
| Present cutoff: no valid baseline or premise gates | **25--35%** | Architecture is coherent, but every large premise remains open. |
| `delta_b > 2.3`, no extraordinary headroom | **below 15%** | The required gain is too large for the directly supported mechanism. |
| `1.7 < delta_b <= 2.3`, tomography positive but systems unproved | **15--25%** | Requires a large composed gain from a half-point-scale historical mechanism. |
| `1.1 < delta_b <= 1.7`, tomography positive but systems unproved | **25--40%** | Plausible gap, still large engineering and composition risk. |
| `0.6 < delta_b <= 1.1`, tomography positive but systems unproved | **35--50%** | Less effect required, but the architecture premises remain unproved. |
| `delta_b <= 0.6`, tomography positive but systems unproved | **45--60%** | Target is close; rules-correct uncertainty and implementation risk remain. |
| Baseline at least 98.2; exact engine, RivalNet, Rival-MF, coverage, and selfish-headroom gates all pass; independent shadow/one-seat evidence covers the gap with margin | **55--65%** | The load-bearing premises have become measurements, but full-game composition remains unproved. |
| Positive fresh paired game gate and absolute projection safely above 100.10 | **65--80%**, depending on interval width | Direct game evidence now dominates the architecture prior. |
| Completed valid 1,000-game target battery | No subjective forecast | Report the result and uncertainty. |

These rows are states of evidence, not a schedule by which confidence rises
automatically. A failed gate lowers or closes the program.

### 20.4 Why 75% is not defensible today

To claim more than 75% now, one would need to be confident simultaneously
that:

1. the fresh incumbent is close enough to 100;
2. at least the remaining gap exists as recoverable selfish headroom;
3. the cheap continuation is both much faster and stably correlated with high
   fidelity strongly enough to improve equal-wall precision;
4. the exact GPU simulator completes enough trajectories;
5. corrected root choices compose across a changing game distribution;
6. retraining retains them without broad regression; and
7. a single frozen candidate survives a fresh target battery.

None is currently established. Assigning 75% would hide conjunction risk,
not express creativity. The best way to maximize the actual chance of 100 is
to expose those premises cheaply and stop protecting failed ideas.

## 21. How Rival differs from the alternatives

### 21.1 Versus NNUE

NNUE is principally a fast, incrementally updated position evaluator used
inside exact search. Stockfish's original implementation exploits small input
changes to update a CPU-friendly accumulator; Stockfish 16.1 later added a
secondary network for positions judged easy.

RivalNet may borrow sparse features, symmetry tying, integer-friendly layers,
and delta updates, but Rival is broader:

- it is an explicit public-information policy as well as an evaluator;
- it predicts incumbent-relative terminal differences and auxiliary intent;
- it may use dense fixed-width GPU recomputation when that is faster;
- it proposes actions and supplies a low-fidelity control variate;
- exact terminal own score, not its static value, confirms overrides; and
- the current transformer remains the high-fidelity policy and fallback.

Calling RivalNet “NNUE” would obscure the central estimator. Calling it
“afterstate evaluator” would also be incomplete: afterstate encoding is only
the first step; the proposal's evidence comes from full legal continuations.

### 21.2 Versus the current transformer

Rival does not assert that a 1--8M model is stronger than the current
transformer per evaluation. The repository's smaller-model results make that
unlikely without a different representation and search boundary.

It instead asks the small model to win a narrower economic contest:

```text
useful terminal-decision information per wall-second
```

The transformer remains mandatory whenever the cheaper system is uncertain
or invalid. Rival beats the current system only if the complete frozen policy
wins a paired gameplay gate. There is no architectural declaration of victory.

### 21.3 Versus Cascadia-NX

NX proposed a structured evaluator and GPU-resident search as the new primary
system. Rival retains its compiler, D6/factor ideas, and systems boundary but
demotes the small model until it proves useful correlation with the incumbent
continuation. Saved compute is spent on terminal policy differences, not just
more of the search axis that already saturated.

### 21.4 Versus Cascadia-Anchor

Anchor is Rival's spine. Rival adds:

- a measured low/high-fidelity bridge to the true Anchor estimand;
- a structured cheap continuation optimized for paired-difference
  correlation;
- an exact resident simulator economics contract;
- frozen terminal relabeling from confirmed labels rather than a permanent expensive
  wrapper; and
- stricter statements about local versus whole-policy safety.

If Rival-MF fails, high-fidelity Anchor remains the scientifically clean
control and may survive in narrow late-game strata.

### 21.5 Versus Cascadia Foundry

Rival retains only seat-local score contracts and diagnostics. It deletes the
centralized plan population, joint four-board genome, resource exchange,
shared memory, team utility, and cooperative confidence claim.

The conceptual difference is absolute:

```text
Foundry-Commons: choose actions for a table-level plan
Rival: each seat independently chooses actions for its own terminal score
```

### 21.6 If forced to name one replacement architecture

No standalone replacement model currently deserves high confidence of
beating the transformer. The most defensible model candidate is the
semantic-compiler plus D6-tied RivalNet family, but only as a cheap policy,
proposal expert, and control variate first.

The architecture I have the most confidence in is therefore a **system that
preserves the transformer and learns when exact terminal evidence justifies
departing from it**. That is less aesthetically pure than a model swap and
more likely to produce a real positive gate.

## 22. Explicit non-recommendations

Do not spend the next major block on:

- a larger transformer or another ordinary fresh-data repeat;
- a wholesale small-model replacement before the correlation/economics gate;
- more ordinary n4096-style root simulations;
- table-total, table-mean, fairness, donation, or coordinated-seat rewards;
- paranoid minimax against three imaginary coalition partners;
- a pure four-seat vector head used as utility;
- risk-sensitive or quantile serving when the target is expected score;
- a learned world model for rules already known exactly;
- a score-contract DSL as the direct production controller;
- an opponent population before a natural-frequency exploit bank exists;
- many competing CUDA contexts on john0;
- uncorrected low-fidelity rollouts relabeled as incumbent rollouts;
- adaptive seed reuse or verdict-block tuning; or
- promotion from validation loss, root regret, throughput, or shadow score.

Each may contain useful diagnostics. None currently has a better evidence-to-
risk ratio than Rival's staged program.

## 23. Implementation blueprint on local hardware

This is an implementation sequence, not authorization to launch or displace
the live chain.

### Phase 0: freeze the post-D1 starting point

- Restore and verify ordinary john0 reachability.
- Let D1 reach its registered boundary without partial-score inspection.
- Resolve the incumbent only through the existing gate and John's promotion
  authority.
- Produce the fresh baseline, rules identity, source hash, seed registry, and
  target gap.

**Artifact:** one canonical `rival_baseline_manifest.json` and complete game
ledger.

### Phase 1: selfish tomography package

- Implement own-board repacking with optimality certificates or explicit
  best-found/proof-gap reporting.
- Implement the CPU T1a high-fidelity Anchor witness with three frozen selfish
  incumbents.
- Add tractable late-game exhaustive integration or certified bounds and
  optimistic resource relaxations.
- Show public-history equality and hidden-future separation under registered
  metamorphic/property tests and by-construction typed boundaries.
- Publish natural-frequency and stratified results.

**Decision:** close, narrow, or fund the terminal-improvement premise.

### Phase 1b: cheap covariance and power falsifier

- Use an existing cheap direct, distilled, or pattern continuation on a small
  CPU/current-bridge cohort.
- Reproduce the exact candidate-selection distribution before measuring
  low/high terminal-difference covariance.
- Derive the fixed bounded interval's `n_H/n_L`, absolute roots/hour,
  games/hour, memory, and 3,000-hour program envelope.
- Close the online path immediately if even optimistic economics fail.

**Decision:** fund the engine only if the terminal-difference premise can fit
an absolute local-hardware budget. This pilot cannot qualify production
Rival-MF.

### Phase 2: semantic compiler preflight

- Inventory every scorer and transition dependency.
- Build canonical integer feature IDs and legal-action hyperedges.
- Add property-based mismatch tests and by-construction invariants against slow
  full recomputation.
- Benchmark dense, component-local, and incremental feature paths on CPU and
  the local CUDA device.
- Record p50/p95/max invalidation fanout.

**Decision:** choose the fastest exact path; do not assume NNUE-style deltas.

### Phase 3: one resident exact simulator

- Port exact transitions and score updates behind a batched state structure.
- Use one owner process, persistent buffers, wavefront queues, and exact
  overflow.
- Require zero trajectory mismatch against the Rust/CPU oracle across the
  registered randomized full games and every exhaustively enumerable rules
  boundary.
- Benchmark complete action pairs and complete games, including failures.

**Decision:** fund only if complete-trajectory economics support the planned
high-fidelity sample count.

### Phase 3b: complete base-policy adapter

- Integrate the full `B_k` transformer-plus-Gumbel serving rule, not only
  rules and scores.
- Keep outer physical scenarios private and independently redeterminize every
  future public policy decision.
- Match packed rows, masks, numerical mode, candidate sets, Gumbel backups,
  refresh branches, exact-K1, RNG domains, tie breaks, fallbacks, and every
  future action against the production bridge.
- Measure full-incumbent terminal pairs/hour, model workspace, lane bytes,
  p99 queue occupancy, OOM behavior, and total required GPU-hours.

**Decision:** no full-incumbent claim without complete policy-trace parity and
absolute throughput. If it fails, retain only a bounded offline CPU labeler or
close.

### Phase 4: bounded RivalNet bakeoff

- Freeze at most three small shapes spanning summary-only,
  component-augmented, and fast/global-correction variants.
- Distill policy/value from broad on-policy data and the resolved incumbent.
- Train terminal-difference and disagreement auxiliaries on disjoint data.
- Select by trajectory speed, confirmed-challenger recall, and low/high
  terminal-difference correlation.

**Decision:** one model or no model. Architecture search does not continue
until one candidate passes the frozen bar.

### Phase 5: Rival-MF calibration and proof

- Allocate disjoint calibration and coverage cohorts.
- Measure stratum-specific costs, variances, and correlations.
- Freeze the coefficient, high/low allocation, coupling, and interval.
- Verify the derived fixed-sample bound in synthetic fixtures where the true
  mean is exactly enumerable, in addition to its analytic bounded-mean basis.
- Validate untouched-root coverage and equal-wall effective precision.

**Decision:** enable only passing strata; otherwise revert to high fidelity or
close.

### Phase 6: terminal appeals shadow and one-seat trial

- Run frozen proposal/confirmation panels without changing live actions.
- Measure activation, completion, and natural-frequency corrected regret.
- Run one `W_k` research-instrument seat against three incumbents with balanced
  seats.
- Compare high-fidelity Anchor, uncorrected low fidelity, and Rival-MF.

**Decision:** fund a label tranche only after actual one-seat evidence.

### Phase 7: one relabel iteration

- Build a balanced broad/hard/adversarial root cohort.
- Confirm exactly one challenger per admitted root on fresh panels and use a
  fixed categorical label weight; add an independent `A` panel before any
  magnitude regression.
- Store full raw ledgers, hashes, and exposure weights.
- Train one ordinary `M_(k+1)` and select one checkpoint under a preregistered
  rule.
- Recheck forced-incumbent and serving parity.

**Decision:** one paired gameplay candidate; no checkpoint fishing.

### Phase 8: game gates and target

- Run the valid `M_(k+1)` versus `B_k` paired promotion gate, at least 100
  complete pairs.
- If CI-positive and John promotes, make `M_(k+1)` the next ordinary base.
- Run at most one more full Rival iteration if the measured residual premise
  remains and the finite work budget permits.
- Freeze the selected ordinary policy and run the 1,000-game absolute target
  battery.

### 23.1 Proposed source and artifact boundaries

Keep durable concepts separated:

```text
cascadiav3/rival-engine/
    Cargo.toml           # exact resident rules/search extension
    src/                 # typed private/public state, ABI and CPU oracle path
    cuda/                # batched transition/compaction kernels where selected

cascadiav3/real-root-exporter/
    src/rival_*.rs       # production B_k trace/parity adapter

cascadiav3/src/cascadiav3/rival/
    compiler.py          # canonical semantic features and action hyperedges
    policy.py            # RivalNet definitions and frozen inference identity
    multifidelity.py     # estimator, coefficients and coverage checks
    appeals.py           # S/H/L and optional A protocols + error ledger
    tomography.py        # unilateral witnesses and bounds

cascadiav3/tests/
    test_rival_compiler_*.py
    test_rival_marginals_*.py
    test_rival_multifidelity_*.py
    test_rival_nonanticipativity_*.py

cascadiav3/reports/rival_<identity>/
    manifest.json
    preregistration.md
    raw_games/
    root_ledgers/
    parity_report.json
    throughput_report.json
    verdict.md
```

Names may change during implementation, but the boundaries may not collapse:
exact rules, model inference, statistical inference, and scientific artifacts
must remain independently testable.

The verification contract includes, at minimum:

```bash
cargo test --workspace
cargo test --manifest-path cascadiav3/rival-engine/Cargo.toml
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \
  python3 -m unittest discover -s cascadiav3/tests -v
```

The exporter command is explicit because workspace Cargo checks do not cover
it in this repository.

### 23.2 Hardware allocation

- john0 owns all CUDA parity, training, and gate evidence, one scientific job
  at a time.
- Mac minis may generate explicitly allocated training shards only; their MPS
  scores are never promotion evidence.
- Offline CPU tomography may run locally if it does not claim john0-equivalent
  throughput or gate validity.
- No cloud compute or paid service is required by the design.

The resident CUDA implementation is designed for the local RTX 5090 class
machine described in the infrastructure document. It must still earn its
actual memory and throughput envelope on that machine.

## 24. Final recommendation

After D1 reaches its boundary, do these in order:

1. establish the fresh canonical baseline and selfish target gap;
2. run unilateral ceiling tomography;
3. run the cheap CPU covariance and absolute power falsifier;
4. requalify the high-fidelity Anchor estimand on tractable roots;
5. run the NX semantic-compiler/rules-engine preflight and complete `B_k`
   policy-trace adapter;
6. train a small RivalNet only if that preflight and work envelope pass;
7. admit Rival-MF only after analytic bounded inference, untouched coverage,
   and equal-wall/absolute-throughput gains;
8. run one complete frozen relabel iteration with `M_(k+1)` as the only
   promotion candidate;
9. require fresh paired gameplay before a second and final iteration; and
10. finish with the unchanged 1,000-game, four-player, pre-habitat-bonus target.

The most likely successful version is not the most exotic one. It is:

> the current strongest transformer policy, protected as a literal anchor,
> corrected at a small number of high-value decisions by exact terminal own
> score, with a cheap structured selfish policy used only where measured
> multifidelity statistics make that correction affordable, and then
> distilled through at most two evidence-gated terminal relabel iterations.

This recommendation is bold where the repository needs a new source of
decision information and conservative where it already owns a strong policy.
It rejects cooperation completely. It also rejects the more subtle optimism
of treating a fast evaluator, an exact endpoint, or a local confidence bound
as game strength before complete games say so.

## 25. Primary literature and exact transfer claims

The table lists the load-bearing external sources. Broader stochastic-game,
representation, and search surveys remain in the three predecessor proposals.

| Primary source | What is published | What Rival transfers; what remains unproved |
| --- | --- | --- |
| [Luckhardt and Irani, *An Algorithmic Solution of N-Person Games* (AAAI 1986)](https://23.aaai.org/Library/AAAI/1986/aaai86-025.php) | `max^n` for constant- and non-constant-sum multiplayer game trees using vector payoffs. | Supports own-coordinate selfish backup semantics. It does not solve Cascadia's stochasticity, approximation, or hidden-future boundary. |
| [Tesauro and Galperin, *On-line Policy Improvement using Monte-Carlo Search* (NeurIPS 1996)](https://proceedings.neurips.cc/paper/1996/hash/996009f2374006606f4c0b0fda878af1-Abstract.html) | Root actions evaluated by terminal Monte Carlo continuation under a base policy; parallel rollout and backgammon decision-error evidence. | Direct precedent for incumbent-anchored one-deviation rollout. Cascadia cost, multiplayer semantics, and effect size remain open. |
| [Bertsekas and Castanon, *Rollout Algorithms for Stochastic Scheduling Problems*](https://doi.org/10.1023/A:1009634810396) | Rollout as an approximate one-step policy-iteration method, with conditional improvement properties. | Supports the policy-iteration framing. Candidate pruning and approximate stochastic multiplayer continuation prevent importing a safety theorem. |
| [Danihelka et al., *Policy improvement by planning with Gumbel*](https://openreview.net/forum?id=bERaNdoegnO) | Gumbel action sampling without replacement, sequential halving, and policy-improvement operators. | Supports the incumbent candidate machinery already in v3. It does not establish terminal rollout or Rival-MF gains. |
| [Castellini et al., *Scalable Safe Policy Improvement via Monte Carlo Tree Search* (ICML 2023)](https://proceedings.mlr.press/v202/castellini23a.html) | Local online MCTS-SPIBB with convergence to the paper's safely improved policy under its assumptions. | Supports baseline-preserving local improvement as a design principle. Rival does not claim the theorem under Cascadia's model and inference. |
| [Laroche, Trichelair, and Tachet des Combes, *SPIBB*](https://proceedings.mlr.press/v97/laroche19a.html) | Baseline bootstrapping in insufficiently supported finite-MDP regions. | Supports literal fallback. The finite-MDP/batch guarantees are not claimed for Rival. |
| [Peherstorfer, Willcox, and Gunzburger, *Optimal Model Management for Multifidelity Monte Carlo Estimation*](https://epubs.siam.org/doi/10.1137/15M1046472) | An unbiased high-fidelity-statistic estimator using correlated surrogate models and optimized sample allocation under stated conditions. | Supports many cheap plus occasional expensive evaluations. Applying it to selected incumbent-relative Cascadia action differences is an original synthesis requiring split selection, fixed coefficients, marginal correctness, and coverage proof. |
| [Khairy and Balaprakash, *Multifidelity Reinforcement Learning with Control Variates* (Neurocomputing 2024)](https://doi.org/10.1016/j.neucom.2024.127963) | A control-variate estimator exploiting low/high-return correlations for state-action value estimation, with theoretical analysis and numerical experiments. | Supports training for return correlation and multifidelity policy evaluation. It does not validate Rival's multiplayer continuation or serving rule. |
| [Veness, Lanctot, and Bowling, *Variance Reduction in Monte Carlo Tree Search*](https://proceedings.neurips.cc/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html) | Common-random-number, antithetic, and control-variate techniques for MCTS, evaluated in three stochastic single-agent settings. | Supports paired action differences and careful coupling. Cascadia's multiplayer dynamic-urn event coupling must be proved separately. |
| [Howard et al., *Time-uniform, nonparametric, nonasymptotic confidence sequences*](https://doi.org/10.1214/20-AOS1991) | Anytime-valid confidence sequences for bounded/controlled observations under the paper's conditions. | Available if later confirmation truly needs planned sequential looks. The first Rival design uses fixed confirmation to reduce inference risk. |
| [Lan et al., *Multiple Policy Value Monte Carlo Tree Search* (IJCAI 2019)](https://www.ijcai.org/proceedings/2019/653) | Small and large policy-value networks combined in MCTS outperformed single-network controls in NoGo experiments. | Supports a measured dual-fidelity model role. It does not establish that Cascadia's cheap model will retain the needed differences. |
| [Stockfish, *Introducing NNUE Evaluation*](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/) | Efficiently updatable neural evaluation exploiting small chess-position changes inside exact search. | Supports the factor/delta economics hypothesis. Cascadia's wide dependency closure and GPU hardware make dense recomputation a live competitor. |
| [Stockfish 16.1](https://stockfishchess.org/blog/2024/stockfish-16-1/) | Official release introduced a secondary NNUE used for positions considered easily decided. | Supports conditional cheap/full evaluation. It is chess CPU evidence, not Cascadia strength evidence. |
| [Koyamada et al., *Pgx*](https://arxiv.org/abs/2303.17503) | Accelerator-native board-game simulation with reported 10--100x throughput over compared Python environments and a Gumbel AlphaZero implementation. | Supports feasibility of a resident batched simulator. It does not establish Cascadia parity or RTX-5090 throughput. |
| [DeepMind Mctx](https://github.com/google-deepmind/mctx) | Open accelerator-native batched MCTS reference implementation. | Supports accelerator-native batched tree-search patterns. Rival's wavefront interpretation, variable menus, exact rules, and bridge must be measured locally. |
| [Sezener and Dayan, *Static and Dynamic Values of Computation in MCTS*](https://arxiv.org/abs/2002.04335) | Explicit computation-value policies with theoretical conditions and competitive experiments. | Supports a later learned compute router. Rival v1 uses fixed strata until unbiased labels and costs exist. |
| [Anthony, Tian, and Barber, *Thinking Fast and Slow with Deep Learning and Tree Search*](https://arxiv.org/abs/1705.08439) | Expert Iteration distills tree-search experts into an apprentice and reported Hex progress. | Supports frozen search-to-policy iteration. It does not prove that Cascadia terminal corrections survive retraining. |
| [Zha et al., *DouZero*](https://proceedings.mlr.press/v139/zha21a.html) | Direct Monte Carlo returns and parallel actors in a three-player stochastic card game, with published leaderboard performance. | Supports exact terminal-return learning in multiplayer stochastic games. Hidden information and DouDizhu's objective differ materially. |
| [Schrittwieser et al., *Planning in Stochastic Environments with a Learned Model*](https://openreview.net/forum?id=X6D9bAHhBQ1) | Explicit decision-afterstate/chance factorization with 2048 and backgammon results. | Supports separating chosen actions from chance. Rival keeps exact known rules rather than learning dynamics. |
| [Wang et al., *Adversarial Policies Beat Superhuman Go AIs*](https://arxiv.org/abs/2211.00241) | Trained adversarial policies exposed transferable failure modes in superhuman Go agents. | Supports a bounded exploit audit. It does not support changing Cascadia's reward or making adversarial training primary. |

### 25.1 Document-specific syntheses not found in the cited search

No source above published this exact integrated Cascadia system. The following
are document-specific application choices or syntheses not found in the cited
search; this is not a universal nonpublication claim:

- using a cheap selfish continuation as a multifidelity control variate for an
  incumbent-relative terminal **action difference**;
- optimizing RivalNet for terminal-difference correlation per complete-
  trajectory wall-second;
- semantic-event coupling for Cascadia's dynamic supply, with independent
  futures as the fail-closed alternative;
- the exact `S/H/L` proposal-confirmation split for this estimator;
- seat-local score contracts admitted only by incremental confirmed-
  challenger yield;
- unilateral ceiling tomography with explicit witness/heuristic/bound labels;
- the combined transformer-anchor, RivalNet, appeals-court, and frozen policy-
  iteration system; and
- the confidence and kill ladder in this report.

These are hypotheses. Distinct integration is not evidence of correctness or
strength.

## 26. Repository evidence and context map

Read current state before this proposal:

- [v3 authoritative README](docs/v3/README.md)
- [campaign state and RESUME HERE](docs/v3/CAMPAIGN_STATE.md)
- [research verdict log](docs/v3/RESEARCH_LOG.md)
- [living research agenda](docs/v3/RESEARCH_AGENDA.md)
- [current architecture](docs/v3/ARCHITECTURE.md)
- [rules and serving contract](docs/v3/RULES_CONTRACT.md)
- [radical-directions registry](docs/v3/RADICAL_DIRECTIONS.md)
- [experiment ledger](cascadiav3/EXPERIMENT_LOG.md)

July 16 question and answer context:

- [external research questions](research_questions_7_16.md)
- [literature-backed answers](research_answers_7_16.md)
- [ranked architecture critique](architecture_proposal_critiques_7_16.md)

Predecessor proposals:

- [Cascadia-NX](stochastic_board_game_ai_architecture_research_7_16.md)
- [Cascadia-Anchor](incumbent_anchored_gpu_rollout_policy_improvement_7_16.md)
- [Cascadia Foundry](cascadia_foundry_original_architecture_proposal_7_16.md)

This document supersedes their **combined architecture recommendation**. It
does not erase their detailed source audits, implementation alternatives, or
historical role. Cascadia-NX remains the structured-architecture study;
Cascadia-Anchor remains the full terminal-rollout design; Foundry remains an
original-program-synthesis exploration whose cooperative controller is now
withdrawn.
