# Cascadia Foundry: nonanticipative score-contract program synthesis

**Date:** 2026-07-16

**Status:** OPEN POST-D1 HYPOTHESIS; D1 remains first; zero gameplay evidence

**Objective:** mean seat score at least 100 over 1,000 complete four-player games,
before habitat bonus, under one pinned rules and serving identity

**Design-confidence forecast:** 76%, conditional on the baseline,
methodology, headroom, and systems gates in this report; this is an explicit
subjective forecast, not a measured probability

## Executive verdict

The third proposal should not be another evaluator, another tree search, or a
larger learned policy. It should replace the object being optimized.

The current stack asks, repeatedly:

> What scalar value should I assign to this partial board after this action?

**Cascadia Foundry** asks a different question:

> Which concrete terminal constructions scoring at least 100 are still reachable,
> what public resources and spatial obligations do they require, what reactive
> policy can keep them reachable when the market changes, and which root action
> is supported by the largest diverse population of such successful policies?

Foundry is a GPU-resident population of **executable score-witness terminal
blueprints** and tiny **public-state controller programs**. It plans backward
from exact terminal score, repairs plans as resources disappear, and evaluates
complete reactive controllers on exact dynamic-urn scenario streams. It never
predicts a scalar Q, never learns game dynamics, and never searches the
ordinary move tree.

Its highest-confidence mode also takes the benchmark literally. With habitat
bonuses disabled, the four-player mean is exactly one quarter of table score:

\[
J(\pi)=\mathbb E_\xi\left[
\frac14\sum_{j=0}^{3}S(B_j^{20}(\pi,\xi))
\right].
\]

When the same system controls all four seats, this is one cooperative,
80-decision, public-information resource-allocation problem. It is not four
simultaneous adversaries. The four boards compete only for the shared market
and supply. Foundry therefore makes all four boards bid for scarce future
resources and deliberately leaves an item for a later controlled seat when
that raises total expected table score.

That objective choice is bold and load-bearing. It directly optimizes the
stated self-play metric, but it is not the same claim as “best selfish agent
against arbitrary opponents.” John must explicitly accept the cooperative
benchmark interpretation before a gameplay arm launches. Foundry also defines
a selfish **Sovereign** control so the contribution of cooperation is measured,
not hidden.

The architecture has six proposed original mechanisms:

1. **Score contracts:** exact terminal scoring witnesses become the deployed
   planning representation.
2. **Completion genomes:** each plan contains a legal placement DAG, resource
   obligations, substitutions, and a repair grammar—not a future action
   sequence.
3. **The Score Futures Exchange:** counterfactual plan collapse produces
   scarcity prices and cross-board market opportunity costs.
4. **Scenario braiding:** one reactive public-history program controls every
   sampled hidden world, preventing strategy fusion while preserving genuine
   adaptation after public reveals.
5. **Canonical archive support:** the chosen action is the one supported by
   the largest diverse, lineage-deduplicated population of high-scoring
   programs, not the action of one lucky optimizer winner.
6. **Commitment-collapse refinement:** exact failures expand the blueprint or
   controller language at the missing causal distinction instead of adding
   generic model capacity.

Three independent problem-solving agents, asked to invent separate clean-sheet
systems, converged on this terminal-first architecture. One independently
derived the score-contract lattice, one the score-witness GPU blueprint
portfolio, and one the reactive-program/policy-basin formulation. Convergence
is not evidence that it works, but it is evidence that the proposal follows
naturally from the game’s actual structure rather than from one favorite ML
template.

My subjective design confidence is **76% that the complete Foundry program can
produce a frozen policy whose true mean reaches at least 100**, conditional on all four
of these premises:

- the valid July-16 baseline is at least approximately 98.0;
- cooperative table mean is accepted as the internal objective;
- ceiling tomography exposes at least 10 table points (2.5 mean-seat points)
  of honest, nonanticipatively recoverable headroom; and
- the exact GPU simulator sustains the registered throughput bar.

That number is not promotion evidence. Before those premises are measured,
Foundry has zero current-rules strength evidence. If a premise fails, the
forecast falls and the corresponding form closes. The report makes the 75%
claim falsifiable instead of decorating an untested architecture with false
precision.

## 1. The first-principles reframe

### 1.1 Cascadia is monotone construction under shared supply

For one seat under the pinned all-A, no-habitat-bonus identity, terminal score
is exactly:

\[
G(B)=
\sum_{h=1}^{5}L_h(B)
+S_{\mathrm{bear}}(B)
+S_{\mathrm{elk}}(B)
+S_{\mathrm{salmon}}(B)
+S_{\mathrm{hawk}}(B)
+S_{\mathrm{fox}}(B)
+N(B).
\]

Every personal turn adds one habitat tile and may add one wildlife token. A
placed tile is never removed. An opponent cannot damage another board; it can
only consume or expose shared market resources. Every score term has a small,
constructive witness:

- Bear A: disjoint pairs, with exact breakpoints at one through four pairs;
- Elk A: disjoint straight lines of length one through four;
- Salmon A: nonbranching connected runs;
- Hawk A: isolated hawks;
- Fox A: distinct neighboring wildlife types;
- habitat: largest connected component for each of five terrains; and
- Nature Tokens: exact keystone acquisition, spending, and terminal retention.

The board is therefore not an arbitrary image. It is a partially completed
constraint object with visible obligations, conflicts, deadlines, and legal
substitutions.

### 1.2 The true benchmark is a team MDP

Let \(s_t\) be the complete public table state and \(a_t\) the active seat’s
legal action. Because exactly one seat acts at a time and all four boards are
public, homogeneous four-seat self-play can be represented as one finite team
MDP:

\[
s_{t+1}\sim P(\cdot\mid s_t,a_t),
\qquad
r_t=0\ \text{for}\ t<80,
\qquad
r_{80}=\frac14\sum_j S(B_j).
\]

There is no mathematical need for `max^n` under that metric. A selfish policy
is a methodological convention, not an identity of the objective. A resource
that produces 0.4 points for the current board but 1.2 for the next controlled
board should be left, if it is likely to survive and doing so maximizes table
score.

The historical July-9 champion mean was 98.2975. Crossing 100 from that
reference requires 6.81 additional table points per game. Across 80 decisions,
that is only 0.085 table point per decision on average. Complementarities mean
those gains will not accrue linearly, but the scale is important: Foundry does
not need to discover a new game. It needs to stop destroying roughly one Bear
pair, one habitat connection, or several small market opportunities across the
whole table.

### 1.3 Why scalar value is the wrong intermediate object

The measured root problem has a median top-two completed-Q gap of 0.049 and a
median pairwise standard error of 0.051. Nearly half of decisions are
noise-flippable. Increasing model size, data, ordinary simulations, and several
value-head variants did not remove the plateau.

Foundry does not attempt a better scalar approximation to the same noisy
quantity. It carries the actual long-term commitments:

- “two more Bear-compatible sockets complete the third pair”;
- “this exposed habitat port can merge components of sizes four and three”;
- “a Salmon endpoint must remain unbranched”;
- “this Hawk reservation forbids a neighboring Hawk”;
- “this Fox socket needs two still-missing wildlife types”;
- “this tile role has three exact catalog substitutions”; and
- “this resource is more valuable to seat two before the market cycles.”

An action is valuable when it satisfies, preserves, or cheaply repairs many
high-scoring futures—not because one network emits 0.07 more Q.

## 2. Claim boundary and evidence labels

This report distinguishes five kinds of statement.

- **Identity:** follows exactly from the pinned rules or metric, such as mean
  seat score equaling table score divided by four.
- **Repo-direct:** measured or implemented in the repository under the stated
  historical identity.
- **Adjacent published work:** a known ingredient or neighboring method, not
  evidence for Foundry.
- **Original synthesis:** a mechanism designed in this report and not located
  in the targeted prior-art search.
- **Forecast:** an explicit subjective probability or engineering projection,
  never strength evidence.

Important boundaries:

- No admissible July-16 baseline battery exists at this report’s cutoff.
- The 98.2975 score and 1.2038 game-mean standard deviation come from the
  complete July-9 historical ledger and are planning references only.
- Foundry has not generated one current-rules action, plan, or game.
- “Original” means the named mechanism and integrated architecture were
  created here and not found in the targeted search. It cannot prove that no
  person anywhere independently conceived a related fragment.
- The 76% forecast is conditional and subjective. The repository’s promotion
  rules, paired confidence interval, and 1,000-game target remain unchanged.
- A cooperative self-play result is labeled as such. It does not establish
  selfish head-to-head strength against humans or unrelated agents.

## 3. System overview

```text
                  exact A-card scorer and rules
                              |
                    terminal design foundry
                              |
              score-witness completion blueprints
                              |
       +----------------------+----------------------+
       |                      |                      |
  seat-0 portfolio       seat-1 portfolio       seats 2/3
       |                      |                      |
       +---------- Score Futures Exchange ---------+
                              |
             reactive controller-program population
                              |
        exact public-consistent dynamic scenario streams
                              |
                     scenario-braided GPU run
                              |
            direct exact terminal objective fitness
                              |
          diverse high-score canonical archive support
                              |
               freeze complete policy capsule
                              |
             fresh-stream full-capsule check
                              |
            committed public policy through terminal
```

The complete policy has four layers:

1. **Compile:** construct exact-scored terminal boards and reverse them into
   alternative-rich completion contracts.
2. **Bind and repair:** match the current four boards to compatible plans and
   repair plans around irreversible actual placements.
3. **Synthesize:** evolve tiny public-state programs and cross-board scarcity
   prices over exact dynamic-urn scenario streams.
4. **Commit:** freeze the canonical archive into a complete reactive capsule,
   compare that capsule with the registered default on fresh scenario streams,
   and follow the winner through terminal.

The committed plan population persists between real turns without new
scenario optimization. Long-term intent is stored as explicit public policy
memory rather than rediscovered from a tensor every decision.

## 4. Score contracts: exact terminal score as representation

### 4.1 Contract definition

A score contract is a machine-verifiable claim:

> If this set of spatial, resource, and incompatibility obligations is
> fulfilled, the materialized terminal extension is legal and scores exactly
> \(y\) under the pinned rules identity.

One concrete representation is:

```rust
struct ScoreContract {
    contract_id: Hash,
    rules_id: Hash,
    canonical_terminal_board: PackedBoard,
    exact_score: ScoreBreakdown,
    placement_dag: BitMatrix,
    clauses: Vec<Clause>,
    resource_requirements: ResourceVector,
    conflict_mask: BitSet,
    repair_grammar: Vec<RepairRule>,
    d6_canonical_hash: Hash,
}

struct Clause {
    kind: ClauseKind,
    cells: CellMask,
    legal_tile_ids: TileBitSet,
    legal_rotations: u8,
    wildlife_domain: u8,
    prerequisites: BitSet,
    diagnostic_attribution_delta: i16,
    deadline_slack: u8,
}
```

The terminal board is retained. Clauses accelerate matching and repair; they
never replace exact materialization and rescoring. Because scoring obligations
overlap, a per-clause delta is diagnostic only under a frozen attribution
order; it is never treated as an independently exact, additive score.

Under the no-habitat-bonus identity, every admitted contract must also satisfy
`exact_score.total == exact_score.base_total` and carry a zero habitat-bonus
vector. A mismatch is an identity failure, not a soft scoring discrepancy.

### 4.2 Contract families for AAAAA

The first grammar contains:

- `BearPair(k)`: create the \(k\)-th exact pair while preventing a third Bear
  from joining the component;
- `ElkLine(length, axis)`: reserve a straight disjoint line and its valid
  construction orders;
- `SalmonRun(length, endpoints)`: extend a connected component while every
  Salmon retains degree at most two; `endpoints` permits two endpoints for a
  path or zero for a valid cycle;
- `HawkIsolation(cell)`: preserve a no-Hawk adjacency exclusion zone;
- `FoxDiversity(cell, types)`: fill a Fox neighborhood with distinct species;
- `HabitatPort(terrain, component_a, component_b)`: preserve or realize a
  component merge;
- `KeystoneOption(tile, wildlife)`: model the exact Nature Token gain and its
  later draft/wipe option; and
- compound clauses whose one tile/wildlife placement advances multiple score
  systems.

The exact scorer provides both payoff and counterexamples. If a clause claims
an 8-point Salmon run but materialization scores 0 because it branches, the
contract is invalid and the compiler fails closed.

### 4.3 Terminal design generation and backward completion lattice

The offline foundry does not try to enumerate every possible board. It creates
legal 20-turn construction traces with a staged hybrid:

1. seed from completed legal games and score-system-specific constructive
   templates;
2. run quality-diversity large-neighborhood search over tile identity,
   rotation, coordinate, wildlife assignment, and legal build order;
3. apply exact bounded repair to four-to-eight-cell regions around the weakest
   score witness;
4. retain only candidates whose complete forward replay, catalog use, wildlife
   use, Nature accounting, and exact `ScoreBreakdown` verify; and
5. archive distinct terminal designs in score strata 90, 95, 100, 105, 110,
   and 115+.

This is deliberately solver-agnostic at the outer loop: evolutionary search,
constraint programming, or both may propose candidates, but only the existing
Rust transition and scorer admit them. The local-hardware feasibility question
is witness yield per CPU-hour, not whether the proposal can exactly enumerate
the global board space.

For every admitted terminal board, repeatedly remove non-starter additions in
every connectivity-preserving reverse order. Merge states that are identical
under:

- D6 rotation/reflection;
- materialized fixed cells;
- unsatisfied obligation signature;
- exact resource domains; and
- placement-precedence constraints.

This yields an AND/OR completion lattice rather than one brittle script. One
108-point board may have thousands of legal build orders and many catalog
substitutions. The lattice records that alternative volume explicitly.

Live boards will rarely be exact subgraphs of an archived design. The compiler
therefore freezes all real cells and applies bounded destroy-and-repair only to
the unfilled intent region. No actual placement is moved in an online plan.

### 4.4 Score witnesses prove spatial construction, not acquisition history

Each admitted blueprint carries:

- complete terminal board serialization;
- exact legal growth order or placement DAG;
- tile identity/rotation assignments;
- wildlife assignments;
- score breakdown from the Rust oracle;
- rules and catalog hashes; and
- D6 canonicalization witness.

This proves that the proposed terminal construction exists, has a legal
spatial growth order given the required resources, and scores as claimed. It
does **not** prove that paired market arrivals, same-turn wildlife placement,
Nature timing, or stochastic play can realize that resource sequence, and it
does not prove optimality. Only a complete chronological `GameState` replay
certifies history reachability. Online reachability belongs to the reactive
controller and scenario evaluation.

## 5. Completion genomes and repair grammars

### 5.1 Persistent four-board genome

A policy particle carries four coordinated plans:

\[
g=(P_0,P_1,P_2,P_3,\Gamma,\lambda,\theta,m),
\]

where:

- \(P_j\) is seat \(j\)’s completion blueprint portfolio member;
- \(\Gamma\) is a bounded repair grammar;
- \(\lambda\) is the shared-resource price vector;
- \(\theta\) parameterizes a tiny semantic controller program; and
- \(m\) is public-history-only persistent memory.

A plan contains:

```rust
struct CompletionPlan {
    terminal_contract: Hash,
    embedding: PackedBinding,
    satisfied: BitSet,
    enabled_steps: BitSet,
    live_substitutions: Vec<ResourceDomain>,
    exact_score_floor: u16,
    exact_score_target: u16,
    remaining_turns: u8,
    deadline_slack: i8,
    resource_demand: ResourceVector,
    repair_budget: u8,
    lineage_id: Hash,
}
```

An individually valid contract is not automatically a valid four-seat genome.
The joint binder rejects any genome whose four plans exceed the pinned global
tile catalog, wildlife inventory, Nature accounting, or remaining public
supply. The Futures Exchange may price a relaxed allocation, but relaxed
infeasible plans never enter exact scenario fitness as though they were legal.

### 5.2 Semantic repair operators

Mutations are legal game concepts, not arbitrary vector noise:

- attach an available tile to an enabled coordinate/rotation;
- substitute a provably equivalent tile role;
- reroute or bridge one habitat boundary;
- move, split, complete, or abandon a Bear pair intent;
- extend, rotate, or redirect an Elk line;
- extend a Salmon endpoint without branching;
- restore Hawk isolation;
- rewire a Fox neighborhood;
- exchange wildlife roles between compatible cells;
- reorder independent placement-DAG steps;
- reserve, spend, or release a Nature Token option;
- transfer resource priority between controlled boards; and
- destroy and rebuild a bounded unfilled terminal region.

Every repaired plan is materialized and rescored. An optional learned repair
ranker may order operators, but it cannot assert legality, score, or policy
value.

### 5.3 Canonical option volume

Raw plan count is meaningless because a generator can duplicate one plan. The
population deduplicates by terminal contract, placement partial order,
resource-demand signature, repair set, and D6 orbit. Diversity is tracked in a
MAP-Elites-style archive whose cells are semantic:

- wildlife portfolio shape;
- habitat topology;
- resource fragility;
- deadline profile;
- cross-board demand balance; and
- first-action identity.

The **effective option volume** of a state is the weighted number of distinct,
still-reachable plan/controller basins after lineage and semantic
deduplication. It rewards multiple ways to succeed, not repeated copies of one
hope.

## 6. The reactive controller language

### 6.1 Programs, not open-loop sequences

A hidden future is never optimized as an action sequence. Each particle owns
one deterministic 32–64 instruction program that maps public state and public
plan memory to a legal action.

Public terminals include:

- exact immediate wildlife, habitat, and Nature deltas;
- distance and gain to every A-card scoring breakpoint;
- habitat component sizes, merge counts, frontier capacity, and exposed
  edges;
- contract compatibility and minimum repair cost;
- remaining public tile/wildlife counts;
- demand-to-supply slack and obligation deadline;
- Nature Token option value;
- market pair coupling cost;
- each board’s marginal use for a visible item;
- estimated survival of a pair until another controlled seat acts;
- phase and turns remaining; and
- geometric flexibility after placement.

Operators are fixed-point and branch-light: add, subtract, min, max, clamp,
bounded product, comparisons, phase gates, and lexicographic composition.
D6 canonicalization normalizes geometry.

For legal action \(a\), a program computes:

\[
u_g(s,a)=
\Delta S_{\mathrm{exact}}(s,a)
+\Delta\Phi_g(s,a)
+\Omega_g(s,a)
-C_g(s,a),
\]

where:

- \(\Phi_g\) is repaired terminal-contract potential;
- \(\Omega_g\) is cross-board market opportunity value; and
- \(C_g\) is fragility, substitution, and deadline cost.

The legal argmax uses a pinned deterministic tie break. The full policy is
small enough to execute millions of times without transformer inference.

The first correct implementation enumerates the full legal compound-action
menu. A plan-directed menu is a later optimization only after Gate 2 proves it
retains the registered fraction of the unrestricted honest-controller ceiling.
No blueprint may silently make an unrepresented legal action disappear.

### 6.2 Controller memory is public policy state

Persistent memory may store:

- active blueprint IDs and embeddings;
- publicly observed resource counts;
- fulfilled/expired obligations;
- current scarcity prices;
- repair history; and
- lineage weights aggregated over independently sampled counterfactual panels.

It may not store a sampled tape ID, a per-tape future fact, an unrevealed tile
identity, or anything derived from the actual game’s hidden suffix. Aggregate
search statistics are allowed only when their counterfactual panels and RNG
domains are derived from public state and are independent of that actual
suffix. Replaying the same public history and serving seed identity must
recreate the same memory hash.

## 7. The Score Futures Exchange

### 7.1 Counterfactual scarcity prices

Resource index \(r\) is atomic: an exact catalog tile identity or one physical
wildlife token. Semantic roles such as “Bear-compatible habitat bridge” are
incidence views over those atoms, never additional inventory. Market-slot pair
coupling, excluded tiles, without-replacement uncertainty, and Nature
transactions are separate constraints.

For board \(j\), define \(\Phi(\mathcal P_j)\) in point units as the fixed-base-
measure weighted exact terminal score of its canonical plans after bounded
public-state repair; collapsed plans receive the registered current-board
floor rather than disappearing from the denominator. Effective option volume
is a separate tie-break/diagnostic and is not blended into \(\Phi\). Remove one
publicly unplaced atomic resource \(r\), repair the portfolio, and define:

\[
\lambda_{j,r}=
\Phi(\mathcal P_j)
-\Phi(\operatorname{Repair}(\mathcal P_j,r^-)).
\]

This is the board’s finite-difference, point-denominated bid under the frozen
portfolio model. It is not a certified causal point value; complete exact
scenario play decides whether the predicted collapse matters.

The global relaxed allocation is:

\[
\max_{P_0,\ldots,P_3}\sum_j S(P_j)
\quad\text{subject to}\quad
\sum_j d_{jr}\le \bar n_r
\quad\forall r.
\]

An iterative price update is:

\[
\lambda_r\leftarrow
\left[
\lambda_r+\eta\left(\sum_j d_{jr}-\bar n_r\right)
\right]_+.
\]

The relaxation does not choose the final action and does not certify value. It
uses catalog-level consumption and explicit incidence constraints so one
physical item cannot satisfy two boards merely because it occupies two
semantic roles. It helps the controller represent shared scarcity. Complete
exact scenarios decide fitness.

### 7.2 Market donation

For each visible pair, the four portfolios submit marginal score bids. The
current seat’s action accounts for:

- its own exact and contract gain;
- which three pairs remain visible;
- the probability each remaining pair survives to a later controlled seat;
- which board can convert it into the most table score; and
- replacement-resource effects from the selected slot.

“Donation” does not move an item directly to another seat. It means selecting a
different legal pair because the opportunity value of leaving this one exceeds
the current board’s marginal use. Exact simulation handles whether it actually
survives.

### 7.3 Commons and Sovereign modes

Two modes isolate methodology and mechanism.

| Mode | Fitness | Cross-board price use | Claim |
| --- | --- | --- | --- |
| Foundry-Commons | exact terminal table score / 4 | shared central memory and prices are load-bearing | directly optimizes the central-control interpretation of the self-play metric |
| Foundry-Sovereign | one seat-local controller’s exact score | no shared memory; public opponent demand predicts pressure only | conventional selfish-agent control |

Sovereign does not use one four-board genome whose later seats can collude with
the root. During unilateral development, one seat-local capsule controls the
target seat against frozen opponent policies. During symmetric evaluation, the
same frozen seat-local capsule is instantiated four times with isolated memory,
cyclic seat-label equivariance, and no cross-seat lineages or prices. This
separates representation gains from central cooperation.

Commons is explicitly a central joint controller. It may share plans, prices,
and public memory across seats and may donate a resource. That is not silently
called “four identical agents”: the policy identity records central control,
and Gate −1 rules on its admissibility. Commons must also pass cyclic
seat-permutation traces plus per-seat/category fairness reporting so a mean
gain cannot hide absolute-seat specialization or pathological sacrifice.

The 76% forecast is for Commons. My corresponding forecast for Sovereign alone
is approximately 55%. A Commons win must never be reported as a generic
competitive-agent result.

This does not reopen the two failed learned table-total serving variants. They
remain closed. Foundry-Commons is not eligible for a gameplay arm merely
because its objective is mathematically aligned with the benchmark. First, the
zero-gameplay tomography in §12 must produce materially new evidence that
exact whole-policy coordination has enough honest headroom; then John must
rule that the cooperative interpretation is an acceptable methodology. Without
both conditions, only Sovereign survives.

## 8. Scenario braiding and the information boundary

### 8.1 One program across every hidden world

At public state \(s\), sample exact conditional hidden-supply scenario streams:

\[
\Xi=\{\xi_1,\ldots,\xi_K\}.
\]

“Tape” below is shorthand for a counter-based scenario RNG stream plus the
conditional hidden inventory—not a fixed precomputed item suffix. The exact
engine consumes that stream through dynamic urn operations. Wildlife returns,
wipes, exclusions, and later draws therefore change the realized sequence
exactly as they do in the real game. The scenario stream is independent of the
physical game seed.

The simulator owns each stream. The controller receives only the public market
after the engine reveals it. For any two lanes:

\[
h_t(\xi)=h_t(\xi')
\Longrightarrow
\pi_g(h_t(\xi))=\pi_g(h_t(\xi')).
\]

Trajectories may diverge only after their public histories diverge. This is the
**scenario braid**: all worlds are tied by one reactive program before a
reveal, then branch honestly after different observations.

The root-search RNG domain is itself derived from the public-history hash and
the frozen serving seed identity. It determines the scenario panel, mutation,
and resampling streams, so exact replay of the same public history reconstructs
the same computation. The real hidden inventory, physical seed, future
wildlife-return schedule, and scenario stream are never policy RNG inputs.

This is stricter than merely deleting hidden features from a tensor.
Scenario-specific optimizers can leak future knowledge through the action they
select even if the final policy input appears public. Foundry forbids separate
per-world action programs.

### 8.2 Mandatory strategy-fusion audit

Each evaluation cohort has an immutable manifest keyed by root genome ID:
controller hash, four blueprint/binding hashes, initial public-memory hash,
repair/mutation state, and public policy-RNG identity. Every lane must match
that manifest before simulation; lane-local optimization is forbidden.

Every simulated decision logs:

- root genome/lineage ID;
- canonical public-history hash;
- controller-program hash;
- public memory hash;
- public-derived search-RNG hash;
- legal-menu hash;
- selected action; and
- scenario-stream ID only in a separate simulator audit field.

Before comparing actions, assert that every lane for one root genome carries
the same expected program, blueprint, initial memory, and mutation-state
hashes; a lane-specific assignment is itself an invalid arm. Then, for a fixed
genome, if two lanes have the same public history and legal menu, their derived
memory and action must match regardless of tape ID. Separately, exact root
replay must
reproduce the scenario-set, population, and elected-action hashes. Test at
least 10,000 hidden permutations per curated state family and 10,000 complete
policy games before strength work. Curated collisions must hold public
serialization fixed while independently changing stack order, excluded tiles,
bag order, physical seed, and future wildlife-return schedule.

### 8.3 Refresh and wipe semantics

The free three-of-a-kind branch remains:

```text
public prelude -> accept/decline decision -> chance replacement -> public draft
```

The controller chooses accept/decline from public information. If accepted,
the engine reveals the replacement and only then does the same program choose
the draft. It may not encode a downstream draft selected using an unrevealed
market. Paid wipes and independent Nature Token drafts follow the exact engine
transaction and public boundary.

## 9. Canonical archive support: broad success, not one lucky winner

### 9.1 Direct terminal fitness

For a design panel \(\Xi_D\), a complete reactive genome has fitness:

\[
J_D(g,s)=
\frac{1}{|\Xi_D|}
\sum_{\xi\in\Xi_D}
\frac14\sum_{j=0}^{3}S_j(g,\xi).
\]

There is no Q target, TD target, value bootstrap, learned world model, or
terminal-score surrogate.

All genomes within a generation see the same conditional physical tapes so
their differences are paired. Each generation rotates to a fresh,
public-hash-derived panel; confirmation panels are never used for mutation or
resampling. A disjoint calibration block compares paired and independent-tape
difference variance. Shared physical tapes are retained only if each policy's
marginal is exact and the measured covariance is useful; otherwise the
election uses independent panels. This is not the previously failed R0.2
rollout-RNG coupling.

### 9.2 Generator-relative empirical policy-basin support

Choosing the single highest-scoring evolved genome recreates the optimizer’s
curse. Foundry therefore builds \(R\) independent fixed-budget
quality-diversity archives and canonicalizes their union into
\(\mathcal A_D(s)\). At most one representative occupies each registered
`semantic cell x lineage` slot. Every surviving representative is then
re-evaluated on one frozen final design block that was not used to generate or
mutate it.

For canonical archive member \(g\), define a frozen score weight

\[
w_g = \exp\{\beta J_D(g,s)\},
\]

and the root action's empirical support

\[
\widehat M_D(a\mid s)=
\frac{
  \sum_{g\in\mathcal A_D(s)}w_g\mathbf 1[\pi_g(s)=a]
}{
  \sum_{g\in\mathcal A_D(s)}w_g
}.
\]

The archive generator may resample and mutate candidates during discovery,
but those operations are **not** claimed to sample a formal posterior. The
statistic above is exact only relative to the frozen generator, semantic
partition, lineage rule, design block, and temperature. It is neither an
unbiased estimate of global policy-space volume nor invariant to a different
generator. Independent archive initializations and sensitivity to every one of
those choices are therefore part of Gate 3.

The discovery loop is:

1. initialize independent archive islands from pinned seeds and library
   members;
2. evaluate candidates on the current shared design panel;
3. retain high-scoring candidates within semantic cells;
4. apply score-aware repair/program mutations under a frozen budget;
5. rotate the exploration panel and repeat;
6. canonicalize and lineage-deduplicate the final union;
7. re-evaluate all survivors on the frozen final design block; and
8. compute \(\widehat M_D\).

An action supported by many different 101–105-point policies is preferred to
one supported only by a single 110-point policy whose success depends on a
fragile resource sequence. “Policy-basin mass” is shorthand for this explicit
generator-relative empirical support, not a Lebesgue measure or Bayesian
posterior.

### 9.3 The deployed object is a policy capsule

Individual genome score is a discovery signal. It is not automatically the
return of the population-vote policy. Foundry therefore materializes the
winning archive as a **policy capsule** containing the frozen canonical
members, weights, plans, deterministic repair budget, semantic partition,
public RNG identity, and tie contract.

At every later public decision, the capsule:

1. conditions or deterministically repairs each member using public facts;
2. removes invalid or expired contracts;
3. recomputes generator-relative support over the surviving members; and
4. plays the supported legal action under the frozen tie rule.

No mutation, resampling, new hidden-scenario optimization, or archive election
occurs after commitment. The reactive programs and repair grammar still adapt
to public reveals. The complete capsule meta-policy \(\Pi_{\mathcal A}\), not
an individual \(\pi_g\), is simulated to terminal and graded by exact score.

The main proposal commits a capsule through the terminal horizon. A
receding-horizon variant that resynthesizes at later real turns is a different
proxy candidate unless its entire deterministic population-update/election
meta-policy is itself executed inside every scenario. It cannot inherit the
capsule's direct-terminal claim.

### 9.4 Fresh-tape applicability certificate

Discovery and confirmation do not use the same scenario streams. The design
stream nominates exactly one frozen non-default policy capsule. Confirmation
compares that complete capsule with the registered default on fresh streams;
it neither reweights the archive nor selects among its lineages. Physical
streams are paired only if the disjoint variance calibration retains that
coupling; otherwise confirmation uses independent exact marginals and the
corresponding two-sample bound.

1. freeze the capsule members, design-derived weights, root action, repair
   budget, and fallback;
2. discard every design score for inference;
3. execute the complete capsule meta-policy and default to terminal under the
   frozen paired-or-independent scenario rule;
4. compute the registered stream-level lower bound against the practical
   margin;
5. require stable sign across stream halves and independent archive
   initializations; and
6. report archive-support ESS and largest-lineage share as representation
   diagnostics, never as the number of statistical samples.

With exactly one design-nominated capsule, the confirmation contrast has no
within-root candidate multiplicity. If a later phase compares multiple
capsules, it freezes that family before confirmation and uses a simultaneous
max-t or Holm-valid bound. An unresolved comparison commits the registered
default for the entire horizon.

This certificate is not Cascadia-Anchor: the challenger is a whole reactive
capsule with terminal plans, not the incumbent after one deviating action.

## 10. Synthesis boundary and committed serving loop

A first game-start synthesis budget is deliberately modest:

- 512 persistent/library genomes;
- 8 shared design tapes;
- 6 resample/mutation generations;
- top 4 independent lineages per leading action; and
- 64 fresh confirmation tapes.

That is 24,576 discovery continuations at the synthesis boundary before fresh
confirmation.
Because the programs are tiny and the rules are resident, the relevant cost is
exact policy plies, not 88M-parameter inference rows.

The `8` and `64` counts are an engineering starting point, not an assertion of
statistical power. The fixed-root variance/power preflight freezes the actual
design and confirmation counts before shadow or gameplay use. If its required
count misses the 10-second synthesis wall, the online form fails Gate 4.

At the registered synthesis boundary:

1. canonicalize the complete public state;
2. bind library contracts and build independent archive islands;
3. run the frozen discovery panels and budget;
4. materialize one canonical policy capsule;
5. evaluate that complete capsule against the default on fresh streams; and
6. commit either the capsule or default through terminal.

On every subsequent real turn, the committed capsule performs only fixed-work,
public-state operations:

1. restore the exact capsule and public memory;
2. deterministically condition/repair members around the public action and
   reveal;
3. enumerate the full legal menu;
4. compute generator-relative support and the frozen tie break;
5. execute the public legal action; and
6. persist the updated public-memory and capsule hashes.

Low support, an empty plan population, or an exhausted repair budget invokes
the capsule's deterministic registered default. Scientific serving uses fixed
work units, not a wall-clock-dependent action. A hardware timeout, parity
failure, provenance mismatch, or replay failure invalidates that scientific
game; an operational watchdog may play safely, but its result is not evidence.

## 11. Counterexample-guided commitment learning

Foundry does not add network capacity whenever it loses. It asks which causal
distinction the current plan language could not express.

### 11.1 Commitment-collapse trace

A commitment collapse occurs when:

- a root action destroys most 100+ plan mass;
- a planned resource becomes scarce and repair fails;
- a terminal blueprint survives symbolically but cannot be constructed
  nonanticipatively;
- the controller repeatedly misses a score breakpoint;
- the capped archive discards the lineage that later wins; or
- Commons gains vanish when all four seats use the policy.

The trace classifies the first failure as one of:

- missing terminal design family;
- missing exact resource equivalence;
- missing repair operator;
- missing controller feature or branch;
- incorrect scarcity/deadline calibration;
- portfolio-cap collapse;
- insufficient runtime search; or
- policy-composition failure.

### 11.2 Refinement loop

1. Reproduce the failure on a frozen, non-live root bundle.
2. Find the earliest public state at which successful and failed lineages need
   different behavior.
3. Add the smallest contract clause, repair, or program primitive that exposes
   that distinction.
4. Replay every prior counterexample and permanent sentinel.
5. Re-run capped-versus-uncapped and design-versus-confirmation checks.
6. Admit the refinement only on a disjoint block.

This borrows the discipline, not the theorem, of counterexample-guided
abstraction refinement. The proposal does not claim formal optimality.

## 12. Ceiling tomography before the large build

Foundry first measures where score is physically available. Do not spend a
month on GPU program search before answering this.

For every diagnostic stage \(k\), distinguish the unknown optimum \(C_k^*\)
from what the implementation actually proves:

\[
L_k \le C_k^* \le U_k.
\]

An executable witness under that stage's information/resource contract gives
a certified lower bound \(L_k\). A mathematically valid relaxation with a
certified solver bound may give \(U_k\). A heuristic optimizer's best score is
only a feasible-witness lower bound; it is never an upper bound merely because
the optimizer stopped. The intended progression—static design, chronological
hindsight replay, known-world control, honest public control, restricted plan
menu, production DSL, production wall, symmetric gameplay—adds information
privileges and then restrictions, but measured heuristic outputs are not
assumed ordered. Never subtract unmatched heuristic scores and call the
difference causal.

### 12.1 Same-resource repacking

Use two deliberately different tests.

1. **Static multiset repacking:** freeze tile identities, wildlife multiset,
   starter, and final Nature accounting, then search for a better legal
   terminal arrangement. A materialized board is a valid static design witness
   but not necessarily an achievable gameplay replay: wildlife cannot be held
   for a later tile, and the original tile/wildlife pairing and timing matter.
   Label this result optimistic and never use it as policy evidence.
2. **Chronology-preserving replay:** freeze the exact 20-turn public draft
   pairs, free-refresh decisions/reveals, paid wipes, Nature transactions, and
   resource receipts. Re-run legal placement decisions through `GameState` in
   their original order. Any better completed replay is a certified
   hindsight-placement witness for that realized chronology. It still uses
   future knowledge and is not an online policy.

A constraint relaxation supplies an upper bound only when its feasible set is
proved to contain the exact problem and the solver emits a valid bound. A tiny
heuristic gain alone cannot close construction; either the upper bound must be
tiny or search coverage must clear a preregistered sufficiency bar.

### 12.2 Four-board resource reallocation

Freeze the table’s acquired resources and optimize their assignment and board
construction across the four seats, first without chronology and then with
turn/resource constraints. The chronology-free form is intentionally
optimistic. Report its best feasible witness and certified relaxation bound
separately; only the chronology-constrained form can witness an achievable
hindsight allocation for that realized table.

Kill the cooperative core if a valid upper bound on the admissible relaxation
exposes less than 10 table points over the current policy. The target needs
roughly 6.8 from the historical reference; the extra margin prices
approximation and current-rules drift.

### 12.3 Known-world diagnostic

Expose exact future supply only to a diagnostic optimizer. This is never a
policy and never strength evidence. A known-world/public difference isolates
information and recourse only under a matched policy class and certified or
comparable optimization budget; otherwise it is descriptive.

### 12.4 Honest public controller

Run one scenario-braided controller across all worlds with scenario RNG
streams private to the simulator. This is the first honest policy diagnostic.
Its certified paired lower bound must expose at least 2.5 mean-seat points over
a fresh baseline before Foundry receives a target-reaching label.

### 12.5 Compression, language, and runtime losses

Compare:

- uncapped versus capped blueprint portfolios;
- full legal menu versus plan-directed menu;
- unrestricted repair search versus fixed repair grammar;
- large controller DSL versus production DSL; and
- unconstrained wall versus production wall.

The first large gap identifies the correct engineering target. It also
prevents a beautiful architecture from hiding that its online approximation
cannot recover its hindsight headroom.

## 13. GPU and local-hardware design

### 13.1 john0 device representation

Use a structure-of-arrays CUDA engine with:

- 127-cell packed board masks plus exact overflow;
- per-terrain edge/component summaries;
- wildlife bitsets and local adjacency masks;
- market and public supply counters;
- dynamic-urn scenario RNG state private to the simulator;
- fixed-size blueprint/repair arrays;
- branch-light DSL bytecode;
- counter-based RNG domains;
- segmented legal-action argmax; and
- wavefront compaction by phase and action-count bucket.

The current Rust engine and scorer remain the oracle. The device engine must be
bit-exact for rules, legality, integer scores, and replay. Floating-point
program fitness is deterministic under a frozen reduction/tie contract.

### 13.2 Memory envelope

A 512-genome by 8-tape design batch has 4,096 live game lanes. Even at a loose
16 KiB per lane, that is about 64 MiB before scratch space. If a mutable plan
binding is 128 bytes, `4,096 lanes x 4 plans x 128 bytes` is about 2 MiB per
root. The immutable contract archive, clauses, placement DAGs, repair grammar,
menus, overflow states, and repair scratch are additional.

Memory is not the primary concern on the RTX 5090. Divergent legal enumeration,
plan repair, exact scoring, and whole-trajectory throughput are the real kill
tests. Measure serialized binding and contract p50/p95/max, total resident
archive working set, peak repair scratch, and end-to-end synthesis memory;
storage arithmetic is not a capacity result.

### 13.3 Throughput bar

Before online optimization, require:

- a disjoint opening/middle/late fixed-root power study that freezes the
  practical margin, leading-capsule family, paired scenario rule, and required
  design/confirmation counts;
- at least 400,000 complete policy plies per second, including legal action
  construction, plan update, controller execution, and exact scoring;
- under 10 seconds for the **precision-required**, fixed-work synthesis budget
  at representative opening, middle, and late roots;
- zero transition mismatches over at least one million randomized reachable
  transitions, including wildlife returns, wipes, exclusions, and Nature
  transactions; and
- zero cohort-manifest, public-policy-trace, or terminal-score mismatches over
  at least 10,000 complete games.

These bars are proposal syntheses, not literature constants. Missing them
closes or shrinks the online form before gameplay.

### 13.4 Mac fleet

john1–john4 remain training-data workers only. They may:

- generate terminal blueprint shards by starter, score tier, and structural
  family;
- run CPU large-neighborhood repair and controller-program islands;
- verify every materialized terminal witness with the Rust scorer;
- mine commitment-collapse examples from complete frozen roots; and
- emit hash-pinned archives for john0.

They never run promotion gates. Fleet shards never auto-fold into the library.
john0 alone performs decisive screens/gates after the current D1 chain reaches
its registered boundary.

## 14. Training without value learning

There is no scalar value loss, Q loss, TD bootstrap, policy imitation target,
or learned dynamics objective.

Genome discovery uses:

\[
Y_{\mathrm{disc}}(g,\xi)=\frac14\sum_j S_j(g,\xi).
\]

The load-bearing deployed-policy datum is instead:

\[
Y_{\mathrm{cap}}(\Pi_{\mathcal A},\xi)
=\frac14\sum_j S_j(\Pi_{\mathcal A},\xi),
\]

computed by the exact engine after the complete committed capsule meta-policy
finishes. Individual-genome fitness may discover a capsule; it cannot certify
the population-vote policy.

Optional exact diagnostics are:

- intended versus realized blueprint score;
- contract settlement/expiry;
- resource substitutions and repair count;
- plan survival by turn;
- wildlife/habitat/Nature attribution;
- root action and canonical archive support;
- paired regret against frozen controls; and
- public-history strategy-fusion audit.

An optional 3–5M parameter D6-tied **RepairNet** may rank symbolic repair
operators. It predicts neither value nor rules. Its targets are exact
destroy-and-repair improvements found by the compiler, and it is retained only
if equal-wall whole-policy evidence improves. Foundry remains functional
without it.

## 15. Why this can cross 100

### 15.1 It attacks the measured gap at the right scale

The historical mean gap is about 1.7 per seat, not 17. The July-15 D1
mechanism evidence, which predates the July-16 rules-identity boundary, shows
stronger search changing 43.2–43.6% of repeat-stable hard-root labels, with
roughly 0.36–0.40 moved-root teacher regret. That does not predict a
current-rules game gain, but it demonstrates unresolved decisions large enough
to matter.

Foundry needs to recover a fraction of that room. It is designed for the
specific cases where immediate/Q estimates are weakest: zero-immediate-score
commitments, mutually exclusive wildlife plans, habitat merges, and market
resources whose best use belongs to a later seat.

### 15.2 Exact terminal fitness removes the learned bootstrap

Every discovery program and the final capsule are graded by actual endpoints.
No value model decides which object is fit. A weak controller can still lose,
but it cannot win the optimizer because a learned Q head hallucinated terminal
value.

Exact endpoints do **not** remove aleatoric return variance. Commons still
scores four stochastic terminal boards and therefore faces the table-total
variance that harmed the prior serving probes. Before fixing any online tape
count, a disjoint fixed-root preflight must measure paired difference variance,
effective selection multiplicity, and the sample size needed for the frozen
override margin. If that \(K\) does not fit the wall budget, close online
Commons rather than treating exact scoring as de-noising.

### 15.3 Canonical archive support attacks selection noise differently

More simulations reduce uncertainty around individual action estimates.
Foundry instead searches for broad regions of policy space that independently
arrive at the same root action and high terminal score. It treats robustness
of the solution set as signal. The single-elite control will test whether that
new statistic actually helps.

### 15.4 Persistent plans model commitments directly

K1 exactness added speed but no score because final actions were already
locally correct. Foundry works where commitment matters: before the terminal
score is fixed, while habitat topology and wildlife compatibility can still be
preserved or destroyed.

### 15.5 Commons unlocks a distinct axis

Ordinary selfish search can spend a shared market resource for a small active-
seat gain even when another controlled board would convert it into more table
score. Full table-native program fitness removes that inconsistency at every
future simulated turn. The previous table-total experiments used learned
table values at selected search locations while the rest of the policy
remained structurally selfish; they are controls, not a test of Foundry-
Commons.

That distinction is a mechanism argument, not new strength evidence. The
failed variants remain closed. Only the exact tomography ladder can supply the
materially new evidence required to open Commons, and only John's methodology
ruling can authorize its gameplay evaluation.

## 16. The 76% forecast

### 16.1 What the number means

The forecast is:

> Conditional on a valid baseline of at least about 98.0, acceptance of the
> cooperative benchmark objective, at least 10 table points of honest
> recoverable headroom, and the GPU throughput bar, I assign a 76% subjective
> probability that the complete Foundry research program produces a frozen
> policy with true mean seat score at least 100.

It is not a frequentist confidence level, posterior from observed Foundry
games, or claim that the next 1,000-game sample already has 76% success
probability. It is also not an unconditional probability from today: the
baseline, headroom, methodology, and throughput premises are not yet known.

### 16.2 Frozen conditional failure budget

The 76% is a pre-evidence engineering forecast built from a
24-percentage-point residual failure budget **after** the four premises pass:

| Residual failure mode after premise gates | Budget |
| --- | ---: |
| production capsule fails to retain honest-controller headroom | 8 points |
| Commons gain disappears under symmetric/fair composition | 5 points |
| development selection fails on fresh policy evidence | 5 points |
| exact device/trace/provenance defect survives preflight | 3 points |
| remaining rules/distribution/final-battery drift | 3 points |
| **Total conditional failure budget** | **24 points** |

These are subjective risk allocations, not independent frequencies or a fitted
probabilistic model. They are frozen here so a later result cannot rewrite the
rationale. Assigning an unconditional probability would additionally require
probabilities that the four premises pass; no evidence supports honest values
for those probabilities yet, so this report does not fabricate them.

### 16.3 Sensitivity

| Condition | Subjective success forecast |
| --- | ---: |
| Baseline at least 98.3; Commons accepted; all headroom/throughput gates pass | 78% |
| Baseline 98.0–98.3; Commons accepted; all gates pass | 76% |
| Baseline 97.5–98.0 | 62% |
| Sovereign-only methodology | 55% |
| Honest headroom below 2.5 mean-seat points | below 35% |
| GPU throughput misses and no cheap offline distillation survives | close online Foundry |

These are forecasts to score later. They must not be silently revised after
the outcome.

### 16.4 Sampling probability is not the main risk

The complete July-9 champion ledger has 100 game-level seat means with:

- mean 98.2975;
- sample standard deviation 1.2038; and
- projected 1,000-game standard error 0.0381 if that variance transferred.

Under a normal approximation, the latent mean required for a 75% chance that a
1,000-game sample mean clears 100 is:

\[
100+0.67449\frac{1.2038}{\sqrt{1000}}
=100.0257.
\]

A true mean of 100.10 would make sampling failure rare. Architecture effect,
rules drift, selection, and implementation error dominate uncertainty—not the
final 1,000-game average.

### 16.5 How Foundry earns the label

The program is called **75%-qualified** only after all of these are frozen and
passed on disjoint evidence:

1. an admissible July-16 baseline exists;
2. honest nonanticipative tomography exposes at least 2.5 mean-seat points;
3. with baseline \(b\) and honest headroom \(h\), the capped production capsule
   recovers at least \(\max(0.70,(100.10-b)/h)\) of that gap—84% when
   \(b=98.0,h=2.5\), and 72% when \(b=98.3,h=2.5\);
4. a fresh paired gate is CI-positive and its effect places the projected
   latent mean above 100.10; and
5. a preregistered nested bootstrap or posterior-predictive audit assigns more
   than 0.75 probability to the future 1,000-game mean clearing 100.

The normal repository promotion CI remains mandatory independently. A
posterior forecast does not replace the 95% paired gate.

## 17. Evidence ladder and preregistered gates

### Gate -1 — objective ruling

Before Commons gameplay, John explicitly rules whether the benchmark permits a
single central joint controller, exact table-mean optimization, shared
cross-seat memory/prices/lineages, deliberate donation, and any seat-aware
asymmetry. The ruling records whether “homogeneous self-play” means four
isolated identical seat-local policies or one centralized equivariant policy.
Commons must pass cyclic seat-permutation and per-seat fairness controls, and
the exact nonanticipative tomography must first supply materially new evidence
beyond the closed learned table-total variants. If either condition fails,
retain Sovereign and revise the 76% forecast downward before gameplay evidence.

### Gate 0 — contract and information semantics

Require:

- every contract materializes to a legal terminal board;
- exact score and category parity with Rust;
- correct placement-DAG reachability;
- exact tile/wildlife conservation;
- cohort-manifest identity across hidden lanes and zero lane-local optimization;
- chronological `GameState` replay for every history-reachability claim;
- D6 canonicalization round trips;
- refresh/wipe decision-chance-draft traces;
- CPU/GPU parity; and
- zero public-history action-hash violations.

One false terminal certificate or one strategy-fusion violation fails closed.

### Gate 1 — ceiling tomography

Before running an optimizer, split complete, non-live current-rules games into
a development block and untouched confirmation block. On both, report
certified \([L_k,U_k]\) intervals and paired uncertainty rather than
best-found scores alone:

- static repacking and chronology-preserving replay expose different design
  and achievable-hindsight gaps;
- four-board reallocation reports feasible witnesses separately from valid
  relaxation upper bounds;
- known-world versus public-controller is called an information diagnostic
  unless policy class and optimization budget are matched; and
- on untouched confirmation, the paired 95% lower bound for the honest public
  controller over baseline must be at least 2.5 mean-seat points.

An admissible coordination relaxation whose valid upper bound is below 10
table points kills Commons immediately. Passing the high-confidence premise
requires the stronger honest-controller confirmation bound above, not merely a
large clairvoyant or chronology-free score.

### Gate 2 — representation retention

With measured baseline \(b\) and honest headroom \(h\), define
\(r_{\min}=\max(0.70,(100.10-b)/h)\). Require:

- capped plans and the production capsule retain at least \(r_{\min}\) of the
  uncapped honest-controller improvement;
- plan-directed menus retain at least 90% of controller ceiling;
- one-resource shocks do not collapse more than half the predicted advantage;
- semantic diversity remains nonzero after deduplication; and
- contract coverage holds across opening, middle, and late strata.

### Gate 3 — canonical archive-support mechanism

Against single-best-genome selection at identical wall:

- at least 20% relative reduction in tape-half action disagreement, with the
  paired 95% interval excluding zero improvement;
- fresh-stream terminal mean noninferior by a frozen 0.05 mean-seat margin;
- archive-support ESS at least `max(32, 0.25 * occupied_slots)`;
- largest lineage share at most 10%; and
- at least 80% elected-action agreement across independent archive
  initializations.

These are proposal bars, not published constants. Scenario-stream count is a
separate statistical sample size and is never replaced by archive ESS.

If canonical archive support does not help, remove it rather than protect
novelty.

### Gate 4 — GPU feasibility

Clear the parity, fixed-root variance/power, and throughput bars in §13.
Benchmark full public-policy plies, not contract bit operations or kernel
calls. Scientific action selection uses fixed work units.

### Gate 5 — phase-local shadow arms

Preregister opening-only, middle-only, late-only, and whole-game Foundry
capsules, each committed from its registered synthesis state through terminal.
Compare:

- Commons versus Sovereign;
- blueprints with and without exchange prices;
- persistent capsule memory versus per-turn reset plans/weights without
  resynthesis;
- canonical archive support versus single elite; and
- Foundry versus equal-wall incumbent Gumbel.

Shadow results choose one frozen candidate on development seeds only.

### Gate 6 — fresh gameplay strength

Run the repository’s preregistered paired gate with at least 100 planned pairs,
correct July-16 identity, clean hashes, and a 95% interval excluding zero for
promotion evidence. Group-sequential looks are permitted only under the
existing rule and frozen before launch.

### Gate 7 — target certification

The final candidate runs exactly 1,000 fresh complete four-player games. The
mean seat score must be at least 100. Commons/Sovereign identity and every
library/program/simulator hash remain frozen.

## 18. Critical controls

| Control | Question answered |
| --- | --- |
| current transformer + Gumbel | does Foundry improve the actual incumbent? |
| equal-wall extra Gumbel | is the gain merely more compute? |
| DSL programs without blueprints | are terminal plans doing work? |
| blueprints without Futures Exchange | are cross-board prices doing work? |
| Foundry-Sovereign | does representation help without cooperation? |
| Foundry-Commons | does exact team optimization unlock the target? |
| single best genome | does canonical archive support reduce optimizer noise? |
| per-turn reset plans/weights | does persistent capsule intent matter? |
| full menu | does plan-menu restriction miss decisive actions? |
| scenario-specific programs | deliberately invalid clairvoyant upper bound; never policy evidence |
| known-future terminal optimizer | oracle diagnostic; an upper bound only when solver-certified |
| incumbent action injected as a genome | measures conservative hybrid value without defining Foundry |

## 19. Failure modes and falsifiers

| Failure | Observable signature | Decision |
| --- | --- | --- |
| little same-resource headroom | certified repacking upper bound near played score | close construction thesis |
| little coordination headroom | valid four-board relaxation upper bound below 10 table points | close Commons core |
| strategy fusion | lane-specific cohort state or same public history, different action | invalidate implementation |
| brittle plans | unseen resource shock destroys most predicted gain | expand repair only if cheap; otherwise close |
| archive duplication | option volume rises without semantic diversity | fix deduplication; prior result void |
| language miss | unrestricted controller wins, DSL fails | add smallest causal primitive or close DSL |
| menu miss | plan-directed menu loses >10% honest ceiling | widen/generate differently; no gameplay |
| optimizer’s curse | design-tape winner fails fresh tapes | lower temperature/more diversity; unresolved falls back |
| GPU divergence | parity or replay mismatch | categorical fail-closed |
| GPU too slow | misses full-ply wall bar | offline teacher/distillation only or close |
| Commons sacrifices one seat pathologically | seat/category collapse despite mean gain | report; apply frozen fairness guard only by methodology ruling |
| unilateral-only gain | symmetric table removes advantage | close deployment policy |
| current-rules baseline much lower | required effect exceeds measured envelope | forecast drops; do not preserve 76% claim |
| fresh paired gate null/negative | no strength evidence | close candidate regardless of elegant internals |

## 20. Why this is not the two previous proposals

| Property | Cascadia-Anchor | Cascadia-NX | Cascadia Foundry |
| --- | --- | --- | --- |
| Core object | incumbent action plus challenger | structured learned evaluator | exact terminal construction programs |
| Continuation | full incumbent for confirmation | tree/world search | committed reactive public-state capsule |
| Learned scalar value | retained | load-bearing | absent |
| Ordinary move tree | terminal rollouts around root | yes | no |
| Long-term memory | incumbent state only | evaluator latent | persistent explicit plans |
| Hidden uncertainty | paired terminal worlds | sampled chance worlds | one program braided across worlds |
| Action rule | confidence-bound override | search argmax | diverse high-score canonical archive support |
| Four-board objective | own-score deviation then symmetric gate | primarily own-score `max^n` | exact table mean in Commons |
| Main risk | nested incumbent cost/statistical power | factor fidelity/GPU port | honest headroom/plan repair/program throughput |

Foundry may reuse an exact GPU rules kernel if NX or Anchor validates one, but
sharing infrastructure does not merge the architectures.

## 21. Why this is not NNUE, MCTS, or ordinary evolutionary planning

### NNUE

NNUE incrementally maps an afterstate to a scalar evaluation. Foundry’s state
is a population of materialized terminal designs, placement partial orders,
resource obligations, repair rules, and controller programs. It can use no
neural evaluator at all.

### MCTS/Gumbel

MCTS expands partial action histories and backs up value estimates. Foundry
optimizes whole reactive policies and terminal intents, then elects a root
action by support across a canonical, lineage-deduplicated archive. It does not
maintain visit counts or a search tree.

### Rolling Horizon Evolution

The cited rolling-horizon evolutionary systems evolve action sequences or
neural controllers over a forward model. Foundry evolves persistent four-board
terminal construction genomes, exact score contracts, cross-board scarcity
prices, and a public-history program. Its root statistic is held-out canonical
archive support rather than best individual fitness.

### Programmatic RL

PIRL/NDPS represents policies in a DSL and directs program search from a neural
oracle. Foundry uses no neural oracle, and its DSL is only one layer. The plan
representation, terminal witnesses, Futures Exchange, scenario braid, and
archive-support election are the proposed intelligence.

### Cooperative value decomposition

VDN/QMIX learn a joint value from agent-wise neural values. Foundry has no
joint Q. It directly scores complete four-board terminal games and uses
explicit resource prices only as controller features.

## 22. Novelty boundary and nearest prior art

The novelty search used both exact-name queries and concept queries for
terminal-board blueprints, reverse completion lattices, counterfactual
resource prices, reactive program populations, scenario nonanticipativity,
and policy-basin action selection. It searched scholarly and general indexes,
then followed primary papers for the nearest mechanisms. At the 2026-07-16
cutoff it found adjacent ingredients, not the integrated architecture or its
named mechanisms. Search absence is not a proof of universal nonpublication.

| Adjacent work | What it contains | What Foundry adds or changes |
| --- | --- | --- |
| [Illuminating search spaces by mapping elites](https://arxiv.org/abs/1504.04909) | a quality-diversity archive of high-performing, behaviorally distinct solutions | semantic terminal-plan cells, exact lineage deduplication, and canonical archive root support rather than one elite per cell |
| [Proof-Carrying Plans](https://research-repository.st-andrews.ac.uk/bitstream/10023/16855/1/padl_pddl_verification.pdf) | the published phrase and formal verification of plans against a planning model | exact Cascadia terminal-score witnesses plus stochastic acquisition/repair contracts; Foundry does not claim the proof-carrying phrase itself is new |
| [Trajectory Balance: Improved Credit Assignment in GFlowNets](https://papers.neurips.cc/paper_files/paper/2022/file/27b51baca8377a0cf109f6ecc15a0f70-Paper-Conference.pdf) | learning to sample diverse terminal objects in proportion to reward | no learned generative flow; a canonical finite controller archive and exact generator-relative action support |
| [Satisficing and Optimal Generalised Planning via Goal Regression](https://ojs.aaai.org/index.php/AAAI/article/view/40938) | regress goals into reusable executable condition-action rules | game-specific exact score contracts, four-board resource obligations, and stochastic public-state capsules |
| [Scenario Trees and Policy Selection](https://arxiv.org/abs/1112.4463) | scenario-based policy generation and out-of-sample policy selection | one public-history executable capsule, exact action-hash audit, and score-contract construction state |
| [Information Set MCTS](https://doi.org/10.1109/TCIAIG.2012.2200894) | information-set search and the established strategy-fusion problem | immutable cohort manifests and cross-hidden-world public-history action equality for a synthesized controller population |
| [Playing Against the Board: RHEA for Pandemic](https://arxiv.org/abs/2103.15090) | short-horizon evolutionary planning, macro-actions, repair in a cooperative board game | persistent exact terminal score contracts, four-board resource exchange, scenario braid, canonical archive root election |
| [Rolling Horizon NEAT](https://arxiv.org/abs/2005.06764) | online evolution of neural controller topology/weights | symbolic reactive program plus score-witness construction genome and exact terminal plan memory |
| [Lazy Cross-Entropy Search Over Policy Trees](https://ojs.aaai.org/index.php/AAAI/article/view/29992) | Monte Carlo search over finite-horizon POMDP policy trees | public-history braid, persistent completion plans, cross-board contracts, generator-relative archive-support rule |
| [Programmatically Interpretable RL](https://proceedings.mlr.press/v80/verma18a.html) | high-level DSL policies and neurally directed program search | no neural oracle; exact construction witnesses and commitment-collapse language growth |
| [Planning by Probabilistic Inference](https://proceedings.mlr.press/r4/attias03a.html) | action posterior conditioned on goal reachability | exact high-score board contracts and empirical mass of diverse reactive policy programs |
| [QMIX](https://proceedings.mlr.press/v80/rashid18a.html) | monotonic neural factorization of cooperative joint value | no learned value factorization; direct terminal team score and explicit scarcity exchange |
| [Progressive-hedging scenario partitioning](https://optimization-online.org/2013/10/4065/) | multistage stochastic programs with nonanticipativity constraints | public-history action hashes and one executable controller across exact game scenarios |
| [Counterexample-guided Cartesian abstraction refinement](https://ojs.aaai.org/index.php/ICAPS/article/view/13605) | refine a planning abstraction from failed abstract solutions | refine scoring contracts/repair/DSL at commitment-collapse traces; no formal optimality claim |
| [Pgx](https://arxiv.org/abs/2303.17503) | accelerator-resident board-game simulation | demonstrates that batched accelerator simulation is feasible on its tested hardware; it does not establish Foundry’s algorithm or RTX-5090 speed |

The claimed original contribution is the integrated Cascadia architecture and
the following mechanisms in their precisely defined form—not generic
proof-carrying plans, goal regression, quality diversity, nonanticipativity,
or reward-weighted terminal objects:

1. exact score contracts and reverse completion lattices as the deployed value
   representation;
2. persistent four-board completion genomes with score-preserving repair
   grammars;
3. the Score Futures Exchange’s counterfactual plan-collapse prices;
4. scenario braiding plus public-history action hashes as an execution
   contract;
5. canonical generator-relative archive support as the root action statistic;
6. commitment-collapse-driven growth of the representation itself; and
7. the certified-interval ceiling-tomography protocol for separating design,
   chronology, information, menu, language, runtime, and composition losses.

No located paper or repository artifact combines these objects. That is a
targeted novelty finding, not proof of universal nonpublication.

## 23. Implementation sequence

### Phase A — freeze objective and identities

Specify Commons and Sovereign objectives, rules ID, score endpoint, public
history, plan/program memory, D6 identity, fallback, and evidence partitions.
Obtain John’s ruling on centralized control, cross-seat memory/prices,
donation, seat equivariance, and fairness before Commons gameplay.

### Phase B — terminal design foundry

Build a Rust terminal-board optimizer and materialized contract format.
Generate score-stratified witnesses; validate exact scorer/category equality,
resource conservation, spatial growth orders, chronological `GameState` replay
where claimed, and D6 deduplication.

### Phase C — ceiling tomography

Run static repacking, chronology-preserving replay, four-board allocation
relaxation, known-world diagnostics, and small honest public-controller bounds
on preregistered development/confirmation games. Report certified intervals;
stop if the target envelope is absent.

### Phase D — CPU completion lattice and DSL

Implement reverse deletion, live binding, bounded repair grammar, controller
bytecode, deterministic legal argmax, and public-memory replay. Run the tiny
late-game scenario braid with dynamic urn RNG streams entirely on CPU.

### Phase E — archive support and committed capsule

Add independent semantic archive islands, lineage deduplication, frozen final
design scoring, generator-relative action support, complete capsule evaluation,
fresh-stream confirmation, and archive-support diagnostics. Compare against a
single elite and never claim formal posterior sampling.

### Phase F — exact GPU engine

Port only the operations needed by Foundry. Prove transition, score, replay,
refresh, controller-memory, cohort-manifest, dynamic-urn, and strategy-fusion
parity before throughput optimization. Measure fixed-root paired variance and
precision-required scenario count before accepting the wall budget.

### Phase G — persistent four-board exchange

Add counterfactual resource shocks, cross-board bids, survival estimates, and
Commons/Sovereign modes. Keep the exact complete committed-capsule policy
load-bearing.

### Phase H — counterexample refinement

Mine commitment-collapse traces on development seeds, expand only the missing
causal primitive, and protect a permanent sentinel suite.

### Phase I — shadow and gameplay ladder

Run phase-local terminal-commitment capsules, equal-wall comparisons, fresh
paired gate, and then the 1,000-game target certification. Never adapt to
partial live-arm output.

### Phase J — optional distillation

If direct Foundry wins but serving is too expensive, distill its public
controller into a small policy. The distilled policy is a new candidate with
its own parity, root-retention, gameplay, and target gates. Distillation does
not inherit Foundry’s evidence.

## 24. Reproducibility and provenance contract

Every artifact pins:

- Git revision;
- rules/scoring/catalog identity;
- terminal-contract compiler and grammar hash;
- blueprint archive hash;
- D6 canonicalization version;
- controller bytecode version and instruction limit;
- mutation/repair operator set;
- population, generation, archive, and temperature settings;
- Commons/Sovereign objective;
- design/confirmation scenario seed blocks;
- dynamic-urn scenario sampler and RNG-domain map;
- immutable cohort manifest and committed capsule hash;
- fixed-work serving budget;
- public-memory schema;
- device numerical and tie contract;
- fallback policy;
- exact simulator build hash; and
- complete per-game replay and terminal score ledger.

No scientific artifact’s only copy lives in a temporary directory. Plan
archives, scenario manifests, raw games, and public-history trace audits are
durable and SHA-256 addressed.

## 25. Recommendation

Fund a **bounded Foundry preflight after the authorized D1 chain reaches its
frozen boundary**.

The first work is not a model and not a long training run. It is ceiling
tomography plus a tiny CPU terminal-contract/controller prototype. Those tests
can destroy the thesis cheaply:

1. Do static and chronology-preserving construction intervals expose real
   placement headroom?
2. Does Commons avoid the 10-table-point upper-bound kill and then clear a
   2.5-mean-seat honest-controller confirmation lower bound?
3. Can exact A-card score contracts cover real incumbent boards?
4. Can one public-state program adapt across hidden worlds without strategy
   fusion?
5. Does canonical archive support produce a more stable, no-worse capsule than
   one evolved winner?
6. Does the precision-required full policy fit the fixed local-5090 work and
   latency bars?

If those answers are yes, Foundry is the most innovative and highest-upside
proposal in the portfolio. It attacks representation, planning, uncertainty,
and the four-seat objective simultaneously while grading every policy by exact
terminal score.

If the headroom tests fail, close it immediately. If Commons is rejected,
retain Sovereign and revise confidence to roughly 55%. If GPU throughput fails
but the contract oracle is strong, use Foundry as an offline teacher and gate
the distilled policy independently.

The north star remains unchanged: a frozen, provenance-clean mean seat score
at least 100 over 1,000 complete games. The invention is not the result. The
result must still be measured.

## 26. Repository context

- [v3 source of truth](docs/v3/README.md)
- [live campaign state](docs/v3/CAMPAIGN_STATE.md)
- [research log and closed directions](docs/v3/RESEARCH_LOG.md)
- [living research agenda](docs/v3/RESEARCH_AGENDA.md)
- [radical directions](docs/v3/RADICAL_DIRECTIONS.md)
- [rules contract](docs/v3/RULES_CONTRACT.md)
- [current architecture](docs/v3/ARCHITECTURE.md)
- [first July-16 proposal: Cascadia-NX](stochastic_board_game_ai_architecture_research_7_16.md)
- [second July-16 proposal: Cascadia-Anchor](incumbent_anchored_gpu_rollout_policy_improvement_7_16.md)
- [July-16 research questions](research_questions_7_16.md)
- [July-16 research answers](research_answers_7_16.md)
- [exact scoring implementation](crates/cascadia-game/src/scoring.rs)
- [current Gumbel implementation](cascadiav3/real-root-exporter/src/gumbel.rs)
- [historical complete July-9 champion games](cascadiav3/reports/rules_20260709_rebaseline_complete/rules_20260709_cycle4_n1024_d16_games.jsonl)
