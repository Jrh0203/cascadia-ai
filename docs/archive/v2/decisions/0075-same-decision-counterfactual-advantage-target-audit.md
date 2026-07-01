# ADR 0075: Same-Decision Counterfactual Advantage Target Audit

Status: rejected on target width on 2026-06-13. The grouped shared-seed sampler
is qualified; no train corpus, model, test, or gameplay domain was opened.

## Context

ADR 0074 qualified complete H6 continuations under public hidden-state
redetermination. An R8 prefix approximated the R16 public-state mean at
0.487-point MAE and 91.14% within-round ordering accuracy. Absolute state value
was rejected because the R16 state means had only 1.945 points of standard
deviation, below the frozen 2.0-point signal-width gate.

The next target removes that narrow game-level offset. At one public decision,
several legal actions can be evaluated under the same hidden futures and
centered against their group mean. This is the decision-local quantity a policy
must rank.

The distinction between state value and action advantage is established in the
dueling-network literature. Counterfactual baselines are also an established
credit-assignment device in multi-agent reinforcement learning. This audit does
not adopt either paper's learning algorithm; it tests whether the corresponding
decision-local target is observable, stable, and affordable in Cascadia.

- [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926)

## Decision

Add a versioned, checksummed grouped counterfactual-advantage dataset:

1. Generate the source trajectory with frozen H6
   `habitat-candidate-lookahead-v1-k8-h6-r4-d4`.
2. Resolve any source-trajectory market prelude first. Record the resulting
   public post-prelude state at a fixed stride. The substantive audit records
   completed turns `0,5,...,75`, giving 16 groups per game, four groups per
   seat, and coverage across the full game.
3. At each recorded post-prelude decision, retain the action selected by H6
   followed by the
   next three highest-ranked distinct actions from H6's existing K8+H6
   frontier after its frozen R4/D4 shallow evaluation. Do not regenerate,
   widen, or rescore the frontier.
4. Retain which of those four actions H6 selected for the factual trajectory,
   the parent public position, exact current decomposed score, public supply,
   each action hash, each observable action afterstate, and H6's shallow mean
   and standard deviation.
5. Generate sixteen deterministic public-redetermination seeds for the group.
   Use the identical ordered seeds for all four candidates.
6. For each candidate and seed, clone the parent state, redetermine only hidden
   order, verify public-supply invariance, apply the candidate, and continue all
   seats to game end with frozen H6.
7. Store every terminal decomposed score and every shared seed. Do not store
   only means, winners, or centered labels.
8. Validate fixed-width headers, checksums, source/executable provenance,
   group/turn/seat sequence, candidate uniqueness and ordering, selected-action
   identity, sample uniqueness, public-supply invariance, score ranges, and
   zeroed unused storage.

For candidate `a` in decision group `g`, the centered target at sample count
`R` is:

`A_R(g,a) = mean_R(return(g,a)) - mean_candidates(mean_R(return(g,*)))`.

The audit computes:

- R16 candidate means, centered advantages, within-group ranges, and standard
  deviations;
- R1/R2/R4/R8 centered drift, correlation, pairwise ordering, exact top-action
  agreement, and top-action regret against R16;
- sample-wise paired standard errors for centered candidate advantages;
- H6 shallow-selected regret against the R16 best candidate;
- phase-binned stability over personal turns 1-5, 6-10, 11-15, and 16-20;
- collection throughput and projected cost for a 128-game train plus 32-game
  validation corpus at R8.

## Frozen Protocol

- Implementation-only smoke: train split index 9,994, one source game, four
  groups at completed turns `0,20,40,60`, four candidates, R2.
- Substantive audit: validation split indices 66,000-66,001, two complete
  source games, 32 groups, four candidates per group, R16.
- Rules: symmetric four-player AAAAA, no habitat bonuses.
- Source and continuation policy: frozen H6 K8/H6/R4/D4.
- Candidate ordering: H6's selected action first, followed by the first three
  differently acting entries in H6's existing ranked order. Ties retain that
  existing stable order.
- Chance boundary: free three-of-a-kind replacement and any future prelude
  action are resolved into the recorded public state. Candidate actions do not
  replay that already-observed chance event.
- Shared sample seeds: BLAKE3 domain
  `cascadia-v2-counterfactual-advantage-v1`, split, game index, completed turn,
  and sample index.
- Execution: local Apple M4 only. Jobs may run in parallel; serialized output
  must be independent of scheduling.
- No model training, train corpus beyond the implementation smoke, test/final
  split access, gameplay comparison, alternate frontier, independent-seed
  retry, threshold change, sample-count retry, or external compute.

The target advances to a separately preregistered MLX experiment only if all
conditions hold:

- every integrity, replay, determinism, provenance, and public-supply check
  passes;
- R8 centered-advantage MAE against R16 is at most 0.50 points;
- R8 centered-advantage Pearson correlation against R16 is at least 0.80;
- R8 within-group pairwise ordering accuracy against R16 is at least 80%;
- R8 exact top-action agreement with R16 is at least 65%;
- mean R16 regret from choosing the R8 winner is at most 0.50 points;
- mean R16 within-group value range is at least 1.50 points;
- mean R16 centered-advantage standard error is at most 0.75 points;
- the projected 160-game R8 corpus takes at most 12 uncontended local hours.

Passing authorizes only a new ADR for a 128-game train and 32-game validation
R8 corpus plus an MLX complete-candidate-set ranker. It does not authorize test
collection or gameplay. Any failed gate rejects this target without rounding,
retrying, changing the frontier, or inspecting another seed.

### Pre-Data Implementation Amendment

The first substantive command stopped before writing a manifest or shard. H6
can return more than four exactly tied leaders, then select one of those ties
with its deterministic decision RNG. The selected action can therefore lie
beyond the first four vector entries. Requiring both "first four" and "retain
the selected action" was impossible.

Before any validation record existed, candidate retention was corrected to the
rule above: selected action first, then the next three ranked distinct
alternatives. The candidate count, H6 frontier, shallow evaluations, seeds,
stride, samples, gates, and validation indices are unchanged. This is an
implementation correction, not a result-driven protocol change.

The next command also stopped before writing a manifest or shard when a fixed
pre-replacement candidate became illegal under a different hidden replacement
order. The recorded decision boundary was therefore corrected, still before
any validation evidence, to the canonical post-prelude public state described
above. This preserves the actual information available when H6 chooses its
placement and prevents a hidden chance outcome from being replayed as though
it were part of the action.

## Required Implementation

- a dedicated grouped dataset schema with fixed-width atomic one-game shards;
- shared public-redetermination seeds represented once per decision group;
- exact parent and action-afterstate features with public supply;
- raw decomposed candidate returns, not pre-averaged labels;
- deterministic collection under Rayon scheduling;
- Rust round-trip, corruption, resume-provenance, and group-invariant tests;
- typed CLI collect, validate, and audit commands;
- one-command smoke and substantive audit targets;
- machine-readable JSON and concise Markdown result reports.

## Implementation Evidence

The only authorized implementation smoke completed on train index 9,994:

- one complete H6 source game;
- four recorded decision groups at completed turns `0,20,40,60`;
- four candidates per group and two shared samples per candidate;
- 32 complete continuations in 13.07 seconds;
- exact fixed-width validation of the manifest, header, group sequence,
  candidate identity, shared seeds, public supply, and all raw returns;
- an independent repeat produced a byte-identical shard with BLAKE3
  `57d574cf90623dd0a8364d52cb78bb5c1c76c116544b39cd4a8c05fbb859145f`;
- strict Clippy, focused Rust tests, formatting, and diff checks passed.

The smoke projects a 160-game R8 corpus at 2.32 uncontended hours, but that
estimate is based on only four groups and is not a gate result. Its provisional
R2 mean group range was 3.0 points. No target-stability conclusion may be drawn
from two samples. The preserved reports are:

- `docs/v2/reports/same-decision-counterfactual-advantage-target-audit-v1-implementation-smoke.json`
  (BLAKE3
  `70b738ea01db7c7e8a2138b3ec435ea3dc5284a61aefdc4e71a6a7a38c0542ae`);
- `docs/v2/reports/same-decision-counterfactual-advantage-target-audit-v1-implementation-smoke.md`
  (BLAKE3
  `0b3e7fb1ada90faece1a85acdee7971c5da869c77fdc3a696afa2925586a2eeb`).

## Result

The two authorized validation games completed 32 decision groups, 128
candidates, and 2,048 full H6 continuations in 694.75 seconds. Both atomic
shards and the aggregate manifest passed validation.

R8 is a strong estimator of the R16 decision target:

- centered-advantage MAE: 0.274 points;
- centered-advantage correlation: 0.855;
- within-group pairwise accuracy: 89.58%;
- exact top-action agreement: 81.25%;
- mean R16 regret from choosing the R8 winner: 0.057 points;
- mean centered-advantage standard error: 0.384 points;
- projected 160-game R8 collection: 7.72 uncontended hours.

The shallow H6 choice itself was the R16-best retained action in only 56.25% of
groups and had 0.338 mean regret. Repeated complete continuations therefore
contain useful corrective ranking signal.

The target nevertheless failed its frozen signal-width gate. The selected
action and its three nearest ranked alternatives spanned only 1.367 points on
average, below the required 1.50. Every other gate passed.

This is not permission to round the threshold or train anyway. The narrow
top-four target is rejected and no train, test, model, or gameplay domain is
authorized. The qualified shared-seed sampler may be reused only under a new
ADR that changes the scientific question: retain rank-stratified contrasts
from across the existing H6 frontier so the model sees both fine winner
distinctions and materially different legal alternatives.

## Maximum Compute

One one-game R2 implementation smoke and one two-game R16 substantive audit.
No retry, sweep, extra game, changed threshold, changed action count, changed
recording stride, changed policy, test access, gameplay, model training, or
external compute.
