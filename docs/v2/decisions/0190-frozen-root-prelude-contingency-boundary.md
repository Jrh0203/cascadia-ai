# ADR 0190: Frozen Root Prelude Contingency Boundary

**Status:** Accepted
**Date:** 2026-06-17
**Experiment:** `o1-public-belief-one-rotation-search-v1`
**Supersedes protocol:** `o1-public-belief-one-rotation-high-regret-v1`
**Authorizes protocol:** `o1-public-belief-one-rotation-high-regret-v2`

## Context

The first four-host launch of ADR 0189 failed before producing any complete
production report. All four primary arms stopped on the same second panel
group while applying a frozen root action:

```text
group=9475110012915721395
root=5fab9de804d78a5e7382f023b2ebfb1d4176ce91c3f6ab4840195d841582d305
tile at HexCoord { q: 0, r: 1 } cannot support Hawk
```

The failure was deterministic and arm-independent. It exposed a mismatch
between the graded-oracle complete-action contract and protocol v1's chance
ordering.

A graded-oracle `TurnAction` may include a market prelude such as the free
three-of-a-kind replacement. The recorded complete action is contingent on
the public market produced by that prelude. Exact replay reconstructs the
hidden supply order needed to reproduce that already observed staged market,
and candidate validation checks that staged market before applying the draft,
tile placement, and wildlife placement.

Protocol v1 redeterminized hidden order before applying the complete root
action. For staged roots, that changed the prelude draw, changed the public
market on which the action was defined, and could make the frozen action
illegal. This was not a weak model result. It was an invalid simulator
boundary.

## Decision

The complete frozen root action is applied before future hidden supply is
redeterminized.

Every trajectory now executes in this order:

1. reconstruct the exact graded-oracle replay state;
2. apply the frozen complete root action, including its recorded staged market
   prelude;
3. retain the resulting public afterstate;
4. sort and redeterminize all remaining hidden tile and wildlife order from
   `BLAKE3(domain, group_id, action_hash, sample_index)`;
5. simulate up to three opponent turns;
6. evaluate the returned focal state or terminal state.

The root chance policy is frozen as:

`condition-on-frozen-complete-turn-staged-prelude-context`

The future hidden-order policy is frozen as:

`sort-and-redeterminize-after-frozen-root-before-opponent-rotation`

The determinization domain is versioned to
`cascadia-v2-o1-public-belief-search-post-root-determinization-v2`.

## Information Boundary

Conditioning on the staged root prelude is not future leakage. The staged
market is part of the frozen candidate's public action identity. Comparing
that candidate while replacing its observed prelude would compare a different
action in a different public state.

No recorded hidden future is retained after the root:

- all remaining tile and wildlife order is sorted and redeterminized;
- the same root/sample determinization is shared by all four arms;
- opponent choices see only the resulting public state;
- labels remain unavailable to search;
- sealed-test and gameplay domains remain closed.

## Invariance Probe

The hidden-order invariance check is moved to the correct causal boundary:

1. apply the frozen root exactly once;
2. clone its public afterstate;
3. adversarially perturb only the remaining hidden order in one clone;
4. redeterminize both clones with the same fixed sample identity;
5. require identical public trajectory results.

This proves that post-root results do not depend on recorded hidden order
while preserving the complete root's public contingency.

## Invalidated Launch

The original protocol-v1 launch is invalid and cannot be resumed:

- bundle:
  `32af22eb973504babfa8b81cf503ba95c76078dbd97cbd92602a31d69fa4f645`;
- authorization:
  `9d619204a2171b0d89de84f509e0602fb3cd627a92ab07620e21fe12c7017991`;
- task prefix: `o1pbs-v1`;
- completed reports: zero;
- scientific comparisons: zero;
- sealed-test rows opened: zero;
- gameplay games run: zero.

The v1 bundle, authorization, queue records, and partial row artifacts remain
immutable evidence. Protocol v2 uses a new bundle, authorization package,
task prefix, preflight directory, run directory, collection directory, and
aggregate path.

## Verification

Before protocol-v2 production authorization:

- a Rust regression test constructs a staged complete action that becomes
  invalid under pre-root redeterminization and proves exact-root-first
  application succeeds;
- a two-group release smoke crosses the previously failing panel group;
- all four arms complete a matched ten-group smoke;
- each ten-group arm accounts for 640 roots, 6,400 trajectories, 19,200
  opponent decisions, ten invariance probes, and 640 candidate hash checks.

## Consequences

ADR 0189 remains the scientific design, success gate, and claim boundary.
This ADR changes only the causal placement of root conditioning and future
hidden-order randomization. No arm, panel row, candidate, budget, model,
metric, threshold, bootstrap, or promotion rule changes.

Only protocol v2 is authorized for production.
