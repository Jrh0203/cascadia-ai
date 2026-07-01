# ADR 0134: Information-Preservation Dependency Closure

Status: accepted

Date: 2026-06-16

Experiment ID: `information-preservation-adversarial-suite-v1`

## Context

ADR 0131 froze fourteen information-preservation pair families and required
unavailable facts to remain explicitly blocked on F1, F2, or F3. F2 completed
the state-footprint census, F3 completed the exact Rust-owned D6 contract, and
F1 has now completed the feature-schema activation census.

F1's authoritative closure artifacts are:

- `artifacts/experiments/feature-schema-activation-census-v1/reports/final-classification.json`;
- `artifacts/experiments/feature-schema-activation-census-v1/reports/final-forward.json`;
- `artifacts/experiments/feature-schema-activation-census-v1/result-manifest.json`; and
- `docs/v2/reports/feature-schema-activation-census-v1-result.md`.

The F1 classification is `feature_schema_activation_census_complete`, with
classification scientific BLAKE3
`f7f8559431f53a461f9464e14ef4cee2119cf3ddcf0bf4e3dd9126ab8bdd91fb`.
The merged census scientific BLAKE3 is
`8906487b91aa0da25f388e2075d15150c8d1499a022cbf2d231987b37f182e65`.

Closing the remaining pairs with handwritten Python geometry, synthetic
legality, hidden refill order, or invented labels would violate ADR 0131. The
closure therefore requires machine-generated witnesses from authoritative Rust
rules APIs, bound to the exact upstream authority chain.

## Decision

F4 owns a deterministic Rust fixture generator at:

`artifacts/experiments/information-preservation-adversarial-suite-v1/fixture-generator`.

It writes:

`artifacts/experiments/information-preservation-adversarial-suite-v1/fixtures/resolved-dependencies-v2.json`.

The generator is a standalone reproducible package that depends on the
workspace `cascadia-game` crate. Receipt schema v2 contains F1, F2, and F3
sections. It does not modify any upstream experiment, the cluster research
queue, or the dashboard ledger.

Python loads the generated receipt fail-closed, validates its schema and frozen
upstream hashes, and deterministically materializes the seven dependency-owned
pairs:

1. `long_salmon_component_context`;
2. `d6_transforms`;
3. `component_bridge`;
4. `equal_immediate_different_future_conflict`;
5. `public_action_equivalence_refill_near_match`;
6. `same_in_radius_different_overflow_consequence`; and
7. `same_compact_latent_different_legal_affordance`.

All fourteen frozen families are now executable. No dependency block remains.

## F1 Authority Receipt

Generation fails unless all four F1 source files match their exact BLAKE3
digests, the result manifest links those files to the frozen scientific hashes,
the report declares F1 complete, and every closed-domain flag remains false.

The receipt also pins the exact F1 census rows for:

- `legacy.habitat_sizes_v1`;
- `legacy.patterns_v1`;
- `graded.action.immediate_score`;
- `graded.action.immediate_deltas`;
- `graded.parent_public_supply`;
- `graded.staged_market`; and
- `graded.staged_public_supply`.

Python independently validates this authority receipt, every exact witness
invariant, the materialized labels, public inputs, boundary contracts,
provenance, and receipt hashes. There is no permissive fallback to the prior v1
receipt.

## F1 Exact Witnesses

### Long Salmon Context

Two legal boards share the exact radius-one public neighborhood around the
origin. The left origin belongs to a straight Salmon component of size five;
the right belongs to a straight Salmon component of size four. Exact component
cells, endpoints, maximum degree, habitat component identities, and scores come
from Rust.

### Component Bridge

A Forest tile at the origin merges two size-one source components into a
size-three component on the left. On the right, the same bridge placement
extends one size-four component to size five. Rust exports exact pre-component
IDs, touched IDs, post-component cells, and largest-habitat values.

### Equal Immediate Score, Different Motif Conflict

The two boards have the same tile layout and identical exact score breakdowns,
including `base_total = 4`. The left has a public Hawk adjacency conflict; the
right has a public branching Salmon conflict. No realized later play or hidden
teacher value is used.

### Public Action Equivalence and Refill Near-Match

The exact-equivalence case uses two distinct rotations of one single-terrain
tile. Rust proves equal preview public afterstates and equal full transitions.

The near-match case compares placing the drafted Elk with legally returning it.
The public one-draw wildlife distributions are order-free and differ exactly:

- place: `[20, 19, 20, 19, 18] / 96`;
- return: `[20, 20, 20, 19, 18] / 97`.

The exact equivalence is required at `public-observable-v1`. The refill
distinction is required at `public-refill-distribution-v1`. Optional assertions
cannot change a boundary verdict or its metrics.

## F2 and F3 Receipts

F2 remains bound to scientific BLAKE3
`c6076545aa93e78902b739eefef1545a23b8f2dbe44770f427a30969511800e5`.
The two legal 23-tile boards share the same radius-6 occupancy and frontier but
have different exact legal masks. Radius-6 clipping loses the distinction; the
127-cell compact contract with an exact overflow sidecar retains it.

F3 remains bound to contract scientific BLAKE3
`db6ac2f9f6ebe2daaa2db603c6c16183512b5d989aed6979e1991e167737633f`.
All twelve D6 transforms over 540 legal rows pass legal-map bijection, policy
and action round-trip, and transition equivariance.

## Reproducibility

Generate the receipt:

```bash
CARGO_TARGET_DIR=target/f4-fixture-generator \
  cargo run --quiet \
  --manifest-path artifacts/experiments/information-preservation-adversarial-suite-v1/fixture-generator/Cargo.toml \
  -- artifacts/experiments/information-preservation-adversarial-suite-v1/fixtures/resolved-dependencies-v2.json
```

Materialize the frozen pair set:

```bash
PYTHONPATH=python .venv/bin/python -m \
  cascadia_mlx.information_preservation_suite \
  --resolved-dependencies \
  artifacts/experiments/information-preservation-adversarial-suite-v1/fixtures/resolved-dependencies-v2.json \
  --materialize-fixtures \
  artifacts/experiments/information-preservation-adversarial-suite-v1/fixtures/pairs-v1.json \
  --stdout none
```

The focused test suite regenerates the Rust receipt into a temporary path and
requires byte equality with the frozen artifact before exercising the
materialized pairs.

## Classification

The complete result is:

`information_preservation_suite_failed`

This is a complete scientific classification, not a dependency or execution
failure:

- fourteen of fourteen pair families are executable;
- zero pairs are dependency-blocked;
- the exact public-observable boundary retains all fourteen families;
- the refill-distribution boundary retains its required near-match;
- the compact 127-cell plus exact-overflow contract retains its legal mask; and
- seven deliberately lossy boundaries fail at least one required assertion.

The failed classification is therefore the intended verdict: the suite found
real information loss. It is not an invalid run and it must not be rewritten as
a pass.

No sealed data, gameplay evaluation, new teacher rollout, cloud host, external
compute, cluster queue edit, or dashboard-ledger edit is part of this closure.

The resulting suite scientific BLAKE3 is:

`f5575ea0c3780894fda656fefd8bab9bb96e2fcc814bceca933002c87a04e178`.
