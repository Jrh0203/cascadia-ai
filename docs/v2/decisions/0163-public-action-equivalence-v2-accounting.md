# ADR 0163: Public-Action Equivalence V2 Accounting

Status: completed; proof tool retained, serving compression rejected

Date: 2026-06-17

Experiment: `s7-public-action-equivalence-foundation-v2`

Protocol: `s7-exact-semantic-transition-v2`

Supersedes: ADR 0162 production attempt

## Context

The V1 source and executable passed formatting, unit tests, clippy, an
adversarial transition suite, a one-group open-data smoke, immutable bundling,
and cross-host checksum fanout. Three disjoint production processes were then
started.

During live code review, before any process produced a shard report, a
fail-closed accounting defect was found. For one serving-safe semantic class
of size `n` split into `k` exact serialized-public subclasses:

```text
serving-safe collapses = n - 1
exact-public collapses = n - k
semantic collapses beyond exact identity = k - 1
```

V1 incorrectly stored the third quantity as `n - k`. The later group-level
identity check would therefore reject ordinary duplicate classes or report
the wrong diagnostic count. No scientific output from V1 is admissible.

## Decision

Terminate every V1 census process and preserve its immutable bundle and
invalid-attempt record. Relaunch only as V2 with:

- experiment ID `s7-public-action-equivalence-foundation-v2`;
- protocol ID `s7-exact-semantic-transition-v2`;
- a checked helper that returns `k - 1`;
- explicit rejection of singleton serving classes and impossible subclass
  counts at that helper boundary; and
- regression tests for one, partial, and fully split exact-public subclasses.

The scientific hypothesis, exact keys, action corpus, three-way modulo
partition, hidden-information boundary, adversarial requirements, promotion
thresholds, and forward/reverse order proof remain unchanged.

## Requalification

Before V2 production:

1. run formatting and all-target checks;
2. run the complete unit suite, including the new accounting regression;
3. run clippy with warnings denied;
4. build one arm64 release executable;
5. pass the three-state adversarial transition suite;
6. pass a real open-data smoke that contains at least one duplicate class;
7. freeze source and binary in a new content-addressed bundle; and
8. verify the complete bundle tree on john2, john3, and john4.

The duplicate-bearing smoke is new. V1's injected duplicate witness exercised
transition parity but did not route through production `GroupRecord`
accounting, which allowed the defect to escape.

## Consequences

V1 is a pipeline failure with no accepted scientific result. It does not
count for or against the public-action equivalence hypothesis.

V2 may proceed only after the stronger smoke demonstrates both exact
transition parity and production class-accounting parity. Any further
scientific-contract change requires another experiment and protocol version.

## Outcome

V2 completed on all three disjoint production shards and classified
`public_action_equivalence_proof_only_futile`.

The frozen aggregate covered all 800 open decisions and 2,995,314 complete
actions. Every source, replay, R3-apply, accounting, and order-invariance check
passed with zero invariant failures. The injected real-data duplicate witness
proved that the equivalence and `k - 1` accounting paths work when a duplicate
exists.

No natural duplicate class existed at any hierarchy level:

- semantic state plus supply;
- serving-safe ordered transition;
- exact public successor within a serving-safe class; or
- exact hidden successor within a serving-safe class.

Train and validation median, P90, P99, maximum, and weighted reductions were
all exactly zero. The promotion gate therefore failed by a wide margin.

The permanent decision is:

- keep the exact equivalence census, adversarial suite, and duplicate witness
  as proof and regression infrastructure;
- do not add an action-class layer to serving;
- do not spend learned-model or gameplay budget on S7 under the current action
  representation; and
- revisit only if a future hierarchical action generator deliberately creates
  multiple serializations of the same exact transition.

See
`docs/v2/reports/public-action-equivalence-foundation-v2-result.md`.
