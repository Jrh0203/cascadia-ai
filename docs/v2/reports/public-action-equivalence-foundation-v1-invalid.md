# Public Action Equivalence Foundation V1 Invalid Attempt

Date: 2026-06-17

Experiment: `s7-public-action-equivalence-foundation-v1`

Protocol: `s7-exact-semantic-transition-v1`

Classification: `public_action_equivalence_invalid`

## Outcome

V1 produced no accepted shard report and no scientific compression estimate.
All three remote census processes were terminated after review found a
production class-accounting defect.

## Preserved Evidence

```text
immutable bundle:
  8952a7ca248aaf30519fe4a8e181533d37d766424ec2957660ccfa0b4ceaf02a
release binary SHA-256:
  6e7052bf9ca54e20f9d81fb9175bb0f763452567aaf7a7ecdfb4d76927d424e6
adversarial report:
  25c0f087b849b5b451e24d945dbc52694e9a813803a2d7c94ebee221c293471e
```

The bundle passed complete cross-host tree verification. The adversarial suite
and a one-group real-data smoke passed, but neither exercised production
accounting for a duplicate class.

## Defect

For a serving-safe class of size `n` split into `k` exact-public subclasses,
V1 used:

```text
n - k
```

for semantic collapses beyond exact-public identity. That is the number of
collapses retained by exact-public grouping. The required difference between
semantic and exact grouping is:

```text
(n - 1) - (n - k) = k - 1
```

The defect could reject or misdescribe the very duplicate classes under
study. It affects scientific bookkeeping, so changing the frozen V1
executable in place was not permitted.

## Disposition

- no V1 shard JSON was emitted;
- no V1 aggregate or order proof exists;
- no V1 result is used for research selection;
- the immutable V1 bundle remains preserved for provenance; and
- ADR 0163 and protocol V2 carry the corrected implementation and stronger
  duplicate-bearing smoke gate.
