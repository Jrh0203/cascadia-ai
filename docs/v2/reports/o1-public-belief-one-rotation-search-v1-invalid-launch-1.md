# O1 Public-Belief Search Invalid Launch 1

Experiment: `o1-public-belief-one-rotation-search-v1`

Invalid protocol: `o1-public-belief-one-rotation-high-regret-v1`

Replacement protocol: `o1-public-belief-one-rotation-high-regret-v2`

Decision: `0190-frozen-root-prelude-contingency-boundary.md`

Status: invalid launch; no scientific result

## Scope

The first production queue entered all four primary arms after successful
bundle fanout, input fanout, authorization, and crossed-host preflight. Every
primary failed deterministically on the same second panel group before any
production report was written.

No replay, collection, aggregation, sealed-test, gameplay, or score task ran.

## Failure

The failing complete action included a free three-of-a-kind market
replacement. Protocol v1 redeterminized hidden supply before applying that
root. The replacement therefore produced a different staged market than the
one recorded in the graded-oracle candidate. The subsequent frozen wildlife
placement was illegal in the altered state:

```text
group=9475110012915721395
sample=0
root=5fab9de804d78a5e7382f023b2ebfb1d4176ce91c3f6ab4840195d841582d305
tile at HexCoord { q: 0, r: 1 } cannot support Hawk
```

## Classification

This launch is administratively and scientifically invalid.

- It is not evidence for or against C0, A0, A2, or S3.
- It contains no comparable arm metric.
- It does not spend the validation domain because no complete report exists.
- It does not authorize threshold changes.
- It does not authorize sealed test, gameplay, or a score claim.

## Preserved Evidence

- bundle ID:
  `32af22eb973504babfa8b81cf503ba95c76078dbd97cbd92602a31d69fa4f645`;
- authorization ID:
  `9d619204a2171b0d89de84f509e0602fb3cd627a92ab07620e21fe12c7017991`;
- queue task prefix: `o1pbs-v1`;
- four failed primary attempts remain in
  `artifacts/cluster/research-queue-v1.json`;
- v1 authorization remains in
  `control/authorization-package/authorization.json`;
- partial run identities and first-row artifacts remain under `runs/`.

Blocked v1 replay, collection, and aggregation tasks are cancelled rather
than repaired in place.

## Root Cause And Permanent Fix

The frozen root is a complete staged public action, not an action template
that can be transplanted across prelude outcomes. Protocol v2 applies that
complete root against exact replay first, then redeterminizes all remaining
hidden future before opponent simulation.

The permanent contract, leakage analysis, and verification evidence are in
ADR 0190. Production restarts only from a new immutable bundle and
authorization.
