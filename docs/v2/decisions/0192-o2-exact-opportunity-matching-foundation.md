# ADR 0192: O2 Exact Opportunity Matching Foundation

**Status:** Accepted for open-train execution  
**Date:** 2026-06-17  
**Experiment:** `o2-exact-opportunity-matching-v1`

## Context

T1 established that direct exact-R2 ranking is stronger than immediate
qualified-leaf rescoring and every tested one-rotation search horizon. O1
showed a small behavioral future-access signal but no eligible integration.
Exact semantic supply is factual infrastructure, while its earlier direct
learned integration regressed protected scarce-supply and independent-draft
slices.

The missing question is relational: which open board opportunities can still
be satisfied by current or unseen public supply before access and competition
make them unavailable?

## Decision

Introduce schema-versioned `OpportunityGraphV1` in `cascadia-data`.

The public-only bipartite graph contains:

- wildlife placement demands with exact Card A one-step completion deltas;
- habitat-frontier demands with exact component-growth deltas;
- current market wildlife and tile components;
- exact unseen wildlife counts and semantic tile archetype counts;
- compatibility edges with canonical archetype rotation masks;
- exact availability fractions, access delay, and opponents before access;
- a deterministic integer teacher value; and
- an exact capacitated maximum-weight matching teacher.

The graph constructor accepts `PublicGameState`, never `GameState`, rollout
targets, R4800 estimates, model predictions, hidden stack order, excluded tile
identity, or hidden wildlife order. Canonical demand IDs are semantic kind,
subject, and public coordinate. Supply IDs are semantic kind and market slot,
wildlife, or archetype identity. Edge IDs are the canonical endpoint pair.

Schema v1 teacher value is:

```text
floor(
  exact_completion_delta * 1,000,000
  * availability_numerator / availability_denominator
  / (1 + access_delay_turns + opponents_before_access)
)
```

The teacher solves the resulting integer capacitated component graph exactly.
It is explicitly a component-feasibility relaxation, not a claim that several
matched components can all be drafted in one turn. Joint paired/independent
draft consistency remains a learned or higher-order assignment question.

Canonical codecs use `CSOPPG1\0` for graphs and `CSOPPM1\0` for matching
summaries, an explicit schema version, exact payload length, and canonical
postcard bytes. Unknown versions, trailing bytes, unsorted IDs, invalid
fractions, dangling edges, and noncanonical encodings fail closed.

## Required Invariants

Before scientific output is inspected:

1. same public state and focal seat produce identical bytes and teacher output;
2. canonical graph and summary codecs round-trip exactly;
3. all 12 D6 transforms produce covariant demand IDs and edge rotation masks,
   invariant supplies, and invariant teacher objective;
4. changing hidden order without changing public state cannot change output;
5. demand, supply, and edge input order cannot change canonical output;
6. the matching solver respects capacities and finds the global integer
   optimum on adversarial fixtures; and
7. the public constructor has no target or model input.

## Scientific Protocol

The frozen protocol is in
`reports/o2-exact-opportunity-matching-v1-preregistration.md`. It reuses only
the 560-group, 35,840-candidate strict open-train top-64 T1 cohort. It does not
rerun T1 search and does not open validation, sealed test, or gameplay.

## Consequences

- john1 owns foundation implementation, invariants, export, deterministic
  summary treatment, analysis, and local classification.
- One immutable contract/fixture bundle may be published only after the
  foundation passes.
- Large row caches stay on john1's external SSD.
- Learned O2 arms are authorized only if the frozen identifiability gate
  passes.
- A null closes this exact schema/teacher mechanism; it does not justify a
  larger neural sweep.
- No offline result can establish a gameplay score or progress toward 100.
