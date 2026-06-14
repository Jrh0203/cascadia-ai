# ADR 0004: Public-Information Determinization

Status: accepted on 2026-06-10.

## Context

The canonical game state owns the exact hidden tile-stack order, excluded
tiles, and wildlife-bag order so seeded simulation can reproduce games. A
search policy must not exploit that information.

## Decision

Search samples an information set by:

1. preserving boards, market, scores, turn, history-derived counters, and all
   other public facts,
2. pooling the unseen tile stack with the unknown excluded tiles,
3. shuffling that pool and restoring the same stack/exclusion counts,
4. independently shuffling the remaining wildlife bag,
5. using the same determinization seeds for every candidate in a comparison.

The operation is deterministic from an explicit `GameSeed`, validates under
the normal game invariants, and cannot change the current legal-action set.
Oracle experiments that retain the actual hidden order must be labeled
separately and cannot promote production behavior.

## Consequences

- Search estimates action value over information-consistent futures.
- Common random numbers reduce paired candidate variance.
- Reproducible tests can verify that no public fact changes.
- Hidden-state sampling remains a rules-engine operation, while seed
  scheduling and rollout policy stay in `cascadia-search`.
