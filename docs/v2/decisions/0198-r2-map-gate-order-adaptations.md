# ADR 0198: R2-MAP Gate-Order Adaptations

Status: accepted before W0 source freeze

## Decision 1: qualify P1 before W0

Run the no-pruning production materialization/performance gate before freezing
W0, even though the original work breakdown placed performance qualification
later. W0 must not freeze a source transaction that is correct only in small
fixtures but cannot enumerate the complete live action set within the resource
contract.

This ordering already found a real correctness defect: v34 passed the targeted
and three-game gates, then the full-100 gate rejected the empirical 92-token
padding bound on a legal late-corpus wildlife sibling. The terminal RED is
preserved at
`reports/runs/run-rust-p1-open-dc7405dd-v34-full100`, with storage receipt
`control/receipts/req-6981d4dd747b44bb95b6c31cb4650dfa.json`. Freezing W0
first would have made the unusable 4x92 live contract authoritative.

P1 remains a qualification gate, not a license to prune actions or change the
scientific objective. A repaired source must repeat the complete targeted,
prefix, and full-100 panels before W0.

## Decision 2: qualify packing only after W7

Move the qualifying John1 MLX packing sweep after W7 creates and commits the
exact 100,000-game bootstrap dataset. A smaller pre-W7 sample cannot determine
the qualifying candidate-width distribution, while requiring the qualifying
sweep before dataset generation creates a circular dependency.

The source freeze still contains the complete strict sweep consumer and report
validator before W7. Those consumers freeze:

- caps 16, 32, 64, and 128;
- exactly twelve epochs;
- no candidate pruning and support through the frozen maximum candidate bound;
- receipt-bound source, dataset, compact-index, generation-manifest, and phase
  barrier identities;
- finite-loss, exact-resume, memory, and zero-swap gates; and
- fail-closed rejection of any non-100,000-game or unrelated dataset.

After W7, the sweep consumes only the committed exact corpus and publishes its
immutable report. No post-W7 code or threshold change is permitted to make a
result qualify.

## Effect

These are dependency-order corrections, not extra synchronization. Game
generation remains embarrassingly parallel across john1, john2, and john3;
John1 training remains single-host; john2 remains canonical storage/build; and
john4 remains unused.
