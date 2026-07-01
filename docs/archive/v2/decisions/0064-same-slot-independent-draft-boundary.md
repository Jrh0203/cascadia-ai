# ADR 0064: Same-Slot Independent Draft Boundary

Status: accepted on 2026-06-12.

## Context

The official Cascadia Nature Token rule permits taking any habitat tile and any
wildlife token. Therefore an independent draft may choose both components from
the same market slot. That action is legal, but it is strictly dominated by the
free paired draft from the same slot because it produces the same drafted
components while spending one Nature Token.

Official rulebook:
<https://www.alderac.com/wp-content/uploads/2021/08/Cascadia-Rules.pdf>

The canonical rules engine must represent every legal action. Ranked gameplay
frontiers, however, should not spend scarce candidate capacity on a move that
can never improve the resulting state.

## Decision

- Keep same-slot independent drafts legal in `cascadia-game`.
- Make `Market::take_independent` transactional by validating both components
  before mutating the market.
- Keep legal-action generation and transition tests for the same-slot case.
- Exclude same-slot independent drafts from greedy, habitat, Bear, and unified
  pattern strategy rankings.
- Preserve same-slot independent actions in stored legal-action datasets.
- Encode the paid legacy bridge action as `wildlife_market_index = Some(slot)`;
  the free paired action remains `None`, so the isolated move type preserves
  the distinction exactly.

This is a strategy-dominance filter, not a rules change. Stored legal-action
datasets remain valid and require no schema migration.

## Verification

- `cargo test -p cascadia-game`
- `cargo test -p cascadia-sim`
- `cargo test -p cascadia-differential --features legacy-teacher`

All targeted tests passed after the change.

## Amendment

ADR 0069's full-corpus parent-prior derivation exposed same-slot independent
records among deterministic legal negatives. The earlier bridge rejection was
stale: the legacy move record distinguishes the paid action from the free pair,
and its market implementation supports same-slot execution. The bridge now
round-trips those actions and verifies Nature Token consumption. Ranked policy
frontiers still exclude the strictly dominated move.
