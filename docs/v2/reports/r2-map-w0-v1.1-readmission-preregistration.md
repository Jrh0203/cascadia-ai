# R2-MAP W0 v1.1 append-only readmission preregistration

Date: 2026-06-18

Status: frozen procedure; awaiting the final W1/W4 legality-repair source
identity

## Purpose

Register the repaired sequential public-market implementation without mutating
the frozen W0 v1 evidence or opening any new seed domain.

## Immutable predecessor

The append-only chain is fixed to:

- repository/canonical v1 manifest formatted SHA-256
  `12555a92ab337eca8d299210e19f5c4bb52298822e82f688ad967ceeaed1f7ec`;
- v1 manifest canonical SHA-256
  `5d88e296810eb5f8c5abc67ebc317ce987a2edb11d97b0c4e55ea873d96e5a65`;
- v1 registration SHA-256
  `7d0336714a1e520c9c99f0d488e48577848f6c0b336ca6257ae987f2548e0d51`;
  and
- paired-power file SHA-256
  `029bb8ab6b432e739157cfb686fb7bd7302e526add968d4cecfab99831b0694f`.

The registration was recovered without SSD access from an immutable Codex
session transcript and is now canonical at
`control/w0-preregistration/registration.json`. The receipt-bound provenance is
`control/w0-preregistration/transcript-recovery-provenance-v1.json`, SHA-256
`40b28702c30ba49c22f984b4774e89119595b4c2c2b070b29bb39a418edf5e88`.
The predecessor remains an immutable stale-negative execution input. Its
historical SSD path strings are evidence, not active storage directives.

## Seed boundary

V1.1 reuses byte-for-byte the 100 seeds derived from
`r2-map-open-reference-performance-100-v1`. V1 outcomes were never opened, so
no outcome or seed selection is reused. The manifest must assert:

- `seed_domain_changed=false`;
- `predecessor_outcomes_opened=false`;
- `old_v1_outcomes_reused=false`;
- `strength_claim_authorized=false`; and
- every protected descriptor has `opened=false` and
  `seed_material_present=false`.

The W0 tool has no interface for protected seed values. The 20-pair blinded,
250-pair development, and 1,000-game final domains remain sealed.

## Source freeze

Wait for the final W1/W4 public-market legality repair and ADR 0197 live-token
capacity repair. Then freeze one
immutable expanded John2 source transaction containing every v1.1 source
binding, the exact protocol fixture, the generator, tests, complete Cargo
workspace closure, and the rendered repository manifest at
`docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json`.

The rendered manifest is valid only if exact regeneration from that same source
transaction is byte-identical. A manifest generated before the final legality
repair is stale and must never be published as W0. The same rule applies to the
rejected 4x92 live shape: a manifest generated before the exact 93-token
replay, 4x139 Rust/Python/service migration, and P1 full-100 resource gate is
stale. ADR 0198 makes P1 a pre-W0 source qualification; P2 and later
optimization comparisons remain post-W7.

## Registration sequence

1. Independently re-read the predecessor registration, v1 manifest, power
   file, and transcript-recovery provenance from John2.
2. Render v1.1 from the final immutable source transaction.
3. Publish identical bytes to
   `control/w0-preregistration/reference-panel-manifest-v1.1.json`.
4. Render `registration-v1.1.json` on John2 so its repository path names the
   immutable source transaction and its canonical path names the John2 control
   object.
5. Publish the registration immutably and reopen both objects.
6. Run Python exact regeneration and registration verification.
7. Run the Rust source rehash and open-panel initializer against v1.1; prove
   the v1 predecessor is rejected.
8. Publish one evidence report binding all source, run, object, cleanup, and
   read receipts, followed by independent John2 and John3 verification.

## Required v1.1 semantics

- Market-choice feasibility is public-universal: a choice may be advertised
  only when it is feasible from the visible market and the public per-species
  bag counts for every hidden bag permutation consistent with those counts.
- The advertised market-choice screen is byte-identical across hidden bag
  permutations with the same visible market and per-species counts, and every
  advertised choice commits successfully for every such permutation.
- An independent Python oracle must reject any Rust screen that is only a
  partial subset of the complete public-universal legal screen.
- Free three-of-a-kind replacement is a public keep/replace decision with zero
  Pinecone spend.
- Paid wipes are sequential stop-or-single-wipe decisions. Each committed wipe
  spends exactly one Pinecone and reveals exactly one new public market before
  the next decision.
- No future refill vector may be observed or scored before commitment.
- Scores are invalidated after every public reveal.
- Every legal market choice and every legal draft action is scored exactly
  once in canonical engine order; no pruning is permitted.
- Live R2-MAP tensors use the proved 4x139 capacity and v3 grouped/market
  request protocols. The archived 4x92 foundation cache remains distinct and
  an obsolete 4x92 live request fails closed.
- Teacher, search, and direct-gameplay paths preserve the same chance
  semantics. Conditional resampling after a choice is selected may not make an
  advertised action executable or silently substitute a different hidden
  outcome.
- Replay retains stage, ordered legal choice hashes, parent public-state hash,
  selected choice hash, resulting public-state hash, and the final bundled
  prelude.

## Acceptance tests

- `tools/test_r2_map_reference_panels.py` passes in full.
- `python/tests/test_r2_map_market_decision.py` independently reconstructs the
  complete public-universal screen and rejects partial, duplicate, reordered,
  stage-mixed, or hidden-state-dependent screens.
- All library tests pass for `cascadia-game`, `cascadia-r2`,
  `cascadia-model`, `cascadia-search`, and `cascadia-data` from the same
  immutable source transaction.
- The minimized v34 93-token replay passes canonical/incremental parity under
  all twelve D6 transforms; slot 138 succeeds, slot 139 fails, and compact
  Rust/Python dataset materialization preserves the same rows.
- P1 passes three games, a prefix beyond the original failing game/turn, and
  the full open 100-game corpus below 4 GiB with zero process swaps and zero
  system swap delta.
- Exact render/verify succeeds twice with identical bytes.
- Repository and canonical John2 v1.1 manifest bytes are identical.
- Registration verification produces one complete v1.1 implementation binding.
- Rust positive v1.1 source/registration verification succeeds.
- Rust negative v1 verification and source/protocol/fixture drift cases fail.
- Exhaustive hidden-permutation tests prove public-universal screen identity,
  successful commitment of every advertised choice, independent Python
  completeness, and identical direct-gameplay/teacher/search chance semantics.
- W5 uses the same implementation binding and open-domain identity.
- No protected seed, W7 packet, controller transition, or terminal/stop object
  is created.

Any source change in a bound file before publication invalidates the candidate
manifest and forces exact regeneration. Any source change after publication
requires a new append-only revision; it cannot overwrite v1.1.
