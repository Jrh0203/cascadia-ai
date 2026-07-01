# R6 Incremental Sparse Accumulator Foundation V1 Preregistration

Date: 2026-06-17

ADR: 0158

Experiment: `r6-incremental-sparse-accumulator-foundation-v1`

Protocol: `r6-apply-undo-parity-and-throughput-v1`

Status: frozen before production

## Question

Can one mutable sparse accumulator apply and undo every exact R3 complete
action edit with authoritative parity while taking at most half the time of
full afterstate construction?

## Frozen Corpus

```text
host: john2
first seed: 5,210,000
games: 4
positions: 320
rayon threads: 10
rules: four-player AAAAA, no habitat bonus
```

Seed `5,200,000` was used only for implementation calibration and is excluded.

## Frozen Procedure

For every complete action:

1. time full authoritative R3 apply over the complete action set;
2. time incremental apply plus undo over the same set;
3. in a separate pass compare each applied accumulator with an independently
   reconstructed authoritative snapshot;
4. undo and compare the exact parent digest.

Stable component keys use relative seat, terrain, and exact sorted members.
Traversal-local R2 component IDs are not accepted as durable identity.

## Frozen Gates

```text
exact apply failures == 0
exact undo failures == 0
exact apply checks == complete actions
exact undo checks == complete actions
authoritative_ns / incremental_apply_undo_ns >= 2.0
```

## Predictions

1. Exact parity will hold over every complete action.
2. Incremental apply plus undo will materially outperform full afterstate
   construction.
3. The speed ratio will remain comfortably above 2x even when undo cost is
   included.

## Invalidators

- source bundle or executable mismatch;
- missing actions or positions;
- timing different action sets;
- omitting undo from incremental timing;
- comparing only canonical hashes without structural parity;
- scientific hash mismatch; or
- changing the 2x gate after launch.

## Claim Boundary

A pass authorizes incremental serving and search integration. It does not
establish model quality or gameplay strength.
