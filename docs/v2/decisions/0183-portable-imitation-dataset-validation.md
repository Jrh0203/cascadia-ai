# ADR 0183: Separate Portable Dataset Validation From Collector Weight Validation

**Status:** Accepted  
**Date:** 2026-06-16  
**Repairs:** O1 reuse-audit invalid launch 1

## Context

The first crossed-host O1 corpus-reuse launch used immutable bundle
`a861a5e5ec44fb14451c2749b6a17006696169516af88f09321ab0a72944cfd4`.
Bundle and dataset fanout completed with whole-tree checksum parity. The john4
primary and john2 replay then both exited in less than one second, before any
trajectory was replayed, with:

```text
Error: Io(Os { code: 2, kind: NotFound, message: "No such file or directory" })
```

`validate_imitation_dataset` called `ImitationTeacherConfig::validate`, which
required the manifest's absolute collector-local weight path to exist. The
frozen manifests correctly retain
`/Users/johnherrick/cascadia/nnue_weights_v4opp_modal_iter3.bin` as provenance,
but that path does not and should not exist on john2 or john4.

The dataset payload already embeds weight byte count and BLAKE3. Requiring the
original weight file during read-only shard validation adds no payload
integrity and makes an otherwise immutable dataset non-portable.

## Decision

Split teacher validation into two contracts:

1. `validate_metadata` checks that strategy, rollout, prefilter, path, byte
   count, and checksum metadata are complete.
2. `validate_weights` additionally opens the referenced weight file and checks
   its byte count and BLAKE3.

Dataset creation and resume continue to require `validate_weights`.
Read-only `validate_imitation_dataset` uses `validate_metadata` plus complete
manifest, shard-size, shard-checksum, header, group, and record validation.

Add a regression test that constructs a valid immutable dataset whose recorded
collector weight path is absent. Read-only validation must pass, while a
collection/resume config using the same absent path must fail.

## Relaunch Rules

- The first bundle and failed task attempts remain immutable evidence.
- Blocked collection and classification tasks from `o1reuse-v1` are
  administratively cancelled.
- No scientific observation was produced, so the preregistered hypotheses,
  metrics, and gates remain unchanged.
- Rebuild the executable and source bundle under a new content hash.
- Relaunch with task prefix `o1reuse-v2`.

## Consequences

Canonical imitation datasets can now be independently validated anywhere the
dataset is copied, without weakening collection-time teacher integrity. The
repair also benefits imitation target, parent-prior, and parent-hidden
consumers that call the shared portable validator.

