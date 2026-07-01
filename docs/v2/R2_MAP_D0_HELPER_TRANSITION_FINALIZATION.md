# D0 Helper Transition Finalization

## Purpose

A helper-migration lineage is an immutable provisional lower-bound snapshot of
accepted old-helper work. It is not a claim that no additional old-helper
transaction completed before each host rotated. This distinction matters when
one host finishes a valid transaction while the three owner-local rotations
are being prepared.

The terminal finalization contract closes that race without rewriting signed
lineage, replaying valid work, or admitting an open-ended amendment mechanism.

## Two-stage contract

1. `cascadia.r2-map.d0-helper-transition.v1` is the signed provisional
   transition. Its accepted set is exactly the migration lineage bound by all
   three migration authorizations.
2. `cascadia.r2-map.d0-helper-transition-finalization.v1` is the one terminal
   signed closure. It binds the provisional transition, the previous
   transition epoch, all three migration-receipt cutoffs, the signed collision
   quarantine manifest, and any exact omitted old-helper transactions.

The finalization's effective accepted set is the provisional ordered prefix
followed by the ordered tail. The schema has no amendment array and a finalized
transition cannot be used as the input to another finalization.

For every finalized chain entry after the first transition epoch,
`previous_transition_sha256` must equal the immediately preceding signed
transition's semantic hash. Both packet-shape validation and execution-time
signature verification enforce this edge in addition to plan-file and helper
continuity, so a validly signed finalization cannot be spliced behind the wrong
prior transition.

## Admission rules

Every tail transaction must:

- be a verified, passing, sealed result bundle signed by the campaign key;
- target the transition's old helper, never its new helper;
- bind the exact packet, report, manifest, bundle, John1 canonical
  materialization receipt, and source-host materialization receipt;
- be absent from both the provisional accepted set and every other tail entry;
- be absent from the signed quarantine report set;
- occupy a unique later position in the frozen old execution plan; and
- finish after its source host's old-helper bootstrap epoch and no later than
  that host's new-helper installation cutoff.

All three migration receipt files and semantic identities are closed into the
finalization. The finalization timestamp must be at or after all three rotation
receipts. Missing, reordered, duplicated, post-cutoff, new-helper, quarantined,
or tampered tail entries fail closed.

## Runtime and aggregate behavior

Work packets embed the complete signed ordered transition chain. Runtime
preclaim verification and final aggregate verification both use the terminal
finalization's effective accepted set. A historical predecessor under an old
helper is authorized only when its exact transaction identity appears in that
set. The research-policy hashes remain a separate domain and are never
compared with execution-plan transition hashes.
