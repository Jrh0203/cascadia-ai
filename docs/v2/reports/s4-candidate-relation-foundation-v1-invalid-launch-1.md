# S4 Candidate-Relation Foundation V1 Invalid Launch 1

Date: 2026-06-17

Experiment: `s4-candidate-relation-foundation-v1`

Classification: `foundation_invalid`

## Failure

Launch one formed `same_draft` by copying the structured action array,
zeroing placement and outcome fields, and viewing the full 128-byte record as
an exact byte key. The action dtype has three unnamed alignment-padding bytes
at offsets 39 through 41. NumPy structured copies preserve named values but do
not define those padding bytes as semantic data. The same canonical draft
could therefore receive different keys.

`same_sibling_plan` concatenated that contaminated draft record with the exact
tile pose and inherited the defect. Union relation topology, linkage, graph
components, and the frozen anchor-width classification consequently lack a
valid exact-key foundation.

## Scope

Unaffected launch-one surfaces:

- observable screen-rank anchor selection;
- R4800 winner retention, confidence coverage, and retained regret;
- `same_frontier`;
- `same_tile_pose`;
- `same_wildlife_destination`;
- `equivalent_afterstate`; and
- row coverage and forward/reverse merge determinism.

Invalidated launch-one surfaces:

- `same_draft`;
- `same_sibling_plan`;
- union relation linkage and topology;
- relation edge totals; and
- `s4_anchor_256_authorized`.

The invalid source bundle is
`950161591fe877ffbb17c2ebc7214b2a90581217795e327a121bc42689c8b188`.
Its reports are preserved under
`artifacts/experiments/s4-candidate-relation-foundation-v1/reports/invalid-launch-1/`.

## Root Fix

The corrected key builder creates a zero-filled structured array and copies
every named field explicitly before zeroing relation-variant fields. Padding
is therefore deterministically zero and cannot influence equality.

A regression test constructs two semantically identical actions, writes
different values into padding bytes 39 through 41, and requires both
`same_draft` and `same_sibling_plan` keys to remain equal. The complete S4
focused suite passes after the fix.

## Recovery

1. Build a new immutable source bundle containing the corrected key builder.
2. Fan it out byte-identically to john2, john3, and john4.
3. Rerun the same modulo-three open-data shards.
4. Collect reports by checksum.
5. Merge in forward and reverse order and require byte-identical output.
6. Apply the original frozen ADR 0151 gates without modification.

No launch-one metric may authorize a neural architecture.
