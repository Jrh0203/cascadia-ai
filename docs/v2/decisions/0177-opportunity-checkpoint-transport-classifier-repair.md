# ADR 0177: Opportunity Checkpoint Transport Classifier Repair

- Status: accepted
- Date: 2026-06-16
- Experiment: `opportunity-cross-attention-mlx-tournament-v1`
- Scope: terminal evidence transport validation and invalid-evidence reporting
- Does not change: data, model topology, initialization, training, checkpoints,
  predictions, metrics, thresholds, bootstrap procedure, selection order, or
  gameplay claims

## Context

The ADR 0173 terminal classifier received all four completed ADR 0166 reports
and byte-identical copies of their selected checkpoint manifests and model
tensors. It rejected the first arm before classification because it compared
the source checkpoint path's intrinsic leaf, such as
`step-000002000-epoch-0000-batch-002000`, with the arbitrary local transport
directory `c0_parent_conditioned`.

The content checks had already established that the copied manifest and model
matched the report. The transport directory is not part of checkpoint
identity. The manifest itself carries the immutable `checkpoint_id`, so that
is the correct object to compare with the source report path.

The invalid-evidence fallback then raised a second exception because Python
cannot order dictionaries while sorting path-identity records. Consequently
the classifier emitted neither a valid scientific verdict nor a durable
invalid-evidence artifact.

Both defects were observed only after all arm reports and checkpoints were
frozen. They affect evidence bookkeeping, not any scientific input or result.

## Decision

Checkpoint collection validation now requires:

1. one report and one collected checkpoint directory for every frozen arm;
2. exact BLAKE3 equality for `checkpoint.json` and `model.safetensors`;
3. equality between the manifest's intrinsic `checkpoint_id` and the leaf of
   the source checkpoint path recorded by the arm report; and
4. equality between the manifest model arm and the report arm.

The local collection directory name is deliberately ignored because it is a
transport concern.

Invalid-evidence report identities are sorted explicitly by their path string,
making the fallback deterministic and order-independent.

## Verification

The repair must pass:

- the complete opportunity classifier and pairwise suites;
- a regression accepting arbitrary transport directory names when intrinsic
  IDs and bytes match;
- a regression rejecting an intrinsic checkpoint-ID mismatch;
- a regression proving invalid evidence is path-sorted without a secondary
  exception;
- Ruff and format checks; and
- content-addressed, read-only repair bundling before queue installation.

The repaired classifier may consume only the already-frozen ADR 0166 reports,
checkpoint bytes, and untouched-C0 control. It may not rerun training or alter
the classifier semantics accepted by ADR 0173.

Qualification completed:

- complete classifier and pairwise suite: 14 passed;
- Ruff: pass;
- Python compilation: pass;
- diff check: pass;
- repair bundle:
  `9803d70158b0da4032a9692714de4ccee720f4ecb2c1341e7ffc47374c318749`;
- classifier BLAKE3:
  `722388631797a5e5b76e313ff1faedcd2ab4dd6cd86735c4201e755991db887b`;
- regression-test BLAKE3:
  `57e3b26378f7e91049e16d4bcfbb2b0220f0d6ef1497ba57126f9a83214d4aec`.

## Consequences

The failed ADR 0173 queue attempt remains preserved as execution evidence. A
new classifier-only task will run from the ADR 0177 repair bundle. Its result
is the first valid terminal classification for this production campaign.
