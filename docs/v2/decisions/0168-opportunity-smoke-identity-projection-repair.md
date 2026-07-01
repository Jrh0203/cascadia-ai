# ADR 0168: Opportunity Smoke Identity Projection Repair

- Status: accepted
- Date: 2026-06-17
- Experiment: `opportunity-cross-attention-mlx-tournament-v1`
- Supersedes: only the failed `oppquery-v2-smoke-compare` validator invocation
- Does not supersede: ADR 0166, its frozen training protocol, any smoke run, any
  model tensor, or any scientific threshold

## Context

The four ADR 0166 bounded smoke runs completed successfully on john1 through
john4. The original cross-host validator nevertheless classified the evidence
as malformed before performing numerical comparisons.

The generated training reports include `paired_panel` and `paired_panel_id` in
their signed `scientific_identity`. The validator's
`_report_scientific_identity` projection omitted those two fields. It therefore
could not reconstruct the identity of any report emitted by the production
trainer, even when both values were `null`.

This was a validator-schema defect. It did not affect training, checkpoint
bytes, batches, predictions, performance measurements, or the frozen
information boundary.

## Decision

1. Preserve the original invalid proof as
   `control/cross-host-smoke-proof-invalid-launch-1.json`.
2. Add `paired_panel` and `paired_panel_id` to the validator identity
   projection.
3. Extend the validator test fixture with the production report shape and add
   a regression test that requires both fields in the reconstructed identity.
4. Freeze the repair under content-addressed repair bundle
   `1f591720ad39381bf8aa60a4ef1fa880ba865249e0958934c023bfd492b7959d`.
5. Re-evaluate the immutable four reports and four checkpoints. Do not rerun
   training.
6. Add a new queue task, `oppquery-v2-smoke-compare-repair-v1`, and make ADR
   0166 authorization depend on that completed task. Retain the failed original
   task and attempt history.

## Frozen Repair

| File | BLAKE3 |
|---|---|
| `tools/opportunity_cross_attention_mlx_smoke_compare.py` | `c2c41ea3cb59f7d2a9da324ab2618f9cfc349b7bc0b12b6852063133400052ca` |
| `tools/test_opportunity_cross_attention_mlx_smoke_compare.py` | `8c87952d698e7ef8678c7869cd31e5b2ad5c0c549ffb2bd00a889ed700a2714b` |

The repaired validator imports the unchanged ADR 0166 Python package from
bundle `b435cd92f27398bffa3167b4b2405c0f879bc121ab101ad09df81e0a3cf39ab6`.

## Verification

- Ruff: pass.
- Validator tests: 3 passed.
- Repaired proof classification:
  `opportunity_query_cross_host_smoke_pass`.
- Proof ID:
  `0aec29500290e5b54c8c556007d2632d75df69b99ddb9022a3c7bf072118c05c`.
- Every host check passed.
- Cross-host loss, parameter, prediction-score, and uncertainty deltas were all
  exactly zero.

## Consequences

No scientific gate is weakened. The repair expands the validator projection to
match the already-signed trainer schema. Production remains blocked until the
new queue task reproduces the canonical proof and ADR 0166 authorization
validates it.
