# Opportunity Smoke Identity Projection Repair V1

## Verdict

**Accepted validator repair.** The original ADR 0166 smoke comparison was
invalid because its report-identity projection omitted two fields emitted by
the production trainer. The immutable smoke evidence passes after the exact
schema repair.

## Root Cause

All four reports signed `paired_panel` and `paired_panel_id` inside
`scientific_identity`. The original validator omitted those keys when
reconstructing the identity and rejected the first report as malformed.

Original invalid artifact:

- Path:
  `artifacts/experiments/opportunity-cross-attention-mlx-tournament-v1/control/cross-host-smoke-proof-invalid-launch-1.json`
- BLAKE3:
  `acd3f57f30bbc26e89be32af2e8420e1290f64e564496aef030a9983c2a50fe0`

## Repair Evidence

- Repair bundle:
  `artifacts/experiments/opportunity-cross-attention-mlx-tournament-v1/repairs/1f591720ad39381bf8aa60a4ef1fa880ba865249e0958934c023bfd492b7959d`
- Repaired proof:
  `control/cross-host-smoke-proof-repair-v1.json`
- Proof BLAKE3:
  `5623db6903c5e26c72421edad761aa200b86ccc5ccd958047997b48974df4f4c`
- Scientific proof ID:
  `0aec29500290e5b54c8c556007d2632d75df69b99ddb9022a3c7bf072118c05c`
- Classification:
  `opportunity_query_cross_host_smoke_pass`

## Cross-Host Result

Across john1, john2, john3, and john4:

- identical three-step batch identities;
- identical candidate counts;
- identical initialization;
- identical final checkpoint tensors;
- zero changed parameters between hosts;
- zero loss difference;
- zero prediction-score difference;
- zero uncertainty difference;
- identical stable rankings;
- exact R6 apply/undo parity;
- no information-boundary violation.

The repair changes no model, data, checkpoint, metric, threshold, or training
decision. It only makes the validator reconstruct the complete signed report
identity.
