# Full-Legal Static-Screen Cohort Preregistration

Status: **accepted**

Date: 2026-06-15

## Evidence

The frozen turn-16 paid-wipe qualification submitted 3,664,806 sparse rows in
23,115 service requests. Of those, 1,826,342 rows arrived in requests of at
most 128 rows and consumed 7,978.305 ms of MLX evaluation. Requests above
1,024 rows processed 1,364,170 rows in 1,674.962 ms.

Static complete screens currently reuse the rollout pipeline's 96-row cohort.
That cohort was chosen to overlap rollout state preparation with inference;
static screens have no producer-consumer pipeline and receive no such benefit.

## Treatment

Keep rollout inference at the accepted 96 states. Add an independent static
screen cohort size used only by complete-action prior and hidden inference.
Sweep `512,1024,2048,4096,8192` rows.

The shared-memory mapping, exact Rust-order MLX kernels, feature rows, row
order, outputs, and logical diagnostics remain unchanged. No action is
removed or deduplicated in this experiment.

## Frozen Screen

- seed `60999`, completed turn `16`;
- R600 trajectory;
- two-token paid-wipe hidden-invariance qualification;
- D8 root determinizations, D2 followup determinizations, width 3;
- all 15 first-wipe subsets;
- one treatment-capable binary;
- opposite host order on john2 and john3.

The reference report SHA-256 is
`dc866e7fa52fbfc09701bc2a78bbd74e5064f88ac676fece39f27e1c8ed2e348`.

## Gates

Each cohort must:

1. produce the reference report byte for byte on both hosts;
2. retain hidden-state invariance and clean shutdown;
3. fit the fixed 8 MiB shared mapping without fallback;
4. improve both hosts versus 96 rows;
5. reduce request count and MLX evaluation time as predicted;
6. avoid material RSS or reliability regression.

Choose the fastest cohort by combined host time. Confirm it in balanced
repeats, then run the complete early/middle/late audit reference. Reject the
entire treatment if the selected cohort does not improve the full contract.

## Outcome

The `4,096`-row cohort won the two-host sweep, passed balanced crossover
confirmation, and reduced the complete frozen audit from `242.43305` to
`217.302693625` seconds with an identical semantic payload.

Acceptance report:
[`full-legal-audit-static-screen-cohort-acceptance-v1.md`](full-legal-audit-static-screen-cohort-acceptance-v1.md).
