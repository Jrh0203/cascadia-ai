# Post-Draft Context Grouping Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Treatment

Complete-screen preparation grouped actions by tile slot, wildlife slot, and
wildlife-placement presence. Each group executed one representative full game
transition to construct the post-draft market and bag context, then reused the
accepted board place/undo and mid-v4 feature context for every placement in
the group.

The reference cloned and executed the complete legacy game independently for
every action.

## Exactness

The treatment and reference produced byte-identical paid-wipe qualification
reports on john2 and john3. Both files had SHA-256
`dc866e7fa52fbfc09701bc2a78bbd74e5064f88ac676fece39f27e1c8ed2e348`.

The feature-gated AI suite passed 89 tests, the differential bridge suite
passed 19 tests, and a dedicated oracle test compared every movement,
immediate-score bit, and ordered sparse feature row.

## Result

One treatment-capable binary was crossed in opposite host order:

| Host | Reference | Grouped | Improvement |
|---|---:|---:|---:|
| john2 | 22.81 s | 22.72 s | 0.395% |
| john3 | 22.26 s | 22.18 s | 0.359% |
| Combined | **22.535 s** | **22.450 s** | **0.377%** |

The mechanism removed full-game clones, but those clones were not the
dominant cost. Ordered sparse feature construction and MLX evaluation still
scale with every complete action in every chance branch.

## Verdict

Reject. A 0.377% source improvement does not materially advance the
242.43305-to-24.243305-second gate and does not justify a second preparation
implementation. All treatment code and its environment switch were removed.

Machine-readable evidence:
[`full-legal-audit-post-draft-context-grouping-rejection-v1.json`](full-legal-audit-post-draft-context-grouping-rejection-v1.json).
