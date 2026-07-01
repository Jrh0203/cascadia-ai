# Complete-Action Frontier Free-Residual Audit V1 Preregistration

ADR 0103 freezes a local objective, optimizer, and representation audit before
another full model treatment.

## Question

Does the current scale-16 objective fail inside the residual box, does frozen
AdamW fail even with one free parameter per action, or does the complete
public-observable model remain unable to realize a locally fit solution?

## Frozen Arms

| Host | First-wave arm | Frozen work |
|---|---|---|
| john1 | analytic optimum | exact box-constrained CE optimum on 64 groups |
| john2 | free residual AdamW | 24 groups through 1,200 updates |
| john3 | projected control | accelerated convex control on the same 24 groups |
| john4 | neural continuation | first of four disjoint 1,200-exposure groups |

As the three short analytic arms finish, john1-john3 backfill the remaining
three neural group shards. This is divisible work, not duplicate training.

## Decision Gates

- The selector ceiling must recover 100% of every target set.
- The analytic CE optimum passes at 95% recall, 75% exact sets, and KKT
  violation at most `1e-8`.
- Projected optimization must match the analytic objective within `1e-7` and
  match its exact selection metrics.
- Frozen free-residual AdamW passes at 1,200 updates with 95% recall and 75%
  exact sets.
- Full-model local continuation passes at 1,200 exposures with 90% recall and
  75% exact sets.

The mechanical outcome selects objective-box mismatch, frozen optimizer
insufficiency, local budget insufficiency, public-observable representation
insufficiency, nonreproduction, unresolved mechanism, or pipeline invalidity.

## Prohibitions

No extra group, seed, optimizer sweep, learning-rate sweep, full 560-group
trainer, width treatment, conflict mitigation, validation-driven selection,
sealed test, gameplay, cloud, Modal, or external compute.

Full algorithms, thresholds, classification precedence, cluster scheduling,
resource gates, and replay requirements are authoritative in ADR 0103.
