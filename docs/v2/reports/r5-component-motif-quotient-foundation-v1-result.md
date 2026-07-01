# R5 Component-and-Motif Quotient Foundation V1 Result

Date: 2026-06-17

ADR: 0157

Classification: `r5_local_geometry_exact_and_quotient_compact`

Outcome: passed

## Result

R5 passed every frozen exactness and compactness gate over 20 four-player
AAAAA games.

| Metric | Result |
|---|---:|
| Positions | 1,600 |
| Complete actions | 2,423,019 |
| Current-board score checks | 6,400 / 6,400 |
| Raw-control affordance failures | 0 |
| Quotient-only underdetermined actions | 2,423,019 / 2,423,019 |
| Local affordance failures | 0 |
| Local score-delta failures | 0 |
| Control parent median / P99 tokens | 329 / 495 |
| Quotient parent median / P99 tokens | 196 / 289 |
| Quotient/control median token ratio | 0.595744 |
| Local patch median / P99 bytes | 44 / 117 |

## Interpretation

The component-and-motif quotient preserves exact current Card A score
semantics but does not contain enough local geometry to prove a complete
action legal. That is the intended result: the quotient does not smuggle raw
board geometry into aggregate objects.

Adding only the selected destination neighborhood, optional wildlife site,
and active nature tokens recovered every legal affordance and immediate score
delta tested. The minimal exact boundary observed here is therefore:

```text
long-range component and motif graph
+ exact action-local geometry
+ public market, player, and semantic-supply state
```

The model-facing parent token surface is 40.4% smaller at the median and 41.6%
smaller at P99 than exact R2. The rich audit postcard is larger than R2
because it serializes variable-length proof evidence; it is not the intended
serving tensor.

## Decision

Promote quotient plus action-local geometry into a matched MLX comparison.
Retain exact R2 as control and full sparse-plus-relational hybrid as the
quality ceiling.

## Provenance

- bundle:
  `3d7f8a563319ba6636a3fb06d3db33fd846f8dbfdd70beeabd3d5a0d509eeee3`
- scientific report:
  `94b4171122b5d0c3af2170836444964cfbed4b7f4a7983ffbcc629a1ea7e91d5`
- host: john1
- seeds: `5,110,000-5,110,019`

## Claim Boundary

This result authorizes learning. It does not show retained R4800 quality or
gameplay improvement.
