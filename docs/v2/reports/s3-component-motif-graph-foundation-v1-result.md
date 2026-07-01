# S3 Component-and-Motif Graph Foundation V1 Result

Date: 2026-06-17

ADR: 0159

Classification: `s3_exact_component_motif_graph_promoted`

Outcome: passed

## Result

S3 passed semantic, symmetry, and opportunity-coverage gates.

| Metric | Result |
|---|---:|
| Positions | 1,120 |
| Board score checks / failures | 4,480 / 0 |
| Action delta checks / failures | 1,120 / 0 |
| D6 checks / failures | 13,440 / 0 |
| Semantic decoder accuracy | 100% |
| Boards with Elk extensions | 1,381 |
| Boards with Salmon continuations | 1,322 |
| Boards with Hawk opportunities | 3,949 |
| Boards with Bear pair opportunities | 1,182 |

View-size medians:

| View | Bytes | Tokens |
|---|---:|---:|
| Raw R2 control | 652 | n/a |
| Component only | 1,163 | 76 |
| Motif only | 255 | 31 |
| Component plus motif | 1,424 | 107 |
| Frontier objects | n/a | 92 |
| Component plus motif plus frontier | 1,656 | 199 |

## Interpretation

The exact graph captures the Card A long-range objects the prior feature
audit identified as missing. It reconstructs score anatomy, immediate action
effects, and symmetry without learned assistance.

Audit bytes are not a compactness win for the combined graph. The strategic
question is whether explicit objects are more learnable than raw tokens at a
matched parameter and serving budget. Motif-only state is compact enough to
be a particularly useful ablation.

## Decision

Authorize four capacity-controlled MLX arms:

1. component only;
2. motif only;
3. component plus motif;
4. component plus motif plus frontier.

Each arm must be compared with exact R2 and R5 quotient-local controls on the
same R3 corpus and labels.

## Provenance

- bundle:
  `3d7f8a563319ba6636a3fb06d3db33fd846f8dbfdd70beeabd3d5a0d509eeee3`
- scientific report:
  `d43a8f00536a7494e71f92730beb842fa2e79b196889dda8a54aa9d5edab5771`
- host: john3
- seeds: `5,310,000-5,310,013`

## Claim Boundary

This result proves exact graph semantics, not learned superiority.
