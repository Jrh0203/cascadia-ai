# Local-Geometry Dropout Preflight Repair V1 Preregistration

Date: 2026-06-16

Decision:
[ADR 0125](../decisions/0125-local-geometry-dropout-preflight-repair.md)

Experiment ID:
`conditional-tile-local-geometry-dropout-preflight-repair-v1`

## Frozen Difference

Repair only the failed resource mechanics from the first preflight:

- stream cache shards rather than retaining all seven;
- eliminate the redundant per-query item copy; and
- use exact tie-safe partition selection instead of a complete key sort.

The selected item set and order must remain byte-identical to the first
preflight. The original epoch-one selection BLAKE3 is
`87a234b381161f78eeefc63199dac85ba342492ed79cee060204a8f36516ed4e`.

## Gates

Every original ADR 0124 preflight gate remains unchanged. The repaired
contract and coverage arms must independently reproduce the original digest.
Preparation overhead must be at most 50%, peak process RSS below 4 GiB, and
process swaps zero.

Only the open train cache may be read. Training and every closed domain remain
closed.
