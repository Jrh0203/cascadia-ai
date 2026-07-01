# Exact Resolved Habitat Preview Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

Within each original `top_habitat_placements` boundary, resolve every occupied
neighbor's union-find root and root size once per frontier cell, then reuse the
snapshot across the six rotations of a dual-terrain tile.

The experiment preserved every scan, replay, candidate, random stream, and MLX
request boundary. Its preregistered minimum was a 0.25% positive combined
source-level speedup.

## Exactness

Board-level tests proved resolved and prepared previews equal before and after
legacy habitat replay history. Existing top-eight and complete-game greedy
parity tests passed.

All four R600 treatment runs reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps and zero policy fallbacks.

## End-To-End Measurement

Both hosts ran treatment, control, treatment, control with unprofiled release
binaries:

| Host | Control mean | Resolved mean | Speedup | Result |
|---|---:|---:|---:|---:|
| john2 | 15.716281 s | 15.743300 s | 0.99828x | 0.172% slower |
| john3 | 16.067420 s | 16.113924 s | 0.99711x | 0.289% slower |
| Combined | **15.891850 s** | **15.928612 s** | **0.99769x** | **0.231% slower** |

Control SHA-256:
`53e62599c9fb9d8c6f232a850d3f7247f668662913297f4ebc36440a21107487`

Treatment SHA-256:
`2d1fa49d20e71c5568357e622c90c134eb9220f2caddabb8497386d69d3d24a4`

## Verdict

Reject. Union-find root traversal was already cheap at these shallow board
depths. Constructing and consuming a second neighbor representation moved
slightly more work into the scan than it removed. The resolved types, methods,
tests, and call-site changes were removed.

Machine-readable evidence:
`docs/v2/reports/exact-resolved-habitat-preview-rejection-v1.json`.
