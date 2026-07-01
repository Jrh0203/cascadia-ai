# Exact Habitat Rotation Batch Rejection

Status: **rejected and removed**

## Hypothesis

`top_habitat_placements` evaluates six rotations of each dual-terrain tile.
The scalar prepared preview rereads the current union-find root and size for
each neighboring habitat on every rotation. The experiment resolved each
neighbor once per frontier cell, then evaluated all six rotations from that
immutable snapshot.

This remained inside one top-placement scan. It did not move work across the
legacy place/undo replay boundary, so the qualified policy's path-compression
history was preserved.

## Exactness

Core replay-history tests compared batched and scalar previews before and after
24 temporary habitat replays. Greedy tests compared the bounded top-eight
scanner with the stable full sort and the optimized qualified policy with the
full reference scanner, including complete AAAAA games.

Both worker measurements reproduced:

- scores `[102, 96, 92, 95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

## End-To-End Screen

The two hosts used opposite order to reduce warmup bias:

| Host | Scalar control | Rotation batch | Treatment delta |
|---|---:|---:|---:|
| john2 | 15.699866 s | 15.882818 s | +0.182952 s |
| john3 | 16.054097 s | 16.072542 s | +0.018445 s |
| Combined | **15.876982 s** | **15.977680 s** | **+0.100698 s** |

The combined speedup was `0.99370x`, a `0.630%` regression. The union-find
trees were already shallow enough that the extra fixed-array initialization
and six batched passes cost more than the avoided root walks.

## Verdict

Reject. The all-rotations API, scanner integration, and temporary parity
assertions were removed. No fresh PGO build was justified after a cross-host
end-to-end regression.

Machine-readable evidence:
`docs/v2/reports/exact-habitat-rotation-batch-rejection-v1.json`.
