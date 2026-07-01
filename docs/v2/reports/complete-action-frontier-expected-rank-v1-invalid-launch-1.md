# Complete-Action Frontier Expected-Rank V1 Invalid Launch 1

Status: invalid before treatment metrics.

The first ADR 0100 wave exposed a cache lookup representation bug. Group IDs
are stored as unsigned 64-bit values, while decoded MLX batches preserve the
same bits in signed `int64`. High-bit IDs therefore appeared negative and did
not match the cache's unsigned keys.

The canonical and independently reproduced cache arrays were already
byte-identical and remain unchanged. Training failed during initial validation
before writing `initial-validation.json`, and the baseline failed before
writing its report. The widest-group gradient diagnostic happened to select a
low-bit ID and completed, but it was quarantined with the wave to preserve one
corrected source identity across every accepted job.

The lookup now normalizes decoded IDs modulo `2^64`, with a dedicated
regression test. The corrected full suite passed, the focused test passed on
all four Macs, and the corrected 103-file runtime bundle was byte-identical.

Preserved evidence:
`artifacts/experiments/complete-action-frontier-expected-rank-v1/invalid-launch-group-id-sign/`.
