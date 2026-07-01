# Opportunity Cross-Attention MLX Tournament: Invalid Launch 1

- Date: 2026-06-17
- Experiment: `opportunity-cross-attention-mlx-tournament-v1`
- Bundle: `b435cd92f27398bffa3167b4b2405c0f879bc121ab101ad09df81e0a3cf39ab6`
- Authorization: `b602504ab7c4fdf73f48cf4c28dbd8dd9189ad7ca13c49d24b83738dba5e3fcc`
- Verdict: invalid launch; no scientific arm result accepted

## Failure

The untouched exact-R2 C0 control scored the complete paired validation panel
and then failed while constructing its canonical panel identity:

```text
OverflowError: Python integer -5482088856184735585 out of bounds for uint64
```

The dataset stores an opaque 64-bit group hash through a signed `i64` surface.
NumPy 2.4 rejects direct conversion of a negative Python integer to `uint64`.
The same terminal-report path is shared by all treatment arms, so the original
bundle could not produce valid final evidence even if optimization completed.
ADR 0172 defines the exact signed-to-unsigned bit-pattern normalization.

## Preserved Work

The three remote treatments were terminated deliberately after trace step 400:

| Host | Arm | Last trace step | Last durable checkpoint |
|---|---|---:|---:|
| john2 | `t1-supply-query` | 400 | 250 |
| john3 | `t2-frontier-query` | 400 | 250 |
| john4 | `t3-combined-query` | 400 | 250 |

All three traces used the same deterministic batch identities through step 400.
Their losses differed only by the expected bounded host-level floating-point
variation. The run directories remain untouched on their originating hosts.
They are audit evidence only and are excluded from classification.

The parent-conditioned arm never started because its dependency on the failed
C0 control remained blocked. The collection and classification tasks also
never started.

## Queue Accounting

The three active claims were closed as `cancelled` with the ADR 0172 defect as
their reason. The blocked parent-conditioned, collection, and classification
tasks were administratively cancelled as superseded. The failed C0-control task
remains failed so the original defect is visible in the queue history.

## Relaunch Contract

The repair uses:

1. a new content-addressed immutable bundle containing the normalization;
2. a distinct task prefix;
3. a launch-scoped artifact root for smoke, authorization, preflight, runs,
   reports, collection, and classification;
4. a fresh four-host smoke proof and production authorization; and
5. four fresh 2,000-step runs from the same exact-R2 warm start.

No checkpoint or optimizer state from this invalid launch may enter the
accepted relaunch.
