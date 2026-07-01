# Full-Legal Multiplexed Stage Profile Preregistration

Status: **closed: completed**

Date: 2026-06-15

## Question

The accepted full-legal teacher now completes in 143.775461 seconds, with
103.119389 seconds in realized-hidden continuations. Exact row reuse is absent
within searches and negligible across multiplexed requests.

Which unavoidable stages now dominate on each Apple Silicon worker, and which
single stage offers enough end-to-end leverage to justify the next treatment?

## Frozen Profile

Run the complete seed-60999 turns 12/39/66 audit on john1, john2, and john3
with:

- `CASCADIA_NNUE_STAGE_TIMINGS=1`;
- `CASCADIA_MLX_STAGE_TIMINGS=1`;
- `CASCADIA_MLX_ACTIVATION_DIAGNOSTICS=1` on john3 only;
- exact K32/R600 champion search;
- exact R1200/R4800 confirmation;
- unchanged paid-wipe and realized-hidden diagnostics;
- accepted public caches and multiplexed trajectory scheduler;
- Card AAAAA, four players, no habitat bonuses.

Rust timings are serialized as exact nanosecond counters. The MLX service emits
one shutdown JSON record containing request, row, feature, decode, graph,
evaluation, materialization, response, request-size, prefix, and optional
activation diagnostics.

This profile changes no model, row, prediction, search, action, or game rule.
It is diagnostic-only and cannot become an accepted timing baseline.

## Correctness Gates

Every worker must preserve:

1. scores `[96,99,92,102]`;
2. terminal state
   `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`;
3. normalized semantic BLAKE3
   `f46ae73349d53d1baa3c69c0f8a3efab5766ed68ef91b6636ad65a3dea340c75`;
4. 33,260 logical neural batches, 55,710,626 logical rows, 44,903,952
   physical rows, 29,151 rollout waves, and 549,517 rollout samples;
5. zero bootstrap samples, zero policy fallbacks, zero bridge fallbacks, zero
   swaps, and clean shutdown;
6. maximum RSS below 1.5 GiB.

## Analysis Contract

Report for each node and combined:

- complete wall and realized-hidden wall;
- every Rust stage in milliseconds and percent of serialized stage total;
- MLX request count, rows, features, evaluation time, and non-evaluation
  service time;
- rows and evaluation time by request-size bucket;
- largest-batch row count, feature count, contiguous prefix opportunity, and
  activation density when available;
- unaccounted wall outside timed Rust stages;
- the top three optimization targets ranked by maximum possible Amdahl gain.

The next treatment must target the highest-ranked stage unless a lower-ranked
stage has materially lower implementation risk and comparable projected
end-to-end leverage. All treatment claims require a separate preregistration.

## Outcome

All three nodes completed the frozen profile with exact semantic parity, zero
process swaps, and bounded memory. Serialized MLX evaluation is the largest
directly measured critical-path stage at 47.813 seconds on the two remote
workers. Cumulative Rust intervals rank opponent advancement and rollout
template preparation next.

Full evidence:
[`full-legal-audit-multiplexed-stage-profile-v1.md`](full-legal-audit-multiplexed-stage-profile-v1.md).
