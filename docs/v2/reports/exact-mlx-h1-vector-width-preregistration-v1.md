# Exact MLX H1 Vector-Width Preregistration

Status: **closed: rejected and removed**

Date: 2026-06-15

## Question

The replicated full-legal profile spends 47.813 seconds in serialized MLX
evaluation, 37.123% of remote wall. Its exact H1 kernel assigns one `float4`
output vector to each Metal thread, so every row uses 128 threads and every
thread independently traverses the same ordered sparse indices.

Can evaluating `float8` or `float16` per thread reduce repeated index-loop and
scheduling overhead while preserving the exact operation order of every
scalar output?

## Treatment

Add two experimental H1 kernels:

- width 8: 64 threads per row, two `float4` accumulators per thread;
- width 16: 32 threads per row, four `float4` accumulators per thread.

The retained width-4 kernel remains the same-binary control. All variants use
a 256-thread Metal threadgroup, so a group covers two, four, or eight rows.
For every hidden output, features are still added in the original CSR order
with the same `float32` additions and ReLU.

The treatment changes no host row, row order, feature order, multiplicity,
weight, bias, later-layer kernel, protocol, prediction order, search state,
random stream, rollout budget, or game rule. It introduces no planner,
metadata, sort, or scatter.

The temporary selector is `CASCADIA_MLX_H1_VECTOR_WIDTH=4|8|16`. Acceptance
removes the selector and retains one production kernel. Rejection removes the
two treatment kernels and selector.

## Correctness Gates

Before timing:

1. all three widths match the Rust-order reference bit for bit on empty,
   duplicate, prefix-related, and arbitrary sparse rows;
2. all focused Python MLX tests pass;
3. all feature-enabled differential and NNUE batch tests pass;
4. every timed report validates and preserves the frozen score, state,
   semantic digest, logical work vector, zero bootstrap, zero fallback, and
   clean shutdown;
5. every run reports zero process swaps and RSS below 1.5 GiB.

Any bit, action, score, diagnostic, or semantic mismatch rejects the
treatment immediately.

## Latin-Square Screen

Run the frozen seed-60999 turn-66 audit once per width on every Mac:

- john1 order: 4, 8, 16;
- john2 order: 8, 16, 4;
- john3 order: 16, 4, 8.

Each run includes the turn-66 full-legal screen, R1200/R4800 confirmation,
paid-wipe diagnostic, multiplexed realized-hidden continuations, Card AAAAA,
four players, no habitat bonuses, and exact K32/R600 search.

A treatment width advances only if:

- combined complete wall improves by at least 1.0% versus width 4;
- combined MLX evaluation time improves by at least 3.0%;
- at least two of three nodes improve in both metrics;
- no node regresses complete wall by more than 1.0%;
- every correctness and resource gate passes.

## Full-Contract Confirmation

The best qualifying width runs an opposite-order ABBA crossover on john2 and
john3 over the complete turns-12/39/66 audit. It is accepted only if:

- both workers improve complete wall;
- combined complete wall improves by at least 1.0%;
- both workers reduce MLX evaluation time;
- exact semantics and the full logical work vector remain unchanged;
- zero-swap and memory gates pass.

The switch-free production run must then reproduce the same result before the
accepted teacher baseline moves.

## Outcome

Both wider geometries failed the Latin-square screen on all three Macs. Width
8 regressed combined complete wall by 1.360% and MLX evaluation by 4.720%.
Width 16 regressed wall by 4.696% and MLX evaluation by 15.210%. No treatment
qualified for full-contract confirmation.

The treatment kernels and selector were removed. Full evidence:
[`exact-mlx-h1-vector-width-rejection-v1.md`](exact-mlx-h1-vector-width-rejection-v1.md).
