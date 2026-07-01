# Full-Legal Multiplexed Trajectory Search Preregistration

Status: **closed - accepted**

Date: 2026-06-15

## Question

The accepted full-legal audit still spends 121.252030 of 162.045309 seconds
inside three realized-hidden diagnostics. Each diagnostic advances eight
independent finalist continuations, but the current implementation solves
those trajectories serially.

Can one exact multi-search scheduler advance the continuations in lockstep,
share one Rayon pool and one MLX evaluator, and combine contemporaneous sparse
requests without changing any independent K32/R600 search?

## Frozen Treatment

Only realized-hidden terminal continuations change:

- retain all eight finalist branches and advance them one completed turn at a
  time;
- continue satisfying exact game-scoped public-decision cache hits before
  scheduling teacher work;
- submit every remaining public state to one
  `ExactMlxLegacyTeacher::select_actions` cohort;
- preserve one independent candidate frontier, sequential-halving state,
  public-state-derived RNG, rollout allocation, row order, and selected action
  per search;
- run search preparation through the process-global Rayon pool rather than
  one pool per process;
- allow at most one outstanding evaluator request per unfinished search;
- wait deterministically until every unfinished search has either submitted
  one request or completed;
- concatenate requests in stable search-index order, execute one exact MLX
  evaluation, and split predictions back by their original ranges;
- perform no cross-request sparse-row deduplication in this experiment, so the
  isolated mechanism is request multiplexing and coordinated CPU work;
- merge per-search bridge and logical neural diagnostics exactly.

The control remains the accepted serial finalist loop. The temporary
qualification switch will be `--disable-multiplexed-realized-hidden`. If
accepted, remove the switch and serial production branch.

The treatment changes no model, weights, candidate limit, rollout budget,
halving rule, rollout horizon, random stream, action order, cache semantics,
public-information boundary, or game rule.

## Correctness Gates

1. Unit tests prove owned sparse requests preserve values and that multiplexed
   responses are split into the exact original request ranges.
2. Deterministic test evaluators show serial and multiplexed searches select
   identical actions and produce identical per-search estimates.
3. The frozen seed-60999 report preserves every semantic record after
   provenance, timing, service, cache, bridge, and batch diagnostics are
   removed.
4. Final scores remain `[96,99,92,102]` and the final-state BLAKE3 remains
   `7b3f520d5441aa2ae9c3d97d87e0cd08299d546a7d1e8398f76fdadedb53fa7d`.
5. Logical neural batches, rows, rollout waves, rollout samples, bootstrap
   count, and policy fallback count remain exact.
6. Every worker shuts down cleanly with zero swaps.

## Mechanism Gates

Advance only if:

1. at least 700 exact searches execute through multi-search cohorts;
2. at least 90% of multiplexed evaluator batches combine two or more search
   requests;
3. physical evaluator service batches fall by at least 50% relative to the
   logical evaluator requests;
4. no request is reordered within a search and no prediction range mismatch
   occurs.

## Performance Gates

Run one treatment-capable binary in opposite order:

- john2: control, treatment, treatment, control;
- john3: treatment, control, control, treatment.

Accept only if:

1. complete uncontended wall time improves by at least 10% on both hosts;
2. combined complete wall time improves by at least 15%;
3. combined realized-hidden wall time improves by at least 15%;
4. maximum RSS and allocator peak remain below 1.5 GiB per process;
5. no host swaps, falls back, bootstraps, or leaks a worker;
6. the result remains positive in a final switch-free production run.

If exact multiplexing reduces service requests but misses the full-wall gate,
reject it rather than retain scheduler complexity without sufficient leverage.

Intermediate measurements authorize no large audit collection.

## Outcome

The treatment passed every correctness, mechanism, cross-host performance,
memory, and production gate. The qualification switch and serial production
branch were removed.

Full evidence:
[`full-legal-audit-multiplexed-trajectory-search-acceptance-v1.md`](full-legal-audit-multiplexed-trajectory-search-acceptance-v1.md).
