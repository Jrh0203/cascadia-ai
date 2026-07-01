# ADR 0158: R6 Incremental Sparse Accumulator Foundation

Status: completed; foundation passed

Date: 2026-06-17

Experiment: `r6-incremental-sparse-accumulator-foundation-v1`

Protocol: `r6-apply-undo-parity-and-throughput-v1`

Research-plan item: R6

## Context

Complete-action ranking and search repeatedly construct sibling afterstates
from one parent. Rebuilding the entire sparse state and relational graph for
every sibling wastes the exact locality already exposed by R3 action edits.

R6 tests whether one mutable active-board accumulator can apply and undo those
edits exactly. This is a prerequisite for a hybrid sparse NNUE plus relational
graph: cheap broad evaluation is useful only if candidate transitions do not
rebuild the world.

## Decision

Implement `IncrementalSparseAccumulator` over:

- active-board occupied tiles;
- market snapshot;
- exact semantic supply;
- active-player public summary;
- legal frontier tokens;
- habitat component objects;
- wildlife motif objects;
- completed turns; and
- current relative seat.

For every complete legal action from every production position:

1. construct the authoritative R3 public afterstate;
2. apply the R3 prelude, placement, board, market, supply, frontier,
   component, motif, and turn edits to the accumulator;
3. compare the accumulator with an independently reconstructed authoritative
   snapshot;
4. undo the action; and
5. require the exact parent digest to return.

## Stable Component Identity

R2 component IDs are traversal-local. Adding one tile may renumber components
that did not change. Incremental state must not treat those IDs as durable.

R6 keys components by a BLAKE3 digest over:

- a schema domain separator;
- relative seat;
- terrain; and
- sorted exact component members.

Frontier touches are normalized to those keys before parity comparison. This
is a representation correction, not an equality relaxation: all component
members, sizes, contact edges, and frontier consequences remain exact.

## Timing Protocol

At each position, over the identical complete-action set:

- authoritative timing applies every action through the full R3 afterstate
  path and consumes its canonical record hash;
- incremental timing applies and undoes every action on one accumulator and
  consumes a changed accumulator field;
- exact parity is checked in a separate untimed pass; and
- nanosecond totals are accumulated over the complete corpus.

The implementation does not omit undo cost from the incremental measurement.

## Production Corpus

| Variable | Value |
|---|---:|
| Host | john2 |
| First seed | `5,210,000` |
| Games | 4 |
| Positions | 320 |
| Rayon threads | 10 |
| Rules | four-player AAAAA, no habitat bonus |

Calibration seed `5,200,000` is excluded from production evidence.

## Promotion Rule

Exact parity requires:

```text
apply failures == 0
undo failures == 0
apply checks == complete actions
undo checks == complete actions
```

The throughput gate requires:

```text
authoritative apply time / incremental apply-plus-undo time >= 2.0
```

Both must pass for classification
`r6_incremental_apply_undo_promoted`.

## Consequences

A pass authorizes:

- incremental relational feature maintenance;
- sibling-action batching without parent reconstruction;
- search tree apply/undo;
- sparse NNUE delta evaluation; and
- later cross-turn tree reuse.

A parity failure blocks all incremental use. A throughput failure preserves
the implementation as exact negative evidence but does not place it in the
serving path.

## Claim Boundary

R6 measures transition mechanics, not learned value quality. It does not show
that a hybrid model ranks actions better, scores more points, or reaches the
100-point target.

## Outcome

The production run classified R6 as
`r6_incremental_apply_undo_promoted`.

- 320 positions and 506,425 complete actions were evaluated;
- all 506,425 incremental afterstates matched the independently reconstructed
  authority;
- all 506,425 undo operations restored the exact parent digest;
- authoritative full apply consumed 127.257 seconds of aggregate timed work;
- incremental apply plus undo consumed 2.162 seconds; and
- the measured ratio was 58.864821x against the frozen 2x gate.

The accumulator was 3,092 bytes median and 5,262 bytes P99. The result
authorizes integration into sibling-action evaluation and search. See
`docs/v2/reports/r6-incremental-sparse-accumulator-foundation-v1-result.md`.
