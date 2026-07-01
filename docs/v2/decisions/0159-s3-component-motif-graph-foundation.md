# ADR 0159: S3 Component-and-Motif Graph Foundation

Status: completed; foundation passed

Date: 2026-06-17

Experiment: `s3-component-motif-graph-foundation-v1`

Protocol: `s3-card-a-semantic-decoder-census-v1`

Research-plan item: S3

## Context

The strongest historical gain came from adding opponent detail, which
indicates that missing structured information matters more than blind search
budget. The current sparse state still asks a generic network to rediscover
habitat topology and wildlife motifs from individual cells.

S3 makes those long-range objects explicit and tests their exactness before
spending MLX training budget.

## Decision

Derive one exact graph per relative board with:

### Habitat

- component members and size;
- matching internal edges;
- open boundary;
- frontier contacts;
- cycle rank;
- bridge count;
- articulation count;
- size rank;
- merge-frontier count; and
- largest possible one-tile merge.

### Wildlife

- Bear connected components, singleton and pair opportunities, and oversize
  risk;
- Elk axis-aligned lines, endpoints, legal extensions, and overlap;
- Salmon connected paths, endpoints, branch conflicts, valid runs, and legal
  continuations;
- Hawk conflict graph, isolated Hawks, and isolated-placement opportunities;
- Fox centers, neighbor diversity, missing types, and compatible cells.

### Frontier

- degree distribution;
- habitat bridge opportunities;
- repeated component contacts;
- maximum resulting habitat size; and
- summed resulting habitat size.

## Four Exact Views

Measure canonical bytes and token counts for:

1. habitat components only;
2. wildlife motifs only;
3. components plus motifs;
4. components plus motifs plus frontier.

These are mechanical ablations. They do not yet train four models.

## Validation

At every position:

- decode exact current Card A score anatomy for all four boards;
- select one deterministic complete action;
- independently reconstruct its afterstate;
- decode the exact 12-part immediate score delta; and
- compare invariant graph signatures under all twelve D6 transforms.

The corpus must also observe at least one:

- Elk extension;
- Salmon continuation;
- Hawk isolated-placement opportunity; and
- Bear pair-completion opportunity.

This prevents a semantically empty corpus from passing.

## Production Corpus

| Variable | Value |
|---|---:|
| Host | john3 |
| First seed | `5,310,000` |
| Games | 14 |
| Positions | 1,120 |
| Board score checks | 4,480 |
| Action-delta checks | 1,120 |
| D6 checks | 13,440 |
| Rayon threads | 10 |

Calibration seed `5,300,000` is excluded from production evidence.

## Promotion Rule

Classify `s3_exact_component_motif_graph_promoted` only when:

```text
semantic decoder accuracy >= 99%
D6 transform failures == 0
D6 checks == positions * 12
all four opportunity families are observed
```

All individual failure counts remain visible even when the 99% aggregate gate
would otherwise pass.

## Consequences

A pass authorizes capacity-controlled MLX tests of:

- component only;
- motif only;
- component plus motif; and
- component plus motif plus frontier.

Those learned tests must use the same R3 actions, labels, initialization,
optimization budget, and serving protocol.

## Claim Boundary

This foundation proves semantic construction and symmetry. It does not prove
retained R4800 regret, tile-stage recall, Elk/Salmon/Hawk ranking gains,
gameplay improvement, or 100-point performance.

## Outcome

The production run classified S3 as
`s3_exact_component_motif_graph_promoted`.

- 4,480 board-score and 1,120 selected-action delta checks had zero failures;
- all 13,440 D6 checks passed;
- semantic decoder accuracy was 100%;
- Elk extensions appeared on 1,381 relative boards;
- Salmon continuations appeared on 1,322;
- Hawk opportunities appeared on 3,949; and
- Bear pair opportunities appeared on 1,182.

Median exact view sizes were 1,163 bytes for components, 255 for motifs, 1,424
combined, and 1,656 combined plus frontier. Median token counts were 76
components, 31 motifs, and 92 frontier objects. S3 advances to matched learned
ablations. See
`docs/v2/reports/s3-component-motif-graph-foundation-v1-result.md`.
