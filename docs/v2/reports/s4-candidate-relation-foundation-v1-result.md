# S4 Candidate-Relation Foundation V1 Result

Date: 2026-06-17

ADR: 0151

Experiment: `s4-candidate-relation-foundation-v1`

Status: corrected production census complete

Classification: `s4_anchor_256_authorized`

## Verdict

Use 256 observable screen-ranked anchors for the first S4 neural comparison.
The corrected 256-anchor surface passed every frozen oracle-retention and
relation-coverage gate over all 240 validation decisions and 860,203 complete
legal actions.

The smaller 128-anchor surface preserved every validation confidence set and
had low regret, but linked 93.41% of complete validation candidates to an
anchor, below the frozen 95% requirement. It remains a useful serving-cost
ablation, but it is not the primary context width.

This result authorizes a separately preregistered neural comparison. It does
not establish gameplay strength or a score improvement.

## Immutable Evidence

| Identity | Value |
|---|---|
| Corrected source bundle | `69512ef62dd125d0231541a4bcf55cfeb861ef2eae79c75082cb90df72bdfc34` |
| Open-data verification | `a056aceadb7f53c01dc87c8a39d95a7866bac6df93b050c45cc860de2b8b87ea` |
| R3 cache | `0de6365fe5dfe57329298e1c3370baeddf14e6edc5909fa930c234d1abc97156` |
| Aggregate report | `2b977892c9b899d2fb9b38cfeb1b2e10c9a4f778650cf68dbadc78b28a33c7fc` |
| Aggregate SHA-256 | `eafc423a86be1587b842cf02dd93b9f8cd82533468c153a7e652b6eeb13212e6` |

Forward and reverse merges are byte-identical at 181,121 bytes.

Launch one used bundle
`950161591fe877ffbb17c2ebc7214b2a90581217795e327a121bc42689c8b188`
and was invalidated because non-semantic structured-action padding entered
two exact relation keys. Its reports remain preserved under
`reports/invalid-launch-1`; see
`s4-candidate-relation-foundation-v1-invalid-launch-1.md`.

## Validation Result

| Anchors | Winner retained | Confidence coverage | R4800 regret | Winner linked | All candidates linked |
|---:|---:|---:|---:|---:|---:|
| 64 | 71.67% | 98.75% | 0.1130 | 99.58% | 84.03% |
| 128 | 81.25% | **100.00%** | 0.0576 | **100.00%** | 93.41% |
| 256 | 90.42% | **100.00%** | **0.0287** | **100.00%** | **98.37%** |

Winner retention is descriptive, not a gate: every candidate remains a query
to the context model. The binding question is whether the anchor set preserves
the high-fidelity target and connects complete queries to relevant
alternatives.

At 256 anchors:

- early, middle, and late linkage was 99.68%, 98.51%, and 97.63%;
- `1..512`, `513..2048`, `2049..4096`, and `4097+` linkage was 100.00%,
  100.00%, 99.09%, and 97.93%;
- every phase and action-width stratum had 100% confidence-set coverage;
- every validation winner linked to at least one other anchor; and
- every anchor had at least one exact-relation sibling.

The result directly covers the middle-game and wide-action regimes implicated
by the completed compact-arm R3 failure atlases.

## Relation Anatomy At 256 Anchors

| Relation | Query linkage | Winner linkage | Anchor siblings |
|---|---:|---:|---:|
| Same draft | 67.90% | 98.75% | 99.79% |
| Same frontier | **92.79%** | **99.58%** | 99.84% |
| Same tile pose | 82.56% | 97.08% | 90.43% |
| Same wildlife destination | 45.83% | 95.42% | 97.34% |
| Same sibling plan | 12.06% | 57.50% | 37.75% |
| Equivalent afterstate | 5.61% | 42.92% | 22.36% |

The corrected draft relation is much denser than launch one reported. It
links more than two thirds of complete queries and nearly every winner while
remaining semantically distinct from frontier and tile-pose relations.
Sibling plans are also three times denser than the invalid launch suggested.

The useful surface is therefore not a generic global moment. Frontier,
tile-pose, and exact draft provide broad reach; wildlife, sibling-plan, and
equivalent-afterstate relations disambiguate the local competitive set.

## Compute Consequence

Across validation, dense attention over 256 anchors requires 15,728,640
pair-score evaluations. Sixteen inducing points require 1,966,080, an exact
8x reduction before projection and feed-forward costs.

The corrected explicit relations contain 6,154,696 overlapping pair edges.
Expanding every relation into separate pairwise attention is therefore not
the efficient first implementation. The neural tournament should compare:

1. independent radius-one/global-edit candidate scoring;
2. 256 anchors summarized through 16 inducing latents;
3. bounded exact relation-neighbor summaries for draft, frontier, tile pose,
   wildlife destination, sibling plan, and equivalent afterstate; and
4. the combined inducing-plus-relation model.

Every complete candidate must query the shared context in chunks. The model
must not discard candidates outside the anchor set.

## Distributed Execution

| Host | Shard | Train groups | Validation groups |
|---|---:|---:|---:|
| john2 | `row % 3 == 0` | 187 | 80 |
| john3 | `row % 3 == 1` | 187 | 80 |
| john4 | `row % 3 == 2` | 186 | 80 |

The corrected merge covered all 560 train and 240 validation rows exactly
once. The immutable bundle was whole-tree verified on all three hosts, shard
reports were checksum-collected, and forward and reverse merges produced the
same report identity and bytes.

## Consequences

1. Freeze 256 anchors and 16 inducing latents for the first S4 comparison.
2. Preserve exact relation IDs; do not replace them with mean/max moments.
3. Score every complete candidate as a query; anchors provide context, not a
   hard shortlist.
4. Use bounded, stable relation neighbors rather than materializing all
   6.15 million overlapping validation edges.
5. Use the R3 radius-one plus exact-global representation as the compact
   treatment substrate unless the final ADR 0150 classifier invalidates the
   experiment foundation.
6. Keep 128 anchors as a preregistered serving-cost ablation because the
   corrected linkage gap to the gate is only 1.59 percentage points.
7. Keep dense all-pairs attention and post-compression ADR 0096 context
   closed.
