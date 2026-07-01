# V2 Performance

Measurements were collected on the project machine described in
`HARDWARE.md`: a 10-core Apple M4 Mac mini with 16 GB unified memory. Commands
use release builds unless the benchmark harness controls the profile.

## Deterministic Criterion Baseline

| Benchmark | Measured time |
|---|---:|
| Initial legal-action generation | 12.872-14.740 us |
| Initial board scoring | 230.20-232.46 ns |
| Active-board preview | 351.89-380.14 ns |
| Transactional transition | 4.8790-4.9075 us |
| Complete four-player random game | 4.3789-4.4862 ms |

Run the deterministic suite with:

```bash
cargo bench -p cascadia-game
cargo bench -p cascadia-sim
```

## Search Profile

The first promoted search retained four exact-greedy candidates and evaluated
each across four public-information determinizations for four greedy plies.
Native `sample` profiling identified repeated `score_board` calls and
`Board::largest_habitat` as the dominant hot path.

The optimized implementation flattens candidate-by-determinization jobs across
Rayon and performs exact dependency-aware delta rescoring after each tile and
wildlife placement. Full scoring remains the reference implementation.

On pilot seeds 20300-20319:

| Implementation | Paired wall time | Games per second |
|---|---:|---:|
| Full-board rescoring | 153.461 s | 0.130 |
| Exact delta rescoring | 41.115 s | 0.486 |

Speedup: **3.73x**.

The optimized run reproduced all aggregate metrics, category deltas, and every
per-seed result bit for bit. `cascadia-sim` also tests delta scoring against
full scoring for every A-D wildlife-card family over every legal candidate in
a representative midgame state.

The 50-game confirmation suite, seeds 20400-20449, was also reproduced exactly:
wall time fell from 232.233 to 111.129 seconds, a 2.09x speedup. The run shared
the machine briefly with compilation work, so the uncontended 20-game
reproduction is the cleaner micro-comparison; both establish unchanged
behavior and materially higher throughput.

## Product Budgets

The versioned acceptance contract is
[`config/performance-budgets-v1.json`](../../config/performance-budgets-v1.json).
`make performance-check` verifies every budget against checksummed canonical
reports, and `make performance-report` regenerates the machine-readable and
Markdown qualification artifacts.

| Tier | Complete game | P90 decision | P99 decision |
|---|---:|---:|---:|
| Instant, exact immediate-score greedy | <=0.50 s | <=25 ms | <=75 ms |
| Interactive, pattern-aware K8/H6/B8/M4 | <=1.25 s | <=50 ms | <=100 ms |
| Research, final-five R8 c90 | <=10 s | <=1,000 ms | <=3,000 ms |

The Apple-GPU boundary additionally requires at least 50,000 evaluations per
second and at most 1 ms P99 latency for batch 32.

The API runs interactive and research evaluation on a blocking worker so CPU
work cannot stall the async request executor. The product and research
benchmarks are `make benchmark` and `make benchmark-research`; determinized search remains
available through `make lookahead-benchmark`.

On the 50-game direct product control, pattern-aware averaged 3.64 ms per
decision with 6.86 ms P90 and 48.49 ms maximum. It scored 91.890 against K8 at
90.775 while running 14.49x faster. Its shared-frontier optimization replaced
three redundant legal-action enumerations and reproduced every pre-optimization
pilot score exactly.

Canonical reports now time strategy selection at every decision and publish
mean, P50, P90, P99, and maximum latency separately from whole-game wall time.
New JSON and Markdown artifacts also embed the full typed command
configuration, hardware and toolchain details, executable checksum, Git status
digest, complete v2 source digest, and input model/run manifest checksums.

## Dataset Collection

Value and grouped-ranking collectors execute complete, deterministically seeded
games in parallel within each shard, then restore game-index order before the
single atomic shard write. Search inside each game uses the same Rayon pool, so
nested work is cooperatively scheduled instead of creating per-game thread
pools. Dataset bytes, manifests, checksums, and resume cursors remain identical
for a fixed executable and configuration regardless of worker scheduling.

## Terminal Policy Improvement

The original R8 terminal teacher repeatedly recomputed connected habitat
components and wildlife scores for candidate states. The production path now:

- analyzes all habitat components once per board;
- evaluates a candidate tile through a reusable placement context;
- recomputes only wildlife-dependent components after wildlife placement;
- stores bounded board coordinate sets in stack-backed collections;
- caches each pre-existing legal `(wildlife, coordinate)` score once per
  decision and evaluates only placements introduced by a candidate tile.

Full scoring remains the oracle. Tests compare the optimized result with full
scoring for every legal action across Card A-D wildlife variants. The
ten-seed pattern-aware reference also remains exactly identical in every score
and breakdown.

| Workload | Before | After | Result |
|---|---:|---:|---|
| Pattern-aware ten-seed reference | 0.506 s/game | 0.075 s/game | 6.8x faster |
| Full-game R8 reference seed | 273.884 s | 80-85 s before final cache | exact parity |
| Final-five R8 hybrid pilot | 11.512 s/game | 7.506 s/game | runtime gate passed |
| Final-five R8 hybrid confirmation | n/a | 7.530 s/game | 382 ms P90 decision |
| Final-five R8 c90 confirmation | n/a | 6.995 s/game | 362 ms P90 decision |

The exact wildlife cache slightly increases overhead in the already-cheap
pattern-aware strategy because it uses a general hash map. Its value is in
terminal search, where the same placement is rescored many times. A future
specialized fixed-index cache may recover that small ordinary-play overhead,
but the current implementation is correct, bounded, and comfortably within
the product budget.

The research terminal policy uses the exact pattern-aware action as an anchor and
only replaces it when eight shared terminal samples put a challenger's
one-sided 90% paired lower bound above zero. On seeds 28000-28049 it measured
87.4 ms mean over all decisions, 1.2 ms median, 362 ms P90, 881 ms P99, and
2.253 seconds maximum in its historical confirmation. Its corrected
requalification measured 172.6 ms P90. The long tail occurs only in terminal
search; all pre-cutoff decisions are the interactive policy exactly. ADR 0068
demoted it after the corrected run failed the non-Bear wildlife guardrail.

## Exact MLX R600 Rollout Optimization

A native `sample` profile of the qualified K32/R600 teacher showed that the
MLX service was not the limiting component. One complete game submitted
6,135,934 neural rows in 4,014 batches, while service startup took only 102 ms.
The dominant native stacks were:

- `board_potential`;
- `Board::place_tile` and `Board::undo`;
- `Board::frontier`;
- `candidate_moves_pub`;
- wildlife scoring.

IPC reads and feature extraction were much smaller. Moving more inference work
to MLX would therefore not address the current wall-time bottleneck.

Two exact optimizations now cover the repeated CPU work:

1. `candidate_moves_pub` has a default-on thread-local single-entry cache. The
   key stores the exact relevant public state rather than a digest, includes
   tile rotations, and therefore cannot change play through a hash collision.
   Hidden bag order is intentionally absent because candidate generation does
   not read it. `CASCADIA_MCE_CACHE=0` remains available for direct legacy
   performance A/B runs.
2. Candidate potential evaluation derives the post-placement frontier from the
   already-computed parent frontier. It no longer rescans every placed tile for
   every candidate. Unit tests compare the derived frontier and potential with
   full recomputation across every initial market tile, frontier coordinate,
   and legal rotation.

The repeated-state microbenchmark now measures a cached
`candidate_moves_pub` call at 0.05 us and a complete cached
`pick_best_move_nnue` at 260.86 us. With the cache disabled, the same binary
measures 773.45 us and 1,047.59 us respectively. Real games have a lower hit
rate because rollout trajectories diverge.

The strict same-seed R600 comparison is the representative result:

| Metric | Before | After |
|---|---:|---:|
| Treatment time per game | 147.217 s | 137.910 s |
| Mean treatment decision | 1,840.21 ms | 1,723.87 ms |
| Wall-time reduction | - | **6.32%** |
| Effective speedup | 1.000x | **1.067x** |

The before/after runs had identical baseline and treatment score breakdowns,
bridge diagnostics, 4,014 neural batches, 6,135,934 neural rows, 3,800 rollout
waves, and 46,150 rollout samples. The full `cascadia-ai` library suite passed
71 tests.

### Full-terminal R600 acceleration

The first cache/frontier pass exposed more exact redundancy in the native
candidate path. The completed implementation keeps the same K32/R600
sequential-halving policy and runs every rollout to the terminal position.
There is no learned leaf, rollout cutoff, reduced candidate set, reduced
rollout budget, or policy fallback.

The additional exact work reductions are:

1. Group active rollout states by an exact public-state key and generate the
   deterministic move template once per unique state in each wave.
2. Separate that template from state-specific bag and opponent features, then
   prepare each state's neural rows from the shared template.
3. Place and undo candidate moves on each rollout board rather than cloning
   the board for every candidate afterstate.
4. Preview the exact post-placement habitat total without mutating
   union-find state.
5. Update habitat-frontier and empty-slot potential from a parent-board
   context instead of rescanning the complete board.
6. Reuse tile-placement previews across all wildlife pairings for the same
   market tile.
7. Produce the candidate set and legacy greedy fallback in one traversal.
8. Score each species' best pre-existing wildlife placement once per decision.
9. Reuse wildlife choice and potential across the six rotations of the same
   tile coordinate; only habitat connectivity is rotation-dependent.

The public-state key includes tile rotations and the stable wildlife insertion
order observed by legacy tie-breaking. It stores the exact values rather than
a digest, so key collision cannot alter play. Tests compare habitat previews,
local potential updates, cached candidate generation, the combined greedy
fallback, and batch policy behavior with their reference implementations.

Reproduce the local full-search measurement with:

```bash
make legacy-nnue-mlx-full-strength-speed
```

Seed 34400 fell from the initial 141.027-second profile to 37.457 seconds:

| Metric | Initial full R600 | Optimized full R600 |
|---|---:|---:|
| Treatment seconds | 141.027 | 37.457 |
| Effective speedup | 1.00x | **3.77x** |
| Rollout limit | none | none |
| Bootstrapped samples | 0 | 0 |
| Policy fallbacks | 0 | 0 |
| Mean score | 95.500 | 96.250 |

The score values are included as a regression signal, not as a strength
estimate from one game. The optimized path is deterministic on repeated seed
34400 runs. The final rotation-reuse step also reproduced the preceding
optimized run's score and all search diagnostics exactly while reducing
treatment time from 60.395 to 37.537 seconds.

The strength and sustained-throughput verification then ran 50 fresh games,
seeds 34500-34549, across john1, john2, and john3 with the same executable,
model, weights, and R600 contract:

| Metric | Result |
|---|---:|
| Games / seat scores | 50 / 200 |
| Mean score | **96.345** |
| Game-block 95% CI | `[95.892,96.798]` |
| Historical champion estimate | 95.940 |
| Paired control mean | 91.920 |
| Mean paired delta | +4.425 |
| Paired 95% CI | `[+3.814,+5.036]` |
| Paired game record | 48-1-1 |
| Mean treatment time | 36.335 s/game |
| Single-Mac projected speedup | **3.88x** |
| Three-node treatment wall | 656.483 s |
| Treatment throughput speedup | **10.74x** |
| Complete paired-report wall | 705.655 s |
| End-to-end paired speedup | **9.99x** |
| Rollout limits | all none |
| Bootstrapped samples | all 0 |
| Policy fallbacks | all 0 |

The treatment-throughput comparison projects the original 141.027-second
profile across 50 sequential games and compares it with the slowest optimized
treatment shard. The 9.99x end-to-end number additionally charges the new run
for 50 paired control games and three MLX service lifecycles. The local
algorithmic gain is about 3.8x; concurrent use of all three Macs supplies the
remaining throughput.

The 50-game score is above the historical 95.94 champion estimate and its
confidence interval contains that estimate. More importantly, all 2,306,322
rollout samples reached the terminal policy: the run recorded zero bootstrap
samples, zero policy fallbacks, and clean shutdown on every host. The evidence
artifact is
`docs/v2/reports/full-strength-r600-speedup-v2.json`.

### Score-sacrificing truncated-rollout screening tier

Full terminal R600 search remains the authoritative promotion evaluator. A
new opt-in screening tier stops each rollout after two focal-player moves and
uses MLX to estimate the remaining score:

```bash
make legacy-nnue-mlx-fast-screen
```

The underlying command is:

```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 \
target/release/legacy-teacher exact-mlx-productive-token-compare \
  --server-program /opt/homebrew/bin/uv \
  --model-dir artifacts/models/legacy-nnue-v4opp-mlx-v1 \
  --games 10 \
  --first-seed 34200 \
  --rollouts 600 \
  --rollout-turns 2 \
  --rollout-leaf-timing after-focal-move \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --output /tmp/cascadia-fast-screen.json
```

The rollout limit includes the root candidate. `after-focal-move` evaluates
the second focal afterstate immediately. `after-opponent-round` first lets the
other three players complete the round. Omitting `--rollout-turns` preserves
the exact terminal search.

On the same seed and R600 budget, immediate afterstate bootstrapping reduced
treatment search from 137.910 seconds to 13.014 seconds:

| Metric | Full terminal | Two-turn MLX leaf |
|---|---:|---:|
| Treatment seconds per game | 137.910 | 13.014 |
| Rollout waves | 3,800 | 380 |
| Neural rows | 6,135,934 | 644,040 |
| Effective speedup | 1.00x | **10.60x** |

The 30-game distributed confirmation used fresh raw seeds 34200-34229, ten
games on each local Mac:

| Metric | Result |
|---|---:|
| Mean score | 93.775 |
| Paired control mean | 92.342 |
| Mean paired delta | +1.433 |
| 95% paired CI | `[+0.720,+2.147]` |
| Mean treatment seconds per game | 12.237 |
| Speedup versus current full terminal baseline | **11.27x** |
| Three-node wall time | 161 s |
| Effective cluster throughput | about 671 games/hour |

This tier is intentionally weaker and not promotion-equivalent. On four directly matched
seeds, full terminal search averaged 96.875 while immediate-afterstate
screening averaged 93.188, a -3.688 point gap. Waiting for the opponent round
averaged 93.000 on the same seeds and did not recover the loss.

The runner also supports `--leaf-model-dir` so root priors and rollout actions
can keep using the qualified parent while a second MLX model evaluates only
the bootstrap leaf. The existing rollout-return model regressed badly in that
role. The joint return/ranking model improved the three-seed shallow result by
only 0.333 points and still trailed full search by 4.250 points. Neither is a
qualified screening leaf.

Use the two-turn tier to reject weak ideas quickly. Any candidate that survives
must be rerun without `--rollout-turns` on common seeds before a strength or
promotion claim. Reports record the rollout limit, leaf timing, bootstrap
count, both model manifests when present, and all ordinary provenance.

## Experiment Velocity

R600 remains the promotion budget, but it should not be the first budget used
for every idea. The local three-node cluster should use a progressive funnel:

| Stage | Budget | Purpose | Approximate three-node throughput |
|---|---:|---|---:|
| Integrity smoke | R32, 1-3 paired games | Reject crashes, drift, fallback, and malformed output | minutes |
| Directional screen | R100, 12-30 paired games | Reject clearly negative ideas on common seeds | about 470 games/hour |
| Promotion pilot | R300, 20-50 paired games | Check category tradeoffs and effect stability | about 155 games/hour |
| Confirmation | R600, preregistered sample | Estimate the final effect and promotion decision | about 255 paired games/hour |

The R600 throughput uses the measured 705.655-second wall time for 50 complete
paired reports across three Macs. Lower-budget estimates remain capacity
planning numbers, not strength claims. A candidate advances only when it keeps
the same sign and acceptable wildlife/habitat/token behavior at the next
budget. Final claims remain full-terminal R600.

The two-turn R600 screening tier is an additional, orthogonal first pass. It
keeps the R600 allocation schedule but substitutes an MLX leaf after two focal
moves. Its measured 671-game/hour cluster throughput is appropriate for broad
hypothesis rejection; the R32/R100/R300 funnel remains useful when terminal
rollout semantics themselves are under study.

This changes the expected cost of a failed idea substantially: a 30-game R100
screen takes roughly four minutes of cluster wall time instead of roughly
23 minutes at R600. Paired common seeds should be retained at every stage.
R1200 should not be used routinely: ADR 0061 found only +0.167 points over R600
while increasing runtime from 151.19 to 311.43 seconds per game.

## Packed CSR Pipeline Boundary

An exact transport experiment replaced each pipelined inference request's
`Vec<Vec<u16>>` rows with one CSR offset vector and one contiguous feature
vector. This removed row-vector clones at the pipeline boundary and allowed
the model client to copy the already-packed arrays directly into shared
memory.

Every run reproduced scores `[102,96,92,95]`, 3,920 neural batches,
6,121,807 logical rows, 5,062,305 physical rows, 3,716 rollout waves, 46,207
samples, and zero fallbacks. End-to-end measurements rejected the change:

| Host | Existing rows | Packed CSR | Packed result |
|---|---:|---:|---:|
| john2 | 15.679491 s | 16.216391 s | 3.424% slower |
| john3 | 16.005119 s | 16.257209 s | 1.575% slower |
| Combined | **15.842305 s** | **16.236800 s** | **2.490% slower** |

The packed representation saved allocation bookkeeping but added a complete
feature copy during materialization. The existing row form transfers ownership
of already-built feature vectors and performs the unavoidable contiguous copy
only once, directly into the MLX shared-memory mapping. The packed API and its
tests were removed. Full evidence is in
`docs/v2/reports/exact-packed-csr-pipeline-rejection-v1.json`.

## Resolved Habitat Preview

An exact in-scan experiment resolved occupied-neighbor union-find roots once
per frontier cell and reused them across all six dual-terrain rotations. It
preserved every original scan and replay boundary and passed all preview,
top-eight, complete-game, score, and search-diagnostic parity checks.

The complete treatment was nevertheless 0.231% slower across balanced john2
and john3 measurements. At Cascadia board depths, the accepted root chains are
too shallow for a second neighbor representation to pay for itself. The
experiment was rejected and removed. Evidence is in
`docs/v2/reports/exact-resolved-habitat-preview-rejection-v1.json`.

## Incremental Top-Eight Rejection

Maintaining the qualified greedy top-eight habitat placements in sorted order
during traversal was exact and improved matched non-PGO release builds by
0.549% across john2 and john3. It nevertheless regressed the production
profile-guided build by 0.600% in a crossed two-host comparison and was
removed.

The PGO audit also established that LLVM's default instrumentation counters
must not be collected concurrently by the Rayon hot path. Ten-worker
collection lost updates and produced inconsistent profiles. Race-free
collection with `RAYON_NUM_THREADS=1` produced per-host total counts that
differed by only 1,436 out of 119.47 billion; the resulting binary was used
for the final rejection decision.

Full evidence:
[`exact-incremental-top8-rejection-v1.md`](reports/exact-incremental-top8-rejection-v1.md).

## Adjacency Pair Feature Rejection

Replacing the NNUE wildlife-pair and terrain-pair coordinate conversions with
precomputed adjacency-table reads was exact across 12 complete games and every
frozen R600 diagnostic. It improved matched non-PGO binaries by 0.802%
combined, enough to trigger the preregistered PGO gate, but the host-level
source signs disagreed.

Race-free profiling with one Rayon worker per host produced near-identical
profiles. The resulting production PGO binary regressed by 0.065% on john2
and 0.488% on john3, or 0.275% combined, against the accepted elk-potential
PGO champion. The treatment and its test oracle were removed.

Full evidence:
[`exact-adjacency-pair-features-rejection-v1.md`](reports/exact-adjacency-pair-features-rejection-v1.md).

## Mid-V4 Bag Feature Template Rejection

Precomputing the 43-55 bag- and opponent-dependent sparse feature indices
shared by every candidate afterstate was byte-exact across eight complete
games, all 640 decisions, and the frozen R600 diagnostic.

The crossed non-PGO result was only 0.119% faster on john2 and 0.004% faster
on john3, or 0.062% combined. That failed the preregistered 0.25% advancement
floor, so no fresh PGO build was authorized. The specialized template path
and its test oracle were removed.

Full evidence:
[`exact-mid-v4-bag-feature-template-rejection-v1.md`](reports/exact-mid-v4-bag-feature-template-rejection-v1.md).

## Local Elk Extension Delta Rejection

For non-elk moves, an exact treatment updated Card-A elk potential from only
the newly placed tile and a distinct newly occupied wildlife cell. It matched
complete recomputation across six full games, every exercised afterstate, both
85-test library configurations, and the frozen R600 diagnostic.

The crossed non-PGO source screen regressed by 0.503% on john2 and improved by
only 0.032% on john3. Combined wall time increased from 15.897619 to 15.934611
seconds, a 0.232% regression. The treatment failed its 0.25% advancement gate,
so no PGO build was authorized and the implementation was removed.

Full evidence:
[`exact-local-elk-extension-delta-rejection-v1.md`](reports/exact-local-elk-extension-delta-rejection-v1.md).

## Shared Tile Potential Rejection

An exact Nature Token specialization prepared tile-only potential once when
several candidate combinations shared a market tile and frontier coordinate.
It matched direct incremental potential, complete recomputation, candidate
sets, and every frozen search diagnostic across six complete games.

The crossed non-PGO screen improved john2 by 0.971% and john3 by 1.520%, or
1.244% combined, so it advanced to fresh race-free PGO. The production result
did not hold:

| Host | Accepted PGO | Treatment PGO | Treatment result |
|---|---:|---:|---:|
| john2 | 15.369275 s | 15.382703 s | 0.087% slower |
| john3 | 15.095387 s | 15.025399 s | 0.464% faster |
| Combined | **15.232331 s** | **15.204051 s** | **0.186% faster** |

The treatment failed the preregistered cross-host reproducibility rule and
was removed. A post-removal rebuild has a byte-identical executable text
section to the retained pre-experiment control.

Full evidence:
[`exact-shared-tile-potential-rejection-v1.md`](reports/exact-shared-tile-potential-rejection-v1.md).

## Parent Afterstate Feature Context Acceptance

The qualified `mid-features,v4-opp` path now constructs one exact parent
feature context per rollout policy state. Candidate afterstates reuse
invariant cell, slot, bag, market, opponent, and parent-pattern state and
recompute only blocks affected by the candidate tile and wildlife placement.
The general extractor remains the independent debug/test oracle.

The ordered sparse row matched byte for byte across eight complete AAAAA
games and all 640 turns, including ordinary and independent drafts, same-cell
and different-cell wildlife placement, keystones, no-wildlife moves, and all
six rotations. Both complete library configurations and every frozen R600
diagnostic passed.

Candidate preparation fell by 18.039% on john2 and 17.512% on john3. Matched
non-PGO binaries improved by 2.783% and 2.243% respectively, or 2.515%
combined, without an aggregate peak-RSS regression.

Fresh race-free PGO used one `RAYON_NUM_THREADS=1` R600 profile from each
worker. The resulting production binary remained faster than the accepted
elk-potential PGO champion on both machines:

| Host | Elk PGO | Parent context PGO | Improvement |
|---|---:|---:|---:|
| john2 | 15.333361 s | 15.211514 s | 0.795% |
| john3 | 15.146230 s | 14.826227 s | 2.113% |
| Combined | **15.239795 s** | **15.018871 s** | **1.450%** |

The accepted path is now 9.390x faster than the 141.027296-second frozen
reference. Reaching the 14.102730-second 10x threshold requires another
0.916141 seconds, or 6.100%.

Full evidence:
[`exact-parent-afterstate-feature-context-acceptance-v1.md`](reports/exact-parent-afterstate-feature-context-acceptance-v1.md).

## Opponent Greedy Decision Reuse Rejection

An exact reuse audit tested whether synchronized rollout states frequently
presented the same public decision to a greedy opponent. Across 1,390,050
opponent move requests, only 7,114 were duplicates of another exact
`CandidateCacheKey`: a 0.512% reuse rate.

Even with free grouping and perfectly uniform request cost, that can remove
only about 16 ms from the accepted 3,156.583 ms opponent-advance stage. The
temporary synchronized diagnostic path actually added about 2.55 seconds.
The production deduplication was therefore rejected before implementation,
and all diagnostic code was removed.

Full evidence:
[`exact-opponent-greedy-decision-reuse-rejection-v1.md`](reports/exact-opponent-greedy-decision-reuse-rejection-v1.md).

## Exact Card A Wildlife Score Context Rejection

An immutable Card A context replaced complete wildlife-category rescoring with
exact placement-local updates. It matched the independent scorer for every
legal existing and new-tile wildlife slot across 16 complete AAAAA games,
including the target category and Fox side effects, and preserved every frozen
R600 score and diagnostic.

Crossed non-PGO binaries improved on both workers, but not enough to advance:

| Host | Parent context source | Card A context source | Improvement |
|---|---:|---:|---:|
| john2 | 15.207208 s | 15.169689 s | 0.247% |
| john3 | 15.067608 s | 14.931556 s | 0.903% |
| Combined | **15.137408 s** | **15.050622 s** | **0.573%** |

The result missed the preregistered 1.00% floor. Candidate preparation also
regressed 0.213% on john2, and allocator peak footprint rose 9.162% combined.
The treatment therefore stopped before PGO and was removed. A post-removal
release build is byte-for-byte identical to the retained parent-context source
control.

Full evidence:
[`exact-card-a-wildlife-score-context-rejection-v1.md`](reports/exact-card-a-wildlife-score-context-rejection-v1.md).

## Exact MLX Row Locality Order Rejection

Lexicographically ordering already-deduplicated sparse rows increased
largest-request adjacent trie reduction from 4.934% to 35.615% without
changing a feature, prediction, action, score, or search diagnostic.

The intended device-cache benefit did not survive host bookkeeping. MLX
evaluation was flat on john2 and 1.258% faster on john3, while Rust-side
neural evaluation regressed 11.606% and 10.442% respectively. The crossed
source result was decisively negative:

| Host | Canonical row order | Locality row order | Treatment result |
|---|---:|---:|---:|
| john2 | 15.216156 s | 16.233674 s | 6.687% slower |
| john3 | 14.993586 s | 16.052421 s | 7.062% slower |
| Combined | **15.104871 s** | **16.143048 s** | **6.873% slower** |

The experiment stopped before PGO and was removed. The post-removal release
is byte-for-byte identical to the retained parent-context source control.

Full evidence:
[`exact-mlx-row-locality-order-rejection-v1.md`](reports/exact-mlx-row-locality-order-rejection-v1.md).

## Exact Direct Rollout Template Preparation Acceptance

The qualified pipelined rollout path previously constructed and hash-grouped
complete public-state keys, then reconstructed the same key inside a
thread-local candidate cache. A full audit observed only 12 reusable template
requests among 440,239 requests, or 0.002726%.

The accepted path directly builds and consumes one uncached candidate template
per active rollout state. Scalar and synchronous callers retain their cache;
the qualified production path has no experiment switch or obsolete grouping
branch. Complete default, `mid-features,v4-opp`, Python exact-service, and
pipelined parity suites passed.

Combined key/template/candidate preparation fell 11.752% on john2 and 11.656%
on john3. The crossed source screen improved 3.620% combined. Fresh race-free
PGO used one `RAYON_NUM_THREADS=1` R600 profile per host and remained faster
than the accepted parent-context PGO binary on both machines:

| Host | Parent-context PGO | Direct-template PGO | Improvement |
|---|---:|---:|---:|
| john2 | 15.031477 s | 14.499902 s | 3.536% |
| john3 | 14.869270 s | 14.166401 s | 4.727% |
| Combined | **14.950374 s** | **14.333151 s** | **4.128%** |

Maximum RSS was flat and mean peak footprint fell 8.879%. The accepted path
is now 9.839x faster than the 141.027296-second reference. Reaching the
14.102730-second 10x threshold requires another 0.230422 seconds, or 1.608%.

Full evidence:
[`exact-direct-rollout-template-preparation-acceptance-v1.md`](reports/exact-direct-rollout-template-preparation-acceptance-v1.md).

## Exact MLX Pipeline Cohort Size Rejection

A fixed two-host sweep tested exact inference cohorts of 128, 160, 192, and
256 rollout states against the accepted 96-state default. Every run preserved
the complete frozen score and search vector.

Larger cohorts nearly halved MLX request count and reduced isolated decode,
graph-build, and device-evaluation sums, but weakened producer/consumer overlap
and increased working-set size:

| Cohort states | Combined time | Result vs 96 |
|---:|---:|---:|
| 96 | **14.260874 s** | control |
| 128 | 14.988507 s | 5.102% slower |
| 160 | 14.950687 s | 4.837% slower |
| 192 | 14.944689 s | 4.795% slower |
| 256 | 14.943135 s | 4.784% slower |

Mean maximum RSS rose from 115.0 MB at 96 to 150.7 MB at 128 and 191.3 MB at
256. Every treatment regressed both hosts, so no confirmation or fresh PGO
was authorized and the 96-state production default remains unchanged.

Full evidence:
[`exact-mlx-pipeline-cohort-size-rejection-v1.md`](reports/exact-mlx-pipeline-cohort-size-rejection-v1.md).

## Exact Bounded Pipeline State Slices Acceptance

Every 96-state exact pipeline cohort previously allocated a Boolean mask over
the complete rollout-state population and scanned all states during
preparation. Application repeated the pattern with a full-population position
map before opponent advancement.

Because active indices are strictly increasing, each cohort is exactly
represented by the half-open range from its first through last active state;
any gap is already terminal. The accepted path prepares unfinished states and
advances opponents only inside that bounded mutable slice. Candidate order,
sparse rows, predictions, actions, random streams, and diagnostics are
unchanged.

Matched source diagnostics show that the combined targeted stages fell on
both workers:

| Host | Full-population path | Bounded-slice path | Reduction |
|---|---:|---:|---:|
| john2 | 7,899.706260 ms | 7,827.403737 ms | 0.915% |
| john3 | 7,990.618453 ms | 7,838.457042 ms | 1.904% |

The crossed non-PGO source screen improved 1.004% combined. Fresh PGO used one
complete race-free profile from each worker and remained faster on both:

| Host | Direct-template PGO | Bounded-slice PGO | Improvement |
|---|---:|---:|---:|
| john2 | 14.531643 s | 14.306317 s | 1.551% |
| john3 | 14.128995 s | 14.019792 s | 0.773% |
| Combined | **14.330319 s** | **14.163055 s** | **1.167%** |

Mean maximum RSS fell 0.099%. The allocator footprint high-water mean rose
5.180%, or 3.29 MB, while the maximum treatment observation exceeded the
maximum control by only 1.52 MB. No shutdown or operational regression was
observed.

The accepted path is now 9.957x faster than the 141.027296-second reference.
The mandatory 10x threshold is 14.102730 seconds, leaving 0.060325 seconds or
0.426% to remove.

Full evidence:
[`exact-bounded-pipeline-state-slices-acceptance-v1.md`](reports/exact-bounded-pipeline-state-slices-acceptance-v1.md).

## Exact Shared CSR Validation Ownership Rejection

The private shared-memory NNUE path validated every request in Rust, encoded
canonical CSR offsets and features, then repeated offset, width, and
feature-range scans in Python. A treatment retained all producer validation
and memory-safety bounds while omitting only those duplicate child scans.

The mechanism worked in isolation:

| Host | Control decode/validate | Treatment decode/validate | Reduction |
|---|---:|---:|---:|
| john2 | 313.867380 ms | 139.295181 ms | 55.620% |
| john3 | 322.401333 ms | 138.482135 ms | 57.047% |

The saving did not survive end-to-end scheduling. The accepted bounded-slice
PGO Rust binary was crossed with the treatment-capable service in opposite
balanced orders:

| Host | Control | Treatment | Treatment result |
|---|---:|---:|---:|
| john2 | 14.384470 s | 14.324836 s | 0.415% faster |
| john3 | 13.977943 s | 14.202196 s | 1.604% slower |
| Combined | **14.181207 s** | **14.263516 s** | **0.580% slower** |

Every score and search diagnostic remained exact, and memory fell, but the
treatment failed the both-host and combined wall-time gates. It was removed
in full. The accepted 14.163055-second, 9.957x Phase 0 baseline is unchanged.

Full evidence:
[`exact-shared-csr-validation-ownership-rejection-v1.md`](reports/exact-shared-csr-validation-ownership-rejection-v1.md).

## Exact Market Wildlife Scan Filter Rejection

The qualified greedy evaluator computed the best existing placement for all
five wildlife categories even when the current market and every legal
independent draft could reach only a subset. The exact treatment retained
category order and skipped only unreachable categories.

Template-preparation time fell 0.312% on john2 and 0.772% on john3. The crossed
source screen improved 0.648% combined, but john3 regressed 0.102% and mean
allocator peak footprint rose 4.216%. The treatment failed the registered
both-host and memory gates, so it was removed without PGO.

Full evidence:
[`exact-market-wildlife-scan-filter-rejection-v1.md`](reports/exact-market-wildlife-scan-filter-rejection-v1.md).

## Exact Greedy Placed-Tile Snapshot Elision Rejection

The rollout-opponent greedy evaluator cloned its insertion-ordered
`placed_tiles` vector on each of 1,390,050 qualified requests. An exact
treatment copied each indexed element into a scalar before hypothetical
wildlife scoring, preserving category order, tile order, mutations, and ties.

Opponent-advance time fell on both workers:

| Host | Snapshot control | Indexed treatment | Reduction |
|---|---:|---:|---:|
| john2 | 3,400.107 ms | 3,373.852 ms | 0.772% |
| john3 | 3,417.880 ms | 3,410.115 ms | 0.227% |

The crossed source result did not meet the end-to-end gate:

| Host | Control | Treatment | Treatment result |
|---|---:|---:|---:|
| john2 | 14.532296 s | 14.465929 s | 0.457% faster |
| john3 | 14.295060 s | 14.342268 s | 0.330% slower |
| Combined | **14.413678 s** | **14.404098 s** | **0.066% faster** |

Mean allocator peak footprint rose 4.021%. The registered gate required both
hosts positive and more than 0.25% combined improvement, so the indexed path,
switch, second release monomorphization, and temporary oracle were removed
before PGO. The 14.163055-second accepted baseline is unchanged.

Full evidence:
[`exact-greedy-placed-tile-snapshot-elision-rejection-v1.md`](reports/exact-greedy-placed-tile-snapshot-elision-rejection-v1.md).

## Exact Candidate Placement Allocation Elision Rejection

The candidate-placement ranking loop knows every vector's final length and
contains many equal habitat scores. A combined treatment reserved exact
capacity and replaced the stable sort with an unstable sort whose explicit
frontier-position and rotation keys reproduced the original total order.

Every candidate, score, sparse row, and frozen diagnostic remained exact, but
the additional tie comparisons dominated the removed allocation work:

| Host | Stable control | Reserve + total-order treatment | Regression |
|---|---:|---:|---:|
| john2 template preparation | 4,538.554 ms | 4,662.612 ms | 2.733% |
| john3 template preparation | 4,510.818 ms | 4,670.930 ms | 3.550% |

Retired instructions rose 1.746% on john2 and 1.780% on john3. The
preregistered mechanism gate failed on both machines, so the unstable sort and
combined switch were removed before a formal source screen or PGO.

Full evidence:
[`exact-candidate-placement-allocation-elision-rejection-v1.md`](reports/exact-candidate-placement-allocation-elision-rejection-v1.md).

## Exact Candidate Placement Capacity Reservation Rejection

The successor retained the original stable sort and changed only vector
capacity. Calling `reserve_exact` once reduced template preparation and
retired instructions on both workers:

| Host | Geometric-growth control | Exact reservation | Reduction |
|---|---:|---:|---:|
| john2 template preparation | 4,457.858 ms | 4,415.140 ms | 0.958% |
| john3 template preparation | 4,530.611 ms | 4,442.075 ms | 1.954% |

That local saving did not survive the complete pipeline:

| Host | Control | Treatment | Treatment result |
|---|---:|---:|---:|
| john2 | 14.513657 s | 14.650299 s | 0.941% slower |
| john3 | 14.415170 s | 14.346497 s | 0.476% faster |
| Combined | **14.464414 s** | **14.498398 s** | **0.235% slower** |

All twelve complete diagnostic and timed games reproduced the frozen vector.
Mean maximum RSS rose 0.060% and mean allocator peak footprint rose 2.990%.
The registered both-host and combined-gain gates failed, so reservation, its
switch, the second release monomorphization, and its oracle were removed
without PGO. The 14.163055-second accepted baseline remains unchanged.

Full evidence:
[`exact-candidate-placement-capacity-reservation-rejection-v1.md`](reports/exact-candidate-placement-capacity-reservation-rejection-v1.md).

## Exact Persistent Evaluator Worker Rejection

The qualified pipeline spawned one evaluator thread and allocated one
capacity-one channel pair for every sequential-halving rollout batch. The
treatment retained one worker and channel pair across all rounds in a search
without changing request contents, ordering, cohort size, rollout allocation,
or random streams.

Native samples confirmed the mechanism: sampled worker identities fell from
104 to 21 on john2 and from 107 to 21 on john3, reductions of 79.8% and 80.4%.
The crossed non-PGO source screen improved both hosts:

| Host | Per-batch worker | Per-search worker | Improvement |
|---|---:|---:|---:|
| john2 | 14.631387 s | 14.515858 s | 0.790% |
| john3 | 14.375427 s | 14.324509 s | 0.354% |
| Combined | **14.503407 s** | **14.420184 s** | **0.574%** |

The treatment was therefore productionized and rebuilt from two fresh
race-free R600 profiles. The final crossed PGO result failed the mandatory
both-host and absolute-threshold gates:

| Host | Accepted PGO | Persistent-worker PGO | Result |
|---|---:|---:|---:|
| john2 | 14.458092 s | 14.228214 s | 1.590% faster |
| john3 | 13.997023 s | 14.096939 s | 0.714% slower |
| Combined | **14.227557 s** | **14.162576 s** | **0.457% faster** |

The combined treatment remained 59.847 ms above the 14.102730-second 10x
threshold. Mean maximum RSS rose only 0.076%, and allocator peak footprint
fell 2.571%, so rejection is driven by timing alone. The complete treatment,
switch, and experiment-only tests were removed. The accepted
14.16305453125-second bounded-slice PGO baseline remains unchanged.

Full evidence:
[`exact-persistent-evaluator-worker-rejection-v1.md`](reports/exact-persistent-evaluator-worker-rejection-v1.md).

## Exact Candidate Placement Metadata Packing Rejection

The hottest exact template path stored duplicate axial coordinates in every
candidate-placement sort record. A lossless treatment packed board index and
rotation into one `u16`, retained habitat score in another, and delayed
coordinate reconstruction. The transient record fell from eight bytes to
four without changing vector growth, stable sorting, ties, truncation, or
candidate order.

The intended stage improved on both workers:

| Host | Expanded record | Packed record | Reduction |
|---|---:|---:|---:|
| john2 template preparation | 4,550.043 ms | 4,468.523 ms | 1.792% |
| john3 template preparation | 4,557.436 ms | 4,508.458 ms | 1.075% |

Retired instructions fell 0.0355% on john2 and 0.0268% on john3. All four
mechanism runs and eight formal source runs preserved the complete frozen
score/search vector.

The balanced source screen rejected the treatment:

| Host | Expanded control | Packed treatment | Result |
|---|---:|---:|---:|
| john2 | 14.653264 s | 14.745721 s | 0.631% slower |
| john3 | 14.457369 s | 14.474092 s | 0.116% slower |
| Combined | **14.555317 s** | **14.609906 s** | **0.375% slower** |

Mean maximum RSS fell 0.127% and allocator peak footprint fell 4.622%, but
both-host and combined wall-time gates failed. The packed representation,
switch, dual monomorphization, and temporary oracle were removed without PGO.
The accepted 14.16305453125-second bounded-slice PGO baseline is unchanged.

Full evidence:
[`exact-candidate-placement-metadata-packing-rejection-v1.md`](reports/exact-candidate-placement-metadata-packing-rejection-v1.md).

## Exact Dead Local Outcome Buffer Elision Acceptance

The shared production candidate path allocated and initialized a local
rotation-invariant outcome vector before unconditionally selecting a separate
shared cache. Optimized IR confirmed that the dead allocation survived full
optimization in the accepted binary.

The treatment retains the local vector only for the non-sharing test oracle.
Template preparation and retired instructions fell on both workers:

| Host | Eager local buffer | Elided local buffer | Reduction |
|---|---:|---:|---:|
| john2 template preparation | 4,542.286 ms | 4,452.052 ms | 1.987% |
| john3 template preparation | 4,497.884 ms | 4,476.286 ms | 0.480% |
| john2 retired instructions | 1,087,604,873,751 | 1,082,900,943,846 | 0.433% |
| john3 retired instructions | 1,088,150,435,236 | 1,082,970,687,422 | 0.476% |

The balanced non-PGO source screen passed:

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 14.802802 s | 14.541419 s | 1.766% |
| john3 | 14.466847 s | 14.387534 s | 0.548% |
| Combined | **14.634825 s** | **14.464476 s** | **1.164%** |

Production was rebuilt without the experiment switch from one complete exact
R600 profile per worker. The fresh PGO binary remained faster on both hosts:

| Host | Previous accepted PGO | New production PGO | Improvement |
|---|---:|---:|---:|
| john2 | 14.390736 s | 14.211937 s | 1.242% |
| john3 | 14.090791 s | 13.980756 s | 0.781% |
| Combined | **14.240764 s** | **14.096347 s** | **1.014%** |

Every run reproduced scores `[102,96,92,95]`, 3,920 neural batches,
6,121,807 logical rows, 5,062,305 physical rows, 3,716 rollout waves, 46,207
samples, zero bootstraps, zero fallbacks, and clean shutdown. Mean RSS fell
0.107% and allocator peak footprint fell 4.026% in the PGO comparison.

The accepted path is `10.004528179689x` faster than the frozen
141.027296-second reference. Its `14.096346521`-second crossed mean is
`0.006383079` seconds inside the mandatory 10x threshold. Phase 0 is complete.

Full evidence:
[`exact-dead-local-outcome-buffer-elision-acceptance-v1.md`](reports/exact-dead-local-outcome-buffer-elision-acceptance-v1.md).

## Full-Legal Audit Teacher Performance

The frozen Phase 1 audit reference is 242.433050 seconds for seed 60999 at
completed turns 12, 39, and 66 with complete legal screening, R1200/R4800
confirmation, paid-wipe chance diagnostics, and realized-hidden terminal
continuations. Its independent 10x threshold is 24.243305 seconds.

Five exact optimizations are now production:

| Production step | Complete wall | Improvement vs parent |
|---|---:|---:|
| Frozen reference | 242.433050 s | - |
| 4,096-row static-screen cohorts | 217.302694 s | 10.366% |
| Exact static sparse-row deduplication | 212.191376 s | 2.352% |
| Exact paid-screen public-state cache | 177.686057 s | 16.261% |
| Exact game-scoped public-decision cache | 162.045309 s | 8.802% |
| Multiplexed realized-hidden trajectories | **143.775461 s** | **11.275%** |

The paid-screen cache reuses only collision-checked complete public states.
Its compact representation interns invariant board contexts and stores exact
markets per entry. On the frozen turn-16 screen it evaluates 402 distinct
states for 842 requests, a 52.257% reduction. Opposite-order confirmation
improved john2 36.979%, john3 34.925%, and the combined mean 35.966%.

The later game-scoped decision cache records 118 exact hits across 1,040
public policy requests while retaining only the 80 states on the reusable
champion-root continuation. It reduces policy evaluations 11.346%, logical
rows 9.813%, physical rows 10.011%, and rollout samples 11.041%. Mirrored
confirmation improved john2 8.592%, john3 8.559%, and the combined mean
8.576%.

The realized-hidden scheduler advances each eight-search finalist set in
deterministic lockstep, combines stable sparse requests through one MLX
evaluator, and preserves independent K32/R600 state. It coordinates 907 exact
searches across 120 cohorts, coalesces 99.974% of evaluator batches, and
reduces physical service calls by 86.953%. Mirrored confirmation improves
john2 16.273%, john3 14.494%, and combined wall time 15.391%.

The complete audit remains semantically identical after non-semantic
diagnostics are removed. Switch-free production improves 11.275% over the
decision-cache parent. Maximum RSS rises to a bounded 907,378,688 bytes while
allocator peak changes only +1.821%; both remain below the 1.5 GiB gate. The
accepted teacher is now 1.686192x faster than the reference, with a remaining
5.931x gap to its mandatory threshold. Realized-hidden terminal continuations
still dominate at 103.119 of 143.775 seconds.

Full evidence:
[`full-legal-audit-multiplexed-trajectory-search-acceptance-v1.md`](reports/full-legal-audit-multiplexed-trajectory-search-acceptance-v1.md).

## Cross-Wave Sparse-Row Reuse Rejection

An exact diagnostic retained complete sparse rows within each independent
K32/R600 search and required full equality after fingerprint lookup. Across
the frozen late-turn screen it observed 6,116,501 physical rows spanning
4,355 rollout waves and 104 searches.

Exact rows repeated in a later wave or halving round: **zero**.

The report reproduced the frozen turn-66 semantic digest, exact scores and
logical diagnostics, zero fallbacks, and zero swaps. Diagnostic RSS remained
below 1.5 GiB. The 0.000% reuse rate fails the preregistered 5% gate, so no
full-contract run and no per-search prediction cache are authorized.

This result does not cover duplicate rows between different requests already
coalesced into one multiplexed evaluator batch.

Full evidence:
[`full-legal-audit-cross-wave-row-reuse-diagnostic-rejection-v1.md`](reports/full-legal-audit-cross-wave-row-reuse-diagnostic-rejection-v1.md).

## Cross-Request Sparse-Row Reuse Rejection

The second bounded diagnostic examined complete sparse rows from different
search requests already combined into one evaluator batch. Across all 189
late-turn batches it observed 891,486 rows and found only 247 exact
cross-request duplicates.

The `0.027707%` reuse rate is one row per 3,609. It misses the 50,000-row and
5% advance gates by orders of magnitude. The report remained exact,
zero-swap, and effectively timing-neutral, so no global deduplication or
prediction-scatter treatment is authorized.

Full evidence:
[`full-legal-audit-cross-request-row-reuse-diagnostic-rejection-v1.md`](reports/full-legal-audit-cross-request-row-reuse-diagnostic-rejection-v1.md).

## Multiplexed Stage Profile

The frozen profile ran on john1, john2, and john3 with exact stage counters in
Rust and service-wall counters in the MLX process. All reports retained the
frozen semantic BLAKE3, exact work vector, zero fallbacks, zero process swaps,
and sub-1.5-GiB RSS.

| Host | Complete wall | Hidden wall | MLX eval | Non-eval MLX |
|---|---:|---:|---:|---:|
| john1 | 145.693 s | 104.623 s | 49.408 s | 2.802 s |
| john2 | 128.645 s | 90.289 s | 47.982 s | 2.435 s |
| john3 | 128.943 s | 90.793 s | 47.644 s | 2.502 s |
| john2/john3 mean | **128.794 s** | **90.541 s** | **47.813 s** | **2.469 s** |

MLX evaluation is 37.123% of remote wall and has a 1.590x
perfect-elimination Amdahl ceiling. More than 1,024 rows account for 96.737%
of all rows and 92.954% of evaluation time. The largest request contains
10,148 rows; H1 is 65.331% positive and H2 49.828% positive.

The cumulative Rust interval total averages 862.938 seconds remotely because
independent searches and the evaluator intentionally overlap. Within that
total, neural wait is 56.624%, opponent advancement 21.258%, and
rollout-template preparation 18.867%. Those counters rank CPU capacity but
must not be subtracted directly from wall.

The next treatment therefore targets the exact H1 Metal execution geometry
without repeating rejected host sorting or prefix planners. Opponent
advancement and template preparation remain the next measured CPU targets.

Full evidence:
[`full-legal-audit-multiplexed-stage-profile-v1.md`](reports/full-legal-audit-multiplexed-stage-profile-v1.md).

## Exact H1 Vector-Width Rejection

The first MLX treatment kept every output's feature-addition order exact but
packed two or four `float4` accumulators into each thread. A three-node
Latin-square screen rejected both wider geometries:

| Geometry | Mean complete wall | Wall change | Mean MLX eval | MLX change |
|---|---:|---:|---:|---:|
| Width 4 control | 30.370 s | - | 10,903 ms | - |
| Width 8 | 30.783 s | **+1.360%** | 11,418 ms | **+4.720%** |
| Width 16 | 31.796 s | **+4.696%** | 12,562 ms | **+15.210%** |

Both treatments regressed on every Mac. Additional live accumulators reduced
effective occupancy more than fewer threads and index-loop iterations helped.
All nine reports remained exact, zero-swap, and below 558 MB RSS.

The two kernels and selector were removed. The Python evaluator and rebuilt
audit binary returned byte-for-byte to the stage-profile versions.

Full evidence:
[`exact-mlx-h1-vector-width-rejection-v1.md`](reports/exact-mlx-h1-vector-width-rejection-v1.md).

## Exact H1 SIMD Index Broadcast Rejection

The second H1 treatment retained width 4 and had only SIMD lane zero load each
CSR offset and feature index before `simd_broadcast_first` shared it with the
other 31 lanes. Direct H1 tensors and final outputs remained bit exact.

| Host | Control wall | Treatment wall | Wall change | Control MLX | Treatment MLX | MLX change |
|---|---:|---:|---:|---:|---:|---:|
| john1 | 31.704 s | 32.018 s | +0.989% | 10,574 ms | 10,704 ms | +1.235% |
| john2 | 29.650 s | 29.418 s | -0.780% | 11,088 ms | 10,993 ms | -0.859% |
| john3 | 29.327 s | 29.081 s | -0.838% | 10,807 ms | 10,541 ms | -2.463% |
| Mean | **30.227 s** | **30.172 s** | **-0.180%** | **10,823 ms** | **10,746 ms** | **-0.711%** |

The 0.180% wall and 0.711% MLX gains missed the 1% and 3% advance gates.
Repeated integer index loads are therefore not a material H1 bottleneck on
these GPUs; hardware coalescing or caching already handles them cheaply
relative to first-layer weight traffic. The alternate kernel and selector
were removed, restoring the evaluator and binary byte for byte.

Two different H1 geometry treatments are now closed. The next MLX diagnostic
will time exact H1, H2, and output execution separately before selecting an
intermediate-tensor or kernel-fusion target.

Full evidence:
[`exact-mlx-h1-simd-index-broadcast-rejection-v1.md`](reports/exact-mlx-h1-simd-index-broadcast-rejection-v1.md).

## Exact MLX Layer Profile

The diagnostic-only layer profile completed on all three Macs and was then
removed. H1 ranked first everywhere:

| Host | H1 share | H2 share | Output share |
|---|---:|---:|---:|
| john1 | 78.900% | 14.855% | 6.245% |
| john2 | 76.410% | 16.773% | 6.816% |
| john3 | 76.491% | 16.783% | 6.726% |

john2 and john3 differ by only 0.081 percentage point on H1 share. Every exact
request and row was covered, all reports preserved the frozen semantic digest
and work vector, every process used zero swap, and maximum RSS remained below
1.12 GB.

The complete audit materializes 91,963,295,744 H1 bytes per write or read:
183,926,591,488 bytes across the H1 global write and H2 global read. The next
preregistered treatment fuses H1 and H2 within a Metal threadgroup while
preserving every H1 feature addition and H2 input accumulation in order.

The diagnostic selector and synchronization boundaries were removed. The
service and audit binary returned byte for byte to the accepted production
hashes.

Full evidence:
[`exact-mlx-layer-profile-v1.md`](reports/exact-mlx-layer-profile-v1.md).

## Exact H1-H2 Scalar Threadgroup Fusion Rejection

The first fusion kept two exact H1 rows in 4 KiB of threadgroup memory,
removed the global H1 write/read boundary, and computed H2 after one barrier.
It preserved every output bit but changed H2 from the production
eight-thread-per-row `float8` mapping to 64 scalar threads per row.

| Host | Control wall | Treatment wall | Wall change | Control MLX | Treatment MLX | MLX change |
|---|---:|---:|---:|---:|---:|---:|
| john1 | 31.695 s | 32.161 s | +1.472% | 10,581 ms | 11,103 ms | +4.936% |
| john2 | 29.675 s | 29.894 s | +0.738% | 11,156 ms | 11,577 ms | +3.779% |
| john3 | 29.165 s | 29.587 s | +1.446% | 10,731 ms | 11,149 ms | +3.891% |
| Mean | **30.178 s** | **30.547 s** | **+1.223%** | **10,823 ms** | **11,277 ms** | **+4.193%** |

All twelve reports remained exact, zero-swap, and below 558 MB RSS. The
treatment failed every performance advance gate and was removed. The
evaluator and release binary returned byte-for-byte to the accepted hashes.

One isolated fusion remains registered: retain the exact production H2
mapping inside the fused kernel. That experiment separates intermediate
elimination from the scalar-H2 scheduling regression. Failure closes this
fusion family.

Full evidence:
[`exact-mlx-h1-h2-threadgroup-fusion-rejection-v1.md`](reports/exact-mlx-h1-h2-threadgroup-fusion-rejection-v1.md).

## Realized-Hidden Concurrency Ceiling Rejection

The accepted audit spends 125.950 seconds in eight independent
realized-hidden terminal continuations. A preregistered ceiling diagnostic
tested whether launching complete exact R600 processes concurrently could
recover unused same-Mac capacity before investing in a shared scheduler.

| Host | 1 process | 2 processes | Gain | 4 processes | Gain |
|---|---:|---:|---:|---:|---:|
| john2 | 33.963929 s | 58.449131 s | 1.162171x | 97.096545 s | 1.399182x |
| john3 | 34.230646 s | 58.740134 s | 1.165494x | 97.737026 s | 1.400928x |

All 14 jobs preserved exact R600 scores and diagnostics, used zero bootstrap
samples and policy fallbacks, shut down cleanly, and reported zero swaps.
Aggregate maximum RSS remained below 0.66 GB. The preregistered 1.50x
two-process and 2.50x four-process gates nevertheless failed on both hosts;
four-way efficiency was only about 35%.

Independent same-Mac process fan-out is rejected as a primary optimization.
The three Macs should continue to run independent outer experiments, while
each Mac's realized-hidden continuations move toward one coordinated native
scheduler, one MLX evaluator, shared cross-trajectory batches, and exact
inference-work elimination.

Full evidence:
[`full-legal-audit-realized-hidden-concurrency-rejection-v1.md`](reports/full-legal-audit-realized-hidden-concurrency-rejection-v1.md).

## Game-Scoped Public Decision Cache Acceptance

After the concurrency ceiling closed independent process fan-out, a
preregistered exact cache tested deterministic public decisions that recur
between the champion-root realized-hidden continuation and subsequent outer
play.

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 168.170 s | 153.720 s | **8.592%** |
| john3 | 165.445 s | 151.285 s | **8.559%** |
| Combined | **166.8075 s** | **152.5025 s** | **8.576%** |

All eight crossover reports and the final production report preserve the
frozen semantic BLAKE3
`f46ae73349d53d1baa3c69c0f8a3efab5766ed68ef91b6636ad65a3dea340c75`.
Each treatment records 118 hits, 922 evaluations, and 80 retained entries.
Mean per-run maximum RSS and allocator footprint remain inside the 10% gates
on both hosts, with zero swaps.

The experiment switch and uncached path were removed. Final production
measures 162.045309 seconds, a 1.496082x total speedup over the frozen
242.433050-second reference.

Full evidence:
[`full-legal-audit-public-decision-cache-acceptance-v1.md`](reports/full-legal-audit-public-decision-cache-acceptance-v1.md).

## Multiplexed Trajectory Search Acceptance

The accepted serial finalist loop issued each exact sparse evaluation in a
separate search context. The replacement keeps all eight trajectories
resident, advances them through deterministic barriers, and combines one
outstanding request per unfinished search in stable search-index order.

| Host | Control | Treatment | Improvement |
|---|---:|---:|---:|
| john2 | 153.799921 s | 128.772081 s | **16.273%** |
| john3 | 151.379278 s | 129.437743 s | **14.494%** |
| Combined | **152.589600 s** | **129.104912 s** | **15.391%** |

Combined realized-hidden time improved 20.486%. All eight reports preserved
the frozen semantic BLAKE3, exact logical diagnostics, zero swaps, zero
bootstraps, and zero fallbacks. Maximum treatment RSS was 970,457,088 bytes.

The qualification switch and serial branch were removed. Switch-free
production measures 143.775461 seconds, or 1.686192x versus the reference.

Full evidence:
[`full-legal-audit-multiplexed-trajectory-search-acceptance-v1.md`](reports/full-legal-audit-multiplexed-trajectory-search-acceptance-v1.md).
