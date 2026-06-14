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

| Tier | Policy | Budget |
|---|---|---:|
| Instant | Exact immediate-score greedy | Near-instant |
| Interactive | Pattern-aware K8/H6/B8/M4 | 0.292 s/game direct control |
| Research | Final-five R8 K8/H6/B8/M4 c90 and experiments | Unrestricted local |

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
