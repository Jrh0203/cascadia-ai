# Cascadia V2 Plan To Reach 100

Status: active research program  
Ruleset: four-player AAAAA, no habitat bonuses  
Target: at least 100.000 mean base score under the canonical benchmark  
Compute boundary: john1, john2, john3, and john4 only; no external compute
Research throughput objective: minimize wall time to trustworthy decisions,
with zero duplicate discovery compute and at least 85% useful aggregate CPU
utilization whenever compatible CPU work is queued. Parallelize across
independent scientific decisions before parallelizing one decision across
additional Macs unless measured marginal scaling proves that the latter
shortens the campaign critical path more.

## Position

The strongest qualified canonical-engine reference currently scores **95.744**
over 1,000 held-out games and 4,000 seats, with a game-block 95% confidence
interval of `[95.652,95.837]`. The measured gap to the target is therefore
**4.256 points**.

The existing system combines a hand-built candidate frontier, a historical
NNUE evaluator, and root-level Monte Carlo evaluation. Increasing rollout
counts or widening the same frontier has not reliably compounded strength.
The path to 100 should instead build a self-improving policy-and-search system
that can:

1. represent every legal decision without depending on a brittle hand-built
   candidate cap;
2. evaluate board, market, supply, and opponents jointly;
3. search stochastic public-information futures with the exact Cascadia
   simulator;
4. turn expensive search improvements into a faster MLX policy and value
   model through expert iteration.

Historical research notes are context, not authority. Every hypothesis in this
program must be tested from fresh evidence with preregistered gates. Old null
results prevent accidental exact repetition, but they do not veto a materially
different model, search method, target, or experimental design.

## Definition Of Done

The project is complete when one frozen player satisfies all of the following:

- four-player AAAAA rules with habitat bonuses disabled;
- the same frozen policy occupies all four seats;
- a fresh, untouched 1,000-game final-domain run;
- mean base score of at least **100.000**;
- ideally, a game-block 95% confidence interval whose lower bound exceeds 100;
- no hidden-state leakage or use of future bag order;
- Rust owns rules, legality, scoring, simulation, and search;
- MLX owns all Apple neural training and inference;
- all artifacts are reproducible locally from manifests and checksums;
- the player remains usable through the CLI, API, and interactive web tool.

Development should target **100.25 to 100.50** over at least 500 fresh games
before opening the final 1,000-game suite. That buffer reduces the chance that
the final verdict depends on ordinary benchmark noise.

## Non-Negotiable Constraints

- No external compute. The entire campaign runs on john1, john2, john3, and john4.
- No score claim from unpaired anecdotes or tiny seed sets.
- No promotion from training or validation metrics alone.
- No performance win obtained by reducing candidates, rollouts, search depth,
  stochastic samples, model quality, numerical fidelity, or rules accuracy.
- No truncated rollout, learned leaf, fallback policy, or altered benchmark
  contract may be represented as a pure optimization.
- Pure performance changes must preserve the chosen player's decisions,
  search diagnostics, score breakdown, and strength.
- Research changes may alter behavior, but must pass explicit paired strength
  gates before becoming the new baseline.
- Test, validation, confirmation, and final seed domains remain separated.
- CPU capacity is the campaign's primary scarce resource. Every campaign must
  be designed as an adaptive four-host portfolio before launch, with one
  independent decision-changing workload per healthy Mac whenever possible.
- A model that can reach its preregistered training gate on one Mac is trained
  once during discovery. The other Macs must advance distinct experiments,
  unique data or evaluation shards, shared prerequisites, or reanalysis.
  Duplicate training is allowed only when replication is itself a frozen
  scientific question.
- The scheduler must be work-conserving: no healthy Mac may remain idle while
  compatible decision-changing work is queued, and assignments must be
  reconsidered whenever a job finishes, fails, or unblocks new work.

## Phase 0: Performance Foundation - Closed As A Win

The performance campaign is complete. The project owner accepted the strongest
exact implementation achieved on 2026-06-15 and directed all further work
toward model strength.

The accepted exact full-terminal R600 path has cleared this gate at
**10.004528x** single-Mac throughput versus its 141.027296-second
full-strength reference. Exact shared-memory MLX transport, direct
rollout-template preparation, bounded pipeline state slices, dead local
outcome-buffer elision, and fresh race-free PGO now measure
14.096346521 seconds across john2 and john3. The 10.0x threshold is
14.1027296 seconds, so the accepted path is 0.006383079 seconds inside it.
The full-legal teacher loop is frozen at 242.433050 seconds. Exact
static-screen cohorts, sparse-row deduplication, bounded public-state caches,
game-scoped decision reuse, and deterministic multi-trajectory scheduling
reduced it to **143.775461 seconds**, a **1.686192x exact end-to-end speedup**
without changing semantics. Mirrored confirmation improved john2 by 16.273%,
john3 by 14.494%, and the combined mean by 15.391%. This is the accepted
teacher performance baseline and is declared a win. Its former
24.243305-second aspirational threshold is retired as a blocking requirement.

No additional pure-performance experiment may delay the strength campaign.
Performance work may resume only in response to a measured strength-experiment
bottleneck that prevents execution, and it must remain exact and
strength-neutral.

The scheduler advances eight independent realized-hidden finalists in
lockstep through one Rayon pool and one MLX evaluator. Switch-free production
leaves 103.119389 seconds in realized-hidden continuations. Same-Mac process
fan-out remains closed.
An exact 6,116,501-row diagnostic found zero recurrence across waves or
halving rounds within one search, closing per-search prediction caching. The
cross-request diagnostic then found only 247 duplicates among 891,486
coalesced rows, closing global batch deduplication at 0.027707% reuse. The next
replicated three-node stage profile measured 47.813 seconds of serialized MLX
evaluation, 183.439 cumulative worker-seconds of opponent advancement, and
162.811 cumulative worker-seconds of template preparation on john2/john3.
These measurements remain available for future bottleneck-driven work, but
they no longer define the active roadmap.

The first H1 geometry sweep is closed. Moving from one `float4` accumulator per
thread to two or four increased combined MLX evaluation by 4.720% and 15.210%
respectively, with complete-wall regressions on every Mac. Explicit
SIMD-group sharing of the sparse index stream was then exact but improved
combined wall by only 0.180% and MLX evaluation by only 0.711%, with a
regression on john1. Both treatments were removed.

Replicated per-layer timing measured H1 at 76.410% and 76.491% of
corrected layer time on john2 and john3, which agree within 0.081 percentage
point. Across the complete audit, the global H1 write and H2 read move
183,926,591,488 bytes. The first exact fusion kept H1 in threadgroup memory
but changed H2 from eight `float8` threads per row to 64 scalar threads. It
regressed combined wall 1.223% and MLX evaluation 4.193%, with regressions on
all three Macs, and was removed. A production-H2-geometry follow-up passed
direct parity and was closed before an authorized screen when the performance
campaign ended. Raw post-closure screen files are invalid evidence and do not
change the accepted baseline.

### 0.1 Freeze Representative Benchmarks

Create reproducible, uncontended one-Mac benchmarks for:

- one complete full-strength game;
- one representative early-, middle-, and late-game search decision;
- complete legal-action enumeration and feature construction;
- batched MLX policy and value inference;
- rollout simulation, scoring, and search bookkeeping;
- dataset serialization and artifact writing.

The primary metric is end-to-end wall time and games or searched decisions per
hour. Kernel-only or microbenchmark gains do not satisfy the 10x requirement.
Startup, IPC, feature construction, simulation, inference, and output writing
must all be included.

### 0.2 Profile Before Optimizing

Use native sampling, Instruments, Rust benchmarks, MLX timing, allocation
counts, batch-shape telemetry, and per-stage wall-time accounting. Rank costs
by total campaign impact rather than optimizing whichever function is easiest
to measure.

Likely exact optimization areas include:

- incremental apply/undo, scoring, frontier, and feature updates;
- structure-of-arrays and compact immutable state representations;
- allocation-free action generation and rollout hot paths;
- exact common-subexpression reuse across actions, rollouts, and chance states;
- transposition and public-state caches keyed by complete relevant state;
- batched action encoding and MLX inference across decisions and games;
- persistent Rust-to-MLX transport with fewer copies and synchronizations;
- pipelined CPU simulation and Apple GPU inference;
- tree reuse between turns and reusable afterstate/chance-node statistics;
- work-stealing pools that avoid nested oversubscription;
- release-profile, link-time, and profile-guided optimization;
- faster checksummed artifact encoding outside the measured search critical
  path.

### 0.3 Strength-Neutral Acceptance Gate

An optimization is accepted only if it passes:

1. identical rules, model, weights, search budget, candidates, horizon, random
   streams, and benchmark inputs;
2. unit and property parity against the reference implementation;
3. bit-exact action traces, score breakdowns, and search diagnostics on the
   deterministic parity suite;
4. repeated uncontended measurements with a material end-to-end improvement
   on the bottleneck that justified reopening performance work;
5. no regression in peak memory, reliability, resumability, or user-facing
   latency that creates an operational disadvantage;
6. permanent performance regression tests and documented reproduction
   commands.

If an optimization changes decisions or scores, it is not a pure performance
optimization and does not satisfy this phase, even if its sampled mean appears
similar.

### 0.4 Exit Criteria

- Exact full-terminal R600 throughput improved 10.004528x.
- Exact full-legal teacher throughput improved 1.686192x and is accepted as
  sufficient for the strength campaign.
- Exact full-strength behavioral parity.
- No weaker search contract or approximate shortcut.
- Reproducible benchmark report committed under `docs/v2/reports/`.
- The optimized path becomes the only default used by later experiments.

## Phase 1: Establish The Reachable Ceiling

Before training another model, determine where the missing points actually
live. Run a **Full-Legal Decision Regret Audit**:

1. Collect about 1,000 fresh champion decisions balanced across game phase.
2. Enumerate every canonical legal action at each decision.
3. Apply a cheap all-action screen.
4. Re-evaluate the strongest 64 actions with a substantial public-information
   budget.
5. Re-evaluate the strongest eight actions with a high-confidence budget.
6. Compare the champion's action, the best action in its current frontier, the
   best full-legal action, and a clearly labeled perfect-future diagnostic.
7. Decompose regret by phase, draft choice, wildlife pattern, habitat, Nature
   Token use, opponent pressure, and market survival.

This separates candidate recall, value error, continuation error, opponent
modeling, stochastic uncertainty, and token/prelude policy.

### Active Experiments

The preregistered audit completed 13 fresh games and 1,040 decisions on all
three Macs:

- john1: seeds `61000-61004`;
- john2: seeds `61005-61008`;
- john3: seeds `61009-61012`.

It screened 3,872,079 canonical actions and initially measured 0.350 mean
champion decision regret, or 6.995 points of first-order 20-turn diagnostic
headroom. The locked K1024 recovery then completed over the same 13 games and
all 1,040 decisions. Under the stronger screen it measured 0.436 mean champion
regret, or 8.723 first-order points, with proposal/frontier regret dominant at
0.329. K1024 recalled 99.423% of measured winners and retained only 0.002 mean
regret, passing both substantive screen-width gates.

The corrected K64 plus champion-frontier online oracle completed 12 fresh
paired games with full integrity. It improved 11 games and produced a positive
paired interval, but stopped at 98.583 mean and +2.375 paired points, below
both advance thresholds. Its confirmation domain remains unopened.

The K1024 online oracle then completed 12 fresh paired games with full
integrity and 99.375% top-screen recall. It improved 10 games, but stopped at
98.417 mean and +2.854 paired points, below both advance thresholds. Every
host was positive and the paired interval was `[1.583, 4.188]`; nevertheless,
its confirmation domain is not authorized. K2048 remains closed.

The result separates recall from conversion. Complete-action recall is no
longer the immediate bottleneck, but the expensive oracle does not convert its
0.313 mean local champion regret into a 100-point online player. The next
experiment is therefore a focused graded-value learner, not a larger
brute-force screen or a large self-play loop.

### Exit Criteria

- A credible public-information oracle exceeds 102 mean.
- The measured opportunity contains at least six points of diagnostic
  headroom, giving enough margin to recover the required 4.256 points.
- The dominant error sources are quantified with confidence intervals.

Those gates did fail. Do not begin a large self-play loop. Use the completed
audit and online evidence for the bounded
`complete-action-graded-oracle-ranker-v1` experiment, which directly tests
whether graded high-budget values can improve action ranking without opening
new gameplay seeds. Reassess the oracle and search design before scaling data
generation if that held-out ranking experiment fails.

## Phase 2: Solve Full-Legal Action Recall

Replace the capped handcrafted frontier as the primary source of actions with a
complete, factorized MLX action proposal and ranking system.

Represent and score:

- market draft or independent draft;
- tile coordinate;
- tile rotation;
- wildlife coordinate;
- pre-move refresh and Nature Token choices;
- the observable market and supply state after the action;
- interactions between the action, board, opponents, and likely next market.

All canonical legal actions should be encoded in large, shape-stable MLX
batches. Search may narrow the set after learned scoring, but no legal action
may be unreachable because a heuristic generator omitted it.

### Exit Criteria

- Greater than 98% top-64 recall of the high-budget full-legal teacher.
- Less than 0.15 mean oracle regret for the retained search set.
- Stable recall across early, middle, late, token, and independent-draft
  decisions.
- Throughput remains within the Phase 0 performance envelope.

## Phase 3: Train A New Policy And Value Model

Build a model around the actual decision structure rather than extending the
historical scalar NNUE indefinitely.

### State Representation

- hex-board graph with oriented terrain edges;
- wildlife occupancy, legal slots, connected habitats, and pattern structure;
- public market, visible supply, bag counts, and turn phase;
- all opponent boards, pattern progress, tokens, and draft pressure;
- explicit candidate action tokens and action-to-state cross-attention.

### Outputs

- policy prior over the complete legal action set;
- distributional focal-player return;
- decomposed habitat, Bear, Elk, Salmon, Hawk, Fox, and Nature Token returns;
- uncertainty or return quantiles for search allocation;
- opponent next-pick and market-survival auxiliary predictions;
- optional action-advantage and search-visit targets.

Train and infer with MLX on Apple Silicon. Keep the exact Rust simulator as the
world model; do not learn game dynamics that are already known.

Run model development as a portfolio rather than a mirrored training fleet.
One Mac owns each indivisible MLX training origin. The remaining Macs
simultaneously run different model treatments, data-quality experiments,
teacher evaluation, error analysis, or unique corpus generation. Multiple
Macs may train concurrently only when each trainer tests a distinct
preregistered hypothesis; extra copies of one training run are deferred until
seed sensitivity or confirmation is the question.

### Exit Criteria

- Better held-out full-legal regret than the current evaluator.
- Calibrated value and uncertainty on untouched games.
- No collapse in any game phase or action family.
- Fast policy-only play first reproduces the current champion's strength, then
  exceeds it before search gains are claimed.

## Phase 4: Replace Root-Only MCE With Stochastic Tree Search

Use the policy and value model inside an exact-model public-information search:

- explicit decision, afterstate, and chance nodes;
- public bag sampling and redetermination without hidden-state leakage;
- progressive action widening from the complete learned proposal;
- batched MLX inference across leaves, games, and concurrent searches;
- common random numbers where they reduce comparison variance;
- tree and inference-cache reuse across turns;
- ordinary search treatment of refreshes, independent drafts, and Nature
  Tokens;
- allocation guided by policy prior, uncertainty, and observed return spread.

Relevant conceptual precedents include
[Expert Iteration](https://arxiv.org/abs/1705.08439),
[Sampled MuZero](https://arxiv.org/abs/2104.06303),
[Stochastic MuZero](https://openreview.net/forum?id=X6D9bAHhBQ1), and
[Student of Games](https://arxiv.org/abs/2112.03178). Cascadia should use its
exact simulator rather than a learned world model.

### Exit Criteria

- At least +0.50 paired mean over the frozen champion on 100 fresh games.
- Positive confidence interval after the registered confirmation stage.
- Search gain survives all four seats and does not depend on one game phase.
- Runtime remains compatible with continuous four-node data generation.

## Phase 5: Expert Iteration

Run a closed local actor-learner loop:

1. Search produces improved action distributions and return distributions.
2. MLX trains policy, value, uncertainty, and auxiliary heads.
3. The improved model generates the next searched corpus.
4. Reanalysis refreshes high-value older states under the current search.
5. A population of current and historical checkpoints prevents narrow
   self-play overfitting.
6. Each iteration is paired against the frozen champion and the previous
   iteration.
7. Only confirmed gains become the next data-generating baseline.

Prefer 2,000 to 5,000 searched games per serious iteration after the
performance gate. Allocate cheap search broadly, then spend R600/R1200-class
budgets on uncertain, high-regret, or strategically important decisions.
Prioritize early and middle game decisions because improvements there have the
most downstream leverage.

Keep the champion action in every training candidate set as an anchor until the
learned/search policy beats it. Train from search visits and return
distributions rather than only hard winner labels.

Pipeline expert iteration across the cluster. Shard unique search, self-play,
teacher, and reanalysis units across all four Macs with dynamic work claiming,
while each distinct learner uses one MLX origin. Do not hold three Macs idle
behind a learner barrier: keep the next independent ablations, data audits,
and evaluation domains launch-ready so training latency is overlapped with
decision-changing CPU work.

## Phase 6: Residual-Driven Research

Once the main loop is strong, rerun the Phase 1 audit and attack only measured
remaining losses. Candidate directions may include:

- longer-horizon market survival and opponent draft modeling;
- commitment-aware wildlife planning without species tunnel vision;
- Nature Token option value;
- habitat and wildlife joint allocation;
- search calibration in high-variance market states;
- model capacity or representation changes justified by residual error.

Do not launch broad feature fishing. Every experiment needs a stated mechanism,
expected score source, cost estimate, preregistered gate, and stop condition.

### Exit Criteria

- At least 100.25 mean over 500 fresh development games.
- Positive paired result against the prior champion.
- No unresolved correctness, leakage, or reproducibility issue.

## Phase 7: Final Qualification

Freeze code, model, weights, search configuration, manifests, and executable
fingerprints before opening the final domain.

Run 1,000 fresh games distributed across john1, john2, john3, and john4. Aggregate by
correlated game block and publish:

- mean, standard error, and 95% confidence interval;
- P10, P50, and P90 seat scores;
- full category decomposition;
- paired result against the frozen previous champion;
- decision and game latency;
- host allocation and utilization;
- complete provenance and explicit 100-point verdict.

The target is passed at mean >=100.000. The preferred scientific result is a
lower 95% confidence bound >=100.000.

## Four-Node Compute Plan

The four Macs are one local research cluster. john1 is both coordinator and a
full compute worker; it must not be reserved only for orchestration. The
cluster objective is **validated research progress per wall-clock hour**, not
symmetry, duplicated activity, or utilization for its own sake.

This section is binding methodology for every phase above. Experiment
manifests must describe the complete four-host portfolio, not only the primary
job, and campaign closeouts must evaluate whether the chosen allocation
maximized research throughput.

The default unit of scheduling is an **independent scientific decision**, not a
copy of a process. When four separable hypotheses are ready, run four different
experiments concurrently. When one model can be trained to its decision gate
on one Mac, train it once and use the remaining Macs for different hypotheses,
nonduplicative data generation, teacher evaluation, profiling, ablations, or
reanalysis. Duplicate training is reserved for seed sensitivity, confirmation,
or cross-host reproducibility after a treatment has earned that expense.

CPU throughput is the cluster's primary scarce resource. Campaigns must
therefore be designed as portfolios of independent hypotheses and divisible
evidence jobs before they are launched, so free cores can immediately perform
work that changes a research decision. The scheduler must optimize accepted or
rejected hypotheses per elapsed hour and useful CPU-core-hours, not process
count. Four distinct experiments running concurrently are preferred whenever
their prerequisites and scientific boundaries permit it.

Parallelism must match the work rather than follow a fixed all-host recipe:

- run four independent experiments when four hypotheses can be tested without
  sharing mutable outputs;
- shard unique data generation, labeling, simulation, evaluation, or seed
  blocks across all compatible hosts when every shard contributes new evidence;
- run an indivisible MLX training job on the single best-fit Mac and use the
  other three Macs for independent experiments or prerequisite evidence;
- replicate a trainer only when the replica itself answers a preregistered
  question about variance, reproducibility, portability, or confirmation.

For every campaign, identify the critical-path decision and a side portfolio of
independent decisions that can run while it is blocked. Allocate hosts by
expected decision value per elapsed hour, adjusted for critical-path impact and
measured host throughput. Revisit that allocation whenever work completes,
rather than preserving the original assignment for administrative symmetry.

### Portfolio Allocation Algorithm

Build every campaign as a four-host portfolio before reserving compute. For
each ready job, estimate:

- the scientific decision it can accept, reject, or unblock;
- expected information value and probability of changing that decision;
- wall time and useful CPU-core-hours on each compatible host;
- MLX, memory, disk, and network requirements;
- dependencies, critical-path impact, and latest useful completion time;
- whether the work is independent, a divisible unique shard, a shared
  prerequisite, or a scientifically justified replica.

Rank assignments by expected decision value per host-hour, with a bonus for
shortening the campaign critical path and a penalty for duplication,
interference, queue starvation, and straggler risk. Choose the four-host set
that maximizes expected trustworthy decisions completed before the next
allocation event, not the four individually largest jobs in isolation. This
allows a short prerequisite that unlocks three experiments to outrank a long
standalone pilot, and prevents four copies of one trainer from displacing four
independent questions.

Use the following allocation rules:

1. If four independent, preregistered experiments are ready, run one on each
   Mac.
2. If a model needs training and one Mac can reach its decision gate within the
   experiment budget, assign exactly one training origin. Use the other Macs
   for different models, mechanisms, analyses, or unique evidence.
3. If unique CPU work is divisible, expose small resumable units and let all
   compatible hosts claim dynamically. Rebalance continuously instead of
   waiting at fixed host barriers.
4. If fewer than four independent decisions are ready, run shared
   prerequisites and nonduplicative evidence that unlock or shorten the next
   decisions. Queue preparation itself is required research work.
5. Add a replica only when its result is part of a frozen gate for variance,
   reproducibility, portability, confirmation, or straggler recovery near a
   hard deadline. Record the reason before launch.
6. Stop or reallocate a job immediately when a frozen futility gate proves
   that further compute cannot change its decision. Do not consume the
   remaining budget merely because it was originally reserved.

Training parallelism is across **distinct models or hypotheses**, not copies of
the same seed. A campaign may train draft, tile, wildlife, and value models on
four Macs simultaneously because each answers a different question. It should
not train one model four times unless run-to-run variance is itself the
preregistered question. When a single MLX trainer leaves substantial CPU idle,
first place CPU work on other Macs; co-locate work with that trainer only after
an interference measurement shows a net gain in campaign throughput.

At every scheduling event, compare the planned allocation against two
counterfactuals: four independent experiments and one critical-path job plus
three backfill jobs. Record why the selected portfolio has the shortest
expected time to trustworthy decisions. This makes cluster allocation a
testable research choice rather than an informal convention.

### Marginal Host Allocation

Treat every healthy Mac as a scarce allocation decision. Before assigning an
additional host to work that is already running elsewhere, compare:

- the expected reduction in time to the current decision from adding that
  host;
- the expected time to a separate trustworthy decision if that host instead
  runs the best compatible independent job;
- the amount of new evidence produced in each case;
- communication, synchronization, artifact-transfer, and straggler costs;
- measured CPU scaling and any CPU/MLX interference on the candidate host.

Assign the host to the existing job only when its work is unique, resumable,
and the measured marginal speedup shortens the campaign critical path more
than the best independent alternative. Otherwise launch the independent
experiment. This rule applies recursively: a two-host job must justify the
second Mac, a three-host job must justify the third, and a whole-cluster job
must justify every Mac.

Whole-cluster execution is normally correct for embarrassingly parallel,
nonduplicative CPU work such as unique game generation, teacher labeling,
reanalysis, paired evaluation, and final seed blocks. Single-host execution is
normally correct for an indivisible MLX training origin. Four simultaneous
single-host experiments are normally correct during hypothesis discovery.
These are defaults, not fixed topology: measured marginal decision throughput
chooses the allocation.

Maintain one resource ledger per host with calibrated CPU worker capacity,
memory ceiling, disk budget, and one exclusive MLX-device slot. Job manifests
must declare their dominant resource and measured scaling curve. The scheduler
may safely co-locate a CPU job with an MLX trainer only when a paired
interference check shows higher combined decision throughput, bounded memory,
and no swap; otherwise the MLX origin owns that host and CPU work runs on the
other Macs.

For every long-running critical-path job, prepare at least three independent
backfill lanes before launch. Each lane must have a frozen hypothesis or
decision it can change, a stop rule, compatible-host list, and enough ready
work to occupy a newly free Mac without waiting for experiment design. Queue
depletion behind one trainer is a planning failure, not an inherently serial
campaign.

| Campaign state | john1 | john2 | john3 | john4 |
|---|---|---|---|---|
| Hypothesis discovery | independent experiment A | independent experiment B | independent experiment C | independent experiment D |
| Profiling | primary single-Mac reference | next independent mechanism | next independent mechanism | next independent mechanism |
| Data collection | search/self-play shard | search/self-play shard | search/self-play shard | search/self-play shard |
| MLX pilot training | one frozen pilot trainer | independent experiment | independent experiment | independent experiment |
| MLX confirmation | selected replica or cross-replay | justified seed/host replica | justified seed/host replica | justified seed/host replica |
| Reanalysis | reanalysis shard | reanalysis shard | reanalysis shard | reanalysis shard |
| Candidate screening | independent treatment shard | independent treatment shard | independent treatment shard | independent treatment shard |
| Confirmation/final | deterministic seed shard | deterministic seed shard | deterministic seed shard | deterministic seed shard |

### Scheduling Requirements

- Use a resumable manifest-backed queue with deterministic shard ownership and
  work stealing for unfinished compatible jobs.
- Size divisible CPU work at the smallest independently verifiable unit that
  keeps scheduling overhead negligible. Workers should claim the next ready
  unit dynamically instead of receiving coarse fixed host shards when unit
  runtimes can vary materially.
- Estimate unit cost from a representative sample, longest-processing-time
  order the remaining queue when practical, and reserve enough tail units for
  work stealing. A host barrier is allowed only when the scientific protocol
  requires one; static equal-count partitioning is not evidence of balanced
  work.
- Keep at least two launch-ready, decision-changing jobs for every host expected
  to become free during a campaign window. A long trainer must never be the
  cluster's only prepared work item.
- Maintain a live portfolio of ready work. Every queued job records its
  hypothesis, decision gate, dependencies, resource profile, expected runtime,
  expected information value, compatible hosts, parallel scaling curve,
  artifact destination, and whether another run would be independent work or a
  replica.
- Recompute assignments whenever a job finishes, fails, or unblocks another
  job. Select the assignment that minimizes expected time to the next
  trustworthy research decision while keeping later critical-path work fed.
- Backfill idle capacity with the highest-value compatible job whose runtime
  fits before a known critical-path reservation. Do not leave a host idle
  merely because the next preferred job has not yet unblocked.
- Use measured per-host runtime and memory history rather than assuming the
  Macs are interchangeable. Put the longest critical-path job on the fastest
  compatible host, then pack shorter independent jobs around it to minimize
  cluster makespan.
- Schedule for the shortest time to the next trustworthy decision. Prefer four
  independent, preregistered pilots over four duplicate replicas when one
  machine can train the model within the experiment's decision budget.
- Before launch, record an explicit allocation decision for every healthy host:
  independent hypothesis, divisible evidence shard, justified replica, or
  intentionally idle with a reason. The default discovery allocation is four
  different experiments; the default training allocation is one trainer plus
  three nonduplicative evidence-producing jobs. A symmetric four-host launch
  is never its own justification.
- A new MLX treatment defaults to one training host. Additional training
  replicas require a written reason such as seed sensitivity, model selection,
  cross-host reproducibility, or confirmation of an already promising pilot.
  Replication is a validation tool, not the default use of the cluster.
- While one Mac runs MLX training, assign the other three Macs independent
  mechanisms, CPU-heavy data generation, teacher evaluation, reanalysis,
  correctness work, or disjoint candidate screens. Do not leave them waiting
  merely to mirror the trainer.
- Do not train the same model on all four Macs merely to increase utilization.
  If a single training run answers the preregistered question, extra replicas
  are duplicative and reduce research throughput. Prefer one trainer plus three
  distinct experiments or evidence-producing workloads.
- Run independent treatments concurrently only when their inputs, frozen
  outputs, host locks, and scientific decisions are isolated. Each treatment
  keeps its own manifest, acceptance gate, stop rule, and artifact root.
- Separate discovery from confirmation. Use broad parallelism for cheap
  independent pilots; spend duplicate seeds and cross-host replays only on
  treatments that pass their pilot gate or when the preregistered question is
  explicitly about variance or portability.
- Keep one MLX process per Apple GPU and tune CPU pools per host to avoid
  oversubscription.
- Treat CPU and MLX occupancy as separate resources. Low CPU usage on the one
  necessary MLX trainer does not justify a duplicate trainer; it requires
  CPU-heavy independent work on the other hosts. Co-locate CPU and MLX jobs on
  one host only after an interference benchmark proves higher total throughput.
- Whenever compatible work is queued, every healthy node receives work.
- No healthy node may remain idle for more than five minutes while compatible
  queued work exists.
- Keep enough independent pilot hypotheses and divisible evidence jobs prepared
  that the queue does not drain behind one long training run. Experiment design
  must expose parallel branches before compute becomes available. Maintain at
  least two launch-ready, decision-changing jobs per host expected to become
  free during the next campaign window; preparation and dependency work count
  only when they produce a durable input to a frozen decision gate.
- During CPU-heavy collection, all four Macs participate, including john1.
- CPU-bound workloads must be sharded across all available physical cores with
  measured scaling, bounded memory, and no swap. Increase worker counts until
  throughput stops improving, then retain the measured optimum per host.
- Calibrate each new CPU workload at representative input sizes before a large
  launch. Record throughput at increasing worker counts, choose the highest
  stable no-swap configuration, and reuse that measured host-specific setting
  instead of assuming that logical-core count equals useful parallelism.
- When compatible CPU work is queued, target at least 85% useful aggregate
  physical-core utilization across the healthy cluster over each 30-minute
  campaign window. A miss requires a documented cause and a scheduler,
  implementation, or queue-depth correction before the next large run. Never
  inflate this number with duplicate work that cannot change a research
  decision.
- Treat repeated low CPU utilization as one of four concrete defects:
  insufficient ready work, undersized worker pools, serial implementation
  bottlenecks, or harmful CPU/MLX interference. Measure which defect applies
  and correct it before the next comparable campaign. The correction may be
  deeper queue preparation, finer dynamic sharding, host-specific concurrency
  tuning, or a profiled exact performance change; duplicate training is not a
  utilization remedy.
- During every CPU-bound campaign, sample useful completed units and throughput
  at least once per scheduling interval. Increase or decrease host concurrency
  when marginal throughput changes, memory pressure appears, or stragglers
  dominate. Preserve the measured best worker setting by workload class and
  host for subsequent campaigns.
- Do not stack a CPU-saturating job beside MLX on the same Mac unless a paired
  interference check proves that neither workload loses material throughput.
  Use the other hosts first.
- Use `caffeinate` for every remote long-running shard.
- Resume atomically after SSH interruption, process failure, sleep, or reboot.
- Reject artifacts produced from a different revision, model, manifest,
  ruleset, or executable fingerprint.
- Track queue depth, experiment identity, completed work, failures, CPU,
  memory, GPU/Metal activity where measurable, and host utilization through
  the cluster dashboard.
- Report both aggregate cluster throughput and normalized per-node throughput
  so scaling cannot conceal a single-machine regression.

Cluster utilization is an experimental concern, not an afterthought. For every
large run, the report must include assigned wall time, productive wall time,
idle time with work queued, failures/retries, work completed per node, and the
reason any healthy node was intentionally left idle. CPU utilization should be
raised whenever CPU-bound compatible work exists, but the governing metric is
accepted or rejected hypotheses per wall-clock hour, not raw CPU percentage.

### Throughput Scorecard

Evaluate each campaign window using all of the following:

- trustworthy hypothesis decisions per wall-clock hour;
- critical-path completion time and total cluster makespan;
- useful CPU-core-hours divided by available CPU-core-hours while compatible
  CPU work was queued;
- useful MLX device-hours divided by available MLX device-hours while
  compatible neural work was queued;
- duplicate compute fraction, which should be zero during discovery unless
  replication is the preregistered scientific question;
- idle healthy-node minutes while compatible work was queued;
- useful aggregate CPU utilization during CPU-bound campaign windows, with the
  85% target evaluated independently of MLX-device utilization;
- failed or repeated work caused by orchestration, revision, artifact, or
  resumability defects.

Raw utilization is a diagnostic, not the objective. A Mac running one necessary
MLX trainer can be correctly scheduled even if its CPU is not saturated; the
cluster is poorly scheduled if the other three Macs are idle despite a backlog
of independent or divisible work. Optimize the portfolio as a whole.

### Research Throughput Scheduler

Maintain a launch-ready queue ordered by expected decision value per wall-clock
hour, then by critical-path impact. Every queued job must name the hypothesis
or frozen decision it can change, its expected duration, required cores and
memory, dependencies, artifact destination, and stop rule. Jobs that cannot
change a decision do not consume cluster capacity merely to raise utilization.

The queue uses four workload classes:

1. **Independent experiment**: a separate mechanism with its own gate and the
   highest discovery priority.
2. **Divisible evidence**: disjoint self-play, teacher, evaluation, reanalysis,
   or seed shards whose outputs combine without duplication.
3. **Shared prerequisite**: correctness, dataset, cache, or infrastructure work
   that unlocks multiple experiments.
4. **Replica**: repeated training or evaluation used only for a preregistered
   variance, reproducibility, or confirmation question.

When all four hosts are healthy, the discovery default is four independent
experiments. If fewer than four independent mechanisms are ready, fill the
remaining hosts with divisible evidence or shared prerequisites. A single MLX
training run occupies one host unless replication itself is the experiment;
the other three hosts continue independent CPU-heavy research.

At every job completion, failure, or early stop, recompute the allocation across
all healthy hosts. Prefer, in order:

1. a shared prerequisite whose completion unlocks multiple independent
   experiments, sharded across hosts only when measured scaling justifies it;
2. independent experiments that can accept or reject different mechanisms;
3. nonduplicative evidence generation, evaluation, profiling, or reanalysis
   that shortens a frozen experiment's critical path;
4. confirmation replicas only after their treatment passes its pilot gate or
   when variance, portability, or reproducibility is itself the question.

One-host training is the default because a second identical trainer usually
does not answer a second question. During that training, the scheduler must
keep the remaining hosts on distinct launch-ready work. Whole-cluster
replication is prohibited during discovery. Whole-cluster sharding is reserved
for divisible work such as self-play, teacher labeling, canonical evaluation,
or final seed blocks where each completed shard contributes unique evidence.

Report research throughput as frozen decisions completed per elapsed campaign
hour, alongside useful CPU utilization. If utilization is low while the queue
is nonempty, fix worker scaling, orchestration, host assignment, or queue depth.
If utilization is low because the only critical job is a single MLX trainer,
use the idle hosts to advance independent hypotheses instead of duplicating the
trainer.

Every campaign closeout must state whether the four-host allocation maximized
research throughput in hindsight. Any avoidable idle time, duplicate discovery
compute, poor worker scaling, or unprepared queue slot becomes a concrete
scheduler correction in the next campaign's manifest.

## Experimental Discipline

Every strength experiment follows the same funnel:

1. Write the mechanism and frozen acceptance gates.
2. Run correctness and deterministic replay tests.
3. Run a tiny implementation smoke without drawing a strength conclusion.
4. Run a fresh paired pilot.
5. Run a disjoint paired confirmation only if the pilot passes.
6. Promote only if the confirmation confidence interval passes its gate.
7. Record negative results with enough detail to prevent accidental exact
   repetition.

Performance and strength are separate axes. A strength experiment may be
slower while being investigated, but it cannot replace the production
champion until its deployment budget is defined. A pure performance
optimization may never change player behavior or weaken search.

## Immediate Next Experiment

ADR 0090's four independent diagnostics are complete. They selected
optimization/capacity underfit: train target recall reached only 29.36% with
0.18% exact set recovery, exact observable collisions were zero, the
train/validation gap was only 3.15 points, and misses were broad.

ADR 0091's single-host target-only curriculum is rejected. It improved train
target recall only from 29.36% to 30.97% and validation from 26.21% to
26.29%, recovered no validation target set, and reduced winner recall. A
separate ceiling proved the current score range can recover every set.

ADR 0092's single-extreme top-K boundary pilot is rejected. john3 proved the
full-model update was finite and directionally correct on 10,854 actions.
john4 then optimized all 12 widest validation target sets to 100% exact
recovery inside the ±12 score range. Yet no neural epoch beat the warm start;
loss fell while validation target recall collapsed from 26.21% to 18.76%.

ADR 0093's rank-matched full-boundary pilot is also rejected. Exact gradient
coverage and direct bounded convergence passed, but no epoch beat the warm
start and final validation recall fell to 19.47%. Uniform set cross entropy,
single-extreme boundary loss, and distributed rank-boundary loss have now all
failed under the same representation.

ADR 0094's frozen-embedding audit is complete and rejects head-only work. The
linear probe reached 22.48% train target recall and the nonlinear probe reached
24.67%; both recovered zero exact train sets. john4 reproduced both reports
exactly. The selected model's final 192-dimensional candidate representation
therefore does not preserve the target structure.

ADR 0095's raw-observable bypass audit is complete and rejected. Raw linear,
raw nonlinear, and embedding-plus-raw probes reached only 26.31%, 28.35%, and
30.50% train target recall; each exactly recovered one of 560 train target
sets and no validation set. john4 reproduced all three scientific payloads
exactly. Candidate-wise bypass heads are therefore closed.

ADR 0096's pre-pool context fork is complete and rejected. Candidate only,
exact legacy context, richer global moments, and observable screen-top64
context reached only 28.61%-29.07% train target recall and each exactly
recovered one of 560 train target sets. Every ring replay was bit-identical.
The 192-dimensional vector produced by `candidate_projection` is therefore
classified insufficient; post-projection pooling and head work are closed.

ADR 0097's pre-compression candidate-factor integration fork is complete and
rejected. Wide concatenation, screen-relative context, factor-token attention,
and pairwise-gated interactions reached only 29.39%-30.88% train target recall
and at most 0.18% exact train sets. Every ring replay was bit-identical. The
1,344-dimensional concatenation of the seven projected factors is classified
insufficient, so further heads, pooling, width, and integration mechanisms
over those frozen factors are closed.

The immediate successor must move upstream into factor construction. Compare
distinct raw-observable and candidate-state relation encoders before the
current action, staged-state, and cross-attention factors have each been
compressed to 192 dimensions. Preserve the same open train/validation target,
one distinct mechanism per Mac, bounded MLX memory, ring replay, and closed
sealed-test/gameplay boundaries. At least one arm must materially fit train
before an end-to-end successor is authorized.

ADR 0098 is now complete and rejected. Complete raw flattening, exact local
relations, explicit market transitions, and fresh entity cross-attention
reached 30.29%, 37.87%, 29.94%, and 17.95% train target recall respectively,
with zero exact train or validation target sets in every arm. All ring replays
were bit-identical, memory and swap gates passed, and four distinct hypotheses
resolved in 2,412.43 seconds of cluster wall time. The classification is
`raw_factor_construction_insufficient`.

The immediate successor must not train another neural constructor. It must
audit the supervision itself using four distinct CPU experiments across
john1-john4: finite-R1200 cutoff signal-to-noise, cross-fidelity target
stability, uncertainty-derived soft top-64 membership, and deterministic
ceilings for smoother graded or ordinal targets. Only a target with a
preregistered open-validation ceiling sufficient for the width-64 deployment
contract may authorize one single-host MLX pilot; the other three Macs remain
assigned to independent evidence work.

ADR 0099 completed that audit. The hard cutoff failed decisively: only 10.38%
of validation target slots were statistically separated, no validation set
was completely separated, and 512 teacher resamples recovered only 41.20% of
nominal slots and 2.50% of exact sets. The uncertainty-aware expected-rank
ceiling passed at 100% validation winner recall, 100% confidence coverage,
100% distinguishable recall, and zero regret across every phase.

The immediate experiment is therefore one frozen MLX pilot that learns
continuous expected-rank ordering, not hard membership. Use one training Mac.
Assign the other three Macs independent target-cache verification,
optimization/gradient diagnostics, and generalization/error analysis; do not
duplicate the training seed unless the pilot first passes. Preserve width 64,
frontier anchors, open train/validation splits, and all sealed boundaries.

ADR 0100 completed that pilot and is rejected as
`expected_rank_optimization_underfit`. The selected model reached 32.21% train
target recall and 27.81% validation recall, with 0.18% and 0.42% exact sets.
The origin and replay were bit-identical, the gradient audit passed, and a
bounded residual range of 6 could recover every target set.

The failure mechanism is target dilution. The frozen scale-64 distribution
places only 44.84% of train and 45.51% of validation probability mass inside
the deployed target set, and only about 26% of uniform-start absolute gradient
acts there. A train-selected scale of 16 raises deployed-set mass to 93.76%;
validation independently mirrors it at 93.75%.

The immediate successor is one separately preregistered scale-16
expected-rank pilot with every other model, optimizer, width, seed, dataset,
and selector setting unchanged. Run one trainer. Use the remaining Macs for
distinct signal-alignment, trajectory, and error-anatomy work; do not duplicate
the training run. A broad target-temperature sweep is not authorized.

ADR 0101 completed that successor and is rejected as
`scale16_alignment_insufficient`. The scale-16 target placed 93.76% of train
and 93.75% of validation mass inside the deployed set, with about 48% of
uniform-start absolute gradient acting there. Both independent cache pairs,
the 12 widest-group optimization audit, bounded reachability, and origin/replay
integrity passed.

The selected epoch-10 checkpoint reached only 30.23% train target recall and
27.21% validation recall, with 0.18% and 0% exact sets. Train recall was 1.98
points below ADR 0100 and far below the 42.21% material-alignment threshold.
Concentrating the target therefore fixed supervision geometry without fixing
full-dataset fit.

The immediate successor is a preregistered four-host fit-scaling and
cross-group-interference audit on the existing open data. Run distinct
nested-subset memorization, capacity-scaling, gradient-conflict, and
error-anatomy arms concurrently across john1-john4. The audit must determine
whether expected-rank fit collapses because of finite shared capacity,
destructive gradients between groups, or insufficient public-observable
representation. Do not launch another full 560-group trainer, scale sweep,
second seed, warm start, sealed test, or gameplay run until one mechanism
passes a frozen diagnostic gate.

ADR 0102 completed that audit as
`local_optimization_or_representation_insufficient`. One-group and four-group
memorization reached only 18.92% and 30.28% recall with zero exact recovery;
independent selected-checkpoint adaptation reached 40.66% recall with zero
exact sets. Width 96/192/288 was immaterial. Gradient conflict was strong in
geometry but independent adaptation beat shared adaptation by only 2.23
recall points, so conflict-only work is not authorized.

The campaign used four distinct first-wave jobs and resolved four hypotheses
in 975.30 seconds with zero duplicate discovery training. Four cross-host
confirmation replays were bit-identical. The complete two-wave campaign took
1,961.49 seconds and preserved every sealed boundary.

The immediate successor is one bounded local objective/optimizer audit on the
frozen cohort. Compute the exact box-constrained free-residual optimum of the
current scale-16 loss, compare it with direct free-residual optimization under
the frozen optimizer schedule, and measure exact top-64 recovery throughout.
Use john1-john4 for distinct analytic, optimization-trajectory,
representation-ceiling, and replay/integrity arms. Do not run another full
trainer, width treatment, conflict mitigation treatment, sealed test, or
gameplay experiment until this audit mechanically selects one local mechanism.

ADR 0103 completed that audit as `free_residual_pipeline_invalid`. The exact
scale-16 optimum recovered every one of 64 target sets with KKT error below
`1.5e-15`, proving that the loss and residual box are sound. Frozen AdamW with
one free parameter per action reached only 59.22% recall after 1,200 updates,
and four full neural groups reached only 58.45% with zero exact recovery.

The independent projected control reached 96.47% recall and 79.17% exact
sets, but missed the frozen KKT and objective-gap tolerances by small,
reproducible margins. The campaign is therefore invalid for treatment
selection despite strong optimizer evidence. The only immediate successor is
a preregistered repair of that numerical control. Reuse all other frozen
evidence; do not rerun discovery work or open a model treatment until the
repair passes.

ADR 0104 increased only that control's iteration ceiling from 10,000 to
100,000. It passed the decision-tolerance KKT (`9.194e-9`) and objective-gap
(`8.566e-8`) gates, but only 12 of 24 groups reached the stricter stop rule
and three target selections still disagreed with the analytic solution.
Therefore it closed as `projected_control_repair_invalid`.

The immediate successor is one independent arbitrary-precision reconstruction
of the same frozen 24-group optimum and selector. It must use a separate
high-precision derivation, exact deterministic ordering, fine-grained dynamic
cross-host work claiming, and cross-host replay. It must not increase projected
iterations again, relax a tolerance, rerun frozen AdamW or neural evidence, or
open an optimizer/model treatment before the numerical control is valid.

ADR 0105 completed that campaign operationally but invalidated itself
scientifically. Its preregistration converted expected ranks as integers,
while the frozen inputs are fractional float64 values. The Decimal solver
reached normalization residual `1.81e-94` and KKT violation `1.05e-95`, but
23 of 24 objectives changed. The dynamic queue nevertheless completed 48
origin/replay tasks in 5.97 seconds, peaked at 23 concurrent processes, and
recorded zero idle process-slot seconds while compatible work was queued.

The immediate successor is one corrected replay using `Decimal.from_float` for
every expected-rank input. Keep the same active-set derivation, precision,
groups, gates, dynamic scheduler, source/replay checks, and sealed boundaries.
Do not reinterpret ADR 0105, alter thresholds, or open an optimizer/model
treatment until the exact-float control passes.

ADR 0106 passed that correction completely. It recovered every one of 851
target slots and all 24 target sets, with normalization residual `1.65e-94`,
KKT violation `9e-96`, objective mismatch `1.89e-14`, and offset mismatch
`4.69e-13`. All 24 cross-host replays were bit-identical. The mechanical
classification is `frozen_optimizer_hyperparameters_insufficient`.

The immediate successor is exactly one analytically calibrated local optimizer
mechanism. First demonstrate that it closes the frozen free-residual gap, then
test the same unchanged mechanism on bounded full-model local continuation.
Do not sweep multiple optimizer families, change representation, launch a full
trainer, or open validation, sealed test, or gameplay until this local
mechanism passes its preregistered gates.

ADR 0107's calibrated monotone AdamW passed the free-residual strength gate:
96.24% recall and 70.83% exact sets at 120 updates, then 96.59% recall and
79.17% exact sets terminally. Five groups reached float32 numerical saturation
after 767-1,105 accepted updates and could not accept another monotone proposal
within 16 halvings. Because the preregistration required 1,200 accepted
updates, Stage 1 closed as `calibrated_optimizer_pipeline_invalid`; neural work
did not launch.

ADR 0108 repaired only that stop rule on groups 0, 2, 8, 14, and 23. Every
group exhausted 16 finite proposals below `1e-7` with zero measurable
improvement and finite optimizer state. All five cross-host replays were
bit-identical. Recombining them with the frozen 19 groups retained 96.59%
recall and 79.17% exact sets, so Stage 1 closed as `free_stage_passed`.

ADR 0109 executed that neural stage. Groups 0, 1, and 3 numerically converged
after 49, 6, and 1 accepted updates; group 2 reproducibly failed the frozen
completion rule after 8. All four replays were bit-identical and all resource
gates passed, but the pipeline classification is
`calibrated_optimizer_pipeline_invalid`. Terminal descriptive recall was only
32.39% with zero exact sets, and no group reached 120 exposures.

ADR 0110 completed that no-training forensic audit. Six accepted-rate
histories fit every frozen group 2 statistic, and all imply the same failed
step: proposals descended from `9.896e-5` to `3.020e-9`. The only rejected
condition was an improving diagnostic proposal below the optimizer's `1e-8`
acceptance floor. Domain-consistent completion therefore reclassifies group 2
as numerically converged without a rerun.

The corrected neural pipeline passes, exposing the substantive result:
32.39% terminal recall and zero exact sets. The classification is
`public_observable_representation_insufficient`.

ADR 0111 preregisters that single public-observable representation treatment:
a zero-initialized exact rotation-canonical local-geometry adapter over the
frozen selected model. It uses the calibrated optimizer and all other inputs
frozen, runs four distinct group origins across john1-john4, and replays every
group cross-host after origin evidence is complete. Do not launch a full
trainer, representation sweep, second treatment, validation, sealed test, or
gameplay.

ADR 0111 completed with every pipeline gate passing, 71.13% terminal recall,
and 25% exact target sets. Group 2 fit exactly while groups 0, 1, and 3
remained below the gate after numerical convergence. The classification is
`calibrated_local_geometry_insufficient`; the single representation treatment
is exhausted and no full trainer is authorized.

Before any new learned mechanism, run a no-training forensic over the retained
ADR 0111 reports and frozen group inputs. Measure residual clipping, duplicate
observable rows with conflicting targets, target misses by action relation,
and whether the converged adapter score ordering has remaining feasible
bounded corrections. Use that evidence to preregister one mechanistically
distinct successor or to close this frontier-learning route. Do not launch a
second representation treatment, full trainer, validation, sealed test, or
gameplay from ADR 0111 alone.

ADR 0112 completed that forensic. Every one of the 11,087 local adapter input
rows was unique, no selected-base residual was saturated, and the exact
candidate-independent bounded interval ceiling recovered 100% recall and 100%
exact sets in every group. The classification is
`parameterized_fit_or_optimizer_insufficient`.

One same-representation mechanistic control is now authorized. It must hold
the ADR 0111 architecture, inputs, bounds, groups, rotations, and optimizer
fixed while replacing only the diffuse expected-rank objective with direct
balanced target-membership supervision. This separates objective/gradient
dilution from shared adapter capacity. Do not launch a second representation,
full trainer, validation, sealed test, or gameplay.

ADR 0113 completed that control. Balanced target-membership supervision
perfectly fit the 324-candidate group but converged at only 40.54%-50% recall
on the three 2,975-4,368-candidate groups. Aggregate recall was 59.86% with
25% exact sets. The classification is `shared_adapter_capacity_insufficient`;
the 192-wide shared local adapter is closed.

Do not continue this route through another objective, optimizer, or width
sweep. The next policy-learning design must scale its representational
authority with candidate structure, such as decomposed action factors,
retrieval, or hierarchical proposal construction, and must be separately
preregistered from the accumulated evidence. A full trainer, validation,
sealed test, and gameplay remain closed until a new local sufficiency gate
passes.

ADR 0114 completed the required structural reframe. Conditional hierarchical
retrieval with widths `16 / 32 / 8` achieved 99.27% train recall and 99.18%
validation recall, 95% validation exact target sets, 100% validation winner
retention, and 482 mean validation proposals. Independent retrieval at the
same widths reached only 94.66%, proving that draft-conditioned tile retrieval
and draft+tile-conditioned wildlife retrieval are necessary.

One learned hierarchical retrieval pilot is authorized. It must preserve the
exact factor partition, conditional order, `16 / 32 / 8` budgets,
champion-frontier inclusion, complete legal reachability, open splits, and
top-64 selector. Train the three retrieval stages as separate conditional
problems so model authority scales with factor count rather than complete
action count. A full policy/value trainer, sealed test, and gameplay remain
closed until the learned pilot approaches the 99.18% oracle ceiling and passes
the Phase 2 recall and regret gates.

Execute this pilot as a throughput portfolio, not four copies of one training
run. First produce and fingerprint the shared open-data inputs as a divisible
four-host CPU job. Once those immutable inputs exist, the draft, conditional
tile, and conditional wildlife retrievers are three distinct MLX experiments
and should train concurrently on separate Macs when their measured resource
profiles permit it. The fourth Mac must advance nonduplicative work that
shortens the same decision path: feature/label integrity audits, deterministic
baselines, validation and regret tooling, error decomposition, or a separate
preregistered mechanism. As each stage finishes, immediately backfill its host
from the ready queue and begin dependency-ready integration checks; do not wait
at a three-stage barrier. Replicate a stage only after its own pilot gate
passes or when seed or host sensitivity is the explicit scientific question.
The pilot report must include useful CPU utilization, idle time while
compatible work was queued, per-host work completed, duplicate-compute
fraction, and decisions reached per elapsed hour.

ADR 0115 completed with every pipeline gate passing and the mechanical
classification `hierarchical_proposal_insufficient`. Draft, tile, and wildlife
validation factor recall reached 92.84%, 66.57%, and 100.00%. The integrated
proposal retained 72.48% of validation target actions and 92.08% of winners;
the learned top 64 retained only 18.14% of targets and 58.75% of winners.

The tile stage is the decisive bottleneck. Exact train and validation
model-input collisions were zero. On the eight widest supervised train
queries, the top-32 membership gradient opposed the combined
regression/listwise gradient at cosine `-0.738910`, while the combined
auxiliary norm exceeded the membership norm. An oracle reranker over the union
of learned and screen-prior top 32 reached only 78.29% validation recall, so a
score blend cannot close the gap.

ADR 0116 completed with every pipeline gate passing and the mechanical
classification `target_only_tile_objective_insufficient`. Removing the
regression and listwise terms improved tile recall to 77.21% train and 70.59%
validation, but integrated proposal recall fell to 71.83% and winner retention
to 89.58%. The objective conflict was real but not the dominant remaining
limit. Boundary-only BCE is closed.

ADR 0117 completed with every pipeline gate passing and the classification
`full_data_scale_or_optimization_insufficient`. The unchanged ranker fit the
16-query cohort exactly in 200 updates and the 256-query cohort exactly in
3,200 updates. The larger attention model was slower and ended at 98.83% exact
recovery. Explicit relational attention is closed as the immediate lever.

The anatomy showed only a 0.59-point validation penalty from permuting query
context and no penalty from permuting parent state, while descendant summaries
accounted for 27.47 points. More importantly, ADR 0116's full-cache train
recall improved monotonically through epoch 20, and the medium cohort required
roughly 200 passes for exact fit. The immediate successor must therefore hold
architecture, features, objective, width, seed, optimizer, and data fixed and
increase only full-cache training exposure. One Mac owns the sole MLX origin;
the other three must run cross-host replay preparation, trajectory and
resource checks, integration tooling, tests, and independent
dependency-ready experiments. Those preparation tasks are short-lived and may
not become an excuse for three hosts to wait on the trainer: as each completes,
its host must immediately claim the highest-value compatible job from the
research queue. Duplicate origins and epoch sweeps remain prohibited.

ADR 0119 completed the required backfill portfolio while the john2 origin
continued. Three independent open-data decisions finished on john1, john3, and
john4 in 31.91 seconds with zero duplicate discovery compute. Uniform query
sampling did not pass its replicated mismatch gate. Draft, tile, and wildlife
score dispersion differed by 5.64x on train and 9.42x on validation. Most
decisively, an oracle-factor top-64 selector retained every validation winner
with zero regret but only 74.72% of the full target set.

The post-ADR-0118 branch is therefore mechanical. If extended exposure is
insufficient, run one optimizer-schedule treatment; target-mass resampling and
another uniform epoch extension are closed. If extended exposure is
sufficient, freeze the proposal and train a normalized complete-action
selector; fixed factor aggregation and raw stage-score summation are closed.

ADR 0118 completed as `extended_exposure_tile_insufficient`. Its sole
200-epoch origin reached 99.80% train recall but only 67.75% validation recall;
mixed and integrated winner retention were both 83.75%. Every pipeline and
replay gate passed. ADR 0120 is therefore the active mechanical successor. It
keeps epochs 1-20 at `3e-4`, applies one cosine decay to `3e-6` over epochs
21-200, and changes nothing else. This is the final authorized exposure,
sampling, or optimizer-schedule treatment for the conditional pointwise tile
ranker.

ADR 0121 completed concurrently on the three nontraining Macs. It found no
exact observable-label contradictions and no material input covariate shift.
Instead, fixed-rate late training improved median normalized train boundary
margin by 1.7033 while validation worsened by 1.1260, expanding the
train-validation gap by 2.8293. If ADR 0120 fails, the next separately
preregistered mechanism must therefore be structural regularization. It may
not be another exposure, sampling, or learning-rate schedule variant.

ADR 0122 identified the concrete structural target. Deterministic
within-query block permutations assigned specialization contributions of
0.2446 to local geometry, 0.1056 to descendant summaries, and 0.0457 to tile
identity. If ADR 0120 fails, the next treatment must regularize only the
local-geometry block before considering broader capacity reduction.

ADR 0123 calibrated the contingent rate. Ten percent corruption removed
11.06% of the extended train-validation gap, 25% removed 24.9948% and missed
the frozen gate, and 50% removed 48.39% with under 0.75 points of validation
damage on both checkpoints. If ADR 0120 fails, the next training pilot must
use 50% local-geometry dropout without a rate sweep.

ADR 0124 now freezes that contingent treatment while ADR 0120 continues. It
inherits the exact ADR 0120 model, objective, optimizer, 200-epoch schedule,
seed, and clean inference contract, changing only training-time local-geometry
columns `[8,188)` for the deterministic hash-ranked half of each query.

The first implementation preflight correctly rejected a 4.39 GiB,
367.66%-overhead path. ADR 0125 repaired only shard retention, the redundant
query copy, and exact selection mechanics. The repaired four-host preflight
passed with the original selection digest preserved, 21.16% preparation
overhead, 1.89 GiB peak RSS, zero swaps, and combined scientific BLAKE3
`2b6eacd04b490e3305e10c4603bf42363fdb78f1a8d21cd7f766eeb2441c99e3`.
No ADR 0124 training is authorized unless ADR 0120 finishes valid and
insufficient.
