# R2-MAP Bootstrap Cross-Architecture Gate Preregistration

Status: completed; negative result, no promotion

Date: 2026-06-19

## Recorded outcome (2026-06-20)

The registered smoke passed all gates. The fixed comparison then completed
exactly 250 pairs and 500 physical games. Candidate/control means were
91.604/97.468; paired delta -5.864, standard error 0.281, 95% confidence
interval [-6.414, -5.314], and wins/ties/losses 19/9/222. The outcome is
negative under the rule below. All pair jobs completed in one attempt with
zero swap growth; the independent audit verified exact seed/seat pairing,
AAAAA/no-bonus rules, greedy opponents, candidate exclusion from opponent
seats, replay/Pinecone conservation, clean shutdown, and unique scheduler
provenance.

Authoritative artifacts:

- `/Users/johnherrick/cascadia-bench/r2-map-v1/gates/development-v8/reports/focal-benchmark-complete.md`
- `/Users/johnherrick/cascadia-bench/r2-map-v1/gates/development-v8/reports/fixed-250-independent-audit.json`
- `/Users/johnherrick/cascadia-bench/r2-map-v1/gates/development-v8/reports/scheduler-provenance.json`

R2-MAP is not promoted and expert iteration does not begin. Primary research
returns to NNUE.

## Question

Does the from-scratch iteration-0 R2-MAP bootstrap policy outperform the
qualified exact NNUE K32/R600 policy when each controls the same focal seat
against an otherwise identical frozen greedy field?

This gate answers only the initial architecture question. It does not
authorize expert iteration, promotion, or a claim that R2-MAP has reached the
100-point target.

## Frozen Policies

- Candidate: the verified terminal checkpoint of
  `r2-map-bootstrap-iteration0-value-v1`, served with exhaustive legal action
  enumeration, exact afterstate score plus predicted score-to-go, deterministic
  argmax, NumPy CPU reference inference, and no pruning.
- Before protected inputs are opened, the terminal candidate must pass a
  content-addressed MLX-versus-NumPy fixed-panel and market-decision parity
  check. Every worker verifies that receipt against the serving bundle's
  checkpoint manifest, model weights, and checkpoint verification ID.
- Control:
  `canonical-action-legacy-exact-mlx-v1-k32-r600-lmr-no-paid-prelude`, executed
  through the native exact sparse-NNUE evaluator with K32, R600 sequential
  halving, `MCE_LMR=1`, and `MCE_DIVERSE_PREFILTER=1`.
- Control weights BLAKE3:
  `9e1d568693274fc537ac4f6d6f729abb1ee8da8330a78d1f78a1f62b733de400`.
- Native/MLX rollout-wave parity report BLAKE3:
  `ff10f31941e3a49b6dc9acfc06c34a1d0e7fba5e680ad42494af283c3aafc4dc`.

The parity report is mandatory input. The runner refuses a changed control
weights file, changed parity report, non-R600 budget, or missing qualified
legacy environment setting.

## Paired Design

Each pair contains two physical four-player AAAAA games with no habitat
bonuses:

1. candidate R2-MAP in the registered focal seat;
2. qualified exact NNUE in that focal seat.

The two games share the exact game seed, focal seat, three greedy opponent
identities, opponent RNG domains, rules, and implementation binding. The focal
seat rotates by pair index modulo four. Candidate and control execution order
alternates by pair index. Candidate and control never appear in an opponent
seat. Every completed pair is written atomically and is independently
resumable.

The all-greedy opponent field is intentional: it isolates the architecture
change and avoids paying R600 for three nondifferential opponent seats. The
result is a paired focal-policy comparison, not a reproduction of the
historical all-NNUE 95.744 absolute-score benchmark.

## Stages and Protected Seeds

1. Run the unopened protected 20-pair strength-blinded smoke domain. Aggregate
   only integrity, resource, completion, Pinecone conservation, and throughput
   fields. Do not inspect or use strength outputs.
2. If and only if the smoke completes with 40 valid games, clean process
   shutdown, peak process RSS at or below 4 GiB, no positive swap growth,
   valid replay/Pinecone accounting, and no contract drift, run the unopened
   fixed 250-pair development domain.
3. Submit every pair as one independent `pair-NNNN` work item: 20 items for the
   smoke and exactly 250 for the development comparison. There is no parity,
   host, or manual shard partition. Bacalhau chooses any available john1-john3
   compute node and owns admission, packing, retry, and rescheduling. No
   per-game or per-move cross-node coordination is permitted.

The complete ordered logical request is persisted before its first job is
released. Because pinned Bacalhau v1.9 has a finite over-subscription queue,
the topology-free client derives the safe outstanding-job window from live
advertised CPU, memory, and disk capacity and the pair's declared resources.
Each terminal job releases the next pair in index order. This flow control does
not bind a pair to a host, alter any scientific identity, or weaken Bacalhau's
exclusive placement/retry authority. Restart recovery reattaches by request,
item, and specification hash and never duplicates an accepted pair.

The development input controller requires the completed smoke campaign and
refuses materialization unless its strength-blinded report has exact 20-pair
coverage, 40 valid physical games, clean shutdown, bounded RSS, zero positive
swap growth, Pinecone conservation, and complete scheduler provenance. Its
admission receipt is content-addressed into the development contract.

The protected registry and private seed material already exist below
`control/protected-seeds-v2`. Gate inputs may be materialized only after the
terminal candidate checkpoint is frozen.

## Decision Rule

The fixed 250-pair report is authoritative and must include both absolute arms,
paired score/category/Pinecone deltas, P10/P50/P90, standard error, 95%
confidence interval, win/tie/loss counts, latency, throughput, RSS, and swap.

- Positive: mean paired base-score delta is greater than zero and the 95%
  confidence interval excludes zero on the positive side, with all integrity
  and resource gates passing.
- Negative: the 95% confidence interval is entirely nonpositive, or an
  integrity/resource gate fails.
- Inconclusive: every other statistically valid result.

Only a positive result establishes that this bootstrap architecture beat the
qualified NNUE control in the registered field. No expert-iteration generation
begins under this goal, regardless of outcome.

## Execution Boundary

John1 is the source of truth. After training ends, John1 verifies and freezes
the candidate, builds one arm64 Docker image, pushes it to the private registry,
and submits only its immutable digest. Bacalhau pulls and schedules that image
across the john1-john3 compute fabric. Candidate/control inputs are immutable
content-addressed objects; execution-specific outputs publish to MinIO and are
validated and atomically imported by John1. John4 is not used.

The raw focal contract binds the immutable image digest, candidate-freeze
receipt hash, qualified exact-weight hash, and opponent-field hash. Each
accepted work-item receipt additionally binds request, item, Bacalhau job,
accepted execution, item specification, input archive, and validated output
manifest identities. Retry or rescheduling may change physical execution
provenance but cannot change scientific identity.

The Bacalhau fabric advertises 9 allocatable CPUs on john1 and 10 on each of
john2 and john3. Those values are scheduler capacity, not scientific topology;
pair identity and result aggregation are independent of physical placement.

The cluster dashboard must show training until checkpoint freeze, then image
distribution, blinded smoke progress, fixed-250 progress, and the final
classification. It must never expose protected seeds or smoke strength.
