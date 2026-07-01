# R2-MAP Expert-Iteration Research Implementation Plan

> **Authority:** [`CASCADIA_V2_GOAL.txt`](../../CASCADIA_V2_GOAL.txt) is the
> source of truth. This document describes the executable R2-MAP research flow
> and may not override its scientific, storage, or orchestration boundaries.

Date: 2026-06-20  
Status: complete; protected fixed-250 result negative, no promotion or expert iteration  
Campaign: `r2-map-expert-iteration-v1`

Architecture basis:
[`R2_MULTITASK_ACTION_PERCEIVER_IMPLEMENTATION_PLAN.md`](R2_MULTITASK_ACTION_PERCEIVER_IMPLEMENTATION_PLAN.md)

## Outcome

Determine whether the width-192 exact-R2 fixed-latent Perceiver can beat the
qualified exact-NNUE control. First complete the protected strength-blinded
20-pair smoke, then one preregistered 250-pair paired comparison. Do not start
expert-iteration generation unless that comparison is positive and a later
goal explicitly authorizes continuation.

The implementation is exhaustive: every legal action is materialized and
scored. No MCE, action pruning, approximate serving, remote inference, or
distributed training is used in this initial result.

## Terminal result

The protected comparison completed with 250 independent pairs and 500 physical
games. R2-MAP scored 91.604 versus 97.468 for the qualified exact NNUE K32/R600
control. The paired delta was -5.864 (SE 0.281, 95% CI [-6.414, -5.314]);
wins/ties/losses were 19/9/222. Every pair passed rules, identity, replay,
shutdown, Pinecone-conservation, memory, and zero-swap checks, and every
scheduler job succeeded in one attempt.

The result is negative under the preregistered rule. R2-MAP is not promoted,
expert iteration is not authorized, and primary research returns to NNUE. The
complete report is at
`/Users/johnherrick/cascadia-bench/r2-map-v1/gates/development-v8/reports/focal-benchmark-complete.md`;
the independent audit is alongside it as `fixed-250-independent-audit.json`.

## Execution architecture

John1 is the sole source, image-build, active-artifact, aggregation, dashboard,
and native MLX authority. A private Bacalhau v1.9.0 fabric provides fungible CPU
capacity on john1, john2, and john3. John4 stays visible on the fleet dashboard
but is never scheduled by this campaign.

The scheduler advertises 9 CPUs on john1 and 10 CPUs on each of john2 and
john3, for 29 allocatable CPUs. Workloads declare measured per-item resources;
callers never pre-divide work by host, parity, or a fixed partition.

```text
John1 source tree
  -> one tested linux/arm64 image
  -> private OCI registry, immutable digest
  -> cluster.map / submit_map
  -> Bacalhau chooses john1, john2, or john3
  -> execution-specific MinIO result
  -> validated atomic import on John1

John1 native MLX/Metal training
  -> verified checkpoint
  -> portable frozen candidate input
  -> CPU/reference Bacalhau gate
```

The normal workload path contains no SSH, rsync, Docker archive fanout, worker
source checkout, worker build, host affinity, shared writable tree, or custom
host executor. SSH is reserved for fabric installation, diagnosis, and
recovery. Standard Bacalhau Docker execution is sufficient for this trusted
private cluster.

Use only the main Codex agent. There are no host-owner or nested subagents.

Operational implementation and API details live in
[`docs/cluster_orchestrator.md`](../cluster_orchestrator.md). The source design
and migration acceptance criteria live in [`orchestrator-rewrite.md`](../../orchestrator-rewrite.md).

## Non-negotiable scientific invariants

1. Exactly one newest-model seat appears in each iterative game.
2. The other three seats are greedy or frozen historical policies. The newest
   checkpoint never appears twice.
3. Candidate training uses only the immediately preceding generation round.
   Historical checkpoints are opponent policies, not imitation targets.
4. Every legal action is enumerated, materialized, scored, and mapped back
   exactly once.
5. Inference uses immutable local checkpoint bytes. No per-move network service
   or cross-node synchronization exists.
6. Seed-range work and pair/game work items are disjoint and deterministic.
   Physical node placement and manual partitions are not part of scientific
   identity.
7. MLX/Metal training and checkpoint replay occur only on john1.
8. During native training, no next-round experience is generated. Available
   remote CPU may benchmark only the previously promoted checkpoint.
9. A crash may lose unfinished work only. Verified checkpoints and imported
   result shards are immutable and restart-safe.
10. Hidden order, future refill, policy identity, node identity, campaign ID,
    and dataset split are forbidden model inputs.
11. Promotion requires replay, score, Pinecone, checkpoint, backend-parity,
    latency, memory, and zero-swap integrity.
12. Protected seeds and blinded smoke strength outputs remain unopened until
    their registered transition permits access.

## Model and learning contract

The shared model is the exact-R2 multitask action Perceiver:

- width 192;
- four attention heads;
- 16 fixed latent tokens;
- exact occupied/frontier/component/motif state tokens;
- exact public market and public opponent state;
- complete parent-state and full-afterstate action representation;
- a scalar score-to-go head;
- score-component heads;
- bootstrap and market-decision policy heads;
- public opponent-next-action and market-survival auxiliaries; and
- a protected 24-dimensional multitask subspace with primary-gradient
  authority.

The primary observed target at a focal decision is terminal base score minus
current realized score. Component targets preserve bear, elk, salmon, hawk,
fox, five terrain components, and Pinecone earned/spent/remaining accounting.
The 100,000-game greedy bootstrap may add deterministic action-imitation
supervision; on-policy rounds retain observed-return supervision.

Only public information enters model tensors. D6 serialization, legal-action
order, tensor shapes, candidate cardinality, and selected-action mapping are
versioned and checked at the Rust/Python boundary.

## Data contract

The bootstrap contains exactly 100,000 deterministic, complete, four-player
greedy games with disjoint seed ranges. Each game contains 80 decision records;
the training projection takes the deterministic focal seat
`global_game_index mod 4`, yielding 20 focal records per game.

Every compact shard binds:

- campaign, round, work-item identity, and seed interval;
- image and source identity;
- policy/checkpoint identities for all seats;
- rules, protocol, feature, and inference identities;
- replay events and terminal state hash;
- score anatomy and Pinecone conservation;
- game and record counts; and
- file byte counts, SHA-256, and BLAKE3.

Worker outputs are execution-local. John1 accepts them only after the cluster
output manifest and every declared checksum validate. Reducers reject missing,
duplicate, extra, overlapping, or identity-drifted records.

## Phase state machine

For any later authorized expert iteration `r`:

```text
promoted C[r]
  -> G[r]: one approximately 45-minute topology-free generation window
  -> verify and atomically install every complete disjoint shard
  -> T[r]: native John1 MLX training from G[r] only
     || B[r-1]: topology-free benchmark of the prior checkpoint only
  -> verify/freeze candidate T[r]
  -> topology-free paired gate against C[r]
  -> promote or retain incumbent
```

Generation never overlaps training. Training is never distributed. No node
waits for another during a game. Barriers exist only at dataset install,
training start, candidate freeze, and promotion.

## Native MLX checkpoint and recovery contract

The trainer runs directly on john1 from the exact compact bootstrap index. It
writes branch-aware loss records and atomic checkpoints containing model,
optimizer, state, fixed-prediction panel, and a canonical manifest.

Each selectable checkpoint must pass an independent replay that proves:

- manifest and file hashes;
- exact prediction-panel replay;
- exact next-batch and optimizer resume;
- finite losses, parameters, and gradients;
- zero process swaps and zero positive system swap delta; and
- monotonic branch-aware loss lineage.

`last_verified.json` advances only after those checks. If uncheckpointed loss
events exist after the last verified checkpoint, recovery creates a new logical
branch; it never truncates or rewrites history. The final candidate is valid
only when the training receipt, best-validation checkpoint, terminal step,
last-verified pointer, and independent verification all identify the same
checkpoint.

The portable freeze produced by `tools/r2_map_freeze_candidate.py` additionally
requires MLX-versus-NumPy checkpoint parity before CPU/reference evaluation.

## Paired benchmark protocol

The gate is frozen in
[`reports/r2-map-bootstrap-cross-architecture-250-preregistration-v1.md`](reports/r2-map-bootstrap-cross-architecture-250-preregistration-v1.md).

Each pair runs the same seed, focal seat, greedy opponent field, rules, and RNG
domains twice:

1. candidate R2-MAP in the focal seat;
2. qualified exact-NNUE K32/R600 control in the focal seat.

The focal seat rotates by pair index modulo four. Execution order alternates by
pair index. Candidate and control never appear in opponent seats. Every pair is
one independent `pair-NNNN` work item; Bacalhau decides physical placement,
packing, retry, and rescheduling.

Required outputs include absolute arm statistics and paired deltas for:

- base total with mean, standard error, 95% interval, P10/P50/P90, and
  win/tie/loss;
- aggregate wildlife and bear/elk/salmon/hawk/fox;
- aggregate habitat and mountain/forest/prairie/wetland/river;
- Pinecones earned, independent-draft spend, paid-wipe spend, total spend,
  remaining, and free replacements;
- focal decision latency, throughput, RSS, swap delta, and clean shutdown; and
- replay and Pinecone conservation.

The 20-pair smoke exposes only integrity, resource, completion, and throughput
fields. The fixed-250 result is:

- **positive:** paired mean is positive and its 95% interval excludes zero,
  with all preregistered integrity/resource gates passing;
- **negative:** the interval is wholly nonpositive or a hard gate fails; or
- **inconclusive:** every other valid outcome.

There is no outcome-driven sample-size extension.

## Executable Bacalhau gate

`tools/r2_map_bacalhau_gate.py` is the only active distributed gate launcher.
It:

1. validates the v4 scheduler-managed contract and opponent field;
2. builds a deterministic content-addressed input archive containing the
   campaign, portable candidate, backend parity receipt, and exact NNUE weights;
3. submits one `ContainerInput` per registered pair with explicit CPU, memory,
   disk, and timeouts, while durably recording the whole logical map before the
   first submission;
4. derives a scheduler-capacity admission window from connected advertised
   resources, releases only that many outstanding jobs, and advances the
   window after terminal work without selecting a node;
5. lets Bacalhau admit, place, retry, reschedule, and publish each released
   execution;
6. reconnects by request ID and imports validated outputs;
7. verifies immutable contract/field identity across every work item;
8. merges only the declared pair receipts and work-item summaries; and
9. submits aggregation as another immutable container job before atomically
   installing the final report on john1. The aggregate request ID binds the
   assembled campaign archive SHA-256, and its Bacalhau job name hashes the
   complete request ID to prevent cross-stage update collisions.

After the terminal candidate and canonical image are both frozen, materialize
each protected stage with `tools/r2_map_gate_control.py prepare-bootstrap`.
The command requires `--image-digest`, `--candidate-freeze-receipt`, and
`--exact-weights`; its raw contract therefore commits every pair receipt to the
immutable image, frozen candidate receipt, qualified exact weights, and exact
opponent-field bytes. The Bacalhau result receipt separately records the
request, item, job, accepted execution, item-spec, output-manifest, input, and
image identities. Open the development domain only after the strength-blinded
smoke report passes every registered gate. Development materialization also
requires `--smoke-campaign-directory`; the controller emits and binds a
content-addressed smoke-admission receipt and otherwise refuses to read the
250-pair protected domain.

While a gate is active, the controller persists reconnect-safe Bacalhau
observations under client state: planned, pending-admission, queued, running,
and terminal work-item states plus per-node and aggregate
allocated CPU over the 9/10/10 pool. The final scheduler-provenance report
includes the observation hash, time-weighted mean and peak utilization,
per-node allocation, accepted execution identities, and retry count.
The completed dashboard projection retains that campaign mean/peak utilization
and retry total after the live Bacalhau allocation returns to idle, so the
final benchmark cannot look like an unused cluster.

Example after terminal candidate freeze and image publication:

```bash
PYTHONPATH=python \
python3 tools/r2_map_bacalhau_gate.py \
  --stage smoke \
  --image '100.110.109.6:5000/cascadia/r2-map@sha256:...' \
  --gate-directory /Users/johnherrick/cascadia-bench/r2-map-v1/gates/smoke-input \
  --candidate-freeze /Users/johnherrick/cascadia-bench/r2-map-v1/frozen/candidate \
  --exact-weights /Users/johnherrick/cascadia/nnue_weights_v4opp_modal_iter3.bin \
  --state-directory /Users/johnherrick/cascadia-bench/orchestrator/client-state \
  --artifact-directory /Users/johnherrick/cascadia-bench/orchestrator/accepted \
  --campaign-directory /Users/johnherrick/cascadia-bench/r2-map-v1/gates/smoke
```

The development invocation is identical except for `--stage development` and
its registered input/output directories. The dashboard watcher reads imported
pair receipts and optional Bacalhau request state. It never queries a worker
over SSH and never claims that a scientific pair belongs to a node.

## Storage and lifecycle

Active authoritative research remains under:

```text
/Users/johnherrick/cascadia-bench/r2-map-v1
```

The Bacalhau scheduler, registry, MinIO, and client state remain under:

```text
/Users/johnherrick/cascadia-bench/orchestrator
```

The authored source remains at `/Users/johnherrick/cascadia`. The active root
has a 64-GiB ceiling and must preserve at least 64 GiB free on john1 after an
admitted operation. Only dependency-closed old research may be checksummed,
copied, verified, and then removed into the john2 cold archive. No process uses
`/Volumes/John_1`.

## Performance program

Correctness precedes optimization. Measure the reference path, then optimize
only observed hot spots. Allowed changes include exact incremental R2 state,
encoding shared parent state once, batching independent games/candidates,
removing transport copies, keeping models resident, and CPU/GPU pipelining.

Every optimization must preserve:

- legal action count and ordered identity;
- exact R2 tensor digests or the registered numerical tolerance;
- zero selected-action disagreement on the frozen parity corpus;
- exploration, replay, terminal state, and score anatomy;
- no pruning or silent fallback;
- at most 4 GiB process RSS and no positive swap growth; and
- no per-move checkpoint load or graph recompilation.

## Work packages

- **W0 contracts/reference panels:** versioned public-information, replay,
  implementation-binding, and benchmark contracts. Complete for the active
  bootstrap binding.
- **W1 simulation/data:** heterogeneous seats, deterministic seed leases,
  compact experience, replay, and Pinecone accounting. Complete for bootstrap.
- **W2 model/dataset:** exact-R2 model, packed dataset path, losses, and public
  information validators. Complete for bootstrap training.
- **W3 training/recovery:** native MLX trainer, checkpoint verification,
  branch-safe resume, loss telemetry, and portable freeze. Complete.
- **W4 serving/gameplay:** exhaustive local serving and direct action selection.
  Complete with terminal-checkpoint qualification and backend parity.
- **W5 benchmark/dashboard:** paired focal statistics, pair work-item receipts,
  blinded projection, scheduler visibility, and report aggregation. Complete.
- **W6 orchestration:** Bacalhau fabric, typed client, registry, MinIO, legacy
  queue freeze, topology-free R2 launcher, and validated import. The 9/10/10
  compute deployment, v4 work-item cutover, durable capacity backpressure,
  registry/MinIO/image canaries, and validated result import are complete.
- **W7 initial result:** freeze candidate, run blinded smoke, run fixed-250 if
  smoke passes, and classify. Complete; negative, no promotion.

## Verification matrix

### Model and data

- exact shapes, padding inertness, full action cardinality, D6 round-trip;
- hidden-order and future-refill mutation independence;
- finite active-head losses and primary-gradient authority;
- Rust/Python/NumPy/MLX checkpoint and selected-action parity;
- exactly one newest seat and deterministic opponent field;
- no seed collision across work items or domains;
- replay-to-terminal identity and exact Pinecone conservation.

### Checkpoint and faults

- save/load/next-batch resume parity;
- interruption at file write, fsync, rename, verification, and pointer updates;
- one-byte corruption rejection for every checkpoint component and loss stream;
- incomplete checkpoint exclusion and last-verified recovery;
- protected/incumbent checkpoint retention.

### Benchmark

- candidate/control identity differs only by focal policy;
- one-item-per-pair identity, disjointness, and complete coverage;
- strength-blinded smoke projection;
- fixed-250 statistics and classification;
- interrupted pair resume and order-independent aggregation;
- missing, duplicate, extra, and tampered pair rejection.

### Orchestration

- exact connected membership john1-john3 with no john4 scheduling;
- 9/10/10 advertised CPU capacity for a 29-CPU scheduler pool;
- immutable digest and topology/secret-field rejection;
- ordered map results, partial failures, cancellation, and reconnect;
- idempotent submission and conflicting-spec rejection;
- deterministic failures execute once; actual attempts and retries are
  reported against Cascadia's recorded application-attempt contract;
- worker loss/reschedule, scheduler restart, and unschedulable resources;
- streaming large-object input/output, safe extraction, manifest checksums,
  duplicate-success arbitration, and atomic import;
- 100-job scale uses all three compute nodes;
- dashboard fleet plus scheduler state on desktop and mobile;
- legacy queue remains frozen and read-only.

## Stop and success rules

Stop immediately on replay, legality, identity, score, Pinecone, protected-data,
checkpoint, backend-parity, memory, swap, or no-pruning failure. Preserve the
last verified state and record the exact blocking evidence.

For this initial architecture question, success means a valid positive
fixed-250 result. A negative or inconclusive result is also a complete research
outcome and does not authorize expert iteration.

The broader program goal remains a frozen promoted policy with development mean
at least 100.25 over at least 500 fresh focal games and mean base score at least
100.000 on the untouched 1,000-game final domain, under all integrity gates.

## Definition of done

This plan is complete when:

- terminal native training and independent checkpoint verification pass;
- the portable candidate and immutable image digest are frozen;
- registry, MinIO, and all three Bacalhau workers pass live health;
- unit, integration, worker-failure, scheduler-restart, deterministic-failure,
  large-artifact, scale, and cross-node parity tests pass;
- the blinded 20-pair smoke passes without revealing strength;
- the fixed 250-pair comparison has one immutable report and classification;
- the dashboard truthfully displays fleet, MLX, scheduler, and gate state;
- active artifacts and source remain within their authorized roots; and
- the next action is unambiguous: authorize expert iteration, revise the
  architecture under a new contract, or close the branch.
