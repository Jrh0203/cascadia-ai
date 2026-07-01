# Cascadia V3 Radius-7 Stockfish-Style NNUE Campaign

Status: Part 1 implementation source of truth. Part 2 is checksum-gated and must not begin without John’s explicit approval.

## Scientific objective

Build a CPU-efficient four-player AAAAA Cascadia policy, with habitat bonuses disabled, whose all-V3 mean is at least 100 and whose game-block 95% lower confidence bound is also at least 100. The protected control is the manifest-pinned qualified exact NNUE K32/R600 policy.

## Hard campaign boundary

The campaign has two parts:

1. Part 1 builds, tests, profiles, optimizes, and smoke-qualifies the complete system using only the permanently excluded `engineering_smoke` domain.
2. Part 2 performs bootstrap collection, three independent training origins, ten expert-iteration cycles, promotion tests, and final protected evaluation.

Part 1 terminates in `awaiting_phase2_approval`. Authorization requires the exact `readiness_sha256` written into `part1-readiness.json`. No protected seed, bootstrap game, or scientific training origin may be opened before that transition.

## Representation

The focal board uses the starter-relative radius-7 disk

```text
max(abs(q), abs(r), abs(q + r)) <= 7
```

with 169 stable coordinates and 7,098 spatial rows:

| Channel | Rows |
|---|---:|
| Tile presence | 169 |
| Terrain by coordinate and directed edge | 5,070 |
| Allowed wildlife | 845 |
| Placed wildlife | 845 |
| Keystone | 169 |

All legal radius-21 states remain exact. Up to 23 deterministically sorted overflow entities encode presence, exact `q/r` bins over `[-21, 21]`, six directed terrain edges, allowed and placed wildlife, and keystone state. A 24-way overflow-count block distinguishes every supported count. D6 augmentation transforms absolute coordinates and directed edges, then re-sorts overflow entities. The complete base table is 11,769 rows, including 1,703 public/global rows.

`FullOpportunities` compiles only valid catalog combinations into 83,284 collision-free inference rows. Active rows describe demand, supply, habitat and wildlife completion, market synergy, contested denial, Nature Token actionability, access delay, relative seat, rotation compatibility, and catalog archetype. A 1,624-row train-only virtual-feature table shares coordinate-free semantics, D6 orbit, market item, perspective, relative seat, completion class, and archetype. Export coalesces every virtual row into its exact inference rows; serving performs no factor lookup.

The canonical schema manifest records coordinate order, all 12 D6 permutations, overflow layout, feature dimensions, factor map, catalog checksums, and a canonical checksum.

## Network and arithmetic

The fixed campaign architecture is `cascadia-v3-sfnnv13-radius7-1024x32x32-v1`:

- shared base and opportunity transformer weights;
- separate focal-board and three-opponent-field accumulators, each width 1,024;
- two 512-wide halves per accumulator with product pooling;
- concatenated focal and field pools, width 1,024;
- eight focal-turn phase buckets: `min(7, floor(8 * completed / 20))`;
- per bucket `1024 -> 32`, where 31 values are nonlinear and one is a skip;
- concatenated squared clipped-ReLU and clipped-ReLU, width 62;
- `62 -> 32 -> 1`, plus the skip and focal-only direct potential.

Quantization-aware training begins at step zero. Base weights are int16 and opportunity and dense weights are int8. Sparse feature sums use exact int32 pre-activation accumulators; only the activation boundary clips them to `[0, 256]` and narrows to int16 for product pooling and dense inference. Dense products remain int32. This is mathematically identical to the MLX QAT graph and avoids order-dependent saturation when Cascadia's hundreds of opportunity rows exceed the narrower Stockfish accumulator range. Scales are 256 for the feature transformer, 64 for dense weights, and 16 for output. A step-zero headroom regularizer penalizes per-example focal or field pre-activation maxima above 64 float units (16,384 integer units) with coefficient 0.1, keeping learned parameters comfortably inside the intended operating envelope without changing serving semantics. The MLX QAT forward graph implements the exact integer clipping, flooring, round-away-from-zero, direct-potential, and skip boundaries. Exported float-QAT, MLX integer, NumPy integer, Rust scalar, and Rust NEON outputs are therefore identical.

## Data contracts

- `V3FeatureSchemaManifest` freezes representation and catalog identity.
- `V3AccumulatorStack` supports exact reconstruction and apply/undo.
- `V3ModelManifest` freezes graph, quantization, factor coalescence, origin, checkpoint, and serving identity.
- `V3TrainingEntry` records state identity, focal seat, phase, sparse rows, realized and teacher score-to-go, teacher variance/count, confidence, blend, and provenance.
- `V3TeacherRootLabel` preserves every top-K candidate’s mean, variance, sample count, and rank under the exact shared rollout budget.
- `V3SearchPolicy` exhaustively enumerates legal actions before direct, K32/R64, or K32/R600 selection.
- `V3GameRecord` stores compact replay-authoritative collection data and enforces exactly one newest-model focal seat during expert iteration.
- `V3CampaignState` is an append-only checksum chain with an explicit human authorization transition.

Packed engineering entries and compact scientific game replays are distinct formats. Every engineering artifact is marked `scientific_eligible=false`; the native loader refuses scientific shards unless given a checksum-authorized Phase 2 campaign state.

## Part 1 qualification

Part 1 may generate exactly 2,000 open-domain games (160,000 afterstates). It runs:

- schema, overflow, D6, feature-collision, factor-coalescence, accumulator, saturation, backend-parity, full-legal, replay, RNG, Docker, Bacalhau, dashboard, resource, and shutdown tests;
- one full 160,000-position QAT epoch;
- controlled interruption at the exact loader cursor and comparison with an uninterrupted twin;
- export/reload and 100 groups of 64 cross-backend ranking candidates;
- 100 direct games and eight focal-seat K32/R600 games;
- one Bacalhau worker failure/retry and one trainer restart;
- two measured optimization passes for MLX and gameplay.

Readiness remains red unless every registered gate passes. Gates are never relaxed automatically.

## Part 2 bootstrap

After checksum-bound approval, collect 500,000 games: 100,000 greedy, 200,000 frozen V1 direct, 100,000 mixed frozen, and 100,000 rare-legal softmax. Record all seats. Select 100,000 stratified teacher roots and 20,000 validation roots. Teacher evaluation exhaustively enumerates actions, selects the direct top 32, and spends 600 terminal rollouts by sequential halving.

Labeled roots remain compact replay-authoritative `.v3l` shards. The native Rust batch stream expands their candidate rows on demand; it must not materialize a multi-gigabyte sparse teacher corpus. The stream binds exact source identities, balances all eight phase buckets, applies the scheduled D6 transform and teacher lambda, and emits an exact exposure count.

Targets are normalized by 100 and use power loss 2.4:

```text
target = lambda * teacher_score_to_go + (1 - lambda) * realized_score_to_go
weight = clamp(1 / (0.25 + teacher_SE^2), 0.25, 4.0)
```

Anneal `lambda` from 1.0 to 0.75, sample phase buckets uniformly, and augment D6 online. Calibrate the from-scratch learning rate on 10% of the corpus over `5e-4`, `1e-3`, and `1.5e-3`, choosing quantized validation loss subject to open-domain nonregression. Train three origins with four broad passes, six 50/50 broad/teacher passes, two low-rate consolidation passes, and stochastic weight averaging over the final 20%.

For the checksum-approved 4.30-day capacity envelope, a bootstrap "pass" is a bounded exposure block, not a complete traversal of all 40 million broad afterstates. The immutable schedule is 4.5 million exposures in each of four broad blocks and 2.25 million exposures in each of six mixed plus two consolidation blocks: 36 million per origin. Three 4-million-exposure learning-rate calibration trials plus three 36-million-exposure origins total exactly 120 million MLX exposures, matching the measured projection. Treating every named pass as a full-corpus epoch would be an unapproved 1.44-billion-exposure run. Every block boundary is checkpointed; the last 7.2 million exposures per origin feed online stochastic weight averaging.

At inference, direct action value is `exact_afterstate_score + predicted_score_to_go`. Both terms are combined in quantized output units before ranking. Score-to-go labels use signed subtraction; they are never clipped at zero when a later placement changes an already-scored wildlife motif.

## Ten expert cycles

Every cycle collects exactly 10,000 games. The newest model appears in exactly one evenly rotated focal seat. Of the other three seats, 80% are the qualified frozen V1 direct policy and 20% are sampled deterministically from prior frozen V3 checkpoints; cycle 1 is necessarily all V1 opponents. Only focal entries are retained. Label 2,500 stratified roots with K32/R600, train two origins from the incumbent, select on open validation, and run the paired promotion gate. A rejected or null candidate leaves the incumbent in place but does not terminate the ten-cycle campaign. This bounded design follows the Stockfish-style quality/volume tradeoff: broad realized outcomes plus selective search supervision, rather than paying deep-teacher cost on every trajectory.

Exploration by cycle is:

```text
[0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.035, 0.03, 0.02]
```

Exploration samples only from the exhaustive direct top 32 with decaying Boltzmann temperature. Each origin trains three passes at `3e-5`, `3e-5`, and `1e-5`; data mix is 50% current, 30% prior three cycles, and 20% bootstrap/older. Phase and score quantiles are equalized. Checkpoints include model, optimizer, RNG, loader cursor, and data-manifest identity.

Each cycle pass is exactly 400,000 exposures. The current-cycle half is 120,000 focal replay rows plus 80,000 exact teacher rows; 120,000 rows come from the preceding three cycles; and 80,000 rows split evenly between bootstrap replay and bootstrap teacher data. Cycle 1 has no preceding cycle, so that 120,000-row share is reassigned evenly across the two bootstrap sources. Score quartiles are measured separately inside each of the eight phase buckets, then the native stream emits exact groups containing one row from every phase-by-within-phase-quartile cell. This makes both marginals uniform without creating structurally impossible global score/phase cells.

Promotion evaluates direct, K32/R64, K32/R600, and equal-wall-time tiers with a bounded mixture-betting e-process. Each tier uses 100–500 paired observations, null `delta <= -0.10`, alternative `delta >= +0.15`, alpha 0.05, and beta 0.05. Promotion requires every alternative boundary plus clean integrity and resource gates. Otherwise the incumbent remains. Cycle 1 retains its already-opened v1 rule over the raw legal `[-200, 200]` score-delta range. Before any Cycle 2 pair domain opens, cycles 2–10 are pre-registered on rule v2: the decision estimand winsorizes each paired delta to `[-25, 25]`, while reports preserve the raw mean and all final protected evaluation remains raw. This robust estimand keeps rare catastrophic pairs from dominating iterative model selection and gives the registered 0.15-point alternative usable power without weakening alpha or beta. The version boundary and power audit are recorded in `docs/v3/PROMOTION_GATE_POWER_AUDIT.md`.

Treatment and control share the fixed architecture, so the equal-wall-time tier gives both the identical K32/R600 work budget and records focal decision time; a treatment/control aggregate time ratio above 1.20 is a resource regression. Promotion data opens in 100-pair increments and stops on the first registered boundary or at 500 pairs. A rejected candidate is still retained as a frozen historical opponent, but an identity equal to the newest champion is never placed in another seat.

## Final evaluation

After cycle 10, freeze the champion before opening the protected domain. Run exactly 250 focal-seat pairs against the pinned qualified exact K32/R600 control, then 1,000 all-V3 games. Extend to 4,000 games if the interval can plausibly cross 100. The report includes paired and unpaired intervals, percentiles, W/T/L, histograms, every animal, aggregate wildlife, terrain, Nature Tokens, Pinecones, throughput, latency, memory, swap, overflow, and per-cycle learning curves.

The all-V3 domain is scheduled in immutable 100-game increments. After each fully terminal and reconciled distributed increment, John1's dedicated worker VM and the idle John2/John3 Docker VMs are trimmed and restarted serially, with no live containers and a complete 9/10/10 Bacalhau fabric check after each restart. This lifecycle releases deleted VirtioFS model inputs before the next increment; it never deletes retained research archives and the canonical image is repulled by digest when necessary. Final results are published as an authoritative JSON report plus a human-readable Markdown report.

Protected seed-domain keys are generated from operating-system randomness only after the checksum chain enters `final_protected_comparison`; no fixed seed value exists in source, manifests, or earlier artifacts. Paired treatment/control games derive the same game seed, rotate the focal seat, and keep opponent and search RNG domains fixed. The all-V3 score evaluation uses K32/R600 in all four seats.

## Compute and storage

John1 is the sole code/image/artifact authority and native MLX trainer. Bacalhau schedules independent container work over 9 CPUs on John1 and 10 each on John2 and John3; callers never select a host or hand-shard even/odd work. John4 remains dashboard-visible and receives no campaign work. During each cycle's MLX training, John1 is reserved for Metal and two 10-CPU jobs benchmark 500 games each with the previous frozen checkpoint in exactly one rotated focal seat against three V1 opponents. The 10-CPU request makes John1 ineligible by construction. The benchmark reports mean, P10/P50/P90, score histogram, animal means, terrain means, Nature Tokens, and Pinecones; it is permanently ineligible for scientific training. John2/John3 do not generate the blocked next cycle.

The only new root is `/Users/johnherrick/cascadia-bench/v3-nnue`. It may consume at most 40 GiB while preserving at least 50 GiB free on John1. Every planned write is preflighted against both limits.
