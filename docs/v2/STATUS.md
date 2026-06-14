# Cascadia V2 Status

Last updated: 2026-06-14

## Active Phase

**Phase 5 and 6: fair hidden-state search, MLX ranking distillation,
same-budget CRN, geometry-only value learning, absolute counterfactual state
value, nearest-neighbor action advantage, and estimator qualification are
closed. R12 rank-stratified contrast is qualified; its corrected corpus is
actively collecting across john1 and john2, while john3 is qualified for the
single frozen MLX complete-candidate-set ranker run.**

ADR 0070 is rejected. The identifiability audit showed that the historical
independent-seed teacher usually does not produce a statistically unique
winner. ADR 0071's three-game CRN pilot was positive, but ADR 0072 rejected it
on 20 fresh pairs at -0.363 with 95% CI `[-1.129,+0.404]`. No new apprentice
will be trained from CRN labels. ADR 0073 then tested exact hex adjacency and
oriented terrain-edge message passing on fresh H6 validation. It improved
pairwise log loss but regressed final correlation from 0.393 to 0.342 and MAE
from 2.541 to 2.798, so the sealed test remained unopened. ADR 0074 qualified
an R8 public-redetermination sampler at 0.487-point drift and 91.1% pairwise
accuracy, but rejected absolute state value because R16 expected totals had
only 1.945 points of standard deviation. ADR 0075 then qualified shared R8
same-decision returns, but the selected action and its three nearest H6
alternatives spanned only 1.367 points versus 1.50 required.
ADR 0076 widened the mean R16 contrast to 2.803 points and passed every
fidelity, regret, uncertainty, integrity, and cost gate except exact R8 winner
agreement, which reached 20 of 32 groups or 62.50% versus 65% required.
ADR 0077 then passed every frozen gate on fresh games: R12 reached 0.204 MAE,
0.968 correlation, 92.19% pairwise accuracy, 78.13% exact winner agreement,
and 0.037 mean regret while projecting to 10.33 local hours for 160 games.

## Completed

- Read and adopted `CASCADIA_V2_GOAL.txt` as the authoritative objective.
- Created branch `codex/cascadia-v2` without discarding the existing dirty
  worktree.
- Inspected running processes. No benchmark or training process was active.
  Stopped an orphaned `tail -F` monitor from an old head-to-head run. Left the
  existing v1 web server running because it is an active product surface, not
  stale compute.
- Captured local hardware and toolchain details in `HARDWARE.md`.
- Audited the v1 architecture, configuration surface, test distribution,
  artifact layout, data formats, benchmark semantics, and primary debt.
- Defined the v2 architecture, benchmark contract, migration sequence, and
  initial ADRs.
- Established a locked, project-owned MLX environment with uv-managed CPython
  3.12.13 and MLX 0.31.2.
- Verified evaluated computation on `Device(gpu, 0)` and added a tested,
  machine-readable device probe.
- Implemented the independent v2 rules engine, all 20 base wildlife cards,
  habitat scoring, deterministic setup, transactional actions, apply/undo,
  stable replay serialization, hashing, and component-conservation checks.
- Added a test-only v1/v2 differential crate with independently expected
  AAAAA wildlife and habitat fixtures, plus property tests proving exact
  board undo and complete seeded-game replay/hash determinism.
- Implemented the canonical legal-action generator and complete 1-4 player
  game tests.
- Added deterministic simulation, symmetric strategies, the canonical
  benchmark CLI, game-block confidence intervals, percentiles, category
  breakdowns, and JSON reports.
- Added per-decision mean/P50/P90/P99/max latency accounting and self-auditing
  report provenance: typed command configuration, hardware/toolchain,
  executable checksum, Git status digest, complete v2 source digest, and input
  artifact manifest checksums.
- Added a versioned, executable performance contract for instant, interactive,
  research, and batch-32 MLX execution. All 11 gates pass against checksummed
  canonical reports; the qualification is reproducible with
  `make performance-check`.
- Centralized Rust provenance in `cascadia-provenance` and made value/ranking
  collection resume reject source or executable drift before appending another
  shard.
- Established fresh 50-game baselines: random 33.620 and greedy 86.265 mean
  base score.
- Added disjoint train/validation/test/final seed namespaces and a compact
  fixed-record dataset format with parallel collection, atomic resumption,
  per-shard manifests, BLAKE3 checksums, and Rust/Python validation.
- Added vectorized memory-mapped decoding into MLX entity tensors.
- Added an MLX entity-set attention model with decomposed score heads,
  held-out metrics, atomic safetensors checkpoints, exact optimizer/cursor
  resumption, and corruption tests.
- Added best-checkpoint tracking, atomic standalone promotion, and promoted
  model integrity validation.
- Hardened training resumption so dataset, model, optimizer, runtime, and
  source provenance must match the checkpoint exactly; only a larger epoch
  budget and the explicit resume path may differ.
- Added a versioned batched Rust/MLX inference protocol and an experimental
  learned afterstate strategy running through the canonical simulator.
- Executed complete GPU train/checkpoint/resume/promotion/inference smoke runs.
- Rejected and removed `pattern-v1` after a 10-game pilot scored 78.65 and
  collapsed Bear scoring to 0.10.
- Added the stateless canonical `cascadia-api` replay service and a standalone
  responsive React/TypeScript product with setup, human/AI seats, legal staged
  turns, market actions, rotation, wildlife placement, undo/redo, persistence,
  save/load, history, scoring, and candidate analysis.
- Verified the web product through Rust API tests, TypeScript unit tests,
  Playwright desktop/mobile flows, production builds, screenshots, and a clean
  zero-vulnerability npm audit.
- Collected the first substantive training split: 256 greedy games, 20,480
  positions, 8 checksummed shards.
- Collected and validated a disjoint 64-game, 5,120-position validation split.
- Trained the first substantive MLX value model locally for 20 epochs and
  selected epoch 8 at 2.817 held-out MAE.
- Corrected the experimental afterstate boundary so model candidates include
  the post-turn market, token count, phase, and acting-seat perspective.
- Rejected the unconstrained value policy at 41.25 mean in the mandatory
  gameplay smoke despite its low held-out MAE.
- Rejected exact-greedy-top-8 plus MLX ranking in a 20-game paired trial:
  -2.688 points, 95% CI -3.518 to -1.857, with 18 losses in 20 games.
- Implemented reproducible public-information hidden-state determinization
  with common random numbers and no actual-stack leakage.
- Confirmed the original stateful search schedule, then superseded it before
  product promotion because a stateless API could not reproduce its RNG
  cursor without replaying prior searches.
- Promoted `determinized-lookahead-v2-k4-r4-d4` through the shared CLI/API
  policy path after a disjoint 50-game confirmation: 89.435 mean, +2.555
  paired, 95% CI 1.915-3.195, record 44-1-5.
- Enabled the interactive strength tier and four-ply candidate analysis in the
  local web product.
- Profiled the promoted search with native sampling, flattened
  candidate-by-determinization parallelism, and replaced full-board candidate
  rescoring with exact dependency-aware delta rescoring.
- Reproduced all 20 pilot games bit for bit after optimization while reducing
  paired wall time from 153.461 to 41.115 seconds, a 3.73x speedup.
- Isolated candidate breadth, hidden-state sample count, and rollout horizon on
  one shared development suite. K8 produced the strongest pilot signal.
- Measured K4 candidate value recall at 83.25% over 400 decisions; 67 decisions
  excluded a strictly higher-valued K8 action, with misses concentrated early.
- Promoted `determinized-lookahead-v2-k8-r4-d4` after a disjoint 50-game
  search-to-search confirmation: 90.270 mean, +0.965 paired, 95% CI
  0.418-1.512, record 31-4-15.
- Confirmed K16 breadth as a positive research result at 91.555 mean and
  +0.745 paired, 95% CI 0.187-1.303, but did not promote it because the lower
  bound missed its pre-registered +0.25 gate.
- Confirmed a Bear-aware K8+B8 candidate union against promoted K8 on a
  disjoint 50-game suite: 91.215 mean, +0.865 paired, 95% CI 0.320-1.410,
  +3.56 Bear, record 31-5-14, and 5.54 seconds per game.
- Completed the preregistered generic-K16 control. K8+B8 was statistically
  tied with K16 at +0.120, 95% CI -0.346 to 0.586, while shifting +2.56 into
  Bear. It is retained as a research teacher but not promoted to the product.
- Rejected the K16+B8 superset in a 10-game pilot at -0.150; more uncalibrated
  candidate breadth again traded non-Bear wildlife for Bear.
- Implemented and smoke-tested the search-distillation pipeline end to end:
  grouped checksummed datasets, listwise MLX training, exact checkpoint resume,
  scalar batch serving, Rust gameplay inference, and ranking metrics.
- Generalized the value-dataset writer for manifest-owned custom policies and
  added resumable H6 trajectory collection with final decomposed targets,
  enabling strong-policy MLX leaf/value training from fresh local games.
- Collected 128 ranking-train games and 32 disjoint validation games from the
  frozen K8+B8 teacher: 145,447 labeled candidates across 12,800 decision
  groups.
- Trained and promoted the fresh MLX `entity-set-ranker-v1` checkpoint after it
  passed held-out ranking gates: 0.800 pairwise accuracy, 0.250 mean top-one
  regret, and 0.663 value-difference correlation.
- Rejected that ranker as a standalone policy at -2.225 versus K8 and rejected
  unanchored and six-anchor rollout prefilters because both materially
  collapsed non-Bear wildlife.
- Implemented exact habitat-cohesion candidate generation using real matching
  terrain edges and distinct draft-and-tile placements.
- Confirmed `habitat-candidate-lookahead-v1-k8-h6-r4-d4` against promoted K8
  on 50 disjoint games: 91.760 mean, +1.090 paired, 95% CI 0.558-1.622,
  +0.725 habitat, +0.240 wildlife, record 36-2-12, and 4.81 seconds per game.
- Completed the generic-K16 control. H6 was +0.515 and 15% faster, but its
  confidence interval crossed zero and habitat was flat, so H6 remains a
  confirmed research teacher rather than a product promotion.
- Rejected H8 on runtime, H4 on strength, and H6/D8 on a null horizon result.
- Completed the resumable independent v1 champion reproduction over 50 games
  and 200 seats: 95.895 all-seat mean, 95% CI 95.480-96.310, P90 99. The
  adapter corrected the legacy CLI's seat-zero-only aggregate and checksummed
  both binary and weights.
- Finalized the H6 diagnosis against that reference: the 4.135-point gap is
  1.255 habitat, 2.345 wildlife, and 0.535 Nature Tokens. Bear alone is 5.070
  points behind, partly offset by stronger H6 Elk, Hawk, and Fox.
- Rejected the preregistered K8+H6+B8 union against H6: -0.300 total, +2.075
  Bear, -2.500 aggregate non-Bear wildlife, nearly flat habitat, and an
  acceptable 8.15 seconds per game. Direct Bear candidate injection still
  shifts species rather than improving total allocation.
- Collected 128 fresh H6 ranking games and 32 disjoint validation games:
  10,240 train groups / 129,894 candidates and 2,560 validation groups /
  32,525 candidates, with complete shard validation.
- Trained the fresh H6 MLX ranker to validation futility and selected epoch 5:
  0.792 pairwise accuracy, 0.334 top-one regret, and 0.759 value-difference
  correlation, clearing every held-out gate.
- Rejected its anchored K16+H8 wide-frontier pilot at the gameplay gate:
  +0.175 paired, 95% CI -1.322 to 1.672, -0.325 habitat, balanced wildlife,
  3-0-7 record, and 7.34 seconds per game. It missed the preregistered +0.25
  advancement threshold, so no confirmation was run.
- Rejected the H6-ranker rollout policy at its mandatory full-configuration
  runtime smoke: 196.0 treatment seconds per game, 2.42-second mean decisions,
  4.10-second P90, and 10.41-second maximum. This exceeded the registered
  120-second research ceiling, so the ten-game strength pilot was not run.
- Collected and validated 256 H6 value-training games and 64 disjoint
  validation games, totaling 20,480 train positions and 5,120 held-out
  positions with final decomposed acting-seat targets.
- Rejected the H6 value model before gameplay. Its best checkpoint improved
  total MAE to 2.538 and passed wildlife component regression checks, but total
  correlation was only 0.212 versus the registered 0.50 gate. The downstream
  value-leaf strength pilot was therefore not run.
- Rejected the self-only H6-ranker rollout policy at its mandatory runtime
  smoke: 62.58 treatment seconds per game, 782 ms mean decisions, 1.57-second
  P90, and 5.53-second maximum. This exceeded the registered 30-second ceiling,
  so the ten-game strength pilot was not run.
- Rejected the exact K16+H8 wide frontier against H6: -0.325 paired, 95% CI
  -1.652 to 1.002, +0.325 habitat, -0.450 wildlife, -0.200 Nature Tokens, and a
  3-0-7 record. Candidate breadth is not fixing the frozen rollout objective.
- Implemented the missing rules-derived pattern-aware baseline. Its first
  pilot was strongly positive against greedy at +3.575, 95% CI 2.909-4.241,
  with +0.925 habitat, +1.375 wildlife, +1.275 Nature Tokens, +4.125 Bear, and
  a 10-0-0 record. The exact configuration was rejected because 2.291 seconds
  per game missed its preregistered two-second runtime gate; profiling and a
  separately registered behavior-preserving cost-control pass follow.
- Profiled that baseline and found three redundant full legal-action
  enumerations. A unified frontier pass reproduced every pilot score and
  category value exactly while reducing runtime to 0.506 seconds per game
  (4.53x faster) and mean decision latency to 6.31 ms. The frozen policy has
  advanced to a disjoint 50-game confirmation.
- Confirmed pattern-aware over 50 disjoint games: 91.525 mean, +4.890 paired
  versus greedy, 95% CI 4.296-5.484, +1.295 habitat, +1.730 wildlife, +1.865
  Nature Tokens, +3.875 Bear, a 50-0-0 record, and 0.820 seconds per game. A
  direct paired K8 product control is now registered.
- Promoted pattern-aware to the interactive product tier after a direct
  50-game K8 control: 91.890 versus 90.775, with K8-minus-pattern delta -1.115
  and 95% CI -1.696 to -0.534. Pattern-aware added 0.480 total wildlife and
  ran 14.49x faster. The API, web defaults, Make target, and documentation now
  share the promoted policy.
- Rejected pattern-aware as an H6 rollout replacement. The full-config smoke
  passed at 15.41 treatment seconds per game, but the disjoint ten-game pilot
  scored -0.550 paired with 95% CI -1.796 to 0.696 and a 4-0-6 record.
  Habitat (-0.050) and wildlife (-0.400) stayed inside guardrails, but the
  primary +0.25 gate failed, so no 50-game confirmation was run.
- Qualified a fresh canonical V2 teacher at 96.350 over ten paired games, with
  +4.375 paired gain and 95% CI 2.938-5.812, while keeping all 800 selected
  actions legal and using zero fallbacks.
- Rejected direct imitation on the production K8+H6+B8 frontier after exact
  recall was only 51.25% over 160 decisions; early, middle, and late recall
  all missed their frozen gates.
- Registered full-legal explicit-action imitation: MLX learns from canonical
  teacher positives plus structured legal negatives, while production scores
  the complete legal action set so proposal recall is 100%.
- Completed the full-legal imitation implementation smoke. The final model
  encodes one shared public state and lightweight action rows, reducing the
  exhaustive policy from 21.878 to 3.966 seconds per game and 49.56 ms mean
  decision latency. The grouped shard format stores a 5,120-candidate game in
  418,672 bytes, 11.64x smaller than the discarded repeated-state format.
- Added independent Rust/Python shard tests, framed service tests, exhaustive
  legal-action tests, exact checkpoint resumption, untouched-test evaluation,
  integrity-gated standalone promotion, and complete Make/runbook commands.
- Rejected the first full-legal MLX apprentice before promotion and gameplay.
  The selected epoch-three checkpoint reduced validation loss from 4.1584 to
  2.9488 and passed untouched-test top-one at 20.08%, but top-five recall was
  51.09% versus the 55% gate and MRR was 0.347 versus 0.40. Pairwise accuracy
  was 88.08%, so broad ordering was learned but the chosen teacher action was
  not concentrated near the top reliably enough.
- Preserved the rejected train, validation, and test corpora with exact
  manifests and checksums. No gameplay seed was opened. The successor must
  use a fresh preregistered test split and must retain teacher margin,
  uncertainty, and candidate-source evidence that v1 omitted.
- Rejected action-query cross-attention on validation. Its selected checkpoint
  regressed to 18.98% top-one, 48.83% top-five, and 0.337 MRR, so no fresh
  test split was opened. A reciprocal-immediate-rank residual improved v1 in
  leave-one-game-out analysis to 23.36% top-one and 58.05% top-five, but its
  0.3968 MRR remained too borderline for a post-hoc promotion attempt.
- Rejected end-to-end training around that prior. The best residual checkpoint
  reached 21.33% top-one, 54.84% top-five, and 0.371 MRR. Pairwise accuracy
  rose to 89.87%, but every frozen top-rank gate failed.
- Rejected exact sixfold hex-rotation augmentation. Its best checkpoint
  reached 20.70% top-one, 48.05% top-five, and 0.342 MRR. Geometry capacity,
  monotonic calibration, and symmetry augmentation have now all failed to
  repair the one-hot teacher target.
- Implemented and exercised a complete local search-guided policy-iteration
  loop: apprentice-owned trajectories, H6 counterfactual labels, model-bound
  dataset manifests, aggregate training, warm start, anti-forgetting
  validation, resumable checkpoints, and model H2H evaluation.
- Rejected iteration 1 before gameplay. The best epoch improved balanced
  listwise loss by 0.002535 and apprentice top-one regret by 0.017773 while
  slightly improving the original H6 validation distribution, but it missed
  the preregistered 0.03 regret gate. No model was promoted.
- Corrected the two-turn wildlife-commitment evaluator so its Bellman horizon
  is capped by the acting seat's exact remaining turns. A late-game regression
  test proves it becomes exactly equal to the one-turn policy when only one
  future personal turn remains.
- Rejected the phase-capped commitment policy after its ten-game pilot. It
  improved paired score by 0.650, Bear by 1.900, total wildlife by 0.950, and
  won 8-0-2, but lost 0.950 aggregate non-Bear wildlife and 0.650 habitat,
  violating both registered mechanism guardrails. No confirmation was run.
- Implemented a fair terminal policy-improvement oracle using shared
  public-information redeterminizations, the frozen pattern frontier, and
  full-game pattern-aware continuations. Its complete search and CLI suites
  are deterministic, legal, terminal-scored, and strict-Clippy clean.
- Rejected the R2 terminal teacher at its three-game qualification gate. It
  improved Bear by 1.250, total wildlife by 1.167, and habitat by 0.917, but
  gained only 0.250 total with extreme paired variance and lost 1.833 Nature
  Tokens. No terminal-label dataset was collected.
- Qualified the variance-controlled R8 terminal teacher for MLX data
  generation. It scored 94.833 versus pattern-aware at 93.500 over the
  registered three-game suite: +1.333 paired, +1.750 Bear, +0.333 wildlife,
  +1.417 habitat, -0.417 Nature Tokens, and a 2-0-1 record. At 185.88 seconds
  per game it is a research oracle, not a product policy.
- Stopped the first terminal-ranker collection before training after proving
  its candidate records exposed the actual hidden post-draft refill. The
  complete 64-game train split and partial nine-game validation split are
  quarantined with checksummed manifests; no model or gameplay result was
  produced.
- Replaced the flawed boundary with typed `PublicGameState` afterstates that
  stop before refill, preserve partial independent-draft market slots, and are
  byte-identical under hidden-stack redetermination. Bumped the feature schema
  to `compact-entity-v2`, making every old dataset and model fail closed.
- Corrected ranking evaluation to report tie-aware top-one value recall
  separately from strict single-index agreement. Exact-value ties no longer
  turn candidate array order into a playing-strength metric.
- Completed the corrected terminal R8 dataset: 64 train games, 16 validation
  games, 6,400 groups, and 96,070 candidates. Cross-language validation found
  70,926 paired-draft records, 25,144 independent-draft records, and zero
  hidden-refill leaks.
- Rejected the corrected entity-set terminal ranker before promotion. Epoch 8
  improved selection loss and passed pairwise accuracy (0.680) plus
  value-difference correlation (0.508), but missed mean top-one regret
  (0.968 versus <=0.75) and tie-aware top-one recall (0.276 versus >=0.45).
  No gameplay was run.
- Diagnosed the representation bottleneck: the independent full-afterstate
  scorer receives no action identity or newly placed marker and only barely
  improves on immediate rank-1 regret. The next learned policy will encode the
  candidate delta explicitly.
- Implemented and froze the registered `action-delta-ranker-v1` successor:
  versioned checksummed action-ranking datasets, exact source-label
  enrichment, explicit draft/placement/prelude/category-delta features,
  changed-entity markers, grouped MLX training, untouched-test evaluation,
  fail-closed promotion, and batched Rust gameplay inference.
- Preserved the hidden-state-safe boundary throughout action enrichment.
  Candidate afterstates remain observable and pre-refill; deterministic
  replay hash-matches every action, immediate value, and afterstate byte to
  its corrected R8 source before writing.
- Completed `make action-ranking-smoke` end to end with disposable tiny data:
  terminal collection, exact continuation metadata, enrichment, validation,
  MLX training, test reporting, and Rust-to-MLX gameplay. The one-epoch smoke
  retained initialization and failed its intentionally underpowered test
  gates, so it is implementation evidence only and not a strength result.
- Completed the substantive action-delta experiment with 64 train, 16
  validation, and 16 untouched test games. Epoch 7 improved validation
  selection loss from 2.6640 to 2.5599, and untouched test passed pairwise
  accuracy (0.670) plus value-difference correlation (0.495).
- Rejected `action-delta-ranker-v1` before promotion and gameplay. Untouched
  mean top-one regret was 0.968 versus the 0.75 gate and tie-aware recall was
  0.273 versus 0.45. The promotion command correctly refused the artifact, so
  zero gameplay seeds were consumed.
- Implemented the exact first-rotation opponent-conditioned wildlife model,
  including public supply inference, free three-kind and automatic four-kind
  replacement, marginal-score opponent drafts, and hidden-order invariance.
  A generating-function dynamic program exactly matched exhaustive replacement
  enumeration while cutting the seed-25999 smoke from 5.206 to 3.348 seconds
  with identical scores.
- Rejected `pattern-competition-v1` after its frozen ten-game pilot. It gained
  0.875 total, 1.275 Bear, 0.400 habitat, and 0.725 Nature Tokens, but lost
  0.250 total wildlife and 1.525 aggregate non-Bear wildlife. Parallel runtime
  was 10.316 seconds per game versus the five-second gate. Three gates failed,
  so no confirmation was run.
- Rejected the anchored `pattern-portfolio-v1` successor. It passed runtime
  at 2.866 seconds per game and converted the prior 1.525 non-Bear loss into a
  0.500 gain with flat habitat, but paired score was only +0.025 and Bear fell
  0.575. Preserving promoted first-turn allocation removed the cross-turn
  strength signal; no confirmation was run.
- Profiled and removed repeated habitat rescoring from the terminal teacher.
  A reusable habitat component analysis, per-tile evaluation context, split
  tile/wildlife rescoring, and bounded stack collections preserve every score
  on the ten-seed pattern reference while reducing pattern-aware runtime from
  0.506 to 0.066 seconds per game. The exact R8 reference seed fell from
  273.884 to roughly 80-85 seconds.
- Rejected the registered final-four-turn R8 hybrid by its frozen gates. It
  produced a tightly positive +0.475 paired result with 95% CI
  `[+0.197,+0.753]`, an 8-1-1 record, +0.225 total wildlife, +0.225 habitat,
  5.895 seconds per game, and 315 ms P90 move latency. It missed the +0.500
  score gate by 0.025 and Bear was -0.100, so no confirmation was run.
- Rejected the registered final-five-turn successor at its confirmation
  mechanism guardrail. Its pilot passed every gate at +1.000 with Bear +0.750.
  The disjoint 50-game confirmation retained +0.425 with 95% CI
  `[+0.198,+0.652]`, a 35-6-9 record, +0.145 wildlife, +0.340 habitat, and
  7.530 seconds per game. Bear was +0.200 against the frozen +0.250
  confirmation requirement, so the strategy was not promoted.
- Added exact per-decision wildlife-placement caching to the action generator.
  Combined with habitat analysis and split delta scoring, it made final-five
  R8 search pass its runtime gate while preserving all A-D scoring and exact
  ten-seed pattern-aware behavior.
- Rejected the wildlife-diverse terminal frontier at two mechanism gates. It
  passed score at +0.550 with an 8-0-2 record, Bear +1.625, habitat +0.500,
  10.976 seconds per game, and 574 ms P90 latency. Total wildlife was only
  +0.100 and aggregate Elk+Salmon+Hawk+Fox fell 1.525. Adding candidates
  amplified the same allocation tradeoff instead of repairing it.
- Rejected confidence-gated W2 terminal improvement at the remaining non-Bear
  guardrail. The one-sided 90% paired lower bound raised the pilot to +0.825
  with a positive interval, 9-0-1 record, +0.350 wildlife, +0.575 habitat,
  and 8.951 seconds per game. It roughly halved non-Bear damage to -0.800 but
  did not eliminate it.
- Promoted the same fixed confidence rule on the original K8+H6+B8 frontier.
  The disjoint 50-game confirmation scored 91.915 against 91.495, a +0.420
  paired gain with 95% CI `[+0.179,+0.661]` and a 28-9-13 record. Bear
  (+0.080), total wildlife (+0.115), non-Bear wildlife (+0.035), habitat
  (+0.365), Nature Tokens (-0.060), runtime (6.995 seconds per game), and P90
  decision latency (362 ms) all passed their frozen gates.
- Added that confirmed strategy to the Rust API and web client as the available
  `strong` tier. Instant and interactive remain unchanged; strong applies the
  final-five R8 c90 operator and falls back exactly to pattern-aware before its
  cutoff.
- Rejected the first direct MLX distillation of that policy before test
  access. The fixed run collected 2,560 train groups and 640 validation groups,
  then selected epoch 12 at 0.966878 lower-bound MSE, 0.760583 correlation,
  0.095716 policy regret, and 0.765625 exact agreement. It recovered only
  0.006711 of selected challengers against the frozen 0.35 gate. The untouched
  test labels remain sealed and no gameplay ran.
- Preregistered a groupwise successor that learns the actual anchor-versus-
  challenger decision with balanced cross-entropy and an auxiliary lower-bound
  head. The architecture, class weight, checkpoint metric, validation gates,
  and no-threshold-tuning rule are frozen in ADR 0026.
- Rejected that groupwise successor before test access. Its selected epoch-4
  checkpoint passed regression, regret, agreement, false-positive, and
  correlation gates but selected no challenger at all. Unthresholded exact
  challenger recall was only 0.154.
- Traced both distillation failures to seed-dependent teacher noise: R8 sample
  seeds include the hidden game seed, while model inputs correctly expose only
  public afterstates. ADR 0027 now freezes a direct R32-versus-R8 strength test
  before any higher-sample dataset is collected.
- Rejected R32 after its ten-game pilot. It passed runtime at 28.217 seconds
  per game and preserved categories, but scored 91.300 versus R8 at 91.425:
  -0.125 paired, 95% CI `[-0.616,+0.366]`, and a 4-0-6 record. No confirmation
  or higher-sample label collection ran.
- Preregistered signed score-to-go value learning on the exact prior H6 seed
  domains. It isolates residual target semantics while preserving trajectory,
  architecture, optimizer, and validation gates.
- Implemented and verified its fixed-width Rust/Python dataset, signed residual
  model, resumable trainer, and Apple MLX smoke path. A two-game parallel probe
  committed two ordered one-game shards and passed checksums plus target
  identities; the frozen 256/64 collection is authorized.
- Rejected signed score-to-go before gameplay. The full 20-epoch MLX run
  selected epoch 13 at 2.568601 reconstructed-final MAE and 0.397451
  correlation. Residual correlation was 0.991700 and every component gate
  passed, but even the maximum correlation at any epoch was only 0.414201
  against the frozen 0.50 gate. No model was promoted.
- Rejected target-level perfect-information policy improvement. The corrected
  seat-rotated diagnostic scored 93.150 versus pattern-aware at 91.375:
  +1.775 with 95% CI `[+0.299,+3.251]` and an 8-0-2 record. Exact future
  knowledge added 11.325 Bear but lost 4.975 Elk and 4.050 Hawk, exposing a
  continuation allocation limit rather than an uncertainty-only gap.
- Confirmed that structural wildlife candidate recall remains material after
  removing Monte Carlo winner's curse. The exact W2 frontier scored 93.975
  versus 92.625 for the exact base frontier: +1.350 with 95% CI
  `[+0.704,+1.996]` and a 9-0-1 record. Fox gained 1.775, while the other four
  wildlife species summed to -0.525 and habitat fell 0.500. Wider candidates
  are useful but remain below the 97 diagnostic boundary and the 100 target.
- Rejected Fox-only transfer through the promoted public-state terminal
  evaluator. Against strong, the focused F2 treatment tied all ten seed
  blocks for exactly 0.000 score gain. Fox moved only +0.050, total wildlife
  -0.025, non-Fox wildlife -0.075, habitat was flat, and runtime passed at
  5.525 seconds per game. Exact candidate value exists, but R8 c90 did not
  identify profitable use of it.
- Confirmed multi-turn continuation as a separate source of value. A
  perfect-information final-five width-16 focal beam improved the exact W2
  one-step oracle from 92.900 to 93.650: +0.750 with 95% CI
  `[+0.400,+1.100]` and a 9-1-0 record. Habitat gained 0.250, wildlife 0.175,
  and Nature Tokens 0.325. The result remains below the 97 diagnostic boundary
  and 6.350 points below target.
- Rejected the first public focal-beam teacher. Four redeterminizations and a
  width-four final-five beam scored 92.850 versus strong at 92.925:
  -0.075 with 95% CI `[-0.565,+0.415]`. Bear gained 0.475, non-Bear wildlife
  fell 0.500, total wildlife fell 0.025, and runtime passed at 114.312 seconds
  per game. No confirmation or MLX collection was authorized.
- Rejected category-preserving exact beam retention. The frozen width-16
  portfolio beam scored 94.075 versus scalar at 94.025: +0.050 with 95% CI
  `[-0.048,+0.148]` and a 1-9-0 record. Habitat and Nature Tokens each gained
  0.050 while wildlife fell 0.050. Nine seed blocks tied exactly, so scalar
  pruning is not the missing continuation mechanism.
- Rejected W4 at every exact focal layer. Its +3.250 runtime smoke was a false
  positive; the ten-seed pilot scored 93.075 versus W2 at 94.000: -0.925 with
  95% CI `[-2.426,+0.576]`. Fox gained 1.425, but Bear fell 1.250, wildlife
  0.550, and habitat 0.400.
- Removed repeated full frontier-record clones and unnecessary complete sorts
  from scalar pattern selection. The general optimization reproduced every
  W2/W4 smoke score and category exactly while reducing W4 runtime from
  232.275 to 163.404 seconds, a 29.7% improvement.
- Rejected root-only W4 as a null mechanism. With future layers fixed at W2,
  treatment scored 94.625 versus 94.550: +0.075 with 95% CI
  `[-0.030,+0.180]`; eight blocks tied, habitat and wildlife were flat, and
  only Nature Tokens gained 0.075.
- Rejected doubled exact beam capacity. B32 scored 94.100 versus B16 at
  94.075: +0.025 with 95% CI `[-0.024,+0.074]`; nine blocks tied and only
  0.025 total wildlife moved.
- Accepted the public beam-state value target. Across 32 final-five groups and
  586 candidates, disjoint R8 batches reached 0.9914 raw value correlation,
  0.9365 centered-advantage correlation, 65.625% top-action agreement, and
  0.1133 mean regret. Both Rust and Python validated the real checksummed
  shards. ADR 0039 freezes the first authorized MLX continuation-value run.
- Completed the full ADR 0039 frozen corpus locally: 32 train games with
  10,116 candidates, 8 validation games with 2,561 candidates, and 8 sealed
  test games with 2,548 candidates. Every one-game shard passed Rust
  validation and carries matching source and executable provenance.
- Added exact sibling-opponent replay inside the final-five beam and replaced
  allocation-heavy Bear readiness scoring with an equivalent single board
  pass. Reference-equivalence tests, all 54 search tests, and strict Clippy
  passed; frozen candidate counts remained identical across restarts.
- Rejected `mlx-public-beam-value-v1` on validation. Its Apple GPU run finished
  20 epochs in 85.51 seconds and passed centered correlation at 0.6730, but
  failed terminal MAE (2.7682), raw correlation (0.5830), exact top agreement
  (0.1406), and mean regret (0.7280). The sealed test evaluator correctly
  remained locked, so no test metrics, model promotion, or gameplay result
  exist.
- Implemented and rejected the preregistered joint candidate-set successor.
  All 49 Python tests and Ruff passed, grouped serving was verified, and a
  real GPU gradient update succeeded. The selected epoch-5 checkpoint improved
  immediate-score regret from 0.4873 to 0.3730 and top-value recall from
  0.2969 to 0.3516, with 0.7891 centered correlation. It still missed the
  fixed 0.35 regret and 0.40 recall gates. Sealed test and gameplay remained
  locked. Further neural architecture changes on this corpus are closed.
- Rejected the full online public R8/B16 focal-beam oracle. It scored 92.167
  versus promoted strong at 92.500 across the frozen three-block
  qualification: -0.333 with 95% CI `[-0.987,+0.320]` and a 0-2-1 record.
  Runtime passed at 200.143 seconds per block, but total wildlife fell 0.500,
  non-Bear wildlife fell 1.167, and treatment crossed the preregistered 92.50
  absolute rejection floor. Repeatable public beam labels are therefore not
  sufficient evidence of a strong policy target.
- Rejected phase-decayed structural potential before validation. A complete
  125-policy, 4,000-game train sweep selected opportunity 1.00, habitat 0.00,
  and Bear readiness 0.75 at 92.117 versus the included production tuple at
  91.992: only +0.125 against the frozen +0.40 gate. Bear improved by 1.023,
  but non-Bear wildlife and habitat paid for it. No held-out seed was opened.
- Built an isolated, feature-gated public-state bridge to the historical V1
  NNUE/MCE policy. It reconstructs hidden inventory without actual-stack
  leakage, filters malformed root records through canonical V2 transition
  validation, revalidates selections, and records complete provenance.
- Rejected score-exact legacy reuse after discovering a real V1 rules defect:
  its greedy Elk A partition scored a connected line-of-three plus line-of-two
  layout at 13 instead of the official maximum interpretation of 14. The
  layout is now a permanent trusted differential regression.
- Qualified that evaluator as an explicitly approximate, non-promotable action
  teacher under canonical V2 execution. Across ten untouched blocks it scored
  96.350 versus promoted strong at 91.975: +4.375 paired, 95% CI
  `[+2.938,+5.812]`, a 10-0-0 record, +2.350 wildlife, +2.250 habitat, and
  -0.225 Nature Tokens. All 800 selected actions and 24,069 retained K32
  records were canonical, with zero fallbacks or score mismatches. Fresh MLX
  action imitation is authorized.
- Rejected winner-only full-legal imitation and three validation-only
  successors. The original model learned broad pairwise ordering but missed
  top-five and MRR; cross-attention, a trained immediate residual, and exact
  hex rotations did not repair the selected-action concentration gap.
- Instrumented the historical teacher to return every sequential-halving
  rollout mean, standard deviation, and sample allocation. This exposed and
  removed randomized hash-map iteration from its candidate prefilter; a
  deterministic parity audit now rejects any selected-action drift.
- Measured that the old 64-action corpus retained only 595 of 2,344 teacher
  estimates in a complete R2 game, or 25.38%. Sidecar enrichment of that
  corpus was rejected as structurally incomplete.
- Built paired one-game `.cim` and `.imv` collection that retains the complete
  teacher and pattern frontiers in 96 actions. R2 train and validation smokes
  aligned 2,344/2,344 and 2,434/2,434 estimates, respectively, and exact
  no-op resume passed.
- Added strict Rust and Python evidence validation plus a resumable MLX
  distributional trainer. A real Apple-GPU smoke restored optimizer and cursor
  state and improved validation loss across two epochs. ADR 0053 freezes the
  first substantive full-frontier experiment before its seeds are opened.
- Passed the mandatory R600 teacher parity gate over a complete 80-decision
  game. The original and instrumented paths selected identical actions across
  2,400 candidate estimates in 292.017 seconds; the checksummed report now
  authorizes frozen substantive collection.
- Completed the ADR 0053 full-frontier corpus locally: 64 train and 16
  validation games, 614,400 retained actions, and 191,662/191,662 teacher
  estimates aligned. Every paired one-game shard passed Rust and Python
  validation.
- Rejected the first distributional apprentice on validation. Loss improved
  from 1.832475 to 1.534834 and value-difference correlation passed at
  0.444333, but top-one was 13.750%, top-five 38.438%, MRR 0.269223,
  teacher-frontier coverage 71.406%, and pairwise accuracy 67.975%. No test,
  promotion, or gameplay domain was opened.
- Fixed two scale defects exposed by the substantive run: paired readers now
  stream shard-owned bytes without exhausting file descriptors, and no-op
  ranking resumes preserve the original final report and cumulative runtime.
- Preregistered and implemented ADR 0054. The new model fixes immediate score
  in the final point prediction, zero-initializes only the continuation
  residual, and trains that residual from uncertainty-weighted R600 means.
  Exact unit tests and a two-epoch Apple-GPU resume smoke pass.
- Rejected ADR 0054 on fresh validation. Anchored loss fell from 4.984072 to
  0.984838, but selected top-one regressed from 18.91% to 17.19%,
  value-difference correlation fell from 0.5675 to 0.3805, and conditional
  regret rose to 1.1573. No test, promotion, or gameplay domain was opened.
- Diagnosed the target failure quantitatively: only 0.438% of train and 0.456%
  of fresh-validation continuation-residual variance was within action
  groups. Absolute regression learned the action-independent state workload
  instead of the decision-local continuation advantage.
- Screened group-centered advantage on the already-open validation split. It
  improved teacher-frontier coverage and value geometry but still regressed
  exact top-one to 17.50%, so no fresh R600 split was spent and the unused
  training path was removed.
- Passed ADR 0055's exact MLX port of the qualified historical NNUE. The
  checksummed 11,231-512-64-1 artifact matched Rust within 0.00004197 points
  on all 80 real trajectory states, repeated bit-deterministically, and
  sustained 40,569 batch-32 evaluations per second on `Device(gpu, 0)`.
- Made repeated sparse features an explicit compatibility invariant. Every
  real fixture record contained duplicates, totaling 1,170 repeated
  occurrences with maximum multiplicity five; MLX preserves them exactly.
- Passed ADR 0056's long-lived sparse MLX service boundary. Service output was
  bit-identical to direct MLX, remained within 0.00004197 points of Rust over
  all 80 fixture states, shut down cleanly, and sustained 7,589 batch-32
  evaluations per second at 4.70 ms P99 end to end.
- Prevented service verification from regenerating ADR 0055 artifacts.
  `make legacy-nnue-mlx-service` now consumes the immutable qualified model
  and fixture, so later experiments cannot rewrite earlier evidence.
- Rejected ADR 0057 during its implementation smoke. The evaluator-independent
  batch search reproduced native exactly, but standard MLX changed one action
  and seven allocations after near-tie drift and took 6.05x native runtime.
- Passed ADR 0058's packed CSR, Rust-order MLX operation. It is bit-identical
  to Rust over all 80 fixture states and sustains 75,176 batch-32 evaluations
  per second at 0.698 ms P99, 9.9x ADR 0056's service throughput.
- Passed ADR 0059's full search integration. Exact MLX matched all 2,494 R32
  estimates and all three R600 spots bit for bit, with zero fallbacks and a
  1.073x native runtime ratio over the complete 80-decision trajectory.
- Passed ADR 0060's fresh gameplay reproduction. Exact MLX scored 95.800
  versus promoted strong at 92.275, a +3.525 paired gain with 95% CI
  `[+2.388,+4.662]`, while all 800 selections remained legal and every neural
  forward ran locally through MLX.
- Rejected ADR 0061's R1200 budget increase. It added only +0.167 over R600
  across three paired games while doubling runtime to 311.43 seconds/game;
  the next probe changes root candidate recall instead of resampling.
- Rejected ADR 0062's generic K64 root widening. It scored 96.667 versus
  K32's 96.583 across three paired games, only +0.083 with 95% CI
  `[-5.151,+5.318]`, while losing 1.500 habitat. Both arms remained legal,
  fallback-free, and below 156 seconds/game, so the rejection is a strength
  result rather than an infrastructure failure.
- Accepted ADR 0064's rules/strategy boundary: same-slot independent Nature
  Token drafts remain officially legal, but ranked frontiers exclude them
  because the free paired draft produces the same components without spending
  a token. The first ADR 0063 smoke was quarantined as invalid implementation
  evidence before rerunning the unchanged frozen protocol.
- Rejected ADR 0063's exact H6 semantic root union. It scored 96.500 versus
  K32's 96.167, a +0.333 paired gain with 95% CI `[-1.485,+2.152]`, missing
  the +0.500 gate despite +0.250 habitat, +0.083 wildlife, neutral tokens, 27
  selected novel actions, zero fallback, and slightly lower runtime. Candidate
  breadth and habitat semantics are now closed; the next lever is value
  representation or multi-turn planning.
- Completed ADR 0065's authorized R32 implementation smoke at train index
  93,000: 80 decisions produced 16,282 trajectory records and 2,351 grouped
  root records in 43.9 seconds. The 10.9 MB shard and manifest revalidated
  byte-for-byte, duplicate sparse features were preserved, and the exact MLX
  process shut down cleanly.
- Corrected ADR 0065's Pearson definition before any substantive index opened.
  Raw parent Pearson was already 0.99071 because personal turn dominates
  score-to-go, making the registered +0.02 gate impossible. The frozen gate
  now uses pooled within-personal-turn residual Pearson; the parent smoke
  value is 0.53179, while raw Pearson remains diagnostic.
- Implemented the complete MLX rollout-return pipeline: strict streaming `.nnv`
  decoder, differentiable six-tensor value model, exact-kernel validation,
  resumable AdamW checkpoints, phase and root-ranking metrics, immutable-parent
  checks, validation gates, and atomic schema-2 derived artifacts. An
  unchanged-value packaging smoke remained bit-identical to Rust on all 80
  fixture states.
- Rejected ADR 0065 on held-out validation before gameplay. The frozen R600
  datasets contain 245,603 train and 121,246 validation trajectory records.
  The selected epoch reduced RMSE from 5.1433 to 2.9912 and improved
  within-turn residual Pearson from 0.4185 to 0.4776, but root pairwise
  accuracy regressed from 0.71834 to 0.71682 and selected-action top-one
  regressed from 0.28125 to 0.27500. Gameplay seeds remain unopened. This
  closes pure trajectory-return fine-tuning and redirects the next model
  experiment toward an objective that explicitly preserves action ordering.
- Rejected ADR 0066's joint return and root-ranking successor. It reused
  only ADR 0065's immutable train split, adds group-centered root regression
  plus selected and soft teacher listwise losses, and evaluated on fresh
  validation indices 95,000-95,001. RMSE improved 42.56%, residual Pearson
  improved 0.0610, and selected-action top-one rose from 27.50% to 29.38%.
  However, pairwise accuracy fell from 70.91% to 70.57% and conditional regret
  worsened from 1.013 to 1.281. It was rejected before gameplay. This closes
  the exact centered-Huber plus selected/soft-listwise objective.
- Preregistered and implementation-qualified ADR 0067's public focal
  open-loop tree. Its integrity gate found and fixed a shared redetermination
  defect: fixed public sample seeds previously depended on the source hidden
  vector order. Unseen tile and wildlife multisets are now canonicalized
  before shuffling. All 50 rules tests, 60 search tests, two CLI tests, strict
  focused Clippy, release build, budget accounting, determinism,
  hidden-order invariance, legality, and replay checks pass. The registered
  runtime smoke then scored 93.000 versus 92.500 for strong (+0.500), with
  11.866 seconds per block and 689 ms P90 latency. Wildlife gained 0.250,
  non-Bear wildlife was flat, habitat hit the -0.500 guardrail, and tokens
  gained 0.750.
- Rejected ADR 0067 after its unchanged ten-block pilot. The public open-loop
  tree scored 92.375 versus 92.350 for corrected strong: +0.025 with 95% CI
  `[-0.856,+0.906]`. It gained 0.425 non-Bear wildlife and 0.200 habitat but
  lost 0.125 total wildlife, crossed the 92.750 rejection floor, and failed
  both score gates. Runtime was excellent at 14.782 seconds per block and
  898 ms P90 latency. Confirmation seeds `35000-35049` remain unopened.
- Preregistered ADR 0068's correction-only 50-block requalification of
  promoted strong against pattern-aware on fresh seeds `35100-35149`. The
  strategy remains frozen; only the now canonical public redetermination
  operator differs from its historical promotion run.
- ADR 0068 retained a clear score gain but failed requalification. Corrected
  terminal search scored 92.100 versus pattern-aware at 91.580: +0.520 with
  95% CI `[+0.260,+0.780]`. Bear gained 0.460 and total wildlife gained
  0.085, but Elk+Salmon+Hawk+Fox fell 0.375, violating the original balanced
  allocation gate. The strategy is demoted to explicit `research` status;
  pattern-aware is now the strongest promoted product policy. API schema 2
  names the unrestricted tier `research` instead of claiming `strong`.
- Preregistered and implementation-qualified ADR 0069's exact-parent
  candidate-set residual. The new checksummed `.imp` sidecar replayed and
  hash-matched all 15,360 smoke actions, exact MLX supplied every parent
  prior, and a one-epoch Apple-GPU run completed in 0.410 seconds with atomic
  checkpoint and gate reporting. The R2 one-game smoke is infrastructure
  evidence only; the frozen R600 validation, test, and gameplay domains remain
  unopened.
- Rejected ADR 0069 on its single authorized fresh validation run. The
  selected epoch improved loss, top-five recall, and teacher coverage, but
  failed six gates: top-one, MRR, pairwise accuracy, value-difference
  correlation, conditional regret, and train top-one. The weak train gain
  identifies representation underfit rather than validation-only overfit.
  Test and gameplay domains remain sealed.
- Completed and rejected ADR 0070 on its single authorized fresh validation
  run. All 614,400 hidden records aligned, the exact scalar parity and
  checkpoint reload gates passed, and epoch 30 reduced validation loss from
  1.522383 to 1.417843. Selected top-one improved only 0.078 percentage point,
  top-five 0.703 point, MRR 0.002335, pairwise accuracy 0.405 point, and
  regret 0.001245 point; train top-one was unchanged. Six gates failed. Test
  and gameplay domains remain sealed, and further residuals on this historical
  representation are closed.
- Completed the no-training MCE teacher identifiability audit over 5,120 train
  and 1,280 fresh-validation decisions. Only 18.359% of validation winners
  cleared a 95% normal difference test, 6.953% had non-overlapping intervals,
  and the mean 95% confidence set contained 10.140 actions. Opening decisions
  were only 6.563% distinguishable with 16.309 actions in the average set.
  Exact-action imitation of the independent-seed argmax is closed.
- Preregistered ADR 0071 before implementation. It changes only rollout seed
  coupling: the treatment shares an ordered seed prefix across candidates
  within each sequential-halving round at the unchanged K32/R600/LMR budget.
  Seed 35,699 is reserved for smoke and 35,700-35,702 for a conditional
  three-game pilot.
- Completed ADR 0071's implementation and gameplay gates. Independent search
  remained bit-exact against native Rust over 80 R32 decisions and three R600
  spots. The R600 smoke passed at +1.25. The three-game pilot scored 97.083
  versus 95.917, +1.167 paired with 95% CI `[+0.578,+1.756]`, a 3-0-0 record,
  +0.583 wildlife, -0.083 habitat, +0.667 Nature Tokens, zero fallback, and
  essentially unchanged runtime.
- Preregistered ADR 0072 before opening confirmation seeds 35,703-35,722. It
  requires 20 fresh paired games, paired gain at least +0.50, a strictly
  positive 95% lower bound, CRN mean at least 96.0, unchanged category and
  runtime guards, exact integrity, and no retries.
- Completed and rejected ADR 0072. Independent scored 95.775 versus 95.413
  for CRN, a -0.363 paired delta with 95% CI `[-1.129,+0.404]` and an 8-1-11
  record. Wildlife was -0.100, habitat -0.350, and Nature Tokens +0.088.
  All 3,200 selected actions were legal, both arms had zero fallback, runtime
  was 150.18 versus 148.34 seconds/game, and both services shut down cleanly.
  This closes same-budget CRN as a valid strength rejection.
- Implemented and rejected ADR 0073's fresh edge-aware MLX value model.
  Exact axial adjacency, directed terrain-edge matching, four-seat graph
  encoding, rotation augmentation, and within-round pairwise training all
  passed implementation and integrity tests. On 64 fresh games, correlation
  regressed from 0.393 to 0.342, MAE regressed by 0.257, and pairwise accuracy
  improved only 0.648 percentage point. Pairwise log loss and every wildlife
  guardrail passed. Batch-256 P90 inference was 0.346 ms per position on
  Metal. The sealed test and gameplay domains remain unopened.
- Implemented and completed ADR 0074's counterfactual target audit. The
  checksummed collector retained 2,560 complete H6 continuations across every
  state of two fresh games. R8 approximated R16 at 0.487 MAE, 91.14%
  within-round accuracy, and a projected 13.86-hour 256-game cost. The
  absolute target still failed its frozen width gate at 1.945 standard
  deviation versus 2.0 required. No training, test, or gameplay was opened.
- Implemented and rejected ADR 0075's grouped counterfactual-advantage target.
  The collector retained raw decomposed returns for 128 legal candidates under
  2,048 shared-seed complete continuations. R8 reached 0.274 centered MAE,
  0.855 correlation, 89.58% pairwise accuracy, 81.25% exact winner agreement,
  and 0.057 mean regret. The sole failed gate was mean group range: 1.367
  points versus 1.50 required. No train corpus, model, test, or gameplay was
  opened.
- Implemented and rejected ADR 0076's rank-stratified contrast target. The
  selected/high/median/low groups widened mean R16 range to 2.803 points; R8
  reached 0.353 centered MAE, 0.931 correlation, 85.42% pairwise accuracy,
  and 0.145 mean winner regret. Exact winner agreement was 62.50%, one group
  below the frozen 65% gate, so no corpus, model, test, or gameplay was opened.
- Implemented and accepted ADR 0077's R12 estimator on fresh validation. R12
  improved exact winner agreement from R8's 56.25% to 78.13% on the same
  groups and reduced mean regret from 0.283 to 0.037 while passing every
  fidelity, width, uncertainty, integrity, and cost gate. Train/validation
  corpus collection is authorized; test and gameplay remain closed.
- Preregistered and implemented ADR 0078's MLX R12 set-ranker pipeline. The
  strict memory-mapped reader, public-supply-aware zero-initialized model,
  uncertainty-weighted objective, resumable trainer, gate evaluator, CLI, and
  Make workflows pass 102 Python tests and 197 affected Rust tests. Two real
  four-group R12 implementation datasets completed 384 continuations; GPU
  train, checkpoint, resume, and deterministic best-checkpoint evaluation all
  passed.
- Invalidated ADR 0078's first substantive collection before training after a
  repeated mandatory four-of-a-kind chain exposed an unconditioned impossible
  chance branch in validation game 70,019. The complete 128-game train corpus
  and 19-game partial validation corpus are archived and prohibited from use.
  H6 rollouts and R12 continuations now deterministically rejection-sample
  only `WildlifeBagEmpty` trajectories, with the versioned contract bound into
  every new manifest. The exact failing game and the full MLX smoke both pass
  under the correction. Corrected substantive collection is active; test and
  gameplay remain closed.
- Added a live local cluster dashboard for john1, john2, and john3 with
  read-only SSH telemetry, normalized utilization, health, uptime, and active
  Cascadia workload visibility.
- Added durable cluster utilization history on john1: a 30-second background
  sampler, crash-tolerant seven-day JSONL retention, bounded 1D/7D API
  aggregation, and responsive CPU and memory charts verified in desktop and
  mobile Chrome.
- Replaced the temporary dashboard dev processes with a launchd-managed
  release server that serves the built frontend and API together at the
  stable Tailscale port 5187.
- Provisioned john2 and john3 as reproducible local workers under the
  distributed-compute goal contract. john1 is collecting ADR 0078 train
  indices 69,000-69,127, john2 is collecting disjoint validation indices
  70,000-70,031 with the exact collector binary, and john3 passed the locked
  Python 3.12.13 / MLX 0.31.2 GPU device and six-test ADR 0078 model preflight.
  The live execution record is
  `docs/v2/reports/adr-0078-distributed-execution.md`.
- Preregistered ADR 0079 before ADR 0078 validation completed. A complete
  validation pass conditionally opens only fresh test indices 71,000-71,031
  under the unchanged R12 contract and exact selected checkpoint. Failure
  keeps that corpus unopened; even a test pass does not authorize gameplay or
  promotion.
- Implemented the sealed-test handoff before validation completes. The
  supervisor now records proof that test data was absent on all three nodes
  before authorization, collects only after a full pass, and uses a frozen
  external MLX evaluator that checks checkpoint identity and replays
  validation bit-exactly. Seventeen focused tests pass, including transport,
  fail-closed recovery, evaluator, and module-size
  regression guard.
- Split the unattended cluster workflow into owned runtime, collection,
  transport, training, and sealed-test modules. john2 uses Tailscale first and
  an identity-pinned LAN route only after an SSH transport failure. The stable
  launchd entrypoint is 73 lines and every orchestration module remains below
  its enforced size limit.
- Closed the v2 debt gate: all v2 Rust packages pass strict no-dependency
  Clippy with warnings denied, Python and frontend lint are clean, and the
  oversized CLI, search, policy-improvement, oracle, and pattern modules are
  split by ownership. Structural tests cap the CLI entrypoint at 300 lines and
  active v2 Rust production modules at 1,500 lines.
- Moved the superseded v1 rules, AI, CLI, embedded web app, historical scripts,
  and historical reports under `legacy/`. Production v2 crates remain
  independent; only the test-only differential boundary can import v1.
- Re-ran the complete format, lint, and test gate after the dashboard,
  orchestration hardening, and source decomposition: 223 Rust tests, 119
  Python tests, and 7 frontend tests passed with zero failures.
- Added a single troubleshooting entrypoint covering uv/MLX, SSH workers,
  resumable artifacts, launchd services, ports, browser evidence, and
  reproducibility escalation. All local Markdown links resolve.

## Evidence Collected

- Repository size: approximately 110 GB.
- Git directory: approximately 449 MB.
- Generated AlphaZero v2 runs: approximately 23 GB across two primary run
  directories.
- Current Rust source surface: approximately 44,000 lines.
- Runtime configuration reads in AI/CLI: 105.
- Existing default test run observed before branch creation:
  181 tests passed, zero failed.
- Existing AlphaZero-v2 feature-gated suite observed before branch creation:
  32 tests passed, zero failed.

Those test results establish that the current tree is executable; they do not
certify the v2 rules or benchmark contract.

## Immediate Work

1. Finish and independently validate the active distributed 128-game train
   plus 32-game validation R12 rank-stratified corpus under ADR 0078's
   explicit stable-market conditioning contract.
2. Train the preregistered MLX complete-candidate-set ranker while
   retaining the shared-seed sampler, post-prelude public state,
   explicit public supply and opponent-board context, raw decomposed returns,
   uncertainty, and exact immediate score.
3. Apply the frozen validation branch: leave ADR 0079 unopened on any failed
   gate, or execute its already sealed 32-game test and one validation replay
   after a complete pass.
4. Keep the independent exact-MLX K32/R600 teacher as the 95.775 local
   research reference; do not collect CRN labels or reopen ADR 0073's test
   domain.
5. Continue toward the 100-point final gate; the historical MLX port remains a
   research teacher, not a final solution.

## Promotion State

No v2 neural model has been promoted to gameplay. All `compact-entity-v1`
learned-action results are retained as historical measurements but are
methodologically superseded by the hidden-refill finding. Exact greedy is the
production `instant` strategy, `pattern-aware-v1-k8-h6-b8-m4` is the promoted
`interactive` strategy and strongest promoted product policy, and
`late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`
remains available only through the explicit `research` tier. H6 remains a
confirmed research teacher at 91.760 mean. Historical weights and reports are
reference artifacts only.
