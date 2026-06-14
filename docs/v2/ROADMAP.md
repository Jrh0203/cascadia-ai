# Cascadia V2 Roadmap

## Phase 0: Evidence

- [x] Adopt authoritative goal.
- [x] Create v2 branch.
- [x] Inspect and clear stale compute.
- [x] Record hardware/toolchain.
- [x] Audit v1.
- [x] Define architecture and benchmark contract.
- [x] Establish project-owned MLX environment and run Metal smoke/performance.
- [x] Build canonical v1 adapter and reproduce baseline.
- [x] Record fresh v2 random and greedy correctness/performance artifacts.
- [x] Record a timed, explicitly non-canonical v1 champion reference artifact.

## Phase 1: Deterministic Game

- [x] Create independent v2 workspace crates.
- [x] Define canonical domain types and `TurnAction`.
- [x] Implement pure validation and transactional apply/undo.
- [x] Implement deterministic setup and separated RNG streams.
- [x] Implement stable snapshots, replay, and hashing.
- [x] Port independently verified AAAAA scoring.
- [x] Add fixtures, property tests, and v1 differential tests.

## Phase 2: Simulation And Evaluation

- [x] Implement random/greedy baselines and evaluate a pattern-aware pilot.
- [x] Implement symmetric parallel match runner.
- [x] Implement seed-suite artifacts and game-block statistics.
- [x] Add paired strategy-comparison statistics.
- [x] Add deterministic criterion benchmark targets.
- [x] Record uncontended criterion measurements on this machine.
- [x] Generate machine-readable reports.
- [x] Generate Markdown reports from the same result artifact.

## Phase 3: Web Product

- [x] Create typed API application layer.
- [x] Build standalone responsive frontend.
- [x] Implement setup, play, undo/redo, save/load, analysis, and accessibility.
- [x] Add browser tests and desktop/mobile visual verification.

## Phase 4: MLX Foundation

- [x] Create `cascadia_mlx` package and one-command environment setup.
- [x] Define dataset/model/run manifests and sharded data format.
- [x] Implement encoders, model service, training, validation, and resumption.
- [x] Train fresh policy/value baselines on local hardware.

## Phase 5-6: Strength Campaign

- [x] Quantify candidate recall and score-loss sources.
- [ ] Run registered architecture, search, distillation, and self-play studies.
- [x] Audit R600 teacher winner identifiability before another distillation
      run; close exact-action imitation after only 18.4% of validation winners
      cleared a 95% difference test.
- [x] Test same-budget common-random-number sequential halving under ADR 0071;
      pilot passed at +1.167 and 97.083 mean.
- [x] Confirm CRN over 20 fresh paired games under ADR 0072; rejected at
      -0.363 with 95% CI `[-1.129,+0.404]`.
- [x] Test explicit hex adjacency and oriented terrain-edge message passing
      under ADR 0073; rejected on fresh validation after correlation and MAE
      regressed despite a small pairwise log-loss improvement.
- [x] Audit repeated counterfactual public-state returns under ADR 0074; R8
      was stable and locally affordable, but absolute R16 state means failed
      the frozen signal-width gate.
- [x] Audit nearest-neighbor same-decision action advantages under ADR 0075;
      R8 was stable and accurate, but the top-four R16 range missed the frozen
      1.50-point width gate at 1.367.
- [x] Audit rank-stratified counterfactual contrasts across the existing H6
      frontier; width passed at 2.803 points, but R8 exact winner agreement
      missed its frozen gate by one of 32 groups.
- [x] Qualify an R12 rank-stratified shared-seed estimator on fresh validation
      games; all gates passed at 78.13% exact winner agreement and 0.037 mean
      regret.
- [x] Diagnose and permanently correct ADR 0078's finite-market collection
      failure; archive the unconditioned corpus, version deterministic
      rejection conditioning in the teacher manifest, and replay the exact
      failing game plus the full MLX smoke.
- [ ] Complete the corrected ADR 0078 collection of the authorized 128-game
      train and 32-game validation R12 corpus on the local john1/john2
      cluster. Both disjoint splits are actively collecting.
- [ ] Train the implemented MLX public-supply-aware complete-candidate-set
      ranker and apply every frozen validation gate.
- [x] Preregister ADR 0079's fresh 32-game sealed test before ADR 0078
      validation is known; keep it unopened unless every validation gate
      passes.
- [x] Implement ADR 0079's conditional authorization, exact collector
      handoff, frozen external MLX evaluator, and bit-exact validation replay
      before any validation metric or test record exists.
- [x] Promote only confirmed paired improvements.
- [x] Confirm positive late-terminal policy-improvement signal without
      weakening frozen mechanism guardrails.
- [x] Requalify the confidence-gated policy after canonical redetermination;
      demote it to explicit research after it failed the original non-Bear
      allocation guardrail.
- [ ] Reach and validate the 100-point target or document the strongest factual
      result with remaining uncertainty.

## Phase 7-8: Product Completion

- [x] Meet versioned latency and throughput budgets for instant, interactive,
      research, and batch-32 MLX execution.
- [x] Eliminate warning/debt backlog in v2. Compiler/lint debt is clean, the
      CLI and research monoliths are split by ownership, and structural tests
      prevent regression.
- [x] Complete documentation and one-command workflows, including generated
      CLI reference freshness checks.
- [x] Move v1 crates, historical scripts, and reports to the explicit
      `legacy/` reference boundary.
- [ ] Rehearse clean checkout.
- [ ] Run 1,000-game held-out validation and publish final report.
