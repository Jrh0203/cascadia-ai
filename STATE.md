# Cascadia V3 Overnight Campaign State

Last updated: 2026-06-24 02:53 EDT (live system clock)

## Objective

Regenerate the Part 1 readiness package from the optimized V3 code, authorize Phase 2 with the new checksum, then execute the bootstrap and ten 10,000-game expert-iteration cycles under `docs/v3/CASCADIA_V3_RESEARCH_SPEC.md`. Produce an overnight report before John returns.

## Current state

- Campaign controller: `cycle-01-training` (Part 2). Cycle 1 collection, verification, and exact teacher labeling are complete; the first of two independent MLX candidate origins is training on John1 while John2/John3 benchmark the frozen parent.
- Protected seeds: sealed. Bootstrap collection, replay verification, deterministic root selection, exact labeling, repair, corpus reconciliation, three-origin training, selection, parity, and champion freezing are complete.
- Green readiness checksum: `b15e44af519060936416af23e9af1795dd37149309a3f65bc86941d5399b3ebe`; every registered gate passed.
- Phase 2 was checksum-authorized by John at 2026-06-21 01:24 EDT.
- Revised measured projection: 4.30 active days and 9.08 GiB under the 10K-cycle/80%-V1 contract.
- Docker daemon: Colima is running. Its daemon is explicitly configured for the HTTP-only John1 research registry.
- Final canonical expert-cycle image was built and published by John1 with embedded Rust parity/worker tests passing: `100.110.109.6:5000/cascadia/v3-worker@sha256:2037941a6a0eeaba162e38b6e1d1232332ad384fda2cb65521fd59f7393d5923`.
- Bootstrap origin 3 is the registered winner: validation power loss `0.00205615`, RMSE `6.38273`, MAE `4.84128`, open mean `85.60156`, and serving weights `c162d67835327abffc8f9b14850a58b9fa3bf816f86fb4d7c2ea58718d1223fe`.
- Winner parity passed across 6,400 rows: Rust/MLX quantized and scalar/NEON outputs were bit-identical; float/quantized top-1 and top-32 agreement were both 100%.
- Completed bootstrap collection image: `100.110.109.6:5000/cascadia/v3-worker@sha256:0aebd73680246cb09fc9266c95f17c1e3aacf3b1101247b631be224ddd8513b1`.
- Canonical repaired and bounded-stream worker image: `100.110.109.6:5000/cascadia/v3-worker@sha256:b93906ca67aeb6a4ada0ec801cf72724f590b62b9fca086c7fa0b7f9bed3f70d`.
- Canonical publication receipt: `/Users/johnherrick/cascadia-bench/v3-nnue/smoke/image-publication-stage2-bounded-stream-v1.json`.
- `STATE.md` is explicitly excluded from source-identity hashing, so operational progress logging cannot create false code drift.
- Live Bacalhau audit: John1, John2, and John3 are connected at 9/10/10 CPUs; John4 remains excluded from compute.
- An excluded one-game Docker preflight and a real Bacalhau/S3 preflight on John3 both passed with the active image. The latter returned a validated V1-direct replay artifact in one attempt.
- The preflight exposed and fixed a transport-boundary bug: registered `CASCADIA_*` artifact-envelope variables are now explicitly classified as evaluator-neutral, while all unregistered policy knobs remain rejected. The focused regression and embedded worker tests pass.
- The Part 1 dashboard verifier now supports the live schema-v3 `nodes` array as well as the legacy host-map envelopes; its focused tests pass and the live infrastructure receipt is green.

## Active work

1. Durable request `v3-bootstrap-collection-b15e44af-0aebd736-v1` completed all 500,000 games in all 250 scheduler-owned shards with zero failures in 4,638.59 seconds (77.31 minutes).
2. The deterministic 120,000-root selector/splitter, exact-budget K32/R600 label worker, and label-to-training converter are implemented. Smoke evidence proves 600 rollouts exactly and 32 training rows from the sampled root.
3. The second John1-built immutable image containing the exact teacher-label path passed its embedded 27-test V3 suite, four worker-policy tests, worker health check, and publication identity checks: `100.110.109.6:5000/cascadia/v3-worker@sha256:ab2d49849931f7a535c5b313f5f1a5fc36663a85e6b6eb3252b67d70d731af3d`.
4. Replay verification completed all 250 shards with exactly 500,000 records and 40,000,000 replay-expanded training entries. The compact corpus manifest passed and the campaign atomically advanced to `bootstrap_labeling`.
5. Deterministic root selection completed across the 500K-game corpus: 40,000,000 positions considered, 640,786 oversampled candidates, 6,407 strata, exactly 100,000 teacher roots, and 20,000 validation roots. The 120 immutable root shards passed their BLAKE3 manifests. Of the 120 exact K32/R600 label jobs, 111 produced complete validated shards and nine were rejected as partial (`38,40,42,43,44,50,54,68,110`). Their legal V2 boards reached absolute q/r coordinate 11, outside the historical V1 21x21 storage origin.
6. A permanent lossless coordinate-frame adapter is implemented for the legacy teacher: each player board is translated independently into V1's fixed window, zero coordinates are retained throughout the already-qualified domain, selected actions are inverse-mapped exactly, score parity is checked after translation, and any truly unrepresentable span over 20 cells is rejected rather than clipped. The complete 50-test differential library passes. The first actual failed root (validation shard 10, root 22, maximum absolute coordinate 11) now completes all 600 rollouts with 32 legal candidate estimates, zero fallbacks, and exact score/action checks.
7. Repair request `v3-bootstrap-label-repair-b15e44af-1d5629dd-v1` completed all nine rejected shards: 9,000 roots, 5.4 million exact rollouts, and 245,912 candidate estimates in 2,532.64 seconds. The strict reconciliation gate accepted exactly 111 original plus nine repaired shards, yielding 120,000 roots and 72,000,000 rollouts. Corpus manifest SHA-256: `6d6e6207347ad80031c7e798e3f45688aaa18b3b0619078d062d1c62907b5960`.
8. The original bootstrap recipe is superseded. Although `5e-4` passed calibration and origin gates at 4.5M and 5.008M exposures, it crossed the historical int16 pre-activation storage boundary at 9M. The complete lineage is checksum-verified on John2. Serving now retains exact int32 pre-activation sums and clips only at the int16 activation boundary, matching MLX QAT mathematically; QAT also includes an explicit maximum-absolute own/field headroom penalty from step zero. All three learning-rate calibrations will rerun before any replacement origin is admitted.
9. The ten-cycle controller, promotion harness, protected final comparison, all-V3 evaluation, and final report path are implemented and tested. During each John1 MLX cycle-training phase, two 10-CPU parent benchmarks occupy John2 and John3; the CPU request makes John1 ineligible by construction.
10. Final calibration lineage `aaaf6b4de850aa7827f21d826562951452ab4101a1181584e37386d7f02f79d2` is live under launchd supervision. At 11:22 EDT the MLX trainer and native stream were both active; the stream was using roughly 3.5 CPU cores, MLX had a 6.4-GiB physical footprint, John1 retained 56 GiB free, and no first checkpoint had yet been published. The dashboard's zero-step display is expected until the exact 4M-exposure atomic calibration checkpoint appears.
11. A 5-second live sample at 11:39 EDT proved the current calibration is input-bound in opportunity-graph feature expansion, not stalled: the native stream is continuously using about 3.6 CPU cores while MLX waits for batches. The permanent source now assigns all eight reserved preprocessing threads to single-source bootstrap blocks, splits them 4/4 for mixed bootstrap blocks, and deterministically apportions the same eight-thread ceiling across expert-cycle sources. This removes broad-only idle capacity and prevents five-source cycle oversubscription. Nineteen focused tests and Ruff pass. Calibration 1 continues on its already-loaded four-thread process; subsequent processes receive the corrected budget without changing example order, targets, or model math.
12. Headroom-QAT calibration 1 completed its exact 4,000,000-example gate in 5,857.06 active seconds with final recorded loss 0.0518383. Its quantized serving export (`a60b3c8cbd3c0ce0c0d58dccf7623b1ef39b8d55a9136d8836167cde48bbe0dc`) passed both fixed Rust games. The admitted cached validation evaluated 554,624 source-pinned rows in 242.28 seconds: power loss 0.00421064, RMSE 8.24280, MAE 6.73893. The 32-game open-domain mean was 80.34375 with no serving failure. Calibration 2 began automatically.
13. Repeated validation feature construction is removed from the critical path. John1 published Docker image `100.110.109.6:5000/cascadia/v3-worker@sha256:a13551652256ce05ef9ffe40da3484dd516ddbe07e3a771ec696d1ad8fa3bff6`; all 20 Bacalhau jobs completed in 45.61 seconds. The admission gate correctly uses each immutable label receipt's actual up-to-32 candidate count: 20,000 roots, 554,624 rows, 9,651 realized rows, 544,973 counterfactual rows, and 1,898,634,630 bytes. Every source, receipt, and packed-shard hash passed. A nine-minute partial compact evaluation was preserved then replaced without repeating the completed 4M training checkpoint.

## Material progress log

- 02:03 EDT — Teacher-label conversion smoke passed: one labeled root produced one realized and 31 counterfactual rows. Runtime teacher-lambda annealing is implemented and unit-tested so the immutable labels can be reused from lambda 1.0 through 0.75.
- 02:09 EDT — Published the exact-label worker image without changing or restarting the live collection request. Publication receipt: `/Users/johnherrick/cascadia-bench/v3-nnue/smoke/image-publication-stage2-teacher-label-v1.json`.
- 02:24 EDT — Froze the bootstrap MLX schedule at `/Users/johnherrick/cascadia-bench/v3-nnue/control/bootstrap-training-schedule.json` (current SHA-256 `233557bb84538a5cb5f7e3492dfb48e1702e93e71753898a00b0d4dcaa686732`). It totals exactly 120M approved exposures: three 4M learning-rate trials plus three 36M origins, with 12 bounded blocks, lambda 1.0→0.75, exact phase balancing, online D6, atomic block checkpoints, and final-20% SWA.
- 02:24 EDT — Added and tested the durable Phase 2 replay-verification/label executor. It rejects any fabric other than connected John1/John2/John3 at 9/10/10 CPUs and uses 0.75-GiB one-core jobs so labeling can occupy all 29 authorized CPUs.
- 02:27 EDT — The new exact-label image passed a real Bacalhau pull, container health run, S3 publication, artifact validation, and import in one attempt (`v3-exact-label-health-ab2d4984`).
- 02:35 EDT — The live dashboard now overlays scheduler-owned bootstrap progress (completed/total games, work-item states, throughput, and ETA) without mutating the campaign state chain. The API process classifier no longer mistakes Codex transcript text for an active MLX job; the focused Rust and dashboard tests pass, and the release service was restarted successfully with John1–John4 visible.
- 02:41 EDT — Removed a projected ~10-GiB teacher-data expansion from the critical path. MLX now streams compact replay-authoritative `.v3l` labels and expands candidate features on demand with exact phase/D6/lambda scheduling. The prior one-root converter remains only a parity oracle; full science will not materialize redundant sparse rows.
- 02:36 EDT — Bootstrap collection completed 250/250 shards and 500,000/500,000 games with zero failed work items. The completion receipt is preserved at `/Users/johnherrick/cascadia-bench/v3-nnue/phase2/bootstrap/collection/completion-receipt.json`.
- Live checkpoint — Closed the bootstrap training handoff: planned calibration stops now cap the native stream at exactly 4,000,000 exposures rather than overshooting a batch boundary; checkpoint evaluation measures the QAT/quantized validation loss directly from compact labels; and calibration/origin selection may choose on validation loss only among candidates noninferior to the strongest same-seed open-game candidate.
- Live checkpoint — Added and tested the autonomous bootstrap-training runner (`tools/v3_bootstrap_train_pipeline.py`) and launched it under a macOS sleep guard. Focused lint, unit tests, release builds, and a two-game direct-policy execution smoke all pass with zero swap growth.
- 02:39 EDT — Fixed the bootstrap barrier to validate the monitor's actual completion contract (`work_items=250`, `games=500000`, empty failures) rather than a nonexistent `succeeded` field. The first failure receipt was archived intact and the runner resumed from the completed shards; no games were regenerated.
- 02:39 EDT — Published the complete expert/final image after 27 V3 library tests and six worker tests passed. A real two-node Bacalhau preflight materialized the split model bundle, loaded V1 and V3, rotated exactly one V3 focal seat, and returned two validated games.
- 02:39 EDT — Verified the focal-only expert root selector against four complete games: exactly 80 focal afterstates were considered, not 320 all-seat afterstates.
- 02:50 EDT — Published the final canonical expert/final image with complete parent-benchmark anatomy. Each John2/John3 training-side benchmark is bounded to 500 games and reports mean, P10/P50/P90, score histogram, per-animal means, per-terrain means, Nature Tokens, and Pinecones. The ten-cycle controller is running under `caffeinate` and waiting at the bootstrap gate.
- 02:53 EDT — The final image passed a real two-node Bacalhau model-bundle and focal-benchmark preflight. Both 10-CPU jobs succeeded and returned score/anatomy summaries from the immutable image.
- 03:00 EDT — Replay verification completed 250/250 shards in 1,339.19 seconds. Totals were exact: 500,000 records, 40,000,000 expanded training entries, and 528,294,561 source bytes. The corpus manifest passed and state transition 7 advanced to `bootstrap_labeling`.
- 03:01 EDT — Archived four obsolete Part-1 smoke checkpoints (4,769,923,072 logical bytes) to `/Users/john2/cascadia-archive/v3-nnue-part1-smoke`. Two independent `rsync --checksum --itemize-changes` comparisons returned an empty diff before local removal. The latest exact-resume checkpoint for each smoke run remains on John1. Campaign footprint fell from 8.0 GiB to 3.6 GiB and free disk rose to 61 GiB.
- 03:02 EDT — Added a tested checkpoint lifecycle for Phase 2: completed runs compact to one exact-resume checkpoint; completed calibration and losing-origin optimizer state retires only after immutable evaluation and serving exports exist; promotion retains only the current champion's resume state. Focused V3 Python coverage is now 61 passing tests.
- 03:13 EDT — Deterministic root selection and splitting completed: 40,000,000 positions considered, 640,786 oversampled candidates, 6,407 strata, exactly 100,000 teacher roots and 20,000 validation roots, split into 120 checksum-pinned 1,000-root shards. Exact K32/R600 labeling then launched as 120 scheduler-owned jobs on John1–John3; the initial scheduler state was 27 running and 93 queued, with John4 excluded.
- 03:37 EDT — Corrected the live dashboard contract for nonbenchmark scheduler phases. The V3 mirror had become `invalid` because generic labeling work items were incorrectly required to equal benchmark pairs. The Rust reader now validates generic scheduler progress independently unless a benchmark is active, and the web command deck shows distributed work items, scheduler states, retries, and phase progress. Eleven API tests, seven focused UI tests, Clippy, the production web build, and release API build passed; the service restarted and `/api/v1/cluster/r2-map` is `fresh` with `bootstrap_labeling`, 29 running and 91 queued. Running science was not restarted.
- 03:37 EDT — Labeling remains fully CPU-bound and healthy: John1 is near 74% CPU and John2/John3 near 99%, with 27 CPU allocations visible (7/10/10; two admitted jobs await capacity) and no failed terminal items. The registered capacity model allocates about 8.6 hours to the 120K bootstrap-label share; the first phase-ordered shards contain expensive early-game roots, so the absence of a completed 1,000-root shard in the first 24 minutes is not a stall.
- 03:59 EDT — The first exact-label wave produced seven successful immutable 1,000-root shards after 45.99 minutes. Scheduler state is now 7 succeeded, 29 active/admitted, 84 queued, zero failures. Read-only inspection before publication confirmed every running container's `.v3l` file was growing (roughly 1.3–1.8 MB), independently ruling out a hidden stall. The seven John1 jobs finished first because Bacalhau currently has 7/10/10 physical containers active; the controller immediately replenished John1 with seven new shards without restarting any completed work.
- 04:10 EDT — The complete first physical label wave passed: 27/120 shards and 27,000/120,000 roots succeeded in 3,411.36 seconds, with 29 active/admitted, 64 queued, and zero failures. Observed aggregate throughput is 7.91 roots/s; a constant-rate projection leaves about 3.27 hours, and later-phase roots are expected to be cheaper. This is materially faster than the preregistered 8.6-hour bootstrap-label allowance.
- 04:42 EDT — Exact labeling advanced to 38/120 shards and 38,000/120,000 roots, with 29 active/admitted, 53 queued, and zero failures after 5,342.39 seconds. Rolling aggregate throughput is 7.11 roots/s and the constant-rate remaining estimate is about 3.20 hours. The durable request continues to replenish capacity automatically.
- 05:07 EDT — Exact labeling reached 54/120 shards and 54,000/120,000 roots after 6,823.41 seconds. All 54 terminal items succeeded; 29 remain active/admitted and 37 queued. Aggregate throughput recovered to 7.91 roots/s, with a dashboard ETA of 8,340 seconds (2.32 hours). Campaign footprint remains 3.8 GiB and John1 retains 60 GiB free, above the 50-GiB floor. The training and expert controllers remain live at their barriers.
- 06:02 EDT — Exact labeling reached 81/120 shards and 81,000/120,000 roots after 10,121.35 seconds. All terminal shards succeeded, 29 are active/admitted, only 10 remain queued, and there are still zero failures or retries. Aggregate throughput is 8.00 roots/s and the live remaining estimate is about 4,873 seconds (1.35 hours). Storage remains 3.8 GiB with 60 GiB free.
- 07:08 EDT — Scheduler execution ended after 14,078.56 seconds, but result validation correctly rejected nine application-level failures that Bacalhau had classified as completed transport jobs. The 111 complete immutable shards remain valid. The nine affected item indices are `38,40,42,43,44,50,54,68,110`; every failure was a legal V2 coordinate at q/r 11 crossing the retained V1 engine's fixed `[-10,10]` origin window. The partial `.v3l` files and failure receipts are preserved and excluded.
- 07:14 EDT — Implemented and tested the root-cause repair, not a fallback: deterministic per-board coordinate frames with exact inverse action mapping, translation-invariant score checks, zero-frame preservation on the prior qualified domain, and explicit rejection of genuinely unrepresentable spans. All 50 differential library tests pass. A local R600 replay of the exact first failed validation root succeeded in 0.128 seconds with maximum absolute coordinate 11, 44/44 expanded candidates legal, 32/32 prefiltered candidates legal, 600 rollouts, and zero fallbacks. Campaign footprint is 4.1 GiB and John1 has 59 GiB free.
- 07:23 EDT — John1 published the canonical coordinate-frame worker image `100.110.109.6:5000/cascadia/v3-worker@sha256:1d5629dd59d6086f8d22ffa14a7117ba1925fb0ee74b0d49cfa44445196154fa`; its embedded 27 V3 library tests, six campaign-worker tests, and health check passed. Repair request `v3-bootstrap-label-repair-b15e44af-1d5629dd-v1` launched exactly the nine rejected root shards, with all nine running across the authorized scheduler fabric. A tested reconciliation gate will require exactly 111 original plus nine repair shards, validate every root/receipt/label checksum, preserve both image lineages, and hard-link only complete artifacts into the training corpus. The expert-cycle controller was restarted at its bootstrap barrier with this repaired canonical image so later mostly-V1 collection cannot encounter the same coordinate-window defect.
- 08:05 EDT — All nine repair jobs succeeded. The reconciler validated every source-root hash, label artifact, receipt, request lineage, and image identity before hard-linking the complete 120-shard training corpus. Totals are exact: 120,000 roots, 72,000,000 R600 rollouts, and zero admitted partial shards. Campaign state atomically advanced to `bootstrap_training`; protected seeds remain sealed.
- 08:07 EDT — John1 began calibration trial 1 (`lr=5e-4`, batch 8,192, exact 4M-exposure stop) using the compact replay/teacher stream, online phase balancing, D6 augmentation, and QAT. The pipeline, MLX process, native Rust streamer, sleep guard, and expert controller are live. The stale pre-repair training-waiter failure receipt was archived under `phase2/bootstrap/training/failure-history`; it is not a current failure.
- 08:10 EDT — A live process sample proved the first calibration was not using Metal productively: the native loader spent its wall time serially rebuilding opportunity graphs on one CPU and had emitted no batch, checkpoint, loss, or parameter update. The attempt was interrupted before scientific state changed and preserved under `phase2/bootstrap/training/interrupted-history` with its failure receipt.
- 08:19 EDT — Qualified the permanent stream fix. Compact games and labeled roots now expand through bounded deterministic Rayon chunks; broad/teacher stream headers flush immediately; and compact games avoid one redundant complete replay. At 8,192 rows, game-stream construction improved 4.83→1.55 seconds (3.12×), the rare-softmax case improved 6.09→1.90 seconds (3.21×), and immediate header publication makes the two-source cold-start projection 6.41× faster. One-thread and four-thread game/teacher streams are byte-identical. Two batch-stream tests, 27 V3 library tests, six container worker tests, and the container health check pass.
- 08:20 EDT — John1 published the complete source as canonical image `100.110.109.6:5000/cascadia/v3-worker@sha256:ce0e7c6f3b4cb5fd1c1ed01a166d976d44181988c927527a9784dafada47b08a` and restarted calibration trial 1 from exposure zero. This is not lost completed work: the interrupted attempt had produced no update or checkpoint. MLX now has a live Metal evaluation stack and the native producer is concurrently reconstructing future batches.
- 08:27 EDT — Sustained-load profiling exposed an unbounded uniform-phase sampler: common phase buckets retained every surplus row while rarer phases governed emission. Producer RSS reached 1.0 GiB and global swap grew monotonically. The attempt was stopped before its first atomic checkpoint. The exact emitted-phase contract is now preserved with deterministic 64-row per-stratum queues; surplus inputs are rejected rather than retained indefinitely. D6 and target preparation also moved into the parallel stage.
- 08:30 EDT — The final bounded producer processed 100,000 balanced examples in 17.07 seconds (5,858 examples/s), peaked at 132,333,568 resident bytes, and incurred zero process swaps. One-thread/four-thread game output and teacher output remain byte-identical. John1 published final canonical image `100.110.109.6:5000/cascadia/v3-worker@sha256:b93906ca67aeb6a4ada0ec801cf72724f590b62b9fca086c7fa0b7f9bed3f70d`; the embedded 27-library-test, six-worker-test, and health gates passed.
- 08:50 EDT — Calibration reached its exact 4M planned stop, then correctly refused the atomic checkpoint because repeated image builds and parity outputs had pushed John1 just below the 50-GiB free-space floor. No partial checkpoint was published. Only V3-generated temporary streams, superseded local Docker layers already preserved in the registry, and 12.1 GiB of rebuildable Cargo dev artifacts were removed. John1 now has 59 GiB free; campaign data remains intact.
- 08:52 EDT — Added and tested a start-of-run storage preflight to both bootstrap and cycle trainers, while retaining every per-checkpoint guard. Future unsavable runs now fail before MLX allocation or corpus consumption. Six focused trainer tests and Ruff pass. Calibration trial 1 restarted from exposure zero with the bounded producer; the expert controller remains safely at the bootstrap gate.
- 09:10 EDT — Calibration 1 (`lr=5e-4`) completed and atomically checkpointed exactly 4,000,000 broad exposures in 1,074.34 seconds (3,723 examples/s end to end), 489 batches, and 57 GiB free remaining. Batch loss moved from 0.22260 to 0.04989. Quantized evaluation passed over 554,624 validation examples at 3,496.5 examples/s: power loss 0.00430596, RMSE 8.47472 score points, MAE 6.99336. Its 32-game open-domain smoke mean was 80.1953 with zero overflow states. Calibration 2 (`lr=1e-3`) launched automatically; no current failure exists.
- 09:37 EDT — Calibration 2 (`lr=1e-3`) completed its exact 4,000,000 exposures in 1,065.71 seconds and wrote an atomic checkpoint, but direct-game evaluation deterministically rejected the serving export with `AccumulatorOverflow`. This is a scientific candidate rejection, not a campaign failure. A durable `evaluation-failure.json` records the command, log digest, and return code. The bootstrap controller now tolerates immutable evaluation failures, excludes those candidates from selection, and resumes without rerunning them; seven focused pipeline tests and Ruff pass. Calibration 3 (`lr=1.5e-3`) launched automatically.
- 09:39 EDT — The dashboard training overlay now publishes a model as `latest_verified_checkpoint` only after its complete evaluation has `passed=true`; a numerically invalid but atomically written checkpoint can no longer replace the last serving-qualified model. Eight dashboard tests and Ruff pass, and the live projection was refreshed.
- 09:57 EDT — Calibration 3 (`lr=1.5e-3`) completed exactly 4,000,000 exposures in 1,071.08 seconds and failed the same deterministic serving `AccumulatorOverflow` gate. The immutable selection chose calibration 1 at `5e-4`, the only rate with a valid quantized serving bundle; protected data remains sealed.
- 10:00 EDT — Fixed the completed-selection resume path exposed by the two scientific rejections. Rejected runs now have an explicit lifecycle that requires a passed training report, immutable evaluation-failure receipt, checkpoint manifest, and serving manifest before optimizer state can be retired. Failure evidence remains intact. Eleven focused lifecycle/pipeline tests and Ruff pass; calibration storage fell from roughly 2.5 GiB to 315 MiB and John1 returned to 57 GiB free.
- 10:07 EDT — Added a bounded real-serving guard to every bootstrap atomic checkpoint. The in-memory model is quantized and exported, two fixed open-domain games execute through Rust integer inference, and a checksum-bound pass/failure receipt is written before training continues. A failure stops at the latest exact-resume checkpoint rather than wasting the remainder of a 36M origin. The first pre-checkpoint origin attempt had no update or checkpoint and was archived before this contract was added. Sixteen focused Python tests, Ruff, all 27 V3 library tests, and a release build pass. Bootstrap origin 1 restarted from exposure zero under the new guard.
- 10:30 EDT — Bootstrap origin 1 reached 4,500,000 exposures in 1,264.37 active seconds and passed its first real quantized-serving checkpoint gate. The exact-resume checkpoint model digest is `cf7d24c4aea1fda8b36b23c6f5492a3658496dad96309eee5604443a5c89016f`; the exported serving weights digest is `fe34b6544f189381b1c3c9732d25da323db20c61453d2ddc762b8f5425b4fc45`. Both fixed open-domain games completed without overflow (eight seat-scores, mean 81.75), and block 2 began automatically. The dashboard now publishes checkpoint-integrity-qualified active checkpoints and computes ETA by summing per-run elapsed time; nine tests and Ruff pass. It reports 16.5M/120M total scheduled exposures and a measured bootstrap remainder near 7.8 hours.
- 10:36 EDT — Origin 1 also passed its 5,007,904-exposure periodic checkpoint gate; serving weights digest `ea5a652e546c172c3fcd96e87b004a58ad701cefbc8367092406be5109f8cdd5`, two-game/eight-seat mean 81.375, no overflow. Checkpoint game reports now use exposure-specific immutable filenames and record their BLAKE3 digest plus score vector. The bootstrap controller already has an advisory single-writer lock; nine focused controller tests pass. A transient duplicate launch was detected before another checkpoint could be published, both speculative workers were stopped, and exactly one lock-owning controller resumed from the verified 5,007,904 checkpoint. The obsolete pre-final-image expert controller was stopped; no cycle can start until John1 publishes and explicitly restarts the final canonical worker image.
- 10:39 EDT — The repeated short-lived controller exits were traced to interactive tool-session SIGTERM, not MLX, data, or checkpoint failure. The single-writer controller is now genuinely headless under detached `nohup` plus `caffeinate`; a 90-second post-launch survival check confirmed the same PID still owned the advisory lock and its one trainer/one native stream remained active. Global swap decreased during the check (14,182.5→14,102.5 MiB) and memory pressure reported 71% free. Training continues from the verified 5,007,904 checkpoint without replaying prior exposures.
- 10:58 EDT — The new guard caught a genuine scientific rejection at exactly 9,000,000 origin exposures: quantized Rust inference raised `AccumulatorOverflow` on the fixed open-domain games. The rejected serving weights digest is `b8c081bd2ed743ad784dda2ea8670666d34a55cb79f30d6c656bbde0d866f5f0`. The checkpoint, serving bundle, stderr, optimizer state, and failure receipt were preserved; protected seeds remain sealed. No continuation from that checkpoint is permitted.
- 11:04 EDT — Implemented the root training-contract repair. QAT now adds a differentiable headroom term over each example's maximum absolute own/field accumulator, with limit 64 float units (16,384 integer units, half the int16 range) and coefficient 0.1. Values already inside the limit incur exactly zero penalty; excursions are pulled back while remaining far outside the clipped-activation decision region. This preserves inference architecture and scales while giving unseen states 2× numerical headroom. Sparse and CSR paths share the same contract; bootstrap and expert-cycle manifests record it. Twenty-five focused model/trainer/controller tests and Ruff pass.
- 11:04 EDT — Because the specification requires QAT from step zero, the prior calibrations and partial origin were not reused. Their 2.7-GiB dependency-closed lineage was copied with rsync checksums to `john2:/Users/john2/cascadia-archive/v3-nnue-stage2-invalid-pre-headroom/training-pre-headroom-v1`; a checksum dry run was empty before local removal. John1 recovered to 56 GiB free. The headroom-QAT calibration campaign restarted from exposure zero under the detached single-writer controller. Its v2 run manifest records the 64/16,384/0.1 contract.
- 11:07 EDT — Shell-level detachment was still vulnerable to the desktop tool process-group lease and was replaced with native macOS launchd supervision. `com.cascadia.v3.bootstrap` is a validated LaunchAgent with repository working directory, explicit absolute arguments, `caffeinate`, durable stdout/stderr paths, and the existing advisory single-writer lock. Launchd reports state `running`, PID 13685 with PPID 1, one trainer, one stream, and no exits. The zero-update shell attempt left no checkpoint and calibration 1 restarted from exposure zero under launchd.
- 11:12 EDT — Closed the accumulator migration before admitting replacement training. The previously failing 9M bundle completed two fixed Rust games with eight valid seat scores and no runtime overflow under exact int32 pre-activation accumulation. The old and corrected runtimes produced identical 16-seat score vectors and final-state hashes on the last safe 5.008M checkpoint. The corrected expert worker reproduced the prior five-game shard checksum exactly in 2.739 seconds at ten threads, and the single-thread all-V3 path remained inside the registered 6.36x speedup envelope. Twenty-seven release V3 tests, six release worker tests, 25 focused Python tests, Clippy, Ruff, and worker health passed. Evidence is frozen at `profiles/exact-wide-v1/qualification.json`; protected seeds remain sealed.
- 11:12 EDT — The launchd calibration attempt was deliberately stopped before its first checkpoint because final source qualification was still changing its manifest identity. The zero-checkpoint attempt is preserved as `training-interrupted-pre-final-source-v1`; no completed scientific exposure was discarded. The final runtime binaries now have BLAKE3 digests `41234795...` (batch stream), `c94c349b...` (serving smoke), and `29558881...` (campaign worker), and John1 has 56 GiB free.
- 11:14 EDT — Closed the last immutable-lineage gap: bootstrap and expert-cycle run manifests now share a deterministic identity over all 16 V3 MLX modules, the atomic checkpoint implementation, lockfile/project metadata, Python 3.12.13, MLX 0.31.2, and the host platform. Twenty-nine focused tests and Ruff pass. The pre-identity zero-checkpoint attempt is preserved separately, and launchd restarted the final calibration lineage from exposure zero. Its manifest is `aaaf6b4d...`, training-source identity `4a3b1a16...`, batch-stream digest `41234795...`, and serving-guard digest `c94c349b...`; the dashboard reports `bootstrap_training` with John1 MLX active.
- 11:40 EDT — Live profiling isolated the replacement calibration's dominant wait to native opportunity-feature expansion at the loader's conservative four-thread default. Bootstrap now gives a single active stream eight threads and concurrent broad/teacher streams four each. Expert-cycle training uses a deterministic quota-aware allocator whose active streams sum to exactly eight threads (2/2/2/1/1 in later five-source cycles), eliminating both idle CPUs and the prior potential 20-thread oversubscription. Nineteen focused tests and Ruff pass; the running calibration remains uninterrupted and later processes adopt the source-qualified allocation.
- 12:56 EDT — Headroom-QAT calibration 1 completed exactly 4M exposures and passed its real quantized Rust serving gate. Elapsed active training was 5,857.06 seconds, latest loss 0.0518383, serving weights digest `a60b3c8...`, and the two-game/eight-seat mean was 80.625 with no accumulator overflow. Validation evaluation began immediately. All evidence is atomic and checksum-bound; protected seeds remain sealed.
- 13:00 EDT — Qualified the final training path after the anomalous long-lived calibration. Direct native streaming of 80K examples took 14.04 seconds at four threads and 11.22 seconds at eight; complete Python streaming reached 4,123 and 4,686 examples/s respectively. The proposed argmax/gather headroom subgradient was value/gradient-identical but 3% slower, so the original maximum-absolute penalty remains. Most importantly, a distinct-batch from-scratch run with the full headroom loss and AdamW trained 80K examples in 22.87 seconds (3,498 examples/s), within 6.1% of the archived 3,723 examples/s baseline. Qualification BLAKE3 is `c8ed3552...` at `profiles/headroom-qat-v2/qualification.json`. Calibration 2 is the sustained 4M confirmation gate; no training-math change was admitted.
- 13:03 EDT — Eliminated repeated compact-label expansion from the remaining validation path. John1 published image `a1355165...`; all 20 one-core Bacalhau jobs finished in 45.61 seconds. The first admission attempt correctly rejected an overly strict local assumption of exactly 32 candidates per root. Reconciliation then used the already-completed artifacts and each immutable source receipt's actual candidate count—no job was rerun. The final cache contains 20,000 roots, 554,624 rows, and 1.77 GiB with every source/output hash verified.
- 13:16 EDT — Cut calibration 1 over to the reusable validation cache. The incomplete compact pass had published no result and was preserved with a checksum-bound interruption receipt; the exact 4M training checkpoint was reused. Cached quantized validation finished in 242.28 seconds at 2,289 examples/s with power loss 0.00421064, RMSE 8.24280, and MAE 6.73893. Its 32 open games averaged 80.34375. Calibration 2 launched automatically with `--expansion-threads 8` confirmed in the live process.
- 13:31 EDT — Reprofiled calibration 2 on a representative qualified-V1 shard. Opportunity-graph construction remains the dominant replay-expansion cost. Increasing Rayon's deterministic chunk from 32 to 128 games produced byte-identical 68,409,280-byte output, but sustained 80K-example time improved only 11.77→11.48 seconds (2.5%) while peak RSS increased. The candidate was rejected, source and release binary were restored exactly to BLAKE3 `41234795...`, and the paused no-checkpoint calibration resumed without replay or mutation. Null evidence BLAKE3: `43713f08...`.
- 13:48 EDT — Calibration 2 reached the twelfth of roughly 32 broad shards needed for its exact 4M exposure stop after 31.5 active minutes. The eight-thread stream remains healthy at about four to five effective CPU cores; swap usage is stable at 14.86 GiB and memory pressure remains noncritical. A constant-rate estimate is roughly 52 minutes to the atomic checkpoint. No further speculative performance change is admitted while this sustained confirmation runs.
- 13:50 EDT — Rechecked the launchd process tree after a transient process-list ambiguity. The single authoritative controller (PID 53857), calibration-2 MLX trainer (PID 56929), and native stream (PID 56947) are all alive under the same immutable run manifest; no restart, replay, or checkpoint mutation occurred. John1 has 52 GiB free, so the 50-GiB storage guard remains active with about 2 GiB of margin.
- 13:51 EDT — Prevented a deterministic calibration-3 storage deadlock by retiring only calibration 1's completed exact-resume checkpoint through the existing verified lifecycle. Its immutable training report, quantized validation, open-game results, serving bundle, manifests, and checksums remain; the retirement receipt proves 1,192,469,653 bytes reclaimed and `scientific_outputs_preserved=true`. John1 returned to 53 GiB free while calibration 2 continued uninterrupted.
- 13:54 EDT — Completed the long-horizon storage fix before the three bootstrap origins. The two superseded Part-1 final smoke checkpoints (2,384,938,961 logical bytes) were copied into their existing lineages at `john2:/Users/john2/cascadia-archive/v3-nnue-part1-smoke`; each passed both a SHA-256 tree-digest match and an independent empty `rsync --checksum` dry run before local checkpoint removal. Part-1 reports/manifests remain on John1, the cold-archive receipt is `smoke/part1-final-checkpoint-cold-archive.json`, and free disk rose to 54 GiB without interrupting calibration 2.
- 13:57 EDT — Verified the live dashboard projection is fresh during calibration 2: `bootstrap_training`, John1 intent `train` with `training.active=true`, John2/John3 visible as benchmark workers, and John4 visible but excluded. Corrected the overnight report's stale pre-headroom interpretation; under the final exact-wide/headroom lineage only calibration 1 is complete, calibration 2 is active, calibration 3 is pending, and no learning rate or bootstrap origin has yet been selected.
- 14:00 EDT — Calibration 2's first attempt was intentionally stopped after about 42 minutes, before its first checkpoint or training report, because direct inspection found MLX had retained an 8.2-GiB graphics footprint with 3.2 GiB swapped. The root cause was an unbounded free-buffer cache across variable sparse shapes. Bootstrap and cycle trainers now pin the free-cache ceiling to 512 MiB in their immutable manifests. The zero-checkpoint attempt is preserved under `interrupted-history/calibration-2-pre-cache-limit-20260621T1358`; no completed exposure was discarded.
- 14:04 EDT — The cache-bounded replacement is qualified live. After five sustained minutes its physical footprint is 2.7 GiB (4.1-GiB peak), graphics allocation 2.1 GiB, and process-local swapped memory zero—down from 8.2/8.2/3.2 GiB. Its run manifest records `mlx_cache_limit_bytes=536870912`; twenty focused tests and Ruff pass. The replacement remains at exposure zero lineage and is progressing with the same corpus, ordering, loss, and optimizer math.
- 14:12 EDT — The cache limit remains stable after eleven sustained minutes: 2.7-GiB physical footprint, 4.1-GiB peak, 2.2-GiB graphics allocation, and zero process-local swap; global swap fell by another 48 MiB. Subsequent bootstrap and expert-cycle trainers now publish atomic loss/exposure progress every 25 steps, closing the dashboard's zero-until-checkpoint gap without changing training math. Twenty-four focused tests and Ruff pass. Calibration 2 continues uninterrupted under its already-frozen manifest.
- 14:49 EDT — Calibration 2 reached shard 19 of roughly 32 after 50 active minutes. The fixed 512-MiB MLX free-cache ceiling remains effective: 2.87-GiB physical footprint, roughly 2.25-GiB graphics allocation, zero process-local swapped pages, and global swap down another 104 MiB since 14:12. The remaining constant-rate estimate is about 34 minutes to its exact 4M checkpoint. No correctness or resource gate has failed.
- 14:02 EDT — Restored storage contingency after macOS allocated another 1-GiB swapfile. Twenty-five Part-1-only payloads permanently excluded from Stage 2—the 2,000-game engineering corpus rows, duplicate engineering parity weights/fixtures, and engineering-trained weights—were copied to `john2:/Users/john2/cascadia-archive/v3-nnue-part1-engineering-final`. All 891,118,500 bytes passed identical per-file SHA-256 lines, aggregate SHA-256 `b1a40275...`, and an empty independent rsync checksum dry run before local removal. Reports/manifests remain on John1, the receipt is `smoke/part1-engineering-payload-cold-archive.json`, free disk is 53.80 GiB, and the restarted calibration 2 remained uninterrupted.
- 14:58 EDT — The replacement calibration-2 attempt passed one continuous hour under the same launchd controller, trainer, stream, and immutable manifest. Process-free waits plus short read-only probes caused no restart; the producer remains CPU-active, free disk is 53.80 GiB, and swap is stable/decreasing. No checkpoint has been published yet, so supervision continues without touching the run.
- 15:33 EDT — Headroom-QAT calibration 2 (`lr=1e-3`) completed exactly 4,000,000 exposures in 5,001.07 active seconds, atomically checkpointed, and passed its real integer-serving gate (weights `13cdc9b...`, eight seat scores, mean 80.25, zero overflow). Cached validation passed 554,624 examples in 225.02 seconds: power loss 0.00352748, RMSE 7.41231, MAE 5.79718. Its 32 open games averaged 81.90625 with zero overflow, materially ahead of calibration 1's 80.34375 and lower validation errors. Evidence BLAKE3 values are `883490fc...` training, `7357a67a...` serving integrity, `d839211e...` evaluation, and `0f6a0d2b...` open games. Calibration 3 (`lr=1.5e-3`) launched automatically at 15:32; selection remains sealed until its identical gates finish.
- 15:35 EDT — Reconciled calibration 3's source identity `04776d09...` against calibration 2's `c76fe345...` before admitting comparison. Only the bootstrap/cycle trainer wrappers differ; all model, loss, optimizer, data, export, feature, checkpoint, runtime, and dependency hashes are identical. After removing origin/rate/seed/source hashes, the immutable manifests differ only by `loss_progress_every_steps=25`. The current wrappers merely throttle atomic `loss.json` publication for live progress; they do not alter batches, gradients, updates, targets, RNG, checkpoint math, or serving export. Calibration 3 therefore remains scientifically comparable, with the operational provenance difference explicit.
- 15:59 EDT — Calibration 3 reached 1,228,800/4,000,000 exposures (30.7%) at step 150 under the unchanged launchd process tree. Live batch loss fell from 1.553 at step 50 to 0.00577 at step 100 and 0.000585 at step 150. This is training progress only, not a selection result; integer serving, cached validation, and open games remain mandatory. Free disk is stable at 52.57 GiB.
- 16:23 EDT — Calibration 3 passed halfway at 2,252,800/4,000,000 exposures (56.3%), step 275, with live batch loss 0.001005. The same controller/trainer/stream remain active and free disk is stable at 52.57 GiB. The checkpoint and all downstream scientific gates are still pending.
- 17:08 EDT — Calibration 3 (`lr=1.5e-3`) completed exactly 4M exposures in 5,081.24 active seconds and passed real integer serving (weights `0a75a04e...`, mean 83.625 over eight seats, zero overflow), cached validation (power loss 0.00422297, RMSE 7.98151, MAE 6.19678), and 32 open games (mean 82.80469, zero overflow). Although calibration 2 had the lower validation loss, its paired open-score delta versus calibration 3 was -0.8984 with lower 95% bound -1.8115, failing the registered -1.0 nonregression margin; calibration 1 also failed. Calibration 3 was the sole eligible rate and selection `9b83375a...` therefore chose `1.5e-3` exactly as registered. All three calibration optimizer checkpoints were retired with scientific outputs preserved, restoring 53.58 GiB free. Bootstrap origin 1 of 3 launched at 17:08 with run manifest `7d4c34ee...`, source `04776d09...`, selected rate `1.5e-3`, live loss telemetry, QAT, and real-serving checkpoint guards.
- 18:47 EDT — Bootstrap origin 1 passed its first durable recovery gate at exactly 4,500,000 exposures/550 updates. Atomic checkpoint manifest BLAKE3 is `adcb3a2a...`; the real-serving integrity receipt `f802b80e...` records quantized weights `1f726d0d...`, eight fixed seat scores `[81,83,85,78,82,84,88,83]`, mean 83.0, and zero overflow. Block 2 began automatically from the in-memory state while the checkpoint remains a complete exact-resume boundary. Free disk is 54.52 GiB.
- 19:00 EDT — Origin 1 passed its first periodic checkpoint at 5,007,904 exposures/612 updates. Checkpoint manifest BLAKE3 is `996db3d0...`; serving receipt `b24ccd93...` binds weights `ee7fde9a...`, eight seat scores `[74,91,87,81,81,80,85,88]`, mean 83.375, and zero overflow. `latest.json` atomically points to this newer exact-resume boundary; block 2 continues and John1 retains 53.40 GiB free.
- 12:07 EDT — The immutable calibration-1 process remains healthy under its original four-thread source identity but has made the throughput impact concrete: 53 minutes elapsed without the 4M terminal checkpoint, versus 1,074.34 seconds for the pre-headroom lineage. Samples show continuous native expansion, 67–86% Metal utilization, stable/decreasing swap, and safe disk. Because the 11:40 source-qualified loader change doubles the next single-stream run to eight expansion threads, calibration 1 will finish intact and calibration 2 will provide the controlled end-to-end speed measurement before any further training-math change is considered.
- 10:27 EDT — Bootstrap origin 1 passed its first real-serving checkpoint at exactly 4,500,000 exposures. The quantized bundle digest is `fe34b6544f189381b1c3c9732d25da323db20c61453d2ddc762b8f5425b4fc45`; both fixed Rust direct games completed with no accumulator failure. The atomic resume checkpoint and checksum-bound integrity receipt are durable, and training continued automatically into block 2.
- 10:30 EDT — The first interval checkpoint beyond the 4M calibration horizon also passed at 5,007,904 exposures. Its quantized digest is `ea5a652e546c172c3fcd96e87b004a58ad701cefbc8367092406be5109f8cdd5`; both integer-serving games completed cleanly. Origin 1 therefore remains serving-safe after crossing the calibration budget.
- 10:34 EDT — Replaced eight full late-origin SWA snapshots with an exact online running average. Each update writes a new checksum-bound generation, atomically advances its state pointer, then removes the prior generation; a crash can always recover the last complete average. Final averaging is mathematically identical while persistent SWA storage falls from about 3.0 GiB to one 379-MiB model plus the final copy. Six focused trainer tests, 26 combined campaign tests, and Ruff pass. Origin 1 was deliberately resumed from its verified 5,007,904-exposure checkpoint so the bounded SWA implementation is active before the 28.8M SWA window; no completed exposure was repeated.
- 10:40 EDT — A turn-boundary relaunch briefly created two bootstrap controllers against the same 5,007,904-exposure checkpoint. Both were stopped before either reached another write; `latest.json` and both integrity receipts remained unchanged. The controller and MLX trainer now each hold an OS-level nonblocking exclusive lock for their full lifetime. A live duplicate launch returned `already-running` while exactly one controller, one trainer, and one native stream remained. Sixteen focused lock/pipeline/trainer tests and Ruff pass. Origin 1 resumed exactly from step 612.
- 10:43 EDT — Closed the bootstrap-to-expert orchestration gap. After the winning origin passes full evaluation and parity, the locked controller now builds and publishes the final John1 source-identical Docker image, runs its embedded worker health path locally, writes a checksum-bound bootstrap/image handoff, advances to cycle 1 using that evidence, and directly supervises the locked ten-cycle expert controller with the immutable image digest. A valid publication is reused only when its complete workspace source identity is exact. Nineteen focused orchestration/trainer tests and Ruff pass. Exactly one live controller and trainer resumed origin 1 from step 612.

- 16:21 EDT — Calibration 3 (`lr=1.5e-3`) reached 2,252,800/4,000,000 exposures under the same launchd controller and immutable scientific contract. Its latest published batch loss was 0.00100491. The MLX process remained below 1 GiB RSS, global swap continued to decline, and John1 retained 53 GiB free; no integrity, memory, or disk gate failed.

- 18:13 EDT — Bootstrap origin 1 reached 2,867,200/36,000,000 exposures at the selected `1.5e-3` rate. The run retained the same controller/trainer/stream identities, latest published batch loss was 0.00250193, John1 had 56 GiB free, and swap had fallen by roughly 1.7 GiB since calibration. Its first 4.5M real-serving checkpoint remained pending.

- 18:58 EDT — Bootstrap origin 1 passed both its 4.5M block-boundary and 5,007,904-exposure periodic real-serving gates under the final headroom/exact-wide lineage. The latter quantized weights digest was `ee7fde9a...`; two fixed games produced eight valid seat scores with mean 83.375 and no runtime failure. Training continued automatically into block 2 and reached 5,114,400 exposures; John1 retained 53 GiB free and swap continued to decline.
- 20:20 EDT — Bootstrap origin 1 completed block 2 and passed the 9,000,000-exposure atomic recovery and real-serving integrity gate. The checkpoint manifest BLAKE3 is `4babf705...`; all model, optimizer, state, and game-report digests were recomputed and match the manifest. Quantized weights `05584f79...` completed two fixed games with eight seat scores `[85,88,76,80,81,76,89,80]`, mean 81.875, zero overflow states, zero swap growth, and integrity receipt `c8a56704...`. `latest.json` points to step 1,100; John1 retains 53.39 GiB free, and block 3 may proceed from this exact-resume boundary.
- 20:46 EDT — Origin 1 passed the 10,007,616-exposure periodic recovery and serving gate while continuing block 3. Recomputed model, optimizer, state, and game hashes match checkpoint manifest `e78fc0f0...`; serving receipt `53430556...` binds quantized weights `7c0b04d0...` to scores `[79,85,80,74,75,85,85,81]`, mean 80.5, zero overflow, and zero swap growth. `latest.json` now points to step 1,223, training has already advanced beyond 10.22M exposures, and John1 retains 53.39 GiB free.
- 21:54 EDT — Origin 1 completed block 3 at exactly 13,500,000 exposures and passed its atomic recovery plus real integer-serving gate. Recomputed file hashes match checkpoint manifest `3cae8da0...`; serving receipt `642b561e...` binds quantized weights `a2b0e75f...` to scores `[88,86,80,84,80,84,86,80]`, mean 83.5, zero overflow, and zero swap growth. `latest.json` points to step 1,650; block 4 began automatically with a fresh deterministic stream, and John1 retains 53.38 GiB free.
- 22:26 EDT — Origin 1 passed the 15,007,328-exposure periodic checkpoint and real integer-serving gate during block 4. Recomputed model, optimizer, state, and report hashes match checkpoint manifest `49624d62...`; serving receipt `412cfd4d...` binds weights `14f48e12...` to scores `[85,72,88,80,75,75,84,81]`, mean 80.0, zero overflow, and zero swap growth. `latest.json` points to step 1,834; the same controller resumed immediately, and John1 retains 53.38 GiB free.
- 23:37 EDT — Origin 1 completed the four broad-data passes at exactly 18,000,000 exposures and passed the phase-boundary recovery and real integer-serving gate. Recomputed hashes match checkpoint manifest `9505a5e7...`; serving receipt `787d4b73...` binds weights `c6935dab...` to scores `[80,89,79,81,81,84,84,84]`, mean 82.75, zero overflow, and zero swap growth. The trainer entered block 5 correctly with two deterministic streams, a 50/50 broad/teacher mix, and teacher lambda 1.0; it had already reached 18,204,800 exposures when verified. John1 retains 53.39 GiB free.
- 00:45 EDT — Origin 1 passed both the 20,007,040-exposure periodic gate and the 20,250,000-exposure completion of mixed block 5. Recomputed file hashes match checkpoint manifests `b8b20c71...` and `943f435c...`. The periodic serving bundle averaged 85.125 over scores `[83,85,82,81,88,82,93,87]`; the block-boundary bundle averaged 82.875 over `[82,78,87,78,82,83,89,84]`. Both had zero overflow and zero swap growth. Block 6 began automatically with the registered 50/50 mix and teacher lambda annealed from 1.0 to `0.96428571`; John1 retains 53.38 GiB free.
- 01:53 EDT — Origin 1 completed mixed block 6 at exactly 22,500,000 exposures. Recomputed hashes match checkpoint manifest `b11afe6b...`; serving receipt `abd7a15b...` binds weights `7866f863...` to scores `[80,88,81,85,76,77,83,84]`, mean 81.75, zero overflow, and zero swap growth. `latest.json` points to step 2,752. Block 7 began automatically with the 50/50 broad/teacher mix and teacher lambda correctly annealed to `0.92857143`; automatic lifecycle retirement leaves 54.36 GiB free.
- 03:05 EDT — Origin 1 completed mixed block 7 at exactly 24,750,000 exposures. Recomputed hashes match checkpoint manifest `c819251a...`; serving receipt `70ab05ba...` binds weights `260e6662...` to scores `[94,89,81,80,88,89,86,87]`, mean 86.75, zero overflow, and zero swap growth. `latest.json` points to step 3,028. Block 8 began automatically with the 50/50 broad/teacher mix and teacher lambda correctly annealed to `0.89285714`; John1 retains 54.35 GiB free.
- 03:21 EDT — Origin 1 passed the 25,003,952-exposure periodic checkpoint during mixed block 8. Recomputed hashes match checkpoint manifest `0079a203...`; serving receipt `95d12fbd...` binds weights `b5958664...` to scores `[84,89,83,85,86,83,85,80]`, mean 84.375, zero overflow, and zero swap growth. `latest.json` points to step 3,059; block 8 has already advanced to 25,339,824 exposures at teacher lambda `0.89285714`, and John1 retains 54.34 GiB free.
- 04:18 EDT — Origin 1 completed mixed block 8 at exactly 27,000,000 exposures. Recomputed hashes match checkpoint manifest `c43239f4...`; serving receipt `e8340a0d...` binds weights `65a2e7bf...` to scores `[81,80,83,81,82,83,86,82]`, mean 82.25, zero overflow, and zero swap growth. `latest.json` points to step 3,304. Block 9 began automatically with the 50/50 broad/teacher mix and teacher lambda correctly annealed to `0.85714286`; live progress reached 27,172,032 exposures and John1 retains 54.34 GiB free.
- 05:16 EDT — Origin 1 crossed the registered 28.8M SWA threshold and atomically published online SWA generation 1 at exposure 28,802,240. `swa/state.json` records count 1 and average digest `470771bf...`; an independent BLAKE3 recomputation matches. Exactly one 397,489,027-byte average plus its 229-byte atomic pointer exists, confirming the bounded online implementation rather than retaining full model snapshots. Training continued to 29,015,232 exposures in mixed block 9; John1 retains 53.96 GiB free.
- 05:31 EDT — Origin 1 completed mixed block 9 at exactly 29,250,000 exposures. Recomputed hashes match checkpoint manifest `42e369fc...`; serving receipt `e8e46662...` binds weights `9d9b30b5...` to scores `[89,88,84,87,90,86,87,87]`, mean 87.25, zero overflow, and zero swap growth. `latest.json` points to step 3,580. Block 10 began automatically with the 50/50 broad/teacher mix and teacher lambda correctly annealed to `0.82142857`; live progress reached 29,413,840 exposures and John1 retains 53.94 GiB free.
- 05:57 EDT — Origin 1 passed the 30,003,664-exposure periodic checkpoint during mixed block 10. Recomputed hashes match checkpoint manifest `72adfef7...`; serving receipt `d43d36a7...` binds weights `0f5ba762...` to scores `[86,87,85,91,88,85,89,89]`, mean 87.5, zero overflow, and zero swap growth. Online SWA generation 2 was atomically published at exposure 29,700,560 with count 2 and digest `8b329e88...`; generation 1 is absent, proving bounded replacement. Training continued to 30,233,040 exposures and John1 retains 53.95 GiB free.
- 06:41 EDT — Origin 1 completed the sixth and final 50/50 broad/teacher block at exactly 31,500,000 exposures. Recomputed hashes match checkpoint manifest `78d3efc7...`; serving receipt `fcc8333f...` binds weights `05b949cb...` to scores `[92,90,91,85,79,85,86,89]`, mean 87.125, zero overflow, and zero swap growth. Online SWA advanced atomically to generation 4 at the same exposure with count 4 and digest `4d856117...`, retaining exactly one average. Low-rate consolidation block 11 then began at the registered 0.2 multiplier: learning rate `3e-4`, 50/50 broad/teacher mix, and lambda `0.78571429`; it reached 31,655,648 exposures with 53.94 GiB free.
- 07:52 EDT — Origin 1 completed low-rate consolidation block 11 at exactly 33,750,000 exposures. Recomputed hashes match checkpoint manifest `7c4b4e9d...`; serving receipt `01d12455...` binds weights `df6ca7a7...` to scores `[88,86,81,85,84,83,83,85]`, mean 84.375, zero overflow, and zero swap growth. Online SWA generation 6 was already durable at exposure 33,302,240 with count 6 and digest `36325a7b...`, retaining one average. Final consolidation block 12 began automatically at learning rate `3e-4`, 50/50 broad/teacher mix, and terminal lambda `0.75`; live progress reached 33,897,456 exposures and John1 retains 53.94 GiB free.
- 08:30 EDT — Origin 1 passed its last intermediate checkpoint at 35,003,376 exposures. Recomputed hashes match checkpoint manifest `7493e4c2...`; serving receipt `7529914a...` binds weights `5b42bd49...` to scores `[84,81,82,92,85,88,88,87]`, mean 85.875, zero overflow, and zero swap growth. Online SWA advanced to generation 8 at exposure 35,101,680 with count 8 and digest `e0c73edf...`, retaining one average. Final consolidation continued to 35,126,256 exposures and John1 retains 53.95 GiB free.
- 09:14 EDT — Bootstrap origin 1 completed all 36,000,000 exposures and passed every terminal gate. Training report `e3c0af6b...` records 27M broad plus 9M teacher examples, 12 completed blocks, 4,408 updates, 56,704.76 active seconds, and nine SWA samples; final SWA digest is `6f2f6ce1...`. The 36M serving guard averaged 82.625 with zero overflow/swap growth. Full cached validation passed 554,624 examples at power loss 0.00364493, RMSE 7.98817, and MAE 5.77423. Its 32 open games averaged 84.91406 over 128 seat scores with zero overflow and zero swap growth; evaluation digest is `160ab3e8...`, serving weights digest `90e4a6eb...`. Verified compaction retained the terminal exact-resume checkpoint and reclaimed 1,589,958,857 bytes (`eff78eb9...`), raising free space to 54.95 GiB.
- 09:14 EDT — Bootstrap origin 2 launched automatically from scratch under its own immutable manifest and seed, while preserving the same final source identity `04776d09...`, selected `1.5e-3` rate, corpus, schedule, QAT, and serving guards. It reached 409,600 exposures/50 updates with broad loss 0.04442 under one trainer and one native stream. Origin 1 remains scientifically frozen; no origin selection occurs until origins 2 and 3 pass the same gates.
- 10:37 EDT — Bootstrap origin 2 completed block 1 at exactly 4,500,000 exposures and passed its first independent atomic recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `cfea2701...`; serving receipt `2a322030...` binds weights `4fa20146...` to scores `[81,80,78,75,84,77,76,84]`, mean 79.375, zero overflow, and zero swap growth. `latest.json` points to step 550, block 2 began automatically, and John1 retains 53.82 GiB free.
- 10:53 EDT — Origin 2 passed its 5,007,904-exposure periodic recovery and integer-serving gate. Recomputed hashes match checkpoint manifest `6d794589...`; serving receipt `6a3d7fb2...` binds weights `07526094...` to scores `[79,74,75,82,81,80,85,90]`, mean 80.75, zero overflow, and zero swap growth. `latest.json` points to step 612; block 2 continued to 5,319,200 exposures. Two exact-resume checkpoints temporarily occupy 2.4 GiB, leaving 52.70 GiB free, still above the hard 50-GiB floor; verified lifecycle compaction remains enabled.
- 12:09 EDT — Origin 2 completed block 2 at exactly 9,000,000 exposures and passed the atomic recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `8955ccc1...`; serving receipt `701050f3...` binds weights `79954c99...` to scores `[86,83,83,84,82,75,88,81]`, mean 82.75, zero overflow, and zero swap growth. `latest.json` points to step 1,100; block 3 began automatically. Lifecycle compaction removed the superseded 4.5M checkpoint and retained the 5.008M plus 9M recovery boundaries, leaving 52.69 GiB free.
- 12:35 EDT — Origin 2 passed its 10,007,616-exposure periodic recovery and integer-serving gate. Recomputed hashes match checkpoint manifest `097b62e3...`; serving receipt `9bbf979c...` binds weights `bccec2bc...` to scores `[79,78,83,73,81,86,83,82]`, mean 80.625, zero overflow, and zero swap growth. `latest.json` points to step 1,223; lifecycle compaction now retains the 9M and 10.008M boundaries, block 3 continued to 10,433,600 exposures, and John1 retains 52.70 GiB free.
- 13:43 EDT — Origin 2 completed block 3 at exactly 13,500,000 exposures and passed the atomic recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `86b42d16...`; serving receipt `83518fbd...` binds weights `3d1cb7a1...` to scores `[84,90,82,74,85,82,84,81]`, mean 82.75, zero overflow, and zero swap growth. `latest.json` points to step 1,650; block 4 began automatically and reached 13,704,800 exposures. Compaction retains only the 10.008M and 13.5M recovery boundaries, leaving 52.68 GiB free.
- 14:16 EDT — Origin 2 passed its 15,007,328-exposure periodic recovery and integer-serving gate. Recomputed hashes match checkpoint manifest `e7830c2f...`; serving receipt `9b4ad571...` binds weights `905a4dae...` to scores `[86,84,80,79,82,80,84,87]`, mean 82.75, zero overflow, and zero swap growth. `latest.json` points to step 1,834; compaction retains the 13.5M and 15.007M boundaries, block 4 continued to 15,343,200 exposures, and John1 retains 52.68 GiB free.
- 15:17 EDT — Origin 2 completed the four broad-only blocks at exactly 18,000,000 exposures and passed the phase-boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `17dfd47b...`; serving receipt `22221ba1...` binds weights `5a5c7fef...` to scores `[82,83,87,83,74,73,88,83]`, mean 81.625, zero overflow, and zero swap growth. `latest.json` points to step 2,200. Mixed block 5 began automatically with two deterministic streams, 50/50 broad/teacher data, and lambda `1.0`; it reached 18,204,800 exposures. Compaction retains the 15.007M and 18M boundaries with 52.67 GiB free.
- 16:28 EDT — Origin 2 passed both the 20,007,040 periodic checkpoint and the 20,250,000 completion of mixed block 5. Recomputed hashes match checkpoint manifests `c15556c3...` and `f32b1e7b...`. Their integer-serving means were 82.375 over `[82,85,77,75,80,80,91,89]` and 84.25 over `[86,84,83,77,83,85,91,85]`; both had zero overflow and zero swap growth. Block 6 began automatically with the 50/50 mix and lambda annealed to `0.96428571`; it reached 20,446,608 exposures. Compaction retains these two latest recovery boundaries with 52.65 GiB free.
- 17:39 EDT — Origin 2 completed mixed block 6 at exactly 22,500,000 exposures and passed the recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `831bdd13...`; serving receipt `a4b58c34...` binds weights `ad6515a6...` to scores `[82,77,82,80,83,78,84,83]`, mean 81.125, zero overflow, and zero swap growth. The earlier isolated 0.3809 batch loss did not persist; the next published loss returned to 0.000705 before the gate. Block 7 began automatically with lambda `0.92857143`, reaching 22,688,416 exposures. Compaction retains the 20.25M and 22.5M boundaries with 52.62 GiB free.
- 18:50 EDT — Origin 2 completed mixed block 7 at exactly 24,750,000 exposures and then passed the adjacent 25,003,952 periodic gate in block 8. Recomputed hashes match checkpoint manifests `cbec1131...` and `20962171...`. Their integer-serving means were 85.125 over `[84,82,88,82,85,88,89,83]` and 84.0 over `[81,84,87,87,76,84,88,85]`; both had zero overflow and zero swap growth. Block 8 continued with lambda `0.89285714`. Compaction retains these two newest boundaries with 52.60 GiB free.
- 20:02 EDT — Origin 2 completed mixed block 8 at exactly 27,000,000 exposures and passed the recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `624cf5cf...`; serving receipt `af247813...` binds weights `22b3093e...` to scores `[83,83,86,75,84,83,86,84]`, mean 83.0, zero overflow, and zero swap growth. Block 9 began automatically with lambda `0.85714286`, reaching 27,172,032 exposures. Compaction retains the 25.004M and 27M boundaries with 52.59 GiB free.
- 21:03 EDT — Bootstrap origin 2 entered its registered final-20% SWA window. At exactly 28,802,240 exposures the crash-recoverable online average was atomically initialized as the sole retained `average-001.safetensors`; `swa/state.json` records count 1 and matching BLAKE3 `25947f63...`. Training remained under the same controller/trainer lineage and advanced through 29,015,232 exposures in mixed block 9 with teacher lambda `0.85714286`; John1 retained 52 GiB free.
- 21:12 EDT — Origin 2 completed mixed block 9 at exactly 29,250,000 exposures and passed the recovery plus real quantized-serving gate. Recomputed model/optimizer/state hashes match checkpoint manifest `8f1ffa72...`; integrity receipt `9a953399...` binds serving weights `f6e4cf82...` to scores `[84,86,90,82,80,84,83,88]`, mean 84.625, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 10 began automatically with the 50/50 broad/teacher mix and teacher lambda `0.82142857`.
- 21:40 EDT — Origin 2 passed the 30,003,664-exposure periodic exact-resume and serving gate in mixed block 10. Recomputed hashes match checkpoint manifest `a1a08869...`; integrity receipt `2bf6f789...` binds weights `dd24ef8a...` to scores `[73,89,91,80,83,82,84,88]`, mean 83.75, 100% radius-7 hot path, zero overflow, and zero swap growth. Online SWA generation 2 is the sole retained average at exposure 29,700,560 with count 2 and digest `e7960440...`; training continued beyond 30.03M exposures.
- 22:29 EDT — Origin 2 reached 31.5M and atomically published SWA generation 4, but the storage guard refused the 31.5M checkpoint before John1 could cross the 50-GiB free-space floor. No partial checkpoint or integrity receipt was admitted. Launchd restarted once from the verified 30,003,664-exposure checkpoint under the same manifest; exact loader/optimizer/RNG continuation is replaying only the uncheckpointed 1.496M exposures. The pre-restart SWA state is safe because exact continuation deterministically reconstructs the same 31.5M weights and the online-SWA last-exposure guard prevents duplicate inclusion.
- 22:51 EDT — Removed the recurring storage constraint by cold-archiving the dependency-closed, superseded R2-MAP campaign tree from John1 to `john2:/Users/john2/cascadia-archive/r2-map-v1-john1-20260622`. The archive contains 8,287 files and 17,371,319,558 logical bytes. Checksum-mode rsync returned zero differences and independently computed per-file aggregate SHA-256 values match at `9a76bf52...` before source deletion. Final receipt BLAKE3 `32e8e19b...` is at `control/cold-archive-r2-map-v1-john1-20260622.json`; John1 free space rose from about 52 GiB to 64 GiB. The resumed origin-2 trainer remained active throughout.
- 23:25 EDT — The first automatic resume correctly stopped at 30.438M rather than corrupt SWA: the atomic forward SWA journal was at 31.5M while the admitted model checkpoint remained at 30.004M, and the old strict stale-state guard could not distinguish exact replay. Implemented the permanent recovery contract: validate the sole SWA generation and event sequence, preserve it, and skip only scheduled SWA events it already contains while exact optimizer/loader/RNG replay reconstructs those model states. A checksum-bound operational source migration proves `campaign_train.py` is the only training-source change and that the scientific run contract is otherwise byte-identical; 12 focused and 36 combined tests plus Ruff pass. Migration receipt `9c5596a4...` transitions source `04776d09...` to `afd654ae...`; migrated run manifest canonical identity is `3be139a7...`. Live replay receipt `72f7d81c...` binds checkpoint 30,003,664 to four-sample SWA digest `1b15d31d...` and replay target 31.5M. The trainer relaunched under that contract with 64 GiB free.
- 00:24 EDT — The migrated origin-2 trainer crossed the formerly failing 30.6M SWA event and advanced to 30,847,440 exposures without interruption. The forward journal remained byte-identical at count 4, last exposure 31.5M, and digest `1b15d31d...`, proving the replay rule skipped the already represented event without double-counting. The controller, MLX trainer, and two native streams remained live with 64 GiB free.
- 00:50 EDT — Origin 2 completed the exact replay and admitted the 31,500,000-exposure checkpoint under migrated manifest `3be139a7...`. Recomputed hashes match checkpoint manifest `9018c48c...`; serving receipt `c44cd02e...` binds weights `e67ffa5f...` to scores `[82,80,85,76,79,82,90,87]`, mean 82.625, 100% radius-7 hot path, zero overflow, and zero swap growth. The four-sample SWA digest remained `1b15d31d...`, proving exact recovery without duplicate averaging. Low-rate consolidation block 11 began automatically at learning rate `3e-4`, 50/50 mix, and lambda `0.78571429`, reaching 31,655,648 exposures with 63 GiB free.
- 02:02 EDT — Origin 2 completed low-rate consolidation block 11 at exactly 33,750,000 exposures. Recomputed hashes match checkpoint manifest `42df4b61...`; serving receipt `3a5817dc...` binds weights `1cc279d7...` to scores `[81,82,82,69,86,87,92,81]`, mean 82.5, 100% radius-7 hot path, zero overflow, and zero swap growth. Online SWA advanced normally beyond the replay horizon to generation 6 at exposure 33,302,240 with digest `9d2029cd...`. Final block 12 began automatically at learning rate `3e-4` and terminal lambda `0.75`, reaching 33,897,456 exposures with 63 GiB free.
- 02:43 EDT — Origin 2 passed its last intermediate checkpoint at 35,003,376 exposures. Recomputed hashes match checkpoint manifest `ffcd3689...`; serving receipt `814206d1...` binds weights `bb63aad1...` to scores `[84,79,85,80,87,80,86,84]`, mean 83.125, 100% radius-7 hot path, zero overflow, and zero swap growth. Online SWA generation 8 is durable at exposure 35,101,680 with digest `3fe2a40b...`; final consolidation continued through 35,126,256 exposures with 63 GiB free.
- 03:13 EDT — Bootstrap origin 2 completed the full 36,000,000-exposure schedule after the bounded storage/SWA recovery, with the same 27M broad plus 9M teacher contract, 12 blocks, 4,408 updates, terminal lambda 0.75, and nine SWA samples. Training report BLAKE3 is `0030e959...`; final SWA is `288b917c...`. Terminal checkpoint manifest `72f14b08...` is bound to migrated run manifest `3be139a7...`; serving receipt `f413a0e1...` binds quantized weights `79894e73...` to scores `[82,88,80,83,79,83,88,85]`, mean 83.5, 100% radius-7 hot path, zero overflow, and zero swap growth. Cached validation and 32 open games began immediately.
- 03:24 EDT — Bootstrap origin 2 passed full evaluation. Cached quantized validation processed 554,624 examples in 234.59 seconds at 2,364 examples/s: power loss `0.00499415`, RMSE `9.16236`, and MAE `6.67397`. Its 32 open games/128 seat scores averaged exactly 85.0 with 100% radius-7 hot path, zero overflow, and zero swap growth. Evaluation digest is `46eff5ee...`; serving weights are `88360c88...`. Verified compaction retained the terminal exact-resume checkpoint and reclaimed 1,589,959,423 bytes. Origin 3 launched automatically from independent seed 83,103 under run manifest `b65d75bd...` and replay-safe source `afd654ae...`; origin selection remains sealed until its identical gates finish.
- 05:02 EDT — Bootstrap origin 3 completed block 1 at exactly 4,500,000 exposures and passed its first independent recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `12a05ef4...`; serving receipt `22282683...` binds weights `4adb5442...` to scores `[80,80,85,71,86,81,87,88]`, mean 82.25, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 2 began automatically under immutable run manifest `b65d75bd...`; John1 retained 63 GiB free.
- 05:22 EDT — Origin 3 passed its 5,007,904-exposure periodic recovery and integer-serving gate. Recomputed hashes match checkpoint manifest `9fa42864...`; serving receipt `51d80f44...` binds weights `e70bfdf8...` to scores `[71,84,81,74,79,84,80,87]`, mean 80.0, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 2 continued beyond 5.52M exposures with 62 GiB free.
- 06:46 EDT — Origin 3 completed block 2 at exactly 9,000,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `81b5ef60...`; serving receipt `13cc8590...` binds weights `c2404426...` to scores `[82,85,81,84,80,78,90,84]`, mean 83.0, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 3 began automatically and reached 9.41M exposures with 62 GiB free.
- 07:08 EDT — Origin 3 passed its 10,007,616-exposure periodic recovery and integer-serving gate. Recomputed hashes match checkpoint manifest `14574e7a...`; serving receipt `59ca99a7...` binds weights `ffabd973...` to scores `[86,79,81,80,87,78,83,85]`, mean 82.375, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 3 continued beyond 10.43M exposures with 62 GiB free.
- 08:16 EDT — Origin 3 completed block 3 at exactly 13,500,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `1011a717...`; serving receipt `85a9a79c...` binds weights `9b96a5bd...` to scores `[82,85,81,84,85,78,86,85]`, mean 83.25, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 4 began automatically with 62 GiB free.
- 08:50 EDT — Origin 3 passed its 15,007,328-exposure periodic recovery and integer-serving gate. Recomputed hashes match checkpoint manifest `e47d7345...`; serving receipt `39e7abd7...` binds weights `b4d56201...` to scores `[87,84,84,85,86,87,84,80]`, mean 84.625, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 4 continued beyond 15.14M exposures with 62 GiB free.
- 09:51 EDT — Origin 3 completed the four broad-only blocks at exactly 18,000,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `69eee807...`; serving receipt `7c6f125d...` binds weights `75bea755...` to scores `[83,88,81,80,79,85,90,92]`, mean 84.75, 100% radius-7 hot path, zero overflow, and zero swap growth. Mixed block 5 began automatically with the 50/50 broad/teacher mix and lambda `1.0`; John1 retained 62 GiB free.
- 11:07 EDT — Origin 3 passed both the 20,007,040 periodic checkpoint and the 20,250,000 completion of mixed block 5. Serving receipts `943f21eb...` and `5ec33b54...` bind weights `88e32f84...` and `e6511a9e...` to fixed-seat means 85.375 and 84.0 respectively; both had 100% radius-7 hot path, zero overflow, and zero swap growth. The 20.25M checkpoint manifest is `91e173d2...`. Block 6 began automatically with the 50/50 mix and lambda annealed to `0.96428571`; John1 retained 62 GiB free.
- 12:22 EDT — Origin 3 completed mixed block 6 at exactly 22,500,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `26510b00...`; serving receipt `23d594b8...` binds weights `29c62a30...` to scores `[80,88,85,83,87,82,79,83]`, mean 83.375, 100% radius-7 hot path, zero overflow, and zero swap growth. The isolated 0.3158 teacher-batch loss did not persist; the next reported broad/teacher losses returned below 0.0011 before the gate. Block 7 began at lambda `0.92857143` and reached 22.69M exposures with 62 GiB free.
- 13:38 EDT — Origin 3 completed mixed block 7 at 24,750,000 exposures and passed the adjacent 25,003,952 periodic gate in block 8. Serving receipts `9c8b590f...` and `719fe7ba...` bind weights `1164c4b7...` and `1c6a768f...` to fixed-seat means 83.75 and 85.625; both had 100% radius-7 hot path, zero overflow, and zero swap growth. Latest checkpoint manifest is `c426bf81...`. Block 8 continued at lambda `0.89285714` with 62 GiB free.
- 14:48 EDT — Origin 3 completed mixed block 8 at exactly 27,000,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `fdc487f4...`; serving receipt `f40344da...` binds weights `1f5e5646...` to scores `[91,81,78,91,91,88,89,86]`, mean 86.875, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 9 began at lambda `0.85714286` and reached 27.17M exposures with 62 GiB free.
- 15:40 EDT — Origin 3 entered its registered final-20% SWA window. The crash-recoverable online average was atomically initialized at exposure 28,802,240 as the sole retained `average-001.safetensors`, count 1, with matching digest `eb06ab4a...`; training continued to 28.81M exposures in mixed block 9. John1 retained 61 GiB free.
- 15:56 EDT — Origin 3 completed mixed block 9 at exactly 29,250,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `b15da0c7...`; serving receipt `44f99148...` binds weights `347c34ed...` to scores `[85,88,85,75,82,83,87,81]`, mean 83.25, 100% radius-7 hot path, zero overflow, and zero swap growth. Block 10 began automatically at lambda `0.82142857`; John1 retained 61 GiB free.
- 16:24 EDT — Origin 3 passed the 30,003,664-exposure periodic recovery and integer-serving gate. Recomputed hashes match checkpoint manifest `4108e453...`; serving receipt `df70bbf9...` binds weights `c72b71fb...` to scores `[84,89,87,86,81,85,83,87]`, mean 85.25, 100% radius-7 hot path, zero overflow, and zero swap growth. Online SWA generation 2 is the sole retained average at exposure 29,700,560 with digest `77481196...`; mixed block 10 continued beyond 30.028M exposures with 61 GiB free.
- 17:14 EDT — Origin 3 completed the sixth and final mixed block at exactly 31,500,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `fd5fec1d...`; serving receipt `d780596d...` binds weights `d14bfda6...` to scores `[78,83,88,87,83,84,88,88]`, mean 84.875, 100% radius-7 hot path, zero overflow, and zero swap growth. SWA advanced atomically to generation 4 at the boundary with digest `da58562c...`. Low-rate consolidation block 11 began at learning rate `3e-4` and lambda `0.78571429`; John1 retained 61 GiB free.
- 18:32 EDT — Origin 3 completed low-rate consolidation block 11 at exactly 33,750,000 exposures and passed the boundary recovery plus integer-serving gate. Recomputed hashes match checkpoint manifest `cf578c7c...`; serving receipt `4d98fd2f...` binds weights `3b8ffdb1...` to scores `[85,81,84,82,89,79,85,85]`, mean 83.75, 100% radius-7 hot path, zero overflow, and zero swap growth. SWA generation 6 remained durable at exposure 33,302,240 with digest `73101c02...`. Final block 12 began at `3e-4` and terminal lambda `0.75`, reaching 33.897M exposures with 61 GiB free.
- 19:13 EDT — Origin 3 passed its last intermediate checkpoint at 35,003,376 exposures. Recomputed hashes match checkpoint manifest `8dcb2b75...`; serving receipt `7f7dd4f2...` binds weights `59d2fb21...` to scores `[85,85,88,83,87,82,87,86]`, mean 85.375, 100% radius-7 hot path, zero overflow, and zero swap growth. SWA generation 8 is durable at exposure 35,101,680 with digest `70ab89ab...`; final consolidation continued with 61 GiB free.
- 19:48 EDT — Bootstrap origin 3 completed the full 36,000,000-exposure schedule: 27M broad plus 9M teacher examples, 12 blocks, 4,408 updates, terminal lambda 0.75, and nine SWA samples. Training report BLAKE3 is `26bea82f...`; final SWA is `aab46449...`. Terminal checkpoint manifest `a143fe5c...` is bound to run manifest `b65d75bd...`; serving receipt `6d723a21...` binds quantized weights `2db7d039...` to scores `[84,90,78,79,84,76,80,81]`, mean 81.5, 100% radius-7 hot path, zero overflow, and zero swap growth. Cached validation and 32 open games began immediately.
- 19:59 EDT — Bootstrap origin 3 passed full evaluation and won the registered three-origin selector. Cached validation processed 554,624 examples in 240.31 seconds: power loss `0.00205615`, RMSE `6.38273`, and MAE `4.84128`; 32 open games/128 seat scores averaged `85.60156`, with 100% radius-7 hot path, zero overflow, and zero swap growth. Origins 1 and 2 failed the registered -1.0 open nonregression margin against origin 3; origin 3 was sole eligible and also had the lowest validation loss. Evaluation digest is `dbdf08b0...`, selection digest `dcd3eece...`, and serving weights `c162d678...`.
- 20:00 EDT — Froze origin 3 as the bootstrap champion after complete cross-backend parity: Rust/MLX quantized, Rust scalar/NEON, and MLX/NumPy were bit-identical across 6,400 rows; float/quantized top-1 and top-32 agreement were both 100%. Parity digest is `1dd7f732...`; frozen champion receipt is `ff738272...`. John1 built and published canonical image `100.110.109.6:5000/cascadia/v3-worker@sha256:2037941a...`, passed local worker health, and wrote stage-2 handoff `e409d024...` without opening protected seeds. Campaign state advanced to `cycle-01-collecting`.
- 20:01 EDT — The first expert-controller launch stopped before dispatch because the bootstrap launch agent had not exported the existing object-store environment. No cycle artifact or seed was lost. Installed dedicated headless launch agent `com.cascadia.v3.expert`, which sources the existing mode-0600 cluster secret file at runtime without copying credential values into code, plist, logs, or manifests. Resolution receipt is in `phase2/bootstrap/training/failure-history/expert-startup-missing-object-store-20260623.json`; the active failure marker was retired only after live recovery.
- 20:02 EDT — Expert cycle 1 collection is live under the canonical image and frozen bootstrap champion. Exactly 100 shards × 100 games are registered for the required 10,000 games; 29 jobs are running across the authorized 9/10/10 CPU fabric and 71 are scheduler-queued under the registered maximum-outstanding backpressure. Cycle 1 necessarily uses all V1 opponents, exploration `0.10`, disjoint scheduler-assigned game indices, and exactly one newest-model focal seat. Protected seeds remain sealed.
- 20:31 EDT — Cycle 1 collection completed all 100 scheduler-owned shards and exactly 10,000 games in 1,856.23 seconds with zero failures. The completion gate confirmed 10,000 newest-model focal seat-games, 30,000 qualified-V1 opponent seat-games, no prior-V3 opponents (the required cycle-1 edge case), scheduler-owned placement, 13,520,711 source bytes, and protected seeds still sealed.
- 20:33 EDT — Replay verification accepted all 100 immutable game shards in 90.94 seconds: exactly 10,000 records and 200,000 focal-only expanded training entries. The campaign advanced atomically to `cycle-01-labeling`; corpus receipt `8439aadd...` is the transition evidence.
- 20:34 EDT — Deterministic focal-only selection considered the 200,000 eligible positions and selected exactly 2,500 teacher roots. The lossless split produced 25 checksum-pinned shards of 100 roots each. Exact K32/R600 labeling launched across the authorized cluster under image `sha256:2037941a...`; 22 jobs were running and three queued, with no terminal failures. John1 retained 62 GiB free and protected seeds remained sealed.
- 20:43 EDT — Cycle 1 teacher labeling completed all 25 immutable shards and exactly 2,500 roots in 530.55 seconds with zero failures. The accepted 5,962,700-byte label corpus contains 70,727 exhaustive top-32 candidate estimates backed by exactly 1,500,000 terminal rollouts. The campaign advanced atomically to `cycle-01-training`; candidate origin 1 started from the frozen bootstrap champion on John1 while two scheduler-owned parent-benchmark jobs ran exclusively on John2 and John3. Protected evaluation seeds remain sealed and John1 retains 61 GiB free.
- 20:52 EDT — Origin 1's pre-update quantile census exposed a finite-corpus contract error: its 70,727 current-cycle candidate labels could not satisfy the exact 80,000-example quota through the single permitted source traversal. This happened before a run manifest, checkpoint, optimizer update, or model admission. The permanent loader contract now permits deterministic replay across the complete 12-element D6 augmentation group while `max_examples` still enforces every exact source quota; early producer EOF also preserves native status and stderr. Ten focused tests plus Ruff passed, and the exact 25-shard reproduction emitted all 80,000 requested rows with return code zero. Recovery receipt is `phase2/cycles/cycle-01/training/failure-history/current-teacher-single-traversal-20260623T2049.json`; the headless controller relaunched successfully and origin 1 restarted from the unchanged bootstrap parent.
- 20:55 EDT — John2/John3 completed the required one-iteration-behind parent benchmark while John1 prepared origin 1. The reconciled immutable result covers 1,000 focal games: mean `84.082`, P10/P50/P90 `79/84/89`, terrain means forest `5.110`, mountain `5.014`, prairie `5.020`, river `4.748`, wetland `5.140`; wildlife means bear `4.097`, elk `13.031`, fox `14.329`, hawk `12.540`, salmon `13.299`; Nature Tokens/Pinecones `1.754`. The jobs were constrained to 10 CPUs each, excluding John1, and the benchmark data is correctly marked ineligible for scientific training.
- 21:01 EDT — The repaired 400,000-row quantile census completed, then cycle 1 exposed its second pre-update edge case: `recent` correctly has a zero quota in cycle 1, but the pass loop indexed its absent thread allocation before skipping the source. No checkpoint, optimizer update, or model state was admitted. Added an explicit positive-quota source schedule that rejects both missing and extraneous allocations; 12 focused tests plus Ruff passed. Receipt: `phase2/cycles/cycle-01/training/failure-history/zero-quota-recent-source-20260623T2100.json`. The controller was deliberately restarted after the source update so the active process cannot retain the prior module, and origin 1 again starts from the unchanged parent.
- 21:35 EDT — Cycle 1 origin 1 completed pass 1 at exactly 400,000 exposures with the registered 120K current-broad, 80K current-teacher, 100K older-broad, and 100K older-teacher mixture. Latest loss is `0.00050769`; all four source losses remained finite and stable after recovery. Atomic checkpoint `step-000000051-epoch-0000-batch-000000` is bound to run manifest `c9512a1f...` and checksum-manifests its 397,489,027-byte model, 794,978,846-byte optimizer, and complete trainer state. Pass 2 began automatically at learning rate `3e-5`; John1 retains 57 GiB free.
- 21:58 EDT — Origin 1 completed pass 2 at exactly 800,000 exposures with latest loss `0.00065547`; the second exact-quota mixture remained finite across every source. Atomic checkpoint `step-000000102-epoch-0000-batch-000000` is bound to the same `c9512a1f...` manifest, and the two-checkpoint retention policy remains within the storage floor. Final pass 3 began at the registered `1e-5` learning rate; John1 retains 56 GiB free.
- 22:23 EDT — Cycle 1 origin 1 completed all three registered passes and exactly 1,200,000 exposures in 4,218.15 active seconds. Terminal loss is `0.00052157`; final checkpoint `step-000000153-epoch-0000-batch-000000` contains the complete model, optimizer, loader, and schedule state under run manifest `c9512a1f...`. Quantized validation and the 32-game open nonregression evaluation began immediately; no strength selection has been made.
- 22:35 EDT — Removed a redundant expert-cycle evaluation bottleneck: the cycle pipeline had re-expanded all 20,000 immutable validation roots for every candidate instead of consuming bootstrap's already qualified cache. The pipeline now requires the cache manifest to match every source path/BLAKE3/size and verifies all 20 cached shard BLAKE3 values before use. Production validation confirmed 20 shards, 554,624 identical rows, and 1,898,634,630 bytes; 13 focused tests plus Ruff pass. The interrupted raw re-expansion had not emitted an evaluation result. Origin 1 evaluation restarted against the verified `.v3t` cache; migration receipt is `phase2/cycles/cycle-01/training/evaluation-cache-migration-20260623.json`.
- 22:47 EDT — Cycle 1 origin 1 passed full evaluation. Cached quantized validation processed 554,624 rows in 229.94 seconds at 2,412 examples/s: power loss `0.00169432`, RMSE `5.71411`, MAE `4.24259`. Its 32 open games/128 seat scores averaged `86.46094`, with 100% radius-7 hot path, zero overflow, and zero swap growth; serving weights are `0c01cf50...`. Compaction retained only the terminal exact-resume checkpoint and restored John1 to 55 GiB free. Independent origin 2 launched from seed 90012; selection remains sealed until its identical gates complete.
- 22:52 EDT — Increased cycle-training preprocessing from eight to 16 bounded expansion threads before origin 2 wrote a run manifest or checkpoint. Four two-batch source queues had left roughly half of John1's CPU idle under backpressure; the registered cycle-1 allocation is now 5/3/4/4 threads for current-broad/current-teacher/older-broad/older-teacher. This changes throughput only: an exact 32-row scientific stream was byte-identical at one and eight threads (`sha256:0393e5e8...`), and 13 focused tests plus Ruff pass. The controller relaunched origin 2 under source `502267f3...`; receipt is `phase2/cycles/cycle-01/training/preprocessing-budget-migration-20260623.json`.
- 23:12 EDT — Rolled back the 16-thread preprocessing trial after direct measurement showed no throughput gain and approximately 1.66 GiB of swap growth before the first optimizer update. Feature expansion is memory-bandwidth bound near five effective cores, so extra producer threads only increased pressure. The controller was stopped, the unadmitted 16-thread run manifest archived, and the verified eight-thread 2/2/2/2 budget restored under source `1a74284f...`; 13 focused tests plus Ruff pass. Origin 2 had no loss stream, checkpoint, or admitted model state. Rollback receipt: `phase2/cycles/cycle-01/training/preprocessing-budget-rollback-20260623.json`.
- 23:54 EDT — Cycle 1 origin 2 completed pass 1 at exactly 400,000 exposures after the verified rollback. Latest loss is `0.00049399`; all source losses remained finite and swap declined throughout the admitted run. Atomic checkpoint `step-000000051-epoch-0000-batch-000000` is bound to independent run manifest `0649df44...`, which records the restored 2/2/2/2 allocation and the same training source identity as origin 1. Pass 2 began automatically at `3e-5`.
- 00:20 EDT — Cycle 1 origin 2 completed pass 2 at exactly 800,000 exposures with latest loss `0.00055107`. Atomic checkpoint `step-000000102-epoch-0000-batch-000000` is bound to manifest `0649df44...`; loss, memory, and swap remained stable. Final pass 3 began automatically at the registered `1e-5` learning rate.
- 00:47 EDT — Cycle 1 origin 2 completed all three passes and exactly 1,200,000 exposures in 5,142.19 active seconds. Terminal loss is `0.00046910`; final checkpoint `step-000000153-epoch-0000-batch-000000` captures complete exact-resume state under manifest `0649df44...`. The 21.9% runtime spread versus origin 1, despite identical exposure count and backend, confirms seed-dependent balanced-source expansion remains the dominant training bottleneck. Cached quantized validation began immediately.
- 00:58 EDT — Cycle 1 origin selection completed. Origin 2 validation yielded power loss `0.00183525`, RMSE `5.95038`, MAE `4.41848`, and open mean `86.11719`; it failed the registered -1.0 paired nonregression margin against origin 1 with lower 95% bound `-1.15558`. Origin 1 is the sole eligible candidate and also has lower validation loss (`0.00169432`) plus higher open mean (`86.46094`). Selection receipt names `cycle-01-origin-1`; the losing origin was retired and selected-candidate cross-backend parity began. Protected seeds remain sealed.
- 01:01 EDT — Froze cycle 1 origin 1 as the promotion candidate after full parity over 6,400 rows: Rust/MLX quantized, Rust scalar/NEON, and MLX/NumPy are bit-identical; float/quantized top-1 and top-32 agreement are both 100%, with exact overflow support. Frozen candidate weights SHA-256 is `c3d95b1f...`; parity report SHA-256 is `0efee90a...`. Campaign state advanced to `cycle-01-promotion`, and the registered first 100 paired comparisons launched across the 29-CPU fabric. Protected seeds remain sealed because this is the open cycle gate, not final evaluation.
- 01:43 EDT — Rejected the entire first promotion worker attempt after artifact reconciliation exposed 42 deterministic application failures: all 20 K32/R64 and 20 K32/R600 items used report spellings that Rust Clap rejects, and two equal-wall-time items reached search without the mandatory legacy environment. The 38 outputs that happened to complete are also excluded because they lack the qualified search contract. Implemented versioned contract `v2-cli-tier-legacy-env`, mapping worker tiers to `k32r64`/`k32r600` and pinning `MCE_LMR=1`, `MCE_DIVERSE_PREFILTER=1`; the sequential reader now accepts only v2 request directories. Six focused tests, Ruff, compile, and a local corrected equal-wall-time pair passed. Receipt: `phase2/cycles/cycle-01/promotion/invalid-worker-contract-20260624.json`. No invalid result entered a promotion statistic.
- 23:13 EDT — Reloaded the persistent `com.cascadia.v3.expert` LaunchAgent and restored the single headless expert controller after the preprocessing rollback. Exactly one controller, one cycle-1 pipeline, and one origin-2 trainer are live; origin 2 restarted its pre-update census from the unchanged bootstrap parent and immutable cycle corpus under the verified eight-thread budget. No scientific exposure was repeated because the discarded trial had published neither a loss sample nor checkpoint.
- 23:54 EDT — Cycle 1 origin 2 completed pass 1 at exactly 400,000 exposures with loss `0.0004939865` and atomically published `step-000000051-epoch-0000-batch-000000`. Independent BLAKE3 recomputation matches model `0de80c25...`, optimizer `45bc4996...`, and state `6e501371...`; metadata binds completed pass 1 to eight-thread run manifest `0649df44...`. Pass 2 began automatically, swap remained flat at 14,007.62 MiB, and John1 retains 53 GiB free.
- 00:20 EDT — Origin 2 completed pass 2 at exactly 800,000 exposures with finite loss `0.0005510697`. Atomic checkpoint `step-000000102-epoch-0000-batch-000000` independently revalidated model `053ac2d5...`, optimizer `cb41fe0d...`, and state `fc8951c3...` against manifest `0649df44...`. Final pass 3 began at LR `1e-5`; swap declined to 13,478.19 MiB and John1 retains 54 GiB free.
- 00:47 EDT — Cycle 1 origin 2 completed all three passes and exactly 1,200,000 exposures in 5,142.19 active seconds. Final loss was `0.0004691003` at LR `1e-5`; terminal checkpoint `step-000000153-epoch-0000-batch-000000` independently revalidated model `50273a94...`, optimizer `27296e2d...`, and state `1e24c1b9...` against run manifest `0649df44...`. Cached validation, serving export, and the fixed open-game gate began automatically; protected seeds remain sealed.
- 00:58 EDT — Origin 2 passed serving qualification (weights `3efe4b35...`) but lost the registered origin selector. Its 554,624-row quantized validation loss/RMSE/MAE were `0.0018352451`/`5.95038`/`4.41848`, versus origin 1's `0.0016943230`/`5.71411`/`4.24259`; open mean was 86.11719 versus 86.46094, and its paired lower-95 nonregression bound was `-1.15558`, below the registered `-1.0` margin. Selection receipt therefore freezes origin 1 and marks origin 2 ineligible. Origin 2 was lifecycle-retired, restoring 55 GiB free; the selected candidate entered 6,400-row parity verification with protected seeds still sealed.
- 01:00 EDT — Selected origin 1 passed the full candidate parity/freeze gate across 6,400 rows: Rust/MLX quantized, scalar/NEON, and MLX/NumPy were bit-identical; float/quantized top-1 and top-32 agreement were both 100%; overflow remained exact. Frozen candidate receipt binds model manifest SHA256 `71108628...`, weights SHA256 `c3d95b1f...`, parity SHA256 `0efee90a...`, and evidence SHA256 `48c46445...`. Campaign transition 12 atomically advanced to `cycle-01-promotion`; the always-valid four-tier test started with protected seeds sealed.
- 22:23 EDT — Cycle 1 origin 1 completed all three registered passes and exactly 1,200,000 exposures in 4,218.15 active seconds. Final loss was finite at `0.0005215654` with LR `1e-5`; passing report `training-report.json` binds run manifest `c9512a1f...`. Terminal checkpoint `step-000000153-epoch-0000-batch-000000` independently revalidated model `399477b3...`, optimizer `7b1b0739...`, and state `9b3033f7...` against its atomic manifest. Open validation/export qualification began automatically; protected seeds remain sealed.
- 22:47 EDT — Origin 1 passed cached quantized validation and open-game nonregression qualification. Validation covered 554,624 rows at loss `0.0016943230`, RMSE `5.71411`, and MAE `4.24259`; export produced serving-compatible weights `0c01cf50...`. The fixed 32-game/128-seat open domain averaged `86.46094`, ran with 100% radius-7 hot-path coverage, zero overflow, and zero swap growth. Independent origin 2 started from the same frozen bootstrap parent with seed `90012`; protected seeds remain sealed and John1 retains 55 GiB free.
- 20:43 EDT — Cycle 1 exact teacher labeling completed all 25 shards in 530.55 seconds with zero failures: exactly 2,500 roots, 1,500,000 K32/R600 rollouts, 70,727 candidate estimates, and 5,962,700 output bytes. The reconciled teacher-label corpus passed and the campaign advanced atomically to `cycle-01-training`.
- 20:44 EDT — Candidate origin 1 started on John1 from the frozen bootstrap-origin-3 checkpoint. Its registered schedule is three exact 400,000-exposure passes at learning rates `3e-5`, `3e-5`, and `1e-5`; source quotas are 50% current-cycle data and 50% bootstrap/older replay for cycle 1. In parallel, exactly two 10-CPU frozen-parent benchmark jobs are running on John2 and John3; John1 is ineligible for them. Protected seeds remain sealed and John1 retains 61 GiB free.
- 20:50 EDT — Origin 1 stopped before publishing any checkpoint during its score-quantile census because a native batch stream closed early. Collection, replay verification, teacher roots, and labels remain immutable and reusable. The immediate diagnostic defect was permanent: the Python reader raised at EOF before collecting the native producer's status and stderr. The boundary now preserves both; two focused stream regressions plus all six existing cycle-training tests pass. The incident is preserved at `phase2/cycles/cycle-01/failure-history/origin1-native-stream-eof-20260623T205011.json`; no protected seed opened and no scientific data will be regenerated.
- 21:00 EDT — The diagnostic relaunch passed all four 400K census sources and exposed the deterministic training root cause: cycle 1 correctly has a zero `recent` replay quota, but the pass-opening loop indexed the inactive source's absent thread allocation before skipping it (`KeyError: recent`). A permanent active-source schedule now binds threads only to nonzero quotas and rejects both missing and extraneous allocations. Twelve focused stream/cycle-training tests and Ruff pass. The incident receipt was amended to `root_cause_fixed_awaiting_checkpoint`; the same immutable inputs will be relaunched without regenerating games or labels.
- 21:21 EDT — The clean replacement origin is live under the corrected source identity. The failed pre-checkpoint manifest (zero checkpoints) was checksum-archived as `failure-history/origin1-precheckpoint-run-manifest-20260623T2100.json` (`sha256:c9a4f82f...`) before relaunch. The new run completed its four-source census, bound a fresh manifest, and opened all four pass-1 balanced streams without the zero-quota exception. No first loss publication or 400K checkpoint exists yet; native workers are CPU-bound in opportunity-graph expansion, so recovery remains provisional until that checkpoint.
- 21:21 EDT — The one-iteration-behind frozen-parent benchmark completed on John2/John3 and was adopted idempotently after controller recovery: 1,000 games, mean `84.082`, P10/P50/P90 `79/84/89`, wildlife means bear `4.097`, elk `13.031`, fox `14.329`, hawk `12.540`, salmon `13.299`, Nature Tokens/Pinecones `1.754`, with a complete score histogram. These engineering benchmark games are not training eligible.
- 21:31 EDT — Corrected origin 1 published its first live MLX loss sample after 204,800 exposures (step 25, pass 1, source `current_broad`): `0.0005400403` at LR `3e-5`. This proves the corrected zero-quota schedule through actual optimization, not only initialization. The four native streams remain healthy; profiling attributes the long first half-pass to exact opportunity-graph feature construction. The 400K atomic checkpoint remains the next recovery gate.
- 21:36 EDT — Origin 1 completed pass 1 at exactly 400,000 exposures and atomically published checkpoint `step-000000051-epoch-0000-batch-000000`. Loss was `0.0005076904` at LR `3e-5`; checkpoint metadata binds completed pass 1 to corrected run manifest `c9512a1f...`, model BLAKE3 `c11c2b84...`, and complete model/optimizer/state hashes. Pass 2 opened automatically. The cycle-1 startup incident is now classified `recovered` without regenerating collection or labels.
- 00:43 EDT — Origin 2 completed the deterministic replay through exactly 31,500,000 exposures and atomically published the previously refused checkpoint. The real quantized Rust serving gate passed: weights `e67ffa5f...` produced scores `[82,80,85,76,79,82,90,87]`, mean 82.625. `latest.json` now points to step 3,856; the preserved four-sample SWA journal remains byte-identical at digest `1b15d31d...`, proving no duplicate averaging. Low-rate consolidation block 11 started automatically at LR `3e-4` and lambda `0.78571429`; John1 retains 63 GiB free.
- 02:00 EDT — Origin 2 completed low-rate consolidation block 11 at exactly 33,750,000 exposures and passed its exact-resume plus integer-serving gate. Weights `1cc279d7...` produced scores `[81,82,82,69,86,87,92,81]`, mean 82.5. Online SWA reached count 6 at exposure 33,302,240 with digest `9d2029cd...`. Final block 12 started automatically at LR `3e-4` and terminal lambda `0.75`, advancing beyond 33.89M exposures with 63 GiB free.
- 03:23 EDT — Bootstrap origin 2 completed all 36,000,000 registered exposures and all 12 blocks after the verified recovery, with nine SWA samples and terminal lambda 0.75. Final weights `88360c88...` passed quantized validation (power loss 0.00499415, RMSE 9.16236, MAE 6.67397) and 32 open games/128 seats (mean 85.0, 100% radius-7 hot path, zero overflow). Its validation metrics trail origin 1, while its open mean leads by 0.08594; no selection is made until origin 3 completes. Lifecycle compaction retained one exact-resume checkpoint. Independent origin 3 launched automatically from seed 83103 under the registered `1.5e-3` rate and current immutable source contract; John1 retained 63 GiB free.
- 03:25 EDT — Retired the resolved origin-2 SWA replay failure marker from the active namespace into immutable `failure-history/swa-forward-journal-origin2-20260622T2312.json` (BLAKE3 `21dc41c...`) and refreshed the dashboard. No active bootstrap failure remains; origin 3 continued uninterrupted.
- 05:00 EDT — Bootstrap origin 3 completed broad block 1 at exactly 4,500,000 exposures and passed its first exact-resume plus real integer-serving gate. Weights `4adb5442...` produced scores `[80,80,85,71,86,81,87,88]`, mean 82.25. `latest.json` points to step 550; block 2 began automatically and advanced beyond 4.70M exposures with 63 GiB free.
- 06:37 EDT — Origin 3 completed broad block 2 at exactly 9,000,000 exposures and passed its recovery plus integer-serving gate. Weights `c2404426...` produced scores `[82,85,81,84,80,78,90,84]`, mean 83.0. `latest.json` points to step 1,100; block 3 began automatically and advanced beyond 9.20M exposures with 62 GiB free.
- 07:01 EDT — Origin 3 passed the 10,007,616-exposure periodic exact-resume and serving gate in broad block 3. Weights `ffabd973...` produced scores `[86,79,81,80,87,78,83,85]`, mean 82.375. `latest.json` points to step 1,223; training continued beyond 10.22M exposures with 62 GiB free.
- 08:14 EDT — Origin 3 completed broad block 3 at exactly 13,500,000 exposures and passed its recovery plus integer-serving gate. Weights `9b96a5bd...` produced scores `[82,85,81,84,85,78,86,85]`, mean 83.25. `latest.json` points to step 1,650; broad block 4 began automatically and advanced beyond 13.90M exposures with 62 GiB free.
- 08:49 EDT — Origin 3 passed the 15,007,328-exposure periodic recovery and serving gate in broad block 4. Weights `b4d56201...` produced scores `[87,84,84,85,86,87,84,80]`, mean 84.625. `latest.json` points to step 1,834; training continued beyond 15.13M exposures with 62 GiB free.
- 09:51 EDT — Origin 3 completed all four broad-only blocks at exactly 18,000,000 exposures and passed the phase-boundary recovery plus real integer-serving gate. Weights `75bea755...` produced scores `[83,88,81,80,79,85,90,92]`, mean 84.75. `latest.json` points to step 2,200; mixed block 5 began automatically with two four-thread streams, the registered 50/50 broad/teacher mix, and teacher lambda 1.0. John1 retains 62 GiB free.
- 11:09 EDT — Origin 3 passed both the 20,007,040 periodic checkpoint and the 20,250,000 completion of mixed block 5. Weights `88e32f84...` and `e6511a9e...` produced fixed-seat means 85.375 and 84.0; both real-serving gates passed. Teacher batches in blocks 5–6 show a heavy-tailed loss distribution (median 0.0030, P90 0.2899, 50/175 above 0.1, max 3.5286) while broad batches remain stable (median-scale, mean 0.00886, one above 0.1). Subsequent losses returned to 0.00444 and checkpoint gameplay remained stable, so this is not treated as divergence; the most likely explanation is the deliberately active accumulator-headroom penalty on hard teacher states, but final validation remains the deciding evidence. Block 6 continued at lambda `0.96428571` with 62 GiB free.
- 12:21 EDT — Origin 3 completed mixed block 6 at exactly 22,500,000 exposures and passed its exact-resume plus real integer-serving gate. Weights `29c62a30...` produced scores `[80,88,85,83,87,82,79,83]`, mean 83.375. The teacher-loss tail contracted in block 6 (P90 0.1791, mean 0.06488, max 1.8647) and the first published block-7 loss was 0.000591, supporting the decision to continue rather than classify divergence. Block 7 began automatically at lambda `0.92857143`, reaching 22.69M exposures with 62 GiB free.
- 13:39 EDT — Origin 3 completed mixed block 7 at exactly 24,750,000 exposures and passed the adjacent 25,003,952 periodic gate in block 8. Weights `1164c4b7...` and `1c6a768f...` produced fixed-seat means 83.75 and 85.625; both exact-resume and real-serving gates passed. Block 8 continued at lambda `0.89285714` with a current loss of 0.000587 and 62 GiB free.
- 14:49 EDT — Origin 3 completed mixed block 8 at exactly 27,000,000 exposures and passed its exact-resume plus real integer-serving gate. Weights `1f5e5646...` produced scores `[91,81,78,91,91,88,89,86]`, mean 86.875—the strongest origin-3 checkpoint smoke so far. Block 9 began automatically at lambda `0.85714286`, reaching 27.38M exposures with current loss 0.000678 and 62 GiB free.
- 15:45 EDT — Origin 3 entered the registered final-20% SWA window. At exactly 28,802,240 exposures it atomically initialized `average-001.safetensors`; `swa/state.json` records count 1 and matching BLAKE3 `eb06ab4a...`. Training continued through 28,810,432 exposures in mixed block 9 with lambda `0.85714286`; John1 retains 61 GiB free.
- 16:00 EDT — Origin 3 completed mixed block 9 at exactly 29,250,000 exposures and passed its exact-resume plus real-serving gate. Weights `347c34ed...` produced scores `[85,88,85,75,82,83,87,81]`, mean 83.25. Block 10 began automatically at lambda `0.82142857`; online SWA advanced atomically to count 2 at exposure 29,700,560 with digest `77481196...`. Training reached 29.62M published exposures with current loss 0.000686 and 61 GiB free.
- 16:29 EDT — Origin 3 passed the 30,003,664-exposure periodic exact-resume and real-serving gate in mixed block 10. Weights `c72b71fb...` produced scores `[84,89,87,86,81,85,83,87]`, mean 85.25. `latest.json` points to step 3,672; training continued beyond 30.44M exposures at lambda `0.82142857`, current loss 0.000416, with 61 GiB free.
- 17:19 EDT — Origin 3 completed mixed block 10 at exactly 31,500,000 exposures and passed its exact-resume plus real-serving gate. Weights `d14bfda6...` produced scores `[78,83,88,87,83,84,88,88]`, mean 84.875. Online SWA reached count 4 at exactly 31.5M with digest `da58562c...`. Low-rate consolidation block 11 began automatically at LR `3e-4` and lambda `0.78571429`, reaching 31.66M exposures with 61 GiB free.
- 18:33 EDT — Origin 3 completed low-rate consolidation block 11 at exactly 33,750,000 exposures and passed its exact-resume plus real-serving gate. Weights `3b8ffdb1...` produced scores `[85,81,84,82,89,79,85,85]`, mean 83.75. Online SWA reached count 6 at exposure 33,302,240 with digest `73101c02...`. Final block 12 began automatically at LR `3e-4` and terminal lambda `0.75`, reaching 34.10M exposures with 61 GiB free.

- 00:45 EDT — Bootstrap origin 1 completed its first 50/50 broad/teacher block at exactly 20,250,000 exposures. Both the 20,007,040 periodic gate (weights `ff5cf7ad...`, fixed-seat mean 85.125) and the block-boundary gate (weights `fe860365...`, mean 82.875) passed atomic recovery plus real quantized Rust serving. Block 6 started automatically with two deterministic four-thread streams and teacher lambda annealed from 1.0 to 0.96428571. John1 retained 53 GiB free and global swap continued to decline.

- 05:11 EDT — Bootstrap origin 1 completed mixed block 8 at 27,000,000 exposures and passed its real-serving gate (weights `65a2e7bf...`, eight-seat mean 82.25). Block 9 progressed to 28,810,432 exposures with teacher lambda `0.85714286`. The final-20% online SWA path activated exactly as registered: atomic state `swa/state.json` points to `average-001.safetensors`, count 1, last exposure 28,802,240, and both record the matching BLAKE3 `470771bf...`. The bounded 397,489,027-byte average is crash recoverable; disk remained 54 GiB free and swap continued to decline.

- 05:56 EDT — Origin 1 passed the 30,003,664-exposure periodic recovery and serving gate in mixed block 10. Recomputed hashes match the checkpoint manifest; quantized weights `0f5ba762...` completed the fixed two-game gate with eight-seat mean 87.5, the strongest checkpoint-smoke mean yet in this origin, and zero runtime failure. Online SWA generation 2 is the sole retained average; `state.json` and `average-002.safetensors` agree on BLAKE3 `8b329e88...` at exposure 29,700,560. Training continued past 30.23M exposures with teacher lambda `0.82142857`.

- 09:07 EDT — Bootstrap origin 1 completed the full registered 36,000,000-exposure schedule in 56,704.76 active seconds. All twelve blocks, every periodic/block serving gate, terminal lambda 0.75, and nine-sample online SWA completed. Final quantized weights `90e4a6eb...` passed cached validation (power loss 0.00364493, RMSE 7.98817, MAE 5.77423) and 32 open games/128 seat scores (mean 84.91406, 100% radius-7 hot path, zero overflow). Checkpoint compaction retained the latest exact-resume checkpoint and reclaimed 1,589,958,857 bytes. Bootstrap origin 2 launched automatically from independent seed 83102 under the same selected rate and immutable contract; John1 had 55 GiB free.

- 02:47 EDT — The contract-correct promotion replay completed 76/80 shards;
  four K32/R600/equal-wall-time shards failed with typed
  `Rules(WildlifeBagEmpty)` from invalid redeterminized terminal samples. The
  V3 policy now deterministically rejects only that sample and resamples a
  domain-separated hidden order without consuming another rollout budget unit;
  unrelated errors remain fatal. The release image stage passed all 28 V3
  library tests, worker tests, and health. John1 published immutable image
  `100.110.109.6:5000/cascadia/v3-worker@sha256:b3c0704d...` under contract
  `v3-conditioned-rollout`. Both earlier attempts are excluded wholesale; the
  exact four-tier 0..99 domain will replay. Protected seeds remain sealed.
- 03:38 EDT — The first scientifically eligible cycle-1 promotion increment
  completed all 80 work items: exactly 100 paired comparisons in each of the
  direct, K32/R64, K32/R600, and equal-wall-time tiers (400 tier-pairs and 800
  physical games) under immutable image `sha256:b3c0704d...`, with zero worker
  failures. The always-valid test did not cross either promotion boundary, so
  the controller correctly classified the result as inconclusive and opened
  only the registered 100..199 increment. Twenty-one of its 80 work items were
  running and 59 queued at launch. No invalid-attempt output entered the
  statistic, and protected seeds remain sealed.

- 03:45 EDT — The replacement promotion domain passed all 80/80 worker shards,
  including all four exact items that previously hit invalid hidden samples.
  This closes the conditioning incident with 400 paired tier observations and
  800 physical games under one immutable image. At the registered 100-pair
  sequential look, all tiers remained open: mean deltas were direct `-0.32`,
  K32/R64 `-0.04`, K32/R600 `+0.48`, and equal-wall-time `+0.32`; integrity and
  resource gates passed, but no statistical boundary fired. The controller
  therefore opened the registered 100–199 increment. Protected seeds remain
  sealed.

- 03:55 EDT — Reconciled the campaign ledger against the physical APFS
  container and found its stored `free_bytes` observation stale: John1 has
  29.95 GiB physically free, below the registered 50 GiB reserve. The live
  Colima worker disk accounts for the difference and cannot be moved because
  it currently hosts Bacalhau, MinIO, the registry, and cycle-1 promotion jobs.
  Docker reports 23.37 GB of unused images as safely reclaimable while the
  canonical `sha256:b3c0704d...` image and infrastructure images remain in
  use. The current read-mostly promotion increment is allowed to drain; before
  Cycle 2 collection, unused images will be pruned, the sparse VM disk trimmed,
  and physical free space revalidated above 50 GiB. No campaign artifact was
  deleted and protected seeds remain sealed.

- 04:27 EDT — Restored the physical storage invariant without touching any
  scientific artifact or live service. Docker removed only images unused by
  every container, then guest `fstrim` released 34.9 GiB of sparse Colima
  blocks; APFS free space rose from 29.95 to 54.13 GiB. The separate default
  Colima profile was proven empty (zero containers, one unreferenced image),
  absent from the live campaign socket contract, stopped, and deleted with its
  data disk, raising John1 to 60.66 GiB free. The active `cascadia-r2` profile,
  canonical worker image, Bacalhau, MinIO, registry, and all 29 promotion jobs
  remained healthy. The second 100-pair increment also completed 80/80 and
  remained statistically open; the registered 200–299 increment started.
  Protected seeds remain sealed.

- 04:38 EDT — Promotion pairs 100–199 completed 80/80 shards with no failure.
  At 200 pairs per tier, all four observed means are positive: direct `+0.205`,
  K32/R64 `+0.235`, K32/R600 `+0.465`, and equal-wall-time `+0.135`.
  Integrity and resource gates remain clean, but the conservative always-valid
  e-process has not crossed a promotion or retention boundary. The controller
  opened the registered 200–299 increment; protected seeds remain sealed.

- 04:43 EDT — Completed a promotion-gate power audit before any Cycle 2 pair
  domain exists. The frozen Cycle-1 v1 e-process is valid but structurally
  underpowered: at its 500-pair maximum, even a zero-variance `+0.15` sequence
  yields e=`1.3883`, and it requires constant `+1.7200` to reach threshold 20.
  Cycle 1 remains byte-for-byte on v1, including after restart. Cycles 2–10 are
  now pre-registered on schema v2, which preserves the same bounded mixture-
  betting process, hypotheses, alpha/beta, look schedule, unanimity, and gates
  but uses a robust paired-delta estimand winsorized at `[-25,25]`; raw means
  remain reported and final protected evaluation remains entirely raw. Nine
  focused tests and Ruff pass. Frozen-v1 SHA256 is `9a6ac0d9...`, v2 SHA256 is
  `345fe38d...`, and the audit is `docs/v3/PROMOTION_GATE_POWER_AUDIT.md`
  (`b2c2f3cb...`). Cycle 1's active workers were unaffected and protected seeds
  remain sealed.

- 05:21 EDT — Cycle 1 promotion pairs 200–299 completed all 80 shards and the
  frozen v1 rule remained open at 300 pairs per tier. Raw means are direct
  `+0.310`, K32/R64 `+0.233`, K32/R600 `+0.310`, and equal-wall-time `-0.077`;
  integrity and resource gates pass. The active process continues byte-for-byte
  on v1, while the audited v2 rule is pre-registered only for Cycles 2–10.
  Increment 300–399 is live and protected seeds remain sealed.

- 05:28 EDT — Recovered Cycle 1 promotion after the controller encountered one
  successful-but-empty Bacalhau REST response while admitting the 300–399
  increment. Scientific workers continued uninterrupted and their completed
  artifacts were preserved. The REST adapter now retries truncated/empty JSON
  responses with the same bounded exponential policy as transient network and
  HTTP failures; focused retry/exhaustion tests pass. The LaunchAgent is live,
  has adopted the existing jobs by immutable request/item identity, and reports
  35 succeeded with the remaining jobs admitted or running. No completed work
  was replayed. The dashboard scheduler query and campaign progress aggregation
  now show newest jobs and cumulative tier-pairs correctly. Protected seeds
  remain sealed.

- 06:19 EDT — Cycle 1 promotion pairs 300–399 completed 80/80 after one
  transient Bacalhau list-response decode error; the durable request resumed
  without resubmitting completed work. The frozen v1 look remains open at 400
  pairs: direct `+0.368`, K32/R64 `+0.298`, K32/R600 `+0.343`, and
  equal-wall-time `-0.085`. Integrity/resource gates pass. The final registered
  400–499 increment is live; protected seeds remain sealed.

- 07:10 EDT — Cycle 1 closed its full registered 500-pair-per-tier promotion
  domain with 4,000 physical games, zero integrity failures, and no protected
  seeds opened. The candidate was directionally positive in every tier:
  direct `+0.254`, K32/R64 `+0.132`, K32/R600 `+0.300`, and equal-wall-time
  `+0.082`. The frozen v1 rule reached its maximum without crossing a boundary,
  so the registered verdict is `retain-incumbent-inconclusive` and the
  bootstrap champion remains the Cycle-1 champion. The storage guard then
  correctly refused the Cycle-2 transition below the 50 GiB reserve. Published
  worker blocks and generated dev-profile build artifacts were reclaimed;
  John1 returned to 51.6 GiB free, the immutable champion receipt advanced the
  campaign to `cycle-02-collecting`, and the controller resumed. Cycles 2–10
  use the prospectively registered v2 promotion rule. Protected seeds remain
  sealed.

- 07:13 EDT — Cycle 1 reached its registered 500-pair maximum in every tier
  with 400/400 worker shards and 4,000 physical games valid under the final
  conditioning contract. The frozen v1 verdict is
  `retain-incumbent-inconclusive`; raw means remained positive in every tier:
  direct `+0.254`, K32/R64 `+0.132`, K32/R600 `+0.300`, and equal-wall-time
  `+0.082`. Integrity and resource gates passed, but no e-process boundary
  fired. The bootstrap checkpoint is frozen as the Cycle 1 champion; report
  SHA256 is `5ef3ca4c...`, champion receipt SHA256 `755f4c6e...`. Campaign state
  advanced atomically to `cycle-02-collecting`. Exactly 100 ten-thousand-game
  shards are registered; 29 are running and 71 scheduler-queued with 80% V1
  opponents and protected seeds sealed.

- 08:12 EDT — Cycle 2 collection and replay verification completed exactly:
  10,000 games, 200,000 expanded focal training entries, 80% qualified-V1
  opponent seats, 20% prior-V3 opponent seats, and 100/100 passing collection
  and verification shards. To restore durable headroom, 288 dependency-closed
  pre-V3 root artifacts (10.95 GiB) were copied to John2, independently
  SHA-256 verified, and only then removed from John1. The archive receipt is
  `cascadia-bench/v3-nnue/reports/storage/legacy-root-artifacts-2026-06-24.receipt.json`;
  the manifest SHA256 is `82f85382962054e9686397e1742ea3d27ce6b8205586958324349aa8762c22da`.
  John1 now has 54.1 GiB free. Sparse Colima trimming is integrated into the
  storage guard as a one-shot reclaim-and-remeasure step without weakening the
  50 GiB invariant. Cycle 2 advanced to teacher labeling; protected seeds
  remain sealed.

- 10:18 EDT — Cycle 2 collection and replay verification completed: exactly
  10,000 games, 10,000 newest focal seats, 24,000 qualified-V1 opponent seats,
  6,000 prior-V3 opponent seats, and 200,000 focal training entries. All 100
  collection and 100 verification shards passed. The first transition attempts
  correctly refused while APFS free space was below 50 GiB; no artifact was
  mutated or deleted. After purgeable space was released, the verified state
  recorded 58.09 GB free and advanced to `cycle-02-labeling`. Exactly 2,500
  roots are now running in 25 K32/R600 jobs. Protected seeds remain sealed.

- 10:28 EDT — Cycle 2 labeling completed 25/25 shards: exactly 2,500 roots,
  1,500,000 K32/R600 rollouts, and 71,129 candidate estimates. Campaign state
  advanced to `cycle-02-training` with 57.95 GB free. Two independent candidate
  origins will fine-tune from the retained bootstrap champion while John2/John3
  benchmark that frozen parent. Protected seeds remain sealed.

- 10:39 EDT — The lagged Cycle-2 parent benchmark completed 1,000 games on
  John2/John3 with mean `84.082`, P10/P50/P90 `79/84/89`, and full
  wildlife/terrain/token/Pinecone anatomy while John1 training continued. Its
  first launch exposed a stale dependency on the old repo-root V1 filename;
  the benchmark now consumes the campaign-pinned `qualified-v1.bin`, with a
  focused regression test. Cycle-2 origin-1 MLX training remains uninterrupted
  and protected seeds remain sealed.

- 09:05 EDT — Cycle-2 origin-1 reached its first 400,000-exposure pass, then
  stopped cleanly before the atomic checkpoint because concurrent APFS usage
  left less than the registered 50-GiB floor plus the 1.25-GiB checkpoint
  reserve. No partial checkpoint was published, so that pass is being replayed
  deterministically from the frozen bootstrap parent. Two retired bootstrap
  origins were copied in full to John2 and checksum-verified; only their local
  SWA tensors were removed, while all local manifests, evaluations, integrity
  receipts, and serving bundles remain. Regenerable Colima, Playwright, and
  cached Codex-updater payloads were also removed. John1 now has 53.94 GiB free,
  enough for the trainer's two-checkpoint retention window; the controller is
  running again. Receipt:
  `cascadia-bench/v3-nnue/control/cold-archive-bootstrap-retired-origins-20260624.json`.
  Protected seeds remain sealed.

- 09:14 EDT — The repeated pre-checkpoint storage failure is permanently
  repaired. The native MLX guard now performs one exact-profile Colima
  `fstrim`, remeasures both the 40-GiB campaign ceiling and the 50-GiB physical
  reserve, and still refuses any genuine overage; 24 focused trainer tests and
  Ruff pass. A further 719 untracked, pre-V3 generated artifacts (2.12 GiB)
  were copied to John2, verified by full rsync checksums plus independent file
  count/byte totals, and only then removed locally. John1 has 60,185,829,376
  free bytes; after reserving a 1.25-GiB atomic checkpoint it retains
  58,843,652,096 bytes, comfortably above the invariant. The failed 400K
  attempt published no checkpoint, so its complete log/manifest/loss evidence
  was retired under `cycle-02/training/failure-history` and Cycle-2 origin-1
  will restart deterministically from the same frozen bootstrap parent and
  unchanged corpus. Protected seeds remain sealed.

- 09:36 EDT — Live Cycle-2 profiling found a second, independent throughput
  defect before any model exposure: quota-proportional preprocessing assigned
  one thread each to the costly older broad/teacher balanced streams, while
  three cheaper producers became pipe-blocked and their CPUs could not be
  reused. The zero-exposure attempt was stopped and preserved with a durable
  failure receipt. Allocation now uses all nine authorized John1 CPU slots by
  measured producer cost (`1/3/1/2/2` for current broad/current teacher/recent/
  older broad/older teacher). Data order, quotas, D6 sequence, RNG, and MLX
  math are unchanged; Rayon expansion is byte-order deterministic across
  thread counts. Twenty-five focused Python tests, Ruff/diff checks, and the
  Rust serial-versus-parallel identity test pass. Cycle-2 origin-1 is
  restarting from the same frozen parent and corpus; protected seeds remain
  sealed.

- 09:56 EDT — The optimized Cycle-2 origin crossed its census handoff and
  opened all five live balanced producers with the exact registered nine-core
  allocation: current broad `1`, current teacher `3`, recent `1`, older broad
  `2`, older teacher `2`. This verifies the controller-to-worker wiring rather
  than merely the allocation unit tests. All producers are advancing through
  the rare-stratum replay; no model exposure, error, swap growth, or protected
  seed access has occurred yet.

- 10:11 EDT — Cycle-2 origin-1 completed pass 1 at exactly 400,000 exposures
  and atomically published checkpoint `step-000000050-epoch-0000-batch-000000`.
  Independent reload recovered step `50`, exposure `400000`, schedule block
  `1`, and batch cursor `0`; all model, optimizer, and state files are bound by
  recorded BLAKE3 hashes. The complete first 25 loss samples are byte-for-byte
  identical to the earlier deterministic attempt, proving the nine-core
  allocation changed latency only. After the 1.11-GiB checkpoint, John1 still
  has 58,360,631,296 free bytes, safely above the unchanged 50-GiB floor, and
  swap delta remains zero. Pass 2 is live; protected seeds remain sealed.

- 10:54 EDT — Cycle-2 origin-1 completed pass 2 at exactly 800,000 exposures
  and atomically published checkpoint `step-000000100-epoch-0000-batch-000000`.
  A fresh loader recovered step `100`, exposure `800000`, schedule block `2`,
  and batch cursor `0`. The two-generation recovery window is intact: both the
  400K and 800K checkpoints retain checksum-bound model, optimizer, and state
  payloads. John1 has 57,146,941,440 free bytes with the V3 tree at 8.0 GiB,
  leaving enough room to publish pass 3 before pruning the oldest generation.
  Pass 3 is live, swap remains flat, and protected seeds remain sealed.

- 11:21 EDT — Cycle-2 origin-1 completed all three passes and exactly
  1,200,000 exposures with final loss `0.0004897956`. Independent reload of
  final checkpoint `step-000000150-epoch-0000-batch-000000` recovered global
  step `150`, schedule block `3`, and a clean cursor; checkpoint pruning
  correctly retained the 800K and 1.2M recovery generations. Measured active
  training time was 5,689.4 seconds. The full-pass trace showed that the
  first-batch-tuned `1/3/1/2/2` allocation overfed teacher streams and left
  both 120K broad streams as the final tail. Before origin-2 started, the
  operational allocation was prospectively retuned to the measured-makespan
  profile `2/2/2/1/2` (current broad/current teacher/recent/older broad/older
  teacher). Data order and training math remain thread-count invariant;
  25 focused tests, Ruff/diff checks, and the Rust serial/parallel identity
  test pass. Protected seeds remain sealed.

- 11:39 EDT — Cycle-2 origin-1 passed its quantized open evaluation and
  serving export: 554,624 validation examples at 2,346.7 examples/s,
  power-loss `0.00231966`, RMSE `6.5817` points, MAE `4.9139` points, plus 32
  open games. The serving bundle is compatible and bound to weights BLAKE3
  `d6adc5f1...`; evaluation SHA256 is `9b54ea65...`. Origin-2 then completed
  its census and opened all five live producers with the full-pass allocation
  recorded in its immutable manifest: `2/2/2/1/2`. Dashboard progress is
  1.2M/2.4M candidate exposures. Protected seeds remain sealed.

- 11:57 EDT — Cycle-2 origin-2 completed pass 1 at exactly 400,000 exposures
  and atomically published checkpoint
  `step-000000050-epoch-0000-batch-000000`; independent reload recovered step
  `50`, exposure `400000`, schedule block `1`, and a clean cursor. Pass 1 took
  about 16 minutes with the full-pass `2/2/2/1/2` producer allocation, versus
  about 35 minutes for origin-1's earlier allocation. Origin-1 lifecycle
  compaction has reclaimed 1,192,469,561 bytes while retaining its complete
  1.2M exact-resume checkpoint and verified serving/evaluation artifacts.
  Colima sparse blocks were trimmed and 708 MiB of rebuildable Cargo dev-profile
  artifacts were cleaned; John1 now has 51.696 GiB free, leaving 456.5 MiB
  above the 50-GiB floor after reserving the next 1.25-GiB atomic checkpoint.
  Origin-2 pass 2 is live, swap remains flat, and protected seeds remain sealed.

- 12:27 EDT — Cycle-2 origin-2 completed pass 2 at exactly 800,000 exposures
  and atomically published checkpoint
  `step-000000100-epoch-0000-batch-000000`. Its immutable state records global
  step `100`, schedule block `2`, a clean batch cursor, and 2,774.996 seconds
  of active training; pass 2 itself consumed 1,768.985 seconds. The checkpoint
  is manifest-bound and contains checksum-bound model, optimizer, and trainer
  state payloads. Pass 3 opened all five producers normally. To preserve the
  exact 50-GiB physical floor through the final atomic write, only rebuildable
  npm, Cargo registry, and Cargo target intermediates were cleaned; all
  required release executables remain present and the live workers are
  unaffected. John1 now has 51.888 GiB free, leaving 653.2 MiB above the floor
  after the registered 1.25-GiB checkpoint reserve. Protected seeds remain
  sealed.

- 13:01 EDT — Cycle-2 training and open selection completed. Origin-2 reached
  1,200,000 exposures in 4,162.8 active seconds with final loss `0.00046410`,
  26.8% faster than origin-1, and passed export plus open validation at
  2,383.0 examples/s. Both origins passed the 128-game open nonregression
  gate. The registered selection rule chose origin-1 because its quantized
  validation loss was lower (`0.00231966` versus `0.00283868`), despite the
  faster origin-2 having a slightly higher open-game mean (`85.633` versus
  `85.484`). Origin-2's checkpoints were retired only after its serving and
  evaluation evidence was complete; origin-1 retains its exact-resume final
  checkpoint. Cross-backend qualification passed with Rust/MLX and
  scalar/NEON bit identity plus `1.0` float/quantized top-32 agreement. Frozen
  candidate weights SHA256 are `bb062ddb10734a8395151e79bb44707ca417beee01be6830d582aa098757eddf`.
  The state machine advanced to `cycle-02-promotion`; protected seeds remain
  sealed.

- 13:50 EDT — Cycle-2 promotion completed the registered first look: 100
  pairs in each of four tiers, 800 physical games, and all 80 immutable worker
  items succeeded under the canonical Docker digest. Raw paired means are
  direct `+0.45`, K32/R64 `+0.40`, K32/R600 `+0.21`, and equal-wall-time
  `-0.17`. No v2 always-valid boundary fired, so the controller correctly
  opened the disjoint 100–199 increment without inspecting or touching the
  protected final-evaluation domain. Protected seeds remain sealed.

- 14:38 EDT — Cycle-2 promotion completed 200 pairs per tier with all 160
  immutable work items and 1,600 physical games passing. Cumulative raw paired
  means are direct `+0.415`, K32/R64 `+0.110`, K32/R600 `+0.320`, and
  equal-wall-time `+0.005`; all four are now nonnegative, but no registered
  always-valid boundary has fired. The controller opened disjoint pairs
  200–299. Protected seeds remain sealed.

- 15:30 EDT — Cycle-2 promotion reached 300 pairs per tier with all 240 work
  items and 2,400 physical games passing. Cumulative raw means remain positive
  in every tier: direct `+0.157`, K32/R64 `+0.390`, K32/R600 `+0.270`, and
  equal-wall-time `+0.100`. The registered v2 boundaries remain open, so
  disjoint pairs 300–399 started. Protected seeds remain sealed.

- 16:20 EDT — Cycle-2 promotion reached 400 pairs per tier with all 320 work
  items and 3,200 physical games passing. Cumulative raw paired means are
  direct `+0.193`, K32/R64 `+0.193`, K32/R600 `-0.045`, and equal-wall-time
  `-0.098`. No v2 boundary fired; the controller opened the final registered
  400–499 increment. Protected seeds remain sealed.

- 17:14 EDT — Cycle-2 promotion completed its full 500-pair-per-tier maximum:
  400 immutable work items and 4,000 physical games all passed. The frozen v2
  verdict retained the incumbent as inconclusive. Final raw paired means were
  direct `+0.256`, K32/R64 `+0.150`, K32/R600 `+0.078`, and equal-wall-time
  `-0.012`; none crossed an always-valid boundary. The canonical bootstrap
  checkpoint therefore remains champion for Cycle 3. The first transition
  attempt correctly refused because completed VirtioFS inputs were still held
  open by the Colima VM, temporarily reducing physical free space. With no
  live jobs, a full guest trim plus controlled exact-profile Colima restart
  closed those deleted descriptors and restored John1 from 3.2 to 131 GiB
  free; MinIO and the registry restarted under their existing `unless-stopped`
  policies. The controller resumed from durable evidence without rerunning a
  game and advanced to `cycle-03-collecting` across all 29 CPUs. Protected
  seeds remain sealed.

- 17:16 EDT — The VirtioFS reclaim is now a permanent, idempotent campaign
  lifecycle step. After each future promotion increment, the controller first
  verifies that every immutable item succeeded and artifacts reconciled,
  refuses to touch the dedicated VM if any non-control container is live,
  trims the guest, restarts the exact Colima profile, verifies MinIO and the
  registry recovered, enforces the 50-GiB floor, and writes a durable receipt.
  Eight targeted promotion/reclaim tests pass, Ruff is clean, and Cycle-3
  collection remained live throughout the host-only source change. Protected
  seeds remain sealed.

- 18:07 EDT — Cycle-3 collection completed exactly 10,000 games in 100/100
  immutable shards; verification and corpus reconciliation passed, and the
  state advanced to `cycle-03-labeling`. The next-cycle lifecycle now applies
  the same terminal-evidence VM reclaim after both collection and teacher
  labeling, not only promotion, preventing model-input descriptor accumulation
  across every Docker-heavy stage. Nine targeted reclaim/data/promotion tests
  pass and Ruff is clean. The already-running Cycle-3 pipeline is selecting its
  2,500 teacher roots; protected seeds remain sealed.

- 18:16 EDT — Cycle-3 teacher search completed all 25/25 shards and exactly
  2,500 registered roots; label reconciliation passed and the state advanced
  to `cycle-03-training`. Origin-1 is live on John1 from the retained bootstrap
  champion while John2 and John3 run the previous-checkpoint benchmark. The
  campaign tree is 6.52 GiB with 108 GiB physically free, and the protected
  domain remains sealed.

- 18:51 EDT — Cycle-3 origin-1 completed pass 1 at exactly 400,000 exposures
  and atomically published `step-000000050-epoch-0000-batch-000000`. The
  checksum-bound state records global step `50`, schedule block `1`, a clean
  cursor, 1,530.3 active seconds, and the registered `2/2/2/1/2` expansion
  allocation. John1 retains 104 GiB free after the 1.11-GiB checkpoint. Pass 2
  is live and protected seeds remain sealed.

- 19:16 EDT — Cycle-3 origin-1 completed pass 2 at exactly 800,000 exposures
  and published its second atomic checkpoint. State reload metadata records
  global step `100`, schedule block `2`, a clean cursor, and 2,862.3 active
  seconds. John1 has 103 GiB free, pass 3 is live, and protected seeds remain
  sealed.

- 19:31 EDT — Cycle-3 origin-1 completed all three passes and exactly
  1,200,000 exposures in 3,949.6 active seconds, with final loss `0.00045510`.
  Final checkpoint `step-000000150-epoch-0000-batch-000000` is manifest-bound
  with global step `150`, schedule block `3`, and a clean cursor. Quantized
  open evaluation is live; protected seeds remain sealed.

- 19:41 EDT — Cycle-3 origin-1 passed serving export and open quantized
  evaluation: 554,624 examples at 2,272.0 examples/s, power loss `0.00235544`,
  RMSE `6.6063`, and MAE `4.9335` score points. Its serving weights are bound
  by BLAKE3 `11c40aa5...`. Origin-2 started independently from the same frozen
  parent and corpus; protected seeds remain sealed.

- 20:21 EDT — Cycle-3 origin-2 completed pass 1 at exactly 400,000 exposures
  and published a manifest-bound atomic checkpoint with global step `50`,
  schedule block `1`, a clean cursor, and 1,800.8 active seconds. Pass 2 is
  live; protected seeds remain sealed.

- 20:57 EDT — Cycle-3 origin-2 completed all three passes and exactly
  1,200,000 exposures in 3,967.1 active seconds with final loss `0.00044564`.
  Its final exact-resume checkpoint is manifest-bound and complete. Quantized
  open evaluation is live; John1 has 100 GiB free and protected seeds remain
  sealed.

- 21:08 EDT — Cycle-3 open selection chose origin-2 under the frozen
  minimum-validation-loss rule: `0.00235124` versus origin-1's `0.00235544`.
  Both passed the 128-game nonregression gate; origin-2's open mean was
  `85.391` versus `85.555`, a statistically acceptable `-0.164` delta under
  the registered `-1.0` margin. The selected bundle passed Rust/MLX and
  scalar/NEON bit identity with `1.0` float/quantized top-32 agreement and was
  frozen under weights SHA256 `e9c1dc0d6c2d3a38f07d0668270f7b7e88b28d19b5d22e13326e2472c761ca46`.
  Cycle-3 promotion is live; protected seeds remain sealed.

- 22:05 EDT — Cycle-3 promotion completed its first registered look: 100 pairs
  per tier, 800 physical games, and all 80 immutable worker items passed. Raw
  paired means are direct `-0.34`, K32/R64 `-0.17`, K32/R600 `+0.26`, and
  equal-wall-time `-0.15`; no v2 boundary fired. The post-increment reclaim
  initially stopped before opening the next domain because the LaunchAgent's
  minimal `PATH` could not satisfy a Docker helper invoked by Colima, leaving
  the exact profile between restart steps. The reclaim environment now binds
  Homebrew and system binary directories explicitly; six focused tests and
  Ruff pass. The exact profile recovered both MinIO and the registry, then a
  checksum-bound reclaim receipt recorded 23,559,790,592 bytes reclaimed and
  140,952,244,224 bytes free. Disjoint pairs 100–199 are now running, and
  protected seeds remain sealed.

- 00:23 EDT — Cycle-3 promotion completed its second registered look at 200
  pairs per tier and 1,600 cumulative physical games. All immutable work items,
  integrity checks, and resource checks passed. Raw paired means are direct
  `+0.105`, K32/R64 `-0.270`, K32/R600 `+0.155`, and equal-wall-time `-0.035`;
  all intervals still include zero and no v2 boundary fired. The post-block
  lifecycle reclaimed another 11,075,096,576 bytes and verified the exact
  control services before block 200–299.

- 00:24 EDT — During idempotent admission of block 200–299, Bacalhau's jobs
  endpoint transiently timed out after all four client attempts. Five work
  items had already succeeded and the other registered jobs remained durable;
  no completed pair or seed was lost. The scheduler endpoint is healthy again.
  Corrected the campaign LaunchAgent from `KeepAlive=false` to
  `KeepAlive.SuccessfulExit=false`, so a nonzero transient controller exit is
  now restarted automatically while a successful completed campaign remains
  stopped. `plutil` validation passed, the exact controller resumed under one
  advisory lock, and it is reconciling the same block 200–299. John1 has 122
  GiB free and protected seeds remain sealed.

- 00:29 EDT — Live stack and endpoint timing isolated the transient failure:
  Bacalhau v1.9's label-filtered job-list recovery took 12–30 seconds after the
  ledger grew past 3,500 jobs, while that recovery call inherited the generic
  15-second transport timeout. Added a dedicated 60-second `list_jobs` timeout
  without relaxing health or per-job calls. All 22 Bacalhau adapter/client unit
  tests pass, Ruff and diff checks are clean. The resumed request recovered all
  80 original job IDs, then advanced immediately to 43/80 succeeded; there was
  no resubmission, pair repetition, or seed change. Protected seeds remain
  sealed.

- 01:09 EDT — Cycle-3 promotion completed the recovered block without a
  duplicate or failure: 300 pairs per tier and 2,400 cumulative physical games
  now pass. Raw means are direct `+0.133`, K32/R64 `-0.137`, K32/R600 `+0.220`,
  and equal-wall-time `+0.030`; no registered boundary fired. The lifecycle
  reclaimed 16,113,311,744 bytes, restored 141,340,729,344 bytes free, and
  verified the control plane. Disjoint pairs 300–399 are live with 35 workers
  running at admission. Protected seeds remain sealed.

- 01:41 EDT — The failure-only LaunchAgent restart policy exercised correctly
  during block 300–399. Bacalhau v1.9 returned HTTP 500 `no changes detected`
  for an idempotent replay while embedding the already-created Job ID in the
  response. The controller restarted once, recovered the original domain, and
  advanced to 56/80 succeeded without duplicating a work item. Normalized this
  scheduler response at the adapter boundary into a successful recovered
  submission; unrelated HTTP 500 responses still retry and fail normally. All
  24 focused adapter/client tests pass, Ruff and diff checks are clean. Block
  300–399 remains live and protected seeds remain sealed.

- 01:58 EDT — Cycle-3 promotion completed its fourth look: 400 pairs per tier,
  3,200 cumulative physical games, and every work item valid. Raw means are
  direct `+0.258`, K32/R64 `-0.168`, K32/R600 `+0.235`, and equal-wall-time
  `+0.135`; no registered boundary fired. The lifecycle reclaimed
  12,023,623,680 bytes and restored 139,086,925,824 bytes free. The final
  registered Cycle-3 domain, disjoint pairs 400–499, is live. Protected seeds
  remain sealed.

- 02:56 EDT — Cycle-3 promotion completed the registered maximum: 500 pairs
  per tier and exactly 4,000 physical games. All integrity and resource gates
  passed. Final raw paired means are direct `+0.096`, K32/R64 `+0.008`,
  K32/R600 `+0.268`, and equal-wall-time `+0.270`. Every tier reached the
  maximum without crossing the always-valid alternative boundary, so the
  preregistered verdict is `retain-incumbent-inconclusive`; the bootstrap
  champion remains the parent. The final lifecycle reclaimed 15,764,115,456
  bytes and restored 140,099,936,256 bytes free. Cycle-4 advanced atomically
  to collection and its 100 immutable 100-game shards are live under the
  registered 80% V1-opponent mixture. Protected seeds remain sealed.

- 03:44 EDT — Cycle-4 collection and distributed replay verification passed:
  exactly 10,000 games, 40,000 seat-games, 200,000 focal training entries, and
  100/100 immutable shards. The realized seat composition is exactly 24,000 V1
  opponent seats, 6,000 prior-V3 opponent seats, and 10,000 newest focal seats.
  The corpus is scientifically eligible and bound by canonical SHA256
  `1b6e4fdcc6fa0fa3cd657d059eff34194abe42880d11226290f79818a3fa4db2`.
  Post-collection lifecycle reclaimed 31,020,163,072 bytes and restored
  141,224,415,232 bytes free. Campaign phase advanced atomically to Cycle-4
  teacher labeling; protected seeds remain sealed.

- 03:55 EDT — Cycle-4 teacher labeling and reconciliation passed: exactly
  2,500 roots, 70,479 candidate estimates, and 1,500,000 K32/R600 rollouts in
  25/25 shards. The 6,102,462-byte eligible teacher corpus is bound by
  canonical SHA256
  `2eef88d476567fe180b64d6896063bde4f13b373d05584665903445c9b8fd3f2`.
  The labeling lifecycle reclaimed 171,048,960 bytes. Campaign phase advanced
  atomically to Cycle-4 training; origin-1 is live natively on John1 from the
  retained bootstrap parent while John2 and John3 benchmark the previous
  checkpoint. Protected seeds remain sealed.

- 04:42 EDT — Cycle-4 origin-1 completed pass 1 at exactly 400,000 exposures
  and atomically published `step-000000050-epoch-0000-batch-000000` with global
  step `50`, schedule block `1`, and a clean pass-boundary cursor. The complete
  1.19-GiB checkpoint includes model, optimizer, state, and checksum manifest;
  pass 2 is live. The two John2/John3 parent benchmarks also completed all
  1,000 ineligible games: mean `84.082`, P10/P50/P90 `79/84/89`, with John1
  excluded by the 10-CPU request as registered. John1 has 130 GiB free, swap
  use is decreasing, and protected seeds remain sealed.

- 04:46 EDT — Preventive John2 worker maintenance completed between scheduler
  phases with zero running containers. Docker removed only unused image layers,
  then the default Colima guest was trimmed and restarted. This released
  168,424,692 KiB of host space: `df` increased from 1,393,380 KiB to
  169,818,072 KiB free. Docker recovered on version 29.5.2 and the authoritative
  Bacalhau membership check again showed John1/John2/John3 `CONNECTED` with
  9/10/10 CPUs and Docker execution enabled. The canonical image remains in the
  John1 registry for content-addressed repull. Cycle-4 training was not
  interrupted and protected seeds remain sealed.

- 04:48 EDT — The same idle, evidence-gated Colima trim/restart completed on
  John3 with no running containers. `df` increased from 144,873,448 KiB to
  353,549,772 KiB free, releasing 208,676,324 KiB retained by the worker VM.
  Post-maintenance validation again passed the complete 9/10/10 scheduler
  fabric and Docker-engine gate. Both CPU workers now have ample headroom for
  the remaining expert-cycle and protected-evaluation jobs; Cycle-4 MLX
  training remained live and protected seeds remain sealed.

- 04:51 EDT — The remote disk leak now has a permanent evidence-gated
  lifecycle: every completed collection, teacher-label, promotion increment,
  and final-evaluation increment serially trims/restarts idle John2 and John3,
  refuses any live container, requires at least 50 GiB free, and revalidates
  the full scheduler fabric after each worker. Final all-V3 evaluation is
  bounded into immutable 100-game increments. Final reporting now includes
  complete per-cycle loss curves, champion and checksum provenance, anatomy
  percentiles and histograms, latency/throughput, seven-day fleet memory
  telemetry, per-host swap before/after/delta, and authoritative JSON plus
  Markdown outputs. Twenty targeted lifecycle, orchestration, aggregation, and
  report tests pass; Ruff, bytecode compilation, and diff checks are clean.
  Cycle-4 origin-1 pass 2 remains live with four batch streamers; protected
  seeds remain sealed.

- 04:52 EDT — The complete targeted V3 Python regression surface passed:
  124/124 tests across campaign state, collection, training, promotion,
  lifecycle reclaim, final aggregation, and reporting. The remote lifecycle
  also now waits boundedly for terminal Bacalhau containers and for scheduler
  reconnection, preventing a harmless teardown delay from aborting a cycle.

- 04:55 EDT — Live dashboard training telemetry now recognizes structurally
  verified atomic expert-cycle checkpoints without continuously rereading the
  1.19-GiB payload. It checks the model/optimizer/state size manifest, all
  recorded BLAKE3 identities, clean pass-boundary cursor, and run-manifest
  binding; scientific resume/evaluation still performs full content checks.
  The top-level dashboard now shows origin-1's verified pass-1 checkpoint,
  602,880/2,400,000 total Cycle-4 exposures, approximately 189 examples/s, and
  a 9,516-second live ETA. Fourteen dashboard/campaign tests pass. Protected
  seeds remain sealed.

- 05:00 EDT — Cycle-4 origin-1 completed pass 2 at exactly 800,000 exposures,
  global step `100`, schedule block `2`, and a clean zero-offset cursor. The
  atomic `step-000000100-epoch-0000-batch-000000` checkpoint's model,
  optimizer, and state sizes match its manifest, the run binding matches
  `31fdd0f4...`, and an independent full BLAKE3 reread passed all three files.
  Mean training loss fell from `0.00115697` in pass 1 to `0.000622164` in pass
  2; the boundary loss was `0.000468796`. Pass 3 began immediately at the
  registered `1e-5` learning rate with five native batch producers. Dashboard
  identity/progress updated correctly, John1 retains 129 GiB free, swap is
  decreasing, and protected seeds remain sealed.

- 05:22 EDT — The long Cycle-4 pass-3 fill was traced precisely to
  `older_broad`: selecting 40,000 phase/score-balanced examples requires
  scanning the 500,000-game bootstrap corpus over D6 replay, but the prior cost
  model assigned only one expansion thread because it considered output quota
  rather than scan amplification. Future training subprocesses now assign this
  source three of the fixed nine preprocessing threads while preserving the
  exact quotas and total CPU budget. Rust's indexed parallel expansion test
  proves identical record values and canonical order across thread counts;
  11 cycle-training tests, Ruff, and diff checks pass. The running origin-1
  process remains untouched on its frozen manifest; origin-2 and subsequent
  cycles receive the latency-only correction. Protected seeds remain sealed.

- 05:42 EDT — Cycle-4 origin-1 completed all three registered passes at
  exactly 1,200,000 exposures and global step `150`; its atomic final
  checkpoint has a clean pass boundary and the run manifest reports success.
  Export/reload and open qualification passed: 554,624 validation positions at
  2,570 examples/s, quantized power loss `0.002531609`, RMSE `6.8045` score
  points, and a serving-compatible 105-MiB bundle with weights BLAKE3
  `d7e9d869...`. Origin-2 then started independently from the same frozen
  bootstrap parent with seed `90042` and the verified preprocessing allocation
  improvement. Protected seeds remain sealed.

- 06:05 EDT — The live origin-2 producer trace refined the preprocessing cost
  model again without touching its frozen run: three-thread `older_broad`
  completed first, two-thread `older_teacher` completed next, and two-thread
  `recent` filled its pipe, while one-thread current broad and current teacher
  remained CPU-bound after fourteen minutes. Future origins now use the
  balanced fixed-nine allocation `2/2/1/2/2` for current broad, current
  teacher, recent, older broad, and older teacher. Quotas, ordering, records,
  training math, and the active origin-2 manifest are unchanged. All 11 cycle
  training tests and Ruff pass. Protected seeds remain sealed.

- 06:15 EDT — Cycle-4 origin-2 completed pass 1 at exactly 400,000
  exposures/global step `50` and began pass 2 immediately. Its first atomic
  checkpoint has a clean cursor and run binding `5a86128c...`; an independent
  full reread reproduced model, optimizer, and state BLAKE3 values
  `6e2325c5...`, `80aa6028...`, and `3ddb358f...`. Mean pass-1 loss was
  `0.000981716`, with boundary loss `0.000715258`. The pass took approximately
  25 minutes after manifest freeze versus about 40 minutes for origin-1,
  demonstrating a measured 1.6x first-pass improvement from the initial
  producer correction. Dashboard progress and checkpoint identity are exact;
  protected seeds remain sealed.

- 06:39 EDT — Cycle-4 origin-2 completed pass 2 at exactly 800,000
  exposures/global step `100` and entered pass 3 at the registered `1e-5`
  learning rate. The atomic checkpoint is clean and bound to `5a86128c...`;
  independent full BLAKE3 reads match its model `14a93d6b...`, optimizer
  `9b11801a...`, and state `e2b7eb5e...` manifest entries. Mean pass-2 loss was
  `0.000625834` and boundary loss was `0.000452915`. Dashboard progress is
  exactly 2,000,000/2,400,000 Cycle-4 exposures with about 28 minutes of
  origin training remaining. Protected seeds remain sealed.

- 07:06 EDT — Cycle-4 origin-2 completed all three passes at exactly
  1,200,000 exposures/global step `150` in 4,496.05 training seconds. Mean
  pass-3 loss was `0.000569610` and boundary loss was `0.000486575`. Its final
  atomic checkpoint is clean and bound to `5a86128c...`; independent full
  BLAKE3 reads match model `fbb73146...`, optimizer `74756758...`, and state
  `432b9fdc...`. Quantized validation/export qualification started
  automatically. Protected seeds remain sealed.

- 07:16 EDT — Origin-2 qualification passed 554,624 quantized validation
  examples at 2,511.83/s with power loss `0.002203216`, RMSE `6.4130`, and MAE
  `4.7846` points. Although this validation loss beat origin-1's `0.002531609`,
  origin-2 failed the preregistered open-domain nonregression gate: paired mean
  delta `-0.1953`, SE `0.4458`, lower 95% `-1.0690` versus the `-1.0` margin
  over 128 pairs. The selection rule therefore correctly retained origin-1,
  whose 100-group x 64-candidate Rust/MLX parity gate started immediately.
  Protected seeds remain sealed.

- 07:17 EDT — The selected origin-1 candidate passed the complete 100-group,
  6,400-row cross-backend parity gate: Rust scalar/NEON, Rust/MLX quantized,
  and MLX/NumPy were bit-identical; float/quantized top-1 and top-32 agreement
  were both `1.0`, maximum absolute error was zero, and overflow remained
  exact. The frozen candidate has weights SHA256 `1667ee32...` and model
  manifest SHA256 `7727a752...`. Campaign state advanced atomically to
  `cycle-04-promotion`, and the four-tier always-valid paired test launched on
  the canonical Docker workers. Protected seeds remain sealed.

- 08:10 EDT — Cycle-4 promotion completed its first immutable 100-pair look
  per tier (800 physical games) with every one of 80 work items successful.
  Raw paired means were direct `-0.21`, K32/R64 `+0.39`, K32/R600 `+0.16`, and
  equal-wall-time `+0.45`; every integrity/resource gate passed, but no
  always-valid boundary fired, so the registered second 100-pair domain began
  automatically. Evidence-gated lifecycle maintenance reclaimed 13.49 GB on
  John1 plus 25.28 GB across John2/John3 and reverified the full Docker fabric.
  Protected seeds remain sealed.

- 08:13 EDT — First-look timing showed a correct but avoidable scheduling
  tail: K32/R600 averaged 141.95 seconds/physical game and equal-wall-time
  123.29 seconds/game, while five-pair items forced the 200 long-tier games
  into coarse waves and a 50-minute makespan. Future cycle promotion processes
  now shard at one pair per immutable item (400 items/look), preserving every
  registered pair, RNG domain, tier, and search budget while approaching the
  29-CPU work-conservation bound. The already-running Cycle-4 process remains
  frozen on five-pair shards. Fourteen promotion/lifecycle tests, Ruff, and
  bytecode compilation pass. Protected seeds remain sealed.

- 08:58 EDT — Cycle-4 promotion completed the cumulative 200-pair look per
  tier (1,600 physical games total), with all work items successful and every
  integrity/resource gate green. Cumulative paired means are direct `-0.09`,
  K32/R64 `+0.385`, K32/R600 `+0.01`, and equal-wall-time `+0.025`. No
  always-valid alternative or null boundary fired, so the registered third
  disjoint 100-pair domain proceeds after lifecycle reclamation. Protected
  seeds remain sealed.

- 09:47 EDT — Cycle-4 promotion completed 300 cumulative pairs per tier
  (2,400 physical games), again with every work item and integrity/resource
  gate passing. Paired means are now directionally positive in all tiers:
  direct `+0.07`, K32/R64 `+0.22`, K32/R600 `+0.0567`, and equal-wall-time
  `+0.0033`. The always-valid evidence remains below the alternative boundary,
  so the fourth disjoint 100-pair look proceeds after lifecycle reclamation.
  Protected seeds remain sealed.

- 10:02 EDT — A transient Bacalhau HTTP 503 restarted the fourth-look
  orchestrator after its durable 80-item request had opened. Because the new
  one-pair optimization changed the default job definition, strict recovery
  correctly refused to reinterpret that already-opened five-pair request. The
  orchestration boundary now reads and checksum-validates an opened managed
  request's frozen shard size, validates its complete tier/domain cardinality,
  and uses the new one-pair default only for unopened requests. Sixteen focused
  promotion/lifecycle tests and static checks pass. The controller reconnected
  to the original request with no duplicate or reopened pairs; fourth-look
  progress recovered at 40 succeeded, 39 running, one queued. Protected seeds
  remain sealed.

- 10:38 EDT — The recovered fourth Cycle-4 promotion look completed with all
  80 original work items successful, bringing the test to 400 pairs per tier
  and 3,200 physical games. Cumulative means are direct `+0.07`, K32/R64
  `+0.0825`, K32/R600 `+0.05`, and equal-wall-time `-0.0825`; all integrity and
  resource gates pass, and no sequential boundary fired. The mandatory final
  100-pair domain will be the first unopened request to use pair-granular
  scheduling. Protected seeds remain sealed.

- 11:37 EDT — Cycle-4 promotion completed the exact registered maximum of 500
  pairs per tier (4,000 physical games). Final paired means were direct
  `+0.046`, K32/R64 `+0.082`, K32/R600 `+0.158`, and equal-wall-time `-0.022`;
  all gates passed, but every tier reached `inconclusive-maximum`, yielding the
  honest `retain-incumbent-inconclusive` verdict. The bootstrap champion remains
  frozen as Cycle-4 champion (`weights SHA256 5eb9fcec...`). The one-pair final
  look took 3,479 seconds because tier-major admission serialized its 400 jobs;
  future unopened requests now interleave all four tiers per pair, while
  durable recovery preserves any opened order exactly. Sixteen focused tests
  pass. State advanced atomically to Cycle-5 collection, which launched the
  exact 10,000-game/80%-V1 contract with prior candidates in the frozen pool.
  Protected seeds remain sealed.

- 12:39 EDT — Cycle-5 collection and distributed replay verification passed:
  exactly 10,000 games, 40,000 seat-games, 200,000 focal training entries, and
  all 100 shards. Realized composition exactly matches the registered contract:
  24,000 V1 opponent seats, 6,000 prior-V3 opponent seats, and 10,000 newest
  focal seats. The 14,099,541-byte scientific corpus is internally bound by
  canonical SHA256 `54c4a3a5...` and the transition evidence by SHA256
  `77482e69...`. Evidence-gated lifecycle maintenance reclaimed 66.00 GB across
  John2/John3 and reverified the full fabric. State advanced to Cycle-5
  labeling; 2,500-root stratified selection is live. Protected seeds remain
  sealed.

- 12:47 EDT — Cycle-5 teacher selection, labeling, and reconciliation passed:
  2,500 stratified roots from 200,000 positions across 2,222 observed strata,
  70,885 candidate estimates, exactly 1.5 million K32/R600 rollouts, and all
  25 shards. The eligible 6,108,948-byte label corpus is canonically bound by
  SHA256 `b842cea2...`; transition evidence is SHA256 `9d66a9d2...`. State
  advanced atomically to Cycle-5 training. Origin-1 started on John1 from the
  retained bootstrap parent while John2/John3 run the one-iteration-behind
  parent benchmark. Protected seeds remain sealed.

- 12:56 EDT — Cycle-5 origin-1 completed its exact 400,000-example score-
  quantile census and atomically bound its run manifest. The registered
  1,200,000-exposure mixture is unchanged (`120K` current broad, `80K` current
  teacher, `120K` recent, `40K` older broad, `40K` older teacher per pass).
  The balanced preprocessing allocation is confirmed live as `2/2/1/2/2`
  threads respectively. Training has entered pass 1; the parent benchmark
  remains active on John2/John3 and protected seeds remain sealed.

- 13:20 EDT — Cycle-5 origin-1 pass 1 completed at exactly 400,000 exposures
  and step 50. The atomic checkpoint is bound to run-manifest BLAKE3
  `c1fa7205...`; independent BLAKE3 recomputation matched the model,
  optimizer, and state files exactly. Latest pass-1 loss is `0.000863489` at
  LR `3e-5`. Pass 2 opened immediately with all five registered streams;
  protected seeds remain sealed.

- 13:44 EDT — Cycle-5 origin-1 pass 2 completed at exactly 800,000 exposures
  and step 100. Independent BLAKE3 verification again matched the atomic
  model, optimizer, and state payloads. Latest loss fell to `0.000570756` at
  LR `3e-5`. Pass 3 opened at the registered `1e-5` consolidation rate;
  protected seeds remain sealed.

- 14:10 EDT — Cycle-5 origin-1 completed all three registered passes at
  exactly 1,200,000 exposures and step 150 in 4,435.87 training seconds.
  Final loss is `0.000513757` at LR `1e-5`. Independent BLAKE3 verification
  matched the final atomic model, optimizer, and state payloads. Qualification
  evaluation and serving export are now running; protected seeds remain sealed.

- 14:20 EDT — Cycle-5 origin-1 qualification passed and serving export froze
  successfully. Quantized validation covered 554,624 examples at 2,490.76/s:
  power loss `0.002827816`, RMSE `7.1305`, MAE `5.3283`. Export weights BLAKE3
  is `a10b5931...`. Checkpoints were compacted only after qualification, and
  independent origin-2 training started from the same parent with seed 90052.
  Protected seeds remain sealed.

- 14:29 EDT — Cycle-5 origin-2 completed its independent 400,000-example
  quantile census and bound the run manifest with seed 90052. It independently
  confirms the exact 1.2M-exposure mixture and the balanced `2/2/1/2/2`
  preprocessing allocation. Origin-2 pass 1 is active; protected seeds remain
  sealed.

- 14:58 EDT — Cycle-5 origin-2 pass 1 completed at exactly 400,000 exposures
  and step 50. The checkpoint is atomically bound to run-manifest BLAKE3
  `e784f0de...`; independent BLAKE3 verification matched model, optimizer, and
  trainer state. Latest loss is `0.000635899` at LR `3e-5`. Pass 2 opened;
  protected seeds remain sealed.

- 15:23 EDT — Cycle-5 origin-2 pass 2 completed at exactly 800,000 exposures
  and step 100. Independent BLAKE3 verification matched the atomic checkpoint
  payloads. Latest loss is `0.000462304` at LR `3e-5`. Pass 3 opened at
  `1e-5`; protected seeds remain sealed.

- 15:53 EDT — Cycle-5 origin-2 completed all three passes at exactly 1,200,000
  exposures and step 150 in 5,063.02 training seconds. Final loss is
  `0.000439587` at LR `1e-5`; independent BLAKE3 verification matched the
  final atomic checkpoint payloads. Qualification/export is active and
  protected seeds remain sealed.

- 16:05 EDT — Cycle-5 origin selection and candidate qualification completed.
  Origin-2 had the lower quantized validation loss (`0.002291419`, RMSE
  `6.5349`) but failed the registered open nonregression gate by 0.0576 points
  (`lower95=-1.05756` versus the fixed `-1.0` margin), so origin-1 was selected
  honestly. Origin-1 passed 6,400-row scalar/NEON/MLX parity with bit identity,
  zero float/quantized error, and top-1/top-32 agreement of 1.0. The frozen
  candidate evidence SHA256 is `0d50d9a9...`; state advanced atomically to
  Cycle-5 promotion and protected seeds remain sealed.

- 17:07 EDT — Cycle-5 promotion look 1 completed the exact 100 pairs per tier
  (800 physical games) with all 400 one-pair jobs accepted in 3,233.14 seconds.
  Integrity and resource gates passed; no always-valid boundary fired. Paired
  means were direct `-0.83`, K32/R64 `-0.40`, K32/R600 `+0.18`, and equal-wall
  `-0.75`, so the registered test correctly continues to 200 pairs. Profiling
  found 54,339 focal CPU-seconds but 400 repeated model-bundle stages, limiting
  fleet efficiency to roughly 62%. Future unopened promotion processes now
  use four-pair, pair-major items: 4x fewer bundle stages while preserving every
  pair/RNG/tier domain and the sequential estimand. Opened Cycle-5 requests
  remain checksum-frozen at one pair. Twelve targeted tests plus Ruff and
  bytecode compilation pass. Protected seeds remain sealed.

- 17:54 EDT — Cycle-5 promotion look 2 completed exactly 200 cumulative pairs
  per tier (1,600 physical games) with all integrity/resource gates passing.
  Cumulative paired means tightened to direct `-0.320`, K32/R64 `-0.095`,
  K32/R600 `+0.065`, and equal-wall `-0.170`; no always-valid boundary fired,
  so the registered test continues to 300 pairs. The second frozen one-pair
  request completed in 3,191.69 seconds. Protected seeds remain sealed.

- 18:48 EDT — Cycle-5 promotion look 3 completed exactly 300 cumulative pairs
  per tier (2,400 physical games) with all gates passing. Cumulative paired
  means are direct `+0.0967`, K32/R64 `-0.0967`, K32/R600 `-0.0367`, and
  equal-wall `-0.0967`. No always-valid boundary fired, so the test continues
  to 400 pairs. The third frozen one-pair request completed in 3,156.09
  seconds. Protected seeds remain sealed.

- 19:43 EDT — Cycle-5 promotion look 4 completed exactly 400 cumulative pairs
  per tier (3,200 physical games), again with every gate passing. Cumulative
  paired means are direct `+0.1450`, K32/R64 `+0.0025`, K32/R600 `+0.0275`,
  and equal-wall `-0.0850`. No always-valid boundary fired, so the mandatory
  final 100-pair look will run to the registered 500-pair maximum. Look 4 took
  3,263.38 seconds. Protected seeds remain sealed.

- 20:38 EDT — Cycle-5 promotion completed the exact registered maximum of 500
  pairs per tier (4,000 physical games). Final paired means are direct
  `+0.012`, K32/R64 `+0.082`, K32/R600 `+0.116`, and equal-wall `-0.168`.
  Every integrity/resource gate passed, but all four tiers ended at
  `inconclusive-maximum`; the honest verdict is
  `retain-incumbent-inconclusive`. The bootstrap incumbent remains Cycle-5
  champion. Independent SHA256 verification matches model `77ed15e2...` and
  weights `5eb9fcec...`; champion evidence SHA256 is `81d6f405...`. State
  advanced atomically to Cycle-6 collection, which is live across the canonical
  fabric. Protected seeds remain sealed.

- 21:03 EDT — Cycle-6 collection encountered a transient Bacalhau HTTP 503
  after 29/100 shards had succeeded. The LaunchAgent restarted the controller
  automatically and resumed the checksum-frozen managed request from its
  durable state: the same 29 shards remain terminal, with no duplicate or
  reopened seed domains. Collection is live again under new controller/data-
  pipeline processes. Protected seeds remain sealed.

- 21:25 EDT — A second Bacalhau 503 occurred at 47/100 Cycle-6 collection
  shards. The root orchestration path is now hardened: transient scheduler
  status/result read failures are recorded in progress and retried inside the
  phase monitor until the existing phase deadline, rather than terminating the
  controller. Ten targeted pipeline/API tests plus Ruff and py_compile pass.
  The LaunchAgent was safely reloaded against the durable request; no worker
  job was cancelled or reopened. Hardened monitoring is live and healthy at
  75/100 succeeded shards with zero post-reload scheduler errors. Protected
  seeds remain sealed.

- 21:41 EDT — Cycle-6 collection and distributed replay verification passed:
  exactly 10,000 games, 40,000 seat-games, 200,000 focal entries, and all 100
  shards. Composition is exact at 24,000 V1 opponent seats, 6,000 prior-V3
  opponent seats, and 10,000 newest focal seats. The 14,099,518-byte corpus is
  canonically bound by SHA256 `770e73c4...`; transition evidence is SHA256
  `0ea1a3d8...`. Hardened monitoring completed collection without another
  teardown and replay verified every record. State advanced atomically to
  Cycle-6 labeling; protected seeds remain sealed.

- 21:50 EDT — Cycle-6 teacher selection, labeling, and reconciliation passed:
  2,500 roots selected from 200,000 positions across 2,204 observed strata,
  70,922 candidate estimates, exactly 1.5 million K32/R600 rollouts, and all
  25 shards. The 6,113,896-byte eligible corpus is canonically bound by SHA256
  `763392f3...`; transition evidence is SHA256 `9eef3c74...`. State advanced
  atomically to Cycle-6 training. Origin-1 started from the retained bootstrap
  parent while John2/John3 benchmark the lagged champion. Protected seeds
  remain sealed.

- 21:58 EDT — Cycle-6 origin-1 completed its independent 400,000-example
  quantile census and atomically bound the seed-90061 run manifest. The exact
  1.2M-exposure mixture and balanced `2/2/1/2/2` preprocessing allocation are
  confirmed. Pass 1 is active; protected seeds remain sealed.

- 22:25 EDT — Cycle-6 origin-1 pass 1 completed at exactly 400,000 exposures
  and step 50. Its checkpoint is atomically bound to run-manifest BLAKE3
  `78f1242b...`; independent BLAKE3 verification matched model, optimizer, and
  trainer state. Latest loss is `0.000595781` at LR `3e-5`. Pass 2 opened;
  protected seeds remain sealed.

- 22:47 EDT — Cycle-6 origin-1 pass 2 completed at exactly 800,000 exposures
  and step 100. Independent BLAKE3 verification matched the atomic model,
  optimizer, and trainer-state payloads. Latest loss is `0.000412174` at LR
  `3e-5`. Pass 3 opened at `1e-5`; protected seeds remain sealed.

- 23:16 EDT — Cycle-6 origin-1 completed all three registered passes at
  exactly 1,200,000 exposures and step 150 in 4,631.15 training seconds. Final
  loss is `0.000391433` at LR `1e-5`; independent BLAKE3 verification matched
  the final atomic checkpoint payloads. Qualification/export is active and
  protected seeds remain sealed.

- 23:27 EDT — Cycle-6 origin-1 qualification passed and serving export froze
  successfully. Quantized validation covered 554,624 examples at 2,445.22/s:
  power loss `0.002532401`, RMSE `6.8031`, MAE `5.0816`; export weights BLAKE3
  is `53762248...`. Independent origin-2 training started from the same parent
  with seed 90062. Protected seeds remain sealed.

- 23:35 EDT — Cycle-6 origin-2 completed its independent 400,000-example
  quantile census and bound the seed-90062 run manifest. The exact 1.2M-
  exposure mixture and balanced `2/2/1/2/2` preprocessing allocation are
  confirmed. Pass 1 is active; protected seeds remain sealed.

- 23:58 EDT — Cycle-6 origin-2 pass 1 completed at exactly 400,000 exposures
  and step 50. Its atomic checkpoint is bound to run-manifest BLAKE3
  `e2111ae3...`; independent BLAKE3 verification matched all payloads. Latest
  loss is `0.000712929` at LR `3e-5`. Pass 2 opened; protected seeds remain
  sealed.

- 00:20 EDT (Jun 26) — Cycle-6 origin-2 pass 2 completed at exactly 800,000
  exposures and step 100. Independent BLAKE3 verification matched the atomic
  checkpoint payloads. Latest loss is `0.000475470` at LR `3e-5`. Pass 3
  opened at `1e-5`; protected seeds remain sealed.

- 00:44 EDT (Jun 26) — Cycle-6 origin-2 completed all three passes at exactly
  1,200,000 exposures and step 150 in 4,079.76 training seconds. Final loss is
  `0.000437931` at LR `1e-5`. Independent BLAKE3 verification matched the
  final model (`7956f1ec...`), optimizer (`dee8a231...`), and state
  (`f4289c2b...`) payloads bound to run manifest `e2111ae3...`.
  Qualification/export is active; protected seeds remain sealed.

- 00:53 EDT (Jun 26) — Cycle-6 origin-2 passed quantized qualification over
  554,624 validation examples: power loss `0.002499420`, RMSE `6.7753`, MAE
  `5.0629`, and serving weights BLAKE3 `2c5048c5...`. Although its validation
  loss was lower than origin-1, it failed the preregistered open nonregression
  gate (`delta -0.4297`, lower 95% bound `-1.2156` versus the `-1.0` margin).
  Origin-1 was therefore selected. Its 6,400-row cross-backend gate passed
  scalar/NEON/MLX bit identity and 100% top-1/top-32 agreement.

- 00:55 EDT (Jun 26) — Campaign state atomically advanced to
  `cycle-06-promotion` (transition 32). The first 100-pair look launched as
  exactly 100 scheduler items using the newly qualified four-pair shards; item
  order begins direct/R64/R600/equal-wall at pair indices 0, 4, and onward.
  This preserves every paired domain while reducing model staging fourfold.
  Protected seeds remain sealed.

- 01:47 EDT (Jun 26) — Cycle-6 promotion look 1 completed and validated all
  four tiers at exactly 100 pairs each (400 paired tier observations, 800
  physical games) in 2,786.36 seconds. All 100 four-pair scheduler items
  succeeded; one transient scheduler read was retried in place. No sequential
  boundary fired, so the exact disjoint 100–199 look launched. The measured
  first-look makespan is 1.13x faster than Cycle 5's one-pair layout; compute,
  not model staging, now dominates. Protected seeds remain sealed.

- 02:36 EDT (Jun 26) — Cycle-6 promotion look 2 completed at 200 cumulative
  pairs per tier. All 100 items succeeded, adding 400 paired tier observations
  and 800 physical games in 3,089.98 seconds with no scheduler read errors. No
  sequential boundary fired; the exact disjoint 200–299 look launched.
  Protected seeds remain sealed.

- 03:21 EDT (Jun 26) — Cycle-6 promotion look 3 completed at 300 cumulative
  pairs per tier. All 100 items succeeded, adding 400 paired tier observations
  and 800 physical games in 2,617.75 seconds with no scheduler read errors. No
  sequential boundary fired; the exact disjoint 300–399 look launched.
  Protected seeds remain sealed.

- 03:34 EDT (Jun 26) — Removed the recurring Bacalhau admission-ledger scan
  from future fresh managed requests without weakening recovery. A request
  definition is durably written before its first job; fresh admissions now use
  the existing deterministic idempotency token directly. After a reconnect,
  exactly the first missing item is still recovered by request/item/spec labels
  because it is the only item that can have been accepted between scheduler
  submission and the following atomic state write. Subsequent items are known
  fresh. Twenty-five focused client/API tests, py_compile, and Ruff pass,
  including crash-before-state-write recovery and conflict detection. The live
  opened Cycle-6 process was not restarted or altered; the optimization applies
  when the next campaign child process starts. Protected seeds remain sealed.

- 03:43 EDT (Jun 26) — The launchd supervisor received SIGTERM and restarted
  during Cycle-6 promotion look 4 after 36/100 four-pair items had succeeded.
  Durable request recovery retained all 68 submitted job IDs uniquely and the
  exact 100 unique item keys; no completed or in-flight pair domain was
  reopened. The replacement controller resumed the same 300–399 request.
  This was a controller restart, not a scientific failure; protected seeds
  remain sealed.

- 03:45 EDT (Jun 26) — Safely reloaded the headless controller against the
  checksum-frozen Cycle-6 look-4 request to activate the admission fix. At the
  reload boundary, 68 job IDs were durably assigned and 32 items remained
  unopened; no job was cancelled and no pair domain changed. The new process
  performed the single required orphan-recovery scan, then advanced from 68 to
  87 assigned IDs in 30 seconds instead of scanning the full scheduler ledger
  per item. Monitoring is healthy at 50 succeeded shards and 90 assigned IDs,
  with zero post-reload scheduler errors. Protected seeds remain sealed.

- 04:08 EDT (Jun 26) — Cycle-6 promotion look 4 recovered and completed at
  400 cumulative pairs per tier. All 100 four-pair items ultimately succeeded,
  adding 400 paired tier observations and 800 physical games in about 46
  minutes end to end, including the controller reload. No sequential boundary
  fired; the final exact disjoint 400–499 look launched. Protected seeds remain
  sealed.

- 04:55 EDT (Jun 26) — Cycle-6 promotion completed the registered maximum of
  500 pairs in every tier (4,000 physical games). All tiers reached
  `inconclusive-maximum`: direct `-0.024`, R64 `+0.044`, R600 `+0.040`, and
  equal-wall `-0.042`. The candidate was not promoted; the bootstrap incumbent
  remains champion with manifest `77ed15e2...` and weights `5eb9fcec...`.
  Champion evidence SHA256 is `f083d3a6...`.

- 04:55 EDT (Jun 26) — Campaign transition 33 atomically opened Cycle 7
  collection using the frozen Cycle-6 champion and the registered prior-model
  pool. Protected seeds remain sealed.

- 06:26 EDT (Jun 26) — Cycle-7 collection exposed a worker-memory scaling
  defect in the legacy opponent-pool transport: every one-GiB shard eagerly
  loaded the newest model plus all six prior V3 models. Ninety-six of 100
  shards succeeded, three exhausted retries with an explicit `memory limit
  exceeded`, and the final legacy shard remains live. A fail-closed repair
  path now detects only invalid/missing shards, reconstructs their original
  full-pool payload exactly at two GiB, hard-links original successes and
  repaired outputs into a lineage-recorded 100-shard corpus, and revalidates
  the exact 10K-game / 24K-V1-seat / 6K-prior-seat totals. Future cycles use a
  permanent bounded-memory layout: each shard mounts the newest policy plus
  one prior policy, rotated deterministically and evenly across the full pool.
  Receipt validation now rejects any policy identity outside the shard-local
  declared domain. Ten focused collection/repair/corpus tests, py_compile, and
  Ruff pass. No successful shard was reopened and protected seeds remain
  sealed.

- 06:37 EDT (Jun 26) — The four exact Cycle-7 repair shards completed at two
  GiB in 470.48 seconds with no failure. Reconciliation independently
  revalidated and lineage-bound all 100 item directories: 10,000 games,
  40,000 seat-games, 10,000 newest-model seats, 24,000 V1 seats, and 6,000
  prior-V3 seats across 14,099,221 packed bytes. Items 30, 38, 40, and 96 are
  explicitly sourced from `v3-cycle-07-collect-memory-repair-v1`; the other 96
  retain their original request identity. The repair implementation now also
  auto-detects and exactly reconstructs either the legacy all-prior layout or
  the new shard-local layout, so future bounded-memory requests remain
  recoverable. Twenty-eight focused scheduler/collection/repair/corpus tests,
  py_compile, and Ruff pass. Protected seeds remain sealed.

- 06:41 EDT (Jun 26) — Cycle-7 replay verification passed all 100 reconciled
  shards in 56.62 seconds: 10,000 records expanded to exactly 200,000 focal
  training entries with 14,099,221 source bytes. Corpus evidence advanced the
  campaign atomically to `cycle-07-labeling` (transition 34). Balanced
  bottom-hash selection considered all 200,000 positions, retained 6,342
  oversampled candidates across 2,204 strata, and froze exactly 2,500 teacher
  roots (`d866da55...`). All 25 K32/R600 label shards are now running on the
  scheduler. Protected seeds remain sealed.

- 06:48 EDT (Jun 26) — Cycle-7 teacher search completed all 25 shards in
  461.45 seconds with no retry or failure: 2,500 roots, 71,163 exhaustive
  candidate estimates, 1,500,000 terminal rollouts, and 6,121,624 output bytes.
  Post-stage local and remote worker-storage reclamation is active before the
  immutable label corpus opens training. Protected seeds remain sealed.

- 06:49 EDT (Jun 26) — Cycle-7 label evidence and storage-reclaim receipts
  passed, advancing campaign state atomically to `cycle-07-training`
  (transition 35). Origin-1 has started on John1 from the unchanged bootstrap
  incumbent with seed 90071 and the registered 50/30/20 replay mixture. The
  two-item parent-checkpoint benchmark is assigned to the CPU workers while
  John1 performs native MLX work. Protected seeds remain sealed.

- 07:00 EDT (Jun 26) — Cycle-7 origin-1 completed its deterministic 400K
  source census and checksum-bound run manifest `173689cc...`. The manifest
  confirms seed 90071, 8,192-example batches, three 400K-exposure passes at
  `3e-5/3e-5/1e-5`, and exact per-pass quotas of 120K current broad, 80K
  current teacher, 120K recent, 40K older broad, and 40K older teacher. The
  lagged parent benchmark also completed 1,000 games in 636.68 seconds at mean
  84.082. Origin-1 pass 1 is active; protected seeds remain sealed.

- 07:31 EDT (Jun 26) — Cycle-7 origin-1 completed pass 1 at exactly 400,000
  exposures and step 50. Latest loss is `0.000675579` at LR `3e-5`.
  Independent BLAKE3 verification matched the atomic model (`3f7cf5b2...`),
  optimizer (`f9b7ad60...`), and state (`5169198e...`) payloads bound to run
  manifest `d7ec5313...`. Pass 2 preparation is active; protected seeds remain
  sealed.

- 07:46 EDT (Jun 26) — Cycle-7 origin-1 completed pass 2 at exactly 800,000
  exposures and step 100. Latest loss is `0.000490043` at LR `3e-5`.
  Independent BLAKE3 verification matched the atomic model (`1bf227df...`),
  optimizer (`73ebd6b9...`), and state (`df21c93a...`) payloads. Pass 3 is
  active at LR `1e-5`; protected seeds remain sealed.

- 08:12 EDT (Jun 26) — Cycle-7 origin-1 completed all three passes at exactly
  1,200,000 exposures and step 150 in 4,293.12 training seconds. Final loss is
  `0.000476280` at LR `1e-5`. Independent BLAKE3 verification matched the
  final model (`bd33df8d...`), optimizer (`9acca9c3...`), and state
  (`b25b77df...`) payloads bound to run manifest `d7ec5313...`.
  Qualification/export is active; protected seeds remain sealed.

- 08:21 EDT (Jun 26) — Cycle-7 origin-1 passed quantized qualification over
  554,624 validation examples at 2,302.45/s: power loss `0.002303628`, RMSE
  `6.5397`, MAE `4.8750`, and serving weights BLAKE3 `7f0b7d69...`. Its
  32-game open smoke completed without overflow or swap growth. Origin-2 has
  started independently from the same parent with seed 90072; protected seeds
  remain sealed.

- 09:10 EDT (Jun 26) — Cycle-7 origin-2 completed pass 1 at exactly 400,000
  exposures and step 50. Latest loss is `0.000728944` at LR `3e-5`.
  Independent BLAKE3 verification matched the atomic model (`ce14ff3e...`),
  optimizer (`95661558...`), and state (`bc5e293d...`) payloads bound to run
  manifest `fa9b359c...`. Pass 2 is active; protected seeds remain sealed.

- 09:37 EDT (Jun 26) — Cycle-7 origin-2 completed pass 2 at exactly 800,000
  exposures and step 100. Latest loss is `0.000488101` at LR `3e-5`.
  Independent BLAKE3 verification matched the atomic model (`7eed85c2...`),
  optimizer (`4e6a8e96...`), and state (`3b6068dd...`) payloads. Pass 3 is
  active at LR `1e-5`; protected seeds remain sealed.

- 09:57 EDT (Jun 26) — Cycle-7 origin-2 completed all three passes at exactly
  1,200,000 exposures and step 150 in 4,960.02 training seconds. Final loss is
  `0.000440775` at LR `1e-5`. Independent BLAKE3 verification matched the
  final model (`b8d233b1...`), optimizer (`e83cab85...`), and state
  (`16b4897d...`) payloads bound to run manifest `fa9b359c...`.
  Qualification/export is active; protected seeds remain sealed.

- 10:04 EDT (Jun 26) — Cycle-7 origin-2 passed quantized qualification over
  554,624 examples: power loss `0.002412432`, RMSE `6.6536`, MAE `4.9662`, and
  serving weights BLAKE3 `1db1b1de...`. Both origins passed the registered
  open nonregression gate; origin-1 was selected by the frozen minimum
  quantized-validation-loss rule (`0.002303628` versus `0.002412432`). The
  selected export is undergoing the complete cross-backend parity gate;
  protected seeds remain sealed.

- 10:06 EDT (Jun 26) — The selected Cycle-7 origin-1 passed the complete
  6,400-row gate with scalar/NEON/MLX bit identity, exact overflow support,
  and 100% float/quantized top-1 and top-32 agreement. Campaign state advanced
  atomically to `cycle-07-promotion` (transition 36), and the registered first
  100-pair look launched. Separately, the now-closed Cycle-7 training traces
  showed the one-thread 120K recent-replay producer alone consuming 6–12
  minutes at pass tails after the 40K older-teacher producer had exited. Future
  origins retain the fixed nine-CPU preprocessing budget but move one thread
  from older teacher to recent replay (`2/2/2/2/1`); Rayon indexed expansion
  preserves identical rows, order, batches, and optimizer steps. Thirty-nine
  focused training/scheduler/collection tests, py_compile, and Ruff pass.
  Protected seeds remain sealed.

- 10:57 EDT (Jun 26) — Cycle-7 promotion completed its first registered
  100-pair look with all 100 work items successful, 400 tier-pairs, and 800
  physical games in 3,044.11 seconds. Integrity and resource gates passed.
  Paired mean deltas are direct `+0.23`, equal-wall-time `+0.02`, K32/R64
  `-0.50`, and K32/R600 `-0.51`; every always-valid boundary is `continue`.
  John1 reclaimed 13.89 GB of completed worker storage while preserving the
  registry and MinIO services, and the controller is completing the same safe
  reclaim on John2/John3 before the second 100-pair increment. Protected seeds
  remain sealed.

- 11:44 EDT (Jun 26) — Cycle-7 promotion completed the second registered
  increment with all 100 additional work items successful in 2,682.09
  seconds. Across 200 pairs per tier, integrity and resource gates still pass;
  paired mean deltas are direct `+0.345`, equal-wall-time `-0.060`, K32/R64
  `-0.085`, and K32/R600 `-0.160`. Every always-valid boundary remains
  `continue`. The controller completed local and remote reclaim and launched
  the third increment (`200–299`). Protected seeds remain sealed.

- 12:39 EDT (Jun 26) — The third Cycle-7 promotion increment completed with
  all 100 work items successful in 3,067.49 seconds. Across 300 pairs per tier,
  paired mean deltas are direct `+0.30`, equal-wall-time `-0.10`, K32/R64
  `+0.13`, and K32/R600 `+0.16`. Integrity and resource gates pass, but no
  always-valid boundary has crossed, so the controller reclaimed all three
  workers and launched the fourth registered increment (`300–399`). Protected
  seeds remain sealed.

- 13:24 EDT (Jun 26) — The fourth Cycle-7 promotion increment completed with
  all 100 work items successful in 2,663.11 seconds. Across 400 pairs per tier,
  direct crossed the registered alternative boundary at `+0.4625`; the other
  paired means are equal-wall-time `+0.0725`, K32/R64 `+0.0600`, and
  K32/R600 `+0.1975`, all still `continue`. The overall verdict therefore
  remains `continue`, and the controller reclaimed all workers and launched
  the final permitted increment (`400–499`). Protected seeds remain sealed.

- 14:10 EDT (Jun 26) — Cycle-7 promotion closed at the registered 500-pair
  maximum. Direct crossed the alternative boundary at `+0.418`, while
  equal-wall-time `+0.090`, K32/R64 `+0.106`, and K32/R600 `+0.060` were
  `inconclusive-maximum`; integrity and resource gates passed. The frozen rule
  therefore retained the incumbent. Campaign state advanced atomically to
  `cycle-08-collecting` (transition 37). The new bounded collection layout was
  verified directly in all 100 submitted jobs: each mounts exactly the newest
  model plus one deterministically rotated prior, declares exactly one prior
  model identity, and requests 1 GiB rather than mounting the full prior pool.
  The first 39 jobs are running and 61 are scheduler-queued; protected seeds
  remain sealed.

- 15:10 EDT (Jun 26) — Cycle-8 collection completed cleanly: 100/100 bounded
  1-GiB shards succeeded with zero retries or scheduler errors, yielding
  exactly 10,000 games and 200,000 focal entries in 2,921.15 seconds. All 100
  shards passed replay verification in 58.04 seconds. Root selection considered
  200,000 positions, oversampled 6,351 across 2,214 strata, and selected exactly
  2,500 teacher roots (`8c9a7208...`). All 25 K32/R600 label shards then
  succeeded in 492.56 seconds, producing 70,920 candidate estimates and exactly
  1.5 million rollouts. Campaign state advanced through `cycle-08-labeling` to
  `cycle-08-training` at transition 39. Origin-1 is active on John1 with seed
  90081 and the corrected `2/2/2/2/1` nine-thread preprocessing allocation;
  John2/John3 are running the lagged parent benchmark. Protected seeds remain
  sealed.

- 15:41 EDT (Jun 26) — The first live Cycle-8 pass trace falsified the initial
  `2/2/2/2/1` preprocessing rebalance: recent replay no longer tailed, but the
  expensive 40K older-teacher stream became the sole producer after all four
  peers exited. Future origins now use the measured `1/2/2/2/2` allocation,
  taking the thread from the cheapest current-broad stream and retaining two
  for both teacher streams, recent replay, and older broad. Indexed expansion
  preserves byte-identical row order and optimizer steps. The active origin-1
  retains its already-loaded frozen allocation; 22 focused tests, py_compile,
  and Ruff pass. Protected seeds remain sealed.

- 16:00 EDT (Jun 26) — Cycle-8 origin-1 completed pass 1 at exactly 400,000
  exposures and step 50 with loss `0.000705057`. The atomic checkpoint is
  independently verified against its manifest: model `47019cca...`, optimizer
  `c3ca750f...`, and exact continuation state `69873e31...`, bound to run
  manifest `71dc0b10...`. Pass 2 is active at `3e-5`; protected seeds remain
  sealed.

- 16:30 EDT (Jun 26) — Cycle-8 origin-1 completed pass 2 at exactly 800,000
  exposures and step 100 with loss `0.000498636`. Independent BLAKE3 checks
  match the atomic manifest for model `7619f6a3...`, optimizer `3c6180fc...`,
  and continuation state `9ce317e0...`. Pass 3 is active at `1e-5`; protected
  seeds remain sealed.

- 17:00 EDT (Jun 26) — Cycle-8 origin-1 completed all three passes at exactly
  1.2 million exposures and step 150 in 5,924.39 training seconds. Final loss
  is `0.000472562` at LR `1e-5`. Independent BLAKE3 verification matches the
  final model `34c1b878...`, optimizer `6142675b...`, and continuation state
  `ff5c383e...`, bound to run manifest `71dc0b10...`. Quantized qualification
  and the 32-game open smoke are active; protected seeds remain sealed.

- 17:09 EDT (Jun 26) — Cycle-8 origin-1 passed quantized qualification over
  554,624 validation examples at 2,311.25/s: power loss `0.002609624`, RMSE
  `6.8906`, MAE `5.1612`, and serving weights BLAKE3 `1a267ab4...`. Its
  32-game open smoke passed. Origin-2 started independently from the same
  incumbent with seed 90082 and is the first run using the measured
  `1/2/2/2/2` preprocessing allocation. Protected seeds remain sealed.

- 17:37 EDT (Jun 26) — Cycle-8 origin-2 completed pass 1 at exactly 400,000
  exposures and step 50 with loss `0.000610239`. The corrected `1/2/2/2/2`
  allocation reached the pass boundary in about 29 minutes versus about 51
  minutes for origin-1's flawed allocation, a 43% wall-time reduction without
  changing rows or optimizer steps. Independent BLAKE3 checks match model
  `23d5dc3f...`, optimizer `73fe85d7...`, and continuation state
  `b3646999...`, bound to run manifest `b52fa1a3...`. Pass 2 is active at
  `3e-5`; protected seeds remain sealed.

- 17:58 EDT (Jun 26) — Cycle-8 origin-2 completed pass 2 at exactly 800,000
  exposures and step 100 with loss `0.000420593`. Independent BLAKE3 checks
  match model `86d926e4...`, optimizer `b83b72dc...`, and continuation state
  `6bc9981f...`, bound to run manifest `b52fa1a3...`. Pass 3 is active at
  `1e-5`; protected seeds remain sealed.

- 18:18 EDT (Jun 26) — Cycle-8 origin-2 completed all three passes at exactly
  1.2 million exposures and step 150 in 3,481.46 training seconds, 41.2% faster
  than origin-1 under the prior allocation. Final loss is `0.000398580` at
  `1e-5`. Independent BLAKE3 verification matches model `d57d3665...`,
  optimizer `00ce4b1f...`, and continuation state `b1b84b21...`, bound to run
  manifest `b52fa1a3...`. Qualification is active; protected seeds remain
  sealed.

- 18:26 EDT (Jun 26) — Cycle-8 origin-2 passed quantized qualification with
  power loss `0.002406002`, RMSE `6.6592`, MAE `4.9836`, and weights
  `0aaea8c4...`, but its paired open-domain lower bound `-1.2242` missed the
  registered `-1.0` nonregression margin and made it ineligible. The frozen
  selection rule therefore chose origin-1. Selected origin-1 then passed all
  6,400 parity rows with scalar/NEON/MLX bit identity, exact overflow support,
  and 100% float/quantized top-1 and top-32 agreement. Campaign state advanced
  atomically to `cycle-08-promotion` at transition 40 and launched the first
  100-pair look. Protected seeds remain sealed.

- 19:17 EDT (Jun 26) — Cycle-8 promotion completed its first 100-pair look
  with all 100 work items successful in 2,865.92 seconds. Integrity and
  resource gates pass; paired mean deltas are direct `-0.09`, equal-wall-time
  `-0.08`, K32/R64 `+0.08`, and K32/R600 `-0.01`, with every boundary
  `continue`. The controller completed safe reclaim and launched the second
  increment (`100–199`). Protected seeds remain sealed.

- 20:02 EDT (Jun 26) — Cycle-8 promotion completed the second 100-pair
  increment with all work items successful in 2,780.43 seconds. Across 200
  pairs per tier, paired mean deltas are direct `+0.215`, equal-wall-time
  `-0.055`, K32/R64 `+0.250`, and K32/R600 `-0.140`; integrity and resource
  gates pass and every boundary remains `continue`. Reclaim and the third
  increment are proceeding. Protected seeds remain sealed.

- 20:49 EDT (Jun 26) — The third Cycle-8 promotion increment completed with
  all 100 work items successful in 2,619.50 seconds. Across 300 pairs per tier,
  means are direct `+0.2833`, equal-wall-time `-0.0867`, K32/R64 `+0.1733`,
  and K32/R600 `-0.2033`. Integrity and resource gates pass; every boundary
  remains `continue`. The controller reclaimed all workers and launched pairs
  `300–399`. Protected seeds remain sealed.

- 21:39 EDT (Jun 26) — The fourth Cycle-8 promotion increment completed with
  all work items successful in 2,945.83 seconds. Across 400 pairs per tier,
  means are direct `+0.2650`, equal-wall-time `-0.0925`, K32/R64 `-0.0100`,
  and K32/R600 `+0.0350`. All integrity/resource gates pass, but every boundary
  remains `continue`; the final registered increment (`400–499`) is running.
  Protected seeds remain sealed.

- 22:29 EDT (Jun 26) — Cycle-8 promotion closed at the 500-pair maximum with
  all integrity and resource gates passing. Final paired means were direct
  `+0.252`, equal-wall-time `-0.050`, K32/R64 `+0.018`, and K32/R600
  `+0.036`; all four boundaries were `inconclusive-maximum`, so the incumbent
  was retained. Campaign state advanced atomically to `cycle-09-collecting`
  at transition 41. All 100 Cycle-9 jobs were verified to use the bounded
  newest-plus-one-prior layout at 1 GiB; 39 are running and 61 queued.
  Protected seeds remain sealed.

- 23:26 EDT (Jun 26) — Cycle-9 collection completed 100/100 shards cleanly in
  2,869.53 seconds, yielding exactly 10,000 games and 200,000 focal entries;
  all shards passed replay verification in 57.93 seconds. Root selection
  considered 200,000 positions, oversampled 6,481 across 2,229 strata, and
  selected exactly 2,500 roots (`b17765af...`). All 25 K32/R600 label shards
  succeeded in 492.78 seconds, producing 70,573 estimates and exactly 1.5
  million rollouts. Campaign state advanced through labeling to
  `cycle-09-training` at transition 43. Origin-1 is active with seed 90091 and
  the measured `1/2/2/2/2` allocation; protected seeds remain sealed.

## Invariants

- John1 remains code/image/artifact authority.
- John2 and John3 execute the exact canonical image; John4 is excluded from compute.
- No protected evaluation opens before the final champion is frozen.
- No Phase 2 work may use engineering-smoke records.
- All material transitions and blockers are appended here.
