# V3 Part 1 Performance

All measurements are from the permanently excluded engineering domain on John1. They are capacity evidence, not strength evidence.

## Representation

- Radius-7 natural fast path: **159,987 / 160,000 states (99.991875%)**.
- Exact overflow states observed naturally: **13**; deliberately injected profiles used 48 overflow entities across four boards and retained exact inference.
- FullOpportunities: **83,284** collision-free serving rows and **1,624** train-only factor rows.
- Active opportunity rows: **99.21 focal**, **313.72 three-opponent field**, or **103.23 per board perspective**. This lands inside the planned 100–160 range per accumulator perspective.

## MLX pass 1 and pass 2

| Measurement | Correct baseline | Optimized |
|---|---:|---:|
| Examples/second | 58.95 | 3,174.66 |
| End-to-end speedup | — | **53.85×** |
| Fastest measured batch | — | 8,192 at 3,404.40 examples/s |
| Peak memory | — | 4.57 GB / 17.18 GB (26.6%) |
| Swap growth | — | 0 bytes |

Pass 1 exposed padded sparse tensors, materialized gather pressure, and synchronization gaps. Pass 2 uses native packed records, a double-buffered Rust producer, exact-width CSR batches, fused Metal embedding bags, deterministic feature-sorted sparse gradients, and bounded cache release between batch shapes. Backward/gradient synchronization and native decode are now the dominant costs; both are required by exact-resume and bit-parity semantics.

## Gameplay pass 1 and pass 2

| Measurement | Correct reference | Optimized |
|---|---:|---:|
| Initial-state decisions/second | 3.66 | 39.58 |
| End-to-end speedup | — | **10.82×** |
| Representative late-game decisions/second | — | 6.73 |
| Injected-overflow decisions/second | — | 4.59 |
| Direct all-V3 game, one thread | 49.361 s | 7.757 s (464.1 games/hour) |
| End-to-end direct-game speedup | — | **6.36×** |
| Expert-shaped game, one V3 + three V1, one thread | — | 2.363 s (1,523 games/hour) |
| Expert-shaped game, one V3 + three V1, ten threads | — | 0.535 s (6,724 games/hour) |
| Focal K32/R600 game | — | 15.433 s |

The optimized ranker is bit-identical to the independent reconstruct-every-afterstate reference at initial, middle, late, Nature Token, and overflow-capable feature paths. It caches public/draft context and matching frontiers, traverses boards in place, applies sparse accumulator deltas, uses NEON int8/int16 feature, product-pooling, and dense kernels, and performs allocation-free action evaluation. Serving inference no longer constructs diagnostic traces, and candidate evaluation no longer clones the four-board public state.

On the final full-game single-thread profile, candidate opportunity construction consumes **26.3%** of wall time, tile-level own-accumulator preparation **25.4%**, habitat preparation **15.6%**, and matching-frontier preparation **13.5%**. Candidate accumulator plus dense inference is now **7.5%**; public context, exact scoring, action enumeration, apply/undo, and sorting are individually below 5%. The remaining costs are dominated by exact opportunity semantics and exhaustive legal-action ranking, not dense NNUE inference.

## Correctness and campaign capacity

- Float-QAT, MLX integer, NumPy integer, Rust scalar, and Rust NEON: **bit-identical over 6,400 candidates**.
- Float/quantized top-1 and top-32 agreement: **100%**.
- Interrupted step-10 training and uninterrupted training produced identical model and optimizer tensor states at step 20.
- R600 target: **15.43 s/game**, below the 45-second gate.
- Projected Part 2 storage under the revised contract: **9.08 GiB**, below the 40-GiB ceiling.
- Projected Part 2 active wall time: **4.30 days** including 20% recovery margin, maximum promotion traffic, final evaluation, and reporting. The expert-cycle contract is 10,000 games per cycle with 80% qualified-V1 opponent seats and 2,500 selective teacher roots; the old 41.94-day projection based on one million expert games is retired.

The five-game engineering worker smoke measured 11.815 seconds on one thread and 2.677 seconds with ten Rayon threads. At the measured ten-thread rate, 10,000 games represent about 1.49 worker-hours before cross-node scheduling; the scientific scheduler remains responsible for placement and disjoint game-index ranges. These are open-domain capacity measurements, not Phase 2 authorization or strength evidence.

## Exact-wide accumulator qualification

The first long bootstrap origin exposed that Cascadia's pre-activation opportunity sums can exceed int16 even though the network immediately clips them at the activation boundary. The serving accumulator now stores exact int32 sums and narrows only after clipping. The previously failing 9,000,000-exposure bundle completes the fixed two-game serving gate under the corrected runtime. On the prior serving-safe 5,007,904-exposure bundle, the old and corrected runtimes produce identical scores for all 16 seats in four fixed games.

The corrected runtime retains the optimized capacity envelope: one representative single-thread all-V3 game completed in 7.995 seconds, and the exact expert-worker five-game shard completed in 2.739 seconds with ten threads. The latter reproduced the pre-migration shard checksum and all policy-seat identities exactly, with zero swap growth. The historical 6.36x direct-game speedup therefore remains qualified after the accumulator correction.
