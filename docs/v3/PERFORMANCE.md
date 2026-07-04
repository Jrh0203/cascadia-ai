# V3 Performance Notes

These are engineering measurements and diagnostics, not promotion evidence.

## Tensor Export

Rust-native greedy tensor export on `john0`:

| Format | 1,024 games wall time | Size |
|---|---:|---:|
| Deflated `.npz` | 1:35.36 | 71,413,962 bytes |
| Stored `.npz` | 18.28 s | 1,248,396,657 bytes |

Stored shards were `5.22x` faster and `17.48x` larger in that benchmark. Use
stored shards when GPU input throughput matters and disk headroom exists. Use
deflated shards for archival or transport.

## Prefilter Findings

The p80x2 vanilla public-token transformer prefilter family produced a useful
serving signal but did not justify K16 promotion:

- dedicated fixed 3-seed vanilla public-token ensemble:
  - K16 recall `0.7672`;
  - oracle regret `0.1146`.
- 4 paired complete-game pilot:
  - K16 prefilter search `96.0625`;
  - full-32 sampled search `95.0625`;
  - shadow full-search winner retained on `77.8125%` of decisions.
- 20-seed non-shadow follow-up:
  - K16 prefilter search `95.4625`;
  - full-32 sampled search `96.3500`;
  - mean decision time `2.3558s` vs `4.4617s`;
  - `1.89x` speedup and `47.2%` time reduction.

Interpretation: the bridge and prefilter path work, but K16 is too lossy on
current evidence. Prefer K24 or a stronger retention/search-aware model.

## CascadiaFormer Baseline

The first relation-tail CascadiaFormer-S run completed, but no-search play was
below greedy:

| Menu | Policy | Q | Greedy |
|---|---:|---:|---:|
| K256 | 72.5125 | 57.4750 | 87.3375 |
| K32 | 82.7625 | 77.9250 | 87.3375 |

The corrected greedy-state K32 retention run was much closer:

- locked validation greedy top-1: `0.6780`;
- 100-game model mean: `86.7800`;
- greedy mean: `87.5875`;
- paired delta: `-0.8075`;
- exact greedy-action match: `67.3625%`.

Interpretation: CascadiaFormer can operate near the greedy policy surface, but
it had not yet surpassed greedy before EI-0.

## EI-0 Greedy Search Bootstrap

EI-0 is the first CascadiaFormer run with positive no-search gameplay evidence
against greedy.

Training run:

- source runbook:
  `docs/v3/EI0_GREEDY_SEARCH_BOOTSTRAP_RUNBOOK.md`;
- expert tensor mode: `greedy_search_bootstrap`;
- objective: `search-improved-greedy-retention`;
- filter: strict greedy-prefix K32;
- corpus: 20,000 train roots and 4,000 validation roots;
- search labels: 4 rollouts/action, rollout top-k 4;
- model: CascadiaFormer-S;
- training: 25,000 steps, batch size 192, LR 1e-4;
- selected checkpoint:
  `cascadiav3/checkpoints/full_v3_ei0_greedy_search_bootstrap/guarded_retention_safe_best.manifest.json`.

Measured throughput:

- generation: 1,569 s total;
- train generation: 1,282 s;
- validation generation: 287 s;
- training: 2,457 s;
- roots/s: `15.2964`;
- rollout evals/s: `1,957.9350`;
- train step seconds: `0.09828`.

Training health:

- train/validation tensor invariant status: `pass`;
- strict K32 selected-action drops: `0`;
- max absolute Q invariant error: `0.0`;
- guarded checkpoint step: `7,250`;
- guarded locked validation total: `5.8410`;
- guarded locked validation greedy top-1: `0.69375`;
- guarded locked validation mean greedy rank: `1.8860`;
- guarded locked validation teacher top-1: `0.13025`;
- guarded locked validation teacher advantage over greedy: `2.1672`.

100-game no-search complete-game benchmark:

| Strategy | Mean | P90 | Delta vs Greedy | Greedy Match |
|---|---:|---:|---:|---:|
| Greedy | 87.5575 | 92.0000 | - | - |
| CascadiaFormer policy | 87.7925 | 92.0000 | +0.2350 | 70.1125% |
| CascadiaFormer q | 89.6175 | 94.0000 | +2.0600 | 29.8125% |

20-game search-integrated complete-game benchmark:

| Strategy | Mean | P90 | Delta vs Control | Mean Decision Seconds |
|---|---:|---:|---:|---:|
| CascadiaFormer-search K32 of K64 | 95.8000 | 99.0000 | -1.1750 | 8.8258 |
| Full-search K64 control | 96.9750 | 100.0000 | - | 8.7871 |

Search-gate diagnostics:

- selection head: q;
- retain K32 out of max K64;
- rollouts/action: 16;
- rollout top-k: 4;
- shadow full search: true;
- treatment/control time ratio: `1.0044`, passing the `<=1.20` gate;
- full-search winner retained rate: `79.1875%`;
- mean shadow search regret: `0.0916`;
- p95 shadow search regret: `0.5656`;
- zero-regret decision rate: `80.25%`;
- model score time/root: `0.0170s`;
- search time/root: `8.8088s`.

EI-0 decision-trace forensics:

| Retained K | Full-Search Winner Recall | Non-Shadow Rollout Fraction |
|---:|---:|---:|
| 32 | 79.1875% | 50.0% |
| 40 | 86.6250% | 62.5% |
| 48 | 91.6875% | 75.0% |
| 56 | 96.8750% | 87.5% |
| 64 | 100.0000% | 100.0% |

20-game K56 non-shadow follow-up on the same seed set:

| Strategy | Mean | P50 | P90 | Delta vs Control | Mean Decision Seconds |
|---|---:|---:|---:|---:|---:|
| CascadiaFormer-search K56 of K64 | 96.4125 | 97.0000 | 100.0000 | -0.5625 | 7.8031 |
| Full-search K64 control | 96.9750 | 97.0000 | 100.0000 | - | 8.8327 |

K56 passed the treatment/control timing gate with ratio `0.8834`, saved the
expected `12.5%` of non-shadow rollouts, and narrowed the K32 score gap by
about half. The result does not prove a 100-point agent: over these 20 games,
neither K56 nor full K64 produced a per-game mean score at or above `100`.

20-game K64 rollout-depth ceiling follow-up on the same seed set:

| Strategy | Rollouts/action | Mean | P50 | P90 | Delta vs K64/R16 | Mean Decision Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Full K64 sampled search | 16 | 96.9750 | 97.0000 | 100.0000 | - | 8.8327 |
| Full K64 sampled search | 32 | 96.8375 | 97.0000 | 100.0000 | -0.1375 | 18.4531 |

Doubling samples per action did not improve this 20-seed mean and roughly
doubled per-decision search time. K64/R32 produced no per-game mean at or above
`100`, despite 12 of 80 individual seats scoring at least `100`.

## EI-1 Model-State Expert Iteration

EI-1 was the first model-state expert-iteration run: EI-0 q chose the behavior
actions, sampled search labeled K32 action menus, and the trainer mixed `70%`
new model-state roots with `30%` EI-0 greedy-state bootstrap roots.

Training:

- expert tensor mode: `model_state_search_bootstrap`;
- train/validation roots: `20,000` / `4,000`;
- rollouts/action during labeling: `4`;
- objective: `expert`;
- selection metric: minimum `locked_val_final_q_regret`;
- selected checkpoint: `best_locked_val` at step `15,000`;
- best locked validation final-Q regret: `1.909125`;
- final step locked validation final-Q regret: `2.110875`;
- generation throughput: `10.5309` roots/s;
- training throughput: `0.0978` seconds/step.

100-game no-search complete-game benchmark:

| Strategy | Mean | P50 | P90 | Delta vs Greedy | Greedy Match |
|---|---:|---:|---:|---:|---:|
| Greedy | 87.5450 | 88.0000 | 92.0000 | - | - |
| CascadiaFormer q | 90.7600 | 91.0000 | 94.0000 | +3.2150 | 30.6375% |

This is a real no-search improvement: EI-1 q beat both matched greedy and EI-0
q's prior `89.6175` 100-game mean.

K56 search-integrated follow-up:

| Strategy | Mean | P50 | P90 | Notes |
|---|---:|---:|---:|---|
| CascadiaFormer-search K56 of K64 | 96.4250 | 97.0000 | 100.0000 | recovered from 20 complete candidate games |
| Full-search K64 control | 96.7656 | 97.0000 | 99.0000 | recovered from 16 complete control games |

The K56 run exited before producing its normal final JSON report, so this is
forensic evidence recovered from the decision stream rather than a promotion
gate. It is still directionally clear: EI-1 search remained in the existing
`96-97` band, with a recovered paired delta of `-0.453125` on the 16 completed
control pairs. The harness now journals completed games to `*_games.jsonl` so a
future long search tail cannot erase completed-game score evidence.

Interpretation: the q serving head is now the useful no-search policy, and it
beats greedy by about two points on the first 100-game EI-0 gate. Search
integration reaches a strong absolute mean above `95` and passes the timing
gate. The K56 retained set is a better serving point than K32 on current
evidence, but the current one-ply sampled-search setting itself still sits
below the 100-point target. The benchmark is CPU rollout-bound rather than
GPU-bound: model inference is tiny compared with terminal rollout search.
However, K64/R32 shows that simply spending more samples per action is not
enough. EI-1 shows that model-state iteration can improve no-search play, but
not yet search-integrated score. The next step should improve the
policy/value/rollout target, not just increase rollout count.

## Leak Caveat (2026-07-02)

All search-integrated numbers above were measured with rollouts that observed
the true hidden tile/bag order. They remain internally comparable with each
other but are **legacy-leaky**: honest baselines re-run with
`--rollout-determinize`, and the Gumbel search path is public-information
legal by construction. See GUMBEL_SELFPLAY_CAMPAIGN.md Phase A.

## Gumbel Stack Engineering Facts (local, 2026-07-02)

Measured on the local dev machine (debug/mock-bridge unless noted):

- Afterstate reuse removes one full `GameState` clone + action re-apply per
  rollout; the golden-equality test pins the refactor to bit-identical
  labels with determinization off.
- Tiny self-play export smoke (release build, mock bridge, 8 sims, K-interior
  4, 2 seeds x 6 plies): `2.8 seeds/s`, `17.0 records/s` single-session.
- The batched bridge collates up to 32 roots per forward; batch-of-N equals
  N single evals to 1e-4 (unit-tested). Relation matrices build in numpy
  instead of pure-Python lists.

## Self-Play Generation Throughput (john0, 2026-07-02)

Measured aggregate self-play generation (n=64, k-interior 16, root menu
256, full games, tensor export included):

| Topology | Games/h | Note |
|---|---:|---|
| 6 owned bridge sessions | ~40 | stable reference config |
| 12 owned sessions | ~stall | CUDA context thrash (near-zero progress) |
| 20 games on 1 shared aggregated bridge | ~15-20 | REGRESSION: one Python collate pipeline replaces six |

Benchmark (non-export) games run ~1.07s/decision; self-play decisions run
~4.3s at 6 sessions. The dominant serial cost is the bridge's pure-Python
per-row feature extraction (public-token + semantic-action features), not
the GPU: merged cross-game batches did not help because collate is
row-cost-bound and single-threaded per bridge process.

Conclusion: the generation unlock is moving feature extraction to Rust
(the builders already exist in `feature_tensors.rs` for shard export) and
sending precomputed feature arrays in eval requests, so the bridge only
wraps tensors and runs forwards. The shared aggregated bridge is the right
serving architecture once collate is cheap; until then owned 6-session is
the production config.

## Throughput Optimization Pass (2026-07-02 afternoon, merged)

Three landed changes, all bit-parity-gated (golden-label test byte-identical,
198 workspace tests, packed-vs-raw eval equivalence to 1e-4):

1. **Packed-features eval protocol** (`e558315`): Rust computes token/action/
   relation-tail features and base64-packs them into eval requests; the
   bridge decodes with `np.frombuffer`. Bridge collate: **394ms -> 47ms per
   32 full-menu roots (8.4x)**. Requests also shrink (no raw token/action
   JSON).
2. **Engine hot-path pass** (`0e31d0d`): visitor-based legal-action
   enumeration without per-action materialization, cached neighbor/habitat
   context, flat wildlife-score caches (FxHashMap), exact-parity O(n) top-k
   selection. **rank_greedy_actions 2.1-3.6x faster, greedy rollouts
   2.2-2.6x faster** on mid/late-game states (checksummed bit-parity across
   45 bench sections; differential test vs a reference implementation).
   End-to-end tiny selfplay export (mock bridge): 2.83 -> 4.32 seeds/s.
3. **Shared aggregated bridge** (`7241e73`): one CUDA context, cross-chunk
   request merging. Was a regression while collate dominated; with packed
   features it becomes the intended high-parallelism serving path.

Deployed 2026-07-02 ~16:35: cycle 2 restarted on the optimized stack
(`SHARED_MODEL_SESSION=1`, `MODEL_SESSIONS=16`). Measured steady
generation: **278 games/h** (125 seeds / 27 min) versus the 40 games/h
old-stack owned-6 baseline — a **~7x aggregate throughput improvement**.
A full-scale 1,375-seed cycle is now a ~5h job.

## Throughput Optimization Pass 2 (2026-07-03, merged, not yet deployed)

Model evals dominate post-pass-1 (n=128 labels cost ~3.5x n=64). Two landed
changes targeting the eval path, both parity-gated (golden test byte-identical;
packed==JSON and dedup==no-dedup exact-equality tests; byte-identical mock
selfplay corpus sha256):

1. **Eval-row dedup + per-chunk eval cache** (merge of `8b90c5c`): eval
   features are public-information-only, so simulations sharing a root action
   produce identical rows for early interior plies and argmax opponent
   advances repeat states. `evaluate_rows_deduped()` collapses within-batch
   duplicates and serves cross-call repeats from a blake3-keyed per-worker
   cache (50k-entry cap). Measured at production search shape (n=64, top_m 16,
   k_interior 16, menu 256, det 4): **43.7% of model eval rows eliminated**
   (39,840 requested -> 22,438 sent; savings almost entirely from the
   cross-call cache). Cache-hit rows also skip packed-feature construction.
   Counters (`rows_requested/rows_sent/cache_hits`) print in export summaries.
2. **Packed-response protocol + serving-path pass** (merge of `59bc3a7`):
   negotiated `packed_response` feature — bridge returns base64 f64 LE arrays
   instead of per-action JSON float lists (f32->f64 widening is exact: bit-
   parity with JSON path, test-enforced). Python response encode **11.93ms ->
   1.54ms per 32x256 batch (7.7x)**; Rust decode **1.60ms -> 0.55ms (2.9x)**.
   Forward path: `torch.inference_mode`, one `.cpu()` per output tensor per
   chunk, zero-copy collate, pinned non-blocking H2D, `final_q` on host.
   New env knobs: `CASCADIA_BRIDGE_TF32=1` / `CASCADIA_BRIDGE_AUTOCAST=bf16`
   (default off; NOT bit-parity), `CASCADIA_SHARED_GATHER_US` /
   `CASCADIA_SHARED_ROW_CAP` (defaults 2000/192 = old behavior).

Caveat: removing rows changes GPU batch composition, so float-reduction-order
drift in labels is possible on real GPUs (byte-pinning holds under the
deterministic mock). Expected production effect: evals dominate, so ~44% row
reduction plus response savings should compound to roughly ~1.7-2x generation
throughput at n=128; measure on john0 after cycle 3 lands before quoting.

## Forward-Path Pass 3 (2026-07-03, bridge forward knobs, default off)

Findings and knobs from the CascadiaFormer forward-path pass in
`torch_inference_bridge.py`. All knobs default off; with them unset the bridge
behavior is unchanged (legacy chunker parity is unit-test pinned).

1. **Trunk-factoring verdict: already factored.** The forward runs the public
   tokens `[B, S]` through `token_proj` and the whole `state_encoder` stack
   once per root; action conditioning first enters at `action_proj` / the
   cross-attention queries, and per-action relation structure only in the CGAB
   tail. The shared trunk is the dominant FLOP share (roughly 85% at
   S=64/A=256 and ~95% at small menus, CascadiaFormer-S) and is computed
   exactly once — there is no per-action recomputation to remove, so no
   factored path was added.
2. **`CASCADIA_BRIDGE_BUCKET=1` shape bucketing.** Collate pads token/action
   capacities to power-of-two buckets (floor 8, cap 512, then multiples of
   128), bounding the shape vocabulary for kernel caches and torch.compile.
   The chunker costs chunks at the padded shape so the CGAB cell budget still
   holds. Mask integrity is exact: filling padded rows with garbage instead of
   zeros leaves every real output bit-identical (test-enforced). Bucketed vs
   unbucketed outputs are NOT bit-identical (~2e-7 max drift, tolerance-gated
   test): CPU/GPU reduction kernels block over the padded length, so appending
   even exact zeros regroups the floating-point reduction of the real prefix —
   the SDPA MATH backend shows the same, and the default chunk-max padding
   already admits this drift class. CPU cost: 32 mixed-menu rows,
   CascadiaFormer-S: 657ms -> 675ms min (~3% padding overhead); the intended
   win is CUDA kernel-cache stability and a finite compile shape set.
3. **`CASCADIA_BRIDGE_COMPILE=1` torch.compile.** Wraps the loaded model
   (default mode; `mode="reduce-overhead"` worth benchmarking on the CUDA
   box). Pair with bucketing so recompiles are finite. Falls back to eager on
   failure; CUDA-only warmup over representative bucket shapes at load time.
   CPU smoke-tested (tiny config, tolerance-gated vs eager).
4. **`CASCADIA_BRIDGE_TIMING=1` per-phase timing.** Accumulates per-chunk wall
   time for collate/H2D/forward/D2H/encode plus rows and actions, emits a
   one-line stderr summary every 50 chunks and at shutdown, e.g.
   `[bridge-timing final] chunks=1 rows=4 actions=2097 collate=0.052s
   h2d=0.000s forward=0.307s d2h=0.000s encode=0.000s total=0.360s
   rows/s=11.1`. Zero-cost when off. Use this on john0 to direct the next
   pass.

## CascadiaFormer Trainer Perf Knobs (2026-07-04)

Efficiency pass on `cascadiav3.torch_train_cascadiaformer` motivated by the
CascadiaFormer-M run measuring 2.39 s/step at batch 192 on the RTX 5090 (GPU
99% util at only 189 W, trainer pegging one CPU core). Hard contract, pinned
by `cascadiav3/tests/test_trainer_perf_knobs.py`: with no knobs set, training
is bit-identical to the pre-pass trainer (verified 3-step CPU runs on npz
unweighted, npz weighted 2-source, and jsonl fixtures — all 45 state-dict
tensors `torch.equal`, metrics JSONL byte-identical).

Root causes found in the default path (all fixed exactly-safely, no knob
needed):

1. **Per-microbatch O(corpus) Python shuffle.** Unweighted sampling rebuilt
   and Fisher-Yates-shuffled the full epoch order for EVERY micro-batch
   (`_deterministic_order`): 19 ms at 100k records, 266 ms at 1M records of
   pure single-core Python per micro-batch. Now LRU-cached per
   (record_count, seed, epoch) — cache hit ~0.1 us, identical indices.
2. **17 separate GPU syncs per micro-batch.** `float(loss.detach().cpu())`
   per aggregate key after each backward (and per eval batch). Now one
   stacked fp64 transfer (`_loss_scalars`), bit-identical values.
3. **`enable_nested_tensor` warning.** The nested-tensor encoder fast path is
   inference-only (requires eval mode + no grad) and additionally disqualified
   by `norm_first=True`; the encoder is now built with
   `enable_nested_tensor=False` — zero numeric change, warning gone.
4. **`gradient_checkpointing` config flag was a silent no-op.** The M config
   requests it but no trainer ever applied it; the observed M run was NOT
   checkpointing. Now implemented behind `--grad-checkpoint on` (auto/off
   preserve legacy no-checkpoint behavior; CPU test shows on/off bit-identical
   losses and grads).

Opt-in knobs (defaults preserve bit-identity):

| Knob | Semantics | Parity |
|---|---|---|
| `--data-workers N` (+`CASCADIA_TRAIN_DATA_WORKERS`) | DataLoader workers; seeded index lists stay in the main process, workers only fetch+collate; persistent workers, prefetch (`--prefetch-factor`), pin_memory + non-blocking H2D on CUDA | bit-identical batches/order (test-enforced); CPU bench: data-wait 31.4 -> 1.6 ms per b192 batch with 2 workers |
| `--autocast {auto,off,bf16}` | auto = legacy (bf16 autocast on CUDA, fp32 CPU); off forces fp32; locked-val eval always runs fp32 | train metrics under bf16 not comparable to fp32 runs |
| `--tf32` / `CASCADIA_TRAIN_TF32=1` | TF32 matmul+cudnn on CUDA | changes fp32 numerics |
| `--fused-optimizer` | fused AdamW, CUDA only | different update kernel, not bit-identical |
| `--compile` / `CASCADIA_TRAIN_COMPILE=1` | torch.compile (default mode), eager fallback | numeric drift possible |
| `--grad-checkpoint {auto,on,off}` | activation checkpointing on the state encoder | bit-identical on deterministic kernels |
| `CASCADIA_TRAIN_SDPA=flash\|mem_efficient\|math\|cudnn` | restricts SDPA backends around train forward/backward and eval (comma list = priority) | selection only; flash rejects our merged key-padding attn_mask |
| `CASCADIA_TRAIN_SDPA_LOG=1` | logs enabled SDPA backends + per-shape usability verdicts (`can_use_flash/efficient_attention` with debug reasons) on the first CUDA batch | log only |
| `CASCADIA_TRAIN_TIMING=1` (`_EVERY=K`) | per-phase wall times (data/h2d/forward/backward/optimizer/eval), line every K steps + summary + report `phase_timing`; synchronizes CUDA at phase boundaries, so it slightly perturbs throughput | measurement only |
| `--cgab-fused` / `CASCADIA_CGAB_FUSED=1` | fused CGAB relation tail: count-matmul instead of the `[B, A, seq, d_model]` intermediate (see "Fused CGAB Relation Tail" below) | equivalent math, not bit-identical (~2.4e-7 max dev fp32) |

SDPA reality for this encoder: training never uses the fused/nested
TransformerEncoder path (inference-only), so attention always goes through
`F.scaled_dot_product_attention` with a merged key-padding float attn_mask;
flash rejects arbitrary attn_masks, leaving mem_efficient vs math — use
`CASCADIA_TRAIN_SDPA_LOG=1` on john0 to see which one actually runs for the
M shapes, and `CASCADIA_TRAIN_SDPA` to force alternatives.

## Fused CGAB Relation Tail (2026-07-04, local CPU)

The Forward-Path Pass 3 finding stands: the top memory-bound op is the CGAB
relation tail, which materializes `[B, A, seq, d_model]` (embedding lookup +
mask product both live at peak) just to take a masked mean over relation ids.
That mean is exactly `(counts_per_relation_id / valid_positions) @
embedding_table`, so the fused path builds a `[B, A, 32]` count matrix
(scatter_add, id-0 column zeroed to honor the `padding_idx=0` + `ne(0)`
contract) and one dense matmul instead. Fully expressible — there is no
position-dependent weighting or attention in the tail; gradients are
equivalent too (padding row still gets zero grad).

Knobs (defaults preserve the materialized path bit for bit):

- `CASCADIA_CGAB_FUSED=1` — read at model construction (trainer, bridge, any
  consumer); also `--cgab-fused` on the trainer and
  `model.set_cgab_fused(True)`. NOT bit-identical: floating-point
  reassociation, measured max deviation `2.4e-7` abs across all heads
  (fp32, tiny/S), grads agree to `rtol 1e-5 / atol 1e-6`.
- `CASCADIA_EVAL_CELL_BUDGET=N` — overrides the bridge's rows×actions×seq
  chunking budget (default `EVAL_BATCH_CELL_BUDGET = 2^21`, unchanged). The
  default budget is sized for the materialized tail; with the fused tail it
  over-estimates CGAB cost by ~`d_model`×, so serving can raise it to run
  bigger chunks.

Local CPU measurements (Mac mini, fp32, B=32 rows, A=256 full menu, S=64
tokens, seq=320, warm best-of-5, peak-RSS delta per fresh process):

| Path | Full forward S | Full forward M | CGAB tail S | CGAB tail M (B=25) |
|---|---|---|---|---|
| materialized | 497 ms / 7.9 GiB | 3590 ms / 12.1 GiB | 881 ms / 6.9 GiB | 2040 ms / 9.5 GiB |
| fused | 112 ms / 0.23 GiB | 399 ms / 0.35 GiB | 6.4 ms / 0.08 GiB | 13 ms / 0.13 GiB |

Analytic dominant-intermediate ratio: `seq * d_model / vocab` = 3840× (S) /
7680× (M) at these shapes. The CPU speedup (4.4× S / 9.0× M end-to-end) is
mostly avoided memory traffic; the expected GPU win is memory-bandwidth plus
the ability to raise `CASCADIA_EVAL_CELL_BUDGET` for larger serving chunks.
Equivalence, padding contract, and default-off behavior are pinned by
`cascadiav3/tests/test_cgab_fused.py`.
