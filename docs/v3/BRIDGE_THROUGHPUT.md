# Bridge Throughput Investigation (R2.4)

Status: **CLOSED 2026-07-13** — every lever measured, none cleared its
preregistered bar (EXPERIMENT_LOG 07-13 03:25): pipelining +4.2%
(bit-identical on CUDA, below the 10% bar), CHUNK_ROWS +3.9% bound (eager
forward saturates by batch 32), compile +0.5%, bucket negative and
numerics-drifting. Serving throughput is within ~5% of what this
architecture yields; the remaining ~45% of wall is Rust-side search
compute. All knobs stay landed and default-off — re-price if the model or
topology changes (bigger model raises the forward share). Line references
are to the working tree as of the 07-12 memo.

Measured production facts driving this memo (n1024/d16, jobs12, RTX 5090,
WSL2): GPU util mean/P50/P90 = 63.8%/66%/85% (observed range 2–88%), 350 W,
2403 MiB, exporter ~779% CPU, bridge ~407% CPU. jobs12→24 changed nothing —
the Python bridge is the throughput bound, not client parallelism.

## 1. Request lifecycle

One `SharedBridge` = one spawned Python process = one CUDA context, shared by
all 12 exporter sessions (`--model-sessions 12 --shared-model-session`).

Rust side (`cascadiav3/real-root-exporter/src/model_bridge.rs`):

1. **Submit** — worker threads call `SharedBridgeClient::eval_batch`
   (`model_bridge.rs:883`), which posts an `AggregateJob` to the aggregator
   thread and **blocks** on a reply channel.
2. **Coalesce** — the aggregator (`SharedBridge::spawn`,
   `model_bridge.rs:823`) takes the first pending job, then keeps gathering
   for `CASCADIA_SHARED_GATHER_US` (default 2 ms, `model_bridge.rs:798`) per
   extra job until `CASCADIA_SHARED_ROW_CAP` rows (default 192,
   `model_bridge.rs:804`).
3. **Wire** — merged roots are serialized as one JSON line
   `{"type": "eval_batch_request", "roots": [...]}` and written to the
   bridge's stdin (`model_bridge.rs:245-266`). Feature payloads are
   Rust-precomputed `packed_features` (base64 f32 token/action rows + u8
   relation tail), not raw dictionaries.
4. **Wait** — the aggregator blocks reading the single response line before
   it even starts gathering the next merged batch. There is **no pipelining
   anywhere**: at most one request is in flight per process pair.

Python side (`cascadiav3/src/cascadiav3/torch_inference_bridge.py`):

5. **Decode** — the serve loop (`torch_inference_bridge.py:1120`) is
   single-threaded: `json.loads` of the whole merged line
   (`torch_inference_bridge.py:1121`), dispatch on `eval_batch_request`
   (`:1150`).
6. **Chunk** — `_eval_batch_chunks` (`:197`) splits the merged roots at
   `chunk_size` rows (default `EVAL_BATCH_CHUNK_SIZE = 32`, `:63`; env
   override `CASCADIA_EVAL_CHUNK_ROWS`, `:101`) AND a
   rows×actions×seq cell budget (`EVAL_BATCH_CELL_BUDGET = 2^21`, `:72`;
   production raises it to 2^24 via `CASCADIA_EVAL_CELL_BUDGET`, `:75`).
   **A 192-row merged request therefore becomes ≥6 serial forwards.**
7. **Collate** — `_collate_packed_inference_roots` (`:307`): base64-decode,
   `np.frombuffer` reshape, pad rows into batch-max-shaped numpy buffers,
   one zero-copy `torch.from_numpy` per input (`:349-380`). Optional shape
   bucketing pads to a bounded shape vocabulary
   (`CASCADIA_BRIDGE_BUCKET=1`, `:154`).
8. **H2D** — `_model_inputs_to_device` (`:633`): per-chunk
   `pin_memory().to(device, non_blocking=True)` for exactly the 6 model
   inputs (`:643`); afterstate scores and ids never leave the host.
9. **Forward** — `_model_eval_batch` (`:826`) under `torch.inference_mode`
   (`:878`), optional bf16 autocast (`:665`, ruled label-unsafe — see §3).
   The CascadiaFormer forward (`torch_cascadiaformer.py:406`) is
   root-factored: tokens run the encoder stack once per root; actions enter
   at cross-attention + the CGAB tail (fused via `CASCADIA_CGAB_FUSED=1` in
   production).
10. **D2H** — one `.float().cpu()` per output head per chunk
    (`torch_inference_bridge.py:908-915`); `final_q = exact_afterstate +
    score_to_go` computed on host (`:918`).
11. **Encode** — per-row response dicts; production uses `packed_response`
    (base64 f64, bit-exact vs JSON floats, `:170`); one
    `json.dumps(sort_keys=True)` line back on stdout (`:55`).
12. **Demux** — Rust parses the line, splits results back to the waiting
    jobs (`model_bridge.rs:846-856`).

Phase timing already exists: `CASCADIA_BRIDGE_TIMING=1` (`:698`) accumulates
collate/H2D/forward/D2H/encode per chunk with proper device sync and prints
every 50 chunks. **No production timing capture exists yet** — that is
step 0 of any preregistered experiment.

## 2. Where the wall time plausibly goes

The GPU is idle whenever the serial loop is doing anything except step 9.
With mean util 63.8%, ~36% of wall time is non-forward. Given the code:

- **Serialization points (structural):** gather window → Rust JSON encode →
  Python decode → 6× (collate → H2D → forward → D2H → encode) → Python
  stdout → Rust decode. Every arrow is sequential; the dips to 2% util are
  the loop tail (encode + wire + gather) between merged requests.
- **Bridge at 407% CPU** on a single-GIL process means most of that CPU is
  GIL-free native work: torch CPU ops in collate, base64/numpy, CUDA driver
  threads, and `json.loads/dumps` (GIL-held). The GIL-held share bounds how
  much a Python-internal pipeline thread could overlap — unknown until
  TIMING=1 numbers exist.
- **Forward launch overhead:** CascadiaFormer-M at rows≤32 with short
  sequences is small-kernel-heavy (12 encoder layers × several kernels ×
  ≥6 chunks per merged request). This is exactly the CUDA-graphs
  (`reduce-overhead`) regime.
- **Known-cheap already:** packed responses (7.7× encode win, see
  PERFORMANCE.md "Pass 2"), pinned+non_blocking H2D, inference_mode, fused
  CGAB, one D2H per head per chunk.

Honest unknowns: per-phase split on john0 (never measured in production),
WSL2 pinned-alloc cost, CPU core budget (exporter 779% + bridge 407% vs
core count), and whether the 2 ms gather window is ever the binding
constraint at jobs12.

## 3. Ranked optimization candidates

Ranking = expected gain × confidence ÷ cost, given the measured 63.8% util.
"Gain" is end-to-end serving throughput, honest order of magnitude.

| # | Candidate | Mechanism | Expected gain | Exactness | Cost | How to measure on john0 |
|---|---|---|---|---|---|---|
| 1 | **Request pipelining / double-buffering** — LANDED 2026-07-12, both halves, default off: Rust `CASCADIA_SHARED_INFLIGHT=2+` (aggregator sends while responses are outstanding, FIFO demux, desync fails all in-flight loudly; `model_bridge.rs spawn_with_options`) + Python `CASCADIA_BRIDGE_PIPELINE=1` (stdin reader thread + one-deep deferred finalize; §4) | Overlap host phases (steps 5-8, 10-12) with GPU forward; the serial aggregator blocked end-to-end | **MEASURED 2026-07-13: 1.042x decision / 1.035x wall — below the 10% adoption bar; stays default-off.** Per-seed scores bit-identical on CUDA (all 12 A/B seeds). Each Rust worker blocks on its own evals, so a second full batch is rarely ready inside the gather window; revisit after #2 (CHUNK_ROWS) shifts the balance | bit-identical per request — now production-proven, not just torch-CPU-proven | zero (landed) | A/B DONE (`pipeline_ab_20260713_verdict.md`); rule fired: leave off, revisit after CHUNK_ROWS |
| 2 | **`CASCADIA_EVAL_CHUNK_ROWS=192`** (landed, default 32) | One merged request = 1-2 forwards instead of ≥6; fewer launch/D2H/sync rounds; larger GPU batches | 1.1–1.3x | numerics-drift (chunk-max padding regroups reductions, ~1e-7 class — same drift class the chunker already admits when membership shifts) | zero (landed) | bridge_throughput_probe batch 192 rows/s vs eager; then paired gate |
| 3 | **`CASCADIA_BRIDGE_COMPILE=1` + `CASCADIA_BRIDGE_BUCKET=1`** (compile mode now defaults to `reduce-overhead`, `CASCADIA_BRIDGE_COMPILE_MODE` to override) | CUDA graphs eliminate per-kernel launch overhead; bucketing keeps the recompile set finite | 1.1–1.5x of the forward phase (launch-bound share unknown); CPU smoke shows compiled ≈ eager on tiny shapes, GPU is the real test | numerics-drift (measured on CPU probe smoke: max abs diff ~2e-6 on q, ~6e-8 on priors → **needs paired score gate**) | zero (landed; warmup covers bucket shapes) | probe arms compile / compile_bucket; watch first-chunk latency + recompile count |
| 4 | Bigger effective batches from the client: raise `CASCADIA_SHARED_ROW_CAP` (192→384) and/or gather window once #2 is in | Amortize per-request wire+decode over more rows; GPU batches closer to saturation | 1.05–1.2x | bit-identical per response, but batch composition changes → reduction-order drift in practice | trivial (env) | sweep ROW_CAP × CHUNK_ROWS in a jobs12 paired probe |
| 5 | **Second bridge process** (2 CUDA contexts, 6 sessions each) | Coarse pipelining: one bridge computes while the other decodes/encodes | 1.1–1.3x, partially redundant with #1; context-switch + memory cost (2.4 GiB → ~4.8 GiB, fine on 32 GiB) | bit-identical per request | low-medium (exporter already supports per-session bridges; needs a 2-shard shared mode) | paired jobs12: 1×12 vs 2×6, GPU util + games/h |
| 6 | Response encode micro-cuts (skip `sort_keys=True`, pre-sized dicts) | `json.dumps(sort_keys=True)` on ~192 small dicts per request | ≤1.05x (encode already 7.7×-reduced by packed_response) | bit-identical payload semantics (key order changes only) | trivial | TIMING=1 encode share; only act if >5% |
| 7 | Collate vectorization (batch base64 decode, single padded ndarray fill) | Python loop over ≤192 rows with numpy slice writes | ≤1.05x (packed features already moved the heavy work to Rust) | bit-identical | low | TIMING=1 collate share |
| 8 | CPU affinity / priority for the bridge (pin bridge threads away from the 12 exporter workers, `chrt`/`taskset`) | Exporter at 779% CPU can starve the bridge's serial phases | unknown, 1.0–1.15x | bit-identical | trivial | paired run with `taskset -c` split; watch bridge-phase times |
| 9 | Persistent pinned staging pool (preallocate max-shape pinned buffers, copy-into-slice) | Avoid per-chunk `cudaHostAlloc`/register in `pin_memory()` (`:643`) | ≤1.05x unless WSL2 pinning is pathologically slow (TIMING h2d share will tell) | bit-identical | medium (shape-capacity management) | TIMING=1 h2d share first; deferred — see §5 |
| 10 | ~~bf16/fp16 autocast~~ (`CASCADIA_BRIDGE_AUTOCAST=bf16` exists, `:665`) | halve matmul width | ~1.3–1.8x forward | **ruled out**: 26% action agreement, label-unsafe (INFRASTRUCTURE.md §1) — never for generation or gates | — | closed |
| 11 | ~~TF32~~ (`CASCADIA_BRIDGE_TF32=1`, `:650`) | 10-bit-mantissa matmuls | already adopted | numerics-drift | — | already ON for generation, OFF for batteries (INFRASTRUCTURE.md §1); no further action |
| 12 | `torch.inference_mode` / no_grad | — | none | — | — | already the serving path (`:878`) |

Not viable as specified: cross-session batching beyond the above — all 12
sessions already share one aggregator and one process; there is no
per-session batching left to merge.

## 4. Landed opt-in changes (this pass, all default-off / no-op when unset)

`cascadiav3/src/cascadiav3/torch_inference_bridge.py`:

- `CASCADIA_BRIDGE_COMPILE_MODE` (`_compile_mode`, `:547`): torch.compile
  mode for `CASCADIA_BRIDGE_COMPILE=1`; default now `reduce-overhead`
  (`DEFAULT_COMPILE_MODE`, `:68`); `default` selects torch's default mode.
  NOTE: this changes what `CASCADIA_BRIDGE_COMPILE=1` alone does (was
  torch's default mode); no battery/production script sets COMPILE, so
  nothing shipping changes.
- Compile fallback hardening (`_maybe_compile_model`, `:557`): any
  `torch.compile` exception AND any CUDA warmup-forward failure
  (`_warmup_compiled_model`, `:592`, now returns success) prints a loud
  `bridge: WARNING ... serving eager` to stderr and serves the eager
  model — the bridge never crashes from compile.
- `CASCADIA_EVAL_CHUNK_ROWS` (`_eval_chunk_rows`, `:101`): overrides the
  32-row chunk cap; resolved in `_model_eval_batch` only when the caller
  passes no explicit `chunk_size` (`:857`).
- `bridge_env_provenance()` (`:674`): snapshot of compile/compile_mode/
  tf32/autocast/bucket/cgab_fused/eval_cell_budget/eval_chunk_rows/
  pinned_h2d/timing/ensemble_size, reported in the hello payload as
  `bridge_env` (`:1117`) so every run is attributable.
- `torch_model_throughput_benchmark.py` environment block now records
  `bridge_compile_mode` and `eval_chunk_rows`.
- **`CASCADIA_BRIDGE_PIPELINE=1`** (candidate #1, Python half; landed
  2026-07-12): pipelined serve loop — a stdin reader thread feeds a bounded
  queue (maxsize 4), and model-backed `eval_batch_request`s run phase-split
  (`_PipelinedEvalBatch`: `prepare` = decode+chunk+collate on the host,
  `launch` = H2D + forward with outputs left on device, `finalize` = D2H +
  row building) with a one-deep deferred finalize:
  `read+prepare(N+1) → finalize+write(N) → launch(N+1)`. Strict FIFO
  response order is preserved; when no next line is buffered the loop
  finalizes immediately instead of blocking (holding response N for a
  future request would deadlock an inflight-1 client), so overlap engages
  exactly when the Rust side pipelines (`CASCADIA_SHARED_INFLIGHT=2+`,
  `model_bridge.rs`). Every other message class (hello, shutdown, single
  eval_request, model-less fallback, malformed lines) drains the
  outstanding request first, then runs the serial loop's inline dispatch.
  `_model_eval_batch` itself is untouched — the phase split is an
  operation-for-operation twin, pinned to exact (`==`) output equality
  (incl. packed responses, pairwise/quantile modes, ensembles, TIMING
  accounting) by `tests/test_bridge_pipeline.py`. Default OFF = the
  historical single-threaded serial loop, byte-identical. Reported as
  `pipeline` in `bridge_env`; activation prints one loud
  `bridge: pipeline mode ON ...` stderr line.
- `torch_cascadiaformer_gumbel_benchmark.py` `execution_provenance` now
  records `shared_inflight` (mirrors the Rust `shared_inflight()`
  resolution: default 1, cap 8) and `bridge_pipeline`, so reports
  attribute both halves of the pipelining experiment.

Tests: `cascadiav3/tests/test_bridge_throughput_knobs.py` (13 tests —
defaults-off provenance, hello payload, compile-mode plumbing, compile and
warmup fallback, chunk-rows resolution incl. explicit-arg precedence, probe
helpers).

## 5. Pinned memory: implemented already, pool deferred

`CASCADIA_BRIDGE_PIN_MEMORY` was **not added**: pinned staging +
`non_blocking=True` H2D is already the unconditional CUDA path
(`_model_inputs_to_device`, `:633-643`, landed in Pass 2) — a flag would
either be a no-op or imply the default path is unpinned. The remaining
headroom is a persistent pinned buffer pool (candidate #9): per-chunk
`pin_memory()` allocates and registers fresh host memory every chunk.
Deferred because (a) buffer-capacity management across variable chunk
shapes is real complexity, and (b) there is no evidence yet that H2D is
>2-3% of wall time; decide after a production `CASCADIA_BRIDGE_TIMING=1`
capture. `pinned_h2d: true` is reported in `bridge_env` so reports stay
self-describing.

## 6. Probe + preregistration plan (run on john0 when idle)

Artifacts staged:

- `cascadiav3/scripts/run_bridge_throughput_probe.sh` — refuses to start if
  any `gumbel|torch_inference_bridge` process exists or GPU util ≥10%;
  TF32 hard off; generates dry-run roots (CPU-only exporter pass, menus up
  to 256) unless `ROOTS=` is given; writes
  `cascadiav3/reports/bridge_throughput_probe.{json,md}`.
- `cascadiav3/src/cascadiav3/torch_bridge_throughput_probe.py` — arms
  eager / bucket / compile / compile_bucket through the production
  `_load_model` gate + `_model_eval_batch` path at batches 8/32/96/192,
  warmup+timed reps, plus per-key max-abs-diff vs eager on identical
  inputs. (`torch_model_throughput_benchmark.py` was evaluated for reuse:
  it drives one env config per process and keeps only response digests, so
  it cannot produce the eager-vs-compiled numerics diff in one report; the
  probe reuses its root-loading/packing helpers instead.)

Preregistered decisions:

1. **Step 0 (before any knob):** one production-shape run with
   `CASCADIA_BRIDGE_TIMING=1` to get the collate/h2d/forward/d2h/encode
   split. This memo's ranking assumes forward-dominant-but-serialized; the
   split can reorder candidates 6-9.
2. **Probe thresholds:** adopt a knob combination for a paired gate only if
   rows/s at batch 192 improves ≥10% vs eager (below that, gate cost isn't
   worth it). Record recompile behavior: if compile arms show first-iter
   latency >30 s or per-shape recompiles in steady state, bucketing is
   mandatory or compile is rejected.
3. **Numerics rule:** CPU smoke already shows compile is NOT bit-identical
   (max abs diff ~2e-6 on q). Unless the CUDA probe surprisingly reports
   0.0 diffs, **any compile/bucket/chunk-rows adoption needs a paired
   score gate** (same protocol as the TF32 precedent: paired seeds,
   CI-gated). Diff ≤1e-5 → standard paired gate; diff >1e-3 on priors/q →
   reject without gating (bf16-class risk).
4. **Adoption order:** #2 (CHUNK_ROWS) alone first — cheapest, one knob,
   one gate; then #3 (compile+bucket) on top; #1/#5 (pipelining) as a
   separate Rust-side experiment since it is bit-identical and needs no
   score gate, only a throughput A/B.
