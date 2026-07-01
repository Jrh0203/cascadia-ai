# R4 Adaptive Multi-Resolution Census

Standalone Rust foundation for `r4-focal-nearfield-topology-v1`.

R4 gives the acting board a regular centered near field and represents every
opponent board through explicitly ablated far-field topology:

- radius four: 61 cells;
- radius five: 91 cells;
- exact coordinate overflow for every occupied entity outside the focal disk;
- exact habitat, wildlife, and frontier summary blocks; and
- an exact-far model-visible control.

A complete centered hex disk has `1 + 3r(r + 1)` cells. There is no regular
121-cell centered hex disk: radius five has 91 cells and radius six has 127.
R4 therefore tests both exact regular supports nearest the historical
“roughly 121 cells” observation without clipping the legal state.

The canonical exact envelope is `CSR4AM1`. Local indices plus exact overflow
coordinates reconstruct the accepted R2 `SparsePublicState`; summary tokens
never participate in decoding.

## Frozen Arms

Both radii evaluate the complete H/W/F factorial and one exact control:

| ID | Habitat | Wildlife | Frontier | Exact far |
|---|---:|---:|---:|---:|
| `n0-near-only` | 0 | 0 | 0 | 0 |
| `h-habitat` | 1 | 0 | 0 | 0 |
| `w-wildlife` | 0 | 1 | 0 | 0 |
| `f-frontier` | 0 | 0 | 1 | 0 |
| `hw-habitat-wildlife` | 1 | 1 | 0 | 0 |
| `hf-habitat-frontier` | 1 | 0 | 1 | 0 |
| `wf-wildlife-frontier` | 0 | 1 | 1 | 0 |
| `hwf-all-topology` | 1 | 1 | 1 | 0 |
| `e-exact-far-control` | 0 | 0 | 0 | 1 |

## Verification

```bash
cd /Users/johnherrick/cascadia/tools/r4_adaptive_multires_census

cargo fmt --manifest-path Cargo.toml -- --check
CARGO_BUILD_JOBS=1 cargo test
CARGO_BUILD_JOBS=1 cargo clippy --all-targets -- -D warnings
```

The permanent suite covers:

- every 61/91-cell index and inverse;
- exact local-plus-overflow reconstruction;
- byte-identical packed decode/re-encode;
- all twelve D6 transform/inverse pairs with carried centers;
- target independence;
- malformed packed input rejection; and
- seven legal long-range collision/retention fixtures at both radii.

## Adversarial Artifact

```bash
cargo run --release -- adversarial \
  --output ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/adversarial.json
```

The report records left/right feature hashes for all nine arms, not only a
single pass bit. `n0-near-only` must collide, each registered H/W/F block must
resolve its pair, and HWF plus exact-far must resolve every pair.

## Four-Host Census

Each host owns one unique train/validation part pair:

| Host | Shard | Dataset parts |
|---|---:|---|
| john1 | 0 | train 0, validation 0 |
| john2 | 1 | train 1, validation 1 |
| john3 | 2 | train 2, validation 2 |
| john4 | 3 | train 3, validation 3 |

Run this command on each host after replacing `N` with its shard:

```bash
cd /Users/johnherrick/cascadia/tools/r4_adaptive_multires_census

cargo run --release -- census \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-N \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-N \
  --shard-index N \
  --shard-count 4 \
  --require-frozen \
  --output ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/shards/shard-N.json
```

Frozen mode verifies the exact dataset ID, split, row count, manifest BLAKE3,
and unique part ownership before processing a row. Each shard evaluates both
radii and all arms over its own source records.

## Aggregate And Order Proof

After collecting checksum-verified shard reports:

```bash
cargo run --release -- aggregate \
  --report ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/shards/shard-0.json \
  --report ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/shards/shard-1.json \
  --report ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/shards/shard-2.json \
  --report ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/shards/shard-3.json \
  --adversarial-report ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/adversarial.json \
  --forward-output ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/aggregate-forward.json \
  --reverse-output ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/aggregate-reverse.json \
  --order-proof-output ../../artifacts/experiments/r4-adaptive-multires-foundation-v1/order-proof.json
```

The aggregate:

- requires all eight frozen datasets exactly once;
- requires exactly 60,000 records;
- merges integer histograms before computing quantiles;
- verifies every shard scientific hash;
- normalizes shard order;
- writes forward and reverse aggregate documents; and
- records their document hashes and byte equality in the order proof.

## Promotion Gates

MLX work is authorized only when the classification is `passed`:

- exact mechanics, D6, target independence, and corpus coverage all pass;
- the adversarial scientific report is present and passes;
- packed-state P99 is at most 864 bytes;
- radius-four HWF P99 is at most 256 spatial tokens;
- radius-five HWF P99 is at most 288 spatial tokens; and
- forward/reverse aggregate documents are byte-identical.

Passing establishes a mechanically exact, compact representation candidate.
It does not establish lower MLX latency, ranking noninferiority, higher game
score, or progress to the 100-point mean. Those require the matched learned
successor specified by ADR 0154.

## Bounded Far-Quotient Successor

ADR 0155 keeps the accepted exact `CSR4AM1` radius-four state and adds the
strict little-endian `CSR4BQ1` model envelope. It evaluates four independent
ways to replace the unbounded wildlife-signature and frontier-signature
streams:

| Arm | Host | Structural maximum |
|---|---|---:|
| `q1-seat-marginal` | john1 | 184 |
| `q2-directional` | john2 | 204 |
| `q3-affordance` | john3 | 200 |
| `q4-selective-exact` | john4 | 224 |

Build and run the proof matrix:

```bash
CARGO_BUILD_JOBS=1 cargo build --release \
  --bin r4-bounded-far-quotient-census

./target/release/r4-bounded-far-quotient-census adversarial \
  --require-pass \
  --output ../../artifacts/experiments/r4-bounded-far-quotient-foundation-v1/adversarial.json
```

Each production arm consumes all eight frozen dataset roots and all 60,000
positions. Example:

```bash
./target/release/r4-bounded-far-quotient-census census \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-0 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-1 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-2 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-train-part-3 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-0 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-1 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-2 \
  --dataset-root ../../artifacts/datasets/r0-spatial-position-corpus-v1-source-frozen-validation-part-3 \
  --arm q1-seat-marginal \
  --require-frozen \
  --output ../../artifacts/experiments/r4-bounded-far-quotient-foundation-v1/arms/q1-seat-marginal.json
```

The bounded report independently measures token count, active primitive
scalars, padded scalar slots, canonical bytes, exact source-bucket accounting,
and paired construction-plus-encoding time against the full radius-four HWF
view. The campaign generator is
`tools/r4_bounded_quotient_campaign.py`; it proves four-host adversarial parity
before releasing the four distinct full-corpus arms.

## References

- `docs/v2/decisions/0154-r4-adaptive-multires-foundation.md`
- `docs/v2/decisions/0155-r4-bounded-far-quotient-foundation.md`
- `docs/v2/reports/r4-adaptive-multires-foundation-v1-preregistration.md`
- `docs/v2/reports/r4-bounded-far-quotient-foundation-v1-preregistration.md`
- `docs/v2/RESEARCH_IMPLEMENTATION_PLAN_TO_100.md`
