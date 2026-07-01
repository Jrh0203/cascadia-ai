# R2-MAP Replay-to-MLX Dataset Bridge

The durable training source is the validated, replay-authoritative `.r2sh`
corpus. Production training uses a small immutable game index and disposable,
bounded per-source `.r2map` windows. It never materializes a corpus-scale
expanded stream or a 4x139 padded cache on disk.

## Authority and information boundary

- `crates/cascadia-r2` owns exact sparse state construction, token encoding,
  D6 transforms, public market/player/global features, and compact token rows.
- `GradedOracleActionFeatures` remains the lossless complete-action authority.
  Rust streams its canonical 128 bytes; Python uses the existing decoder to
  produce the frozen 140-float input.
- Parents are public states before the played action. Candidate state is only
  the selected public afterstate from `preview_public_afterstate`, before any
  hidden refill. The bridge never calls `transition` to build a model input.
- A frame represents one observed transition. It does not invent return labels
  for unplayed actions. Serving independently enumerates every legal action and
  does not prune moves.

## Identities and splits

`dataset.json` binds the ordered source shard names, byte sizes, BLAKE3 hashes,
game ranges, game counts, and example counts. Absolute paths and mtimes are not
scientific identity. The dataset hash also binds feature, target, whole-game
split, D6, and protocol schemas. It additionally binds one round identity:
campaign ID, iteration, collection kind, and newest-checkpoint hash. Rust
rejects a source set that mixes any of those fields across shards, and Python
revalidates the same identity before MLX sees a frame. Bootstrap datasets bind
an absent newest-checkpoint hash; iterative datasets require one.

All examples from one game share a deterministic train/validation assignment.
Train D6 IDs use `r2-map-d6-cyclic-offset-v1`: a domain-separated base offset
is hashed once from game ID, post-Stop draft decision ID, and sampler seed,
then `(base + epoch) mod 12` visits every transform exactly once per 12
epochs. Bootstrap completion is accepted only at a positive multiple of that
cycle. Validation and fixed-panel streams use identity transform zero. Sorting
by global game index and turn makes output independent of input shard order or
worker scheduling.

## Wire contract

Each stream begins with `CSDR2MP\0`, schema/header sizes, dataset and stream
configuration hashes, frame/game counts, and a mode code. Every variable-size
frame has a little-endian length, payload BLAKE3, and payload.

The payload includes immutable game/position/action identity, turn/seat,
transform ID, canonical action bytes, exact afterstate base score, current,
residual, and terminal 11-component scores, ordered opponent targets, market
survival targets, one compact parent, and one compact selected afterstate.
Compact states keep board-major/type-major active rows and the exact 52-byte
token payload while omitting every unused board tail. Python reconstructs the
rules-complete 4x139 live padding only for selected batches. The older 4x92
shape remains a historical sparse-foundation cache identity and is rejected by
the live v3 serving protocol.

Market decision frames bind visible wildlife, all five public bag counts, and
their redundant checked total. Their legal screen is the canonical
information-set intersection: Keep/Stop plus every optional replacement that
stabilizes for every hidden bag order consistent with those public counts.
This is direct-gameplay legality, not candidate pruning. ADR 0018/0078's
rejection-conditioned teacher/search expectation remains separate and is not
silently imported into committed gameplay.

Live legality uses the exact O(5) public-count theorem rather than enumerating
hidden orders: two retained species stabilize immediately after the refill
capacity check; one retained species checks only its all-matching completion;
and an empty market is safe exactly when `T >= 4` and
`sum(count // 4) < T // 4`. The recursive multiset solver is retained only as
an independently exhaustive test oracle.

Observed scalar and component losses have a selected-only mask. A frozen
bootstrap-only 1% subset, derived from the post-Stop draft decision identity by
`r2-map-draft-imitation-subset-v1`, regenerates the complete canonical legal
draft screen and applies policy CE at the true selected index. Every other
draft row remains selected-only and has an explicit false policy mask.
Bootstrap market screens carry policy supervision only when they contain a
choice; iterative/explored and benchmark records never silently become
imitation targets.

Opponent and market-survival targets are conditioned on the selected candidate
afterstate, not the pre-action parent. The ordered opponent target covers free
replacement, all paid-wipe masks (count plus 20 fixed ordinals), and the draft.
Per-opponent and per-wipe masks prevent sentinel CE. Market disposition/pair
targets are masked unless all three subsequent opponents exist; final-slot
loss is further masked to tiles that survive. This keeps turns 77-79 valid
primary examples while preserving identifiable auxiliary tasks.

Checkpoint fixed-panel replay retains every frame from the sealed panel but
projects each to its replay-selected action before dense MLX batching. This
keeps the 80-frame regression panel bounded even when a bootstrap imitation
frame contains thousands of legal candidates; exhaustive maximum-width
coverage remains the separate serving-contract smoke.

## Build the compact production index on John2

ADR 0195 makes John2's internal APFS root canonical. Run this section on John2;
all outputs and Rust build artifacts stay under that root:

```bash
export R2_MAP_ROOT=/Users/john2/cascadia-bench/r2-map-v1
export TMPDIR=$R2_MAP_ROOT/tmp/rust
export CARGO_TARGET_DIR=$R2_MAP_ROOT/build/cargo-target

PYTHONPATH=python .venv/bin/python tools/r2_map_compact_dataset.py build-index \
  --shard $R2_MAP_ROOT/datasets/bootstrap/ITER/shard-00000.r2sh \
  --exporter $R2_MAP_ROOT/build/cargo-target/release/cascadia-v2 \
  --output $R2_MAP_ROOT/datasets/bootstrap/ITER/compact-index.json \
  --scratch $R2_MAP_ROOT/tmp/index-ITER \
  --maximum-window-bytes 1073741824
```

Repeat `--shard` for every validated source. Index construction visits one
shard at a time, validates both whole-game splits through Rust, records only
game identity/count/source metadata, and deletes each temporary stream before
moving on. The index binds the same aggregate dataset hash as a monolithic
export. It contains no model tensors.

Before training, run `validate` and `project`. Validation rehashes every `.r2sh`
against the index. Projection must show the compact 100,000-game plan below the
40-GiB per-run gate and the hypothetical expanded corpus above it:

```bash
PYTHONPATH=python .venv/bin/python tools/r2_map_compact_dataset.py validate \
  --index $R2_MAP_ROOT/datasets/bootstrap/ITER/compact-index.json \
  --shard-root $R2_MAP_ROOT/datasets/bootstrap/ITER

PYTHONPATH=python .venv/bin/python tools/r2_map_compact_dataset.py project \
  --index $R2_MAP_ROOT/datasets/bootstrap/ITER/compact-index.json \
  --target-games 100000 --maximum-window-bytes 1073741824 \
  --maximum-prefetch-windows 1
```

At runtime, the John2-side compact materializer deterministically hashes shard
order and game order from `(seed, epoch)`, keeps turns contiguous, and asks the
frozen Rust exporter for the current shard only. Rust computes D6 from the
immutable game/position identity, epoch, and seed. John1's remote adapter opens
an immutable token for each materialized object and consumes bounded verified
ranges directly in memory; it never creates a local window. Python verifies the
manifest, frame checksum, token layout, target algebra, and class bounds, and
pads only the selected in-memory batch. At most one current and one prefetched
window are live remotely. A 1-GiB per-window ceiling, candidate budget, group
budget, source hashes, and John2-root boundary all fail closed.

## Train and recover on John1

`r2_map_train` opens the compact adapter, binds the aggregate dataset and
adapter protocol identities into every checkpoint, trains only selected
observed-return targets, streams whole-game validation one source at a time,
and fully recomputes the fixed prediction panel before advancing
`last_verified`. John1 has no campaign staging directory. The remote client
opens immutable object tokens and supplies bounded, hash-verified ranges
directly to the in-memory adapter. `R2MapTrainer(in_memory=True)` keeps loss
records, model/optimizer state, cursor, sampler, RNG, and checkpoint
serialization in memory. `checkpoint_bundle()` produces a complete immutable
bundle that is streamed into a John2 checkpoint transaction; mutable loss and
pointer objects advance only by hash-CAS after the transaction receipt verifies.
Best-validation selection is reduced incrementally with the frozen
metric/step/manifest tie-break, bounding retained serialized checkpoint state
to the current candidate plus one best candidate.

Resume opens the canonical John2 pointer, loss stream, and checkpoint objects,
verifies every framed payload and receipt, then calls
`R2MapTrainer.resume_from_bundle(...)`. If post-checkpoint loss events exist,
use a new branch ID; the trainer never truncates the append-only loss stream.
The checkpoint cursor stores epoch, shard offset, game offset, and turn offset;
the sampler state stores the seed and protocol. Reopening at that state must
produce the exact same next batch identity and D6 tensors.
The immutable trainer identity also binds the group cap and padded-candidate
budget; resume rejects an adapter whose packing contract differs even if its
immediate next batch would happen to collide.
`r2_map_verify --compare-checkpoint` computes canonical semantic digests over
sorted model and optimizer tensors, so recovery parity does not depend on
container metadata ordering.

The uploader must complete the John2 atomic install and verification receipt
before an in-memory checkpoint is considered recoverable. A test must prove
that John1 creates no campaign file during train, checkpoint, injected failure,
or resume. The six legacy
`--*-manifest/--*-stream` arguments remain available only with
`--allow-reference-expanded-streams` for small regression fixtures. Production
bootstrap must use the compact path. Even explicit reference mode is bounded to
1 GiB by default. The CLI refuses an estimated compact plan above 40 GiB and
never silently falls back to persistent expanded streams.

The concrete production implementation is
`python/cascadia_mlx/r2_map_remote_training.py`, invoked by
`tools/r2_map_john1_train.py`. Each materialization uses a unique registered
John2 run, opens exactly the `.json` and `.r2map` outputs, reads contiguous
receipt-bearing ranges into one bytearray, commits the token/inventory-bound
cleanup, and publishes the complete read/cleanup evidence before the adapter
can consume the window. Export failure uses the distinct failed-run cleanup
token. A cleanup failure fails the training call; it is never downgraded to a
warning.

Checkpoint ordering is loss-stream SHA-CAS, immutable transaction commit,
remote object/provenance reopen, exact fixed-panel verification publication,
then pointer SHA-CAS. `last_verified` resume additionally reopens and validates
the named verification receipt and requires the pointer metadata to bind its
verification id. This ordering makes an unpointed complete checkpoint harmless
and prevents a pointer from naming an unverified or partial bundle.
