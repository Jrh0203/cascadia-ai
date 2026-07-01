# ADR 0194: Compact on-demand R2-MAP training data

Status: accepted as W2/W3 production-scale closure; storage location superseded
by ADR 0195; bootstrap remains unauthorized

## Decision

Validated `.r2sh` shards are the only durable production training corpus.
The training pipeline builds one immutable compact game/source index, then
materializes exact-R2 frames from one source shard at a time into disposable
bounded windows. The adapter
may retain one current window and prefetch at most one next window. It deletes
both on close and removes stale named windows on startup.

Sampling is deterministic by `(seed, epoch)`: shard order and game order are
counter-hashed, while turns remain contiguous within a game. The cursor stores
epoch, shard offset, game offset, and turn offset. The sampler state stores the
seed and protocol. Rust continues to own replay validation, exact sparse state
construction, and D6. Python validates source identity, each local manifest,
frame checksums, target algebra, token layout, and class bounds before creating
only the selected MLX batch.

Whole-game validation is a fresh lazy iterator over source windows. The fixed
prediction panel is rebuilt from the lowest-index sealed validation game. The
same compact dataset hash and adapter protocol are bound into checkpoints, so a
resume must reproduce the exact next batch identity.

## Storage and safety gates

- All authoritative compact sources, indexes, windows, runs, and temporary
  files remain beneath John2's canonical root from ADR 0195. Any John1
  execution staging is bounded, non-authoritative, outside the external SSD,
  and deleted after verified install.
- A source window defaults to at most 1 GiB; prefetch is restricted to zero or
  one window.
- Index construction is sequential and applies the same window bound.
- Before production training, the CLI projects 100,000 games and refuses a
  compact plan above the 40-GiB per-run budget.
- Persistent corpus-scale `.r2map` inputs are reference/test-only and require
  an explicit flag. There is no production fallback to them.
- Index and source identity drift, a cursor outside an indexed game, an
  oversized window, or a window source mismatch fails closed.

## Evidence

The four-game physical corpus contains 49,188 bytes of `.r2sh` and 320 exact
examples. Its compact index binds the same aggregate dataset BLAKE3 as the
previous full-stream export. Projected to 100,000 games:

- compact replay: 1,229,700,000 bytes;
- compact index: 59,600,000 bytes;
- current plus one prefetched 1-GiB window: 2,147,483,648 bytes;
- total peak additional storage: 3,436,783,648 bytes (fits 40 GiB); and
- persistent expanded corpus at 2 MB/game: 200,000,000,000 bytes (fails).

The real SSD smoke observed deterministic repeated batches, exact next-batch
identity after closing and reopening the adapter, streamed validation, an
80-frame fixed panel, zero leftover window files, 105,889,792-byte maximum RSS,
and zero process swaps.

## Consequences

Epoch changes can require each source shard to be replayed once because D6 is
derived on demand. This trades bounded sequential compute for a roughly 200-GB
storage avoidance at bootstrap scale. Shard-local ordering intentionally
reduces source-window churn; it remains a complete deterministic permutation
of training examples each epoch.
