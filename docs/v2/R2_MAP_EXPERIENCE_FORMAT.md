# R2-MAP Experience And Collector Contract

Schema versions: trajectory `2`, collector manifest `1`.

The authoritative R2-MAP training artifact is a compact, replayable game
trajectory. Expanded R2 tensors and legal afterstates are regenerated lazily;
they are not stored in collection shards.

## Identity and information boundary

Every record binds the campaign, iteration, host, global game index, game seed,
four seat policies, checkpoint hashes, exploration schedule, RNG domains,
collector/source/serving hashes, ordered actions, score breakdowns, and terminal
hashes. Iterative records require exactly one newest checkpoint in the focal
seat. Focal seats rotate as `global_game_index % 4`.

Seeds are derived from campaign, phase purpose, iteration, and global game
index. Host identity is deliberately excluded from seed generation; disjoint
host leases therefore cannot silently produce different games for a duplicated
index. Only `john1`, `john2`, and `john3` are accepted.

Selected afterstates are constructed with
`GameState::preview_public_afterstate`. They include the selected action but
exclude hidden market refills. The next decision's public parent contains the
refill only after it has become public.

Each turn records exact Pinecone accounting:

```text
before + earned - wipe_spend - independent_draft_spend = after
```

Bootstrap games expose all 80 decisions as primary examples. Iterative games
expose only the newest model's 20 focal decisions. Component targets reconcile
the five habitats, five animals, and remaining Pinecones to the terminal base
score.

## Shards and recovery

`dataset.json` identifies the immutable collection contract. Each
`shard-NNNNN.r2sh` contains one contiguous game-index prefix and has both an
internal payload checksum and a manifest file checksum. A shard is committed
before an atomically renamed and fsynced manifest update.

Resume verifies every registered shard before doing work. It skips only those
registered shards. A crash-complete but unregistered next shard is discarded
and regenerated; any other unknown artifact, checksum failure, range gap,
protocol change, policy change, or seed-lease change fails closed. An OS file
lock prevents concurrent writers and is released automatically if a process
exits.

## Bootstrap command (frozen v1 evidence)

The existing 100,000-game bootstrap was produced by the host-lease v1 schema
and remains immutable training evidence. Do not copy this interface into a new
campaign. New generation expresses disjoint logical seed ranges as
`ContainerInput` records and lets Bacalhau choose john1-john3 placement.

John1 is the active dataset authority. Historically John1 built one arm64 Docker image and
distributed that frozen image to John2 and John3. Each host received a disjoint
game-index lease and writes only its own bounded output directory or Docker
volume. All three hashes are required registered 64-character hexadecimal
identities; zero hashes are rejected. A representative in-container lease is:

```bash
/usr/local/bin/cascadia-v2 collect-r2-map-bootstrap \
  --output /output/john2/worker-0 \
  --host john2 \
  --first-game-index 33334 \
  --games 33333 \
  --shard-games 256 \
  --collector-hash COLLECTOR_SHA256 \
  --source-hash SOURCE_SHA256 \
  --serving-protocol-hash SERVING_PROTOCOL_SHA256
```

Use the same command with `--resume` and byte-for-byte identical arguments to
continue an interrupted lease.

## Completion validation

`collect-r2-map-bootstrap` and `collect-r2-map-iteration` perform the full
semantic replay and primary-example extraction validation before returning
zero. Dispatch wrappers must capture the successful collector JSON payload and
then run `cascadia_mlx.r2_map_collector_audit`. The audit is intentionally
lightweight: it binds that payload to `dataset.json`, verifies the complete
lease, contiguous shard sequence, byte counts, and every BLAKE3 digest, then
writes `validation.json`, `completion-audit.json`, and
`copied-files.sha256` atomically.

Do not run `validate-r2-map-collector` immediately after a successful collector
in production dispatch. That command repeats the complete replay/example pass
and is reserved for independent forensic revalidation. The standard wrapper is:

```bash
/usr/local/bin/cascadia-v2 collect-r2-map-bootstrap ... >collector-validation.json
python3 -m cascadia_mlx.r2_map_collector_audit \
  --dataset /output/john2/worker-0 \
  --validation-payload collector-validation.json \
  --validation-manifest /output/john2/worker-0/validation.json \
  --receipt /output/john2/worker-0/completion-audit.json \
  --copy-manifest /output/john2/worker-0/copied-files.sha256 \
  --semantic-validation-proof collector-zero-exit
```

The reusable Rust boundary is `R2MapGameRunner`. W4 supplies a runner that maps
the four registered seat identities to four frozen local policy handles and
returns deterministic focal exploration draws. The collector independently
replays and rejects mismatched outputs.
