# R2-MAP Docker worker

All non-MLX R2-MAP collection, simulation, evaluation, and benchmark commands
run in the canonical `linux/arm64` worker image. MLX/Metal training remains a
native john1 operation because Linux Docker cannot expose Apple Metal.

The native John1 lifecycle commands installed by `uv sync --frozen` are:

```bash
cascadia-mlx-r2-map-train --help
cascadia-mlx-r2-map-serve --help
cascadia-mlx-r2-map-verify --help
cascadia-mlx-r2-map-promote --help
```

Their run, checkpoint, compact-index, and disposable-window paths must remain
below `/Users/johnherrick/cascadia-bench/r2-map-v1`.

Build, test, and publish the image once on john1:

```bash
export DOCKER_HOST=unix:///Users/johnherrick/.local/share/cascadia-r2/colima/cascadia-r2/docker.sock
make docker-r2-map-build R2_MAP_IMAGE=cascadia-r2-map:dev
make docker-r2-map-smoke R2_MAP_IMAGE=cascadia-r2-map:dev
make docker-r2-map-bootstrap-smoke R2_MAP_IMAGE=cascadia-r2-map:dev
make docker-r2-map-iteration-smoke R2_MAP_IMAGE=cascadia-r2-map:dev
make cluster-fabric-build-push CLUSTER_IMAGE_TAG=dev
```

The image is non-root, has no MLX dependency, and writes only below `/output`.
Learned-model inference runs locally through the NumPy/OpenBLAS CPU backend,
which loads the same float32 `model.safetensors` as John1's MLX backend and
speaks the identical R2MP v3 framed protocol. Fixed-fixture golden tests require
MLX/CPU logits to match within `2e-5`; complete legal screens are processed in
bounded candidate chunks without pruning or reordering. The chunk size is
explicitly configurable for profiling; the 256-candidate default is selected
from OpenBLAS arm64 worker measurements. Candidate afterstates preserve three
unchanged opponent boards, so those exact repeated rows are encoded once per
chunk and broadcast into the unchanged slots. The service verifies this
invariant before using the fast path and otherwise evaluates the full graph.
Logical legal screens are scored as consecutive ordered frames of at most
1,024 candidates. This stays well below both R2MP v3's 8,192-candidate
protocol ceiling and its 1-GiB tensor-byte ceiling for the fixed public-state
schema, while retaining enough memory headroom for multiple independent local
workers. Every frame remains exhaustive; the Rust client rejects missing,
partial, reordered, or identity-mismatched frame
responses and concatenates all scores in original enumeration order before
argmax or exploration. This is transport partitioning, not candidate pruning.
Production jobs use standard trusted Bacalhau Docker execution with explicit
CPU, memory, disk, and timeouts. They bind immutable image digests and
content-addressed inputs, publish execution-specific outputs to MinIO, and
return strict manifests and checksums. Custom sandbox hardening is not a
cutover gate for this private cluster.

The following direct `docker run` is retained only as a local collector smoke;
distributed campaigns use `cluster.map`/`submit_map` and never address a host:

```bash
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 512 --memory 12g --cpus 9 \
  --mount type=bind,src="$PWD/output",dst=/output \
  cascadia-r2-map:dev collect-r2-map-bootstrap \
  --output /output/john1-bootstrap --campaign-id r2-map-expert-iteration-v1 \
  --iteration 0 --host john1 --first-game-index 0 --games 100 --shard-games 100 \
  --collector-hash "$COLLECTOR_HASH" --source-hash "$SOURCE_HASH" \
  --serving-protocol-hash "$SERVING_PROTOCOL_HASH"
```

`collect-r2-map-iteration` is the corresponding expert-iteration command. It
requires one newest manifest identity and zero to three historical manifest
identities from the frozen serving bundle; missing opponent seats are filled by
the canonical greedy policy. The collector enforces exactly one newest seat,
the iteration exploration schedule, exhaustive legal-action scoring, and
restart-safe checksummed shards.

The iteration smoke target uses `--exact-score-reference`, a deterministic
immediate-score predictor that exercises the complete exhaustive inference and
trajectory pipeline without claiming learned-model strength. Production
generation omits that flag and supplies `--bundle`,
`--newest-manifest-identity`, and any historical manifest identities.
Inside the worker image also pass `--python /usr/bin/python3 --python-path
/opt/cascadia/repo/python`. Mount the frozen bundle and every checkpoint/run path it
names read-only at the same absolute container paths.

## Measured CPU worker topology

The real one-step checkpoint was profiled with complete legal screens and no
candidate pruning. Every completed shard was replay-validated.

| services per host | OpenBLAS threads each | games | wall seconds | projected three-host games / 45 min |
|---:|---:|---:|---:|---:|
| 1 | 8 | 1 | 119.57 | 67.7 |
| 2 | 4 | 2 | 133 | 121.8 |
| 4 | 2 | 4 | 155 | 209.0 |

Use four independent services per worker host, two OpenBLAS threads, a 2.5-CPU
limit, and a 3-GiB memory limit as the measured starting topology. Each service
owns a disjoint game-index range and output directory. Do not share one service
between games: the Rust runner intentionally locks a framed service for the
whole game. The detailed reports are `cpu-stage-profile.json` and
`cpu-throughput-profile.json` under the campaign smoke directory. This
throughput is scientifically usable but remains far below the original
thousand-game-per-window estimate; plan generation volume from the measured
rate, not the earlier assumption.
