# ADR 0078 Distributed Execution Record

Status: complete and rejected on validation on 2026-06-14.

## Allocation

| Host | Work | Split indices | Start (UTC) |
|---|---|---|---|
| `john1` | R12 train collection | 69,000-69,127 | 2026-06-13T19:26:18Z |
| `john2` | R12 validation collection | 70,000-70,031 | 2026-06-14T02:02:50Z |
| `john3` | frozen MLX training and validation | transferred train/validation only | 2026-06-14T05:20:16Z |

The train and validation domains are disjoint and were preregistered in ADR
0078 before either collection began. No statistical game is duplicated across
machines.

## Worker Identity

Both workers use the existing `john2` and `john3` SSH aliases over Tailscale.
Before dispatch:

- each node reported an Apple M4, 16 GB unified memory, AC power, disabled
  system sleep, automatic restart, and more than 390 GiB free;
- neither node had an active Cascadia workload;
- both worker source trees reported Git revision
  `a9918946f66c237a803b23ea299c6a514785ae52`;
- both worker source trees reported v2 source BLAKE3
  `2c761bb49bbc22fc84fbac437242025afe103322cfd5336f4f0950cf183426d4`.

john2 executes the same release collector copied from john1:

`183192792323090bac31de9ba8e4327ae466cb066f844447ef6a8c696fc122d1`

The train manifest records the earlier v2 source digest
`1bcfaf5645bfc478d70af4ca43a4e0595d59cfd393f9be7e0cdf07e299c7d1f8`
with the same executable checksum. The only source-root files modified after
that executable was built are the React cluster dashboard and
`cascadia-api` dashboard endpoints, including the later persistent telemetry
history view. Those dashboard changes remain local to john1 and were not
synced into john2's frozen collector checkout or john3's frozen MLX checkout.
`cargo tree -p cascadia-cli-v2` confirms that `cascadia-api` is not in the
collector dependency graph. No game, search, data, model, simulation,
provenance, CLI, or Python model source changed between the collection
launches or in john3's training snapshot.

## Commands

john1:

```bash
target/release/cascadia-v2 collect-counterfactual-advantage \
  --output artifacts/datasets/r12-counterfactual-advantage-v1-train-128 \
  --games 128 \
  --first-game-index 69000 \
  --split train \
  --groups-per-game 16 \
  --samples-per-candidate 12 \
  --candidate-selection stratified \
  --resume
```

john2:

```bash
./target/release/cascadia-v2 collect-counterfactual-advantage \
  --output artifacts/datasets/r12-counterfactual-advantage-v1-validation-32 \
  --games 32 \
  --first-game-index 70000 \
  --split validation \
  --groups-per-game 16 \
  --samples-per-candidate 12 \
  --candidate-selection stratified \
  --resume
```

The initial john2 launch was stopped before its first manifest or shard because
the lightweight checkout did not contain the Git object required for an
authoritative revision record. The exact commit bundle was installed and the
unchanged command was restarted. No validation output or statistic existed
before the restart.

## MLX Preflight

john3 received the locked CPython 3.12.13 environment and project virtual
environment used on john1. Its device probe reported:

- MLX 0.31.2;
- `Device(gpu, 0)`;
- macOS 26.4 on arm64;
- one-million-element evaluated computation with relative error below
  `8e-8`.

The focused ADR 0078 decoder/model suite passed on john3: six tests, zero
failures. john3 received the checksummed train and validation datasets after
both collections completed and ran the single frozen training job.

## Ownership Collision And Recovery

An unintended duplicate validation collector was discovered on john1 after
john2 completed its registered 32-game corpus. It had produced 17 local shards
under the same dataset ID with john1's current source provenance. Every one of
those 17 shard files was byte-identical to the corresponding john2 shard, but
john1 was not the registered validation producer.

The supervisor correctly stopped before training because the destination
manifest differed. PID 57917 was terminated, and the duplicate was archived
without contributing a record under:

`artifacts/datasets/invalidated/adr-0078-unregistered-john1-validation-afba36f8b1ce`

Its manifest SHA-256 is
`afba36f8b1ce8c3a0b47b64afcdd12ce22d7d85f18d9ba44cec35efc78b48a89`.
The complete john2 manifest SHA-256 is
`5fd3526aec30ce390b1767cb7bca7eb73496e1a16243cca38f0a15d058ecb990`.

The permanent handoff rule now rejects any unregistered local validation
collector before monitoring. During aggregation, only a byte-identical strict
prefix with the same immutable dataset contract may be archived and replaced
atomically by the complete registered producer corpus. Any changed contract,
shard metadata, byte, or non-prefix collision still fails closed.

## Unattended Handoff

`tools/adr0078_cluster_supervisor.py` is running as the one-shot launch agent
`com.johnherrick.cascadia.adr0078-supervisor`. The entrypoint is 73 lines and
delegates to focused runtime, transport, collection, training, and sealed-test
modules. The service:

- waits for the exact registered 128/128 and 32/32 manifests;
- reaches john2 through Tailscale first, then retries SSH and rsync through its
  verified LAN address only on transport exit 255, with strict host-key
  checking pinned to the existing Tailscale host identity;
- fails closed if either collector exits incomplete or makes no manifest
  progress for 45 minutes;
- verifies the frozen executable identity and validates each dataset on its
  producing host;
- transfers validation to john1 and validates both datasets again;
- transfers the exact binary and datasets to john3 and validates them there;
- verifies john3 revision, source digest, and `Device(gpu, 0)`;
- launches only the frozen 20-epoch/patience-five training command, resuming
  the same checkpointed run only after an abrupt interruption, with durable
  resume accounting, remote command identity checks, and a 30-minute
  artifact-progress guard;
- evaluates validation at most once and retrieves the complete run;
- on validation failure, records that ADR 0079 remains unopened;
- only on a complete pass, atomically records proof that the sealed test was
  absent on all three nodes, launches the exact ADR 0079 collector on john2,
  validates and transfers the result, evaluates the byte-identical checkpoint
  once on john3, and replays validation bit-exactly;
- keeps gameplay and promotion closed in every branch.

Eighteen focused orchestration, transport, evaluator, and structure tests
cover module size, manifest and command drift, process-lock ownership, durable
exact resumption, unreachable-host progress preservation, identity-pinned LAN
fallback, stalled training, sealed-test authorization ordering, checkpoint
identity, and validation replay. Live state and logs are under
`artifacts/logs/`.

## Result

The corrected train corpus completed 128 games, 2,048 groups, 8,192
candidates, and 98,304 continuations. The corrected validation corpus
completed 32 games, 512 groups, 2,048 candidates, and 24,576 continuations.
Both validated on their producing hosts, john1, and john3.

The frozen MLX run completed 640 optimizer steps over five epochs in 16.318
seconds on `Device(gpu, 0)` and stopped at the registered patience limit. The
zero-output initialization remained the selected checkpoint. Six validation
gates failed, so the model was rejected and ADR 0079 remained unopened.

Key identities:

- train manifest SHA-256:
  `ed6b64c90327b818dccddb2f7185404c5cd743b65bd5c963d1f19472dad1655c`;
- validation manifest SHA-256:
  `5fd3526aec30ce390b1767cb7bca7eb73496e1a16243cca38f0a15d058ecb990`;
- run manifest SHA-256:
  `2b8615d60f80ba8f6303464927f36e5aa1f140ed6e345c336452401353176ba8`;
- selected checkpoint manifest SHA-256:
  `314ac7fa77138b791f2092a0ef59e1a690fb2caf094f18218d0cab1dab803fd5`;
- validation report SHA-256:
  `6f0b9f87bac3532c1059f80156517b0eaf9e3feef1c0e2e67c81be3956d52629`.

## Completion Checklist

- [x] john1 train manifest reaches 128/128 and validates.
- [x] john2 validation manifest reaches 32/32 and validates.
- [x] validation dataset is copied to john1 and both manifests/checksums are
      independently revalidated.
- [x] both datasets are copied to john3 and revalidated before training.
- [x] the one authorized MLX run completes or stops at frozen patience.
- [x] selected-checkpoint validation gates are evaluated once.
- [x] artifacts are copied back to john1 and checksummed.
- [x] validation fails, so ADR 0079 authorization and its first test record
      remain absent.
- [x] test, inference, gameplay, and promotion remain unopened.
