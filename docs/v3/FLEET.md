# Mac Mini Fleet

The fleet is a pool of general-purpose Cascadia workers. It is not divided
into evidence roles.

## Hosts

| Host | Hardware | Access | Notes |
|---|---|---|---|
| `john1` | Apple M4 mini | local workspace or `ssh john1` | web UI may also run here |
| `john2` | Apple M4 mini | `ssh john2` | usually fast |
| `john3` | Apple M4 mini | `ssh john3` | historically about 10% slower |
| `john4` | Apple M4 mini | `ssh john4` | usually fast |
| `john0` | RTX 5090 workstation | `ssh john0` | CUDA and high-memory work |

Use every available host for generation, evaluation, training, or exact
wildlife search when it improves throughput. MPS and CUDA results do not need
to be bit-identical; compare the game-level outcomes that matter.

## Deployment

The minis normally use `~/cascadia`, a Python virtual environment, and a
native exporter build:

```bash
rsync -a --exclude target --exclude __pycache__ \
  --exclude checkpoints --exclude reports --exclude logs \
  ./ <host>:~/cascadia/

ssh <host> 'cd ~/cascadia && \
  export PATH="$HOME/.cargo/bin:$PATH" && \
  cargo build --release \
    --manifest-path cascadiav3/real-root-exporter/Cargo.toml'
```

There is no source-revision or source-hash preflight. If compatibility is in
doubt, rebuild and run a smoke test.

## Distributing work

Split independent units—seeds, roots, rulesets, count vectors, profiles, or
solver branches—into roughly equal-cost lists. Launch all hosts concurrently.
An ordinary task file with stable logical IDs is sufficient.

For long tasks:

```bash
ssh <host> 'cd ~/cascadia && \
  nohup <command> > cascadiav3/logs/<name>.log 2>&1 & \
  echo $! > cascadiav3/logs/<name>.pid'
```

Watch logs and process state directly. Heartbeats and terminal marker files
are optional conveniences.

## Collection

Merge successful outputs by the logical task key. Check:

- the output parses;
- the expected task is present;
- a returned board has the requested animal counts;
- the independent scorer agrees;
- exact solvers distinguish `optimal`, `infeasible`, and `timeout`.

Do not require source hashes, taskset hashes, artifact hashes, receipt files,
host identity, single-use tags, or a completely terminal fleet before using
finished work. Missing tasks can be relaunched anywhere.

## Existing scripts

The older fleet launchers remain useful starting points:

- `fleet_cbddb_launch.sh` / `fleet_cbddb_gen.sh`;
- `fleet_wildlife_exact_launch.sh`;
- `fleet_all_wildlife_bound_probe_worker.sh`;
- the `fleet_*_worker.sh` wildlife workers.

Their old ledgers and hash/provenance fields are legacy compatibility data.
New or updated paths should treat those fields as optional and should never
fail merely because they differ or are absent.

## Throughput notes

Historical MPS self-play was about 300 seconds per seed at n128/d2 with a
model-S checkpoint, largely limited by the shared MPS bridge. Re-benchmark:
the code, models, and PyTorch stack have changed.

CPU wildlife tasks vary by orders of magnitude. Balance by measured query
cost or dynamic work stealing when possible; equal task counts are often
poorly balanced.

## Stopping work

Resolve the exact PID from `ps` or the job's PID file and stop it. Do not use
a broad pattern that can catch unrelated processes.
