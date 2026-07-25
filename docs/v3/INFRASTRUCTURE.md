# Cascadia Machines and Runbook

This is a practical runbook, not a governance system.

## Machines

| Host | Hardware | Typical use | Access |
|---|---|---|---|
| `john0` | CUDA workstation, RTX 5090, roughly 121 GB RAM | training, generation, search, evaluation | `ssh john0` |
| `john1`–`john4` | Apple M4 minis with MPS | CPU/MPS search, data generation, evaluation, web UI | `ssh johnN` |

Every host may do any job it can execute usefully. Cross-device results need
not be bit-identical. Run independent work in parallel and measure the actual
throughput. Avoid accidental oversubscription by checking the process list
first; there is no one-job policy.

## Build

Local Mac:

```bash
RUSTC="$HOME/.cargo/bin/rustc" "$HOME/.cargo/bin/cargo" build \
  --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src \
  ./venv/bin/python -m unittest discover -s cascadiav3/tests
```

`cascadiav3/real-root-exporter` is a separate Cargo workspace, so build it
explicitly when it changes.

On john0:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CC=/home/john0/.local/bin/zig-cc
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/home/john0/.local/bin/zig-cc
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
source /home/john0/venvs/torch/bin/activate
```

On a mini:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
source "$HOME/cascadia/venv/bin/activate"
```

## Launching work

Foreground runs are ideal while tuning because metrics are visible. For long
runs, ordinary `screen`, `tmux`, or `nohup` is enough:

```bash
mkdir -p cascadiav3/logs
nohup bash <script> > cascadiav3/logs/<name>.log 2>&1 &
echo $! > cascadiav3/logs/<name>.pid
```

Check progress directly:

```bash
tail -f cascadiav3/logs/<name>.log
ps -p "$(cat cascadiav3/logs/<name>.pid)" -o pid,etime,%cpu,%mem,command
nvidia-smi
```

PID files, logs, metrics, and checkpoints are conveniences. They are not
receipts. A stale PID file has no authority over the process list.

Jobs may be stopped, restarted, extended, or retuned when that is the best use
of the machines. Resolve the exact target PID before stopping it.

## Data and checkpoints

- Use packed `.npz` shards for large corpora.
- Keep ordinary numbered checkpoints plus `latest` and `best`.
- Resume when the checkpoint loads and the intended model configuration is
  compatible. Source or dataset hashes are not required.
- Keep data long enough to be useful. Delete obsolete generated artifacts by
  exact path when space is worth more than retention.
- `/tmp` is fine for throwaway work. Put long-lived runs under
  `cascadiav3/{checkpoints,reports,logs,fleet}` so a reboot does not erase
  them.

## Performance knobs

TF32, bf16, compilation, batch size, worker count, bridge sessions, Rayon
threads, and host concurrency are empirical knobs. Benchmark complete
throughput and playing strength; do not preserve an old restriction solely
because a historical campaign used it.

Known environment quirk: on the john0 WSL/CUDA stack,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` has previously crashed the
trainer at its first forward. Re-test after environment upgrades; until then
use it only where it demonstrably works.

## Fleet use

Distribute independent roots, seeds, rulesets, count vectors, or solver
branches across john1–john4. A simple balanced task list plus `ssh` is enough.
Collectors should merge successful outputs by their logical key and report
missing tasks. They must not require taskset/source hashes or receipt files.

See [FLEET.md](FLEET.md) for host-specific paths and access notes.
