# Mac mini fleet (john1–john4): job distribution

Status: operational since 2026-07-22 (CBDDB era). Successor of the
hand-edited fleet3/4/5 workflow from the AAAAA campaign
(`INFRASTRUCTURE.md` §Fleet operations). This is the current reference
for distributing self-play generation across the minis.

## Hosts and access

| Host | HW | Access | Notes |
|---|---|---|---|
| john1 | Mac mini (10 cores) | `ssh john1` → `john1@100.110.109.6`, key `~/.ssh/john0_codex` | Orchestrator + web-UI host — do not run generation while the UI is in use. NOT provisioned for CBDDB yet. |
| john2 | Mac mini M4, 10 cores | `ssh john2` (ssh config) | Provisioned 2026-07-22 |
| john3 | Mac mini M4, 10 cores | `ssh john3` | Provisioned 2026-07-22; consistently ~10% slower than john2/4 |
| john4 | Mac mini M4, 10 cores | `ssh john4` | Provisioned 2026-07-22 |
| john0 | WSL box, RTX 5090, 32 cores | `ssh john0` (port 2222) | GPU box — never part of the fleet; all evals run here exclusively |

"Provisioned" means: current source at `~/cascadia` (rsync'd snapshot,
NOT a git repo), exporter built natively
(`~/cascadia/cascadiav3/real-root-exporter/target/release/...`), and a
torch venv at `~/cascadia/venv` with MPS working (torch 2.12.1).

## The three scripts (`cascadiav3/scripts/`)

All jobs are **seed-sharded self-play generation**: each host gets a
contiguous seed range and produces one tensor shard. There is no other
fleet job type today.

1. **`fleet_cbddb_launch.sh`** — run on the Mac. Allocates contiguous
   seed ranges across hosts, rsyncs the incumbent checkpoint
   (manifest + weights.pt) to each, launches `fleet_cbddb_gen.sh`
   detached (nohup + pid file), and writes a ledger to
   `cascadiav3/fleet/cbddb_<tag>_fleet.json`. Refuses to double-launch
   a tag that already has a ledger.

   ```bash
   CYCLE_TAG=mytag \
   HOSTS="john2 john3 john4" \            # default
   FIRST_SEED=<fresh block> \
   SEEDS_PER_HOST=60 \
   INCUMBENT_DIR=/local/path/to/checkpoint_dir \  # must contain best_locked_val.manifest.json + .weights.pt
   SOURCE_REVISION=<git rev of deployed source> \
   bash cascadiav3/scripts/fleet_cbddb_launch.sh
   ```

2. **`fleet_cbddb_gen.sh`** — runs ON a mini (launched by the above).
   Same gumbel-selfplay arguments as `run_cbddb_cycle.sh` generation
   (n128/d2 default, `--max-actions 8`, plies 80), MPS bridge.
   Artifacts land in `~/cascadia/cascadiav3/fixtures/` as
   `cbddb_<tag>_shard_<host>_{tensor.npz,manifest.json,decisions.jsonl}`;
   log + pid in `~/cascadia/cascadiav3/logs/`.

3. **`fleet_cbddb_collect.sh <tag> status|collect`** — run on the Mac.
   `status` prints per-host progress. `collect` (only when every shard
   is done) pulls shards to `cascadiav3/fleet/staging_<tag>/`, verifies
   each against the ledger (seed range from `metadata.seed_domain`,
   CBDDB ruleset id, npz sha256 vs manifest checksum, zero skipped
   seeds), pushes them to `john0:~/cascadia/cascadiav3/fixtures/`, and
   prints the comma-joined `--train` list. The trainer consumes
   multi-shard input natively (`--train a.npz,b.npz,c.npz`) — there is
   no merge step.

## Throughput (measured 2026-07-22, model-S incumbent, n128/d2)

- **~300 s/seed per host**, flat in session count (6/9/12 all within
  noise) — the single shared MPS bridge process is the bottleneck,
  not CPU. Do not bother re-tuning SESSIONS/RAYON; it was measured.
- Fleet total (3 hosts) ≈ 0.01 seeds/s ≈ **+25–30%** on top of john0
  (john0 does ~50 s/seed at n128/d2 pre-doubling; ~200 s/seed at
  n512/d8 with JOBS=24).
- Consequence: the fleet is only worth using for **cheap-search
  generation** (n128/d2-grade). Teacher-grade labels (n512/d8+) cost
  ~40 min/seed on a mini — always generate those on john0.

## Rules and caveats

- **Seed discipline**: every launch consumes a fresh, never-used seed
  block; audit and record it in `cascadiav3/EXPERIMENT_LOG.md` before
  launching (the launcher prints a reminder). Scratch/burn ranges used
  so far live in the log entries of 2026-07-22. Blocks 2027195000+
  are RESERVED for >105 certification — never touch.
- **Evals never run on the fleet.** MPS vs CUDA float differences make
  cross-hardware numbers incomparable with the john0 eval history.
  Corpus generation is fine (fleet3/4/5 precedent).
- **Provisioning refresh** (after any Rust/feature change): rsync
  source and rebuild —
  ```bash
  rsync -a --exclude target --exclude __pycache__ --exclude fixtures \
    --exclude checkpoints --exclude reports --exclude logs \
    ~/cascadia/Cargo.toml ~/cascadia/Cargo.lock ~/cascadia/crates \
    ~/cascadia/cascadiav3 <host>:~/cascadia/
  ssh <host> 'export PATH=$HOME/.cargo/bin:$PATH; cd ~/cascadia && \
    cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml'
  ```
- **Stopping a shard**: read the pid file
  (`~/cascadia/cascadiav3/logs/cbddb_<tag>_shard_<host>.pid`) and kill
  that explicit pid — never pkill by pattern.
- **john1**: authenticates as `john1@100.110.109.6` with
  `~/.ssh/john0_codex` (config entry exists), but repeated failed auth
  attempts trip macOS sshd per-source penalties — if you get
  "Permission denied" unexpectedly, wait a few minutes. It needs the
  provisioning refresh above before first CBDDB use.
- **Bacalhau/MinIO fabric** (`infra/`, `tools/cluster_*`): working but
  intentionally NOT used for self-play — it runs linux/arm64
  containers with no Metal access. Don't route generation through it.
