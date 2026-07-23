# Mac mini fleet (john1–john4): job distribution

Status: operational since 2026-07-22 (CBDDB era); exact wildlife-catalog
sharding added 2026-07-23. Successor of the
hand-edited fleet3/4/5 workflow from the AAAAA campaign
(`INFRASTRUCTURE.md` §Fleet operations). This is the current reference
for distributing CPU-safe independent work across the minis.

## Hosts and access

| Host | HW | Access | Notes |
|---|---|---|---|
| john1 | Mac mini (10 cores) | local workspace (`Johns-Mac-mini.local`) | Orchestrator + web-UI host; John authorized local CPU exact work for CBDDB on 2026-07-23. |
| john2 | Mac mini M4, 10 cores | `ssh john2` (ssh config) | Provisioned 2026-07-22 |
| john3 | Mac mini M4, 10 cores | `ssh john3` | Provisioned 2026-07-22; consistently ~10% slower than john2/4 |
| john4 | Mac mini M4, 10 cores | `ssh john4` | Provisioned 2026-07-22 |
| john0 | WSL box, RTX 5090, 32 cores | `ssh john0` (port 2222) | GPU box — never part of the fleet; all evals run here exclusively |

"Provisioned" means: current source at `~/cascadia` (rsync'd snapshot,
NOT a git repo), exporter built natively
(`~/cascadia/cascadiav3/real-root-exporter/target/release/...`), and a
torch venv at `~/cascadia/venv` with MPS working (torch 2.12.1).

## Self-play generation scripts (`cascadiav3/scripts/`)

Self-play jobs are **seed-sharded generation**: each host gets a contiguous
seed range and produces one tensor shard.

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

## Exact wildlife-catalog scripts

Pure-wildlife catalog proofs are independent by animal-count vector and use
CPU-only OR-Tools. John authorized distributing this computation on
2026-07-23. This is not gameplay evaluation, training-data generation, or
promotion evidence; each returned witness and proof ledger is validated again
on the orchestrator before import.

The minis use a dedicated CPU-only `~/cascadia/wildlife-venv-py312`, created
from the fleet's Python 3.12 generation interpreter but isolated from its
Torch/MPS packages. Both Python (currently `3.12.13`) and OR-Tools (currently
`9.15.6755`) are exact preflight/worker requirements.
`WILDLIFE_VENV` can select another safe path under `~/cascadia`; the path and
runtime versions are recorded in the launch ledger. The AAAAA run uses only
john2–john4. John authorized the subsequent CBDDB exact run to add local
john1 without stopping its web UI or champion service.

**Live 2026-07-23 09:16 EDT:** AAAAA tag
`aaaaa_exact_tail_fleet3_20260723` runs the 115-vector exact tail as
39/38/38 disjoint shards on john2/john3/john4 from revision `c726df87`.
See the experiment log for pinned hashes and PIDs; use the collector's status
mode rather than inspecting partial scores.

**Next ruleset authorization:** John authorized john1 for the CBDDB exact
catalog on 2026-07-23. The frozen 826-vector taskset is therefore planned as
207/207/206/206 shards on john1–john4. This does not waive preflight: john1
is the local orchestrator workspace and uses the repo `.venv`; john2–john4
remain SSH targets using `wildlife-venv-py312`. Every host must pass the same
exact runtime/source/collision checks before launch. The local worker runs in
a detached named `screen` session. john1's web UI and champion service must
not be stopped or restarted.

1. **`tools/wildlife_catalog_taskset.py`** freezes the currently unresolved
   canonical count vectors from a validated catalog snapshot:

   ```bash
   .venv/bin/python -m tools.wildlife_catalog_taskset \
     --scoring-cards AAAAA \
     --catalog <frozen-catalog-snapshot.json> \
     --output <taskset.json>
   ```

   For the 1,024 arbitrary-card catalog,
   **`tools/all_wildlife_tail_taskset.py`** freezes a complete inclusive slice
   by current unresolved-branch count. It independently checks every selected
   candidate score/count and can require byte-identical boards in a second
   candidate catalog before permitting legacy-proof union:

   ```bash
   .venv/bin/python -m tools.all_wildlife_tail_taskset \
     --catalog <integrated-catalog.json> \
     --candidates <legacy-unionable-candidates.json> \
     --comparison-candidates <current-candidates.json> \
     --min-branches 3 --max-branches 5 \
     --output <taskset.json>
   ```

2. **`fleet_wildlife_exact_launch.sh`** validates the full candidate file,
   taskset, optional imported proof ledger, host idleness, dependency version,
   and source revision before writing a durable launch ledger. It snapshots
   every input, deploys hash-pinned sources, then launches disjoint
   round-robin count shards with a heartbeat and terminal exit file:

   ```bash
   RULESET=aaaaa \
   FLEET_TAG=aaaaa_exact_tail_fleet1_20260723 \
   CANDIDATES=docs/v3/evidence/aaaaa_wildlife_candidates_deep_2026-07-23.json \
   COUNTS_FILE=cascadiav3/fleet/<taskset.json> \
   IMPORT_LEDGER=cascadiav3/fleet/<catalog-snapshot.json> \
   SOURCE_REVISION="$(git rev-parse HEAD)" \
   HOSTS="john2 john3 john4" \
   JOBS=2 SOLVER_WORKERS=4 \
   bash cascadiav3/scripts/fleet_wildlife_exact_launch.sh
   ```

   A tag is single-use. Existing local input state, a ledger, remote output,
   PID file, dependency mismatch, source mismatch, or non-idle host fails the
   launch closed. No process is stopped or replaced.

3. **`fleet_wildlife_exact_collect.sh <tag> status|collect`** reports remote
   PID, heartbeat age, terminal state, and durable-ledger progress. Collection
   is refused until every shard is terminal. It then verifies source,
   candidate, and taskset hashes; shard indices; worker exit codes; every
   board's count vector, connectivity, independent score, and task coverage;
   and writes `collection_manifest.json`. It does not mutate a live local
   catalog—the verified shard ledgers are imported only after its writer exits.

The arbitrary-card proof worker uses a validated five-second child poll by
default (`HEARTBEAT_INTERVAL`, allowed range 1–60 seconds). This is both its
heartbeat cadence and its maximum inter-row handoff latency. The earlier
30-second fixed poll dominated wall time once the near-tail exact queries
fell below four seconds.

Bounded maximization uses
`fleet_all_wildlife_bound_probe_worker.sh`. Its hash-pinned inputs are a
validated all-rules catalog plus an explicit
`all-wildlife-bound-probe-taskset-v1` file. Every task is one unresolved
ruleset/count pair, and every output records the verified witness (if any),
CP-SAT objective bound, refined analytical intersection, and resumable
identity. Collection is performed by
`tools/all_wildlife_bound_probe_collect.py`, which validates every bound and
witness before production-rescoring all selected boards.
Merged bound catalogs serialize `unresolved_count_upper_bounds` in exact
parallel with `unresolved_counts` and union inherited probe paths/hashes.
Subsequent tasksets must be based on that merged catalog, never on the older
analytical-only catalog; this makes repeated passes monotonic. The taskset
builder's `--top-frontier-above SCORE` mode deterministically selects every
count tied for its row's current sound upper above `SCORE`.

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
  Corpus generation is fine (fleet3/4/5 precedent). CPU-only exact
  pure-wildlife count-vector proofs are also allowed under the 2026-07-23
  authorization above; they do not use MPS scores and are independently
  validated before import.
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
- **john1 from another machine**: the legacy SSH entry names
  `john1@100.110.109.6`, but this run's orchestrator is john1 itself and must
  use the launcher's explicit local-host path. Do not loop SSH back to the
  local mini. Remote account/key repair is a separate infrastructure issue.
- **Bacalhau/MinIO fabric** (`infra/`, `tools/cluster_*`): working but
  intentionally NOT used for self-play — it runs linux/arm64
  containers with no Metal access. Don't route generation through it.
