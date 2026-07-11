# Cascadia Training Infrastructure & Cluster Runbook

How to operate the machines, run experiments, and not break in-flight
research. Companion to [RESEARCH_LOG.md](RESEARCH_LOG.md) (science) and
[CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) (live state — read its RESUME
HERE section first, always).

## 1. The machines

| Host | Hardware | Role | Access |
|---|---|---|---|
| **john0** | WSL box, CUDA GPU (~121 GB RAM) | ALL training, ALL gate batteries, primary generation | `ssh john0` |
| **john1–john4** | Apple M4 minis (MPS) | Training-data generation ONLY; john1 also hosts the web UI | `ssh johnN` (note: john1 user is `johnherrick`, john3 is `john3`) |

**Iron rules:**
- **john0 runs strictly one job at a time.** Chain jobs by waiting on the
  previous pid file, never in parallel.
- **Fleet (MPS) output is training data only — never gate batteries.**
  MPS/CUDA fp32 are not bit-identical; gates on minis are invalid.
  Fleet-generated shards are never auto-folded into training: cycle-5 was
  poisoned by fleet labels at high weight. Fold only after a paired
  safety trial at low weight (0.25 verified safe for n256/d4 labels).
- Batteries run TF32 **off**; generation runs TF32 **on**
  (`CASCADIA_BRIDGE_TF32=1`). bf16 serving is label-unsafe (26% action
  agreement) — never use it.

## 2. Building

**Local (Mac):**
```bash
# Plain `cargo`/`python3` are wrong on this machine:
RUSTC=~/.cargo/bin/rustc ~/.cargo/bin/cargo build --manifest-path cascadiav3/real-root-exporter/Cargo.toml
RUSTC=~/.cargo/bin/rustc ~/.cargo/bin/cargo check --workspace   # does NOT cover the exporter!
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src ./venv/bin/python -m unittest discover -s cascadiav3/tests
```
- Homebrew rustc 1.85 shadows rustup 1.96 → pin RUSTC (and RUSTDOC for doc tests).
- System `python3` is 3.9 (too old); the repo-root `venv` is Python 3.12 with
  the required Torch stack.
- **The exporter has its own workspace** — `cargo check --workspace`
  does not compile it. Always build the exporter manifest explicitly
  before shipping exporter changes (a cfg(test)-gated fn once broke only
  the non-test build and silently killed a job chain).

**Source provenance:** `cascadia-provenance` hashes Git-visible source under
its registered roots: tracked files plus untracked files not excluded by
`.gitignore`. It must never recursively hash ignored checkpoints, reports,
logs, venvs, or Rust `target/` output. Those generated trees exceed 11 GiB in
an active checkout, mutate during experiments, and are separately identified
by artifact manifests/checksums. Archive operation without Git uses the same
generated-directory exclusions. The stability and ignored-output invariance
tests pin this contract.

**john0:**
```bash
export PATH=$HOME/.cargo/bin:$PATH BLAKE3_NO_ASM=1 \
  CC=/home/john0/.local/bin/zig-cc \
  CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/home/john0/.local/bin/zig-cc
cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml
```
No system cc on john0 — zig-cc is the linker. Python: `source /home/john0/venvs/torch/bin/activate`.

**Minis:** `PATH=$HOME/.cargo/bin:$PATH cargo build --release ...`;
python via `~/cascadia/venv/bin/python` (torch with MPS).

## 3. Job pattern on john0

Jobs are bash scripts under `/home/john0/cascadia/cascadiav3/logs/`,
launched detached, chained by pid file:

```bash
# in the script: wait for the previous job
while kill -0 $(cat cascadiav3/logs/PREV_job.pid 2>/dev/null) 2>/dev/null; do sleep 60; done
# launching (parens matter — otherwise the pid file lands in $HOME):
ssh john0 'cd /home/john0/cascadia && (nohup bash cascadiav3/logs/X_job.sh > cascadiav3/logs/X_job.log 2>&1 & echo $! > cascadiav3/logs/X_job.pid)'
```
- Give every phase an idempotent skip guard (`[ -s report.json ] && skip`).
- **Never `>/dev/null` a cargo build inside a job** — a compile error
  dies silently under `set -e`.
- `john0` may expose Rust without a system `cc` or libc development package.
  `run_rules_20260709_rebaseline.sh` falls back to Zig `0.13.0` installed in
  the user account from the official tarball only after checking the pinned
  SHA-256; `zig-cc-linker.sh` is the checked-in Cargo/cc-rs adapter. Do not use
  the stale release binary when a target rebuild cannot be proven.
- Launch ssh sessions often exit 255 with a benign broken pipe; verify
  via pid file + log, not the ssh exit code.
- `pgrep -f` self-matches over ssh; use `ps aux | grep -v grep`. Quote
  pkill/pgrep patterns that appear in your own command line; prefer killing
  by pid file and chaining jobs on done-marker files.
- Kill order for a stuck job: job pid first, then
  `pkill -9 -f gumbel-selfplay-tensor-corpus`, then
  `pkill -9 -f torch_inference_bridge` (bridges launch via `sh -c`, so ppid
  checks are unreliable — pkill by name only when nothing else runs).
- macOS ships rsync 2.6.9: ONE remote source per command (use scp when
  fetching multiple remote paths).
- Monitors (Claude-side) run in zsh: `status` is a read-only variable;
  break patterns must not substring-match progress lines; scope error
  greps past the preflight-test section (benign BrokenPipe tracebacks).
- Never end a monitor's remote ssh command with `pgrep`/`grep` whose
  nonzero no-match exit makes the ssh look failed — end remote pipelines
  with `| tail -1` or capture output without exit-code coupling. One
  consolidated watchdog per work-wave. Monitors die with the session: on
  any resume, first check every in-flight job log.

**Waiter/chain pattern:** queued waiters live under `cascadiav3/logs/`
(script + pinned inputs — never `/tmp`), source
`cascadiav3/scripts/lib_waiter.sh`, and call `waiter_wait_for_pids` /
`waiter_gate` at every stage boundary. Pause an idle waiter by touching
`cascadiav3/logs/HOLD_<name>`; resume by removing it. Pausing a waiter
mid-stage requires `kill -STOP <pid>` / `kill -CONT <pid>` (user permission
required, per AGENTS.md). All waiters and watchers emit timestamped
heartbeats; `bash cascadiav3/scripts/campaign_status.sh` is the one-command
read-only snapshot that reports pid liveness, heartbeat freshness, HOLDs,
raw-ledger progress, GPU state, and fleet reachability.

**Durable-first evidence:** the Gumbel benchmark writes raw per-seed game
files to `<report>_raw_games/` beside `--out` by default and refuses a
directory holding stale raw files. `--ephemeral-raw-games` (temp dir) is for
throwaway smokes only. Never launch a scientific arm whose raw ledger lives
in `/tmp`.

**Checkpoint retention:** when a research line closes, prune its
`step_*.pt` intermediates on john0, keeping `best_locked_val*` + SWA +
final weights plus every manifest, metrics, and report file. Rejected
candidates keep their best checkpoint (opponent pool).

**Recovery:** resume training only from a matching manifest and source
identity; never reuse deleted worker-local artifacts as durable evidence.
If a long run is interrupted, fetch manifests and metrics before deciding
to resume, restart, or discard. (Bacalhau CPU-fabric scheduling, when used,
is documented in `docs/BACALHAU_USAGE.md` — the v3 campaign currently
orchestrates the fleet by direct ssh instead.)

## 4. Standard experiment workflow

Before using any baseline report, verify that its ruleset/config identity is
the corrected 2026-07-09 contract from `docs/v3/RULES_CONTRACT.md`. Reports
from the forced three-of-a-kind refresh era are historical only and must not
enter a paired verdict against corrected games.

CUDA shared-bridge concurrency is calibrated with
`cascadiav3/scripts/run_cuda_concurrency_probe.sh`, never by editing a live
gate. It runs matched jobs12/16/24 arms sequentially, profiles the GPU once per
second, validates complete traces and parity, and writes an advisory verdict.
The script is resumable only when every reused arm matches rules, revision,
seeds, search, topology, device, ledger counts, and telemetry length. Apply a
recommended jobs change only in a later reviewed commit; the probe itself
does not modify launchers or defaults.

1. **Benchmark/battery** (100 games, paired seeds):
```bash
python -m cascadiav3.torch_cascadiaformer_gumbel_benchmark \
  --manifest <ckpt.manifest.json> --device cuda \
  --first-seed 2026995000 --games 100 --jobs 12 --batch-runner \
  --gumbel-n-simulations 256 --gumbel-top-m 16 \
  --gumbel-determinizations 4 --gumbel-market-decision-samples 8 \
  --gumbel-blend-weight 0.5 --source-revision "$(git rev-parse HEAD)" \
  --control none --experiment-id <tag> \
  --out cascadiav3/reports/gumbel_<tag>.json
```
2. **Verdict** (paired t-CI vs a baseline report on the same seeds):
   `python3 /tmp/pair_verdict.py <cand.json> <base.json> <label>` on john0
   (pattern: candidate_per_seed → per-seed deltas → `paired_delta_stats`,
   promotion requires the 95% CI to exclude zero at ≥100 games).
   Key baselines: `gumbel_cycle4_gate_n256.json` (96.95, n256/d4),
   `gumbel_probe4_confirm_n1024_d16.json` (98.28, champion config),
   `gumbel_distq_k8_n256.json` (97.38), `gumbel_distq_k8_n1024_d16.json` (98.40).
   For the corrected 2026-07-09 four-arm battery, run
   `python -m cascadiav3.compare_rules_rebaseline --source-revision <rev>`;
   it rejects rules, revision, seed, or budget mismatches before producing the
   paired scalar-vs-distq and scaling verdicts.
3. **EI cycle**: sed-derive from
   `logs/gumbel_selfplay_cycle6_job.sh` (generation n512/d8 w1.0 →
   filter top-64 → materialize relation tail → train M 25k steps with
   `--max-example-passes 4` guard → runbook validation). Trainer is
   mmap-fast (~0.23 s/step); training is ~25 min, generation is the
   overnight cost (~10 h for 1,375 seeds).
4. **Log it**: EXPERIMENT_LOG.md entry per event; RESEARCH_LOG.md verdict;
   CAMPAIGN_STATE.md RESUME HERE updated whenever in-flight state changes.

**Serving env knobs:** `CASCADIA_CGAB_FUSED=1`,
`CASCADIA_EVAL_CELL_BUDGET=16777216`, shard mmap on by default
(`CASCADIA_SHARD_MMAP=0` disables), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## 5. Seed-range registry (do not collide)

| Range | Owner |
|---|---|
| 2026995000–099 | paired gate/probe batteries (gumbel) |
| 2026994000+ | no-search batteries |
| 2026790000 / 2026890000 | cycle-6 train/val corpus |
| 2026800000 / 2026900000 | cycle-7-table (staged, unlaunched) |
| 2026810000 / 2026910000 | distq EI-1 train/val corpus |
| 2026781000+ | fleet3 shards |
| 2026805000+ | fleet4 (staged) · 2026815000+: fleet5 shards |
| 2027070900–0999 | corrected-rules rebaseline battery (all arms, scalar + distq) |
| 2027071360–2003 | 07-09 engineering smokes (exact-K1 1360+, market samples 1500+, S calibration 1700+, q-risk 1900+, shared-batch 2000+) |
| 2027071400–1499 | f35 post-chain paired gates at n256/d4: stage 1 exact-K1 (baseline + K1 arms) and stage 4 market sample-4 (candidate arm paired against the stage-1 samples=8 baseline report) |
| 2027071500–1599 | worlds-allocation screen: cycle4 n256 det4 vs det8, K1 on (preregistered 07-10) |
| 2027071600–1699 | RESERVED (conditional): n1024 det16 vs det32 confirmation, only if the screen is CI+ |
| 2027072100–2124 | R0.1 sigma-calibration sweep screens: 8 paired arms at n256/d4, K1 on (preregistered 07-10 21:55) |
| 2027072200–2299 | RESERVED (conditional): R0.1 sigma confirm gate at n256, winner vs incumbent — touched once, only if the screen floor passes |
| 2027073000–3129 | pairwise label audit (3000+) and v3 fit/selection/validation corpus (3100–29) |
| 2027073300–3301 | parallel leaf-rollout screens |
| 2027073400–3447 | jobs12/16/24 CUDA concurrency calibration (queued) |
| 2027073500–3529 | structured-Q pilot: fit / LR-selection / untouched verdict |
| 2027073600–3749 | structured-Q fit expansion (quarantined) |
| 2027073750–3809 | structured-Q reserves: selection / verdict / replication |

## 6. Fleet operations (john1–4)

Repo at `~/cascadia`, checkpoints under
`cascadiav3/checkpoints/<name>/` (rsync manifest + weights.pt only).
Generation pattern: `~/cascadia/fleet5_gen.sh` (n256/d4, w0.75, 3 model
sessions, ~150 seeds/host, `--gumbel-selfplay-tensor-corpus`). Launch
detached with pid file like john0. Shards land at
`~/cascadia/fleetN_shard_johnN.npz` — fetch to john0, filter/materialize,
then safety-trial before any fold-in.

## 7. Web UI (john1)

- Server: `~/cascadia/webui_run.sh` → `cascadia-api` on **0.0.0.0:8787**
  (Tailscale: http://100.110.109.6:8787), serving `apps/web/dist` +
  `/api/v1/*`.
- Champion strengths spawn `champion_server.sh` (exporter
  `--gumbel-suggest-server`, distq checkpoint, MPS) lazily via
  `CASCADIA_CHAMPION_CMD`; per-request overrides give the
  champion-deep (n1024/d16) tier from the same loaded model.
- Deploy: build `apps/web` locally (`npm run build`), rsync `dist`,
  rebuild `cargo build --release -p cascadia-api` on john1 if the API
  changed, bounce via `webui.pid`. Suggest server picks up new exporter
  binaries on respawn (`pkill -f gumbel-suggest-server`).
- Fleet generation on john1 is paused while the UI is in active use
  (MPS contention); relaunch `fleet5_gen.sh` when done.

## 8. Where everything is documented

- `docs/v3/RESEARCH_LOG.md` — **the experiment record**: architecture,
  every direction tried, verdicts, scaling laws, future directions.
- `cascadiav3/EXPERIMENT_LOG.md` — chronological per-run entries.
- `docs/v3/CAMPAIGN_STATE.md` — live state, in-flight jobs, decision tree.
- `docs/v3/GUMBEL_SELFPLAY_CAMPAIGN.md` — campaign strategy/phases.
- `docs/v3/INFRASTRUCTURE.md` — this file.
