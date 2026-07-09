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
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3.13 -m unittest discover -s cascadiav3/tests
```
- Homebrew rustc 1.85 shadows rustup 1.96 → pin RUSTC (and RUSTDOC for doc tests).
- `python3` is miniconda 3.9 (too old) → use `python3.13`.
- **The exporter has its own workspace** — `cargo check --workspace`
  does not compile it. Always build the exporter manifest explicitly
  before shipping exporter changes (a cfg(test)-gated fn once broke only
  the non-test build and silently killed a job chain).

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
- `pgrep -f` self-matches over ssh; use `ps aux | grep -v grep`.
- Monitors (Claude-side) run in zsh: `status` is a read-only variable;
  break patterns must not substring-match progress lines; scope error
  greps past the preflight-test section (benign BrokenPipe tracebacks).

## 4. Standard experiment workflow

Before using any baseline report, verify that its ruleset/config identity is
the corrected 2026-07-09 contract from `docs/v3/RULES_CONTRACT.md`. Reports
from the forced three-of-a-kind refresh era are historical only and must not
enter a paired verdict against corrected games.

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
