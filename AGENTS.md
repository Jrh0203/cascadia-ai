# Cascadia AI — Agent Rules of Engagement

## Goal

Build a superhuman Cascadia board-game AI, currently through the Cascadia v3
transformer stack. The gate: mean seat score ≥ 100 over 1,000 games of
4-player self-play.

## Standards

- Boil the ocean: ship the complete fix with tests and documentation when it is
  within reach.
- Search before building. Test before shipping.
- No hacks or undocumented shortcuts. If an unavoidable compromise remains,
  document it in `docs/TECH_DEBT.md` with cause, proper fix, and blast radius.

## Source of truth

- **Start every session at `docs/v3/README.md`** — the authoritative status
  and doc map. Then `docs/v3/CAMPAIGN_STATE.md` RESUME HERE for live detail.
- Current `main` and live artifacts outrank any dated handoff snapshot or
  remembered state. Before resuming a claimed in-flight job, verify the live
  state yourself: `bash cascadiav3/scripts/campaign_status.sh` gives the
  one-command snapshot (git sync, pids, heartbeats, HOLDs, ledger progress,
  GPU, fleet).
- Pre-v3 material lives at tag `archive/pre-v3-repo-cleanup-2026-07-01`;
  pruned docs at branch `archive/doc-prune-2026-07-09`. Off-repo data
  archives live at `john0:~/cascadia-archive/` (hash-verified tarballs).

## Documentation discipline (log as you go, never batch)

- **Every run gets a `cascadiav3/EXPERIMENT_LOG.md` entry at the time it
  happens**: purpose, exact config (source revision, rules ID, seeds, search
  settings, hosts), artifacts with SHA-256, result, and the decision taken.
  Failed, invalidated, and discarded runs are logged too — say why.
- `docs/v3/RESEARCH_LOG.md` gets the consolidated verdict whenever a
  direction opens, closes, or changes rank.
- `docs/v3/CAMPAIGN_STATE.md` RESUME HERE and the status section of
  `docs/v3/README.md` are updated at **every material transition** — a job
  starting/finishing, a verdict landing, a blocker appearing. Durable
  Markdown must never lag live conclusions.
- At session end with jobs in flight, write a dated
  `docs/handoffs/handoff-YYYY-MM-DD.md` snapshot (live PIDs, hashes,
  blockers, resume checklist) and link it from `docs/v3/README.md`.

## Scientific discipline

- **Preregister before you peek.** Gates, thresholds, and seed roles are
  fixed before any candidate output exists. Hyperparameter selection and the
  final verdict use disjoint seed blocks; verdict blocks are touched exactly
  once.
- **Promotion** requires ≥ 100 paired games with the 95% CI excluding zero,
  on corrected-rules identity, with clean provenance. Validation loss, smoke
  scores, retention metrics, or a busy process are never strength evidence.
- **Never read partial scores of a live arm**, and never adapt an experiment
  to them.
- **Fail closed.** Hash-pin every artifact; refuse mismatched
  rules/source/seed/search provenance; an incomplete ledger is not published.
- **Closed directions stay closed** without materially new evidence — check
  `RESEARCH_LOG.md` §5 and `RADICAL_DIRECTIONS.md` before proposing work.
- **Seeds are allocated, never reused**: pick fresh disjoint blocks and
  record them in the `INFRASTRUCTURE.md` seed registry.

## Operational safety

- **Never kill or restart a process without explicit user permission.**
  Don't disturb live scientific chains; coordinate with queued waiters
  instead of racing them.
- **Durable-first evidence.** No scientific artifact's only copy may ever
  live under `/tmp` or a process temporary directory. Benchmark runners
  write raw per-seed ledgers to a durable `<report>_raw_games/` directory
  beside the report (the CLI default; `--ephemeral-raw-games` is for
  throwaway smokes only). Watchers and mirrors are redundancy, never the
  primary durability mechanism.
- **Waiters are pausable and reboot-reconstructible.** Queued waiter/chain
  scripts and their pinned inputs live under `cascadiav3/logs/`, never
  `/tmp`. They check a `HOLD_<name>` file before every stage (helpers in
  `cascadiav3/scripts/lib_waiter.sh`) so work can be inserted without
  killing the chain.
- **Background helpers write heartbeats.** Every watcher, mirror, and waiter
  emits timestamped heartbeat lines; status checks verify heartbeat
  freshness, not just pid liveness — a silent helper is presumed dead.
- **Checkpoint retention on line-close.** When a research direction closes,
  prune its `step_*.pt` intermediates, keeping `best_locked_val` + SWA +
  final weights and every manifest/metrics/report file. Rejected candidates'
  best checkpoints stay (opponent-pool material).
- john0 runs strictly one scientific job at a time. The Mac minis
  (john1–john4) generate training data only — never gates; MPS numbers are
  never promotion evidence. Fleet shards are never auto-folded into training.
- Batteries run TF32 **off**; generation may run TF32 on. bf16 serving is
  label-unsafe.
- Destructive file operations follow **archive-then-delete**: hash-verify the
  archived copy (convention: `john0:~/cascadia-archive/<date>/` with
  SHA256SUMS + README) before removing the original. Never clean another
  session's operational files without checking what owns them.
- `cargo check --workspace` does **not** cover the exporter — build
  `cascadiav3/real-root-exporter` explicitly, and never silence build output
  inside job scripts.
- Job/monitor patterns, kill order, and remote build environments are in
  `docs/v3/INFRASTRUCTURE.md` — follow them exactly.

## Git rules

- `HEAD == origin/main` with a clean worktree is the resting state: commit
  and push documentation/state updates as they happen, in small focused
  commits.
- Keep generated data, checkpoints, reports, tensor shards, dependency
  directories, and build outputs out of Git. Small curated evidence is fine.
- Removals of tracked content get an archive tag/branch pointer first, and
  the commit message says how to recover.

## Engineering rules

- Prefer packed tensor `.npz` paths for real training data (schema v4 for new
  Gumbel generation). Keep JSONL only for tiny audit fixtures.
- Radius 6 is the default public board fast path for CascadiaFormer. Exact
  overflow is required for states outside the disk.
- Serving must rank by
  `exact_afterstate_score_active + predicted_score_to_go`.
- Before shipping code, run the relevant subset of:

```bash
cargo check --workspace
cargo test --workspace
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```
