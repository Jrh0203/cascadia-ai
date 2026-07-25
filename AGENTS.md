# Cascadia AI — Working Agreement

## Goal

Build the strongest Cascadia player we can. The current engineering target is
a mean seat score of at least 100 over 1,000 four-player self-play games, but
that number is a progress measure rather than a bureaucracy.

## How we work

- Optimize for playing strength, iteration speed, and understandable results.
- Search before building. Test important behavior before shipping it.
- Prefer direct, maintainable implementations over campaign machinery.
- Fix rules bugs and engineering bugs when found. Record the practical effect
  in the commit or current status note; a new rules identity or formal contract
  process is not required.
- Use any available machine for any useful work. Run independent jobs in
  parallel when resources permit.
- Inspect live metrics and partial results whenever they help make a better
  decision. Stop, extend, retune, or redirect runs based on what is learned.
- Reuse seeds freely. A seed is an input, not a scarce registered asset.
- Promote the strongest checkpoint supported by the evidence available. Paired
  games and confidence intervals are useful analysis tools, not mandatory
  gates.

## What is not required

The project does not use an internal-adversary or chain-of-custody model.
Future work must not be blocked on:

- preregistration, sealed holdouts, fixed decision rules, or no-peek rules;
- source, artifact, manifest, dataset, or executable hash verification;
- receipts, provenance ledgers, seed registries, or rules/scientific IDs;
- HOLD files, mandatory waiters, mandatory heartbeats, or one-job limits;
- host-role restrictions, bit-identical cross-host output, or approval to
  inspect, stop, restart, or retune our own jobs;
- archival ceremony before replacing obsolete generated artifacts.

Existing historical reports may contain those fields. They are inert history,
not current requirements. Hashes that are algorithmic data—game-state IDs,
action IDs, cache keys, binary-format fingerprints, content-addressed object
keys, or signatures required by an external protocol—may remain because they
serve computation/interoperability rather than trust enforcement.

## Source of truth

- Start at `docs/v3/README.md`, then `docs/v3/CAMPAIGN_STATE.md`.
- Current code, current artifacts, and live processes outrank stale notes.
- Keep the live status concise: what is running, the latest useful result, the
  current best checkpoint, and the next strength-improving action.
- `cascadiav3/EXPERIMENT_LOG.md` and older research documents are historical
  notebooks. Updating them is optional.

## Operational defaults

- Check the process list before launching expensive work so we do not
  accidentally oversubscribe a machine.
- Ordinary logs, metrics, numbered checkpoints, `latest`, and `best` are
  enough. Runs should resume from a loadable checkpoint when practical.
- Generated data may live wherever it is convenient and durable enough for
  its expected lifetime. Avoid `/tmp` for long jobs only because reboots erase
  it.
- Destructive operations still need an exact target. Do not delete unrelated
  user data or another live job's files.
- TF32, bf16, batching, compilation, and concurrency are performance knobs.
  Benchmark them and use whichever produces the best end-to-end result.

## Engineering invariants

- The canonical Rust game engine defines legal play and scoring.
- Serving ranks actions by
  `exact_afterstate_score_active + predicted_score_to_go`.
- Radius 6 is the normal CascadiaFormer fast path; states outside it must still
  be handled correctly.
- Training must reject non-finite losses or gradients and write loadable
  checkpoints.
- Prefer packed tensor `.npz` data for large training corpora. JSONL is fine
  for small diagnostics.

Run the relevant tests for the files changed. The broad validation commands
are:

```bash
cargo check --workspace
cargo test --workspace
cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src uv run python -m unittest discover -s cascadiav3/tests -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=python:tools uv run pytest -q tests/cluster_unit tools/test_cluster_*.py
```
