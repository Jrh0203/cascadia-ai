# Experiment queues

Config-driven sequential experiment orchestration for john0. A queue file
lists preregistered experiment stages; `run_experiment_queue.sh` runs them
one at a time (AGENTS.md: one scientific job at a time) so the GPU never
idles between experiments, with no interactive session required.

## Queue file format (JSONL)

One JSON object per line. Blank lines and full-line `#` comments are skipped.

```jsonl
{"name": "worlds_confirm_resume", "script": "cascadiav3/scripts/run_worlds_confirm.sh", "env": {}}
{"name": "ghost_screen", "script": "cascadiav3/scripts/run_bank_screen.sh", "env": {"SCREEN_NAME": "ghost_opponents", "EXTRA_FLAGS": "--gumbel-ghost-opponents"}}
```

- `name` — unique per file, matches `[A-Za-z0-9][A-Za-z0-9._-]*`; it names
  the stage's log (`cascadiav3/logs/queue_<name>.log`), pid file
  (`queue_<name>.pid`), and done marker (`queue_done_<name>`).
- `script` — path relative to the repo root; must exist at launch time.
- `env` — extra environment for the stage. Keys match `[A-Z][A-Z0-9_]*`;
  `SOURCE_REVISION` is reserved (the runner pins it for every stage).
  Values may be strings or numbers; quoting is handled for you.
- `notes` — optional free-text annotation (preregistration pointer etc.).

The whole file is validated **before any stage runs** (fail closed): a
malformed line, duplicate name, bad env key, or missing script rejects the
entire queue. Validate locally before shipping:

```bash
PYTHONPATH=cascadiav3/src python3 -m cascadiav3.experiment_queue cascadiav3/queues/<file>.jsonl
```

## Launching on john0

```bash
ssh john0 'cd /home/john0/cascadia && (nohup bash cascadiav3/scripts/run_experiment_queue.sh cascadiav3/queues/<file>.jsonl > cascadiav3/logs/queue_<date>.log 2>&1 & echo $! > cascadiav3/logs/queue_<date>.pid)'
```

with `SOURCE_REVISION=<rev>` exported in the remote command (the runner
refuses to start unless it matches `HEAD`, or the deployed-revision marker
on snapshot hosts). The runner writes its own pid to
`cascadiav3/logs/queue_runner.pid` and refuses to start while another queue
runner is live. Heartbeats land in the nohup log every `WAITER_POLL_SECONDS`
(default 60) — a silent runner is presumed dead.

## Pause / resume

Touch `cascadiav3/logs/HOLD_experiment_queue` to pause the queue before its
next stage; the runner heartbeats "paused by ..." while holding. Remove the
file to resume. A stage that is already running is not interrupted — use
that stage's own `HOLD_<name>` file (e.g. `HOLD_worlds_confirm`) for
intra-stage gates.

## Done markers (idempotent resume)

A stage that finishes successfully writes
`cascadiav3/logs/queue_done_<name>` (timestamp + source revision). On any
rerun of the queue — after a failure, a reboot, or adding stages to the
file — completed stages are skipped and work resumes at the first stage
without a marker. Delete a stage's marker to force a rerun. A failed stage
writes no marker: the failure is heartbeat-logged with its exit code and
**the queue continues to the next stage**; the runner prints a
stage → COMPLETE/FAILED/SKIPPED summary table at the end and exits 1 if
anything failed.

## Reading results

The queue log records orchestration only. Each stage appends its full
stdout/stderr (build output included) to `cascadiav3/logs/queue_<name>.log`,
and the science lands where the stage script puts it — report/verdict
artifacts under `cascadiav3/reports/` (e.g. `<tag>.json`, `<tag>_verdict.json`,
`<tag>.md` summaries) plus the stage's own `<tag>_complete.json` marker under
`cascadiav3/logs/`. Every stage still gets its own
`cascadiav3/EXPERIMENT_LOG.md` entry at launch time, per AGENTS.md.

## Files

- `queue_20260712_example.jsonl` — worked two-stage example (worlds-confirm
  resume, then the ghost-opponents bank screen). Note: stage 2's
  `run_bank_screen.sh` lands with the ghost-opponents workstream; until it
  exists, validation fails closed on this file by design.
