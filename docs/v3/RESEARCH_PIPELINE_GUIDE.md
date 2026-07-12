# Research Pipeline Guide — run every experiment end to end

Written 2026-07-12. This is the **operator manual** for the Cascadia v3
research pipeline. It assumes no prior context: follow it top to bottom and
you can read the current results, run any experiment, and add new ones —
without writing code. Rules of engagement live in
[`AGENTS.md`](../../AGENTS.md); this guide tells you which buttons exist
and the exact order to press them.

## 0. The one-paragraph mental model

The champion engine (cycle4, ~98.3 mean seat score; goal ≥100) is served by
a Rust binary ("the exporter") that runs Gumbel search with a PyTorch model
("the bridge") on the GPU box **john0** (`ssh john0`, repo at
`/home/john0/cascadia`). Every research idea is a **flag** on that binary.
Ideas are tested in two stages: a **screen** (~6 min: does the idea pick
better actions on 700 frozen positions, judged against mega-search truth?)
and, only if the screen passes its preregistered bar, a **gate** (~5 h: 100
paired self-play games, adopt only if the 95% CI excludes zero). Screens
rank; gates decide; nothing is adopted from a screen alone.

## 1. Read the current results (start here every time)

```bash
ssh john0 'bash /home/john0/cascadia/cascadiav3/scripts/morning_report.sh'
```

That prints one markdown digest: every chain's last heartbeat, every screen
analysis, every gate verdict. Cross-reference each artifact with its
**decision rule** in `cascadiav3/EXPERIMENT_LOG.md` (search for
"PREREGISTERED"). The rule was written before the data existed — apply it
literally; do not invent a new interpretation after seeing numbers.

To check what is running right now:

```bash
ssh john0 'for f in /home/john0/cascadia/cascadiav3/logs/*_20260712.log; do echo "== $f"; tail -1 "$f"; done'
```

## 2. Screen a new idea (~6 minutes of GPU)

A screen replays 700 stored champion positions ("the bank" —
`cascadiav3/reports/puzzle_bank_20260711_n4096/`, resolved by n4096
mega-search) and asks: with your flags on, how much worse/better are the
chosen actions, judged by the bank's values ("bank regret")?

```bash
# 1. Find the deployed revision (your SOURCE_REVISION):
ssh john0 'cat /home/john0/cascadia/cascadiav3/logs/exact_k1_deployed_revision.txt'
# 2. Run the screen (name it; put your flags in EXTRA_FLAGS):
ssh john0 'cd /home/john0/cascadia && SCREEN_NAME=my_idea \
  EXTRA_FLAGS="--gumbel-lcb-c 2.0" SOURCE_REVISION=<paste revision> \
  bash cascadiav3/scripts/run_bank_screen.sh'
# 3. Read it:
ssh john0 'cat /home/john0/cascadia/cascadiav3/reports/puzzle_screen_my_idea_analysis.md'
```

Compare `mean_regret` against the incumbent's `0.2351`
(`puzzle_screen_20260711_incumbent_analysis.json`). Lower is better.
**Before running, write a preregistration entry** in
`cascadiav3/EXPERIMENT_LOG.md`: what bar the screen must clear (e.g. "delta
≤ -0.010 → gate") — copy the format of the 2026-07-12 01:02 entry.

Available experiment flags (all combinable; see `exporter --help`):
`--gumbel-ghost-opponents`, `--gumbel-q-bias-correction`,
`--gumbel-lcb-c <c>`, `--gumbel-refresh-sample-divisor <k>`,
`--gumbel-c-visit/c-scale`, `--gumbel-sigma-norm <scheme>`,
`--gumbel-paired-rollouts`, `--gumbel-depth-rounds <n>`,
`--gumbel-determinizations <d>`, `--gumbel-n-simulations <n>`.

## 3. Gate a screened idea (~5 hours of GPU)

1. **Register a fresh seed block**: add a row to the table in
   `docs/v3/INFRASTRUCTURE.md` §5 (pick the next free 100-seed range —
   never reuse a block).
2. **Preregister** in EXPERIMENT_LOG: arms, block, the exact adopt/close
   rule (standard: adopt iff paired 95% t-CI excludes zero; for speed
   features: CI floor above -0.25 plus the wall saving).
3. **Run** (VARIED_KEYS = every search-provenance key that differs between
   the arms — include `n_simulations` if the arms use different n):

```bash
ssh john0 'cd /home/john0/cascadia && (nohup env \
  SOURCE_REVISION=<revision> GATE_NAME=my_idea_20260712 \
  FIRST_SEED=<block start> VARIED_KEYS="lcb_c" \
  CAND_FLAGS="--gumbel-lcb-c 2.0" \
  bash cascadiav3/scripts/run_paired_gate.sh \
  > cascadiav3/logs/gate_my_idea.log 2>&1 & echo $! > cascadiav3/logs/gate_my_idea.pid)'
```

4. **Read the verdict**: `cascadiav3/reports/gate_my_idea_20260712_verdict.md`
   — apply the preregistered rule, record the outcome in EXPERIMENT_LOG,
   update `docs/v3/README.md` status, commit and push.

## 4. Run several experiments as an unattended sequence

Write a queue file (JSONL; `#` comments allowed) in `cascadiav3/queues/`:

```jsonl
{"name": "my_screen", "script": "cascadiav3/scripts/run_bank_screen.sh", "env": {"SCREEN_NAME": "my_idea", "EXTRA_FLAGS": "--gumbel-lcb-c 2.0"}}
{"name": "my_gate", "script": "cascadiav3/scripts/run_paired_gate.sh", "env": {"GATE_NAME": "my_idea_20260712", "FIRST_SEED": "<registered>", "VARIED_KEYS": "lcb_c", "CAND_FLAGS": "--gumbel-lcb-c 2.0"}}
```

Copy it to john0 (`scp <file> john0:/home/john0/cascadia/cascadiav3/queues/`),
then launch detached:

```bash
ssh john0 'cd /home/john0/cascadia && (nohup env SOURCE_REVISION=<revision> \
  bash cascadiav3/scripts/run_experiment_queue.sh cascadiav3/queues/<file>.jsonl \
  > cascadiav3/logs/queue_run.log 2>&1 & echo $! > cascadiav3/logs/queue_run.pid)'
```

Stages run in order; a failed stage is logged and the queue continues;
re-launching skips completed stages (`queue_done_<name>` markers). Pause
between stages: `touch cascadiav3/logs/HOLD_experiment_queue` (remove to
resume). Full format: `cascadiav3/queues/README.md`. Ready-made conditional
gates (fill placeholders first): `queue_20260712_gates_template.jsonl`.

## 5. Deploy new code to john0 (only if you changed Rust/Python)

```bash
# On your machine, from the repo root, after commit + push + green tests:
REV=$(git rev-parse HEAD)
git archive --format=tar.gz -o /tmp/cascadia-main-$REV.tar.gz HEAD
scp /tmp/cascadia-main-$REV.tar.gz john0:/home/john0/cascadia/cascadiav3/logs/
ssh john0 "cd /home/john0/cascadia && tar -xzf cascadiav3/logs/cascadia-main-$REV.tar.gz \
  && echo $REV > cascadiav3/logs/exact_k1_deployed_revision.txt \
  && export PATH=\$HOME/.cargo/bin:\$PATH BLAKE3_NO_ASM=1 \
     CC=/home/john0/.local/bin/zig-cc \
     CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/home/john0/.local/bin/zig-cc \
  && cargo build --release --manifest-path cascadiav3/real-root-exporter/Cargo.toml"
```

**Never deploy while a job is running** (`ps aux | grep -E "gumbel|bridge"`
must be empty, or wait for the chain to exit). Tests before shipping:
`cargo test --manifest-path cascadiav3/real-root-exporter/Cargo.toml` and
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=cascadiav3/src python3 -m unittest
discover -s cascadiav3/tests`.

## 6. The non-negotiable discipline (why this works)

1. **Preregister before you peek** — rule and seed block written in
   EXPERIMENT_LOG before any output exists.
2. **Fresh seed blocks, registered, never reused**; verdict blocks touched
   once. Never read partial scores of a live arm.
3. **Screens rank, gates decide** — a screen result is never adoption
   evidence (see 07-11: a 7/7-positive screen died at its gate).
4. **One scientific job at a time on john0**; chain with waiters, don't
   race. Never kill a process without John's permission.
5. **Log as you go** — every run gets an EXPERIMENT_LOG entry (config,
   artifacts, verdict, decision); update README/CAMPAIGN_STATE at every
   material transition; commit and push immediately.
6. **Failures are results**: log them loudly and keep the invalid
   artifacts renamed in place (`*_invalid_*` precedent).

## 7. Troubleshooting

- Script dies instantly in a detached shell → PATH: prefix with
  `export PATH="$HOME/.cargo/bin:$PATH:/usr/lib/wsl/lib"` (nvidia-smi and
  cargo are off-PATH on john0; every chain script does this).
- "SOURCE_REVISION does not match" → the marker file
  (`cascadiav3/logs/exact_k1_deployed_revision.txt`) must equal what you
  pass; deploy first (§5).
- "refuses stale raw files" → a previous arm left durable ledgers; rename
  the old `*_raw_games` dir aside (never delete without archiving).
- Screen fails "action menu mismatch" → your screen and the bank must use
  the same ledger, stride, and menu cap; don't change `STRIDE`.
- Where is everything? Scripts: `cascadiav3/scripts/`; results:
  `cascadiav3/reports/`; logs/pids/HOLD files: `cascadiav3/logs/`;
  queues: `cascadiav3/queues/`; the experiment record:
  `cascadiav3/EXPERIMENT_LOG.md`; live state: `docs/v3/CAMPAIGN_STATE.md`.

## 8. What's live and what's next (as of 2026-07-12)

See [`../handoffs/handoff-2026-07-12.md`](../handoffs/handoff-2026-07-12.md)
for tonight's running chains and the morning decision checklist (ghost
gate → conditional n1024 confirmation; refresh gate → adoption decision;
coverage audit rerun). The strategic map — what's closed, what's queued,
and why — is `docs/v3/RESEARCH_LOG.md` §5 and the portfolio at
`claude_max_research_ideas.md`.
