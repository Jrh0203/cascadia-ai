# Cascadia V3 — Source of Truth

**This file is the authoritative entry point for the state of the project.**
Link this in handoffs. It is updated at every material transition, together
with [CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) (live operational detail) and
`cascadiav3/EXPERIMENT_LOG.md` (chronological evidence). If this file and a
dated handoff snapshot disagree, current `main` wins.

Cascadia v3 is the transformer-based training and search stack for pushing
four-player Cascadia beyond the previous neural/search plateau: CascadiaFormer
over packed expert tensors with Gumbel search-supervised action values.

## Status at a glance (updated 2026-07-10 08:40 EDT)

- **Goal:** mean seat score **≥ 100 over 1,000 games** of 4-player self-play.
- **Rules boundary:** the 2026-07-08/09 corrections (optional three-of-a-kind
  refresh as a real decision → chance → draft; wildlife returns to the bag
  before refill) compatibility-broke every earlier number. All promotion
  evidence must carry rules ID `..._rules_2026_07_09`. See
  [RULES_CONTRACT.md](RULES_CONTRACT.md).
- **Corrected-rules scoreboard (100 games, seeds 2027070900..99), COMPLETE:**
  greedy `87.5450` → no-search policy `91.8425` → cycle4 n256/d4 `97.0675`
  → distq-k8 n256/d4 `97.3075` (+0.24 ns) → cycle4 n1024/d16 `98.2975` →
  **distq-k8 n1024/d16 `98.3850` (+0.0875 ns vs scalar)**. Verdict:
  **cycle4 scalar retained as champion** — the heads are statistically tied
  at high budget (reproducing the legacy tie); distq remains strictly the
  better low-budget server. High-budget scaling is CI+ within both heads
  (~+1.1 to +1.2). Gap to 100: ~-1.6.
- **Exact-K1 gate: arms complete and flat (97.265 baseline / 97.235 K1);
  verdict BLOCKED** — the causal comparator failed closed because one seed
  of 100 diverged pre-K1 (jobs12 concurrency numerics; the other 99 are
  causally perfect). Needs a methodology ruling: declared one-seed
  exclusion, lower-concurrency rerun, or invalid-as-run. See EXPERIMENT_LOG
  2026-07-10 08:35.
- **Live now:** f35 post-chain stages 2-5 (`postchain_resume2_f35b0d0b`) —
  structured-Q frozen-head pilot (training) → CUDA packed-throughput probe
  → market sample-4 gate → jobs concurrency calibration. Pause with
  `touch cascadiav3/logs/HOLD_postchain_resume` on john0.
- **Open recovery items:** two one-seed d20 replays — scalar `2027070908`
  and distq `2027070962` (lost to the pre-durable-first temp-dir race). The
  category-mechanism verdict is blocked until both 100-row category ledgers
  exist; the totals verdict above is not blocked.
- **Central scientific finding:** evaluation noise is the binding constraint
  (median decision SNR ≈ 1; ~46% of decisions noise-flippable). Exactness
  beats estimation wherever practical. Ranked directions and closed
  directions: [RESEARCH_LOG.md](RESEARCH_LOG.md) §5.

For exact live PIDs, artifact hashes, and the resume checklist, read
[CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) RESUME HERE first, then the latest
dated handoff ([handoff-2026-07-09.md](../handoffs/handoff-2026-07-09.md) — a
timestamped snapshot, weaker than current `main`).

## Read order for a fresh session

1. This file.
2. [CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) — live state; RESUME HERE first.
3. `cascadiav3/EXPERIMENT_LOG.md` — chronological evidence, newest entries.
4. [RESEARCH_LOG.md](RESEARCH_LOG.md) — consolidated verdicts and ranked
   directions.
5. [RULES_CONTRACT.md](RULES_CONTRACT.md) — rules identity and compatibility
   boundary.
6. [INFRASTRUCTURE.md](INFRASTRUCTURE.md) and
   [cascadiav3/README.md](../../cascadiav3/README.md) — operations and
   command entry points.
7. [RADICAL_DIRECTIONS.md](RADICAL_DIRECTIONS.md),
   [ARCHITECTURE.md](ARCHITECTURE.md),
   [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) — architecture bets and
   methodology.

## Canonical docs

- [Agent Rules of Engagement](../../AGENTS.md): how agents interact with this
  repository — logging, scientific, operational, and git discipline.
- [Campaign State](CAMPAIGN_STATE.md): live working state — in-flight jobs,
  blockers, decision tree.
- [Research Log](RESEARCH_LOG.md): **the experiment record** — architecture,
  every direction tried with verdicts, scaling laws, decision-SNR
  measurement, and ranked future directions.
- [Rules Contract](RULES_CONTRACT.md): official rules identity and the
  resulting baseline compatibility boundary.
- [Radical Directions](RADICAL_DIRECTIONS.md): speculative architecture-level
  bets, each judged against the campaign's measured constraints.
- [Infrastructure Runbook](INFRASTRUCTURE.md): how to operate john0 + the
  mac-mini fleet — builds, job patterns, batteries/verdicts, seed registry,
  fleet rules, recovery, and the web UI deployment.
- [Gumbel Self-Play Campaign](GUMBEL_SELFPLAY_CAMPAIGN.md): the active
  100-point plan — phases, gates, and decision branches.
- [Architecture](ARCHITECTURE.md): model shape, tokenization, relation bias,
  serving semantics, and literature basis.
- [Training Pipeline](TRAINING_PIPELINE.md): data formats, objectives, expert
  iteration, checkpointing, and promotion gates.
- [Performance](PERFORMANCE.md): measured loader/training/gameplay facts.
- Latest handoff: [handoff-2026-07-09.md](../handoffs/handoff-2026-07-09.md).

The implementation package lives in
[cascadiav3/README.md](../../cascadiav3/README.md).

## Standing contracts (durable, not run-by-run status)

- Real training data is packed `.npz` tensor shards (schema v4 for new Gumbel
  generation); JSONL only for tiny audit fixtures.
- Serving must rank by
  `derived_final_q = exact_afterstate_score_active + predicted_score_to_go`.
- Promotion requires ≥100 paired games with a 95% CI excluding zero — never
  validation loss, smoke scores, or process activity.
- john0 runs one scientific job at a time; the Mac minis generate training
  data only and never host gates; fleet shards are never auto-folded.
- Batteries run TF32 off; generation may run TF32 on.
- Trust streamed artifacts, reports, manifests, and paired verdicts — never a
  busy process alone.
- `cargo check --workspace` does **not** cover the exporter; build
  `cascadiav3/real-root-exporter` explicitly.

## Historical recovery

Pre-v3 material (v1/v2 engines, MLX package, old web app, rejected
experiments) was removed from `main` on 2026-07-01; superseded docs were
pruned on 2026-07-09. Recover via:

```bash
git show archive/pre-v3-repo-cleanup-2026-07-01:<path>   # tag, 07-01 cleanup
git show archive/doc-prune-2026-07-09:<path>             # branch, 07-09 prune
```

Off-repo data archives (dead v1/v2 weights, rules-broken fleet shards) live at
`john0:~/cascadia-archive/` with SHA256SUMS and a README.
