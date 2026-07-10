# Cascadia V3 — Source of Truth

**This file is the authoritative entry point for the state of the project.**
Link this in handoffs. It is updated at every material transition, together
with [CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) (live operational detail) and
`cascadiav3/EXPERIMENT_LOG.md` (chronological evidence). If this file and a
dated handoff snapshot disagree, current `main` wins.

Cascadia v3 is the transformer-based training and search stack for pushing
four-player Cascadia beyond the previous neural/search plateau: CascadiaFormer
over packed expert tensors with Gumbel search-supervised action values.

## Status at a glance (updated 2026-07-10 10:30 EDT)

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
- **Exact-K1: ADOPTED (07-10 ruling); exact K2 closed.** John ruled to keep
  K1 and exclude the one concurrency-divergent seed by declaration. Verdict
  on the 99 causally-valid pairs: `-0.0379`, CI `[-0.0859,+0.0101]` —
  score-neutral with a `28.99x` exact-frontier speedup. Seat-0 delta was
  exactly zero: the model already picks score-optimal final actions, so K1
  is pure speed. `--gumbel-exact-endgame-turns 1` is now the
  serving/benchmark default; K2+ plies stay on model inference.
- **Structured-Q head pilot: FAILED its preregistered kill test (07-10).**
  Selected-final RMSE `4.1573` vs teacher `3.5520` (`-17.04%` against a
  required `+10%`); paired CI wholly on the wrong side of zero. Retention
  gates passed (the decomposed head is the better completed-Q predictor)
  but per preregistration the direction is **closed**: no full-model run,
  no gameplay; the 12,000-root expansion and reserve holdouts stay
  quarantined. See RESEARCH_LOG §4.9.
- **Smaller-model serving: closed on john0 CUDA (07-10).** The packed
  throughput probe measured S at only `1.9x` M (XS `2.0x`, tiny `2.8x`) —
  far under the MPS ratios and under the >3x already shown insufficient.
- **Market sample-4 gate: FAIL (07-10) — sample-8 stays.** Paired delta
  `-0.1575`, CI `[-0.4684,+0.1534]`: the floor breaches the preregistered
  `-0.25` noninferiority margin despite a `1.575x` speedup. The comparator's
  trace-frontier premise was invalid for this knob (42/100 seeds diverge
  pre-exposure by mechanism); it now verdicts on score+speedup only.
- **Live now:** the approved one-seed d20 replays (cycle4 `2027070908`
  running, distq `2027070962` next), then the preregistered worlds screen
  (n256 det4 vs det8, block `2027071500..1599`), then the stage-5
  jobs12/16/24 concurrency probe relaunch.
- **Recovery CLOSED (07-10):** both one-seed d20 replays validated
  bit-exact and installed; both 100-row category ledgers exist. The paired
  category attribution (distq minus cycle4, n1024/d16) is **flat in every
  category** (wildlife `+0.145` ns, habitat `-0.050` ns, nature `-0.008`
  ns) — no hidden mechanism trade behind the head tie. Canonical artifact
  set harvested to `cascadiav3/reports/rules_20260709_rebaseline_complete/`.
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
- Exact final-personal-turn evaluation is the serving/benchmark default
  (`--gumbel-exact-endgame-turns 1`, adopted 2026-07-10, score-neutral,
  ~29x faster frontier); K2 and deeper plies stay on model inference.
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
