# Cascadia V3 — Source of Truth

**This file is the authoritative entry point for the state of the project.**
Link this in handoffs. It is updated at every material transition, together
with [CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) (live operational detail) and
`cascadiav3/EXPERIMENT_LOG.md` (chronological evidence). If this file and a
dated handoff snapshot disagree, current `main` wins.

Cascadia v3 is the transformer-based training and search stack for pushing
four-player Cascadia beyond the previous neural/search plateau: CascadiaFormer
over packed expert tensors with Gumbel search-supervised action values.

## Status at a glance (updated 2026-07-12 10:50 EDT)

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
- **Worlds screen: CI+ (07-10 19:15).** det4 `97.1425` vs det8 `97.5650` at
  n256, paired `+0.4225`, CI `[+0.1045, +0.7405]` — first CI+ search-shape
  result under corrected rules. Caveat: det8 cost `1.495x` mean decision
  time at n256 (worlds reduce eval dedup), so the knob is not wall-free.
- **Worlds det16/det32 n1024 confirmation: PAUSED by ruling (07-10 21:10),
  not closed.** Not wall-matched (scaling predicts ~+0.3 for its ~1.5x
  wall) and lowest conviction-per-GPU-hour next to the portfolio; killed
  cleanly with zero completed det16 games (nothing durable lost). Block
  `2027071600..1699` stays reserved for a future rerun.
- **R0.1 sigma calibration: CLOSED (07-11).** 8-arm screen was 7/7
  positive (best c_scale 0.25/topk:8 at +0.70) but the preregistered
  100-seed disjoint-block confirm returned `-0.2325`, CI
  `[-0.5440, +0.0790]` — a shared-baseline screen artifact; lesson and
  knobs (now on `main`, bit-identical defaults) in RESEARCH_LOG §4.10.
- **Concurrency probe: RESOLVED (07-11) — jobs12 retained.** Throughput
  flat across jobs12/16/24 (best `1.051x`), GPU ~66% util everywhere: the
  shared bridge is the bound; R2.4 throughput work moves bridge-side.
  Comparator now classifies cross-jobs trajectory forks descriptively
  (divergence-frontier fix).
- **R0.2 paired rollouts: CLOSED at the preregistered floor (07-11)** —
  gap-variance `-4.4%` vs required `-20%`; secondary CI+ (selection flip
  rate `0.466 → 0.424`) earns it a seat in a future composed serving-v2
  gate. **R1.1a: no cheap cooperative points at the root** (table delta
  `-0.03`/decision at own-Q parity; R1.1b/c deprioritized).
- **R3.6 ceiling probe: DECELERATING (07-11 18:41).** n4096/d16 paired
  `+0.21` vs the stored champion arm (CI `[-0.59, +1.01]`), ~1/3 of the
  log-linear +0.615 — the selfish scaling lane plausibly tops out under
  100. Portfolio reweights to **R1.2 ghost opponents** and **R1.4
  training densification**; velocity stack multiplies.
- **R2.1 puzzle bank: ACCEPTED (07-12 01:15).** 700 champion-ledger roots
  resolved at n4096/d16x2 are the frozen screening truth; incumbent bank
  regret `0.2351`, cross-checked against gate truth. Screens now cost ~6
  min. First wave: **ghost PASS** (`+0.0074` vs `+0.020` bar); q-bias
  structurally null at n256 serving; LCB and combo flat.
- **R0.6(i) refresh-divisor 4: ADOPTED (07-12 05:50).** Paired
  `+0.0375`, CI `[-0.1611, +0.2361]` (floor above the preregistered
  `-0.25` margin) with a `1.243x` mean-decision speedup — score-neutral
  pure speed, the second adopted serving default after exact-K1.
  `--gumbel-refresh-sample-divisor 4` is now the serving/benchmark
  default.
- **R1.2A ghost opponents: CI+ at the wall-matched gate (07-12 10:30) —
  the campaign's first CI+ wall-matched search improvement.** Ghost
  n512/d4 (opponent plies fast-forwarded by top-1 policy, zero model
  evals; reclaimed budget reinvested as 2x sims) vs champion n256/d4:
  paired **`+0.5450`**, CI **`[+0.1823, +0.9077]`**, at `1.049x` wall.
  Graduates per the preregistered rule.
- **R1.3a coverage audit: measured (07-12 10:55) — R1.3 stays open.**
  Valid rerun at rev `1c9211a5` (200/200 roots): greedy-256 cap drops
  the full-menu best in `1.5%` of decisions (above the `<1%` close bar)
  at `+0.30` regret each; mean overall `+0.0045`/root. The cap is safe
  on average with a thin material tail (~0.37 Q/game bound) — R1.3b/c
  remains a priced, modest-upside lane.
- **Live now (session-independent on john0):** the **ghost n1024-tier
  confirmation** (launched 07-12 10:38: champion n1024/d16 vs ghost
  n2048/d16, both arms div4, 100 paired seeds `2027072600..99`,
  ~13-17h; preregistered 10:45). CI+ at ≤1.25x wall makes ghost
  n2048/d16 **champion-designate** — John alone rules on promotion.
- **Research planning:** [`claude_max_research_ideas.md`](../../claude_max_research_ideas.md)
  (repo root, 07-10) — tiered break-100 research portfolio with
  preregisterable kill tests.
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
dated handoff ([handoff-2026-07-10.md](../handoffs/handoff-2026-07-10.md) — a
timestamped snapshot, weaker than current `main`).

## Read order for a fresh session

1. This file.
2. **[RESEARCH_PIPELINE_GUIDE.md](RESEARCH_PIPELINE_GUIDE.md) — the
   operator manual: read results, run screens/gates/queues, deploy —
   every command, end to end, no prior context assumed.**
3. [CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) — live state; RESUME HERE first.
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
- [Bridge Throughput](BRIDGE_THROUGHPUT.md): R2.4 serving-path
  investigation — request lifecycle, the serial-pipeline bound, ranked
  levers, and the staged GPU probe.
- Latest handoff: [handoff-2026-07-10.md](../handoffs/handoff-2026-07-10.md).

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
- Refresh-decision sub-searches run at 1/4 budget
  (`--gumbel-refresh-sample-divisor 4`, adopted 2026-07-12, score-neutral
  noninferior, 1.24x mean-decision speedup); the sample count stays 8.
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
