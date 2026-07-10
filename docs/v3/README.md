# Cascadia V3

Cascadia v3 is the transformer-based training and search stack for pushing
four-player Cascadia beyond the previous neural/search plateau. The live path is
not an NNUE campaign and not a v2 MLX continuation; it is CascadiaFormer over
packed expert tensors with search-supervised action values.

## Canonical Docs

- [Current Handoff](../../handoff-2026-07-09.md): timestamped operational
  snapshot, exact live PIDs/artifacts, blockers, and resume checklist.
- [Research Log](RESEARCH_LOG.md): **the experiment record** — architecture,
  every direction tried with verdicts, scaling laws, decision-SNR
  measurement, and ranked future directions.
- [Rules Contract](RULES_CONTRACT.md): official wildlife-return ordering,
  optional three-of-a-kind policy choice, rules identity, and the resulting
  baseline compatibility boundary.
- [Radical Directions](RADICAL_DIRECTIONS.md): speculative architecture-level
  bets (exact endgame solving, tiny-model/huge-search inversion, pairwise
  comparator head, on-GPU rollouts, decomposed value heads, league play),
  each judged against the campaign's measured constraints.
- [Infrastructure Runbook](INFRASTRUCTURE.md): how to operate john0 + the
  mac-mini fleet — builds, job patterns, batteries/verdicts, seed registry,
  fleet rules, and the web UI deployment.
- [Campaign State](CAMPAIGN_STATE.md): live working state — read the RESUME
  HERE section first when picking up work.
- [Gumbel Self-Play Campaign](GUMBEL_SELFPLAY_CAMPAIGN.md): the active
  100-point plan — Gumbel search with neural leaf values, self-play data
  generation, phases, gates, and decision branches.
- [Architecture](ARCHITECTURE.md): model shape, tokenization, relation bias,
  public-boundary rules, and literature basis.
- [Training Pipeline](TRAINING_PIPELINE.md): data generation, objectives,
  expert iteration, checkpointing, and promotion gates.
- [Operations](OPERATIONS.md): local, john0 GPU, and Bacalhau worker workflows.
- [Performance](PERFORMANCE.md): measured loader/training/gameplay facts.
- [EI-0 Runbook](EI0_GREEDY_SEARCH_BOOTSTRAP_RUNBOOK.md): completed bootstrap
  checklist, resume path, measured timeline, and success gates.

The implementation package lives in
[cascadiav3/README.md](../../cascadiav3/README.md).

## Current Scientific State

Rules compatibility break: the historical score measurements below predate
the 2026-07-09 correction that made the free three-of-a-kind refresh an
explicit accept/decline policy action. They remain architecture evidence but
not promotion controls. Corrected-rules baselines are being regenerated; see
[Campaign State](CAMPAIGN_STATE.md) and [Rules Contract](RULES_CONTRACT.md).

- Real training data should use packed `.npz` tensor shards.
- JSONL is retained only for tiny audit fixtures.
- The default CascadiaFormer board fast path is radius 6: 127 canonical cells
  plus exact overflow entities.
- The model must learn score-to-go, and serving must rank by
  `exact_afterstate_score_active + predicted_score_to_go`.
- EI-0 is the first transformer run with positive no-search gameplay evidence:
  CascadiaFormer-q scored `89.6175` versus greedy `87.5575` over 100 complete
  games.
- EI-0 search-integrated K32 retained search reached `95.8000`, but trailed
  matched full K64 search by `1.1750`.
- EI-0 K56 retained search narrowed the matched full-K64 gap to `0.5625` and
  passed the timing gate with a `0.8834` treatment/control ratio, but both K56
  and full K64 remained below the 100-point target on 20-game mean score.
- EI-1 model-state expert iteration improved no-search q play to `90.7600`
  over 100 games, beating matched greedy by `3.2150` and EI-0 q's `89.6175`.
- EI-1 K56 search remained in the `96-97` score band on recovered evidence:
  `96.4250` over 20 complete candidate games, with no 100-point breakthrough.
- The next improvement should change the policy/value/rollout target, not just
  increase retained width, rollout count, or this exact EI-1 objective.
- K64/R32 showed that raw rollout count is not the bottleneck: the greedy
  rollout policy itself caps the teacher.
- All pre-2026-07-02 search-integrated numbers carry a hidden-information
  leak (rollouts observed the true hidden tile/bag order) and are treated as
  legacy-leaky; honest baselines use `--rollout-determinize`.
- The active strategy is the Gumbel self-play campaign
  ([GUMBEL_SELFPLAY_CAMPAIGN.md](GUMBEL_SELFPLAY_CAMPAIGN.md)): Gumbel top-m
  search with batched model leaf values over determinized states, all-seat
  self-play data with improved-policy targets and real-outcome value labels
  (new generation uses exact-grounded schema v4), and CI-gated promotion at
  100+ paired games. EI-1 was
  terminated in favor of this line.
- Under corrected rules, cycle4 scored `97.0675` and distq-k8 `97.3075` at
  n256/d4 over the same 100 seeds; the `+0.2400` delta was inconclusive. The
  corrected scalar n1024/d16 baseline is complete at `98.2975`; the matching
  distq n1024 arm is live and owns the next strength decision.

## Historical Recovery

The pre-cleanup v1/v2 archive, older planning memos, v2 MLX package, web app,
legacy teacher bridge, and rejected experiment attic were removed from `main`.
Recover them from:

```bash
git show archive/pre-v3-repo-cleanup-2026-07-01:<path>
```
