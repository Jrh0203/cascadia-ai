# Cascadia V3

Cascadia v3 is the transformer, search, self-play, and wildlife-optimization
stack for building a superhuman Cascadia player.

## Current status

The project switched to a lean, strength-first workflow on 2026-07-25.
Preregistration, source and artifact hash gates, receipts, seed registries,
sealed evaluations, host restrictions, and mandatory campaign ledgers are
retired. Older documents and reports retain their historical wording but do
not govern new work.

The player-strength target remains a mean seat score of at least 100 over
1,000 four-player self-play games. The last useful corrected-rules reference
point in the historical campaign was approximately 98.4 at the expensive
n1024/d16 search setting. That result is a comparison point, not a pinned
champion contract.

The wildlife-card catalog currently has:

- one validated board for all 1,024 ordered A/B/C/D rulesets;
- 80 rulesets with completed exact proofs;
- a best known animal-only score of 85, achieved by eight rulesets;
- a sound all-rules interval of [85, 96] under the 20-animal,
  maximum-six-per-species constraint.

The July 26 bound merge eliminated the last two score-97 count branches.
The catalog boards and exact-search artifacts remain useful. Their embedded
hashes and proof-provenance fields are no longer required to continue, combine,
or improve the search.

## Resume here

Read [CAMPAIGN_STATE.md](CAMPAIGN_STATE.md) for the concise live state.
Before launching work, inspect the actual processes and available hardware.
Do not reconstruct or obey the old waiter/receipt chains.

## Active workflow

1. Run or resume the most promising training/search job.
2. Watch ordinary metrics and partial results.
3. Change course when the evidence says to.
4. Compare candidates directly on enough games to make the decision useful.
5. Keep the stronger checkpoint and continue.

Details:

- [TRAINING_PIPELINE.md](TRAINING_PIPELINE.md) — model and data path
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) — machines and practical commands
- [RESEARCH_PIPELINE_GUIDE.md](RESEARCH_PIPELINE_GUIDE.md) — rapid experiment loop
- [FLEET.md](FLEET.md) — Mac mini fleet access
- [WILDLIFE_OPTIMAL_CATALOGS.md](WILDLIFE_OPTIMAL_CATALOGS.md) — AAAAA/CBDDB work
- [ALL_WILDLIFE_RULESET_CATALOG.md](ALL_WILDLIFE_RULESET_CATALOG.md) — all 1,024 rulesets
- [WILDLIFE_ATLAS_WEB_APP.md](WILDLIFE_ATLAS_WEB_APP.md) — catalog viewer

## Historical material

`cascadiav3/EXPERIMENT_LOG.md`, `RESEARCH_LOG.md`, dated handoffs, old gates,
and campaign reports are retained as notebooks. They can inform decisions but
cannot block work.
