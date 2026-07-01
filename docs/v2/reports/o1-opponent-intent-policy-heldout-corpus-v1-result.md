# O1 Policy-Held-Out Sequential Corpus v1 Result

**Completed:** 2026-06-17  
**Experiment:** `o1-opponent-intent-policy-heldout-corpus-v1`  
**Classification:** `policy_held_out_draft_survival_corpus_passed`

## Verdict

The compact O1 corpus passed every preregistered foundation gate and reproduced
exactly on john1 and john2.

It authorizes the matched MLX learnability factorial for:

- public-state controls;
- public state plus recent action history;
- next-draft auxiliary prediction;
- exact tile-survival and tile-plus-wildlife survival prediction;
- policy-held-out validation and sealed test.

It does not authorize gameplay, score claims, paid-wipe intent, strategy
switching, or transfer claims to the v1 champion or a learned v2 policy.

## Corpus

| Role | Games | Windows | Required unseen family |
|---|---:|---:|---|
| Train part 0 | 512 | 38,912 | none |
| Train part 1 | 512 | 38,912 | none |
| Validation | 256 | 19,456 | PatternCompetition |
| Test | 256 | 19,456 | PatternPortfolio |
| Final stress | 128 | 9,728 | Random |
| **Total** | **1,664** | **126,464** | |

The corpus contains 104 immutable `.o1i` shards and 379,392 ordered opponent
action targets. Every model input is 1,189 bytes: one compact public state plus
up to 12 public actions. No 441-cell board tensor is materialized.

## Exactness

- 126,464 of 126,464 records passed forbidden-field mutation checks.
- All 126,464 model inputs are unique within their declared corpus.
- All ten corpus-pair overlap counts are zero.
- All five manifests, every shard header, every checksum, every 76-window game,
  every history age, every relative seat, and every target validated.
- john1 and john2 used executable BLAKE3
  `644635396a1d4703fb95f14539158403e32a77e64132f8bf3bb3a908c7abd9ca`.
- Both hosts produced scientific BLAKE3
  `eaf584928ba0b87340b53c4ec33d1b334fbbe76ced22830c96c58b2b0e819885`.

## Policy Boundary

Training seat support was:

- Greedy: 1,336;
- PatternAware: 1,387;
- PatternCommitment: 1,373;
- Random, PatternCompetition, PatternPortfolio: zero.

Validation contained 452 PatternCompetition seats, sealed test contained 423
PatternPortfolio seats, and final stress contained 184 Random seats. Every
held-out game contained its required family.

## Label Support

All nine audited next-action factors had complete train-to-held-out class
coverage:

- paired and independent drafts;
- every tile and wildlife slot;
- all six rotations;
- all five wildlife species;
- wildlife placement and return;
- free replacement on and off;
- the observed paid-wipe count and slot-count classes.

All four tile dispositions and both pair-survival classes appeared in training
and held-out evaluation. Across 505,856 tile labels:

- 126,464 were consumed by opponent one;
- 84,383 were consumed by opponent two;
- 60,480 were consumed by opponent three;
- 234,529 survived to the focal player's next access;
- 181,590 retained the original public wildlife pairing.

## Scope Limits

The audit found zero paid wildlife wipes in 379,392 target actions. This is a
real policy-support gap, not a serialization bug: all six current v2 simulator
policies decline paid wipes.

Therefore:

- paid-wipe intent training remains unauthorized;
- strategy-switch training remains unauthorized because policies are
  stationary within a game;
- champion generalization remains unauthorized because the held-out families
  are v2 heuristics;
- gameplay promotion remains unauthorized.

## Artifacts

- terminal classification:
  `artifacts/experiments/o1-opponent-intent-policy-heldout-corpus-v1/classification.json`;
- immutable audit bundle:
  `artifacts/experiments/o1-opponent-intent-policy-heldout-corpus-v1/audit-bundles/2dddc4a286b53f5b16142b71f0f8fef874bfebf17874352f825448143ae6fd2f`;
- primary report:
  `artifacts/experiments/o1-opponent-intent-policy-heldout-corpus-v1/closeout-launches/2dddc4a286b53f5b16142b71f0f8fef874bfebf17874352f825448143ae6fd2f/runs/john1-primary.json`;
- crossed-host replay:
  `artifacts/experiments/o1-opponent-intent-policy-heldout-corpus-v1/closeout-launches/2dddc4a286b53f5b16142b71f0f8fef874bfebf17874352f825448143ae6fd2f/collected/john2-replay.json`.

## Next Experiment

Run the four-arm matched MLX factorial on identical train and validation bytes:

1. compact public state only;
2. compact public state plus recent action history;
3. history model plus per-opponent next-draft auxiliary heads;
4. joint intent model plus exact market-survival heads.

Validation selects without opening test. The sealed PatternPortfolio test is
opened once after selection; Random remains descriptive final stress. A
separate corpus must add nature-token-active and champion-like opponents.
