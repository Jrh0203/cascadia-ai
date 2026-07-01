# O1 Policy-Held-Out Sequential Corpus v1 Preregistration

**Frozen before production collection:** 2026-06-17  
**Experiment ID:** `o1-opponent-intent-policy-heldout-corpus-v1`  
**ADR:** `0185-o1-policy-held-out-sequential-corpus.md`

## Question

Can a compact, exact sequential corpus support opponent next-action and
future-market-access learning while making policy identity and dataset identity
unavailable to model inputs?

## Hypotheses

**H1:** Native mixed-policy v2 simulation produces exact 76-window games with
complete three-opponent action and four-tile survival labels.

**H2:** Greedy, PatternAware, and PatternCommitment provide enough train
behavioral diversity to support held-out calibration on PatternCompetition.

**H3:** PatternPortfolio provides a second, stronger held-out behavior family
for sealed test, while Random provides a radical out-of-distribution stress
family.

**H4:** Public recent actions and opponent boards contain predictive intent
signal without exposing policy names, checkpoint IDs, game indices, or split
identity.

**H5:** Exact tile survival remains a balanced enough target for supervised
learning across all policy cohorts.

## Frozen Collection

| Dataset | Split | First index | Games | Required policy |
|---|---|---:|---:|---|
| `o1-opponent-intent-v1-train-part-0` | train | 0 | 512 | none |
| `o1-opponent-intent-v1-train-part-1` | train | 512 | 512 | none |
| `o1-opponent-intent-v1-validation` | validation | 100,000 | 256 | PatternCompetition |
| `o1-opponent-intent-v1-test` | test | 200,000 | 256 | PatternPortfolio |
| `o1-opponent-intent-v1-final-stress` | final | 300,000 | 128 | Random |

Every game uses Card A scoring, four players, and no habitat bonuses.

## Record Contract

The source state is the public state at turn `t + 1`, after focal turn `t` has
been applied and the market refilled. It is encoded from the focal player's
perspective.

History contains up to 12 actions ending with the focal action. Each entry has
an exact age, relative actor seat, draft choice, market semantics, placement,
wildlife placement or return, free replacement, and paid-wipe metadata.

Targets contain:

- three ordered opponent actions at relative seats one, two, and three;
- each target's policy code as evaluation provenance only;
- four exact initial tile IDs;
- consumption by opponent one, two, or three, or survival to next focal access;
- exact-tile plus public-wildlife pair survival;
- terminal seat scores for analysis only.

## Forbidden Inputs

The MLX loader must consume only `model_input_bytes`, which contains:

- public compact position with `game_index = 0` and targets zeroed;
- history count;
- fixed 12-entry public action history.

It must not contain:

- game index;
- split or cohort ID;
- seat-policy codes;
- target policy codes;
- physical tile IDs;
- future actions or survival labels;
- final scores.

## Production Gates

- 1,664 games and 126,464 windows complete.
- Every manifest and shard validates with no repaired-row path.
- Train policy support is restricted to the three declared train families.
- Required held-out policy support is present in every validation, test, and
  stress game.
- Exact crossed-host calibration payloads match.
- No exact model-input hash overlaps across split boundaries.
- Every action-factor class and survival disposition used for training or
  evaluation has nonzero support.

## Interpretation

A pass authorizes a matched MLX learnability experiment:

1. public-state-only control;
2. public state plus recent action history;
3. per-opponent latent intent with next-action auxiliary heads;
4. joint intent plus market-survival heads.

Validation selects without opening test. Final stress remains descriptive and
cannot select a checkpoint. No arm enters gameplay until policy-held-out
calibration and frozen high-regret ranking gates pass.
