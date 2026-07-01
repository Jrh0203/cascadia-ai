# Exact Mid-V4 Bag Feature Template Rejection

Status: **rejected and removed**

Date: 2026-06-15

## Hypothesis

The qualified mid-features plus v4-opp NNUE policy evaluates up to 15
candidate afterstates from one rollout state. Those rows share the wildlife
bag, opponent habitat maxima, and per-opponent detail. The treatment computed
the two bag-dependent feature segments once and appended the resulting
indices to every candidate vector at their original positions.

Candidate generation, feature order, active indices, network inputs, search
boundaries, and random streams were unchanged. The implementation was
compiled only for the exact qualified feature combination.

## Exactness

A test compared templated and ordinary feature vectors byte for byte for
every candidate at all 640 decisions in eight complete seeded four-player
AAAAA games. Both full K32/R600 worker screens also reproduced:

- scores `[102,96,92,95]`, mean `96.25`;
- 3,920 neural batches and 6,121,807 logical neural rows;
- 5,062,305 physical rows;
- 3,716 rollout waves and 46,207 rollout samples;
- zero bootstraps, zero policy fallbacks, and clean shutdown.

## Source-Level Screen

Matched non-PGO release binaries were crossed across john2 and john3.

| Host | Control mean | Treatment mean | Speedup |
|---|---:|---:|---:|
| john2 | 15.661486 s | 15.642902 s | 1.00119x |
| john3 | 15.429961 s | 15.429272 s | 1.00004x |
| Combined | **15.545724 s** | **15.536087 s** | **1.00062x** |

The treatment saved 0.009637 seconds per game, or 0.062%. That missed the
preregistered 0.25% advancement floor by roughly a factor of four.

## Verdict

Reject before PGO. The shared bag fields are real common work, but their
43-55 sparse indices are too small a share of the complete candidate and
neural pipeline to produce a material end-to-end gain. Fresh PGO was not
authorized after the source gate failed.

The feature template, specialized extractor, callback emitter, test oracle,
and preregistration were removed. The ordinary single-source feature
extractor remains the only production implementation.

Machine-readable evidence:
`docs/v2/reports/exact-mid-v4-bag-feature-template-rejection-v1.json`.
