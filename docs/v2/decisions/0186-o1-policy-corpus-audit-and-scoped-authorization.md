# ADR 0186: O1 Policy Corpus Audit And Scoped Authorization

**Status:** Completed  
**Date:** 2026-06-17  
**Experiment:** `o1-opponent-intent-policy-heldout-corpus-v1`

## Context

ADR 0185 freezes five sequential policy cohorts totaling 1,664 games and
126,464 focal windows. A valid manifest is necessary but insufficient for
training authorization. The combined corpus must also prove that:

- the five role contracts are exact;
- policy families are held out as declared;
- provenance and future labels cannot alter model inputs;
- no exact model input crosses a corpus boundary;
- every held-out action-factor class appears in training;
- tile-consumption, tile-survival, and pair-survival classes have support;
- a distinct Mac reproduces the complete scientific result.

The production corpus also exposes a real scope limit: Random, Greedy,
PatternAware, PatternCommitment, PatternCompetition, and PatternPortfolio
never perform a paid wildlife wipe. Treating the absence of this action as
evidence that a model understands nature-token wipe intent would be an
overclaim.

## Decision

Implement `opponent_intent_policy_corpus_audit` as a deterministic Rust
auditor over the five immutable dataset trees.

For every record it:

1. invokes native manifest and shard validation;
2. hashes the exact sanitized `model_input_bytes`;
3. mutates game identity, policy identity, score targets, future actions,
   physical tile identity, survival labels, and terminal scores;
4. requires model input to remain byte-identical after those mutations;
5. counts policy, history-action, target-action, and survival support;
6. compares all ten dataset pairs for exact model-input overlap.

The audit must be replayed on john1 and john2 from:

- one content-addressed source and executable bundle;
- one whole-tree-verified corpus copy;
- distinct host-local paths.

The Python classifier requires exact scientific JSON and BLAKE3 equality. It
then writes one terminal classification and a separate authorization map.

## Authorization Boundary

A pass authorizes:

- public-state-only MLX controls;
- public state plus recent-action-history models;
- next-draft auxiliary heads;
- tile and tile-plus-wildlife survival heads;
- policy-held-out validation and sealed-test calibration.

A pass does not authorize:

- a positive paid-wipe intent head;
- within-game strategy-switch supervision;
- transfer claims to the v1 champion or a learned v2 policy;
- gameplay integration, checkpoint promotion, or a score claim.

Nature-token-active and champion-like policy cohorts are mandatory successors
before those broader claims can be tested.

## Consequences

The corpus can advance O1 learnability without conflating "policy held out"
with "all relevant behavior represented." The closeout artifact remains useful
even if later policy cohorts extend the action support, because the original
five-way split, exact hashes, and authorization boundary are immutable.

## Outcome

john1 and john2 reproduced scientific BLAKE3
`eaf584928ba0b87340b53c4ec33d1b334fbbe76ced22830c96c58b2b0e819885`.
All seven corpus gates passed. The terminal classifier authorized the matched
public-state, history, next-draft, and market-survival MLX factorial while
keeping every broader claim false.
