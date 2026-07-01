# ADR 0067: Public Focal Open-Loop Tree Search

Status: rejected after pilot on 2026-06-12.

## Context

The exact-hidden final-five focal beam proved that coordinated future focal
choices contain useful signal, but both fair public beam variants regressed in
online play. Their root candidates and redeterminizations were evaluated
independently, so evidence from one simulated future focal turn was never
reused by another simulation.

ADR 0066 also closed another loss-only sparse-NNUE fine-tuning attempt:
trajectory calibration improved substantially while held-out action ordering
and costly-error regret did not. The next experiment must change the planning
mechanism rather than another loss weight.

Cascadia's uncertainty is shared concealed supply, not private player state.
The relevant established ideas are information-set MCTS with fresh
redetermination per simulation and stochastic planning with explicit
decision/afterstate/chance structure:

- Cowling, Powley, and Whitehouse, *Information Set Monte Carlo Tree Search*
  (2012): <https://eprints.whiterose.ac.uk/id/eprint/75048/>
- Antonoglou et al., *Planning in Stochastic Environments with a Learned
  Model* (Stochastic MuZero, 2022):
  <https://openreview.net/forum?id=X6D9bAHhBQ1>

Exact public observations branch too widely for a useful late-game tree at a
local budget. This experiment therefore uses an open-loop information-set
tree: a focal action is represented by its deterministic rank in the frozen
public frontier at that turn. The concrete legal action is reconstructed from
each sampled public state. This shares evidence across chance outcomes without
placing concealed stack order in a node key.

## Frozen Treatment

- Canonical four-player AAAAA with habitat bonuses excluded.
- Pattern-aware play before the final five personal turns.
- Fresh public hidden-supply redetermination for every simulation, seeded only
  from the canonical public-state hash and simulation index.
- 128 sequential simulations per searched decision.
- Full mandatory coverage of the first 16 root candidates.
- K8+H6+B8+W2+M4 root and future focal frontiers.
- Opponents use the frozen pattern-aware policy.
- Future focal nodes use square-root progressive widening:
  `ceil(sqrt(node_visits + 1))`.
- UCT on terminal acting-seat base score with exploration coefficient 2.000.
- Terminal pattern-aware rollout on first expansion.
- Final action is the robust root child: most visits, then higher mean, then
  lower deterministic rank.
- A root requiring an unobserved three-of-a-kind refill falls back to the exact
  pattern-aware action. The bundled action API cannot fairly return a concrete
  post-refill draft before that refill is public.

This is distinct from ADRs 0033 and 0041: those built independent concrete
beams for every root candidate and redetermination. ADR 0067 shares future
ranked-action statistics across all simulations.

## Frozen Protocol

- Implementation-only test seeds may be opened before qualification.
- Runtime smoke: seed `34899`.
- Pilot, only after all implementation and smoke gates pass: seeds
  `34900-34909`.
- Confirmation, only after every pilot gate passes: seeds `35000-35049`.
- Baseline:
  `late-conservative-base-policy-improvement-v1-t5-r8-k8-h6-b8-m4-c90`.
- Execution: sequential local CPU blocks.

Runtime smoke gates:

- complete treatment block at most 600 seconds;
- treatment P90 searched-decision latency at most 30 seconds;
- zero illegal actions, replay failures, panics, or budget mismatches.

Pilot gates:

- paired treatment-minus-strong mean at least +0.50;
- treatment absolute mean at least 93.50;
- total wildlife delta at least 0.0;
- Elk+Salmon+Hawk+Fox delta at least -0.50;
- habitat delta at least -0.50;
- Nature Token delta at least -1.0;
- runtime at most 600 seconds per block and P90 latency at most 30 seconds.

Passing the pilot authorizes the frozen 50-block confirmation. Promotion
requires a positive paired 95% confidence-interval lower bound, paired mean at
least +0.25, treatment mean at least 94.00, nonnegative total wildlife,
non-Bear wildlife at least -0.25, habitat at least -0.25, Nature Tokens at
least -0.50, and the same runtime limits.

Ten pilot blocks cannot promote a strategy. No simulations, exploration,
frontier, widening, rollout, cutoff, seed, or threshold tuning is permitted
between stages. A pilot paired gain below +0.25 or treatment mean below 92.75
rejects this mechanism at the frozen budget.

## Required Evidence Before Smoke

- public-state serialization and hashing ignore concealed supply order;
- configuration rejects zero work and budgets that cannot cover the root;
- every retained root action receives a simulation;
- exact root visit count equals the configured simulation budget;
- deterministic repeated analysis is byte-for-byte equivalent;
- hidden-order redetermination leaves analysis unchanged;
- selected actions are legal;
- complete matches are deterministic and replay-verifiable;
- CLI output identifies every frozen treatment parameter;
- formatting, strict lint, and focused tests pass.

## Implementation Evidence

The integrity suite exposed a root-cause defect in the shared engine before the
registered smoke was opened. `redeterminize_hidden` previously shuffled the
existing concealed vectors directly. A uniform shuffle remained
distributionally fair, but the concrete sampled world for a fixed public
sample seed depended on the source game's inaccessible hidden permutation.

The engine now sorts the unseen tile multiset by tile ID and the wildlife
multiset by species before applying the domain-separated shuffle. The same
public facts and sample seed therefore produce the identical complete sampled
state regardless of prior hidden order. Existing and new public-search
invariance tests pass with the corrected operator.

Before opening seed `34899`:

- all 50 `cascadia-game` tests passed;
- all 60 `cascadia-search` tests passed;
- both `cascadia-cli-v2` tests passed;
- strict Clippy passed for `cascadia-game`, `cascadia-search`, and
  `cascadia-cli-v2` with all targets and `-D warnings`;
- the release binary built successfully;
- repeated tree analysis was identical;
- root visits exactly matched the configured budget and covered every
  retained root action;
- hidden-order invariance, selected-action legality, complete-match
  determinism, and replay verification passed.

## Commands

```bash
cargo test -p cascadia-game -p cascadia-search
cargo clippy -p cascadia-game -p cascadia-search -p cascadia-cli-v2 \
  --all-targets -- -D warnings
cargo build --release -p cascadia-cli-v2

target/release/cascadia-v2 public-focal-tree-compare \
  --games 1 --first-seed 34899 --terminal-turns 5 \
  --simulations 128 --root-candidates 16 --exploration-milli 2000 \
  --wildlife-candidates 2 --sequential \
  --output docs/v2/reports/public-focal-open-loop-tree-v1-runtime-smoke-1.json
```

The pilot command is identical except for
`--games 10 --first-seed 34900` and its output path.

## Runtime Smoke

Registered seed `34899` passed every runtime and integrity gate:

- promoted strong mean: 92.500;
- public open-loop tree mean: 93.000;
- paired delta: +0.500;
- treatment runtime: 11.866 seconds per four-seat block;
- treatment mean decision latency: 148.3 ms across all decisions;
- treatment P90 decision latency: 689.1 ms;
- treatment maximum decision latency: 1,391.8 ms;
- total wildlife delta: +0.250;
- Elk+Salmon+Hawk+Fox delta: 0.000;
- total habitat delta: -0.500;
- Nature Token delta: +0.750.

The report is
`docs/v2/reports/public-focal-open-loop-tree-v1-runtime-smoke-1.json`
with BLAKE3
`c6d4313d8bbdda5d7f35d27496cfb15f954509bb6c85a58c0f9ae026addaa7d4`.
The single smoke score is not a strength claim. Its runtime and integrity
result authorizes the unchanged ten-block pilot.

## Pilot Result

Across registered seeds `34900-34909`:

- promoted strong mean: 92.350;
- public open-loop tree mean: 92.375;
- paired delta: +0.025, 95% CI `[-0.856,+0.906]`;
- record: 6 wins, 0 ties, 4 losses;
- treatment runtime: 14.782 seconds per block;
- treatment P90 decision latency: 897.9 ms;
- total wildlife delta: -0.125;
- Elk+Salmon+Hawk+Fox delta: +0.425;
- total habitat delta: +0.200;
- Nature Token delta: -0.050.

The treatment failed the +0.500 paired-score gate, the 93.500 absolute-score
gate, and the nonnegative total-wildlife gate. Its 92.375 mean also crossed
the preregistered 92.750 absolute rejection floor. Runtime, latency, non-Bear
wildlife, habitat, and Nature Token gates passed.

The report is
`docs/v2/reports/public-focal-open-loop-tree-v1-pilot10.json` with BLAKE3
`0a02c618c0797935fe7da224d67c7319c9f6b4d2766e8be028b0e6e7760b67ff`.
No confirmation is permitted, and seeds `35000-35049` remain unopened.

Open-loop sharing by deterministic candidate rank is therefore not the missing
multi-turn planning mechanism at this budget. It is fast enough for further
research, but its action-role abstraction conflates materially different
future public states and adds variance without strength. The production
strategy remains unchanged.
