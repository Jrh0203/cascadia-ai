# ADR 0176: Matched V1 Score Anatomy on Exact Sparse R2

Status: authorized for open-validation execution.

## Context

The research plan proposes KataGo-style score anatomy because Cascadia's total
score is an exact sum of habitat, wildlife, and Nature Token objectives. The
repository already predicts eleven components in several value models, but it
has never run the decisive matched test: identical state representation,
parameter graph, initialization, optimizer, examples, and schedule with only
the scalar-total versus component-anatomy supervision changed.

The user also recalled that a 121-cell state was sufficient in earlier work.
The current exact sparse representation is stronger than a dense 11 by 11
crop: ADR 0145 verified a maximum of 92 active-or-padded sparse objects per
board while preserving occupied tiles, legal frontier, habitat components,
wildlife motifs, exact D6 transforms, and public-state reconstruction. ADR
0174 later measured a maximum of 84 exact sparse objects on the P1 corpus.
This experiment therefore uses no 441-cell lattice and stays below the 121
reference without spatial truncation.

## Decision

Run two supervision arms and exact cross-host replays on the accepted
`perceiver-fixed-latents` R2 trunk:

1. `scalar-total-control`: train only final total-score mean squared error.
2. `component-anatomy`: train normalized eleven-component mean squared error
   plus the existing total-score consistency term.

Both arms keep the same eleven-output nonnegative head. In the scalar control
those channels are unsupervised latent contributors whose sum is the total.
This preserves byte-identical parameter layouts and initial tensors instead of
padding a smaller control with inert parameters.

## Frozen Protocol

- exact R2 cache `c97ce6b...85f8`;
- open train 50,000 positions, open validation 10,000 positions;
- seed `2026061801`;
- 3,000 steps, batch 32;
- AdamW, learning rate `3e-4`, weight decay `1e-4`;
- uniform exact D6 transform sampling;
- full validation at identity transform;
- identical Perceiver trunk and 11-output head;
- scalar primary on john2, anatomy primary on john3;
- scalar replay on john4, anatomy replay on john1.

No test/final split or gameplay result may be opened.

## Promotion Gate

The treatment advances to a complete-action value experiment only if:

- primary and replay final parameter tensors are exact for both arms;
- primary and replay prediction probes are exact for both arms;
- all cache, protocol, parameter-layout, and initialization identities match;
- anatomy total MAE is no more than 0.05 worse than scalar;
- anatomy total correlation improves by at least 0.03;
- anatomy within-round pairwise log loss improves by at least 0.005;
- every component metric is finite;
- the exact sparse per-board capacity remains at or below 121.

Passing does not claim gameplay strength. Failing closes only this basic
component-supervision hypothesis; motif completion, multi-horizon, and
distributional heads remain separately testable hypotheses.
