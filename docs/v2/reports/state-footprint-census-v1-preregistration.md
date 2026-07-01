# State-Footprint Census V1 Preregistration

Date: 2026-06-16

Experiment ID: `state-footprint-census-v1`

ADR 0128 measures the current V2 board support before any compact model is
implemented. It is the F2 prerequisite for the controlled 441, 127, 91, 61,
and sparse representation tournament.

The current V2 rules board uses a 49x49, 2,401-cell backing grid but tracks no
more than 23 occupied indices per player. The tournament's exact control will
preserve the complete coordinate, frontier, and action domain without forcing
that backing grid into a dense neural tensor. The historical 441-cell square
is a diagnostic arm, not the control, and must also preserve exact overflow.

The historical claim is now stated precisely. Complete centered radius-5 and
radius-6 hex disks contain 91 and 127 cells. No complete centered hex disk
contains 121 cells. The legacy repository reports 99.7% and 99.9% per-cell
firing retention at radii 5 and 6 over 50,000 old self-play states, but that
experiment silently dropped overflow and changed several model variables at
once. It is evidence for leverage, not evidence of strength-neutrality.

The new census has two disjoint arms:

- john4 generates 625 pattern-aware four-player AAAAA games from raw seeds
  73000 through 73624, yielding exactly 50,000 pre-move states and 200,000
  board observations; and
- john1 validates and scans every unique group in the open complete-action
  graded-oracle train and validation datasets.

For radii 3 through 8, the report measures occupied, frontier, selected-action,
and complete-candidate support; feature-firing retention; crossing components
and wildlife adjacencies; overflow incidence; fixed and recentered radii;
dense bytes; sparse token counts; public serialization bytes; and extraction
time. Every radius-6-or-larger generated outlier is retained with deterministic
identity. Straight and bent legal elongated boards provide adversarial
overflow controls.

The experiment succeeds only if the complete generated and open-corpus
domains are present, all checksums and count identities pass, every required
table is complete, no radius-6 outlier is truncated, both adversarial controls
overflow, D6 radius invariance passes, and the combined scientific hash is
reproducible. Failure blocks R0 training.

This experiment does not choose 127, 91, 61, sparse, or any learned
representation. It only establishes the exact support and overflow contract
that makes their later comparison scientifically valid.
