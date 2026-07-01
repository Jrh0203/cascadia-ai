# Entity Ranker V2 Terminal R8: Rejected Before Gameplay

The hidden-state-safe terminal distillation pipeline completed exactly as
registered:

- 64 train games, 5,120 groups, 76,974 candidates;
- 16 validation games, 1,280 groups, 19,096 candidates;
- zero hidden-refill records across all 96,070 candidates;
- 13 MLX epochs and 4,160 optimizer steps before validation patience.

The best checkpoint was epoch 8:

| Metric | Initial | Best | Gate |
|---|---:|---:|---:|
| Selection loss | 2.664595 | 2.563423 | improve |
| Mean top-one regret | 1.568164 | 0.968164 | <= 0.75 |
| Pairwise accuracy | 0.552465 | 0.680330 | >= 0.65 |
| Value-difference correlation | 0.098475 | 0.507718 | >= 0.30 |
| Tie-aware top-one recall | 0.153906 | 0.275781 | >= 0.45 |
| Strict top-one accuracy | 0.114063 | 0.200781 | diagnostic |

The model passed broad ordering metrics but failed both best-action gates. It
was not promoted and no gameplay benchmark was run.

The independent afterstate scorer receives no action identity or newly placed
marker. It must infer a one-tile change by independently re-estimating an
almost identical full board for every candidate. Its 0.968 regret only barely
improved on fixed immediate rank 1 at 0.999. Early regret improved, middle was
flat, and late regret regressed. The permanent follow-up is an explicit
action-delta representation, not relaxed gates.
