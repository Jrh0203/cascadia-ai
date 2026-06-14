# Action-Delta Ranker V1 Terminal R8: Rejected Before Gameplay

The explicit-action successor completed exactly as preregistered:

- deterministic enrichment of 64 train games, 5,120 groups, and 76,974
  candidates;
- deterministic enrichment of 16 validation games, 1,280 groups, and 19,096
  candidates;
- collection and enrichment of 16 untouched test games, 1,280 groups, and
  19,131 candidates;
- exact action-hash, immediate-value, observable-afterstate, and teacher-winner
  replay checks for every enriched candidate;
- 12 MLX epochs and 3,840 optimizer steps before validation patience.

The best checkpoint was epoch 7:

| Metric | Initial validation | Best validation | Untouched test | Gate |
|---|---:|---:|---:|---:|
| Selection loss | 2.664025 | 2.559858 | 2.567485 | improve on validation |
| Mean top-one regret | 1.451270 | 0.922656 | 0.967773 | <= 0.75 test |
| Pairwise accuracy | 0.577592 | 0.685155 | 0.670384 | >= 0.65 test |
| Value-difference correlation | 0.153919 | 0.500543 | 0.495073 | >= 0.30 test |
| Tie-aware top-one recall | 0.141406 | 0.290625 | 0.272656 | >= 0.45 test |
| Strict top-one accuracy | 0.096875 | 0.233594 | 0.217969 | diagnostic |

The model learned broad candidate ordering and generalized that signal to the
untouched split, but it failed both best-action gates. Promotion rejected the
failed test report, no model artifact was created, and no gameplay benchmark
was run.

Explicit action identity was not the missing step function. It reduced
initialization error and slightly improved validation fidelity relative to the
independent afterstate encoder, but untouched regret remained essentially at
the prior model's 0.968 level and recall remained near 0.27. The terminal
teacher's per-action means overlap too heavily for this small supervised
ranker to identify tied optima reliably from one public position.

The next experiment must change the decision problem rather than decorate the
same candidate scorer. The strongest remaining evidence points to opponent-
conditioned future market availability and allocation planning: current
pattern opportunity assumes optimistic species supply and does not model which
opponents are likely to consume scarce wildlife before the acting player's
next turn.
