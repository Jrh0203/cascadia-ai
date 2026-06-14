# ADR 0050: Action-Conditioned Cross-Attention Ablation

Status: rejected on validation on 2026-06-12. No sealed test or gameplay seed
was opened.

## Context

ADR 0049 rejected the first full-legal apprentice. The model learned broad
ordering but failed to place the selected teacher action near the top
reliably. Its architecture encoded each board and the market into pooled
state summaries before combining them with an action embedding. That makes
candidate-specific geometry, local wildlife compatibility, and market-slot
relationships unnecessarily indirect.

Validation diagnostics found:

- v1 top-one 19.453%, top-five 50.547%, MRR 0.347332;
- early-turn MRR 0.4170 versus late-turn MRR 0.3376;
- paired-draft MRR 0.3528 versus independent-draft MRR 0.3017;
- the selected teacher action had immediate rank above 16 in 623/1,280
  groups, where v1 still reached 24.719% top-one and 54.896% top-five;
- a validation-tuned reciprocal-immediate-rank residual at coefficient 0.08
  reached 24.063% top-one, 58.203% top-five, and 0.401136 MRR.

The residual probe shows that the learned strategic signal and a monotonic
short-term prior are complementary. It is not promotion evidence because its
coefficient was selected on validation.

## Decision

Implement `shared-state-action-cross-ranker-v2` without changing the dataset,
Rust protocol, or legal-action boundary:

1. Encode each board and the market once per decision with the existing
   self-attention blocks.
2. Preserve the pooled state summary used by v1.
3. Project each candidate action into a query.
4. Cross-attend every action query over the encoded board tokens and market
   tokens.
5. Score the pooled state, action embedding, board context, and market context
   together.

The architecture remains exhaustive, shared-state, locally trained in MLX,
and backward-compatible with v1 checkpoint loading.

## Frozen Validation Experiment

- Train dataset: `canonical-action-imitation-train-a0155b3613e51112`.
- Validation dataset:
  `canonical-action-imitation-validation-4929d2a8a2bb0a0d`.
- Model: hidden 96, four heads, two board blocks, one market block, feed-forward
  multiplier three.
- Optimizer: AdamW, learning rate `1e-4`, weight decay `1e-4`.
- Training: at most 20 epochs, batch 16 groups, patience five, seed 20260613.
- Selection: validation listwise loss only.
- Command: `make train-imitation-cross`.

The ablation advances to a separately registered fresh-test experiment only
if the selected checkpoint:

- improves validation listwise loss over v1's 2.948818;
- reaches top-one accuracy at least 23%;
- reaches top-five recall at least 58%;
- reaches MRR at least 0.40;
- completes without integrity, resume, or non-MLX failure.

No result from the sealed ADR 0049 test split may be inspected or used for
selection. Missing any gate rejects this architecture as the next standalone
apprentice. The validation-tuned residual remains a possible component of a
future preregistered model, not a retroactive repair of either experiment.

## Result

The Apple-GPU run completed seven epochs in 31.0 seconds and stopped after
five non-improving epochs. Epoch two was selected:

- listwise loss 2.978492, worse than v1's 2.948818;
- top-one accuracy 18.984%, below the 23% gate;
- top-five recall 48.828%, below the 58% gate;
- MRR 0.337093, below the 0.40 gate;
- pairwise accuracy 88.199%.

Cross-attention failed every advancement gate and slightly regressed the
pooled v1 ranker. Candidate-specific token attention is therefore not the
dominant missing signal at this data scale and objective. The run is rejected
without opening a fresh test split.

The reciprocal-immediate-rank residual was also stress-tested without opening
test data. Leave-one-game-out coefficient selection chose 0.08 in 13 of 16
folds and produced 23.359% top-one, 58.047% top-five, and 0.396801 MRR.
A 20,000-sample game bootstrap for fixed coefficient 0.08 placed MRR's 95%
interval at 0.379739-0.424915. That is promising but too borderline to justify
a fresh held-out collection as a post-hoc wrapper. The next experiment will
train the strategic residual around a frozen monotonic prior.
