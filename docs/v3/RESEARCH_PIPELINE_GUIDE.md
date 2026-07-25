# Strength-First Research Loop

## The loop

1. Form a concrete hypothesis.
2. Run the cheapest experiment that can change the decision.
3. Watch the result while it runs.
4. Stop weak ideas early; extend promising ones.
5. Compare the best candidate directly with the current best player.
6. Keep the stronger checkpoint and repeat.

There is no preregistration, seed registry, sealed holdout, promotion gate, or
ban on partial results. Confidence intervals and paired seeds are useful when
they clarify uncertainty; they do not control permission to act.

## Quick comparison

Run the Gumbel benchmark with any convenient seeds and game count:

```bash
PYTHONPATH=cascadiav3/src python -m \
  cascadiav3.torch_cascadiaformer_gumbel_benchmark \
  --manifest <checkpoint.manifest.json> \
  --device cuda \
  --first-seed 1 \
  --games 32 \
  --jobs 12 \
  --batch-runner \
  --gumbel-n-simulations 256 \
  --gumbel-top-m 16 \
  --gumbel-determinizations 4 \
  --control none \
  --out cascadiav3/reports/<name>.json
```

Increase the game count when the decision remains uncertain. Reuse the same
seeds for lower-variance candidate comparisons or change them when broader
coverage is more useful.

## Training iteration

The normal expert-iteration path is:

1. generate self-play roots from the current checkpoint;
2. train or fine-tune CascadiaFormer;
3. evaluate the new checkpoint as soon as it exists;
4. continue from the better checkpoint.

Use `cascadiav3/scripts/run_gumbel_selfplay_cycle.sh` for the existing path.
Its configuration is ordinary environment variables. Adjust scale, search,
workers, and precision to fit the current question and hardware.

## Decision quality

Useful signals include:

- mean seat score and its recent trend;
- paired score delta against the current best;
- action agreement/regret on stored hard positions;
- validation loss and selected-action Q error;
- game throughput, GPU utilization, and time per useful label.

No single signal is sacred. A small fast screen can justify more compute; only
actual game strength should settle a close promotion decision.

## Records

Keep a short config, metrics JSONL, and checkpoints. Add a concise note to
`docs/v3/CAMPAIGN_STATE.md` when it helps the next session. The historical
`EXPERIMENT_LOG.md`, research log, handoffs, hashes, receipts, and decision
rules are optional references.
