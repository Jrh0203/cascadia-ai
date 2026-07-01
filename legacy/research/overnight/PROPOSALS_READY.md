# Experiment Launchers

All infrastructure shipped during sym_unexp run. Pick one to launch after
sym_unexp finishes (currently ~3 hr remaining).

## A — Fictitious Self-Play Reservoir (RECOMMENDED NEXT)

Train against a growing reservoir: draft opponents + mce93 + all past iters.
AlphaStar-style league training. Bear-competition signal from mce93,
diversity from draft opponents, non-stationarity from past selves.

```bash
./overnight/train_fsp_reservoir.sh 5
# Saves: nnue_weights_sym_fsp_iter{1..5}.bin
# Per-iter logs: overnight/train_sym_fsp_iter{1..5}.log
# Runtime: ~50 min/iter × 5 = ~4 hours local
```

Opponents each iter:
- iter 1: {random, scarcity, preference, mce93}
- iter N: adds all sym_fsp_iter{1..N-1} to the pool

## B — Temperature-Sampled Self-Play

Switch from ε-greedy random moves to softmax sampling over NNUE scores.
AlphaZero-style exploration. Cleaner than ε-random because it picks
good-but-not-best moves proportionally.

Env var: `CASCADIA_TRAIN_TEMPERATURE=1.0` (overrides --epsilon).
Typical: τ=2.0 early iters, anneal to τ=0.5 late. Manual per-iter control:

```bash
CASCADIA_TRAIN_OPP_POOL="nnue_weights_mce93.bin,random,scarcity,preference" \
CASCADIA_TRAIN_TEMPERATURE=1.0 \
CASCADIA_TRAIN_SEED=11111 \
./target/release/cascadia-cli 100000 --nnue-train \
  --lr 1e-4 --epochs 15 \
  --init-weights nnue_weights_sym_unexp_iter5.bin \
  --weights nnue_weights_sym_temp_iter1.bin
```

## E — Small Architecture (mce93's 256→32)

Smaller model (10 MB vs 89 MB), may generalize better with limited data.
Build to a separate target so the default-arch binary stays intact.

Build (one-time, ~15s incremental):
```bash
cargo build --release --features small-net --bin cascadia-cli --target-dir target-small
```

Train:
```bash
CASCADIA_TRAIN_OPP_POOL="nnue_weights_mce93.bin,random,scarcity,preference" \
CASCADIA_TRAIN_SEED=22222 \
./target-small/release/cascadia-cli 100000 --nnue-train \
  --lr 1e-4 --epochs 15 --epsilon 0.1 \
  --weights nnue_weights_sym_small_iter1.bin
```

NOTE: small-net binary cannot load default-arch models (e.g., sym_unexp_iter5).
Use mce93.bin as the pool NNUE (mce93 loads cleanly into either arch since
it's stored at its own smaller feature count).

## D — Distributional NNUE (variance head) — NOT YET IMPLEMENTED

Biggest change: adds a variance output head to NNUE, trains on (mean, var)
labels derived from rollout score variance. Inference picks candidates by
`mean ± k·stdev` based on whether we're ahead or behind. 2-3 hours of
careful work. Deferred — do after A or B shows promise.

## Validation (always)

Modal HH against mce93:
```bash
python3 -m modal run overnight/head_to_head_modal.py \
  --strategies "mce_new,mce_anchor,mce_anchor,mce_anchor" \
  --strategy-weights "mce_new=nnue_weights_<your_run>.bin,mce_anchor=nnue_weights_mce93.bin" \
  --game-samples 13 \
  --weights nnue_weights_mce93.bin
```

Cost ~$0.22 for N=52 games. Volume caches weights across runs.
