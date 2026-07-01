# CascadiaZero AAAAA With-Bonus Runbook

This pipeline trains and evaluates the CascadiaZero expert-iteration player for
configuration `AAAAA` with habitat bonuses enabled. It is append-only over the
full `v3 + v4-opp + v5-feat` representation via `czero-feat`.

## Build

```bash
cargo build --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli
```

Use `CASCADIA_SCORING_CARDS=A,A,A,A,A` for every collection, training smoke, and
promotion benchmark. Use `--score-target with-bonus` or the `czero_mce` tag to
force with-bonus MCE scoring.

`--collect-czero` is a teacher self-play collector. Every seat is played by the
same bonus-aware teacher. The teacher uses the expanded `mce_wide_v1` candidate
pool, diverse NNUE prefiltering, `MCE_LMR=1`, post-move bag features, and
NNUE-guided rollouts scored through the active `with-bonus` target. The default
prefilter is `--prefilter-k 32`; use `--czero-top-k N` or `--prefilter-k N` to
override it deliberately.

## Smoke

```bash
cargo test --features v4-opp,v5-feat,czero-feat

CASCADIA_SCORING_CARDS=A,A,A,A,A \
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  1 --collect-czero --rollouts 8 --prefilter-k 32 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --out /tmp/czero_smoke.czr

CASCADIA_SCORING_CARDS=A,A,A,A,A \
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  0 --train-czero --samples /tmp/czero_smoke.czr --epochs 1 \
  --init-weights nnue_weights_v4opp_modal_iter3.bin \
  --weights /tmp/czero_smoke.bin

CASCADIA_SCORING_CARDS=A,A,A,A,A \
CASCADIA_SEAT_STRATEGIES=czero_mce:greedy:greedy:greedy \
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  1 --nnue --weights /tmp/czero_smoke.bin \
  --policy-weights /tmp/czero_smoke.policy
```

## Local Training Regimen

```bash
# Stage 2: broad teacher data
CASCADIA_SCORING_CARDS=A,A,A,A,A \
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  300 --collect-czero --rollouts 160 --prefilter-k 32 \
  --weights nnue_weights_v4opp_modal_iter3.bin \
  --out czero_stage2.czr

# Train value + CZP1 policy.
CASCADIA_SCORING_CARDS=A,A,A,A,A \
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- \
  0 --train-czero --samples czero_stage2.czr --epochs 25 --lr 0.00003 \
  --init-weights nnue_weights_v4opp_modal_iter3.bin \
  --weights czero_stage2.bin
```

Repeat collection at `--rollouts 300` for the main stage and `--rollouts 600`
for hard-position data, using the latest `czero_stage*.bin` as the teacher
weights.

## Promotion

Primary metric is with-bonus symmetric head-to-head versus
`nnue_weights_v4opp_modal_iter3.bin + mce_wide_v1`; base score is the guardrail.

```bash
CASCADIA_SCORING_CARDS=A,A,A,A,A \
CASCADIA_SEAT_STRATEGIES=czero_mce:mce_wide_v1:czero_mce:mce_wide_v1 \
CASCADIA_SEAT_WEIGHTS=czero_stage2.bin:nnue_weights_v4opp_modal_iter3.bin:czero_stage2.bin:nnue_weights_v4opp_modal_iter3.bin \
CASCADIA_POLICY_WEIGHTS=czero_stage2.policy \
cargo run --release --features v4-opp,v5-feat,czero-feat --bin cascadia-cli -- 40 --nnue \
  --weights czero_stage2.bin --score-target with-bonus
```

Promote only after 100 games, extending to 200 if the result is close, when
with-bonus mean improves, base score does not regress beyond noise, and alternating
seat win rate improves.
