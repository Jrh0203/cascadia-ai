# Current Objective & Plan — April 16, 2026

## Active Right Now

**Policy network training running** (PID check: `ps aux | grep train-policy`):
```bash
target-mid/release/cascadia-cli 0 --train-policy \
  --weights overnight/policy_net_v1.bin \
  --init-weights nnue_weights_mid_fsp_iter10.bin \
  --load-policy-data overnight/policy_training_data.bin \
  --epochs 500 --policy-lr 0.001 --temperature 2.0
```
- Architecture: PolicyNetwork (10.8K → 512 → 256 → 1), initialized from value NNUE's first layer
- Data: 2000 positions (100 games × 20 turns), 100 cands × 100 MCE rollouts each
- Data file: `overnight/policy_training_data.bin` (70MB, collected on Modal for $1.68)
- At epoch 210/500: loss=4.25, agree=8.4% (still training)
- Output: `overnight/policy_net_v1.bin`

## Objective

**Train a policy network that ranks candidate moves better than the NNUE value head (23% top-1 agreement with MCE).**

The NNUE prefilter drops 41% of MCE-best moves at K=8. If a policy network achieves 50%+ top-1 agreement, we can dramatically improve search quality.

## What we've tried for policy

| Approach | Top-1 | Notes |
|---|---|---|
| NNUE value head (existing) | 23% | Trained with MSE, not ranking loss |
| Frozen h2 (64d) + linear | 4% | Too compressed |
| Delta features linear | 14% | No context awareness |
| Fresh network (random) | 8% | Not enough data for 5.7M params |
| Fine-tuned value NNUE | 4% | Catastrophic forgetting |
| **Wide PolicyNet (256d h2)** | **8.4% at ep210** | **Currently training to 500 epochs** |

## Key Findings This Session

### Value Network
- **mid_fsp_iter10 TIES mce93** at N=100 HH (26% vs 24.7%)
- **mid_fsp_iter10 + diverse_v2 prefilter BEATS mce93** (31% win, 95.01 mean)
- Mid-features (10.8K) much better than full v3 (45K) due to param/data ratio
- FSP reservoir training (growing opponent pool) is the best training recipe

### Rank Correlation (50 games, 83K candidates)
- NNUE top-1 agrees with MCE only 23% of the time
- K=8 prefilter catches 59% of MCE-best (NNUE opp) or 70% (MCE opp)
- Mid-game turns 6-13 are worst: 42-62% miss rate, 2-3 pt gaps
- Bear moves dropped most often (52% miss)
- MCE rollout std ~2.7 (SE ~0.27 at 100 rollouts — estimates are precise)
- Raw data: `overnight/rank_corr_raw_50g_100r.jsonl` + `_mce_opp.jsonl`

### Infrastructure Built
- `crates/cascadia-ai/src/policy_net.rs` — standalone PolicyNetwork (512→256→1)
- `crates/cascadia-ai/src/draft_opponents.rs` — random/scarcity/preference opponents
- `overnight/collect_policy_data_modal.py` — Modal data collection
- `overnight/rank_correlation_modal.py` — Modal rank correlation diagnostic
- `overnight/analyze_rank_corr.py` — offline analysis from JSONL
- `overnight/head_to_head_modal.py` — Modal HH with volume-cached weights
- `overnight/train_fsp_reservoir.sh` — FSP reservoir training
- Feature flags: `--features mid-features` (10.8K), `--features legacy-features` (5.2K)
- Binaries: `target-mid/release/cascadia-cli`, `target-legacy/release/cascadia-cli`
- Search improvements: `MCE_OPP_TEMPERATURE`, `MCE_DIVERSE_PREFILTER`, `MCE_MUTATE_EXPAND`, `--alloc thompson`

## Next Steps After Policy Training Completes

1. **Check policy_net_v1 agreement** — if >15%, test in prefilter
2. **If policy plateaus at ~10%**: need MORE data (collect 500+ games on Modal, ~$5) or different architecture
3. **If policy reaches 30%+**: integrate into prefilter, HH benchmark on Modal
4. **Fallback**: diverse_v2 prefilter already beats mce93 (31% win rate) — ship it

## Key Files

| File | Purpose |
|---|---|
| `nnue_weights_mid_fsp_iter10.bin` | Best value network (ties mce93) |
| `nnue_weights_mce93.bin` | Original champion |
| `overnight/policy_training_data.bin` | 2000 positions of policy training data |
| `overnight/rank_corr_raw_50g_100r.jsonl` | Rank correlation raw data (NNUE opp) |
| `overnight/rank_corr_raw_50g_100r_mce_opp.jsonl` | Rank correlation raw data (MCE opp) |
| `overnight/policy_net_v1.bin` | Policy network (training in progress) |

## Modal Budget
- Spent this session: ~$10.50
- Remaining: ~$9.50 of original $20
- Key costs: rank correlation ($3.40), HH benchmarks ($5.50), policy data ($1.68)

## Champion Command (current best, beats mce93)
```bash
MCE_LMR=1 MCE_DIVERSE_PREFILTER=1 MCE_MUTATE_EXPAND=4 \
./target-mid/release/cascadia-cli N --nnue-rollout-mce \
  --candidates expanded --prefilter-k 8 --alloc halving \
  --rollouts 200 --weights nnue_weights_mid_fsp_iter10.bin
```
