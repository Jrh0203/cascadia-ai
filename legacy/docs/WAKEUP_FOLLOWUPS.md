# Wakeup follow-ups

When you wake up, here are the concrete actions to take in priority order:

## Good news first 🏆

**Your overnight training pipeline produced a real improvement.** I tested
`nnue_weights_hybrid_iter20.bin` (created at 07:00 by your training loop) and got:
- baseline_iter4 (mine, 200g): 95.3
- baseline_iter20 (yours, 200g): **95.7** (+0.4)

**Action: switch the default weight file to `nnue_weights_hybrid_iter20.bin` (or whatever
your latest is when you wake up).**

## Bad news second

The 200-game confirmation killed all my search-side improvements. Specifically:
- LEAF1 with iter4: looked like +0.6 at 50g, was +0.1 at 200g (within noise)
- LEAF1 with iter20: -0.3 at 200g (LEAF1 actively HURTS the new weights)
- All other search techniques: also within noise of baseline at 50g, which means they're
  almost certainly within noise at higher game counts too

**Don't enable `MCE_LEAF_EXPECTIMAX=1` or any of the other env vars I added.** They were
exploration code; none of them help with the better iter20 weights. The implementations
remain in place as opt-in env vars in case you want to revisit.

## Step 1: Check what completed

```bash
./wakeup_status.sh
```

This shows running processes, completed benches sorted by mean, and queue progress.

## Step 2: Read the wakeup report

```bash
$EDITOR WAKEUP_REPORT.md
```

The most important sections:
- TL;DR (the corrected story)
- Results table (the 200g rows are the reliable comparison)
- Why Cascadia is structurally hard for search
- Variance vs accuracy in MCE rollouts (deeper analysis)

## Step 3: Stop the slow benches

The NRPA bench (`pgrep -f "cascadia-cli.*nrpa"`) is making essentially no progress and
won't finish in any reasonable time. Kill it if you want to free CPU. Phase 3, 4, 5
queue scripts may still be running their items — they don't add value beyond what's
already in `bench_results/`.

## Step 5: Stop the slow benches if you don't want them

The NRPA bench (`pgrep -f "cascadia-cli.*nrpa"`) is making essentially no progress and
won't finish in any reasonable time. Kill it if you want to free CPU. The phase 3, 4, 5
queue scripts are at:

- phase 3 (`bp3qbi092`) — running leaf_n1500_50g
- phase 5 (`b2gwv33pl`) — running leaf1_c20

These will keep running their queue items as long as they're alive.

## Step 6: Optional next experiments

Phase 5 has `leaf1_c20` and `leaf1_c10` queued to test LEAF1 + different candidate counts.
Phase 5 also has `leaf1_n1500` to test LEAF1 + 1500 rollouts.

If you want to push further, the most promising untried directions (saved from the
"Speculative directions" section of the wakeup report):

1. Random feature projections for the NNUE
2. Anti-symmetric leaf eval (condition on root candidate)
3. Importance sampling at the leaf
4. Auxiliary value heads (separate "wildlife only" head)
5. Train NNUE on the grown `mce_policy_samples.bin` (~200MB now, ~14K samples per 50g run)

## Files I created (don't delete unless you mean to)

- `WAKEUP_REPORT.md` — main analysis (~700 lines)
- `WAKEUP_FOLLOWUPS.md` — this file
- `overnight_results.md` — per-experiment details
- `wakeup_status.sh` — run on wake to see status
- `verify_winner.sh` — A/B comparison of LEAF1 vs baseline at 200g
- `generate_report.sh` — produces the comparison table from `bench_results/*.log`
- `summarize_benches.sh` — older summary script
- `run_overnight_benches.sh` — original queue (still grinding NRPA)
- `run_overnight_benches_2.sh` — secondary queue (not started)
- `run_phase3.sh`, `run_phase4.sh`, `run_phase5.sh` — phased benchmarks
- `bench_results/` — log files from each completed/running benchmark

## Code changes (uncommitted)

- `crates/cascadia-ai/src/mce.rs` — 4 new leaf eval functions + 5 env vars
- `crates/cascadia-ai/src/nrpa.rs` — new (NRPA module)
- `crates/cascadia-ai/src/ol_mcts.rs` — new (Open-Loop MCTS, uses unsafe pointer descent)
- `crates/cascadia-ai/src/gumbel_mcts.rs` — new (Gumbel AlphaZero standalone)
- `crates/cascadia-ai/src/lib.rs` — register new modules
- `crates/cascadia-cli/src/main.rs` — `--nrpa`, `--ol-mcts`, `--gumbel-mcts` strategies

Run `git diff HEAD` and `git status` to see all changes.
