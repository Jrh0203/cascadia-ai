"""Pure-greedy MCE(N) benchmark. No model.

For each player-0 decision:
    1. Rust enumerates candidate moves.
    2. For each candidate: N rollouts to depth D (greedy for all players).
    3. Each rollout's leaf score is the player-0 score after D-turn rollout.
    4. Candidate value = mean(leaf_score) over its N rollouts.
    5. Pick argmax.

This is "classical" Monte Carlo Evaluation — no value function, just averaging
actual greedy-play outcomes.

Usage:
    python3 bench_greedy_mce.py --games 50 --rollouts 750 --depth 20 --random-seed
"""

import argparse
import struct
import subprocess
import sys
import time

import numpy as np

from bench_gnn import MSG_DONE, MSG_FINAL, read_frame, write_pick, parse_candidate
from bench_gnn_mce import MSG_MCE_EVAL, parse_mce_eval


CASCADIA_CLI = "./target/release/cascadia-cli"


def pick_greedy_mce(num_cands, rollouts, leaves):
    """Average leaf_score across rollouts per candidate. Return argmax index.

    No value-function delta added — this is pure Monte Carlo on actual scores.
    """
    leaf_scores = np.array([leaves[i]["current_score"] for i in range(len(leaves))],
                           dtype=np.float32)
    per_leaf = leaf_scores.reshape(num_cands, rollouts)
    per_cand_avg = per_leaf.mean(axis=1)
    return int(np.argmax(per_cand_avg)), per_cand_avg


def run_benchmark(args):
    cmd = [
        CASCADIA_CLI, str(args.games), "--gnn-mce-bench",
        "--rollouts", str(args.rollouts),
        "--depth", str(args.depth),
    ]
    if args.random_seed:
        cmd.append("--random-seed")
    print(f"Running: {' '.join(cmd)}")
    print(f"Config: {args.games} games, {args.rollouts} rollouts/cand, depth {args.depth} "
          f"(pure greedy MCE — no model)")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None)

    scores = []
    num_decisions = 0
    total_leaves = 0
    t0 = time.time()

    try:
        while True:
            msg_type, payload = read_frame(proc.stdout)
            if msg_type is None:
                break
            if msg_type == MSG_MCE_EVAL:
                num_cands, rollouts, leaves = parse_mce_eval(payload)
                idx, _ = pick_greedy_mce(num_cands, rollouts, leaves)
                write_pick(proc.stdin, idx)
                num_decisions += 1
                total_leaves += len(leaves)
            elif msg_type == MSG_DONE:
                scores.append(struct.unpack("<H", payload)[0])
            elif msg_type == MSG_FINAL:
                break
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)

    elapsed = time.time() - t0

    print(f"\n{'═' * 50}")
    print(f"Greedy MCE({args.rollouts}) bench: {len(scores)} games in {elapsed:.1f}s "
          f"({len(scores)/max(elapsed,1):.2f} g/s)")
    print(f"  {num_decisions} decisions, {total_leaves:,} leaves")
    if scores:
        s = np.array(scores)
        print(f"\n  Mean:    {s.mean():.2f}")
        print(f"  Median:  {int(np.median(s))}")
        print(f"  P10:     {int(np.percentile(s, 10))}")
        print(f"  P90:     {int(np.percentile(s, 90))}")
        print(f"  Min/Max: {s.min()}/{s.max()}")
        print(f"  Stddev:  {s.std():.2f}")


def main():
    p = argparse.ArgumentParser(description="Pure-greedy MCE(N) benchmark (no model)")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--rollouts", type=int, default=750)
    p.add_argument("--depth", type=int, default=20,
                   help="AI-turn depth per rollout. 20 = roll out to ~end of game.")
    p.add_argument("--random-seed", action="store_true")
    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
