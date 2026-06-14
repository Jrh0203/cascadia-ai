#!/usr/bin/env python3
"""Parse a training_merged_iterN.bin file and extract per-game self-play stats.

The file is in MCEP format: magic 'MCEP' followed by sequential samples.
Each sample: u16 nf, nf × u16 features, f32 target.

Each game has exactly 20 samples (one per AI turn). The target on sample i is
`final_score - cumulative_score_at_turn_i`. So:
- Sample 0 of a game: target = final_score - score_after_move_1
- Sample 19 (last) of a game: target = 0

We use first-sample-target + offset_estimate as a proxy for final_score.
For the relative comparison across iterations, the offset cancels out.
"""

import struct
import sys
import os
from pathlib import Path
import numpy as np

MCEP_MAGIC = b'MCEP'
SAMPLES_PER_GAME = 20  # 4-player Cascadia, AI is player 0, AI gets 20 turns


def parse_mcep_file(path):
    """Yield (features, target) tuples from an MCEP file."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != MCEP_MAGIC:
            raise ValueError(f"Bad magic in {path}: {magic}")
        # Read all bytes for fast iteration
        data = f.read()
    pos = 0
    n = len(data)
    while pos + 2 <= n:
        nf = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        if nf > 1024 or pos + nf * 2 + 4 > n:
            break
        feature_bytes = nf * 2
        # Skip the features (we don't need them)
        pos += feature_bytes
        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4
        yield target


def stats_from_file(path):
    """Compute per-game stats. Returns dict of stats."""
    targets = list(parse_mcep_file(path))
    n_samples = len(targets)
    n_games = n_samples // SAMPLES_PER_GAME

    # The "final score proxy" = first target of each game + small offset
    # First target = final_score - score_after_move_1 ≈ final_score - 4
    # We use the raw first-target value; offset is constant across iters
    first_targets = [targets[i * SAMPLES_PER_GAME] for i in range(n_games)]
    arr = np.array(first_targets, dtype=np.float32)

    return {
        'n_games': n_games,
        'mean_first_target': float(arr.mean()),
        'p10_first_target': float(np.percentile(arr, 10)),
        'median_first_target': float(np.percentile(arr, 50)),
        'p90_first_target': float(np.percentile(arr, 90)),
        'min_first_target': float(arr.min()),
        'max_first_target': float(arr.max()),
        # The "final score proxy" adds ~4 (typical score after move 1)
        'estimated_mean_final_score': float(arr.mean()) + 4.0,
    }


def main():
    if len(sys.argv) < 2:
        # Auto-find all v3 merged files
        files = sorted(Path('.').glob('training_merged_iter*.bin'))
    else:
        files = [Path(p) for p in sys.argv[1:]]

    print(f"{'iter':<6} {'n_games':<8} {'mean':<8} {'p10':<6} {'median':<8} {'p90':<6} {'max':<6} {'est_final':<10}")
    print('-' * 70)
    for path in files:
        # Extract iter number from filename
        try:
            iter_n = int(path.stem.replace('training_merged_iter', ''))
        except ValueError:
            iter_n = -1
        try:
            stats = stats_from_file(path)
        except Exception as e:
            print(f"{iter_n:<6} ERROR: {e}")
            continue
        print(f"{iter_n:<6} {stats['n_games']:<8} "
              f"{stats['mean_first_target']:<8.2f} "
              f"{stats['p10_first_target']:<6.0f} "
              f"{stats['median_first_target']:<8.0f} "
              f"{stats['p90_first_target']:<6.0f} "
              f"{stats['max_first_target']:<6.0f} "
              f"{stats['estimated_mean_final_score']:<10.2f}")


if __name__ == '__main__':
    main()
