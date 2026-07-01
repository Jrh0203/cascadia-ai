"""Experiment: how much of the upper-bound score does the AI actually capture?

Process:
1. Load v3 self-play data (training_merged_iterN.bin) — features per sample
2. For each game (every 20 samples), grab the LAST sample = final game state
3. Decode features → (placed_tiles, placed_wildlife)
4. Compute the UB ORACLE on the SAME tile layout but with wildlife re-optimized
   (i.e. "given the tiles you ended up with, what's the max wildlife score
    achievable?")
5. Compare to the AI's actual wildlife score
6. Per wildlife type: capture % = actual / optimal_for_that_type
7. Aggregate across games

This reveals which animals the AI is leaving the most points on the table for.

Usage:
    python3 experiment_capture_rate.py [--samples training_merged_iter15.bin] [--n 20]
"""

import argparse
import struct
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# Reuse helpers from the upper bound module
from ilp_upper_bound import (
    BoardState, neighbors, coord_to_idx, idx_to_coord,
    BEAR_SCORE, ELK_LINE_SCORE, SALMON_RUN_SCORE, HAWK_SCORE,
)
from ilp_upper_bound_cpsat import upper_bound_cpsat, best_salmon_score


# ─────────────────────────────────────────────────────────────────────
# Feature decoder (matches crates/cascadia-ai/src/nnue.rs feature layout)
# ─────────────────────────────────────────────────────────────────────

FEATURES_PER_CELL = 11
NUM_CELLS = 441
CELL_FEATURES = NUM_CELLS * FEATURES_PER_CELL  # 4851

# Wildlife indices (must match crates/cascadia-core/src/types.rs Wildlife enum)
WILDLIFE_NAMES = ["Bear", "Elk", "Salmon", "Hawk", "Fox"]


def decode_board_from_features(feature_indices: List[int]) -> Tuple[Set[int], Dict[int, int]]:
    """Extract (placed_tiles, placed_wildlife) from a sparse feature list.

    Per-cell features (cell_base = idx * 11):
        offset 0-4: wildlife placed (Bear/Elk/Salmon/Hawk/Fox)
        offset 5:   tile present, no wildlife
        offset 6-10: primary terrain (Forest/Prairie/Wetland/Mountain/River)

    A cell is "placed" if any feature in [idx*11, idx*11 + 11) fires.
    """
    placed_tiles: Set[int] = set()
    placed_wildlife: Dict[int, int] = {}
    for fi in feature_indices:
        if fi >= CELL_FEATURES:
            continue
        idx = fi // FEATURES_PER_CELL
        offset = fi % FEATURES_PER_CELL
        placed_tiles.add(idx)
        if offset < 5:
            placed_wildlife[idx] = offset  # 0=Bear, 1=Elk, ...
    return placed_tiles, placed_wildlife


# ─────────────────────────────────────────────────────────────────────
# Cascadia scoring in Python (matches crates/cascadia-core/src/scoring/wildlife/*.rs)
# ─────────────────────────────────────────────────────────────────────

def score_bear_pairs(positions: Set[int]) -> int:
    """Bear Card A: count of isolated pairs (size-2 components)."""
    visited = set()
    pairs = 0
    for pos in positions:
        if pos in visited:
            continue
        # BFS the bear component
        component = set()
        queue = [pos]
        visited.add(pos)
        while queue:
            cur = queue.pop()
            component.add(cur)
            for n in neighbors(cur):
                if n in positions and n not in visited:
                    visited.add(n)
                    queue.append(n)
        if len(component) == 2:
            pairs += 1
    return BEAR_SCORE.get(min(pairs, 4), BEAR_SCORE[4])


def score_elk_lines(positions: Set[int]) -> int:
    """Elk Card A: greedy line assignment, score per line by length (cap 4)."""
    if not positions:
        return 0
    from ilp_upper_bound import LINE_DIRECTIONS

    is_elk = positions
    used = set()
    score = 0

    # Find connected components first
    visited = set()
    components = []
    for pos in is_elk:
        if pos in visited:
            continue
        comp = set()
        queue = [pos]
        visited.add(pos)
        while queue:
            cur = queue.pop()
            comp.add(cur)
            for n in neighbors(cur):
                if n in is_elk and n not in visited:
                    visited.add(n)
                    queue.append(n)
        components.append(comp)

    for component in components:
        # Find all maximal lines in 3 directions
        comp_used = set()
        all_lines: List[List[int]] = []
        in_line = {d: set() for d in range(3)}
        for d, (dq, dr) in enumerate(LINE_DIRECTIONS):
            for pos in component:
                if pos in in_line[d]:
                    continue
                # Walk back to start
                q, r = idx_to_coord(pos)
                while True:
                    prev_q, prev_r = q - dq, r - dr
                    pidx = coord_to_idx(prev_q, prev_r)
                    if pidx is not None and pidx in component:
                        q, r = prev_q, prev_r
                    else:
                        break
                # Walk forward
                line = []
                while True:
                    cidx = coord_to_idx(q, r)
                    if cidx is not None and cidx in component:
                        line.append(cidx)
                        in_line[d].add(cidx)
                        q, r = q + dq, r + dr
                    else:
                        break
                if line:
                    all_lines.append(line)

        # Sort by length desc, greedy fill
        all_lines.sort(key=len, reverse=True)
        for line in all_lines:
            best_run = 0
            current_run = 0
            for p in line:
                if p not in comp_used:
                    current_run += 1
                    best_run = max(best_run, current_run)
                else:
                    current_run = 0
            if best_run >= 1:
                # Mark first best_run unused
                count = 0
                for p in line:
                    if p not in comp_used and count < best_run:
                        comp_used.add(p)
                        count += 1
                    if count >= best_run:
                        break
                score += min(ELK_LINE_SCORE[min(best_run, 4)], ELK_LINE_SCORE[4])
        # Remaining elks score as singles
        for p in component:
            if p not in comp_used:
                score += ELK_LINE_SCORE[1]
    return score


def score_salmon_runs(positions: Set[int]) -> int:
    """Salmon Card A: connected components where each cell has degree ≤ 2."""
    if not positions:
        return 0
    visited = set()
    total = 0
    for pos in positions:
        if pos in visited:
            continue
        component = set()
        queue = [pos]
        visited.add(pos)
        while queue:
            cur = queue.pop()
            component.add(cur)
            for n in neighbors(cur):
                if n in positions and n not in visited:
                    visited.add(n)
                    queue.append(n)
        # Check degree ≤ 2
        valid = all(
            sum(1 for n in neighbors(p) if n in positions) <= 2
            for p in component
        )
        if valid:
            n_cells = len(component)
            total += SALMON_RUN_SCORE.get(min(n_cells, 7), SALMON_RUN_SCORE[7])
    return total


def score_hawks(positions: Set[int]) -> int:
    """Hawk Card A: count of isolated hawks (no hawk neighbors)."""
    isolated = 0
    for pos in positions:
        if not any(n in positions for n in neighbors(pos)):
            isolated += 1
    return HAWK_SCORE.get(min(isolated, 8), HAWK_SCORE[8])


def score_foxes(positions: Set[int], all_wildlife: Dict[int, int]) -> int:
    """Fox Card A: each fox = unique wildlife species count among neighbors."""
    total = 0
    for pos in positions:
        species_seen = set()
        for n in neighbors(pos):
            if n in all_wildlife:
                species_seen.add(all_wildlife[n])
        total += len(species_seen)
    return total


def score_actual_board(placed_wildlife: Dict[int, int]) -> Dict[str, int]:
    """Compute per-species score breakdown for an actual board."""
    by_species = defaultdict(set)
    for c, w in placed_wildlife.items():
        by_species[w].add(c)
    return {
        'bear':   score_bear_pairs(by_species[0]),
        'elk':    score_elk_lines(by_species[1]),
        'salmon': score_salmon_runs(by_species[2]),
        'hawk':   score_hawks(by_species[3]),
        'fox':    score_foxes(by_species[4], placed_wildlife),
    }


# ─────────────────────────────────────────────────────────────────────
# Sample loader (MCEP format)
# ─────────────────────────────────────────────────────────────────────

MCEP_MAGIC = b'MCEP'
MCV2_MAGIC = b'MCV2'


def load_game_final_states(path: str, n_games: int):
    """Load the FINAL sample of each game from a training_merged file.
    Each game has 20 samples (one per AI turn). Returns list of feature lists.
    Auto-detects MCEP (v1) or MCV2 (v2 with aux targets).
    """
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic == MCEP_MAGIC:
            extra_per_sample = 0
        elif magic == MCV2_MAGIC:
            extra_per_sample = 8  # 2 floats: aux_bear, aux_salmon
        else:
            raise ValueError(f"Bad magic: {magic}")
        data = f.read()
    pos = 0
    n = len(data)
    games_loaded = 0
    sample_idx = 0
    samples_per_game = 20
    final_states: List[List[int]] = []

    current_game_samples: List[List[int]] = []

    while pos + 2 <= n and games_loaded < n_games:
        nf = struct.unpack_from('<H', data, pos)[0]
        pos += 2
        if nf > 1024 or pos + nf * 2 + 4 + extra_per_sample > n:
            break
        features = list(struct.unpack_from(f'<{nf}H', data, pos))
        pos += nf * 2
        target = struct.unpack_from('<f', data, pos)[0]
        pos += 4 + extra_per_sample  # skip aux floats if MCV2

        current_game_samples.append(features)
        sample_idx += 1
        if sample_idx == samples_per_game:
            # Take the last sample as the final state
            final_states.append(current_game_samples[-1])
            current_game_samples = []
            sample_idx = 0
            games_loaded += 1

    return final_states


# ─────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='training_merged_iter15.bin')
    parser.add_argument('--n', type=int, default=10, help='Number of games to analyze')
    parser.add_argument('--time-limit', type=float, default=10)
    args = parser.parse_args()

    print(f"=== Capture rate experiment ===")
    print(f"Loading: {args.samples}")
    print(f"Games to analyze: {args.n}")
    print()

    final_states = load_game_final_states(args.samples, args.n)
    print(f"Loaded {len(final_states)} game final states")
    print()

    aggregated = {
        'actual': defaultdict(list),
        'optimal': defaultdict(list),
        'cells': [],
        'wildlife_count': [],
    }

    for game_idx, features in enumerate(final_states):
        placed_tiles, placed_wildlife = decode_board_from_features(features)
        actual = score_actual_board(placed_wildlife)
        actual_total = sum(actual.values())

        # Build a board state with the SAME tiles but EMPTY wildlife
        # Then compute UB assuming we can re-place wildlife optimally over R moves.
        # R = number of cells available for wildlife (ALL cells in the layout).
        # In real games, AI places one wildlife per turn → R = 20 by default.
        # We optimize the FULL board's wildlife layout from scratch.
        ub_board = BoardState(
            placed_tiles=placed_tiles,
            placed_wildlife={},  # empty — re-optimize
            allowed_mask={c: 0b11111 for c in placed_tiles},
            current_score=0,
        )
        moves_avail = min(20, len(placed_tiles))

        t0 = time.time()
        result = upper_bound_cpsat(ub_board, moves_avail, time_limit_s=args.time_limit)
        elapsed = time.time() - t0

        ub = result['upper_bound']
        bd = result['breakdown']

        capture_pct = (actual_total / ub * 100) if ub > 0 else 0
        print(f"Game {game_idx+1:2}: cells={len(placed_tiles):2} wl={len(placed_wildlife):2}  "
              f"actual={actual_total:3} ub={ub:3} capture={capture_pct:5.1f}%  ({elapsed:.1f}s)")
        print(f"          actual:  B{actual['bear']:2} E{actual['elk']:2} S{actual['salmon']:2} H{actual['hawk']:2} F{actual['fox']:2}")
        print(f"          optimal: B{bd['bear']:2} E{bd['elk']:2} S{bd['salmon']:2} H{bd['hawk']:2} F{bd['fox']:2}")

        for sp in ('bear', 'elk', 'salmon', 'hawk', 'fox'):
            aggregated['actual'][sp].append(actual[sp])
            aggregated['optimal'][sp].append(bd[sp])
        aggregated['cells'].append(len(placed_tiles))
        aggregated['wildlife_count'].append(len(placed_wildlife))

    print()
    print("=== Aggregate per-species capture (across all games) ===")
    print(f"{'species':<8} {'actual_avg':<12} {'optimal_avg':<12} {'capture_%':<10} {'gap':<8}")
    for sp in ('bear', 'elk', 'salmon', 'hawk', 'fox'):
        a_vals = aggregated['actual'][sp]
        o_vals = aggregated['optimal'][sp]
        a_avg = sum(a_vals) / len(a_vals) if a_vals else 0
        o_avg = sum(o_vals) / len(o_vals) if o_vals else 0
        cap = (a_avg / o_avg * 100) if o_avg > 0 else 0
        gap = o_avg - a_avg
        print(f"{sp:<8} {a_avg:<12.1f} {o_avg:<12.1f} {cap:<10.1f} {gap:<8.1f}")

    total_actual_avg = sum(sum(v) for v in aggregated['actual'].values()) / len(final_states)
    total_optimal_avg = sum(sum(v) for v in aggregated['optimal'].values()) / len(final_states)
    print()
    print(f"Total actual avg:  {total_actual_avg:.1f}")
    print(f"Total optimal avg: {total_optimal_avg:.1f}")
    print(f"Overall capture %: {total_actual_avg/total_optimal_avg*100:.1f}%")
    print(f"Avg cells/game:    {sum(aggregated['cells'])/len(aggregated['cells']):.1f}")
    print(f"Avg wildlife/game: {sum(aggregated['wildlife_count'])/len(aggregated['wildlife_count']):.1f}")


if __name__ == '__main__':
    main()
