"""CP-SAT (Google OR-Tools) version of the Cascadia upper-bound oracle.

CP-SAT is generally much faster than HiGHS/CBC for combinatorial set-packing
problems. It handles cardinality constraints natively (e.g., "at most one of
these vars is 1") and has aggressive presolve for binary problems.

Same model as ilp_upper_bound.py but using ortools.sat.python.cp_model.

Handles existing wildlife correctly: per-cell wildlife indicators over ALL
placed cells, with existing wildlife pinned to fixed values. Move budget counts
only NEW placements. Patterns (bear pairs, elk lines of length 2/3/4) defined
over the full cell set.

Salmon model (fixed 2026-04-10):
- Per-cell salmon[c] binary
- Degree constraint: each salmon has at most 2 salmon neighbors (non-branching)
- Extended staircase scoring: best_salmon_score(n) = max over partitions of n
  into chunks of length ≤ 7 of sum of SALMON_RUN_SCORE[chunk]. This correctly
  captures multi-chain layouts where 14 salmon = 2×26 = 52, etc.
"""


def _best_salmon_score_table(max_n: int = 100) -> list:
    """Compute best_salmon_score(n) for n in 0..max_n via dynamic programming.

    best_salmon_score(n) = max over partitions of n into parts <=7 of sum SALMON_RUN_SCORE[parts]

    This is the maximum score achievable by any valid spatial layout of n salmon
    cells (single chain capped at 7 = 26, multi-chain layouts add).
    """
    # Defer import — avoid circular reference
    from ilp_upper_bound import SALMON_RUN_SCORE
    dp = [0] * (max_n + 1)
    for n in range(1, max_n + 1):
        best = 0
        for k in range(1, min(n, 7) + 1):
            score = SALMON_RUN_SCORE[k] + dp[n - k]
            if score > best:
                best = score
        dp[n] = best
    return dp


_SALMON_SCORE_TABLE = _best_salmon_score_table(100)


def best_salmon_score(n: int) -> int:
    """Best achievable score for n salmon (multi-chain partition optimum)."""
    if n < 0:
        return 0
    if n >= len(_SALMON_SCORE_TABLE):
        n = len(_SALMON_SCORE_TABLE) - 1
    return _SALMON_SCORE_TABLE[n]

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

# Reuse hex grid + scoring tables from the ILP version
from ilp_upper_bound import (
    GRID_DIM, GRID_CENTER, NUM_CELLS, LINE_DIRECTIONS, NEIGHBOR_OFFSETS,
    coord_to_idx, idx_to_coord, neighbors,
    BEAR_SCORE, BEAR_MARGINAL,
    ELK_LINE_SCORE,
    SALMON_RUN_SCORE,
    HAWK_SCORE, HAWK_MARGINAL,
    BoardState, make_blank_board,
)


def upper_bound_cpsat(
    board: BoardState,
    moves_remaining: int,
    *,
    msg: bool = False,
    time_limit_s: Optional[float] = 15.0,
    num_workers: int = 8,
) -> Dict:
    """Compute the wildlife-only upper bound using CP-SAT.

    Returns a dict similar to ilp_upper_bound.ilp_upper_bound.
    """
    t_setup_start = time.time()

    all_cells_set = set(board.placed_tiles)
    all_cells = sorted(all_cells_set)
    empty_cells = all_cells_set - set(board.placed_wildlife.keys())

    if not all_cells:
        return {
            'upper_bound': 0,
            'breakdown': {'bear': 0, 'elk': 0, 'salmon': 0, 'hawk': 0, 'fox': 0},
            'n_vars': 0,
            'n_constraints': 0,
            'solver_time_s': 0.0,
        }

    model = cp_model.CpModel()

    # ── Per-cell wildlife indicators ──
    bear_v = {c: model.NewBoolVar(f"bear_{c}") for c in all_cells}
    elk_v  = {c: model.NewBoolVar(f"elk_{c}")  for c in all_cells}
    sal_v  = {c: model.NewBoolVar(f"sal_{c}")  for c in all_cells}
    haw_v  = {c: model.NewBoolVar(f"haw_{c}")  for c in all_cells}
    fox_v  = {c: model.NewBoolVar(f"fox_{c}")  for c in all_cells}
    species_vars = [bear_v, elk_v, sal_v, haw_v, fox_v]

    # ── Fix existing wildlife ──
    for c, w_id in board.placed_wildlife.items():
        if c not in all_cells_set:
            continue
        for sp_id, vars_dict in enumerate(species_vars):
            if sp_id == w_id:
                model.Add(vars_dict[c] == 1)
            else:
                model.Add(vars_dict[c] == 0)

    # ── Cell exclusivity (each cell holds at most ONE wildlife) ──
    for c in all_cells:
        model.AddAtMostOne(v[c] for v in species_vars)

    # ── Move budget: only NEW placements consume moves ──
    new_placement_terms = [v[c] for c in empty_cells for v in species_vars]
    if new_placement_terms:
        model.Add(sum(new_placement_terms) <= moves_remaining)

    # ── Bear: atomic pair vars ──
    bear_pairs: List[Tuple[int, int]] = []
    for c in all_cells:
        for n in neighbors(c):
            if n in all_cells_set and n > c:
                bear_pairs.append((c, n))
    bp_vars = [model.NewBoolVar(f"bp_{i}") for i in range(len(bear_pairs))]

    for i, (a, b) in enumerate(bear_pairs):
        # bp[i] => bear[a] AND bear[b]
        model.AddBoolAnd([bear_v[a], bear_v[b]]).OnlyEnforceIf(bp_vars[i])
        # Isolation: bp[i] => no other bear in N(a) ∪ N(b) \ {a, b}
        forbidden = (set(neighbors(a)) | set(neighbors(b))) - {a, b}
        for k in forbidden:
            if k in all_cells_set:
                model.Add(bear_v[k] == 0).OnlyEnforceIf(bp_vars[i])

    # Each cell in at most 1 bear pair (so a bear can't be "in two pairs")
    for c in all_cells:
        pairs_at_c = [bp_vars[i] for i, (a, b) in enumerate(bear_pairs) if c == a or c == b]
        if pairs_at_c:
            model.AddAtMostOne(pairs_at_c)

    # ── Elk: atomic lines of length 2, 3, 4 ──
    elk_lines_by_len: Dict[int, List[Tuple[int, ...]]] = {2: [], 3: [], 4: []}
    for length in (2, 3, 4):
        for d, (dq, dr) in enumerate(LINE_DIRECTIONS):
            for c in all_cells:
                q, r = idx_to_coord(c)
                line_cells = []
                for k in range(length):
                    ni = coord_to_idx(q + k * dq, r + k * dr)
                    if ni is None or ni not in all_cells_set:
                        break
                    line_cells.append(ni)
                if len(line_cells) == length:
                    elk_lines_by_len[length].append(tuple(line_cells))

    el_vars: Dict[int, List] = {}
    for length in (2, 3, 4):
        el_vars[length] = [
            model.NewBoolVar(f"el{length}_{i}")
            for i in range(len(elk_lines_by_len[length]))
        ]

    # Each elk line var requires all cells in the line to have elk
    for length in (2, 3, 4):
        for i, line in enumerate(elk_lines_by_len[length]):
            model.AddBoolAnd([elk_v[c] for c in line]).OnlyEnforceIf(el_vars[length][i])

    # Each elk in at most 1 line
    for c in all_cells:
        lines_at_c = []
        for length in (2, 3, 4):
            for i, line in enumerate(elk_lines_by_len[length]):
                if c in line:
                    lines_at_c.append(el_vars[length][i])
        if lines_at_c:
            model.AddAtMostOne(lines_at_c)

    # ── Hawk isolation via clique cuts (3-cell triangles) ──
    cell_set = all_cells_set
    triangles_seen: Set[Tuple[int, int, int]] = set()
    edges_in_tris: Set[Tuple[int, int]] = set()
    for c1 in all_cells:
        n1 = set(neighbors(c1)) & cell_set
        for c2 in n1:
            if c2 <= c1:
                continue
            n2 = set(neighbors(c2)) & cell_set
            for c3 in n1 & n2:
                if c3 <= c2:
                    continue
                tri = tuple(sorted((c1, c2, c3)))
                if tri not in triangles_seen:
                    triangles_seen.add(tri)
                    model.AddAtMostOne([haw_v[c1], haw_v[c2], haw_v[c3]])
                    for i in range(3):
                        for j in range(i+1, 3):
                            edges_in_tris.add(tuple(sorted((tri[i], tri[j]))))
    # Edges not in any triangle
    for c in all_cells:
        for n in neighbors(c):
            if n in cell_set and n > c:
                if (c, n) not in edges_in_tris:
                    model.AddAtMostOne([haw_v[c], haw_v[n]])

    # ── Bear staircase scoring on number of pairs ──
    pair_count = sum(bp_vars)
    I_bear = [model.NewBoolVar(f"I_bear_{k}") for k in range(1, 5)]
    for k in range(1, 5):
        # I_bear[k-1] = 1 iff pair_count >= k
        # Use linearization: pair_count >= k * I_bear[k-1]
        model.Add(pair_count >= k * I_bear[k-1])
    # Monotonicity
    for k in range(1, 4):
        model.Add(I_bear[k] <= I_bear[k-1])
    # Bear score: sum of marginal[k] * I_bear[k-1]
    bear_score_terms = [BEAR_MARGINAL[k] * I_bear[k-1] for k in range(1, 5)]

    # ── Elk: line scores + isolated ──
    line_2_count = sum(el_vars[2])
    line_3_count = sum(el_vars[3])
    line_4_count = sum(el_vars[4])
    elk_total = sum(elk_v[c] for c in all_cells)
    # isolated_elks = elk_total - 2*line_2 - 3*line_3 - 4*line_4
    elk_score_terms = [
        ELK_LINE_SCORE[2] * line_2_count,
        ELK_LINE_SCORE[3] * line_3_count,
        ELK_LINE_SCORE[4] * line_4_count,
        ELK_LINE_SCORE[1] * (elk_total - 2 * line_2_count - 3 * line_3_count - 4 * line_4_count),
    ]

    # ── Salmon non-branching constraint ──
    # Each salmon has at most 2 salmon neighbors (Cascadia rule: chains are
    # paths or cycles, not branching trees).
    # Linearization: sum_neighbor_salmons <= 2 + (max_nbrs - 2) * (1 - sal_v[c])
    # When sal_v[c]=1: sum <= 2 (forbids branching)
    # When sal_v[c]=0: sum <= max_nbrs (no constraint)
    for c in all_cells:
        nbrs = [n for n in neighbors(c) if n in all_cells_set]
        if len(nbrs) >= 3:
            max_nbrs = len(nbrs)
            model.Add(
                sum(sal_v[n] for n in nbrs)
                <= 2 + (max_nbrs - 2) * (1 - sal_v[c])
            )

    # ── Salmon extended staircase scoring (multi-chain, capped at 9) ──
    # SCORE = best_salmon_score(salmon_count) where best_salmon_score is computed
    # via DP over partitions into chunks of length ≤ 7.
    # Domain assumption: max 9 salmon per player (realistic cap given 20 in bag,
    # 4 players, draft dynamics). Also bounds the indicator variable count.
    MAX_SALMON_PER_PLAYER = 9
    salmon_count = sum(sal_v[c] for c in all_cells)
    model.Add(salmon_count <= MAX_SALMON_PER_PLAYER)
    max_salmon = MAX_SALMON_PER_PLAYER
    I_salmon = [model.NewBoolVar(f"I_salmon_{k}") for k in range(1, max_salmon + 1)]
    for k in range(1, max_salmon + 1):
        model.Add(salmon_count >= k * I_salmon[k-1])
    for k in range(1, max_salmon):
        model.Add(I_salmon[k] <= I_salmon[k-1])
    # Marginals: best_salmon_score(k) - best_salmon_score(k-1)
    salmon_score_terms = []
    for k in range(1, max_salmon + 1):
        marginal = best_salmon_score(k) - best_salmon_score(k - 1)
        if marginal != 0:
            salmon_score_terms.append(marginal * I_salmon[k-1])

    # ── Hawk staircase scoring ──
    hawk_count = sum(haw_v[c] for c in all_cells)
    I_hawk = [model.NewBoolVar(f"I_hawk_{k}") for k in range(1, 9)]
    for k in range(1, 9):
        model.Add(hawk_count >= k * I_hawk[k-1])
    for k in range(1, 8):
        model.Add(I_hawk[k] <= I_hawk[k-1])
    hawk_score_terms = [HAWK_MARGINAL[k] * I_hawk[k-1] for k in range(1, 9)]

    # ── Fox: per-fox spatial adjacency check ──
    SPECIES = ["bear", "elk", "salmon", "hawk", "fox"]
    sp_vars_by_name = {
        "bear": bear_v, "elk": elk_v, "salmon": sal_v, "hawk": haw_v, "fox": fox_v,
    }
    fox_sees_terms = []
    for c in all_cells:
        nbrs = [n for n in neighbors(c) if n in cell_set]
        if not nbrs:
            continue
        for sp in SPECIES:
            fs = model.NewBoolVar(f"fs_{c}_{sp}")
            # fs implies fox at c
            model.Add(fox_v[c] == 1).OnlyEnforceIf(fs)
            # fs implies at least one neighbor has sp
            # fs <= sum(neighbors with sp)
            model.Add(sum(sp_vars_by_name[sp][n] for n in nbrs) >= 1).OnlyEnforceIf(fs)
            fox_sees_terms.append(fs)

    # ── Objective ──
    objective_terms = (
        bear_score_terms
        + elk_score_terms
        + salmon_score_terms
        + hawk_score_terms
        + fox_sees_terms
    )
    model.Maximize(sum(objective_terms))

    n_vars = -1  # CP-SAT doesn't expose a simple var count
    n_constraints = -1
    setup_time = time.time() - t_setup_start

    # ── Solve ──
    solver = cp_model.CpSolver()
    if time_limit_s is not None:
        solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers
    if msg:
        solver.parameters.log_search_progress = True

    t_solve_start = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - t_solve_start

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            'upper_bound': -1,
            'status': solver.StatusName(status),
            'n_vars': n_vars,
            'n_constraints': n_constraints,
            'setup_time_s': setup_time,
            'solver_time_s': solve_time,
        }

    # Extract counts
    bp_val = sum(solver.Value(v) for v in bp_vars)
    el2_val = sum(solver.Value(v) for v in el_vars[2])
    el3_val = sum(solver.Value(v) for v in el_vars[3])
    el4_val = sum(solver.Value(v) for v in el_vars[4])
    elk_total_val = sum(solver.Value(elk_v[c]) for c in all_cells)
    isolated_elks_val = elk_total_val - 2 * el2_val - 3 * el3_val - 4 * el4_val
    salmon_val = sum(solver.Value(sal_v[c]) for c in all_cells)
    hawk_val = sum(solver.Value(haw_v[c]) for c in all_cells)
    fox_val = sum(solver.Value(fox_v[c]) for c in all_cells)
    bear_total_val = sum(solver.Value(bear_v[c]) for c in all_cells)
    fox_score_val = sum(solver.Value(fs) for fs in fox_sees_terms)

    elk_score_val = (
        ELK_LINE_SCORE[2] * el2_val
        + ELK_LINE_SCORE[3] * el3_val
        + ELK_LINE_SCORE[4] * el4_val
        + ELK_LINE_SCORE[1] * isolated_elks_val
    )

    breakdown = {
        'bear': BEAR_SCORE.get(min(bp_val, 4), BEAR_SCORE[4]),
        'elk': elk_score_val,
        'salmon': best_salmon_score(salmon_val),
        'hawk': HAWK_SCORE.get(min(hawk_val, 8), HAWK_SCORE[8]),
        'fox': fox_score_val,
        'counts': {
            'bear_pairs': bp_val,
            'bear_total': bear_total_val,
            'elk_total': elk_total_val,
            'elk_lines_2': el2_val,
            'elk_lines_3': el3_val,
            'elk_lines_4': el4_val,
            'elk_isolated': isolated_elks_val,
            'salmon_cells': salmon_val,
            'hawks': hawk_val,
            'foxes': fox_val,
            'fox_score_per_fox': (fox_score_val / fox_val) if fox_val > 0 else 0,
        },
    }

    return {
        'upper_bound': int(round(solver.ObjectiveValue())),
        'wildlife_only_bound': int(round(solver.ObjectiveValue())),
        'status': solver.StatusName(status),
        'breakdown': breakdown,
        'setup_time_s': setup_time,
        'solver_time_s': solve_time,
        'best_objective_bound': solver.BestObjectiveBound(),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--moves', type=int, default=20)
    parser.add_argument('--time-limit', type=float, default=20)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print(f"=== CP-SAT Upper Bound — wildlife-only ===")
    board = make_blank_board(args.radius)
    print(f"  Cells: {len(board.placed_tiles)}")
    print(f"  Moves: {args.moves}")
    print()

    print("Solving...")
    result = upper_bound_cpsat(board, args.moves,
                                time_limit_s=args.time_limit, msg=args.verbose)
    print(f"  Status: {result.get('status')}")
    print(f"  Upper bound: {result.get('upper_bound')}")
    print(f"  Best obj bound (proven): {result.get('best_objective_bound')}")
    print(f"  Setup: {result['setup_time_s']:.2f}s, Solver: {result['solver_time_s']:.2f}s")
    if 'breakdown' in result:
        bd = result['breakdown']
        print(f"  Breakdown:")
        print(f"    Bear:    {bd['bear']:3} ({bd['counts']['bear_pairs']} pairs, {bd['counts']['bear_total']} total)")
        print(f"    Elk:     {bd['elk']:3} (lines: 2={bd['counts']['elk_lines_2']} 3={bd['counts']['elk_lines_3']} 4={bd['counts']['elk_lines_4']}, iso={bd['counts']['elk_isolated']})")
        print(f"    Salmon:  {bd['salmon']:3} ({bd['counts']['salmon_cells']} cells)")
        print(f"    Hawk:    {bd['hawk']:3} ({bd['counts']['hawks']} isolated)")
        print(f"    Fox:     {bd['fox']:3} ({bd['counts']['foxes']} foxes, {bd['counts']['fox_score_per_fox']:.1f} avg)")
