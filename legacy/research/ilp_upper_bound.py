"""ILP upper-bound oracle for Cascadia.

Computes the maximum achievable score from a given board state, assuming the
player can pick any tile and any wildlife for each remaining move. This is an
UPPER BOUND because we relax bag constraints and allow per-cell wildlife choice.

Uses the wildlife-only formulation (ignores habitat scoring) for tractability.
Card A scoring tables verified against crates/cascadia-core/src/scoring/wildlife/*.rs.

Atomic units (per user's insight):
- bear_pair: 2-cell binary variable, exactly 2 adjacent bears = a "pair"
- elk_line4: 4-cell binary variable, line of exactly 4 elks = 13 points
- salmon[c]: per-cell binary; staircase scoring on total count
- hawk[c]: per-cell binary, must not be adjacent to other hawks
- fox[c]: per-cell binary; scores per-fox count of distinct adjacent species

Tightening techniques applied (researched 2026-04-10):
- Indicator monotonicity for staircase scoring (I[k] <= I[k-1])
- Clique cuts for hawk isolation (3-cell hex triangles instead of pairwise)
- Tight fox bound: per-fox score requires real spatial adjacency to species
- HiGHS performance options: parallel=on, threads=8, mip_heuristic_effort=0.8

Known limitations:
- **Pre-placed wildlife is treated as frozen baseline** (current_score). The ILP
  doesn't model synergies between new and existing wildlife (e.g., new salmon
  adjacent to existing chain extends it; new hawks adjacent to existing hawks
  break isolation). Use only on EMPTY boards or START-of-game states for
  strict correctness. For mid-game states the bound is approximate.
- Wildlife-only formulation ignores habitat (largest-group) scoring.
- LP relaxation is loose (~50% over true optimum) — use MIP path.
- Solver scales poorly past ~40 cells; default 15s time limit + 5% gap is the
  practical sweet spot for most realistic scenarios.

Usage:
    python3 ilp_upper_bound.py                    # run built-in test cases
"""

import argparse
import time
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pulp


# ─────────────────────────────────────────────────────────────────────
# Hex grid (matches crates/cascadia-core/src/hex.rs)
# ─────────────────────────────────────────────────────────────────────

GRID_DIM = 21
GRID_CENTER = 10  # axial coords range from -10 to +10
NUM_CELLS = GRID_DIM * GRID_DIM  # 441

# Three "line" directions in hex (axial). Each goes both ways through a cell.
# These match LINE_DIRECTIONS in hex.rs.
LINE_DIRECTIONS = [(1, 0), (0, 1), (-1, 1)]  # E, SE, SW

# All 6 neighbors of a hex (3 line directions × 2 orientations)
NEIGHBOR_OFFSETS = [
    (1, 0), (-1, 0),
    (0, 1), (0, -1),
    (-1, 1), (1, -1),
]


def coord_to_idx(q: int, r: int) -> Optional[int]:
    """Convert axial (q, r) to flat cell index, or None if out of bounds."""
    qi = q + GRID_CENTER
    ri = r + GRID_CENTER
    if 0 <= qi < GRID_DIM and 0 <= ri < GRID_DIM:
        return qi * GRID_DIM + ri
    return None


def idx_to_coord(idx: int) -> Tuple[int, int]:
    """Convert flat cell index back to axial (q, r)."""
    qi = idx // GRID_DIM
    ri = idx % GRID_DIM
    return (qi - GRID_CENTER, ri - GRID_CENTER)


def neighbors(idx: int) -> List[int]:
    """Return all 6 neighbors of cell idx that are in bounds."""
    q, r = idx_to_coord(idx)
    out = []
    for dq, dr in NEIGHBOR_OFFSETS:
        ni = coord_to_idx(q + dq, r + dr)
        if ni is not None:
            out.append(ni)
    return out


# ─────────────────────────────────────────────────────────────────────
# Card A scoring (verified from crates/cascadia-core/src/scoring/wildlife/*.rs)
# ─────────────────────────────────────────────────────────────────────

# Bear: count of pairs → score
BEAR_SCORE = {0: 0, 1: 4, 2: 11, 3: 19, 4: 27}  # 4+ caps at 27
# Marginal contribution of going from k-1 pairs to k pairs:
BEAR_MARGINAL = {1: 4, 2: 7, 3: 8, 4: 8}

# Elk: line length → per-line score (4+ caps at 13)
ELK_LINE_SCORE = {1: 2, 2: 5, 3: 9, 4: 13}  # we only enumerate length-4 atomic units

# Salmon: run length → per-run score (7+ caps at 26)
SALMON_RUN_SCORE = {1: 2, 2: 4, 3: 7, 4: 11, 5: 15, 6: 20, 7: 26}

# Hawk: count of isolated hawks → score
HAWK_SCORE = {0: 0, 1: 2, 2: 5, 3: 8, 4: 11, 5: 14, 6: 18, 7: 22, 8: 28}
HAWK_MARGINAL = {1: 2, 2: 3, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 6}

# Fox: per-fox, score = number of unique adjacent species (0..5)
# In the wildlife-only relaxation, each fox can theoretically see all 5 species
# nearby. We use 5 as a loose upper bound per fox. A tighter formulation would
# explicitly check neighborhood placements (left as a future refinement).
FOX_MAX_PER_FOX = 5


# ─────────────────────────────────────────────────────────────────────
# Board state (parsed or constructed)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BoardState:
    """A snapshot of a Cascadia board sufficient for ILP upper-bound computation.

    For the wildlife-only relaxation, we only need:
    - The set of cells that have tiles (placed_tiles)
    - Which of those have wildlife already placed (placed_wildlife)
    - The allowed wildlife mask per cell (allowed_mask)
    - Existing wildlife scores (used as starting score)
    """
    placed_tiles: Set[int] = field(default_factory=set)
    placed_wildlife: Dict[int, int] = field(default_factory=dict)  # cell -> wildlife (0..4)
    allowed_mask: Dict[int, int] = field(default_factory=dict)     # cell -> 5-bit mask
    current_score: int = 0  # known starting score (from outside)


# ─────────────────────────────────────────────────────────────────────
# Pattern enumeration helpers
# ─────────────────────────────────────────────────────────────────────

def enumerate_bear_pairs(cells: Set[int]) -> List[Tuple[int, int]]:
    """All ordered pairs (a, b) with a < b that are adjacent in the cell set."""
    pairs = []
    for c in cells:
        for n in neighbors(c):
            if n in cells and n > c:
                pairs.append((c, n))
    return pairs


def enumerate_elk_lines4(cells: Set[int]) -> List[Tuple[int, int, int, int, int]]:
    """All length-4 lines in 3 hex line directions. Returns (c1, c2, c3, c4, dir)."""
    lines = []
    for d, (dq, dr) in enumerate(LINE_DIRECTIONS):
        for c in cells:
            q, r = idx_to_coord(c)
            line_cells = []
            for k in range(4):
                ni = coord_to_idx(q + k * dq, r + k * dr)
                if ni is None or ni not in cells:
                    break
                line_cells.append(ni)
            if len(line_cells) == 4:
                lines.append((line_cells[0], line_cells[1], line_cells[2], line_cells[3], d))
    return lines


def enumerate_salmon_chains(cells: Set[int], max_k: int = 7) -> List[Tuple[int, ...]]:
    """All simple paths (chains) of length 1..max_k in the cell adjacency graph.
    A chain has no branching internally — each interior cell has exactly 2 neighbors
    in the chain.

    For tractability, we enumerate paths starting at each cell and extend up to
    max_k cells. We deduplicate by canonicalizing (smallest endpoint first).

    NOTE: This grows quickly with max_k. For dense regions and max_k=7, can be
    millions of paths. Caller should use only reasonable cell sets (~30-50 cells).
    """
    chains: Set[Tuple[int, ...]] = set()

    def canonical(path: Tuple[int, ...]) -> Tuple[int, ...]:
        # Reverse if last < first; treat (a..b) and (b..a) as the same chain
        if len(path) >= 2 and path[-1] < path[0]:
            return tuple(reversed(path))
        return path

    def dfs(path: List[int], visited: Set[int]):
        chains.add(canonical(tuple(path)))
        if len(path) >= max_k:
            return
        last = path[-1]
        for n in neighbors(last):
            if n not in visited and n in cells:
                # Avoid cycles by only extending if new cell isn't already in path
                visited.add(n)
                path.append(n)
                dfs(path, visited)
                path.pop()
                visited.remove(n)

    for start in cells:
        dfs([start], {start})

    return [tuple(c) for c in chains]


# ─────────────────────────────────────────────────────────────────────
# ILP solver
# ─────────────────────────────────────────────────────────────────────

def ilp_upper_bound(
    board: BoardState,
    moves_remaining: int,
    *,
    wildlife_only: bool = True,
    msg: bool = False,
    time_limit_s: Optional[float] = 15.0,
    mip_gap: Optional[float] = 0.05,
    lp_relax: bool = False,
) -> Dict:
    """Compute the wildlife-only ILP upper bound.

    v2 design (handles pre-placed wildlife correctly):
    - Per-cell wildlife indicator vars over ALL placed cells (existing + empty)
    - Existing wildlife is FIXED to its current value via equality constraints
    - Atomic units (bear pairs, elk lines of length 1/2/3/4) defined over all cells
    - Existing wildlife participates in patterns naturally; constraints
      (isolation, etc.) automatically enforce correctness
    - Move budget counts only NEW placements

    Returns a dict with:
        upper_bound: int — the total max score (existing + new)
        breakdown: dict per-pattern point allocation
        n_vars: int
        n_constraints: int
        solver_time_s: float
    """
    t_setup_start = time.time()

    # All placed cells (existing + empty)
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

    prob = pulp.LpProblem("cascadia_ub_v2", pulp.LpMaximize)

    # ── Per-cell wildlife indicator variables ──
    bear_v = pulp.LpVariable.dicts("bear", all_cells, cat='Binary')
    elk_v  = pulp.LpVariable.dicts("elk",  all_cells, cat='Binary')
    sal_v  = pulp.LpVariable.dicts("sal",  all_cells, cat='Binary')
    haw_v  = pulp.LpVariable.dicts("haw",  all_cells, cat='Binary')
    fox_v  = pulp.LpVariable.dicts("fox",  all_cells, cat='Binary')
    species_vars = [bear_v, elk_v, sal_v, haw_v, fox_v]

    # ── Fix existing wildlife placements ──
    for c, w_id in board.placed_wildlife.items():
        if c not in all_cells_set:
            continue
        for sp_id, vars_dict in enumerate(species_vars):
            if sp_id == w_id:
                prob += vars_dict[c] == 1
            else:
                prob += vars_dict[c] == 0

    # ── Cell exclusivity (each cell holds at most ONE wildlife) ──
    for c in all_cells:
        prob += pulp.lpSum(v[c] for v in species_vars) <= 1

    # ── Move budget: count only NEW placements ──
    new_placement_count = pulp.lpSum(
        v[c] for c in empty_cells for v in species_vars
    )
    prob += new_placement_count <= moves_remaining

    # ── Bear: atomic pair variables ──
    bear_pairs = enumerate_bear_pairs(all_cells_set)
    bp = pulp.LpVariable.dicts("bp", range(len(bear_pairs)), cat='Binary')

    # bear_pair[i,j] requires bear[i] = bear[j] = 1
    for i, (a, b) in enumerate(bear_pairs):
        prob += bp[i] <= bear_v[a]
        prob += bp[i] <= bear_v[b]
        # Isolation: no other bear in N(a) ∪ N(b) \ {a, b}
        forbidden = (set(neighbors(a)) | set(neighbors(b))) - {a, b}
        for k in forbidden:
            if k in all_cells_set:
                prob += bp[i] + bear_v[k] <= 1

    # Each cell in at most 1 bear pair
    for c in all_cells:
        pairs_at_c = [bp[i] for i, (a, b) in enumerate(bear_pairs) if c == a or c == b]
        if pairs_at_c:
            prob += pulp.lpSum(pairs_at_c) <= 1

    # ── Elk: atomic line variables for length 2, 3, 4 ──
    # (Length 1 = isolated elks, computed as "elks not in any line".)
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

    el_vars: Dict[int, Dict[int, object]] = {}
    for length in (2, 3, 4):
        el_vars[length] = pulp.LpVariable.dicts(
            f"el{length}",
            range(len(elk_lines_by_len[length])),
            cat='Binary',
        )

    # Each elk line var requires all underlying elk vars
    for length in (2, 3, 4):
        for i, line in enumerate(elk_lines_by_len[length]):
            for c in line:
                prob += el_vars[length][i] <= elk_v[c]

    # Each elk in at most 1 line (per cell)
    for c in all_cells:
        lines_containing_c = []
        for length in (2, 3, 4):
            for i, line in enumerate(elk_lines_by_len[length]):
                if c in line:
                    lines_containing_c.append(el_vars[length][i])
        if lines_containing_c:
            prob += pulp.lpSum(lines_containing_c) <= 1

    # NOTE: backward-compat aliases used by older code paths
    salmon = sal_v
    hawk = haw_v
    fox = fox_v
    elk_lines = elk_lines_by_len[4]
    el = el_vars[4]
    cells = all_cells_set

    # (Cell exclusivity and move budget already added above; old block removed.)

    # ── Hawk isolation via clique cuts ──
    # Standard LP-tightening: for each triangle (3 mutually-adjacent cells in the
    # hex graph), sum of hawks ≤ 1. This is strictly tighter than the 3 pairwise
    # constraints because the LP relaxation can't fractionally place 0.5 in each.
    triangles_seen: Set[Tuple[int, int, int]] = set()
    for c1 in cells:
        n1 = set(neighbors(c1)) & cells
        for c2 in n1:
            if c2 <= c1:
                continue
            n2 = set(neighbors(c2)) & cells
            common = n1 & n2
            for c3 in common:
                if c3 <= c2:
                    continue
                tri = tuple(sorted((c1, c2, c3)))
                if tri not in triangles_seen:
                    triangles_seen.add(tri)
                    prob += hawk[c1] + hawk[c2] + hawk[c3] <= 1
    # Edge case: edges with no triangle (unusual in dense regions but possible
    # at the boundary). Add the pairwise constraint only for edges not covered
    # by any triangle.
    edges_in_tris: Set[Tuple[int, int]] = set()
    for tri in triangles_seen:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = sorted((tri[i], tri[j]))
                edges_in_tris.add((a, b))
    for c in cells:
        for n in neighbors(c):
            if n in cells and n > c:
                if (c, n) not in edges_in_tris:
                    prob += hawk[c] + hawk[n] <= 1

    # (Bear pair isolation is enforced via per-cell bear_v constraints in the
    # new block above — bp[i] + bear_v[k] <= 1 for forbidden k. No pairwise
    # bp[i] + bp[j] needed.)

    # ── Bear staircase scoring (on number of pairs) ──
    bear_count = pulp.lpSum(bp.values())
    I_bear = pulp.LpVariable.dicts("I_bear", range(1, 5), cat='Binary')
    for k in range(1, 5):
        prob += bear_count >= k * I_bear[k]
    # Monotonicity: I_bear[k] <= I_bear[k-1] (LP-tightening; integer-equivalent)
    for k in range(2, 5):
        prob += I_bear[k] <= I_bear[k-1]
    bear_score = (
        BEAR_MARGINAL[1] * I_bear[1]
        + BEAR_MARGINAL[2] * I_bear[2]
        + BEAR_MARGINAL[3] * I_bear[3]
        + BEAR_MARGINAL[4] * I_bear[4]
    )

    # ── Elk: lines of length 2/3/4 + isolated leftovers ──
    # An elk in a line of length L scores ELK_LINE_SCORE[L] / L per cell, but the
    # value is per LINE not per cell. Score = sum over lines of their value, plus
    # isolated elks each scoring 2.
    line_2_count = pulp.lpSum(el_vars[2].values())
    line_3_count = pulp.lpSum(el_vars[3].values())
    line_4_count = pulp.lpSum(el_vars[4].values())
    elk_total = pulp.lpSum(elk_v[c] for c in all_cells)
    # elks in lines = 2 * line_2 + 3 * line_3 + 4 * line_4
    # isolated elks = elk_total - elks in lines
    elks_in_lines_expr = 2 * line_2_count + 3 * line_3_count + 4 * line_4_count
    isolated_elks_expr = elk_total - elks_in_lines_expr
    elk_score = (
        ELK_LINE_SCORE[2] * line_2_count
        + ELK_LINE_SCORE[3] * line_3_count
        + ELK_LINE_SCORE[4] * line_4_count
        + ELK_LINE_SCORE[1] * isolated_elks_expr
    )

    # ── Salmon non-branching constraint ──
    # Each salmon has at most 2 salmon neighbors (Cascadia rule).
    # Big-M linearization: sum_n_salmon <= 2 + (max_nbrs - 2)*(1 - sal_v[c])
    for c in all_cells:
        nbrs_in = [n for n in neighbors(c) if n in all_cells_set]
        if len(nbrs_in) >= 3:
            max_nbrs = len(nbrs_in)
            prob += pulp.lpSum(sal_v[n] for n in nbrs_in) <= 2 + (max_nbrs - 2) * (1 - sal_v[c])

    # ── Salmon extended staircase scoring (multi-chain, capped at 9) ──
    # Imports for the dp table
    from ilp_upper_bound_cpsat import best_salmon_score
    MAX_SALMON_PER_PLAYER = 9
    salmon_count = pulp.lpSum(sal_v[c] for c in all_cells)
    prob += salmon_count <= MAX_SALMON_PER_PLAYER
    I_salmon = pulp.LpVariable.dicts("I_salmon", range(1, MAX_SALMON_PER_PLAYER + 1), cat='Binary')
    for k in range(1, MAX_SALMON_PER_PLAYER + 1):
        prob += salmon_count >= k * I_salmon[k]
    for k in range(2, MAX_SALMON_PER_PLAYER + 1):
        prob += I_salmon[k] <= I_salmon[k-1]
    salmon_score = pulp.lpSum(
        (best_salmon_score(k) - best_salmon_score(k - 1)) * I_salmon[k]
        for k in range(1, MAX_SALMON_PER_PLAYER + 1)
    )

    # ── Hawk staircase scoring ──
    hawk_count = pulp.lpSum(haw_v[c] for c in all_cells)
    I_hawk = pulp.LpVariable.dicts("I_hawk", range(1, 9), cat='Binary')
    for k in range(1, 9):
        prob += hawk_count >= k * I_hawk[k]
    for k in range(2, 9):
        prob += I_hawk[k] <= I_hawk[k-1]
    hawk_score = pulp.lpSum(
        HAWK_MARGINAL[k] * I_hawk[k] for k in range(1, 9)
    )

    # ── Fox: TIGHT bound — per-fox count of distinct species in neighbors ──
    # For each cell c and species sp, fox_sees[c,sp] = 1 iff (fox is at c) AND
    # (at least one neighbor of c has species sp). fox_score = sum.
    SPECIES = ["bear", "elk", "salmon", "hawk", "fox"]
    sp_vars_by_name = {
        "bear": bear_v, "elk": elk_v, "salmon": sal_v, "hawk": haw_v, "fox": fox_v,
    }

    placement_lin: Dict[Tuple[int, str], object] = {}
    for c in all_cells:
        for sp in SPECIES:
            placement_lin[(c, sp)] = sp_vars_by_name[sp][c]
    # (legacy block, no-op kept for diff readability)
    for c in []:
        bear_terms = [bp[i] for i, (a, b) in enumerate(bear_pairs) if c == a or c == b]
        placement_lin[(c, "bear")] = pulp.lpSum(bear_terms) if bear_terms else 0
        # elk: c is in some elk line
        elk_terms = [el[i] for i, line in enumerate(elk_lines) if c in line[:4]]
        placement_lin[(c, "elk")] = pulp.lpSum(elk_terms) if elk_terms else 0
        placement_lin[(c, "salmon")] = salmon[c]
        placement_lin[(c, "hawk")] = hawk[c]
        placement_lin[(c, "fox")] = fox[c]

    # fox_sees[(c, sp)] is a new binary indicator
    fox_sees_keys = [(c, sp) for c in all_cells for sp in SPECIES]
    fox_sees = pulp.LpVariable.dicts(
        "fox_sees",
        fox_sees_keys,
        cat='Binary',
    )

    for c in all_cells:
        nbrs = [n for n in neighbors(c) if n in all_cells_set]
        for sp in SPECIES:
            prob += fox_sees[(c, sp)] <= fox_v[c]
            if nbrs:
                prob += fox_sees[(c, sp)] <= pulp.lpSum(sp_vars_by_name[sp][n] for n in nbrs)
            else:
                prob += fox_sees[(c, sp)] == 0

    fox_score = pulp.lpSum(fox_sees.values())

    # ── Objective ──
    prob += bear_score + elk_score + salmon_score + hawk_score + fox_score

    n_vars = len(prob.variables())
    n_constraints = len(prob.constraints)
    setup_time = time.time() - t_setup_start

    # If lp_relax requested, relax all integer/binary variables to continuous [0, 1].
    # This gives a (looser but fast) LP relaxation upper bound.
    # NOTE: pulp stores Binary vars internally as cat='Integer' with bounds [0,1],
    # so we relax everything by setting cat = LpContinuous directly.
    if lp_relax:
        for v in prob.variables():
            if v.cat in (pulp.LpInteger, pulp.LpBinary):
                v.cat = pulp.LpContinuous
                if v.lowBound is None:
                    v.lowBound = 0
                if v.upBound is None:
                    v.upBound = 1

    # ── Solve ──
    t_solve_start = time.time()
    # Prefer HiGHS (native arm64 from Homebrew), fall back to CBC.
    solver = None
    # HiGHS performance options (researched 2026-04-10):
    # - parallel=on + threads=8: leverage all M-series cores
    # - mip_heuristic_effort=0.8: find good feasible solutions faster (default 0.05)
    # - presolve=on: shrinks the problem first (default)
    # - mip_detect_symmetry is on by default
    solver_options = [
        "parallel=on",
        "threads=8",
        "mip_heuristic_effort=0.8",
        "presolve=on",
    ]
    if mip_gap is not None:
        solver_options.append(f"mip_rel_gap={mip_gap}")
    for solver_cls in (pulp.HiGHS_CMD, pulp.PULP_CBC_CMD):
        try:
            kwargs = {'msg': msg, 'timeLimit': time_limit_s}
            if solver_cls is pulp.HiGHS_CMD:
                kwargs['options'] = solver_options
            candidate = solver_cls(**kwargs)
            if candidate.available():
                solver = candidate
                break
        except Exception:
            continue
    if solver is None:
        raise RuntimeError("No LP solver available (tried HiGHS, CBC)")
    status = prob.solve(solver)
    solve_time = time.time() - t_solve_start

    if status != 1:  # 1 = Optimal
        return {
            'upper_bound': -1,
            'status': pulp.LpStatus[status],
            'n_vars': n_vars,
            'n_constraints': n_constraints,
            'setup_time_s': setup_time,
            'solver_time_s': solve_time,
        }

    # Extract per-component scores (now over the FULL board, existing + new)
    bp_val = sum(int(round(bp[i].varValue or 0)) for i in range(len(bear_pairs)))
    el2_val = sum(int(round(el_vars[2][i].varValue or 0)) for i in range(len(elk_lines_by_len[2])))
    el3_val = sum(int(round(el_vars[3][i].varValue or 0)) for i in range(len(elk_lines_by_len[3])))
    el4_val = sum(int(round(el_vars[4][i].varValue or 0)) for i in range(len(elk_lines_by_len[4])))
    elk_total_val = sum(int(round(elk_v[c].varValue or 0)) for c in all_cells)
    isolated_elks_val = elk_total_val - 2 * el2_val - 3 * el3_val - 4 * el4_val
    salmon_val = sum(int(round(sal_v[c].varValue or 0)) for c in all_cells)
    hawk_val = sum(int(round(haw_v[c].varValue or 0)) for c in all_cells)
    fox_val = sum(int(round(fox_v[c].varValue or 0)) for c in all_cells)
    bear_total_val = sum(int(round(bear_v[c].varValue or 0)) for c in all_cells)

    fox_score_val = sum(int(round(fox_sees[k].varValue or 0)) for k in fox_sees_keys)
    elk_score_val = (
        ELK_LINE_SCORE[2] * el2_val
        + ELK_LINE_SCORE[3] * el3_val
        + ELK_LINE_SCORE[4] * el4_val
        + ELK_LINE_SCORE[1] * isolated_elks_val
    )
    breakdown = {
        'bear': BEAR_SCORE.get(min(bp_val, 4), BEAR_SCORE[4]),
        'elk': elk_score_val,
        'salmon': SALMON_RUN_SCORE.get(min(salmon_val, 7), SALMON_RUN_SCORE[7]) if salmon_val > 0 else 0,
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
        'upper_bound': int(round(pulp.value(prob.objective) or 0)),
        'wildlife_only_bound': int(round(pulp.value(prob.objective) or 0)),
        'breakdown': breakdown,
        'n_vars': n_vars,
        'n_constraints': n_constraints,
        'setup_time_s': setup_time,
        'solver_time_s': solve_time,
    }


# ─────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────

def make_blank_board(radius: int = 3) -> BoardState:
    """Build an empty board with a hex region of given radius around origin.
    All cells are 'placed tiles' with allowed_mask = all wildlife (any animal).
    Useful for testing upper bound on max-flexibility positions.
    """
    cells: Set[int] = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(q + r) > radius:
                continue
            idx = coord_to_idx(q, r)
            if idx is not None:
                cells.add(idx)
    return BoardState(
        placed_tiles=cells,
        placed_wildlife={},
        allowed_mask={c: 0b11111 for c in cells},
        current_score=0,
    )


def main():
    parser = argparse.ArgumentParser(description='ILP upper-bound oracle for Cascadia')
    parser.add_argument('--radius', type=int, default=3,
                        help='Radius of test hex region (default 3 → 37 cells)')
    parser.add_argument('--moves', type=int, default=20,
                        help='Moves remaining to optimize over')
    parser.add_argument('--verbose', action='store_true', help='Print solver output')
    args = parser.parse_args()

    print(f"=== ILP Upper Bound — wildlife-only relaxation ===")
    print(f"Test board: hex region radius={args.radius}")
    board = make_blank_board(args.radius)
    print(f"  Cells: {len(board.placed_tiles)}")
    print(f"  Moves remaining: {args.moves}")
    print()

    print("Building model and solving...")
    result = ilp_upper_bound(board, args.moves, msg=args.verbose, time_limit_s=60)

    if result.get('upper_bound', -1) < 0:
        print(f"  Status: {result.get('status')}")
        return

    print(f"  Variables: {result['n_vars']}")
    print(f"  Constraints: {result['n_constraints']}")
    print(f"  Setup time: {result['setup_time_s']:.2f}s")
    print(f"  Solver time: {result['solver_time_s']:.2f}s")
    print()
    print(f"  Wildlife-only upper bound: {result['wildlife_only_bound']}")
    print()
    print(f"  Breakdown:")
    bd = result['breakdown']
    print(f"    Bear:    {bd['bear']:3} ({bd['counts']['bear_pairs']} pairs)")
    print(f"    Elk:     {bd['elk']:3} ({bd['counts']['elk_lines']} lines of 4)")
    print(f"    Salmon:  {bd['salmon']:3} ({bd['counts']['salmon_cells']} salmon cells)")
    print(f"    Hawk:    {bd['hawk']:3} ({bd['counts']['hawks']} isolated)")
    print(f"    Fox:     {bd['fox']:3} ({bd['counts']['foxes']} foxes)")


if __name__ == '__main__':
    main()
