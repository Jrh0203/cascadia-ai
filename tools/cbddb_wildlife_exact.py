#!/usr/bin/env python3
"""Exact pure-wildlife optimization primitives for Cascadia CBDDB.

The optimization domain is exactly twenty wildlife tokens on distinct,
connected axial hexes.  Species counts are fixed per solve and capped at six;
habitats, tile compatibility, drafting, Nature tokens, and every other game
mechanic are deliberately absent.

This module contains an executable scoring specification independent of the
Rust production scorer.  The CP-SAT formulation below uses the same scoring
contract but is validated exclusively through :func:`score_tokens` before a
witness is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

SPECIES = ("bear", "elk", "salmon", "hawk", "fox")
SPECIES_CODE = {name: index for index, name in enumerate(SPECIES)}
DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
TOKEN_COUNT = 20
COUNT_CAP = 6
GLOBAL_RADIUS = TOKEN_COUNT - 1

BEAR_C_STANDALONE = (0, 2, 5, 8, 10, 13, 18)
ELK_B_STANDALONE = (0, 2, 5, 9, 13, 15, 18)
# A valid path component of length k has at most 2k+4 distinct surrounding
# hexes. With six salmon, two disjoint triples can each claim ten non-salmon
# neighbors; allowing those neighbor sets to overlap gives the sound 26 bound.
SALMON_D_STANDALONE = (0, 0, 0, 13, 16, 19, 26)


def neighbors(coord: tuple[int, int]) -> set[tuple[int, int]]:
    q, r = coord
    return {(q + dq, r + dr) for dq, dr in DIRECTIONS}


def components(coords: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    result: list[set[tuple[int, int]]] = []
    remaining = set(coords)
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            found = neighbors(frontier.pop()) & remaining
            component.update(found)
            frontier.extend(found)
            remaining.difference_update(found)
        result.append(component)
    return result


def render_tokens(tokens: list[dict[str, int | str]]) -> str:
    occupants = {
        (int(token["q"]), int(token["r"])): str(token["wildlife"])[0].upper() for token in tokens
    }
    minimum_q = min(q for q, _ in occupants)
    maximum_q = max(q for q, _ in occupants)
    minimum_r = min(r for _, r in occupants)
    maximum_r = max(r for _, r in occupants)
    lines = []
    for r in range(minimum_r, maximum_r + 1):
        cells = " ".join(occupants.get((q, r), ".") for q in range(minimum_q, maximum_q + 1))
        lines.append(f"r={r:>2} {' ' * (r - minimum_r)}{cells}")
    return "\n".join(lines)


def _maximize_disjoint_groups(token_count: int, groups: list[tuple[int, int]]) -> int:
    dp = [0] * (1 << token_count)
    for state in range(1, len(dp)):
        first = (state & -state).bit_length() - 1
        for group, score in groups:
            if group & (1 << first) and group & state == group:
                dp[state] = max(dp[state], score + dp[state & ~group])
    return dp[-1]


def _score_bear_c(coords: set[tuple[int, int]]) -> int:
    sizes = [len(component) for component in components(coords)]
    seen = {size for size in sizes if 1 <= size <= 3}
    total = sum({1: 2, 2: 5, 3: 8}.get(size, 0) for size in sizes)
    return total + 3 * int(seen == {1, 2, 3})


def _score_elk_b(coords: set[tuple[int, int]]) -> int:
    ordered = sorted(coords)
    index = {coord: position for position, coord in enumerate(ordered)}
    adjacency = [0] * len(ordered)
    for left, coord in enumerate(ordered):
        for other in neighbors(coord):
            if other in index:
                adjacency[left] |= 1 << index[other]

    groups: list[tuple[int, int]] = [(1 << i, 2) for i in range(len(ordered))]
    for i, j in itertools.combinations(range(len(ordered)), 2):
        if not adjacency[i] & (1 << j):
            continue
        groups.append(((1 << i) | (1 << j), 5))
        for k in range(j + 1, len(ordered)):
            if not (adjacency[i] & (1 << k) and adjacency[j] & (1 << k)):
                continue
            triangle = (1 << i) | (1 << j) | (1 << k)
            groups.append((triangle, 9))
            for fourth in range(len(ordered)):
                if triangle & (1 << fourth):
                    continue
                attached = sum(
                    bool(adjacency[triangle_elk] & (1 << fourth)) for triangle_elk in (i, j, k)
                )
                if attached >= 2:
                    groups.append((triangle | (1 << fourth), 13))
    return _maximize_disjoint_groups(len(ordered), sorted(set(groups)))


def _score_salmon_d(salmon: set[tuple[int, int]], occupants: dict[tuple[int, int], str]) -> int:
    total = 0
    for component in components(salmon):
        if len(component) < 3:
            continue
        if any(len(neighbors(coord) & component) > 2 for coord in component):
            continue
        adjacent_non_salmon = {
            other
            for coord in component
            for other in neighbors(coord)
            if other in occupants and occupants[other] != "salmon"
        }
        total += len(component) + len(adjacent_non_salmon)
    return total


def _ray_between(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[tuple[int, int], ...] | None:
    dq = right[0] - left[0]
    dr = right[1] - left[1]
    for step_q, step_r in DIRECTIONS:
        distance: int | None = None
        if step_q:
            if dq % step_q:
                continue
            distance = dq // step_q
        elif dq != 0:
            continue
        if step_r:
            if dr % step_r:
                continue
            r_distance = dr // step_r
            if distance is not None and distance != r_distance:
                continue
            distance = r_distance
        elif dr != 0:
            continue
        if distance is not None and distance > 1:
            return tuple(
                (left[0] + step * step_q, left[1] + step * step_r) for step in range(1, distance)
            )
    return None


def _maximum_weight_matching(token_count: int, edges: list[tuple[int, int, int]]) -> int:
    by_left: list[list[tuple[int, int]]] = [[] for _ in range(token_count)]
    for left, right, score in edges:
        by_left[left].append((right, score))
        by_left[right].append((left, score))

    @cache
    def solve(available: int) -> int:
        if not available:
            return 0
        first = (available & -available).bit_length() - 1
        without_first = available & ~(1 << first)
        best = solve(without_first)
        for other, score in by_left[first]:
            if without_first & (1 << other):
                best = max(best, score + solve(without_first & ~(1 << other)))
        return best

    return solve((1 << token_count) - 1)


def _score_hawk_d(hawks: set[tuple[int, int]], occupants: dict[tuple[int, int], str]) -> int:
    ordered = sorted(hawks)
    edges: list[tuple[int, int, int]] = []
    for left, right in itertools.combinations(range(len(ordered)), 2):
        between = _ray_between(ordered[left], ordered[right])
        if between is None or any(coord in hawks for coord in between):
            continue
        distinct = {occupants[coord] for coord in between if coord in occupants}
        score = (0, 4, 7, 9)[min(len(distinct), 3)]
        if score:
            edges.append((left, right, score))
    return _maximum_weight_matching(len(ordered), edges)


def _score_fox_b(foxes: set[tuple[int, int]], occupants: dict[tuple[int, int], str]) -> int:
    total = 0
    for fox in foxes:
        counts = {
            species: sum(occupants.get(other) == species for other in neighbors(fox))
            for species in SPECIES[:-1]
        }
        doubled_species = sum(count >= 2 for count in counts.values())
        total += (0, 3, 5, 7)[min(doubled_species, 3)]
    return total


def score_tokens(tokens: list[dict[str, int | str]]) -> tuple[int, int, int, int, int]:
    """Return the independent CBDDB B/E/S/H/F wildlife breakdown."""
    occupants = {(int(row["q"]), int(row["r"])): str(row["wildlife"]) for row in tokens}
    if len(occupants) != len(tokens):
        raise ValueError("tokens overlap")
    if any(species not in SPECIES_CODE for species in occupants.values()):
        raise ValueError("unknown wildlife species")
    positions = {
        species: {coord for coord, wildlife in occupants.items() if wildlife == species}
        for species in SPECIES
    }
    return (
        _score_bear_c(positions["bear"]),
        _score_elk_b(positions["elk"]),
        _score_salmon_d(positions["salmon"], occupants),
        _score_hawk_d(positions["hawk"], occupants),
        _score_fox_b(positions["fox"], occupants),
    )


def count_relaxation(counts: tuple[int, int, int, int, int]) -> int:
    """Sound geometry-free CBDDB upper bound for a fixed count vector."""
    if sum(counts) != TOKEN_COUNT or any(count < 0 or count > COUNT_CAP for count in counts):
        raise ValueError(f"invalid counts: {counts}")
    bear, elk, salmon, hawk, fox = counts
    distinct_between = sum(count > 0 for count in (bear, elk, salmon, fox))
    hawk_pair_score = (0, 4, 7, 9)[min(distinct_between, 3)]
    doubled_non_fox = sum(count >= 2 for count in (bear, elk, salmon, hawk))
    fox_score = (0, 3, 5, 7)[min(doubled_non_fox, 3)]
    return (
        BEAR_C_STANDALONE[bear]
        + ELK_B_STANDALONE[elk]
        + SALMON_D_STANDALONE[salmon]
        + (hawk // 2) * hawk_pair_score
        + fox * fox_score
    )


def count_vectors() -> list[tuple[tuple[int, int, int, int, int], int]]:
    vectors = []
    for counts in itertools.product(range(COUNT_CAP + 1), repeat=len(SPECIES)):
        if sum(counts) == TOKEN_COUNT:
            vectors.append((counts, count_relaxation(counts)))
    vectors.sort(key=lambda item: (-item[1], item[0]))
    return vectors


def normalized_tokens(rows: list[dict[str, Any]]) -> list[dict[str, int | str]]:
    tokens = [
        {"q": int(row["q"]), "r": int(row["r"]), "wildlife": str(row["wildlife"])} for row in rows
    ]
    tokens.sort(key=lambda row: (int(row["r"]), int(row["q"]), str(row["wildlife"])))
    return tokens


@dataclass
class ExactVariables:
    q: list[cp_model.IntVar]
    r: list[cp_model.IntVar]
    total_score: cp_model.IntVar
    species_by_token: list[int]


def species_tokens(counts: tuple[int, int, int, int, int]) -> list[int]:
    order = (
        SPECIES_CODE["fox"],
        SPECIES_CODE["bear"],
        SPECIES_CODE["elk"],
        SPECIES_CODE["salmon"],
        SPECIES_CODE["hawk"],
    )
    return [species for species in order for _ in range(counts[species])]


def adjacency_variables(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
) -> dict[tuple[int, int], cp_model.IntVar]:
    adjacency: dict[tuple[int, int], cp_model.IntVar] = {}
    for left, right in itertools.combinations(range(TOKEN_COUNT), 2):
        dq = model.new_int_var(-2 * GLOBAL_RADIUS, 2 * GLOBAL_RADIUS, f"dq_{left}_{right}")
        dr = model.new_int_var(-2 * GLOBAL_RADIUS, 2 * GLOBAL_RADIUS, f"dr_{left}_{right}")
        model.add(dq == q[right] - q[left])
        model.add(dr == r[right] - r[left])
        adjacent_var = model.new_bool_var(f"adj_{left}_{right}")
        model.add_allowed_assignments([dq, dr], DIRECTIONS).only_enforce_if(adjacent_var)
        model.add_forbidden_assignments([dq, dr], DIRECTIONS).only_enforce_if(
            adjacent_var.negated()
        )
        adjacency[(left, right)] = adjacent_var
    return adjacency


def adjacent(
    adjacency: dict[tuple[int, int], cp_model.IntVar], left: int, right: int
) -> cp_model.IntVar:
    return adjacency[(left, right) if left < right else (right, left)]


def _reified_equal(
    model: cp_model.CpModel, expression: cp_model.LinearExprT, value: int, name: str
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(expression == value).only_enforce_if(result)
    model.add(expression != value).only_enforce_if(result.negated())
    return result


def _reified_at_least(
    model: cp_model.CpModel, expression: cp_model.LinearExprT, value: int, name: str
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(expression >= value).only_enforce_if(result)
    model.add(expression < value).only_enforce_if(result.negated())
    return result


def _reified_at_most(
    model: cp_model.CpModel, expression: cp_model.LinearExprT, value: int, name: str
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(expression <= value).only_enforce_if(result)
    model.add(expression > value).only_enforce_if(result.negated())
    return result


def _conjunction(
    model: cp_model.CpModel, members: list[cp_model.IntVar], name: str
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    for member in members:
        model.add(result <= member)
    model.add(result >= sum(members) - len(members) + 1)
    return result


def _disjunction(
    model: cp_model.CpModel, members: list[cp_model.IntVar], name: str
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(result <= sum(members))
    for member in members:
        model.add(result >= member)
    return result


def _component_constraints(
    model: cp_model.CpModel,
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    members: tuple[int, ...],
    all_species_tokens: list[int],
    component: cp_model.IntVar,
    *,
    maximum_degree: int | None = None,
) -> None:
    member_set = set(members)
    for member in members:
        if maximum_degree is not None:
            model.add(
                sum(adjacent(adjacency, member, other) for other in members if other != member)
                <= maximum_degree + len(members) * (1 - component)
            )
        for outside in all_species_tokens:
            if outside not in member_set:
                model.add(component + adjacent(adjacency, member, outside) <= 1)

    first = members[0]
    rest = members[1:]
    for mask in range(1 << len(rest)):
        left = {first}
        left.update(rest[index] for index in range(len(rest)) if mask & (1 << index))
        if len(left) == len(members):
            continue
        right = member_set - left
        model.add(
            sum(adjacent(adjacency, one, other) for one in left for other in right) >= component
        )


def _direction_cases(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    left: int,
    right: int,
    prefix: str,
) -> list[tuple[cp_model.IntVar, int, int, cp_model.LinearExprT]]:
    direction_cases: list[tuple[cp_model.IntVar, int, int, cp_model.LinearExprT]] = []
    pair_q = q[right] - q[left]
    pair_r = r[right] - r[left]
    for direction_index, (step_q, step_r) in enumerate(DIRECTIONS):
        cross = pair_q * step_r - pair_r * step_q
        projection = pair_q * step_q if step_q else pair_r * step_r
        collinear = _reified_equal(model, cross, 0, f"{prefix}_pair_collinear_{direction_index}")
        separated = _reified_at_least(
            model, projection, 2, f"{prefix}_pair_separated_{direction_index}"
        )
        direction_case = _conjunction(
            model,
            [collinear, separated],
            f"{prefix}_direction_{direction_index}",
        )
        direction_cases.append((direction_case, step_q, step_r, projection))
    return direction_cases


def _between_on_segment(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    left: int,
    target: int,
    direction_cases: list[tuple[cp_model.IntVar, int, int, cp_model.LinearExprT]],
    prefix: str,
) -> cp_model.IntVar:
    between_cases = []
    for direction_index, (direction_case, step_q, step_r, projection) in enumerate(direction_cases):
        target_q = q[target] - q[left]
        target_r = r[target] - r[left]
        target_cross = target_q * step_r - target_r * step_q
        target_projection = target_q * step_q if step_q else target_r * step_r
        target_collinear = _reified_equal(
            model,
            target_cross,
            0,
            f"{prefix}_target_{target}_collinear_{direction_index}",
        )
        after_left = _reified_at_least(
            model,
            target_projection,
            1,
            f"{prefix}_target_{target}_after_{direction_index}",
        )
        before_right = _reified_at_most(
            model,
            target_projection - projection,
            -1,
            f"{prefix}_target_{target}_before_{direction_index}",
        )
        between_cases.append(
            _conjunction(
                model,
                [direction_case, target_collinear, after_left, before_right],
                f"{prefix}_target_{target}_between_{direction_index}",
            )
        )
    return _disjunction(model, between_cases, f"{prefix}_target_{target}_between")


def build_model(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
    initial_tokens: list[dict[str, int | str]] | None = None,
    fix_initial_tokens: bool = False,
) -> tuple[cp_model.CpModel, ExactVariables]:
    upper = count_relaxation(counts)
    if maximum_score is not None:
        upper = min(upper, maximum_score)
    if minimum_score > upper:
        raise ValueError(f"score interval is empty: [{minimum_score}, {upper}]")

    model = cp_model.CpModel()
    species_by_token = species_tokens(counts)
    q = [model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"q_{i}") for i in range(TOKEN_COUNT)]
    r = [model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"r_{i}") for i in range(TOKEN_COUNT)]
    coordinate_id = [
        model.new_int_var(0, (2 * GLOBAL_RADIUS + 1) ** 2 - 1, f"coord_{i}")
        for i in range(TOKEN_COUNT)
    ]
    width = 2 * GLOBAL_RADIUS + 1
    for token in range(TOKEN_COUNT):
        diagonal = model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"diag_{token}")
        model.add(diagonal == q[token] + r[token])
        model.add(
            coordinate_id[token] == (q[token] + GLOBAL_RADIUS) * width + r[token] + GLOBAL_RADIUS
        )
    model.add_all_different(coordinate_id)
    model.add(q[0] == 0)
    model.add(r[0] == 0)

    by_species = {
        species: [index for index, value in enumerate(species_by_token) if value == species]
        for species in range(len(SPECIES))
    }
    for indices in by_species.values():
        for left, right in itertools.pairwise(indices):
            model.add(coordinate_id[left] < coordinate_id[right])

    if initial_tokens is not None:
        rows_by_species = {
            species: sorted(
                (
                    (int(row["q"]), int(row["r"]))
                    for row in initial_tokens
                    if str(row["wildlife"]) == SPECIES[species]
                ),
                key=lambda coord: coord[0] * width + coord[1],
            )
            for species in range(len(SPECIES))
        }
        if any(len(rows_by_species[species]) != counts[species] for species in range(len(SPECIES))):
            raise ValueError("initial token counts do not match model counts")
        anchor_species = species_by_token[0]
        anchor_q, anchor_r = rows_by_species[anchor_species][0]
        translated = {
            species: [(coord_q - anchor_q, coord_r - anchor_r) for coord_q, coord_r in rows]
            for species, rows in rows_by_species.items()
        }
        for token, species in enumerate(species_by_token):
            species_offset = sum(value == species for value in species_by_token[:token])
            initial_q, initial_r = translated[species][species_offset]
            if fix_initial_tokens:
                model.add(q[token] == initial_q)
                model.add(r[token] == initial_r)
            else:
                model.add_hint(q[token], initial_q)
                model.add_hint(r[token], initial_r)

    adjacency = adjacency_variables(model, q, r)
    if enforce_connectivity:
        depth = [model.new_int_var(0, TOKEN_COUNT - 1, f"depth_{i}") for i in range(TOKEN_COUNT)]
        model.add(depth[0] == 0)
        for child in range(1, TOKEN_COUNT):
            parents = []
            for parent in range(TOKEN_COUNT):
                if parent == child:
                    continue
                chosen = model.new_bool_var(f"parent_{child}_{parent}")
                model.add(chosen <= adjacent(adjacency, child, parent))
                model.add(depth[child] > depth[parent]).only_enforce_if(chosen)
                parents.append(chosen)
            model.add_exactly_one(parents)

    # Bear C: exact full components of sizes one through three, plus the set bonus.
    bears = by_species[SPECIES_CODE["bear"]]
    bear_components: list[tuple[tuple[int, ...], int, cp_model.IntVar]] = []
    for size, score in ((1, 2), (2, 5), (3, 8)):
        for members in itertools.combinations(bears, size):
            component = model.new_bool_var(f"bear_component_{'_'.join(map(str, members))}")
            _component_constraints(model, adjacency, members, bears, component)
            bear_components.append((members, score, component))
    for token in bears:
        model.add(
            sum(component for members, _, component in bear_components if token in members) <= 1
        )
    bear_bonus = model.new_bool_var("bear_set_bonus")
    for size in (1, 2, 3):
        model.add(
            bear_bonus
            <= sum(component for members, _, component in bear_components if len(members) == size)
        )
    bear_score = model.new_int_var(0, BEAR_C_STANDALONE[len(bears)], "bear_score")
    model.add(
        bear_score
        == sum(score * component for _, score, component in bear_components) + 3 * bear_bonus
    )

    # Elk B: maximum packing of singles, adjacent pairs, triangles, and strict rhombi.
    elk = by_species[SPECIES_CODE["elk"]]
    elk_groups: list[tuple[tuple[int, ...], int, cp_model.IntVar]] = []
    for token in elk:
        elk_groups.append(((token,), 2, model.new_bool_var(f"elk_single_{token}")))
    for members in itertools.combinations(elk, 2):
        group = model.new_bool_var(f"elk_pair_{'_'.join(map(str, members))}")
        model.add(group <= adjacent(adjacency, *members))
        elk_groups.append((members, 5, group))
    for members in itertools.combinations(elk, 3):
        group = model.new_bool_var(f"elk_triangle_{'_'.join(map(str, members))}")
        for left, right in itertools.combinations(members, 2):
            model.add(group <= adjacent(adjacency, left, right))
        elk_groups.append((members, 9, group))
    for members in itertools.combinations(elk, 4):
        for fourth in members:
            triangle = tuple(token for token in members if token != fourth)
            group = model.new_bool_var(f"elk_rhombus_{'_'.join(map(str, triangle))}_plus_{fourth}")
            for left, right in itertools.combinations(triangle, 2):
                model.add(group <= adjacent(adjacency, left, right))
            model.add(sum(adjacent(adjacency, fourth, member) for member in triangle) >= 2 * group)
            elk_groups.append((members, 13, group))
    for token in elk:
        model.add(sum(group for members, _, group in elk_groups if token in members) <= 1)
    elk_score = model.new_int_var(0, ELK_B_STANDALONE[len(elk)], "elk_score")
    model.add(elk_score == sum(score * group for _, score, group in elk_groups))

    # Salmon D: valid full components of length >=3 plus every distinct adjacent non-salmon.
    salmon = by_species[SPECIES_CODE["salmon"]]
    non_salmon = [token for token in range(TOKEN_COUNT) if token not in salmon]
    salmon_components: list[tuple[tuple[int, ...], cp_model.IntVar, list[cp_model.IntVar]]] = []
    for size in range(3, len(salmon) + 1):
        for members in itertools.combinations(salmon, size):
            component = model.new_bool_var(f"salmon_component_{'_'.join(map(str, members))}")
            _component_constraints(
                model,
                adjacency,
                members,
                salmon,
                component,
                maximum_degree=2,
            )
            claims = []
            for target in non_salmon:
                claim = model.new_bool_var(
                    f"salmon_component_{'_'.join(map(str, members))}_neighbor_{target}"
                )
                model.add(claim <= component)
                model.add(claim <= sum(adjacent(adjacency, member, target) for member in members))
                claims.append(claim)
            salmon_components.append((members, component, claims))
    for token in salmon:
        model.add(
            sum(component for members, component, _ in salmon_components if token in members) <= 1
        )
    salmon_score = model.new_int_var(0, SALMON_D_STANDALONE[len(salmon)], "salmon_score")
    model.add(
        salmon_score
        == sum(
            len(members) * component + sum(claims)
            for members, component, claims in salmon_components
        )
    )

    # Hawk D: line-of-sight edges weighted by distinct intervening non-hawk species,
    # followed by an exact maximum-weight matching over the selected edges.
    hawks = by_species[SPECIES_CODE["hawk"]]
    hawk_edges: list[tuple[int, int, cp_model.IntVar, cp_model.IntVar]] = []
    for left, right in itertools.combinations(hawks, 2):
        prefix = f"hawk_{left}_{right}"
        direction_data = _direction_cases(model, q, r, left, right, prefix)
        selected = model.new_bool_var(f"{prefix}_selected")
        model.add(selected <= sum(case for case, _, _, _ in direction_data))
        between_by_target = {
            target: _between_on_segment(
                model,
                q,
                r,
                left,
                target,
                direction_data,
                prefix,
            )
            for target in range(TOKEN_COUNT)
            if target not in (left, right)
        }
        for blocker in hawks:
            if blocker not in (left, right):
                model.add(selected + between_by_target[blocker] <= 1)

        claims = []
        for species in range(len(SPECIES)):
            if species == SPECIES_CODE["hawk"]:
                continue
            species_between = [between_by_target[target] for target in by_species[species]]
            claim = model.new_bool_var(f"{prefix}_distinct_{species}")
            model.add(claim <= selected)
            model.add(claim <= sum(species_between))
            claims.append(claim)
        distinct = model.new_int_var(0, len(claims), f"{prefix}_distinct_count")
        pair_score = model.new_int_var(0, 9, f"{prefix}_score")
        model.add(distinct == sum(claims))
        model.add_allowed_assignments(
            [selected, distinct, pair_score],
            [[0, 0, 0]]
            + [[1, count, (0, 4, 7, 9)[min(count, 3)]] for count in range(len(claims) + 1)],
        )
        hawk_edges.append((left, right, selected, pair_score))
    for hawk in hawks:
        model.add(
            sum(selected for left, right, selected, _ in hawk_edges if hawk in (left, right)) <= 1
        )
    hawk_upper = (len(hawks) // 2) * 9
    hawk_score = model.new_int_var(0, hawk_upper, "hawk_score")
    model.add(hawk_score == sum(score for _, _, _, score in hawk_edges))

    # Fox B: each fox scores from the number of non-fox species represented at least twice.
    foxes = by_species[SPECIES_CODE["fox"]]
    fox_scores = []
    for fox in foxes:
        doubled = []
        for species in range(len(SPECIES) - 1):
            pair_seen = []
            for left, right in itertools.combinations(by_species[species], 2):
                seen = model.new_bool_var(f"fox_{fox}_species_{species}_pair_{left}_{right}")
                model.add(seen <= adjacent(adjacency, fox, left))
                model.add(seen <= adjacent(adjacency, fox, right))
                pair_seen.append(seen)
            qualifies = model.new_bool_var(f"fox_{fox}_doubled_{species}")
            model.add(qualifies <= sum(pair_seen))
            doubled.append(qualifies)
        doubled_count = model.new_int_var(0, len(doubled), f"fox_{fox}_doubled_count")
        fox_score = model.new_int_var(0, 7, f"fox_{fox}_score")
        model.add(doubled_count == sum(doubled))
        model.add_allowed_assignments(
            [doubled_count, fox_score],
            [[count, (0, 3, 5, 7)[min(count, 3)]] for count in range(len(doubled) + 1)],
        )
        fox_scores.append(fox_score)
    fox_score = model.new_int_var(0, len(foxes) * 7, "fox_score")
    model.add(fox_score == sum(fox_scores))

    total_score = model.new_int_var(minimum_score, upper, "total_score")
    model.add(total_score == bear_score + elk_score + salmon_score + hawk_score + fox_score)
    if maximize:
        model.maximize(total_score)
    return model, ExactVariables(q, r, total_score, species_by_token)


class Progress(cp_model.CpSolverSolutionCallback):
    def __init__(self, total_score: cp_model.IntVar, started: float) -> None:
        super().__init__()
        self.total_score = total_score
        self.started = started
        self.best = -1

    def on_solution_callback(self) -> None:
        score = self.value(self.total_score)
        if score > self.best:
            self.best = score
            print(
                f"candidate={score} bound={self.best_objective_bound:.3f} "
                f"elapsed={time.monotonic() - self.started:.2f}s",
                flush=True,
            )


def solve_counts(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    time_limit: float,
    workers: int,
    seed: int,
    log_search: bool = False,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
    initial_tokens: list[dict[str, int | str]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    model, variables = build_model(
        counts,
        minimum_score,
        maximize=maximize,
        maximum_score=maximum_score,
        enforce_connectivity=enforce_connectivity,
        initial_tokens=initial_tokens,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log_search
    if maximize:
        status = solver.solve(model, Progress(variables.total_score, started))
    else:
        status = solver.solve(model)

    tokens: list[dict[str, int | str]] = []
    model_score: int | None = None
    objective: int | None = None
    breakdown: list[int] | None = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        model_score = int(solver.value(variables.total_score))
        tokens = normalized_tokens(
            [
                {
                    "q": solver.value(variables.q[token]),
                    "r": solver.value(variables.r[token]),
                    "wildlife": SPECIES[species],
                }
                for token, species in enumerate(variables.species_by_token)
            ]
        )
        breakdown = list(score_tokens(tokens))
        objective = sum(breakdown)
        if objective < model_score:
            raise RuntimeError(
                f"independent witness score {objective} is below model score {model_score}"
            )
        if not maximize and objective < minimum_score:
            raise RuntimeError(f"feasibility witness scored {objective}, below {minimum_score}")
        occupied = {(int(row["q"]), int(row["r"])) for row in tokens}
        if enforce_connectivity and len(components(occupied)) != 1:
            raise RuntimeError("model witness is disconnected")
    return {
        "counts": list(counts),
        "count_relaxation": count_relaxation(counts),
        "model_status": solver.status_name(status),
        "objective": objective,
        "model_score": model_score,
        "score_breakdown": breakdown,
        "best_bound": solver.best_objective_bound,
        "wall_seconds": solver.wall_time,
        "branches": solver.num_branches,
        "conflicts": solver.num_conflicts,
        "solver_parameters": {
            "time_limit_seconds": time_limit,
            "workers": workers,
            "random_seed": seed,
            "maximize": maximize,
            "maximum_score": maximum_score,
            "enforce_connectivity": enforce_connectivity,
        },
        "tokens": tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, help="bear,elk,salmon,hawk,fox")
    parser.add_argument("--minimum-score", type=int, default=0)
    parser.add_argument("--maximum-score", type=int)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--feasibility", action="store_true")
    parser.add_argument("--disconnected-relaxation", action="store_true")
    parser.add_argument("--log-search", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.counts.split(","))
    if len(counts) != len(SPECIES):
        parser.error("--counts requires bear,elk,salmon,hawk,fox")
    result = solve_counts(
        counts,  # type: ignore[arg-type]
        args.minimum_score,
        args.time_limit,
        args.workers,
        args.seed,
        args.log_search,
        maximize=not args.feasibility,
        maximum_score=args.maximum_score,
        enforce_connectivity=not args.disconnected_relaxation,
    )
    source = Path(__file__).resolve()
    payload = {
        "schema": "cbddb-wildlife-exact-result-v1",
        "model": "labeled-token-cp-sat-cbddb-v1",
        "model_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "assumptions": {
            "occupied_connected_hexes": TOKEN_COUNT,
            "maximum_per_species": COUNT_CAP,
            "scoring_cards": "CBDDB",
            "other_game_mechanics": "ignored",
        },
        "result": result,
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(output)
    print(encoded, end="")
    return 0 if result["model_status"] != "UNKNOWN" else 2


if __name__ == "__main__":
    sys.exit(main())
