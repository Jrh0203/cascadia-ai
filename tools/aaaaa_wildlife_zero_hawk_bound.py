#!/usr/bin/env python3
"""Relaxed local set-packing upper bound for high-fox AAAAA boards.

This exact auxiliary model handles the unresolved zero-hawk tail in which the
requested threshold forces Bear A and Salmon A to their standalone maxima.
It fixes each possible maximum salmon component shape, models foxes adjacent
to that component, and packs the required bear components and elk scoring
lines without cell overlap. Foxes that are allowed to miss salmon are awarded
optimistic abstract coverage, so the model is a superset of real boards.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache
from typing import Any

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools.aaaaa_wildlife_motif_certificate import (
    DIRECTIONS,
    LINE_DIRECTIONS,
    Shape,
    adjacent,
    boundary,
    free_polyhexes,
    integer_partitions,
)


@cache
def unbranched_shapes(count: int) -> tuple[Shape, ...]:
    result = []
    for row in free_polyhexes(count):
        shape = frozenset(row)
        if all(sum(adjacent(cell, other) for other in shape) <= 2 for cell in shape):
            result.append(shape)
    return tuple(sorted(result, key=lambda shape: tuple(sorted(shape))))


@cache
def elk_partitions_by_score(count: int) -> dict[int, tuple[tuple[int, ...], ...]]:
    grouped: dict[int, list[tuple[int, ...]]] = {}
    scores = {1: 2, 2: 5, 3: 9, 4: 13}
    for partition in integer_partitions(count, 4):
        score = sum(scores[length] for length in partition)
        grouped.setdefault(score, []).append(partition)
    return {score: tuple(rows) for score, rows in grouped.items()}


def maximum_bear_structure(count: int) -> tuple[int, int, int]:
    """Return maximum score, isolated-pair count, and leftover singles."""
    pairs = count // 2
    singles = count % 2
    return base.BEAR_SCORES[min(pairs, 3)], pairs, singles


def maximum_salmon_score(count: int) -> int:
    return base.SALMON_SCORES[count]


def second_bear_score(count: int) -> int:
    maximum, pairs, _ = maximum_bear_structure(count)
    if pairs == 0:
        return maximum
    return base.BEAR_SCORES[min(pairs - 1, 3)]


def second_salmon_score(count: int) -> int:
    candidates = []
    for partition in integer_partitions(count, count):
        if len(partition) == 1:
            continue
        candidates.append(sum(base.SALMON_SCORES[length] for length in partition))
    candidates.append(0)
    return max(candidates)


def group_placements(
    possible_foxes: set[tuple[int, int]],
    salmon: Shape,
    length: int,
) -> tuple[Shape, ...]:
    touching_cells = {
        (q + dq, r + dr)
        for q, r in possible_foxes
        for dq, dr in DIRECTIONS
        if (q + dq, r + dr) not in salmon
    }
    if length == 1:
        return tuple(frozenset({cell}) for cell in sorted(touching_cells))
    result = set()
    for touching_cell in touching_cells:
        for dq, dr in LINE_DIRECTIONS:
            for offset in range(length):
                line = frozenset(
                    (
                        touching_cell[0] + (step - offset) * dq,
                        touching_cell[1] + (step - offset) * dr,
                    )
                    for step in range(length)
                )
                if line.isdisjoint(salmon):
                    result.add(line)
    return tuple(sorted(result, key=lambda shape: tuple(sorted(shape))))


def pair_placements(possible_foxes: set[tuple[int, int]], salmon: Shape) -> tuple[Shape, ...]:
    touching_cells = {
        (q + dq, r + dr)
        for q, r in possible_foxes
        for dq, dr in DIRECTIONS
        if (q + dq, r + dr) not in salmon
    }
    result = set()
    for left in touching_cells:
        for dq, dr in DIRECTIONS:
            right = (left[0] + dq, left[1] + dr)
            pair = frozenset({left, right})
            if pair.isdisjoint(salmon):
                result.add(pair)
    return tuple(sorted(result, key=lambda shape: tuple(sorted(shape))))


def selected_placement_variables(
    model: cp_model.CpModel,
    placements: tuple[Shape, ...],
    required_count: int,
    name: str,
) -> tuple[list[cp_model.IntVar], cp_model.IntVar]:
    selected = [model.new_bool_var(f"{name}_{index}") for index in range(len(placements))]
    abstract = model.new_int_var(0, required_count, f"{name}_abstract")
    model.add(sum(selected) + abstract == required_count)
    return selected, abstract


@dataclass(frozen=True)
class ShapeBound:
    status: str
    local_coverage: int | None
    fox_upper: int | None
    total_upper: int | None
    branches: int
    conflicts: int
    wall_seconds: float


def solve_shape_bound(
    counts: tuple[int, int, int, int, int],
    salmon: Shape,
    local_fox_count: int,
    elk_partition: tuple[int, ...],
    workers: int,
    time_limit: float,
) -> ShapeBound:
    bear_count, _, salmon_count, hawk_count, fox_count = counts
    if hawk_count != 0 or len(salmon) != salmon_count:
        raise ValueError("zero-hawk shape bound received incompatible counts")
    if not 0 <= local_fox_count <= fox_count:
        raise ValueError("invalid local fox count")

    model = cp_model.CpModel()
    possible_foxes = set(boundary(salmon))
    fox_at = {
        cell: model.new_bool_var(f"fox_{index}")
        for index, cell in enumerate(sorted(possible_foxes))
    }
    model.add(sum(fox_at.values()) == local_fox_count)

    _, bear_pairs, bear_singles = maximum_bear_structure(bear_count)
    bear_pair_shapes = pair_placements(possible_foxes, salmon)
    bear_pair_vars, _ = selected_placement_variables(
        model, bear_pair_shapes, bear_pairs, "bear_pair"
    )
    bear_single_shapes = group_placements(possible_foxes, salmon, 1)
    bear_single_vars, _ = selected_placement_variables(
        model, bear_single_shapes, bear_singles, "bear_single"
    )

    elk_shapes_by_length = {
        length: group_placements(possible_foxes, salmon, length)
        for length in set(elk_partition)
    }
    elk_vars_by_length: dict[int, list[cp_model.IntVar]] = {}
    for length, required in itertools.groupby(sorted(elk_partition)):
        required_count = len(tuple(required))
        selected, _ = selected_placement_variables(
            model,
            elk_shapes_by_length[length],
            required_count,
            f"elk_{length}",
        )
        elk_vars_by_length[length] = selected

    occupants: dict[tuple[int, int], list[cp_model.IntVar]] = {
        cell: [variable] for cell, variable in fox_at.items()
    }
    bear_terms: list[tuple[Shape, cp_model.IntVar]] = []
    for shapes, variables in (
        (bear_pair_shapes, bear_pair_vars),
        (bear_single_shapes, bear_single_vars),
    ):
        for shape, variable in zip(shapes, variables, strict=True):
            bear_terms.append((shape, variable))
            for cell in shape:
                occupants.setdefault(cell, []).append(variable)
    elk_terms: list[tuple[Shape, cp_model.IntVar]] = []
    for length, variables in elk_vars_by_length.items():
        for shape, variable in zip(elk_shapes_by_length[length], variables, strict=True):
            elk_terms.append((shape, variable))
            for cell in shape:
                occupants.setdefault(cell, []).append(variable)
    for terms in occupants.values():
        model.add(sum(terms) <= 1)

    bear_coverage = []
    elk_coverage = []
    self_coverage = []
    for index, (cell, present) in enumerate(fox_at.items()):
        bear = model.new_bool_var(f"bear_coverage_{index}")
        bear_neighbors = [
            variable
            for shape, variable in bear_terms
            if any(adjacent(cell, target) for target in shape)
        ]
        model.add(bear <= present)
        model.add(bear <= sum(bear_neighbors))
        bear_coverage.append(bear)

        elk = model.new_bool_var(f"elk_coverage_{index}")
        elk_neighbors = [
            variable
            for shape, variable in elk_terms
            if any(adjacent(cell, target) for target in shape)
        ]
        model.add(elk <= present)
        model.add(elk <= sum(elk_neighbors))
        elk_coverage.append(elk)

        sees_fox = model.new_bool_var(f"self_coverage_{index}")
        model.add(sees_fox <= present)
        model.add(
            sees_fox
            <= sum(variable for other, variable in fox_at.items() if adjacent(cell, other))
        )
        self_coverage.append(sees_fox)

    abstract_foxes = fox_count - local_fox_count
    if abstract_foxes:
        # Optimistically give every fox its self observation and each abstract
        # fox both bear and elk observations for free.
        local_expression = sum(bear_coverage) + sum(elk_coverage)
        fixed_fox_score = local_fox_count + fox_count + 2 * abstract_foxes
    else:
        local_expression = sum(bear_coverage) + sum(elk_coverage) + sum(self_coverage)
        fixed_fox_score = local_fox_count
    local_upper = 3 * local_fox_count if not abstract_foxes else 2 * local_fox_count
    local_score = model.new_int_var(0, local_upper, "local_coverage")
    model.add(local_score == local_expression)
    model.maximize(local_score)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        return ShapeBound(
            status=solver.status_name(status),
            local_coverage=None,
            fox_upper=None,
            total_upper=None,
            branches=solver.num_branches,
            conflicts=solver.num_conflicts,
            wall_seconds=solver.wall_time,
        )
    local_value = int(solver.value(local_score))
    fox_upper = fixed_fox_score + local_value
    bear_score, _, _ = maximum_bear_structure(bear_count)
    elk_score = sum({1: 2, 2: 5, 3: 9, 4: 13}[length] for length in elk_partition)
    total_upper = bear_score + maximum_salmon_score(salmon_count) + elk_score + fox_upper
    return ShapeBound(
        status="OPTIMAL",
        local_coverage=local_value,
        fox_upper=fox_upper,
        total_upper=total_upper,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        wall_seconds=solver.wall_time,
    )


def relaxed_upper_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    workers: int = 1,
    per_shape_time_limit: float = 30.0,
) -> dict[str, Any]:
    bear, elk, salmon, hawk, fox = counts
    if hawk != 0:
        raise ValueError("this bound currently covers zero-hawk vectors only")
    count_upper = base.count_relaxation(counts)
    if target > count_upper:
        return {"status": "INFEASIBLE", "upper_bound": count_upper, "cases": 0}
    maximum_bear, _, _ = maximum_bear_structure(bear)
    maximum_salmon = maximum_salmon_score(salmon)
    maximum_elk = base.STANDALONE_SCORES[base.SPECIES_CODE["elk"]][elk]
    maximum_fox = fox * 4
    if second_bear_score(bear) + maximum_salmon + maximum_elk + maximum_fox >= target:
        raise ValueError("target does not force the maximum bear structure")
    if maximum_bear + second_salmon_score(salmon) + maximum_elk + maximum_fox >= target:
        raise ValueError("target does not force one maximum salmon component")

    best = -1
    best_case: dict[str, Any] | None = None
    cases = 0
    total_wall = 0.0
    for elk_score, partitions in elk_partitions_by_score(elk).items():
        required_fox = target - maximum_bear - maximum_salmon - elk_score
        if required_fox > maximum_fox:
            continue
        maximum_salmon_misses = min(fox, maximum_fox - max(0, required_fox))
        for partition in partitions:
            for missing_salmon in range(maximum_salmon_misses + 1):
                local_foxes = fox - missing_salmon
                for shape_index, shape in enumerate(unbranched_shapes(salmon)):
                    result = solve_shape_bound(
                        counts,
                        shape,
                        local_foxes,
                        partition,
                        workers,
                        per_shape_time_limit,
                    )
                    cases += 1
                    total_wall += result.wall_seconds
                    if result.status != "OPTIMAL":
                        return {
                            "status": "UNKNOWN",
                            "upper_bound": None,
                            "cases": cases,
                            "failed_case": {
                                "elk_partition": list(partition),
                                "missing_salmon_foxes": missing_salmon,
                                "salmon_shape_index": shape_index,
                                "result": result.__dict__,
                            },
                            "aggregate_solver_wall_seconds": total_wall,
                        }
                    if int(result.total_upper) > best:
                        best = int(result.total_upper)
                        best_case = {
                            "elk_partition": list(partition),
                            "missing_salmon_foxes": missing_salmon,
                            "salmon_shape_index": shape_index,
                            "result": result.__dict__,
                        }
    return {
        "status": "INFEASIBLE" if best < target else "RELAXATION_FEASIBLE",
        "upper_bound": best,
        "target": target,
        "cases": cases,
        "best_case": best_case,
        "aggregate_solver_wall_seconds": total_wall,
    }
