#!/usr/bin/env python3
"""Threshold feasibility bound for split-singleton Salmon-A tail branches."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools import aaaaa_wildlife_zero_hawk_bound as zero
from tools.aaaaa_wildlife_gap_two_salmon_pair_bound import split_singleton_shapes
from tools.aaaaa_wildlife_motif_certificate import Shape, adjacent, boundary


@dataclass(frozen=True)
class FeasibilityResult:
    status: str
    branches: int
    conflicts: int
    wall_seconds: float


def _selected_terms(
    model: cp_model.CpModel,
    placements: tuple[Shape, ...],
    required: int,
    name: str,
) -> list[cp_model.IntVar]:
    selected, _ = zero.selected_placement_variables(model, placements, required, name)
    return selected


def solve_split_shape_feasibility(
    counts: tuple[int, int, int, int, int],
    salmon: Shape,
    local_fox_count: int,
    elk_partition: tuple[int, ...],
    required_fox_score: int,
    *,
    workers: int,
    time_limit: float,
) -> FeasibilityResult:
    bear_count, _, salmon_count, hawk_count, fox_count = counts
    if salmon_count != 2 or len(salmon) != 2 or any(
        adjacent(left, right) for left, right in itertools.combinations(salmon, 2)
    ):
        raise ValueError("split branch requires two nonadjacent singleton salmon")
    if fox_count != 6 or fox_count - local_fox_count not in (0, 1):
        raise ValueError("split branch supports zero or one salmon-missing fox")

    model = cp_model.CpModel()
    salmon_neighbors = set(boundary(salmon))
    outer_foxes = {
        (q + dq, r + dr)
        for q, r in salmon_neighbors
        for dq, dr in zero.DIRECTIONS
        if (q + dq, r + dr) not in salmon and (q + dq, r + dr) not in salmon_neighbors
    }
    missing = fox_count - local_fox_count
    possible_foxes = salmon_neighbors | (outer_foxes if missing else set())
    fox_at = {
        cell: model.new_bool_var(f"fox_{index}")
        for index, cell in enumerate(sorted(possible_foxes))
    }
    model.add(sum(fox_at[cell] for cell in salmon_neighbors) == local_fox_count)
    if missing:
        model.add(sum(fox_at[cell] for cell in outer_foxes) == missing)

    _, bear_pairs, bear_singles = zero.maximum_bear_structure(bear_count)
    bear_pair_shapes = zero.pair_placements(possible_foxes, salmon)
    bear_pair_vars = _selected_terms(model, bear_pair_shapes, bear_pairs, "bear_pair")
    single_shapes = zero.group_placements(possible_foxes, salmon, 1)
    bear_single_vars = _selected_terms(model, single_shapes, bear_singles, "bear_single")
    hawk_vars = _selected_terms(model, single_shapes, hawk_count, "hawk")
    elk_shapes_by_length = {
        length: zero.group_placements(possible_foxes, salmon, length)
        for length in set(elk_partition)
    }
    elk_vars_by_length = {
        length: _selected_terms(
            model,
            elk_shapes_by_length[length],
            sum(value == length for value in elk_partition),
            f"elk_{length}",
        )
        for length in set(elk_partition)
    }

    occupants: dict[tuple[int, int], list[cp_model.IntVar]] = {
        cell: [variable] for cell, variable in fox_at.items()
    }
    bear_terms: list[tuple[Shape, cp_model.IntVar]] = []
    for shapes, variables in (
        (bear_pair_shapes, bear_pair_vars),
        (single_shapes, bear_single_vars),
    ):
        for shape, variable in zip(shapes, variables, strict=True):
            bear_terms.append((shape, variable))
            for cell in shape:
                occupants.setdefault(cell, []).append(variable)
    hawk_terms = list(zip(single_shapes, hawk_vars, strict=True))
    for shape, variable in hawk_terms:
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

    coverage_terms = []
    self_terms = []
    species_terms = {
        "bear": bear_terms,
        "elk": elk_terms,
        "hawk": hawk_terms,
    }
    for index, (fox, present) in enumerate(fox_at.items()):
        for species, terms in species_terms.items():
            if counts[base.SPECIES_CODE[species]] == 0:
                continue
            covered = model.new_bool_var(f"{species}_coverage_{index}")
            neighbors = [
                variable
                for shape, variable in terms
                if any(adjacent(fox, target) for target in shape)
            ]
            model.add(covered <= present)
            model.add(covered <= sum(neighbors))
            coverage_terms.append(covered)
        self_coverage = model.new_bool_var(f"self_coverage_{index}")
        neighbors = [
            other_present
            for other, other_present in fox_at.items()
            if other != fox and adjacent(fox, other)
        ]
        model.add(self_coverage <= present)
        model.add(self_coverage <= sum(neighbors))
        self_terms.append(self_coverage)

    # Every local fox has its one Salmon-A species observation. All remaining
    # positive observations are represented explicitly above.
    model.add(local_fox_count + sum(coverage_terms) + sum(self_terms) >= required_fox_score)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)
    return FeasibilityResult(
        status=solver.status_name(status),
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        wall_seconds=solver.wall_time,
    )


def split_branch_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    workers: int = 1,
    per_shape_time_limit: float = 30.0,
) -> dict[str, Any]:
    bear, elk, salmon_count, hawk, fox = counts
    if salmon_count != 2 or fox != 6:
        raise ValueError("split branch requires two salmon and six foxes")
    bear_score = zero.maximum_bear_structure(bear)[0]
    hawk_score = base.HAWK_SCORES[hawk]
    salmon_score = 4
    cases = 0
    wall = 0.0
    unknown = None
    for elk_score, partitions in sorted(zero.elk_partitions_by_score(elk).items(), reverse=True):
        required_fox = target - bear_score - salmon_score - hawk_score - elk_score
        if required_fox > 5 * fox:
            continue
        maximum_misses = min(1, 5 * fox - required_fox)
        for partition in partitions:
            for missing in range(maximum_misses + 1):
                for shape_index, shape in enumerate(split_singleton_shapes()):
                    result = solve_split_shape_feasibility(
                        counts,
                        shape,
                        fox - missing,
                        partition,
                        required_fox,
                        workers=workers,
                        time_limit=per_shape_time_limit,
                    )
                    cases += 1
                    wall += result.wall_seconds
                    case = {
                        "elk_partition": list(partition),
                        "elk_score": elk_score,
                        "required_fox_score": required_fox,
                        "missing_salmon_foxes": missing,
                        "salmon_shape_index": shape_index,
                        "salmon_shape": sorted(shape),
                        "result": result.__dict__,
                    }
                    if result.status in ("OPTIMAL", "FEASIBLE"):
                        return {
                            "status": "RELAXATION_FEASIBLE",
                            "upper_bound": None,
                            "target": target,
                            "cases": cases,
                            "feasible_case": case,
                            "aggregate_solver_wall_seconds": wall,
                        }
                    if result.status != "INFEASIBLE" and unknown is None:
                        unknown = case
    if unknown is not None:
        return {
            "status": "UNKNOWN",
            "upper_bound": None,
            "target": target,
            "cases": cases,
            "failed_case": unknown,
            "aggregate_solver_wall_seconds": wall,
        }
    return {
        "status": "INFEASIBLE",
        "upper_bound": target - 1,
        "target": target,
        "cases": cases,
        "shape_count": len(split_singleton_shapes()),
        "aggregate_solver_wall_seconds": wall,
    }
