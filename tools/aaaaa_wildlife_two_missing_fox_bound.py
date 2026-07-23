#!/usr/bin/env python3
"""Finite local bound for AAAAA branches with two salmon-missing foxes.

The target cases have six foxes and exactly two salmon.  If a maximum salmon
pair is missed by two foxes, reaching the challenged score requires all other
Fox-A observations, including a fox observation for each missing fox.  Missing
foxes either remain in the salmon cluster's second ring or are adjacent to one
another.  The former is covered by the existing explicit-ring model; this
module exhausts the latter at every interacting separation and one factorized
far representative.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cache
from typing import Any

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools import aaaaa_wildlife_hawk_packing_bound as hawk
from tools import aaaaa_wildlife_zero_hawk_bound as zero
from tools.aaaaa_wildlife_gap_one_salmon_bound import INTERACTION_DISTANCE, hex_distance
from tools.aaaaa_wildlife_motif_certificate import (
    DIHEDRAL_TRANSFORMS,
    DIRECTIONS,
    Shape,
    adjacent,
    boundary,
)

SCREEN_CASES = (
    ((4, 5, 2, 3, 6), 67),
    ((5, 5, 2, 2, 6), 64),
    ((3, 5, 2, 4, 6), 63),
    ((3, 6, 2, 3, 6), 63),
)


@dataclass(frozen=True)
class ColoredLayout:
    salmon: Shape
    remote_fox_pair: Shape
    factorized_far_case: bool


@dataclass(frozen=True)
class LayoutBound:
    status: str
    fox_upper: int | None
    total_upper: int | None
    branches: int
    conflicts: int
    wall_seconds: float


def _canonical_colored(salmon: Shape, foxes: Shape) -> tuple[tuple[tuple[int, int], ...], ...]:
    images = []
    for (qq, qr), (rq, rr) in DIHEDRAL_TRANSFORMS:
        salmon_image = {(qq * q + qr * r, rq * q + rr * r) for q, r in salmon}
        fox_image = {(qq * q + qr * r, rq * q + rr * r) for q, r in foxes}
        for anchor_q, anchor_r in salmon_image | fox_image:
            images.append(
                (
                    tuple(sorted((q - anchor_q, r - anchor_r) for q, r in salmon_image)),
                    tuple(sorted((q - anchor_q, r - anchor_r) for q, r in fox_image)),
                )
            )
    return min(images)


def _expanded_region(shape: Shape, radius: int) -> set[tuple[int, int]]:
    region = set(shape)
    for _ in range(radius):
        region.update(boundary(region))
    return region


@cache
def interacting_layouts() -> tuple[ColoredLayout, ...]:
    layouts: dict[tuple[tuple[tuple[int, int], ...], ...], ColoredLayout] = {}
    for salmon in zero.unbranched_shapes(2):
        forbidden = set(salmon) | boundary(salmon)
        region = _expanded_region(salmon, INTERACTION_DISTANCE + 1)
        for left in region - forbidden:
            for dq, dr in DIRECTIONS[:3]:
                right = (left[0] + dq, left[1] + dr)
                pair = frozenset({left, right})
                if pair & forbidden:
                    continue
                distance = min(hex_distance(fox, fish) for fox in pair for fish in salmon)
                if not 2 <= distance <= INTERACTION_DISTANCE:
                    continue
                key = _canonical_colored(salmon, pair)
                layouts[key] = ColoredLayout(
                    salmon=frozenset(key[0]),
                    remote_fox_pair=frozenset(key[1]),
                    factorized_far_case=False,
                )

        # At component distance eight, no Bear pair, Hawk singleton, fox edge,
        # or Elk line of length at most four spans the clusters.  One relative
        # placement therefore represents every farther translation.
        maximum_q = max(q for q, _ in salmon)
        pair = frozenset({(maximum_q + INTERACTION_DISTANCE + 1, 0),
                          (maximum_q + INTERACTION_DISTANCE + 2, 0)})
        key = _canonical_colored(salmon, pair)
        layouts[key] = ColoredLayout(
            salmon=frozenset(key[0]),
            remote_fox_pair=frozenset(key[1]),
            factorized_far_case=True,
        )
    return tuple(
        sorted(
            layouts.values(),
            key=lambda row: (
                row.factorized_far_case,
                tuple(sorted(row.salmon)),
                tuple(sorted(row.remote_fox_pair)),
            ),
        )
    )


def _selected_terms(
    model: cp_model.CpModel,
    placements: tuple[Shape, ...],
    required: int,
    name: str,
) -> list[cp_model.IntVar]:
    selected, _ = zero.selected_placement_variables(model, placements, required, name)
    return selected


def solve_layout_bound(
    counts: tuple[int, int, int, int, int],
    layout: ColoredLayout,
    elk_partition: tuple[int, ...],
    *,
    workers: int,
    time_limit: float,
) -> LayoutBound:
    bear_count, _, salmon_count, hawk_count, fox_count = counts
    if salmon_count != 2 or fox_count != 6:
        raise ValueError("two-missing-fox bound requires two salmon and six foxes")
    salmon = layout.salmon
    remote = layout.remote_fox_pair
    if len(salmon) != 2 or len(remote) != 2 or not any(
        adjacent(left, right) for left, right in itertools.combinations(remote, 2)
    ):
        raise ValueError("invalid fixed layout")

    model = cp_model.CpModel()
    local_cells = sorted(boundary(salmon) - remote)
    local_fox = {
        cell: model.new_bool_var(f"local_fox_{index}") for index, cell in enumerate(local_cells)
    }
    local_count = fox_count - len(remote)
    model.add(sum(local_fox.values()) == local_count)
    possible_foxes = set(local_cells) | set(remote)

    _, bear_pairs, bear_singles = zero.maximum_bear_structure(bear_count)
    bear_pair_shapes = tuple(
        shape
        for shape in zero.pair_placements(possible_foxes, salmon)
        if shape.isdisjoint(remote)
    )
    bear_pair_vars = _selected_terms(model, bear_pair_shapes, bear_pairs, "bear_pair")
    single_shapes = tuple(
        shape
        for shape in zero.group_placements(possible_foxes, salmon, 1)
        if shape.isdisjoint(remote)
    )
    bear_single_vars = _selected_terms(model, single_shapes, bear_singles, "bear_single")
    hawk_vars = _selected_terms(model, single_shapes, hawk_count, "hawk")

    elk_shapes_by_length = {
        length: tuple(
            shape
            for shape in zero.group_placements(possible_foxes, salmon, length)
            if shape.isdisjoint(remote)
        )
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
        cell: [variable] for cell, variable in local_fox.items()
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

    term_sets = {
        "bear": bear_terms,
        "elk": elk_terms,
        "hawk": hawk_terms,
    }
    coverages = []
    self_coverages = []
    all_fox_cells = set(local_cells) | set(remote)
    for index, fox in enumerate(sorted(all_fox_cells)):
        present: cp_model.IntVar | int = local_fox.get(fox, 1)
        for species, terms in term_sets.items():
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
            coverages.append(covered)
        sees_fox = model.new_bool_var(f"self_coverage_{index}")
        fox_neighbors: list[cp_model.IntVar | int] = []
        for other in all_fox_cells - {fox}:
            if adjacent(fox, other):
                fox_neighbors.append(local_fox.get(other, 1))
        model.add(sees_fox <= present)
        model.add(sees_fox <= sum(fox_neighbors))
        self_coverages.append(sees_fox)

    local_score = model.new_int_var(0, 4 * fox_count, "non_salmon_fox_coverage")
    model.add(local_score == sum(coverages) + sum(self_coverages))
    model.maximize(local_score)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        return LayoutBound(
            status=solver.status_name(status),
            fox_upper=None,
            total_upper=None,
            branches=solver.num_branches,
            conflicts=solver.num_conflicts,
            wall_seconds=solver.wall_time,
        )
    fox_upper = local_count + int(solver.value(local_score))
    bear_score, _, _ = zero.maximum_bear_structure(bear_count)
    elk_score = sum({1: 2, 2: 5, 3: 9, 4: 13}[size] for size in elk_partition)
    total_upper = (
        bear_score
        + zero.maximum_salmon_score(salmon_count)
        + base.HAWK_SCORES[hawk_count]
        + elk_score
        + fox_upper
    )
    return LayoutBound(
        status="OPTIMAL",
        fox_upper=fox_upper,
        total_upper=total_upper,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        wall_seconds=solver.wall_time,
    )


def remote_pair_branch_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    workers: int = 1,
    per_layout_time_limit: float = 30.0,
) -> dict[str, Any]:
    bear, elk, salmon, hawk, fox = counts
    if salmon != 2 or fox != 6:
        raise ValueError("remote pair branch requires exactly two salmon and six foxes")
    non_fox_max = (
        zero.maximum_bear_structure(bear)[0]
        + base.STANDALONE_SCORES[base.SPECIES_CODE["elk"]][elk]
        + zero.maximum_salmon_score(salmon)
        + base.HAWK_SCORES[hawk]
    )
    if target - non_fox_max != 28:
        raise ValueError("target must force every non-salmon Fox-A observation")

    best = -1
    best_case = None
    cases = 0
    wall = 0.0
    for elk_score, partitions in zero.elk_partitions_by_score(elk).items():
        if (
            zero.maximum_bear_structure(bear)[0]
            + zero.maximum_salmon_score(salmon)
            + base.HAWK_SCORES[hawk]
            + elk_score
            + 28
            < target
        ):
            continue
        for partition in partitions:
            for layout_index, layout in enumerate(interacting_layouts()):
                result = solve_layout_bound(
                    counts,
                    layout,
                    partition,
                    workers=workers,
                    time_limit=per_layout_time_limit,
                )
                cases += 1
                wall += result.wall_seconds
                if result.status != "OPTIMAL":
                    return {
                        "status": "UNKNOWN",
                        "upper_bound": None,
                        "cases": cases,
                        "failed_case": {
                            "elk_partition": list(partition),
                            "layout_index": layout_index,
                            "layout": {
                                "salmon": sorted(layout.salmon),
                                "remote_fox_pair": sorted(layout.remote_fox_pair),
                                "factorized_far_case": layout.factorized_far_case,
                            },
                            "result": result.__dict__,
                        },
                        "aggregate_solver_wall_seconds": wall,
                    }
                if int(result.total_upper) > best:
                    best = int(result.total_upper)
                    best_case = {
                        "elk_partition": list(partition),
                        "layout_index": layout_index,
                        "layout": {
                            "salmon": sorted(layout.salmon),
                            "remote_fox_pair": sorted(layout.remote_fox_pair),
                            "factorized_far_case": layout.factorized_far_case,
                        },
                        "result": result.__dict__,
                    }
    return {
        "status": "INFEASIBLE" if best < target else "RELAXATION_FEASIBLE",
        "upper_bound": best,
        "target": target,
        "cases": cases,
        "layout_count": len(interacting_layouts()),
        "best_case": best_case,
        "aggregate_solver_wall_seconds": wall,
    }


def explicit_second_ring_branch_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    workers: int = 1,
    per_shape_time_limit: float = 30.0,
) -> dict[str, Any]:
    bear, elk, salmon, hawk_count, fox = counts
    if salmon != 2 or fox != 6:
        raise ValueError("second-ring branch requires exactly two salmon and six foxes")
    non_fox_max = (
        zero.maximum_bear_structure(bear)[0]
        + base.STANDALONE_SCORES[base.SPECIES_CODE["elk"]][elk]
        + zero.maximum_salmon_score(salmon)
        + base.HAWK_SCORES[hawk_count]
    )
    if target - non_fox_max != 28:
        raise ValueError("target must force every non-salmon Fox-A observation")

    best = -1
    best_case = None
    cases = 0
    wall = 0.0
    maximum_elk = base.STANDALONE_SCORES[base.SPECIES_CODE["elk"]][elk]
    for partition in zero.elk_partitions_by_score(elk)[maximum_elk]:
        for shape_index, shape in enumerate(zero.unbranched_shapes(salmon)):
            result = hawk.solve_shape_bound(
                counts,
                shape,
                fox - 2,
                partition,
                workers,
                per_shape_time_limit,
                explicit_missing_foxes=True,
            )
            cases += 1
            wall += result.wall_seconds
            if result.status != "OPTIMAL":
                return {
                    "status": "UNKNOWN",
                    "upper_bound": None,
                    "cases": cases,
                    "failed_case": {
                        "elk_partition": list(partition),
                        "salmon_shape_index": shape_index,
                        "result": result.__dict__,
                    },
                    "aggregate_solver_wall_seconds": wall,
                }
            if int(result.total_upper) > best:
                best = int(result.total_upper)
                best_case = {
                    "elk_partition": list(partition),
                    "salmon_shape_index": shape_index,
                    "result": result.__dict__,
                }
    return {
        "status": "INFEASIBLE" if best < target else "RELAXATION_FEASIBLE",
        "upper_bound": best,
        "target": target,
        "cases": cases,
        "best_case": best_case,
        "aggregate_solver_wall_seconds": wall,
    }


def refined_maximum_salmon_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    workers: int = 1,
    per_case_time_limit: float = 30.0,
) -> dict[str, Any]:
    explicit = explicit_second_ring_branch_bound(
        counts,
        target,
        workers=workers,
        per_shape_time_limit=per_case_time_limit,
    )
    if explicit["status"] == "UNKNOWN":
        return {"status": "UNKNOWN", "upper_bound": None, "explicit_second_ring": explicit}
    remote = remote_pair_branch_bound(
        counts,
        target,
        workers=workers,
        per_layout_time_limit=per_case_time_limit,
    )
    if remote["status"] == "UNKNOWN":
        return {
            "status": "UNKNOWN",
            "upper_bound": None,
            "explicit_second_ring": explicit,
            "remote_adjacent_pair": remote,
        }
    upper = max(int(explicit["upper_bound"]), int(remote["upper_bound"]))
    return {
        "status": "INFEASIBLE" if upper < target else "RELAXATION_FEASIBLE",
        "upper_bound": upper,
        "target": target,
        "explicit_second_ring": explicit,
        "remote_adjacent_pair": remote,
    }
