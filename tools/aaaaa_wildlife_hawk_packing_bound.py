#!/usr/bin/env python3
"""Relaxed local set-packing upper bound for forced-maximum Hawk-A cases.

This extends the zero-hawk local packing model with optimistic local hawk
singletons. It is valid only when the requested threshold forces Bear A,
Salmon A, and Hawk A to their standalone maxima. Hawk isolation constraints
are deliberately dropped, making the model a superset of real boards.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools import aaaaa_wildlife_zero_hawk_bound as zero
from tools.aaaaa_wildlife_motif_certificate import Shape, adjacent, boundary

SCREEN_CASES = (
    ((6, 4, 1, 3, 6), 68),
    ((6, 5, 1, 2, 6), 68),
    ((4, 4, 1, 5, 6), 66),
    ((4, 6, 1, 3, 6), 66),
    ((4, 6, 4, 2, 4), 65),
    ((3, 5, 4, 3, 5), 63),
)
CERTIFICATE_CASES = SCREEN_CASES[-2:]


def second_hawk_score(count: int) -> int:
    if count <= 1:
        return 0
    return base.HAWK_SCORES[count - 2]


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
    *,
    explicit_missing_foxes: bool = False,
) -> ShapeBound:
    bear_count, _, salmon_count, hawk_count, fox_count = counts
    if hawk_count <= 0 or len(salmon) != salmon_count:
        raise ValueError("hawk shape bound received incompatible counts")
    model = cp_model.CpModel()
    salmon_neighbors = set(boundary(salmon))
    if explicit_missing_foxes:
        outer_foxes = {
            (q + dq, r + dr)
            for q, r in salmon_neighbors
            for dq, dr in zero.DIRECTIONS
            if (q + dq, r + dr) not in salmon and (q + dq, r + dr) not in salmon_neighbors
        }
        possible_foxes = salmon_neighbors | outer_foxes
    else:
        outer_foxes = set()
        possible_foxes = salmon_neighbors
    fox_at = {
        cell: model.new_bool_var(f"fox_{index}")
        for index, cell in enumerate(sorted(possible_foxes))
    }
    model.add(sum(fox_at[cell] for cell in salmon_neighbors) == local_fox_count)
    if explicit_missing_foxes:
        model.add(sum(fox_at[cell] for cell in outer_foxes) == fox_count - local_fox_count)
    else:
        model.add(sum(fox_at.values()) == local_fox_count)

    _, bear_pairs, bear_singles = zero.maximum_bear_structure(bear_count)
    bear_pair_shapes = zero.pair_placements(possible_foxes, salmon)
    bear_pair_vars, _ = zero.selected_placement_variables(
        model, bear_pair_shapes, bear_pairs, "bear_pair"
    )
    single_shapes = zero.group_placements(possible_foxes, salmon, 1)
    bear_single_vars, _ = zero.selected_placement_variables(
        model, single_shapes, bear_singles, "bear_single"
    )
    hawk_vars, _ = zero.selected_placement_variables(
        model, single_shapes, hawk_count, "hawk"
    )

    elk_shapes_by_length = {
        length: zero.group_placements(possible_foxes, salmon, length)
        for length in set(elk_partition)
    }
    elk_vars_by_length: dict[int, list[cp_model.IntVar]] = {}
    for length, required in itertools.groupby(sorted(elk_partition)):
        selected, _ = zero.selected_placement_variables(
            model,
            elk_shapes_by_length[length],
            len(tuple(required)),
            f"elk_{length}",
        )
        elk_vars_by_length[length] = selected

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

    coverages: dict[str, list[cp_model.IntVar]] = {"bear": [], "elk": [], "hawk": []}
    term_sets = {"bear": bear_terms, "elk": elk_terms, "hawk": hawk_terms}
    self_coverage = []
    for index, (cell, present) in enumerate(fox_at.items()):
        for species, terms in term_sets.items():
            covered = model.new_bool_var(f"{species}_coverage_{index}")
            neighbors = [
                variable
                for shape, variable in terms
                if any(adjacent(cell, target) for target in shape)
            ]
            model.add(covered <= present)
            model.add(covered <= sum(neighbors))
            coverages[species].append(covered)
        sees_fox = model.new_bool_var(f"self_coverage_{index}")
        model.add(sees_fox <= present)
        model.add(
            sees_fox
            <= sum(variable for other, variable in fox_at.items() if adjacent(cell, other))
        )
        self_coverage.append(sees_fox)

    abstract_foxes = 0 if explicit_missing_foxes else fox_count - local_fox_count
    local_species_coverage = sum(
        variable for values in coverages.values() for variable in values
    )
    if explicit_missing_foxes:
        local_expression = local_species_coverage + sum(self_coverage)
        fixed_fox_score = local_fox_count
        local_upper = 4 * fox_count
    elif abstract_foxes:
        local_expression = local_species_coverage
        fixed_fox_score = local_fox_count + fox_count + 3 * abstract_foxes
        local_upper = 3 * local_fox_count
    else:
        local_expression = local_species_coverage + sum(self_coverage)
        fixed_fox_score = local_fox_count
        local_upper = 4 * local_fox_count
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
    bear_score, _, _ = zero.maximum_bear_structure(bear_count)
    elk_score = sum({1: 2, 2: 5, 3: 9, 4: 13}[length] for length in elk_partition)
    total_upper = (
        bear_score
        + zero.maximum_salmon_score(salmon_count)
        + base.HAWK_SCORES[hawk_count]
        + elk_score
        + fox_upper
    )
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
    explicit_missing_foxes: bool = False,
) -> dict[str, Any]:
    bear, elk, salmon, hawk, fox = counts
    if hawk <= 0:
        raise ValueError("hawk packing bound requires at least one hawk")
    bear_max, _, _ = zero.maximum_bear_structure(bear)
    salmon_max = zero.maximum_salmon_score(salmon)
    hawk_max = base.HAWK_SCORES[hawk]
    elk_max = base.STANDALONE_SCORES[base.SPECIES_CODE["elk"]][elk]
    fox_max = fox * 5
    if zero.second_bear_score(bear) + salmon_max + hawk_max + elk_max + fox_max >= target:
        raise ValueError("target does not force maximum bear score")
    # A singleton salmon is necessarily a valid unbranched component and
    # always scores two; there is no attainable second-best state.
    salmon_second = -10**6 if salmon == 1 else zero.second_salmon_score(salmon)
    if bear_max + salmon_second + hawk_max + elk_max + fox_max >= target:
        raise ValueError("target does not force one maximum salmon component")
    if bear_max + salmon_max + second_hawk_score(hawk) + elk_max + fox_max >= target:
        raise ValueError("target does not force maximum hawk score")

    best = -1
    best_case: dict[str, Any] | None = None
    cases = 0
    total_wall = 0.0
    for elk_score, partitions in zero.elk_partitions_by_score(elk).items():
        required_fox = target - bear_max - salmon_max - hawk_max - elk_score
        if required_fox > fox_max:
            continue
        maximum_salmon_misses = min(fox, fox_max - max(0, required_fox))
        for partition in partitions:
            for missing_salmon in range(maximum_salmon_misses + 1):
                local_foxes = fox - missing_salmon
                for shape_index, shape in enumerate(zero.unbranched_shapes(salmon)):
                    result = solve_shape_bound(
                        counts,
                        shape,
                        local_foxes,
                        partition,
                        workers,
                        per_shape_time_limit,
                        explicit_missing_foxes=explicit_missing_foxes,
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
        "explicit_missing_foxes": explicit_missing_foxes,
    }


def certificates(
    catalog_path: Path,
    *,
    workers: int,
    per_shape_time_limit: float,
) -> dict[str, Any]:
    started = time.monotonic()
    catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    results = []
    for counts, target in CERTIFICATE_CASES:
        incumbent = zero.validate_incumbent(catalog_payload, counts)
        bound = relaxed_upper_bound(
            counts,
            target,
            workers=workers,
            per_shape_time_limit=per_shape_time_limit,
            explicit_missing_foxes=True,
        )
        if bound["status"] != "INFEASIBLE" or bound["upper_bound"] != incumbent["score"]:
            raise RuntimeError(
                f"{counts}: bound {bound['status']} / {bound['upper_bound']} "
                f"does not certify incumbent {incumbent['score']}"
            )
        results.append(
            {
                "counts": list(counts),
                "excluded_score": target,
                "certified_upper_bound": int(bound["upper_bound"]),
                "proof_complete": True,
                "proof_method": "hawk_one_loss_relaxed_local_packing_infeasible",
                "bound": bound,
                "incumbent": incumbent,
            }
        )
    source = Path(__file__).resolve()
    zero_source = source.with_name("aaaaa_wildlife_zero_hawk_bound.py")
    exact_source = source.with_name("aaaaa_wildlife_exact.py")
    return {
        "schema": "aaaaa-hawk-one-loss-local-packing-certificates-v1",
        "proof_complete": True,
        "configuration": {
            "workers": workers,
            "per_shape_time_limit_seconds": per_shape_time_limit,
            "explicit_missing_foxes": True,
        },
        "relaxation": {
            "whole_board_connectivity_required": False,
            "target_forces_maximum_bear_score": True,
            "target_forces_one_maximum_salmon_component": True,
            "target_forces_maximum_hawk_score": True,
            "hawk_isolation_dropped": True,
            "bear_isolation_dropped": True,
            "noncovering_scoring_groups_may_be_abstract": True,
            "salmon_missing_foxes_placed_explicitly": True,
            "forced_local_cells_must_not_overlap": True,
        },
        "ortools_version": ORTOOLS_VERSION,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "zero_hawk_support_source_sha256": hashlib.sha256(zero_source.read_bytes()).hexdigest(),
        "exact_scorer_source_sha256": hashlib.sha256(exact_source.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--per-shape-time-limit", type=float, default=30.0)
    args = parser.parse_args()
    payload = certificates(
        args.catalog,
        workers=args.workers,
        per_shape_time_limit=args.per_shape_time_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "elapsed_seconds": payload["elapsed_seconds"],
                "results": [
                    {
                        "counts": row["counts"],
                        "optimum": row["certified_upper_bound"],
                        "cases": row["bound"]["cases"],
                    }
                    for row in payload["results"]
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
