#!/usr/bin/env python3
"""Relaxed exact upper bounds for AAAAA gap-three cases with two salmon."""

from __future__ import annotations

from typing import Any

from tools import aaaaa_wildlife_exact as base
from tools import aaaaa_wildlife_hawk_packing_bound as hawk
from tools import aaaaa_wildlife_zero_hawk_bound as zero
from tools.aaaaa_wildlife_gap_one_salmon_bound import INTERACTION_DISTANCE, hex_distance
from tools.aaaaa_wildlife_motif_certificate import canonical_free_shape

SCREEN_CASES = (
    ((4, 5, 2, 3, 6), 67),
    ((5, 5, 2, 2, 6), 64),
    ((3, 5, 2, 4, 6), 63),
    ((3, 6, 2, 3, 6), 63),
)


def split_singleton_shapes() -> tuple[frozenset[tuple[int, int]], ...]:
    canonical = set()
    for q in range(-INTERACTION_DISTANCE, INTERACTION_DISTANCE + 1):
        for r in range(-INTERACTION_DISTANCE, INTERACTION_DISTANCE + 1):
            distance = hex_distance((0, 0), (q, r))
            if 2 <= distance <= INTERACTION_DISTANCE:
                canonical.add(canonical_free_shape({(0, 0), (q, r)}))
    canonical.add(canonical_free_shape({(0, 0), (INTERACTION_DISTANCE + 1, 0)}))
    return tuple(frozenset(row) for row in sorted(canonical))


def branch_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    salmon_score: int,
    shapes: tuple[frozenset[tuple[int, int]], ...],
    *,
    workers: int,
    per_shape_time_limit: float,
) -> dict[str, Any]:
    bear, elk, _, hawk_count, fox = counts
    bear_score, _, _ = zero.maximum_bear_structure(bear)
    hawk_score = base.HAWK_SCORES[hawk_count]
    fox_max = fox * 5
    best = -1
    best_case = None
    cases = 0
    wall = 0.0
    for elk_score, partitions in zero.elk_partitions_by_score(elk).items():
        required_fox = target - bear_score - salmon_score - hawk_score - elk_score
        if required_fox > fox_max:
            continue
        missing_limit = min(fox, fox_max - max(0, required_fox))
        for partition in partitions:
            for missing_salmon in range(missing_limit + 1):
                for shape_index, shape in enumerate(shapes):
                    # With two misses, the missing foxes could form a remote
                    # pair. Keep the older abstract relaxation for soundness;
                    # zero or one miss is exhaustively explicit in the outer ring.
                    explicit = missing_salmon <= 1
                    result = hawk.solve_shape_bound(
                        counts,
                        shape,
                        fox - missing_salmon,
                        partition,
                        workers,
                        per_shape_time_limit,
                        explicit_missing_foxes=explicit,
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
                                "missing_salmon_foxes": missing_salmon,
                                "salmon_shape_index": shape_index,
                                "result": result.__dict__,
                            },
                            "aggregate_solver_wall_seconds": wall,
                        }
                    # The low-level two-salmon score is five. Two singleton
                    # components score four.
                    total_upper = int(result.total_upper) - (5 - salmon_score)
                    if total_upper > best:
                        best = total_upper
                        best_case = {
                            "elk_partition": list(partition),
                            "missing_salmon_foxes": missing_salmon,
                            "salmon_shape_index": shape_index,
                            "salmon_shape": sorted(shape),
                            "explicit_missing_foxes": explicit,
                            "total_upper": total_upper,
                            "result": result.__dict__,
                        }
    return {
        "status": "INFEASIBLE" if best < target else "RELAXATION_FEASIBLE",
        "upper_bound": best,
        "cases": cases,
        "best_case": best_case,
        "aggregate_solver_wall_seconds": wall,
        "salmon_score": salmon_score,
        "shape_count": len(shapes),
    }


def relaxed_upper_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    workers: int = 1,
    per_shape_time_limit: float = 30.0,
) -> dict[str, Any]:
    maximum = branch_bound(
        counts,
        target,
        5,
        (frozenset({(0, 0), (1, 0)}),),
        workers=workers,
        per_shape_time_limit=per_shape_time_limit,
    )
    if maximum["status"] == "UNKNOWN":
        return {"status": "UNKNOWN", "upper_bound": None, "maximum_salmon": maximum}
    split = branch_bound(
        counts,
        target,
        4,
        split_singleton_shapes(),
        workers=workers,
        per_shape_time_limit=per_shape_time_limit,
    )
    if split["status"] == "UNKNOWN":
        return {
            "status": "UNKNOWN",
            "upper_bound": None,
            "maximum_salmon": maximum,
            "split_salmon": split,
        }
    upper = max(int(maximum["upper_bound"]), int(split["upper_bound"]))
    return {
        "status": "INFEASIBLE" if upper < target else "RELAXATION_FEASIBLE",
        "upper_bound": upper,
        "target": target,
        "maximum_salmon": maximum,
        "split_salmon": split,
    }
