#!/usr/bin/env python3
"""Exact relaxed upper bound for AAAAA counts (3, 6, 3, 3, 5).

The challenged score 62 is one below the geometry-free ceiling. The only
possible losses are one Elk-A point, one Salmon-A point, or one Fox-A point.
Maximum-salmon branches use the explicit one-loss local packing model. The
seven-point salmon branch must be a pair plus singleton; by pigeonhole, one
component covers at least three of the five foxes. We model that component and
its local foxes exactly while awarding the other component/foxes optimistic
abstract coverage, yielding a sound superset.
"""

from __future__ import annotations

from typing import Any

from tools import aaaaa_wildlife_hawk_packing_bound as hawk
from tools import aaaaa_wildlife_zero_hawk_bound as zero
from tools.aaaaa_wildlife_motif_certificate import canonical_free_shape

COUNTS = (3, 6, 3, 3, 5)
TARGET = 62
INTERACTION_DISTANCE = 7


def hex_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    dq = right[0] - left[0]
    dr = right[1] - left[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def joint_split_salmon_shapes() -> tuple[frozenset[tuple[int, int]], ...]:
    """All pair+singleton separations that can interact, plus one far case."""
    pair = {(0, 0), (1, 0)}
    canonical = set()
    for q in range(-INTERACTION_DISTANCE, INTERACTION_DISTANCE + 2):
        for r in range(-INTERACTION_DISTANCE, INTERACTION_DISTANCE + 1):
            singleton = (q, r)
            distance = min(hex_distance(singleton, cell) for cell in pair)
            if 2 <= distance <= INTERACTION_DISTANCE:
                canonical.add(canonical_free_shape(pair | {singleton}))
    # At component distance eight, possible fox cells are at least six apart;
    # no length-four Elk line or shorter local motif can span both clusters.
    canonical.add(canonical_free_shape(pair | {(9, 0)}))
    return tuple(frozenset(row) for row in sorted(canonical))


def maximum_salmon_branch(
    *, workers: int, per_shape_time_limit: float
) -> dict[str, Any]:
    best = -1
    best_case = None
    cases = 0
    wall = 0.0
    for elk_score, partitions in zero.elk_partitions_by_score(6).items():
        required_fox = TARGET - 4 - 8 - 8 - elk_score
        if required_fox > 25:
            continue
        missing_salmon_limit = 25 - required_fox
        for partition in partitions:
            for missing_salmon in range(missing_salmon_limit + 1):
                for shape_index, shape in enumerate(zero.unbranched_shapes(3)):
                    result = hawk.solve_shape_bound(
                        COUNTS,
                        shape,
                        5 - missing_salmon,
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
                                "missing_salmon_foxes": missing_salmon,
                                "salmon_shape_index": shape_index,
                                "result": result.__dict__,
                            },
                            "aggregate_solver_wall_seconds": wall,
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
        "status": "INFEASIBLE" if best < TARGET else "RELAXATION_FEASIBLE",
        "upper_bound": best,
        "cases": cases,
        "best_case": best_case,
        "aggregate_solver_wall_seconds": wall,
    }


def split_salmon_branch(
    *, workers: int, per_shape_time_limit: float
) -> dict[str, Any]:
    best = -1
    best_case = None
    cases = 0
    wall = 0.0
    # Score seven is exactly a size-two component plus a singleton. At least
    # one component is assigned three or more of the five salmon observations.
    for anchor_size in (1, 2):
        proxy_counts = (3, 6, anchor_size, 3, 5)
        for local_foxes in range(3, 6):
            abstract_foxes = 5 - local_foxes
            for partition in zero.elk_partitions_by_score(6)[18]:
                for shape_index, shape in enumerate(zero.unbranched_shapes(anchor_size)):
                    result = hawk.solve_shape_bound(
                        proxy_counts,
                        shape,
                        local_foxes,
                        partition,
                        workers,
                        per_shape_time_limit,
                    )
                    cases += 1
                    wall += result.wall_seconds
                    if result.status != "OPTIMAL":
                        return {
                            "status": "UNKNOWN",
                            "upper_bound": None,
                            "cases": cases,
                            "failed_case": {
                                "anchor_size": anchor_size,
                                "local_foxes": local_foxes,
                                "elk_partition": list(partition),
                                "salmon_shape_index": shape_index,
                                "result": result.__dict__,
                            },
                            "aggregate_solver_wall_seconds": wall,
                        }
                    # The low-level proxy omits the abstract foxes' salmon
                    # observations. Award all of them for free in this branch.
                    fox_upper = int(result.fox_upper) + abstract_foxes
                    total_upper = 4 + 18 + 7 + 8 + fox_upper
                    if total_upper > best:
                        best = total_upper
                        best_case = {
                            "anchor_size": anchor_size,
                            "local_foxes": local_foxes,
                            "elk_partition": list(partition),
                            "salmon_shape_index": shape_index,
                            "fox_upper": fox_upper,
                            "total_upper": total_upper,
                            "result": result.__dict__,
                        }
    return {
        "status": "INFEASIBLE" if best < TARGET else "RELAXATION_FEASIBLE",
        "upper_bound": best,
        "cases": cases,
        "best_case": best_case,
        "aggregate_solver_wall_seconds": wall,
    }


def joint_split_salmon_branch(
    *, workers: int, per_shape_time_limit: float
) -> dict[str, Any]:
    best = -1
    best_case = None
    cases = 0
    wall = 0.0
    for partition in zero.elk_partitions_by_score(6)[18]:
        for shape_index, shape in enumerate(joint_split_salmon_shapes()):
            result = hawk.solve_shape_bound(
                COUNTS,
                shape,
                5,
                partition,
                workers,
                per_shape_time_limit,
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
                        "salmon_shape": sorted(shape),
                        "result": result.__dict__,
                    },
                    "aggregate_solver_wall_seconds": wall,
                }
            # The low-level model assigns the maximum connected size-three
            # salmon score of eight. The fixed pair+singleton scores seven.
            total_upper = int(result.total_upper) - 1
            if total_upper > best:
                best = total_upper
                best_case = {
                    "elk_partition": list(partition),
                    "salmon_shape_index": shape_index,
                    "salmon_shape": sorted(shape),
                    "total_upper": total_upper,
                    "result": result.__dict__,
                }
    return {
        "status": "INFEASIBLE" if best < TARGET else "RELAXATION_FEASIBLE",
        "upper_bound": best,
        "cases": cases,
        "best_case": best_case,
        "aggregate_solver_wall_seconds": wall,
        "interaction_distance": INTERACTION_DISTANCE,
        "shape_count": len(joint_split_salmon_shapes()),
    }


def relaxed_upper_bound(
    *, workers: int = 1, per_shape_time_limit: float = 30.0
) -> dict[str, Any]:
    maximum = maximum_salmon_branch(
        workers=workers, per_shape_time_limit=per_shape_time_limit
    )
    if maximum["status"] == "UNKNOWN":
        return {"status": "UNKNOWN", "upper_bound": None, "maximum_salmon": maximum}
    split = split_salmon_branch(workers=workers, per_shape_time_limit=per_shape_time_limit)
    if split["status"] == "UNKNOWN":
        return {
            "status": "UNKNOWN",
            "upper_bound": None,
            "maximum_salmon": maximum,
            "split_salmon": split,
        }
    upper = max(int(maximum["upper_bound"]), int(split["upper_bound"]))
    return {
        "status": "INFEASIBLE" if upper < TARGET else "RELAXATION_FEASIBLE",
        "upper_bound": upper,
        "target": TARGET,
        "maximum_salmon": maximum,
        "split_salmon": split,
    }


def refined_relaxed_upper_bound(
    *, workers: int = 1, per_shape_time_limit: float = 30.0
) -> dict[str, Any]:
    maximum = maximum_salmon_branch(
        workers=workers, per_shape_time_limit=per_shape_time_limit
    )
    if maximum["status"] == "UNKNOWN":
        return {"status": "UNKNOWN", "upper_bound": None, "maximum_salmon": maximum}
    split = joint_split_salmon_branch(
        workers=workers, per_shape_time_limit=per_shape_time_limit
    )
    if split["status"] == "UNKNOWN":
        return {
            "status": "UNKNOWN",
            "upper_bound": None,
            "maximum_salmon": maximum,
            "joint_split_salmon": split,
        }
    upper = max(int(maximum["upper_bound"]), int(split["upper_bound"]))
    return {
        "status": "INFEASIBLE" if upper < TARGET else "RELAXATION_FEASIBLE",
        "upper_bound": upper,
        "target": TARGET,
        "maximum_salmon": maximum,
        "joint_split_salmon": split,
    }
