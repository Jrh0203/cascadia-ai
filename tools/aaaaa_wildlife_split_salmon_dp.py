#!/usr/bin/env python3
"""Exact bitset packing for split-Salmon AAAAA tail branches.

This module is logically equivalent to the finite relaxation in
``aaaaa_wildlife_split_salmon_feasibility`` but removes its placement-variable
symmetry.  It enumerates fox cell sets, constructs only inclusion-minimal
per-species covers of the required fox observations, and tests those covers
for disjoint cell packing.

Unselected Bear/Elk/Hawk groups remain abstract, exactly as in the original
relaxation.  Bear/Hawk isolation and whole-board connectivity remain dropped.
Consequently, an exhaustive ``INFEASIBLE`` result is a sound upper-bound
certificate for legal boards.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from tools import aaaaa_wildlife_exact as base
from tools import aaaaa_wildlife_zero_hawk_bound as zero
from tools.aaaaa_wildlife_gap_two_salmon_pair_bound import split_singleton_shapes
from tools.aaaaa_wildlife_motif_certificate import Shape, adjacent, boundary

Coord = tuple[int, int]


@dataclass(frozen=True)
class PackingStats:
    fox_layouts: int = 0
    deficit_branches: int = 0
    cover_searches: int = 0
    cover_configurations: int = 0
    packing_nodes: int = 0

    def add(self, **values: int) -> PackingStats:
        row = asdict(self)
        for key, value in values.items():
            row[key] += value
        return PackingStats(**row)


@dataclass(frozen=True)
class PackingResult:
    status: str
    wall_seconds: float
    stats: PackingStats
    witness: dict[str, Any] | None


def _outer_foxes(salmon: Shape) -> tuple[Coord, ...]:
    local = set(boundary(salmon))
    return tuple(
        sorted(
            {
                (q + dq, r + dr)
                for q, r in local
                for dq, dr in zero.DIRECTIONS
                if (q + dq, r + dr) not in salmon
                and (q + dq, r + dr) not in local
            }
        )
    )


@cache
def fox_layouts(salmon: Shape, missing_salmon_foxes: int) -> tuple[tuple[Coord, ...], ...]:
    """Enumerate exact six-fox layouts for a fixed split-Salmon shape."""

    if len(salmon) != 2 or any(
        adjacent(left, right) for left, right in itertools.combinations(salmon, 2)
    ):
        raise ValueError("split branch requires two nonadjacent singleton salmon")
    if missing_salmon_foxes not in (0, 1):
        raise ValueError("only zero or one salmon-missing fox is supported")
    local = tuple(sorted(boundary(salmon)))
    outer = _outer_foxes(salmon)
    local_count = 6 - missing_salmon_foxes
    rows = []
    for chosen_local in itertools.combinations(local, local_count):
        outer_choices: Iterable[tuple[Coord, ...]]
        outer_choices = (
            ((cell,) for cell in outer) if missing_salmon_foxes else ((),)
        )
        for chosen_outer in outer_choices:
            rows.append(tuple(sorted((*chosen_local, *chosen_outer))))
    return tuple(rows)


def _minimal_masks(masks: Iterable[int]) -> tuple[int, ...]:
    """Discard occupancy supersets, which can only make packing harder."""

    kept: list[int] = []
    kept_set: set[int] = set()
    for mask in sorted(set(masks), key=lambda value: (value.bit_count(), value)):
        subset = mask
        dominated = False
        while True:
            if subset in kept_set:
                dominated = True
                break
            if subset == 0:
                break
            subset = (subset - 1) & mask
        if dominated:
            continue
        kept.append(mask)
        kept_set.add(mask)
    return tuple(kept)


def _cover_configurations(
    required_foxes: tuple[Coord, ...],
    all_foxes: tuple[Coord, ...],
    placement_types: tuple[tuple[tuple[Shape, ...], int], ...],
    cell_index: dict[Coord, int],
) -> tuple[int, ...]:
    """Return inclusion-minimal occupancy masks covering every required fox."""

    if not required_foxes:
        return (0,)
    required_index = {cell: index for index, cell in enumerate(required_foxes)}
    full_cover = (1 << len(required_foxes)) - 1
    fox_set = set(all_foxes)
    candidates: list[tuple[int, int, int]] = []
    by_fox: list[list[int]] = [[] for _ in required_foxes]
    for type_index, (placements, cap) in enumerate(placement_types):
        if cap <= 0:
            continue
        for shape in placements:
            if not shape.isdisjoint(fox_set):
                continue
            cover = 0
            for fox in required_foxes:
                if any(adjacent(fox, cell) for cell in shape):
                    cover |= 1 << required_index[fox]
            if not cover:
                continue
            occupancy = sum(1 << cell_index[cell] for cell in shape)
            candidate_index = len(candidates)
            candidates.append((occupancy, cover, type_index))
            for fox_index in range(len(required_foxes)):
                if cover & (1 << fox_index):
                    by_fox[fox_index].append(candidate_index)
    if any(not row for row in by_fox):
        return ()

    caps = tuple(cap for _, cap in placement_types)
    found: set[int] = set()
    visited: set[tuple[int, tuple[int, ...], int]] = set()

    def search(covered: int, used: tuple[int, ...], occupied: int) -> None:
        if covered == full_cover:
            found.add(occupied)
            return
        state = (covered, used, occupied)
        if state in visited:
            return
        visited.add(state)
        uncovered = full_cover & ~covered
        choices = [
            fox_index for fox_index in range(len(required_foxes)) if uncovered & (1 << fox_index)
        ]
        fox_index = min(
            choices,
            key=lambda index: sum(
                used[candidates[candidate][2]] < caps[candidates[candidate][2]]
                and not (occupied & candidates[candidate][0])
                for candidate in by_fox[index]
            ),
        )
        for candidate_index in by_fox[fox_index]:
            placement, cover, type_index = candidates[candidate_index]
            if used[type_index] >= caps[type_index] or occupied & placement:
                continue
            next_used = list(used)
            next_used[type_index] += 1
            search(covered | cover, tuple(next_used), occupied | placement)

    search(0, (0,) * len(caps), 0)
    return _minimal_masks(found)


def _pack_disjoint(configuration_sets: list[tuple[int, ...]]) -> tuple[bool, list[int], int]:
    """Find one disjoint choice from every configuration set."""

    ordered = sorted(enumerate(configuration_sets), key=lambda row: len(row[1]))
    chosen = [0] * len(configuration_sets)
    nodes = 0

    def search(position: int, occupied: int) -> bool:
        nonlocal nodes
        nodes += 1
        if position == len(ordered):
            return True
        original_index, options = ordered[position]
        for option in options:
            if occupied & option:
                continue
            chosen[original_index] = option
            if search(position + 1, occupied | option):
                return True
        return False

    feasible = search(0, 0)
    return feasible, chosen if feasible else [], nodes


def _self_observations_hold(
    foxes: tuple[Coord, ...],
    allowed_missing: Coord | None,
) -> bool:
    for fox in foxes:
        if fox == allowed_missing:
            continue
        if not any(adjacent(fox, other) for other in foxes if other != fox):
            return False
    return True


def _deficit_branches(
    foxes: tuple[Coord, ...],
    deficit: int,
) -> tuple[tuple[str | None, Coord | None], ...]:
    if deficit < 0 or deficit > 1:
        raise ValueError("bitset split solver currently supports deficit zero or one")
    rows: list[tuple[str | None, Coord | None]] = [(None, None)]
    if deficit:
        rows.extend(
            (kind, fox)
            for kind in ("bear", "elk", "hawk", "self")
            for fox in foxes
        )
    return tuple(rows)


def solve_split_shape_packing(
    counts: tuple[int, int, int, int, int],
    salmon: Shape,
    missing_salmon_foxes: int,
    elk_partition: tuple[int, ...],
    required_fox_score: int,
) -> PackingResult:
    """Decide one fixed split-Salmon arithmetic/geometry branch exactly."""

    started = time.monotonic()
    bear_count, _, salmon_count, hawk_count, fox_count = counts
    if salmon_count != 2 or fox_count != 6:
        raise ValueError("split packing requires two salmon and six foxes")
    deficit = 5 * fox_count - required_fox_score - missing_salmon_foxes
    if deficit not in (0, 1):
        raise ValueError("branch does not force all but at most one Fox-A observation")

    possible_foxes = set(boundary(salmon)) | set(_outer_foxes(salmon))
    _, bear_pairs, bear_singles = zero.maximum_bear_structure(bear_count)
    single_shapes = zero.group_placements(possible_foxes, salmon, 1)
    bear_types = (
        (zero.pair_placements(possible_foxes, salmon), bear_pairs),
        (single_shapes, bear_singles),
    )
    elk_types = tuple(
        (
            zero.group_placements(possible_foxes, salmon, length),
            sum(value == length for value in elk_partition),
        )
        for length in sorted(set(elk_partition))
    )
    hawk_types = ((single_shapes, hawk_count),)
    all_shapes = [
        shape
        for placements, _ in (*bear_types, *elk_types, *hawk_types)
        for shape in placements
    ]
    cells = sorted({cell for shape in all_shapes for cell in shape})
    cell_index = {cell: index for index, cell in enumerate(cells)}

    stats = PackingStats()
    for foxes in fox_layouts(salmon, missing_salmon_foxes):
        stats = stats.add(fox_layouts=1)
        for missing_kind, missing_fox in _deficit_branches(foxes, deficit):
            stats = stats.add(deficit_branches=1)
            if not _self_observations_hold(
                foxes, missing_fox if missing_kind == "self" else None
            ):
                continue
            configuration_sets = []
            descriptions = []
            for kind, placement_types in (
                ("bear", bear_types),
                ("elk", elk_types),
                ("hawk", hawk_types),
            ):
                required = tuple(
                    fox
                    for fox in foxes
                    if not (missing_kind == kind and missing_fox == fox)
                )
                configurations = _cover_configurations(
                    required,
                    foxes,
                    placement_types,
                    cell_index,
                )
                stats = stats.add(
                    cover_searches=1,
                    cover_configurations=len(configurations),
                )
                if not configurations:
                    break
                configuration_sets.append(configurations)
                descriptions.append(kind)
            if len(configuration_sets) != 3:
                continue
            feasible, chosen, nodes = _pack_disjoint(configuration_sets)
            stats = stats.add(packing_nodes=nodes)
            if feasible:
                return PackingResult(
                    status="RELAXATION_FEASIBLE",
                    wall_seconds=time.monotonic() - started,
                    stats=stats,
                    witness={
                        "foxes": [list(cell) for cell in foxes],
                        "missing_observation": (
                            None
                            if missing_kind is None
                            else {"kind": missing_kind, "fox": list(missing_fox)}
                        ),
                        "occupancy_masks": dict(zip(descriptions, chosen, strict=True)),
                    },
                )
    return PackingResult(
        status="INFEASIBLE",
        wall_seconds=time.monotonic() - started,
        stats=stats,
        witness=None,
    )


def split_branch_packing(
    counts: tuple[int, int, int, int, int],
    target: int,
) -> dict[str, Any]:
    """Exhaust every split-Salmon branch capable of reaching ``target``."""

    bear, elk, salmon_count, hawk, fox = counts
    if salmon_count != 2 or fox != 6:
        raise ValueError("split branch requires two salmon and six foxes")
    bear_score = zero.maximum_bear_structure(bear)[0]
    hawk_score = base.HAWK_SCORES[hawk]
    salmon_score = 4
    cases = []
    started = time.monotonic()
    for elk_score, partitions in sorted(zero.elk_partitions_by_score(elk).items(), reverse=True):
        required_fox = target - bear_score - salmon_score - hawk_score - elk_score
        if required_fox > 5 * fox:
            continue
        maximum_misses = min(1, 5 * fox - required_fox)
        for partition in partitions:
            for missing in range(maximum_misses + 1):
                for shape_index, salmon in enumerate(split_singleton_shapes()):
                    result = solve_split_shape_packing(
                        counts,
                        salmon,
                        missing,
                        partition,
                        required_fox,
                    )
                    row = {
                        "elk_partition": list(partition),
                        "elk_score": elk_score,
                        "required_fox_score": required_fox,
                        "missing_salmon_foxes": missing,
                        "salmon_shape_index": shape_index,
                        "result": {
                            **asdict(result),
                            "stats": asdict(result.stats),
                        },
                    }
                    cases.append(row)
                    if result.status == "RELAXATION_FEASIBLE":
                        return {
                            "status": "RELAXATION_FEASIBLE",
                            "target": target,
                            "case_count": len(cases),
                            "feasible_case": row,
                            "wall_seconds": time.monotonic() - started,
                        }
    return {
        "status": "INFEASIBLE",
        "upper_bound": target - 1,
        "target": target,
        "case_count": len(cases),
        "cases": cases,
        "wall_seconds": time.monotonic() - started,
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.counts.split(","))
    if len(counts) != 5:
        raise ValueError("counts must contain five comma-separated integers")
    payload = split_branch_packing(counts, args.target)  # type: ignore[arg-type]
    if args.output:
        _write_atomic(args.output, payload)
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
