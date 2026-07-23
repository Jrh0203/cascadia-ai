#!/usr/bin/env python3
"""Deterministic motif-incompatibility certificate for one AAAAA tail case.

For counts (bear, elk, salmon, hawk, fox) = (3, 6, 6, 0, 5), the geometry-
free AAAAA upper bound is 62. Reaching it requires every species to attain its
standalone maximum. This module exhaustively enumerates a deliberate superset
of those maximum-score layouts and proves that even the superset is empty.

The relaxation drops whole-board connectivity and lets any bear component or
elk scoring line that covers no fox live abstractly at infinity. It preserves
only forced local cells and their non-overlap. Therefore infeasibility here is
a sound upper-bound certificate for real connected boards.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections.abc import Iterable
from functools import reduce
from pathlib import Path
from typing import Any

from tools.aaaaa_wildlife_exact import components, score_tokens

COUNTS = (3, 6, 6, 0, 5)
EXCLUDED_SCORE = 62
CERTIFIED_UPPER_BOUND = 61
DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
LINE_DIRECTIONS = DIRECTIONS[:3]
DIHEDRAL_TRANSFORMS = (
    ((1, 0), (0, 1)),
    ((0, -1), (1, 1)),
    ((-1, -1), (1, 0)),
    ((-1, 0), (0, -1)),
    ((0, 1), (-1, -1)),
    ((1, 1), (-1, 0)),
    ((0, 1), (1, 0)),
    ((-1, 0), (1, 1)),
    ((-1, -1), (0, 1)),
    ((0, -1), (-1, 0)),
    ((1, 0), (-1, -1)),
    ((1, 1), (0, -1)),
)
ELK_GROUP_SCORE = {1: 2, 2: 5, 3: 9, 4: 13}

Coord = tuple[int, int]
Shape = frozenset[Coord]


def adjacent(left: Coord, right: Coord) -> bool:
    return (right[0] - left[0], right[1] - left[1]) in DIRECTIONS


def boundary(shape: Iterable[Coord]) -> set[Coord]:
    occupied = set(shape)
    return {
        (q + dq, r + dr)
        for q, r in occupied
        for dq, dr in DIRECTIONS
        if (q + dq, r + dr) not in occupied
    }


def normalize(shape: Iterable[Coord]) -> tuple[Coord, ...]:
    cells = tuple(shape)
    return min(
        tuple(sorted((q - anchor_q, r - anchor_r) for q, r in cells))
        for anchor_q, anchor_r in cells
    )


def canonical_free_shape(shape: Iterable[Coord]) -> tuple[Coord, ...]:
    cells = tuple(shape)
    images = []
    for (qq, qr), (rq, rr) in DIHEDRAL_TRANSFORMS:
        image = {(qq * q + qr * r, rq * q + rr * r) for q, r in cells}
        images.append(normalize(image))
    return min(images)


def free_polyhexes(size: int) -> set[tuple[Coord, ...]]:
    """Enumerate every free connected polyhex through ``size`` by induction."""
    if size < 1:
        raise ValueError("polyhex size must be positive")
    shapes = {((0, 0),)}
    for _ in range(2, size + 1):
        shapes = {
            canonical_free_shape(set(shape) | {(q + dq, r + dr)})
            for shape in shapes
            for q, r in shape
            for dq, dr in DIRECTIONS
            if (q + dq, r + dr) not in shape
        }
    return shapes


def valid_six_salmon_shapes() -> list[Shape]:
    """All free connected six-hex components with no branching vertex."""
    result = []
    for row in free_polyhexes(6):
        shape = frozenset(row)
        if all(sum(adjacent(cell, other) for other in shape) <= 2 for cell in shape):
            result.append(shape)
    return sorted(result, key=lambda shape: tuple(sorted(shape)))


def coverage_mask(foxes: tuple[Coord, ...], cell: Coord) -> int:
    mask = 0
    for index, fox in enumerate(foxes):
        if adjacent(fox, cell):
            mask |= 1 << index
    return mask


def foxes_have_no_isolate(foxes: tuple[Coord, ...]) -> bool:
    return all(any(left != right and adjacent(left, right) for right in foxes) for left in foxes)


def relaxed_bear_local_placements(
    foxes: tuple[Coord, ...], blocked: set[Coord]
) -> set[Shape]:
    """Forced local cells for every relaxed maximum-score three-bear layout.

    A Bear-A score of four with three bears requires exactly one pair and one
    singleton. A non-covering pair or singleton is represented abstractly by
    omitting it from the returned local shape.
    """
    fox_boundary = boundary(foxes) - blocked
    full = (1 << len(foxes)) - 1
    placements: set[Shape] = set()

    # The singleton covers every fox; its pair can be placed abstractly.
    for singleton in fox_boundary:
        if coverage_mask(foxes, singleton) == full:
            placements.add(frozenset({singleton}))

    # At least one endpoint of a locally relevant pair covers a fox. The
    # other endpoint may be just outside the fox boundary.
    pair_edges = {
        frozenset({left, (left[0] + dq, left[1] + dr)})
        for left in fox_boundary
        for dq, dr in DIRECTIONS
        if (left[0] + dq, left[1] + dr) not in blocked
    }
    for pair in pair_edges:
        left, right = tuple(pair)
        pair_mask = coverage_mask(foxes, left) | coverage_mask(foxes, right)
        if pair_mask == full:
            placements.add(pair)  # The singleton may be abstract.
        for singleton in fox_boundary - pair:
            if adjacent(singleton, left) or adjacent(singleton, right):
                continue
            if pair_mask | coverage_mask(foxes, singleton) == full:
                placements.add(pair | {singleton})
    return placements


def integer_partitions(total: int, maximum: int) -> Iterable[tuple[int, ...]]:
    def visit(remaining: int, largest: int, prefix: tuple[int, ...]) -> Iterable[tuple[int, ...]]:
        if remaining == 0:
            yield prefix
            return
        for value in range(min(remaining, largest), 0, -1):
            yield from visit(remaining - value, value, (*prefix, value))

    return visit(total, maximum, ())


def optimal_elk_partitions(count: int) -> tuple[tuple[int, ...], ...]:
    rows = list(integer_partitions(count, 4))
    best = max(sum(ELK_GROUP_SCORE[value] for value in row) for row in rows)
    return tuple(row for row in rows if sum(ELK_GROUP_SCORE[value] for value in row) == best)


def local_line_placements(
    foxes: tuple[Coord, ...], length: int, blocked: set[Coord]
) -> list[tuple[Shape, int]]:
    """Every straight line of ``length`` touching at least one fox neighbor."""
    result: dict[Shape, int] = {}
    for touching_cell in boundary(foxes) - blocked:
        for dq, dr in LINE_DIRECTIONS:
            for touching_offset in range(length):
                line = frozenset(
                    (
                        touching_cell[0] + (step - touching_offset) * dq,
                        touching_cell[1] + (step - touching_offset) * dr,
                    )
                    for step in range(length)
                )
                if line & blocked:
                    continue
                result[line] = reduce(
                    int.__or__, (coverage_mask(foxes, cell) for cell in line), 0
                )
    return list(result.items())


def relaxed_elk_local_placements(
    foxes: tuple[Coord, ...], blocked: set[Coord]
) -> set[Shape]:
    """Forced local cells for every relaxed maximum-score six-elk packing."""
    full = (1 << len(foxes)) - 1
    placements: set[Shape] = set()
    for first_length, second_length in optimal_elk_partitions(6):
        first = local_line_placements(foxes, first_length, blocked)
        second = local_line_placements(foxes, second_length, blocked)
        for line, mask in first:
            if mask == full:
                placements.add(line)  # The other group may be abstract.
        for line, mask in second:
            if mask == full:
                placements.add(line)
        for first_line, first_mask in first:
            for second_line, second_mask in second:
                if first_line.isdisjoint(second_line) and first_mask | second_mask == full:
                    placements.add(first_line | second_line)
    return placements


def enumerate_relaxed_superset() -> dict[str, int | bool]:
    stats = {
        "free_polyhexes_size_6": 0,
        "valid_salmon_shapes": 0,
        "fox_boundary_sets_no_isolates": 0,
        "fox_sets_with_relaxed_bear_coverage": 0,
        "fox_sets_with_relaxed_bear_and_elk_coverage": 0,
        "nonoverlapping_relaxed_realisations": 0,
    }
    salmon_shapes = valid_six_salmon_shapes()
    stats["free_polyhexes_size_6"] = len(free_polyhexes(6))
    stats["valid_salmon_shapes"] = len(salmon_shapes)

    for salmon in salmon_shapes:
        for foxes in itertools.combinations(sorted(boundary(salmon)), 5):
            if not foxes_have_no_isolate(foxes):
                continue
            stats["fox_boundary_sets_no_isolates"] += 1
            blocked = set(salmon) | set(foxes)
            bears = relaxed_bear_local_placements(foxes, blocked)
            if not bears:
                continue
            stats["fox_sets_with_relaxed_bear_coverage"] += 1
            elk = relaxed_elk_local_placements(foxes, blocked)
            if not elk:
                continue
            stats["fox_sets_with_relaxed_bear_and_elk_coverage"] += 1
            compatible = sum(
                bear.isdisjoint(elk_group) for bear in bears for elk_group in elk
            )
            stats["nonoverlapping_relaxed_realisations"] += compatible
    stats["infeasible"] = stats["nonoverlapping_relaxed_realisations"] == 0
    return stats


def load_incumbent(catalog_path: Path) -> dict[str, Any]:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    row = next(row for row in payload["results"] if tuple(row["counts"]) == COUNTS)
    tokens = list(row["tokens"])
    breakdown = list(score_tokens(tokens))
    if sum(breakdown) != CERTIFIED_UPPER_BOUND:
        raise ValueError(
            f"catalog incumbent scores {sum(breakdown)}, expected {CERTIFIED_UPPER_BOUND}"
        )
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    if len(tokens) != 20 or len(occupied) != 20 or len(components(occupied)) != 1:
        raise ValueError("catalog incumbent is not a valid connected 20-token board")
    return {"score": sum(breakdown), "score_breakdown": breakdown, "tokens": tokens}


def certificate(catalog_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    stats = enumerate_relaxed_superset()
    if not stats["infeasible"]:
        raise RuntimeError("relaxed motif superset has a realization; certificate failed")
    source = Path(__file__).resolve()
    return {
        "schema": "aaaaa-motif-incompatibility-certificate-v1",
        "counts": list(COUNTS),
        "excluded_score": EXCLUDED_SCORE,
        "certified_upper_bound": CERTIFIED_UPPER_BOUND,
        "proof_complete": True,
        "proof_method": "standalone_maximum_motif_incompatibility",
        "relaxation": {
            "whole_board_connectivity_required": False,
            "noncovering_bear_components_may_be_abstract": True,
            "noncovering_elk_lines_may_be_abstract": True,
            "forced_local_cells_must_not_overlap": True,
        },
        "enumeration": stats,
        "incumbent": load_incumbent(catalog_path),
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = certificate(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload["enumeration"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
