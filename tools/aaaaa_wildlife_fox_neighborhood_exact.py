#!/usr/bin/env python3
"""AAAAA exact model with realizable radius-two fox-neighborhood tables.

Fox-A pressure is governed not only by fox-fox adjacency, but by whether two
foxes can share an adjacent non-fox witness.  Two foxes share a witness only
when their hex distance is at most two.  This wrapper classifies every fox
pair as distance 1, distance 2, or farther, then constrains each set of up to
five foxes to a radius-two relation graph realizable on the hex lattice.

For six foxes the six overlapping five-fox tables are a sound local
consistency relaxation; for at most five foxes the table is exact.  The
coordinate model remains authoritative, so these redundant tables only
improve propagation and never remove a legal board.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import time
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools.aaaaa_wildlife_motif_certificate import (
    DIHEDRAL_TRANSFORMS,
    normalize,
)

Coord = tuple[int, int]
MAX_TABLE_SIZE = 5
DISTANCE_TWO = tuple(
    (q, r)
    for q in range(-2, 3)
    for r in range(-2, 3)
    if max(abs(q), abs(r), abs(q + r)) == 2
)
NEAR_OFFSETS = (*base.DIRECTIONS, *DISTANCE_TWO)


def _transform(
    coord: Coord,
    transform: tuple[tuple[int, int], tuple[int, int]],
) -> Coord:
    (qq, qr), (rq, rr) = transform
    q, r = coord
    return qq * q + qr * r, rq * q + rr * r


def _canonical_shape(coords: Iterable[Coord]) -> tuple[Coord, ...]:
    source = tuple(coords)
    return min(
        normalize(_transform(cell, transform) for cell in source)
        for transform in DIHEDRAL_TRANSFORMS
    )


def _hex_distance(left: Coord, right: Coord) -> int:
    dq = right[0] - left[0]
    dr = right[1] - left[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def _state_code_for_order(
    coords: tuple[Coord, ...],
    order: tuple[int, ...],
) -> int:
    code = 0
    multiplier = 1
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            distance = _hex_distance(coords[order[left]], coords[order[right]])
            code += (distance if distance <= 2 else 0) * multiplier
            multiplier *= 3
    return code


@cache
def _near_connected_shapes(size: int) -> frozenset[tuple[Coord, ...]]:
    if size < 1:
        raise ValueError("shape size must be positive")
    shapes = {((0, 0),)}
    for _ in range(2, size + 1):
        shapes = {
            _canonical_shape((*shape, (q + dq, r + dr)))
            for shape in shapes
            for q, r in shape
            for dq, dr in NEAR_OFFSETS
            if (q + dq, r + dr) not in shape
        }
    return frozenset(shapes)


@cache
def _connected_state_codes(size: int) -> frozenset[int]:
    permutations = tuple(itertools.permutations(range(size)))
    return frozenset(
        min(_state_code_for_order(shape, order) for order in permutations)
        for shape in _near_connected_shapes(size)
    )


def _state_adjacency(size: int, row: tuple[int, ...]) -> list[int]:
    adjacency = [0] * size
    index = 0
    for left in range(size):
        for right in range(left + 1, size):
            if row[index]:
                adjacency[left] |= 1 << right
                adjacency[right] |= 1 << left
            index += 1
    return adjacency


def _components(adjacency: list[int]) -> list[list[int]]:
    unseen = (1 << len(adjacency)) - 1
    result = []
    while unseen:
        first = (unseen & -unseen).bit_length() - 1
        component = 1 << first
        previous = 0
        while component != previous:
            previous = component
            frontier = component
            while frontier:
                token = (frontier & -frontier).bit_length() - 1
                frontier &= frontier - 1
                component |= adjacency[token]
        unseen &= ~component
        result.append([token for token in range(len(adjacency)) if component & (1 << token)])
    return result


def _component_code(
    vertices: list[int],
    states: list[list[int]],
) -> int:
    codes = []
    for order in itertools.permutations(vertices):
        code = 0
        multiplier = 1
        for left in range(len(order)):
            for right in range(left + 1, len(order)):
                code += states[order[left]][order[right]] * multiplier
                multiplier *= 3
        codes.append(code)
    return min(codes)


@cache
def neighborhood_rows(size: int) -> tuple[tuple[int, ...], ...]:
    """Return all realizable labeled radius-two relation rows."""

    if not 1 <= size <= MAX_TABLE_SIZE:
        raise ValueError(f"table size must be between one and {MAX_TABLE_SIZE}")
    edge_count = size * (size - 1) // 2
    result = []
    for row in itertools.product(range(3), repeat=edge_count):
        adjacency = _state_adjacency(size, row)
        states = [[0] * size for _ in range(size)]
        index = 0
        for left in range(size):
            for right in range(left + 1, size):
                states[left][right] = row[index]
                states[right][left] = row[index]
                index += 1
        if all(
            _component_code(component, states) in _connected_state_codes(len(component))
            for component in _components(adjacency)
        ):
            result.append(row)
    return tuple(result)


def _variables_by_name(model: cp_model.CpModel) -> dict[str, cp_model.IntVar]:
    return {
        variable.name: model.get_int_var_from_proto_index(index)
        for index, variable in enumerate(model.proto.variables)
    }


def add_fox_neighborhood_tables(
    model: cp_model.CpModel,
    variables: base.ExactVariables,
) -> None:
    named = _variables_by_name(model)
    foxes = [
        token
        for token, species in enumerate(variables.species_by_token)
        if species == base.SPECIES_CODE["fox"]
    ]
    if len(foxes) < 2:
        return

    states: dict[tuple[int, int], cp_model.IntVar] = {}
    for left, right in itertools.combinations(foxes, 2):
        dq = named[f"dq_{left}_{right}"]
        dr = named[f"dr_{left}_{right}"]
        adjacent = named[f"adj_{left}_{right}"]
        distance_two = model.new_bool_var(f"fox_distance_two_{left}_{right}")
        model.add_allowed_assignments([dq, dr], DISTANCE_TWO).only_enforce_if(distance_two)
        model.add_forbidden_assignments([dq, dr], DISTANCE_TWO).only_enforce_if(
            distance_two.negated()
        )
        state = model.new_int_var(0, 2, f"fox_neighborhood_state_{left}_{right}")
        model.add(state == adjacent + 2 * distance_two)
        states[(left, right)] = state

    table_size = min(MAX_TABLE_SIZE, len(foxes))
    rows = neighborhood_rows(table_size)
    for subset in itertools.combinations(foxes, table_size):
        relations = [
            states[(left, right)]
            for left, right in itertools.combinations(subset, 2)
        ]
        model.add_allowed_assignments(relations, rows)


def build_model(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
    initial_tokens: list[dict[str, int | str]] | None = None,
    fix_initial_tokens: bool = False,
) -> tuple[cp_model.CpModel, base.ExactVariables]:
    model, variables = base.build_model(
        counts,
        minimum_score,
        maximize=maximize,
        maximum_score=maximum_score,
        enforce_connectivity=enforce_connectivity,
        initial_tokens=initial_tokens,
        fix_initial_tokens=fix_initial_tokens,
    )
    add_fox_neighborhood_tables(model, variables)
    return model, variables


def solve_counts(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    time_limit: float,
    workers: int,
    seed: int,
    *,
    maximum_score: int | None,
    enforce_connectivity: bool,
    initial_tokens: list[dict[str, int | str]] | None,
) -> dict[str, Any]:
    started = time.monotonic()
    model, variables = build_model(
        counts,
        minimum_score,
        maximize=False,
        maximum_score=maximum_score,
        enforce_connectivity=enforce_connectivity,
        initial_tokens=initial_tokens,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    tokens: list[dict[str, int | str]] = []
    objective: int | None = None
    score_breakdown: list[int] | None = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        tokens = [
            {
                "q": int(solver.value(variables.q[token])),
                "r": int(solver.value(variables.r[token])),
                "wildlife": base.SPECIES[species],
            }
            for token, species in enumerate(variables.species_by_token)
        ]
        tokens.sort(key=lambda row: (int(row["r"]), int(row["q"]), str(row["wildlife"])))
        score_breakdown = list(base.score_tokens(tokens))
        objective = sum(score_breakdown)
        if objective < minimum_score:
            raise RuntimeError(f"model witness scored {objective}, below {minimum_score}")
        occupied = {(int(row["q"]), int(row["r"])) for row in tokens}
        if enforce_connectivity and len(base.components(occupied)) != 1:
            raise RuntimeError("fox-neighborhood model emitted a disconnected witness")
    return {
        "counts": list(counts),
        "minimum_score": minimum_score,
        "model_status": solver.status_name(status),
        "objective": objective,
        "score_breakdown": score_breakdown,
        "best_bound": solver.best_objective_bound,
        "wall_seconds": solver.wall_time,
        "elapsed_seconds": time.monotonic() - started,
        "branches": solver.num_branches,
        "conflicts": solver.num_conflicts,
        "solver_parameters": {
            "time_limit_seconds": time_limit,
            "workers": workers,
            "random_seed": seed,
            "maximum_score": maximum_score,
            "enforce_connectivity": enforce_connectivity,
            "fox_neighborhood_table_size": min(MAX_TABLE_SIZE, counts[-1]),
        },
        "tokens": tokens,
    }


def _parse_counts(value: str) -> tuple[int, int, int, int, int]:
    counts = tuple(int(part) for part in value.split(","))
    if len(counts) != 5:
        raise argparse.ArgumentTypeError("counts must contain five comma-separated integers")
    return counts  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--counts", required=True, type=_parse_counts)
    parser.add_argument("--minimum-score", required=True, type=int)
    parser.add_argument("--maximum-score", type=int, default=68)
    parser.add_argument("--time-limit", type=float, default=60)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--disconnected", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    row = next(
        row for row in catalog["results"]
        if tuple(int(value) for value in row["counts"]) == args.counts
    )
    tokens = list(row["tokens"])
    result = solve_counts(
        args.counts,
        args.minimum_score,
        args.time_limit,
        args.workers,
        args.seed,
        maximum_score=args.maximum_score,
        enforce_connectivity=not args.disconnected,
        initial_tokens=tokens,
    )
    source = Path(__file__).resolve()
    payload = {
        "schema": "aaaaa-fox-neighborhood-calibration-v1",
        "proof_complete": result["model_status"] == "INFEASIBLE",
        "result": result,
        "ortools_version": ORTOOLS_VERSION,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "exact_source_sha256": hashlib.sha256(
            source.with_name("aaaaa_wildlife_exact.py").read_bytes()
        ).hexdigest(),
        "catalog": {
            "path": str(args.catalog),
            "sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        },
        "table_rows": {
            str(size): len(neighborhood_rows(size))
            for size in range(1, MAX_TABLE_SIZE + 1)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": result["model_status"],
                "branches": result["branches"],
                "conflicts": result["conflicts"],
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
