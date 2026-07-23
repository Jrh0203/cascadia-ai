#!/usr/bin/env python3
"""Derive exact Fox-A two-target-species observation capacities.

The useful cross-edge support decomposes into connected components.  Each
component has at most ``fox + first + second`` vertices and therefore fits,
after translation, in the complete radius-``n-1`` disk.  CP-SAT proves every
connected component maximum; a three-dimensional DP combines components and
isolated vertices without assuming global connectivity.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from tools.cbddb_wildlife_exact import adjacent
from tools.derive_hex_bipartite_edge_bounds import CAP, adjacency_variables

PAIR_COUNT = CAP**3


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def connected_component_maximum(
    fox_count: int,
    first_count: int,
    second_count: int,
    *,
    seconds: float,
    workers: int,
) -> dict[str, Any]:
    if not all(1 <= count <= CAP for count in (fox_count, first_count, second_count)):
        raise ValueError("component class sizes must be in 1..6")
    token_count = fox_count + first_count + second_count
    radius = token_count - 1
    width = 2 * radius + 1
    model = cp_model.CpModel()
    q = [model.new_int_var(-radius, radius, f"q_{i}") for i in range(token_count)]
    r = [model.new_int_var(-radius, radius, f"r_{i}") for i in range(token_count)]
    coordinate_id = [
        model.new_int_var(0, width**2 - 1, f"coord_{i}")
        for i in range(token_count)
    ]
    for token in range(token_count):
        diagonal = model.new_int_var(-radius, radius, f"diag_{token}")
        model.add(diagonal == q[token] + r[token])
        model.add(
            coordinate_id[token] == (q[token] + radius) * width + r[token] + radius
        )
    model.add_all_different(coordinate_id)
    model.add(q[0] == 0)
    model.add(r[0] == 0)
    class_ranges = (
        range(fox_count),
        range(fox_count, fox_count + first_count),
        range(fox_count + first_count, token_count),
    )
    for indices in class_ranges:
        for left, right in itertools.pairwise(indices):
            model.add(coordinate_id[left] < coordinate_id[right])

    adjacency = adjacency_variables(model, q, r, radius)
    targets = range(fox_count, token_count)
    depth = [
        model.new_int_var(0, token_count - 1, f"depth_{token}")
        for token in range(token_count)
    ]
    model.add(depth[0] == 0)
    for child in range(1, token_count):
        possible_parents = targets if child < fox_count else range(fox_count)
        selected_parents = []
        for parent in possible_parents:
            selected = model.new_bool_var(f"parent_{child}_{parent}")
            model.add(selected <= adjacent(adjacency, child, parent))
            model.add(depth[child] > depth[parent]).only_enforce_if(selected)
            selected_parents.append(selected)
        model.add_exactly_one(selected_parents)

    first_targets = range(fox_count, fox_count + first_count)
    second_targets = range(fox_count + first_count, token_count)
    observes_both = []
    for fox in range(fox_count):
        first_seen = model.new_bool_var(f"fox_{fox}_sees_first")
        first_degree = sum(adjacent(adjacency, fox, target) for target in first_targets)
        model.add(first_degree >= 1).only_enforce_if(first_seen)
        model.add(first_degree == 0).only_enforce_if(first_seen.negated())
        second_seen = model.new_bool_var(f"fox_{fox}_sees_second")
        second_degree = sum(adjacent(adjacency, fox, target) for target in second_targets)
        model.add(second_degree >= 1).only_enforce_if(second_seen)
        model.add(second_degree == 0).only_enforce_if(second_seen.negated())
        both = model.new_bool_var(f"fox_{fox}_sees_both")
        model.add(both <= first_seen)
        model.add(both <= second_seen)
        model.add(both >= first_seen + second_seen - 1)
        observes_both.append(both)
    objective = model.new_int_var(0, fox_count, "foxes_observing_both")
    model.add(objective == sum(observes_both))
    model.maximize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 20260723
    started = time.monotonic()
    status_code = solver.solve(model)
    status = solver.status_name(status_code)
    return {
        "foxes": fox_count,
        "first_targets": first_count,
        "second_targets": second_count,
        "status": status,
        "maximum": (
            int(solver.objective_value)
            if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE)
            else None
        ),
        "best_bound": (
            int(solver.best_objective_bound)
            if status_code != cp_model.MODEL_INVALID
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
        "branches": solver.num_branches,
        "conflicts": solver.num_conflicts,
    }


def combine_components(component: list[list[list[int]]]) -> list[list[list[int]]]:
    result = [
        [[0] * (CAP + 1) for _ in range(CAP + 1)]
        for _ in range(CAP + 1)
    ]
    for foxes in range(CAP + 1):
        for first in range(CAP + 1):
            for second in range(CAP + 1):
                best = 0
                for component_foxes in range(1, foxes + 1):
                    for component_first in range(1, first + 1):
                        for component_second in range(1, second + 1):
                            best = max(
                                best,
                                component[component_foxes][component_first][
                                    component_second
                                ]
                                + result[foxes - component_foxes][
                                    first - component_first
                                ][second - component_second],
                            )
                result[foxes][first][second] = best
    return result


def collect_shards(paths: list[Path]) -> dict[str, Any]:
    proofs_by_counts: dict[tuple[int, int, int], dict[str, Any]] = {}
    shard_hashes = {}
    for path in paths:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        if payload.get("schema") != "hex-dual-observation-bound-shard-v1":
            raise ValueError(f"{path}: unexpected schema")
        shard_hashes[str(path)] = hashlib.sha256(encoded).hexdigest()
        for proof in payload["proofs"]:
            counts = (
                int(proof["foxes"]),
                int(proof["first_targets"]),
                int(proof["second_targets"]),
            )
            if counts in proofs_by_counts:
                raise ValueError(f"duplicate component counts {counts}")
            proofs_by_counts[counts] = proof
    expected = {
        (foxes, first, second)
        for foxes in range(1, CAP + 1)
        for first in range(1, CAP + 1)
        for second in range(1, CAP + 1)
    }
    if set(proofs_by_counts) != expected:
        raise ValueError("component coverage is not exact")
    proofs = [proofs_by_counts[counts] for counts in sorted(expected)]
    for proof in proofs:
        if proof["status"] == "OPTIMAL":
            if proof["maximum"] != proof["best_bound"]:
                raise ValueError("component objective/bound mismatch")
        elif proof["status"] != "INFEASIBLE":
            raise ValueError("one or more component proofs are incomplete")
    component = [
        [[0] * (CAP + 1) for _ in range(CAP + 1)]
        for _ in range(CAP + 1)
    ]
    for proof in proofs:
        if proof["status"] == "OPTIMAL":
            component[proof["foxes"]][proof["first_targets"]][
                proof["second_targets"]
            ] = int(proof["maximum"])
    if any(
        component[foxes][first][second] != component[foxes][second][first]
        for foxes in range(1, CAP + 1)
        for first in range(1, CAP + 1)
        for second in range(1, CAP + 1)
    ):
        raise ValueError("target-class symmetry check failed")
    return {
        "schema": "hex-dual-observation-bound-derivation-v1",
        "proof_complete": True,
        "cap": CAP,
        "connected_component_maximum": component,
        "global_maximum": combine_components(component),
        "shard_sha256": shard_hashes,
        "proofs": proofs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=PAIR_COUNT)
    parser.add_argument("--collect", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.collect is not None:
        payload = collect_shards(args.collect)
        _write_atomic(args.output, payload)
        print(json.dumps({"proof_complete": True, "components": PAIR_COUNT}))
        return 0
    if not (0 <= args.start_index < args.end_index <= PAIR_COUNT):
        parser.error(f"expected 0 <= start-index < end-index <= {PAIR_COUNT}")
    triples = [
        (foxes, first, second)
        for foxes in range(1, CAP + 1)
        for first in range(1, CAP + 1)
        for second in range(1, CAP + 1)
    ][args.start_index : args.end_index]
    proofs = [
        connected_component_maximum(
            foxes,
            first,
            second,
            seconds=args.seconds,
            workers=args.workers,
        )
        for foxes, first, second in triples
    ]
    incomplete = [
        proof
        for proof in proofs
        if proof["status"] not in ("OPTIMAL", "INFEASIBLE")
    ]
    payload = {
        "schema": "hex-dual-observation-bound-shard-v1",
        "proof_complete": not incomplete,
        "cap": CAP,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "configuration": {
            "seconds_per_component": args.seconds,
            "workers": args.workers,
            "random_seed": 20260723,
        },
        "proofs": proofs,
    }
    _write_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "proof_complete": not incomplete,
                "resolved_components": len(proofs) - len(incomplete),
                "components": len(proofs),
            },
            sort_keys=True,
        )
    )
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
