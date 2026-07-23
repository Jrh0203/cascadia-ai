#!/usr/bin/env python3
"""Derive exact cap-six cross-adjacency bounds on the infinite hex lattice.

Every cross-edge graph decomposes into connected components.  A component
with ``left + right`` vertices has hex-graph diameter at most that value minus
one, so after translation it lies in the finite disk modeled here.  CP-SAT
proves the maximum edge count for each connected component size.  A small
integer DP then combines components and isolated vertices, producing a
globally exact table without assuming that an optimum is connected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from tools.cbddb_wildlife_exact import adjacency_variables, adjacent

CAP = 6


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def connected_component_maximum(
    left_count: int,
    right_count: int,
    *,
    seconds: float,
    workers: int,
) -> dict[str, Any]:
    if not (1 <= left_count <= CAP and 1 <= right_count <= CAP):
        raise ValueError("component sides must both be in 1..6")
    token_count = left_count + right_count
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
    for indices in (range(left_count), range(left_count, token_count)):
        for first, second in zip(indices, list(indices)[1:], strict=False):
            model.add(coordinate_id[first] < coordinate_id[second])

    adjacency = adjacency_variables(model, q, r)
    cross_edges = [
        adjacent(adjacency, left, right)
        for left in range(left_count)
        for right in range(left_count, token_count)
    ]

    # Require the cross-edge support to be one connected bipartite component.
    depth = [
        model.new_int_var(0, token_count - 1, f"depth_{token}")
        for token in range(token_count)
    ]
    model.add(depth[0] == 0)
    for child in range(1, token_count):
        child_is_left = child < left_count
        possible_parents = (
            range(left_count, token_count)
            if child_is_left
            else range(left_count)
        )
        selected_parents = []
        for parent in possible_parents:
            selected = model.new_bool_var(f"parent_{child}_{parent}")
            model.add(selected <= adjacent(adjacency, child, parent))
            model.add(depth[child] > depth[parent]).only_enforce_if(selected)
            selected_parents.append(selected)
        model.add_exactly_one(selected_parents)

    objective = model.new_int_var(0, left_count * right_count, "cross_edges")
    model.add(objective == sum(cross_edges))
    model.maximize(objective)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 20260723
    started = time.monotonic()
    status_code = solver.solve(model)
    status = solver.status_name(status_code)
    return {
        "left": left_count,
        "right": right_count,
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


def combine_components(component: list[list[int]]) -> list[list[int]]:
    result = [[0] * (CAP + 1) for _ in range(CAP + 1)]
    for left in range(CAP + 1):
        for right in range(CAP + 1):
            best = 0
            for component_left in range(1, left + 1):
                for component_right in range(1, right + 1):
                    best = max(
                        best,
                        component[component_left][component_right]
                        + result[left - component_left][right - component_right],
                    )
            result[left][right] = best
    return result


def collect_shards(shard_paths: list[Path]) -> dict[str, Any]:
    proofs_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    shard_hashes = {}
    for path in shard_paths:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        if payload.get("schema") != "hex-bipartite-edge-bound-shard-v1":
            raise ValueError(f"{path}: unexpected shard schema")
        shard_hashes[str(path)] = hashlib.sha256(encoded).hexdigest()
        for proof in payload["proofs"]:
            pair = (int(proof["left"]), int(proof["right"]))
            if pair in proofs_by_pair:
                raise ValueError(f"duplicate component pair {pair}")
            proofs_by_pair[pair] = proof
    expected = {
        (left, right)
        for left in range(1, CAP + 1)
        for right in range(1, CAP + 1)
    }
    if set(proofs_by_pair) != expected:
        missing = sorted(expected - set(proofs_by_pair))
        extra = sorted(set(proofs_by_pair) - expected)
        raise ValueError(f"incomplete component coverage; missing={missing}, extra={extra}")
    proofs = [proofs_by_pair[pair] for pair in sorted(expected)]
    incomplete = [proof for proof in proofs if proof["status"] != "OPTIMAL"]
    if incomplete:
        raise ValueError(
            "component proofs are incomplete: "
            + ", ".join(f"{row['left']},{row['right']}={row['status']}" for row in incomplete)
        )
    component = [[0] * (CAP + 1) for _ in range(CAP + 1)]
    for proof in proofs:
        if proof["maximum"] != proof["best_bound"]:
            raise ValueError(f"objective/bound mismatch for {proof['left']},{proof['right']}")
        component[proof["left"]][proof["right"]] = int(proof["maximum"])
    if any(
        component[left][right] != component[right][left]
        for left in range(1, CAP + 1)
        for right in range(1, CAP + 1)
    ):
        raise ValueError("left/right symmetry check failed")
    return {
        "schema": "hex-bipartite-edge-bound-derivation-v1",
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
    parser.add_argument("--end-index", type=int, default=CAP * CAP)
    parser.add_argument("--collect", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.collect is not None:
        payload = collect_shards(args.collect)
        _write_atomic(args.output, payload)
        print(
            json.dumps(
                {
                    "proof_complete": True,
                    "optimal_components": len(payload["proofs"]),
                    "components": CAP * CAP,
                },
                sort_keys=True,
            )
        )
        return 0
    if not (0 <= args.start_index < args.end_index <= CAP * CAP):
        parser.error(f"expected 0 <= start-index < end-index <= {CAP * CAP}")
    pairs = [
        (left, right)
        for left in range(1, CAP + 1)
        for right in range(1, CAP + 1)
    ][args.start_index : args.end_index]
    proofs = [
        connected_component_maximum(
            left,
            right,
            seconds=args.seconds,
            workers=args.workers,
        )
        for left, right in pairs
    ]
    incomplete = [proof for proof in proofs if proof["status"] != "OPTIMAL"]
    payload = {
        "schema": "hex-bipartite-edge-bound-shard-v1",
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
                "proof_complete": payload["proof_complete"],
                "optimal_components": len(proofs) - len(incomplete),
                "components": len(proofs),
            },
            sort_keys=True,
        )
    )
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
