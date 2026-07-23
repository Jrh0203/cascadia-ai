#!/usr/bin/env python3
"""AAAAA exact model with canonical Fox-A witnesses and local ring tables.

This extends the radius-two fox-neighborhood model by assigning every scored
Fox-A species observation to its lowest-index adjacent token.  The assignment
is deterministic, so it removes witness symmetry.  Foxes assigned to the same
token must be mutually within distance two, and every assigned triple must
have the exact distance pattern of three distinct cells in a six-cell ring.

All constraints are redundant consequences of the exact coordinates and
Fox-A scoring rule.  They expose the coupling between per-species coverage and
the realizable fox-neighborhood tables without changing the legal boards.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from functools import cache
from pathlib import Path
from typing import Any

from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools import aaaaa_wildlife_fox_neighborhood_exact as neighborhood


def _variables_by_name(model: cp_model.CpModel) -> dict[str, cp_model.IntVar]:
    return {
        variable.name: model.get_int_var_from_proto_index(index)
        for index, variable in enumerate(model.proto.variables)
    }


def _pair_name(prefix: str, left: int, right: int) -> str:
    return f"{prefix}_{min(left, right)}_{max(left, right)}"


def _relation_row(coords: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    row = []
    for left in range(len(coords)):
        for right in range(left + 1, len(coords)):
            dq = coords[right][0] - coords[left][0]
            dr = coords[right][1] - coords[left][1]
            distance = max(abs(dq), abs(dr), abs(dq + dr))
            row.append(distance if distance <= 2 else 0)
    return tuple(row)


@cache
def ring_triple_rows() -> tuple[tuple[int, int, int], ...]:
    rows = {
        _relation_row(tuple(cells[index] for index in order))
        for cells in itertools.combinations(base.DIRECTIONS, 3)
        for order in itertools.permutations(range(3))
    }
    return tuple(sorted(rows))


@cache
def conditional_ring_triple_rows() -> tuple[tuple[int, ...], ...]:
    rows = []
    for witnesses in itertools.product(range(2), repeat=3):
        allowed_relations = (
            ring_triple_rows()
            if all(witnesses)
            else itertools.product(range(3), repeat=3)
        )
        rows.extend((*witnesses, *relations) for relations in allowed_relations)
    return tuple(rows)


def add_canonical_fox_witnesses(
    model: cp_model.CpModel,
    variables: base.ExactVariables,
) -> None:
    named = _variables_by_name(model)
    by_species = {
        species: [
            token
            for token, token_species in enumerate(variables.species_by_token)
            if token_species == species
        ]
        for species in range(len(base.SPECIES))
    }
    foxes = by_species[base.SPECIES_CODE["fox"]]
    witnesses: dict[tuple[int, int, int], cp_model.IntVar] = {}

    for fox in foxes:
        for species, species_targets in by_species.items():
            targets = [target for target in species_targets if target != fox]
            selected = []
            earlier_edges: list[cp_model.IntVar] = []
            for target in targets:
                edge = named[_pair_name("adj", fox, target)]
                chosen = model.new_bool_var(f"fox_witness_{fox}_{species}_{target}")
                model.add(chosen <= edge)
                for earlier in earlier_edges:
                    model.add(chosen + earlier <= 1)
                model.add(chosen >= edge - sum(earlier_edges))
                selected.append(chosen)
                earlier_edges.append(edge)
                witnesses[(fox, species, target)] = chosen
            distinct = named[f"fox_distinct_{fox}_{species}"]
            model.add(sum(selected) == distinct)

    for species, targets in by_species.items():
        for target in targets:
            observing = [
                fox for fox in foxes
                if (fox, species, target) in witnesses
            ]
            for left, right in itertools.combinations(observing, 2):
                adjacent = named[_pair_name("adj", left, right)]
                distance_two = named[_pair_name("fox_distance_two", left, right)]
                model.add(
                    witnesses[(left, species, target)]
                    + witnesses[(right, species, target)]
                    <= 1 + adjacent + distance_two
                )
            for triple in itertools.combinations(observing, 3):
                witness_vars = [
                    witnesses[(fox, species, target)]
                    for fox in triple
                ]
                relation_vars = [
                    named[_pair_name("fox_neighborhood_state", left, right)]
                    for left, right in itertools.combinations(triple, 2)
                ]
                model.add_allowed_assignments(
                    [*witness_vars, *relation_vars],
                    conditional_ring_triple_rows(),
                )


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
    model, variables = neighborhood.build_model(
        counts,
        minimum_score,
        maximize=maximize,
        maximum_score=maximum_score,
        enforce_connectivity=enforce_connectivity,
        initial_tokens=initial_tokens,
        fix_initial_tokens=fix_initial_tokens,
    )
    add_canonical_fox_witnesses(model, variables)
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
            raise RuntimeError("fox-witness model emitted a disconnected witness")
    return {
        "counts": list(counts),
        "minimum_score": minimum_score,
        "model_status": solver.status_name(status),
        "objective": objective,
        "score_breakdown": score_breakdown,
        "best_bound": solver.best_objective_bound,
        "wall_seconds": solver.wall_time,
        "branches": solver.num_branches,
        "conflicts": solver.num_conflicts,
        "solver_parameters": {
            "time_limit_seconds": time_limit,
            "workers": workers,
            "random_seed": seed,
            "maximum_score": maximum_score,
            "enforce_connectivity": enforce_connectivity,
            "fox_neighborhood_table_size": min(neighborhood.MAX_TABLE_SIZE, counts[-1]),
            "canonical_fox_witnesses": True,
            "common_witness_ring_triples": True,
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
    result = solve_counts(
        args.counts,
        args.minimum_score,
        args.time_limit,
        args.workers,
        args.seed,
        maximum_score=args.maximum_score,
        enforce_connectivity=not args.disconnected,
        initial_tokens=list(row["tokens"]),
    )
    source = Path(__file__).resolve()
    payload = {
        "schema": "aaaaa-fox-witness-calibration-v1",
        "proof_complete": result["model_status"] == "INFEASIBLE",
        "result": result,
        "ortools_version": ORTOOLS_VERSION,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "neighborhood_source_sha256": hashlib.sha256(
            source.with_name("aaaaa_wildlife_fox_neighborhood_exact.py").read_bytes()
        ).hexdigest(),
        "exact_source_sha256": hashlib.sha256(
            source.with_name("aaaaa_wildlife_exact.py").read_bytes()
        ).hexdigest(),
        "catalog": {
            "path": str(args.catalog),
            "sha256": hashlib.sha256(args.catalog.read_bytes()).hexdigest(),
        },
        "ring_triple_rows": len(ring_triple_rows()),
        "conditional_ring_triple_rows": len(conditional_ring_triple_rows()),
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
