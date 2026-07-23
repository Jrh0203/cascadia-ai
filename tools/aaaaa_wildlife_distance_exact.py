#!/usr/bin/env python3
"""AAAAA exact model with symmetry-free shortest-path connectivity.

The scoring and coordinate formulation is imported from the hash-pinned base
model.  Instead of choosing one of many equivalent parent/depth arborescences,
this variant makes every token depth equal its graph distance from token zero:

    depth(v) = 1 + min(depth(u) for adjacent u)

The equations are feasible exactly when every token is connected to the root.
They add no geometric restriction and remove spanning-tree assignment
symmetry from connected infeasibility proofs.
"""

from __future__ import annotations

import time
from typing import Any

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base


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
        enforce_connectivity=False,
        initial_tokens=initial_tokens,
        fix_initial_tokens=fix_initial_tokens,
    )
    if not enforce_connectivity:
        return model, variables

    adjacency = base.adjacency_variables(model, variables.q, variables.r)
    depth = [
        model.new_int_var(0, base.TOKEN_COUNT - 1, f"shortest_depth_{token}")
        for token in range(base.TOKEN_COUNT)
    ]
    model.add(depth[0] == 0)
    for token in range(1, base.TOKEN_COUNT):
        candidates = []
        for other in range(base.TOKEN_COUNT):
            if other == token:
                continue
            candidate = model.new_int_var(1, base.TOKEN_COUNT, f"depth_via_{token}_{other}")
            edge = base.adjacent(adjacency, token, other)
            model.add(candidate == depth[other] + 1).only_enforce_if(edge)
            model.add(candidate == base.TOKEN_COUNT).only_enforce_if(edge.negated())
            candidates.append(candidate)
        model.add_min_equality(depth[token], candidates)
    return model, variables


def solve_counts(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    time_limit: float,
    workers: int,
    seed: int,
    log_search: bool = False,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
    initial_tokens: list[dict[str, int | str]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    model, variables = build_model(
        counts,
        minimum_score,
        maximize=maximize,
        maximum_score=maximum_score,
        enforce_connectivity=enforce_connectivity,
        initial_tokens=initial_tokens,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log_search
    if maximize:
        status = solver.solve(model, base.Progress(variables.total_score, started))
    else:
        status = solver.solve(model)

    tokens: list[dict[str, int | str]] = []
    objective: int | None = None
    model_score: int | None = None
    score_breakdown: list[int] | None = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        model_score = int(solver.value(variables.total_score))
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
        if maximize and objective != model_score:
            raise RuntimeError(
                f"model witness scored {score_breakdown}, not claimed objective {model_score}"
            )
        if not maximize and objective < minimum_score:
            raise RuntimeError(f"feasibility witness scored {objective}, below {minimum_score}")
        occupied = {(int(row["q"]), int(row["r"])) for row in tokens}
        if enforce_connectivity and len(base.components(occupied)) != 1:
            raise RuntimeError("shortest-path model emitted a disconnected witness")
    return {
        "counts": list(counts),
        "count_relaxation": base.count_relaxation(counts),
        "model_status": solver.status_name(status),
        "objective": objective,
        "model_score": model_score,
        "score_breakdown": score_breakdown,
        "best_bound": solver.best_objective_bound,
        "wall_seconds": solver.wall_time,
        "branches": solver.num_branches,
        "conflicts": solver.num_conflicts,
        "solver_parameters": {
            "time_limit_seconds": time_limit,
            "workers": workers,
            "random_seed": seed,
            "maximize": maximize,
            "maximum_score": maximum_score,
            "enforce_connectivity": enforce_connectivity,
            "connectivity_formulation": "shortest_path_min_equality_v1",
        },
        "tokens": tokens,
    }
