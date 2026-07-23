#!/usr/bin/env python3
"""AAAAA exact model with a complete occupied-center radius bound.

Every connected graph on twenty vertices has an occupied vertex whose graph
eccentricity is at most ten.  Hex distance never exceeds distance in the
occupied adjacency graph, so every legal twenty-token board has an occupied
cell within hex distance ten of every token.  This module adds that necessary
condition to the hash-pinned labeled-token model.  It does not assume that a
particular species is central and therefore excludes no legal board.
"""

from __future__ import annotations

import time
from typing import Any

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base

CENTER_RADIUS = base.TOKEN_COUNT // 2


def add_occupied_center_bound(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
) -> None:
    """Require some occupied token to cover the board within radius ten."""
    selected = [model.new_bool_var(f"board_center_{token}") for token in range(base.TOKEN_COUNT)]
    model.add_exactly_one(selected)
    center_q = model.new_int_var(-base.GLOBAL_RADIUS, base.GLOBAL_RADIUS, "board_center_q")
    center_r = model.new_int_var(-base.GLOBAL_RADIUS, base.GLOBAL_RADIUS, "board_center_r")
    for token, choice in enumerate(selected):
        model.add(center_q == q[token]).only_enforce_if(choice)
        model.add(center_r == r[token]).only_enforce_if(choice)

    for token in range(base.TOKEN_COUNT):
        model.add(q[token] - center_q >= -CENTER_RADIUS)
        model.add(q[token] - center_q <= CENTER_RADIUS)
        model.add(r[token] - center_r >= -CENTER_RADIUS)
        model.add(r[token] - center_r <= CENTER_RADIUS)
        model.add(q[token] + r[token] - center_q - center_r >= -CENTER_RADIUS)
        model.add(q[token] + r[token] - center_q - center_r <= CENTER_RADIUS)


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
    if enforce_connectivity:
        add_occupied_center_bound(model, variables.q, variables.r)
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
            raise RuntimeError("occupied-center model emitted a disconnected witness")

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
            "occupied_center_radius": CENTER_RADIUS if enforce_connectivity else None,
        },
        "tokens": tokens,
    }
