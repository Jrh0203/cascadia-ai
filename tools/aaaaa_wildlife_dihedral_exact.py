#!/usr/bin/env python3
"""AAAAA exact model with sound dihedral lex-leader constraints.

The base labeled-token model removes translation and same-species label
symmetry, but still admits rotated and reflected copies.  This wrapper keeps
only representations whose permutation-invariant species moment is minimal
among every dihedral transform for which the same anchor token remains the
lexicographically first token of its species.  Identity is always one such
transform, so every physical board retains at least one representation.
"""

from __future__ import annotations

import time
from typing import Any

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base

# Each transform is ((q coefficient, r coefficient) for q', likewise for r').
# The first six are rotations. The final six are those rotations after q/r
# reflection. All twelve preserve the axial hex lattice.
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


def transformed(
    q: cp_model.LinearExprT,
    r: cp_model.LinearExprT,
    transform: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[cp_model.LinearExprT, cp_model.LinearExprT]:
    (qq, qr), (rq, rr) = transform
    return qq * q + qr * r, rq * q + rr * r


def add_dihedral_lex_leader(
    model: cp_model.CpModel,
    variables: base.ExactVariables,
) -> None:
    anchor_species = variables.species_by_token[0]
    anchor_group = [
        token
        for token, species in enumerate(variables.species_by_token)
        if species == anchor_species
    ]
    width = 2 * base.GLOBAL_RADIUS + 1
    original_key = sum(
        (species + 1) * (11 * variables.q[token] + variables.r[token])
        for token, species in enumerate(variables.species_by_token)
    )

    for transform_index, transform in enumerate(DIHEDRAL_TRANSFORMS[1:], start=1):
        anchor_still_first = []
        for token in anchor_group[1:]:
            tq, tr = transformed(variables.q[token], variables.r[token], transform)
            positive = model.new_bool_var(f"dihedral_{transform_index}_anchor_before_{token}")
            difference = width * tq + tr
            model.add(difference >= 1).only_enforce_if(positive)
            model.add(difference <= 0).only_enforce_if(positive.negated())
            anchor_still_first.append(positive)

        valid = model.new_bool_var(f"dihedral_{transform_index}_anchor_valid")
        if anchor_still_first:
            for positive in anchor_still_first:
                model.add(valid <= positive)
            model.add(valid >= sum(anchor_still_first) - len(anchor_still_first) + 1)
        else:
            model.add(valid == 1)

        transformed_key = sum(
            (species + 1) * (11 * transformed(variables.q[token], variables.r[token], transform)[0]
                             + transformed(variables.q[token], variables.r[token], transform)[1])
            for token, species in enumerate(variables.species_by_token)
        )
        model.add(original_key <= transformed_key).only_enforce_if(valid)


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
    add_dihedral_lex_leader(model, variables)
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
            raise RuntimeError("dihedral model emitted a disconnected witness")

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
            "symmetry_breaking": "conditional_dihedral_species_moment_v1",
        },
        "tokens": tokens,
    }
