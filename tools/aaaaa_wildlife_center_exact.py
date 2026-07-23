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
