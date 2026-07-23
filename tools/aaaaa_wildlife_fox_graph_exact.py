#!/usr/bin/env python3
"""AAAAA model variant with an explicit realizable fox-adjacency table."""

from __future__ import annotations

import itertools
from functools import cache

from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base


def _rotate(coord: tuple[int, int]) -> tuple[int, int]:
    q, r = coord
    return -r, q + r


def _canonical_shape(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    variants = []
    for reflected in range(2):
        oriented = [(q, -q - r) if reflected else (q, r) for q, r in coords]
        for rotation in range(6):
            if rotation:
                oriented = [_rotate(coord) for coord in oriented]
            for anchor_q, anchor_r in oriented:
                variants.append(tuple(sorted((q - anchor_q, r - anchor_r) for q, r in oriented)))
    return min(variants)


def _graph_code_for_order(coords: tuple[tuple[int, int], ...], order: tuple[int, ...]) -> int:
    code = 0
    bit = 0
    for left in range(len(order)):
        for right in range(left + 1, len(order)):
            one = coords[order[left]]
            other = coords[order[right]]
            if (other[0] - one[0], other[1] - one[1]) in base.DIRECTIONS:
                code |= 1 << bit
            bit += 1
    return code


@cache
def _connected_graph_codes(size: int) -> frozenset[int]:
    shapes = {((0, 0),)}
    for _ in range(2, size + 1):
        next_shapes = set()
        for shape in shapes:
            occupied = set(shape)
            for q, r in shape:
                for dq, dr in base.DIRECTIONS:
                    target = (q + dq, r + dr)
                    if target not in occupied:
                        next_shapes.add(_canonical_shape(occupied | {target}))
        shapes = next_shapes
    return frozenset(
        min(_graph_code_for_order(shape, order) for order in itertools.permutations(range(size)))
        for shape in shapes
    )


def _adjacency_from_mask(size: int, mask: int) -> list[int]:
    adjacency = [0] * size
    bit = 0
    for left in range(size):
        for right in range(left + 1, size):
            if mask & (1 << bit):
                adjacency[left] |= 1 << right
                adjacency[right] |= 1 << left
            bit += 1
    return adjacency


def _component_code(vertices: list[int], adjacency: list[int]) -> int:
    codes = []
    for order in itertools.permutations(vertices):
        code = 0
        bit = 0
        for left in range(len(order)):
            for right in range(left + 1, len(order)):
                if adjacency[order[left]] & (1 << order[right]):
                    code |= 1 << bit
                bit += 1
        codes.append(code)
    return min(codes)


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


@cache
def _all_fox_graph_rows(fox_count: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    edge_count = fox_count * (fox_count - 1) // 2
    rows = []
    for mask in range(1 << edge_count):
        adjacency = _adjacency_from_mask(fox_count, mask)
        if any(
            _component_code(component, adjacency) not in _connected_graph_codes(len(component))
            for component in _components(adjacency)
        ):
            continue
        rows.append(
            (
                tuple((mask >> bit) & 1 for bit in range(edge_count)),
                sum(bool(neighbors) for neighbors in adjacency),
            )
        )
    return tuple(rows)


@cache
def fox_graph_rows(fox_count: int, minimum_nonisolated: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        row
        for row, nonisolated in _all_fox_graph_rows(fox_count)
        if nonisolated >= minimum_nonisolated
    )


def minimum_nonisolated_foxes(counts: tuple[int, int, int, int, int], minimum_score: int) -> int:
    fox_count = counts[base.SPECIES_CODE["fox"]]
    nonfox_upper = sum(base.STANDALONE_SCORES[species][counts[species]] for species in range(4))
    present_nonfox = sum(count > 0 for count in counts[:4])
    return max(0, min(fox_count, minimum_score - nonfox_upper - fox_count * present_nonfox))


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
    foxes = [
        token
        for token, species in enumerate(variables.species_by_token)
        if species == base.SPECIES_CODE["fox"]
    ]
    if len(foxes) < 2:
        return model, variables

    named = {
        variable.name: model.get_bool_var_from_proto_index(index)
        for index, variable in enumerate(model.proto.variables)
        if len(variable.domain) == 2 and variable.domain[0] == 0 and variable.domain[1] == 1
    }
    edges = [named[f"adj_{left}_{right}"] for left, right in itertools.combinations(foxes, 2)]
    minimum_nonisolated = minimum_nonisolated_foxes(counts, minimum_score)
    model.add_allowed_assignments(edges, fox_graph_rows(len(foxes), minimum_nonisolated))
    return model, variables
