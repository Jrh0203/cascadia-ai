#!/usr/bin/env python3
"""Exact achievement-certificate model for every Cascadia wildlife card.

The coordinate model places exactly twenty tokens of fixed species counts on
distinct, connected axial hexes.  Each card encoding exposes scoring objects
whose selected value can never exceed the production score of the represented
board.  Conversely, every production scoring decomposition has a corresponding
selection.  Maximizing the certificate is therefore exact, while infeasibility
at a threshold is a valid proof that no board with those counts reaches it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from tools import all_wildlife_rules as rules
from tools.cbddb_wildlife_exact import (
    GLOBAL_RADIUS,
    ExactVariables,
    _between_on_segment,
    _component_constraints,
    _direction_cases,
    adjacency_variables,
    adjacent,
    species_tokens,
)

SPECIES = rules.SPECIES
SPECIES_CODE = {name: index for index, name in enumerate(SPECIES)}
TOKEN_COUNT = rules.TOKEN_COUNT
COUNT_CAP = rules.COUNT_CAP
DIRECTIONS = rules.DIRECTIONS


def _dihedral_axial(
    coord: tuple[int, int],
    transform: int,
) -> tuple[int, int]:
    q, r = coord
    rotations = (
        (q, r),
        (-r, q + r),
        (-q - r, q),
        (-q, -r),
        (r, -q - r),
        (q + r, -q),
    )
    rotated_q, rotated_r = rotations[transform % 6]
    if transform >= 6:
        return rotated_r, rotated_q
    return rotated_q, rotated_r


def _in_centroid_wedge(coord: tuple[int, int]) -> bool:
    q, r = coord
    return q >= r >= 0


def _anchor_centroid_initial_coordinates(
    initial_tokens: list[dict[str, Any]],
    species_by_token: list[int],
) -> list[tuple[int, int]]:
    by_species = {
        species: sorted(
            (
                (int(row["q"]), int(row["r"]))
                for row in initial_tokens
                if str(row["wildlife"]) == SPECIES[species]
            ),
        )
        for species in range(len(SPECIES))
    }
    expected = [species_by_token.count(species) for species in range(len(SPECIES))]
    if any(len(by_species[species]) != expected[species] for species in range(len(SPECIES))):
        raise ValueError("initial token counts do not match model counts")
    fox_species = SPECIES_CODE["fox"]
    if not by_species[fox_species]:
        raise ValueError("anchor-centroid symmetry requires at least one fox")

    candidates = []
    for anchor in by_species[fox_species]:
        for transform in range(12):
            transformed = {
                species: [
                    _dihedral_axial(
                        (coord[0] - anchor[0], coord[1] - anchor[1]),
                        transform,
                    )
                    for coord in coords
                ]
                for species, coords in by_species.items()
            }
            centroid = (
                sum(q for coords in transformed.values() for q, _ in coords),
                sum(r for coords in transformed.values() for _, r in coords),
            )
            if not _in_centroid_wedge(centroid):
                continue
            ordered_by_species = {
                species: sorted(coords)
                for species, coords in transformed.items()
            }
            ordered_by_species[fox_species] = [
                (0, 0),
                *sorted(coord for coord in transformed[fox_species] if coord != (0, 0)),
            ]
            offsets = [0] * len(SPECIES)
            ordered = []
            for species in species_by_token:
                ordered.append(ordered_by_species[species][offsets[species]])
                offsets[species] += 1
            if all(
                max(abs(q), abs(r), abs(q + r)) <= GLOBAL_RADIUS
                for q, r in ordered
            ):
                candidates.append(ordered)
    if not candidates:
        raise AssertionError("no anchor-centroid canonical image for initial board")
    return min(candidates)


def _component_score(
    model: cp_model.CpModel,
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    tokens: list[int],
    sizes: dict[int, int],
    name: str,
    upper: int,
    *,
    maximum_degree: int | None = None,
) -> cp_model.IntVar:
    components: list[tuple[tuple[int, ...], int, cp_model.IntVar]] = []
    for size, score in sizes.items():
        for members in itertools.combinations(tokens, size):
            selected = model.new_bool_var(f"{name}_component_{'_'.join(map(str, members))}")
            _component_constraints(
                model,
                adjacency,
                members,
                tokens,
                selected,
                maximum_degree=maximum_degree,
            )
            components.append((members, score, selected))
    for token in tokens:
        model.add(sum(selected for members, _, selected in components if token in members) <= 1)
    result = model.new_int_var(0, upper, f"{name}_score")
    model.add(result == sum(score * selected for _, score, selected in components))
    return result


def _bear_score(
    model: cp_model.CpModel,
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    bears: list[int],
    variant: str,
) -> cp_model.IntVar:
    if variant == "A":
        components: list[tuple[tuple[int, ...], cp_model.IntVar]] = []
        for members in itertools.combinations(bears, 2):
            selected = model.new_bool_var(f"bear_a_pair_{'_'.join(map(str, members))}")
            _component_constraints(model, adjacency, members, bears, selected)
            components.append((members, selected))
        for bear in bears:
            model.add(sum(selected for members, selected in components if bear in members) <= 1)
        pair_count = model.new_int_var(0, len(bears) // 2, "bear_a_pair_count")
        result = model.new_int_var(0, 19, "bear_score")
        model.add(pair_count == sum(selected for _, selected in components))
        model.add_allowed_assignments(
            [pair_count, result],
            [[count, (0, 4, 11, 19, 27)[min(count, 4)]] for count in range(len(bears) // 2 + 1)],
        )
        return result
    if variant == "B":
        return _component_score(
            model,
            adjacency,
            bears,
            {3: 10},
            "bear_b",
            rules._standalone_bear(len(bears), variant),
        )
    if variant == "C":
        components: list[tuple[tuple[int, ...], int, cp_model.IntVar]] = []
        for size, score in ((1, 2), (2, 5), (3, 8)):
            for members in itertools.combinations(bears, size):
                selected = model.new_bool_var(f"bear_c_component_{'_'.join(map(str, members))}")
                _component_constraints(model, adjacency, members, bears, selected)
                components.append((members, score, selected))
        for bear in bears:
            model.add(sum(selected for members, _, selected in components if bear in members) <= 1)
        bonus = model.new_bool_var("bear_c_set_bonus")
        for size in (1, 2, 3):
            model.add(
                bonus
                <= sum(selected for members, _, selected in components if len(members) == size)
            )
        result = model.new_int_var(0, rules._standalone_bear(len(bears), variant), "bear_score")
        model.add(result == sum(score * selected for _, score, selected in components) + 3 * bonus)
        return result
    if variant == "D":
        return _component_score(
            model,
            adjacency,
            bears,
            {2: 5, 3: 8, 4: 13},
            "bear_d",
            rules._standalone_bear(len(bears), variant),
        )
    raise AssertionError(variant)


def _packing_score(
    model: cp_model.CpModel,
    groups: list[tuple[tuple[int, ...], int, cp_model.IntVar]],
    tokens: list[int],
    upper: int,
    name: str,
) -> cp_model.IntVar:
    for token in tokens:
        model.add(sum(selected for members, _, selected in groups if token in members) <= 1)
    result = model.new_int_var(0, upper, f"{name}_score")
    model.add(result == sum(score * selected for _, score, selected in groups))
    return result


def _elk_a_score(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    elk: list[int],
) -> cp_model.IntVar:
    groups = [((token,), 2, model.new_bool_var(f"elk_a_single_{token}")) for token in elk]
    for length, score in ((2, 5), (3, 9), (4, 13)):
        for ordered in itertools.combinations(elk, length):
            for direction_index, (dq, dr) in enumerate(((1, 0), (1, -1), (0, 1))):
                selected = model.new_bool_var(
                    f"elk_a_line_{length}_{'_'.join(map(str, ordered))}_{direction_index}"
                )
                for step in range(1, length):
                    model.add(q[ordered[step]] == q[ordered[0]] + step * dq).only_enforce_if(
                        selected
                    )
                    model.add(r[ordered[step]] == r[ordered[0]] + step * dr).only_enforce_if(
                        selected
                    )
                groups.append((ordered, score, selected))
    return _packing_score(model, groups, elk, rules._standalone_elk(len(elk), "A"), "elk")


def _elk_b_score(
    model: cp_model.CpModel,
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    elk: list[int],
) -> cp_model.IntVar:
    groups = [((token,), 2, model.new_bool_var(f"elk_b_single_{token}")) for token in elk]
    for members in itertools.combinations(elk, 2):
        selected = model.new_bool_var(f"elk_b_pair_{'_'.join(map(str, members))}")
        model.add(selected <= adjacent(adjacency, *members))
        groups.append((members, 5, selected))
    for members in itertools.combinations(elk, 3):
        selected = model.new_bool_var(f"elk_b_triangle_{'_'.join(map(str, members))}")
        for left, right in itertools.combinations(members, 2):
            model.add(selected <= adjacent(adjacency, left, right))
        groups.append((members, 9, selected))
    for members in itertools.combinations(elk, 4):
        for fourth in members:
            triangle = tuple(token for token in members if token != fourth)
            selected = model.new_bool_var(
                f"elk_b_rhombus_{'_'.join(map(str, triangle))}_plus_{fourth}"
            )
            for left, right in itertools.combinations(triangle, 2):
                model.add(selected <= adjacent(adjacency, left, right))
            model.add(
                sum(adjacent(adjacency, fourth, member) for member in triangle) >= 2 * selected
            )
            groups.append((members, 13, selected))
    return _packing_score(model, groups, elk, rules._standalone_elk(len(elk), "B"), "elk")


def _center_adjacency(
    model: cp_model.CpModel,
    center_q: cp_model.IntVar,
    center_r: cp_model.IntVar,
    q: cp_model.IntVar,
    r: cp_model.IntVar,
    name: str,
) -> cp_model.IntVar:
    delta_q = model.new_int_var(-2 * GLOBAL_RADIUS - 1, 2 * GLOBAL_RADIUS + 1, f"{name}_dq")
    delta_r = model.new_int_var(-2 * GLOBAL_RADIUS - 1, 2 * GLOBAL_RADIUS + 1, f"{name}_dr")
    model.add(delta_q == q - center_q)
    model.add(delta_r == r - center_r)
    result = model.new_bool_var(name)
    model.add_allowed_assignments([delta_q, delta_r], DIRECTIONS).only_enforce_if(result)
    model.add_forbidden_assignments([delta_q, delta_r], DIRECTIONS).only_enforce_if(
        result.negated()
    )
    return result


def _elk_d_score(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    elk: list[int],
) -> cp_model.IntVar:
    if not elk:
        return model.new_constant(0)
    memberships: list[list[cp_model.IntVar]] = []
    stage_scores = []
    active = []
    for stage in range(len(elk)):
        stage_active = model.new_bool_var(f"elk_d_stage_{stage}_active")
        active.append(stage_active)
        center_q = model.new_int_var(
            -GLOBAL_RADIUS - 1, GLOBAL_RADIUS + 1, f"elk_d_stage_{stage}_center_q"
        )
        center_r = model.new_int_var(
            -GLOBAL_RADIUS - 1, GLOBAL_RADIUS + 1, f"elk_d_stage_{stage}_center_r"
        )
        centered = [
            _center_adjacency(
                model,
                center_q,
                center_r,
                q[token],
                r[token],
                f"elk_d_stage_{stage}_adjacent_{token}",
            )
            for token in elk
        ]
        members = [model.new_bool_var(f"elk_d_stage_{stage}_member_{token}") for token in elk]
        memberships.append(members)
        for local, member in enumerate(members):
            model.add(member <= stage_active)
            model.add(member <= centered[local])
            earlier = sum(memberships[prior][local] for prior in range(stage))
            model.add(member >= stage_active + centered[local] - earlier - 1)
        member_count = model.new_int_var(0, len(elk), f"elk_d_stage_{stage}_count")
        stage_score = model.new_int_var(0, 21, f"elk_d_stage_{stage}_score")
        model.add(member_count == sum(members))
        model.add_allowed_assignments(
            [stage_active, member_count, stage_score],
            [[0, 0, 0]]
            + [[1, count, (0, 2, 5, 8, 12, 16, 21)[count]] for count in range(1, len(elk) + 1)],
        )
        stage_scores.append(stage_score)
        if stage:
            model.add(active[stage - 1] >= stage_active)
    for local in range(len(elk)):
        model.add_exactly_one(memberships[stage][local] for stage in range(len(elk)))
    result = model.new_int_var(0, rules._standalone_elk(len(elk), "D"), "elk_score")
    model.add(result == sum(stage_scores))
    return result


def _elk_score(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    elk: list[int],
    variant: str,
) -> cp_model.IntVar:
    if variant == "A":
        return _elk_a_score(model, q, r, elk)
    if variant == "B":
        return _elk_b_score(model, adjacency, elk)
    if variant == "C":
        return _component_score(
            model,
            adjacency,
            elk,
            {1: 2, 2: 4, 3: 7, 4: 10, 5: 14, 6: 18},
            "elk_c",
            rules._standalone_elk(len(elk), variant),
        )
    if variant == "D":
        return _elk_d_score(model, q, r, elk)
    raise AssertionError(variant)


def _salmon_score(
    model: cp_model.CpModel,
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    salmon: list[int],
    all_tokens: list[int],
    variant: str,
) -> cp_model.IntVar:
    tables = {
        "A": {1: 2, 2: 5, 3: 8, 4: 12, 5: 16, 6: 20},
        "B": {1: 2, 2: 4, 3: 9, 4: 11, 5: 17, 6: 17},
        "C": {3: 10, 4: 12, 5: 15, 6: 15},
    }
    if variant in tables:
        return _component_score(
            model,
            adjacency,
            salmon,
            tables[variant],
            f"salmon_{variant.lower()}",
            rules._standalone_salmon(len(salmon), variant, TOKEN_COUNT - len(salmon)),
            maximum_degree=2,
        )
    if variant == "D":
        non_salmon = [token for token in all_tokens if token not in salmon]
        components: list[tuple[tuple[int, ...], cp_model.IntVar, list[cp_model.IntVar]]] = []
        for size in range(3, len(salmon) + 1):
            for members in itertools.combinations(salmon, size):
                selected = model.new_bool_var(f"salmon_d_component_{'_'.join(map(str, members))}")
                _component_constraints(
                    model,
                    adjacency,
                    members,
                    salmon,
                    selected,
                    maximum_degree=2,
                )
                claims = []
                for target in non_salmon:
                    claim = model.new_bool_var(
                        f"salmon_d_component_{'_'.join(map(str, members))}_neighbor_{target}"
                    )
                    model.add(claim <= selected)
                    model.add(
                        claim <= sum(adjacent(adjacency, member, target) for member in members)
                    )
                    claims.append(claim)
                components.append((members, selected, claims))
        for token in salmon:
            model.add(sum(selected for members, selected, _ in components if token in members) <= 1)
        upper = rules._standalone_salmon(len(salmon), variant, TOKEN_COUNT - len(salmon))
        result = model.new_int_var(0, upper, "salmon_score")
        model.add(
            result
            == sum(
                len(members) * selected + sum(claims) for members, selected, claims in components
            )
        )
        return result
    raise AssertionError(variant)


def _hawk_lines(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    by_species: dict[int, list[int]],
) -> list[
    tuple[
        int,
        int,
        cp_model.IntVar,
        dict[int, cp_model.IntVar],
    ]
]:
    hawks = by_species[SPECIES_CODE["hawk"]]
    lines = []
    for left, right in itertools.combinations(hawks, 2):
        prefix = f"hawk_{left}_{right}"
        direction_data = _direction_cases(model, q, r, left, right, prefix)
        selected = model.new_bool_var(f"{prefix}_visible")
        model.add(selected <= sum(case for case, _, _, _ in direction_data))
        between = {
            target: _between_on_segment(model, q, r, left, target, direction_data, prefix)
            for target in range(TOKEN_COUNT)
            if target not in (left, right)
        }
        for blocker in hawks:
            if blocker not in (left, right):
                model.add(selected + between[blocker] <= 1)
        lines.append((left, right, selected, between))
    return lines


def _hawk_score(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    by_species: dict[int, list[int]],
    variant: str,
) -> cp_model.IntVar:
    hawks = by_species[SPECIES_CODE["hawk"]]
    if variant == "A":
        isolated = []
        for hawk in hawks:
            selected = model.new_bool_var(f"hawk_a_isolated_{hawk}")
            for other in hawks:
                if other != hawk:
                    model.add(selected + adjacent(adjacency, hawk, other) <= 1)
            isolated.append(selected)
        count = model.new_int_var(0, len(hawks), "hawk_a_count")
        result = model.new_int_var(0, (0, 2, 5, 8, 11, 14, 18)[len(hawks)], "hawk_score")
        model.add(count == sum(isolated))
        model.add_allowed_assignments(
            [count, result],
            [[value, (0, 2, 5, 8, 11, 14, 18)[value]] for value in range(len(hawks) + 1)],
        )
        return result

    lines = _hawk_lines(model, q, r, by_species)
    if variant == "B":
        qualifying = []
        for hawk in hawks:
            selected = model.new_bool_var(f"hawk_b_qualifying_{hawk}")
            for other in hawks:
                if other != hawk:
                    model.add(selected + adjacent(adjacency, hawk, other) <= 1)
            model.add(
                selected <= sum(line for left, right, line, _ in lines if hawk in (left, right))
            )
            qualifying.append(selected)
        count = model.new_int_var(0, len(hawks), "hawk_b_count")
        result = model.new_int_var(0, (0, 0, 5, 9, 12, 16, 20)[len(hawks)], "hawk_score")
        model.add(count == sum(qualifying))
        model.add_allowed_assignments(
            [count, result],
            [[value, (0, 0, 5, 9, 12, 16, 20)[value]] for value in range(len(hawks) + 1)],
        )
        return result
    if variant == "C":
        result = model.new_int_var(0, 3 * len(hawks) * (len(hawks) - 1) // 2, "hawk_score")
        model.add(result == 3 * sum(line for _, _, line, _ in lines))
        return result
    if variant == "D":
        weighted = []
        for left, right, selected, between in lines:
            claims = []
            for species in range(len(SPECIES)):
                if species == SPECIES_CODE["hawk"]:
                    continue
                claim = model.new_bool_var(f"hawk_{left}_{right}_distinct_{species}")
                model.add(claim <= selected)
                model.add(claim <= sum(between[target] for target in by_species[species]))
                claims.append(claim)
            distinct = model.new_int_var(0, len(claims), f"hawk_{left}_{right}_distinct")
            pair_score = model.new_int_var(0, 9, f"hawk_{left}_{right}_score")
            model.add(distinct == sum(claims))
            model.add_allowed_assignments(
                [selected, distinct, pair_score],
                [[0, 0, 0]]
                + [[1, count, (0, 4, 7, 9)[min(count, 3)]] for count in range(len(claims) + 1)],
            )
            weighted.append((left, right, selected, pair_score))
        for hawk in hawks:
            model.add(
                sum(selected for left, right, selected, _ in weighted if hawk in (left, right)) <= 1
            )
        result = model.new_int_var(0, (len(hawks) // 2) * 9, "hawk_score")
        model.add(result == sum(score for _, _, _, score in weighted))
        return result
    raise AssertionError(variant)


def _seen_from_either(
    model: cp_model.CpModel,
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    left: int,
    right: int,
    target: int,
    name: str,
) -> cp_model.IntVar:
    result = model.new_bool_var(name)
    model.add(result <= adjacent(adjacency, left, target) + adjacent(adjacency, right, target))
    return result


def _fox_score(
    model: cp_model.CpModel,
    adjacency: dict[tuple[int, int], cp_model.IntVar],
    by_species: dict[int, list[int]],
    variant: str,
) -> cp_model.IntVar:
    foxes = by_species[SPECIES_CODE["fox"]]
    if variant == "A":
        claims = []
        for fox in foxes:
            for species in range(len(SPECIES)):
                claim = model.new_bool_var(f"fox_a_{fox}_distinct_{species}")
                targets = [target for target in by_species[species] if target != fox]
                model.add(claim <= sum(adjacent(adjacency, fox, target) for target in targets))
                claims.append(claim)
        result = model.new_int_var(0, len(foxes) * len(SPECIES), "fox_score")
        model.add(result == sum(claims))
        return result
    if variant == "B":
        fox_scores = []
        for fox in foxes:
            doubled = []
            for species in range(len(SPECIES) - 1):
                pairs = []
                for left, right in itertools.combinations(by_species[species], 2):
                    seen = model.new_bool_var(f"fox_b_{fox}_species_{species}_pair_{left}_{right}")
                    model.add(seen <= adjacent(adjacency, fox, left))
                    model.add(seen <= adjacent(adjacency, fox, right))
                    pairs.append(seen)
                qualifies = model.new_bool_var(f"fox_b_{fox}_doubled_{species}")
                model.add(qualifies <= sum(pairs))
                doubled.append(qualifies)
            count = model.new_int_var(0, len(doubled), f"fox_b_{fox}_count")
            score = model.new_int_var(0, 7, f"fox_b_{fox}_score")
            model.add(count == sum(doubled))
            model.add_allowed_assignments(
                [count, score],
                [[value, (0, 3, 5, 7)[min(value, 3)]] for value in range(len(doubled) + 1)],
            )
            fox_scores.append(score)
        result = model.new_int_var(0, len(foxes) * 7, "fox_score")
        model.add(result == sum(fox_scores))
        return result
    if variant == "C":
        claims = []
        for fox in foxes:
            choices = []
            for species in range(len(SPECIES) - 1):
                choice = model.new_bool_var(f"fox_c_{fox}_species_{species}")
                choices.append(choice)
                for target in by_species[species]:
                    claim = model.new_bool_var(f"fox_c_{fox}_{species}_target_{target}")
                    model.add(claim <= choice)
                    model.add(claim <= adjacent(adjacency, fox, target))
                    claims.append(claim)
            model.add_at_most_one(choices)
        result = model.new_int_var(0, len(foxes) * 6, "fox_score")
        model.add(result == sum(claims))
        return result
    if variant == "D":
        edges = []
        for left, right in itertools.combinations(foxes, 2):
            selected = model.new_bool_var(f"fox_d_pair_{left}_{right}")
            model.add(selected <= adjacent(adjacency, left, right))
            doubled = []
            for species in range(len(SPECIES) - 1):
                target_seen = {
                    target: _seen_from_either(
                        model,
                        adjacency,
                        left,
                        right,
                        target,
                        f"fox_d_{left}_{right}_species_{species}_target_{target}",
                    )
                    for target in by_species[species]
                }
                pairs = []
                for one, two in itertools.combinations(by_species[species], 2):
                    pair = model.new_bool_var(
                        f"fox_d_{left}_{right}_species_{species}_pair_{one}_{two}"
                    )
                    model.add(pair <= target_seen[one])
                    model.add(pair <= target_seen[two])
                    pairs.append(pair)
                qualifies = model.new_bool_var(f"fox_d_{left}_{right}_doubled_{species}")
                model.add(qualifies <= selected)
                model.add(qualifies <= sum(pairs))
                doubled.append(qualifies)
            count = model.new_int_var(0, len(doubled), f"fox_d_{left}_{right}_count")
            score = model.new_int_var(0, 11, f"fox_d_{left}_{right}_score")
            model.add(count == sum(doubled))
            model.add_allowed_assignments(
                [selected, count, score],
                [[0, 0, 0]]
                + [[1, value, (0, 5, 7, 9, 11)[value]] for value in range(len(doubled) + 1)],
            )
            edges.append((left, right, selected, score))
        for fox in foxes:
            model.add(
                sum(selected for left, right, selected, _ in edges if fox in (left, right)) <= 1
            )
        result = model.new_int_var(0, (len(foxes) // 2) * 11, "fox_score")
        model.add(result == sum(score for _, _, _, score in edges))
        return result
    raise AssertionError(variant)


def build_model(
    ruleset: str,
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
    initial_tokens: list[dict[str, Any]] | None = None,
    fix_initial_tokens: bool = False,
    break_anchor_centroid_symmetry: bool = False,
) -> tuple[cp_model.CpModel, ExactVariables]:
    cards = rules.parse_ruleset(ruleset)
    if counts not in rules.count_vectors():
        raise ValueError(f"invalid counts: {counts}")
    upper = rules.count_upper(counts, ruleset)
    if maximum_score is not None:
        upper = min(upper, maximum_score)
    if minimum_score > upper:
        raise ValueError(f"score interval is empty: [{minimum_score}, {upper}]")

    model = cp_model.CpModel()
    species_by_token = species_tokens(counts)
    q = [model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"q_{i}") for i in range(TOKEN_COUNT)]
    r = [model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"r_{i}") for i in range(TOKEN_COUNT)]
    width = 2 * GLOBAL_RADIUS + 1
    coordinate_id = [model.new_int_var(0, width**2 - 1, f"coord_{i}") for i in range(TOKEN_COUNT)]
    for token in range(TOKEN_COUNT):
        diagonal = model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"diag_{token}")
        model.add(diagonal == q[token] + r[token])
        model.add(
            coordinate_id[token] == (q[token] + GLOBAL_RADIUS) * width + r[token] + GLOBAL_RADIUS
        )
    model.add_all_different(coordinate_id)
    model.add(q[0] == 0)
    model.add(r[0] == 0)

    by_species = {
        species: [index for index, value in enumerate(species_by_token) if value == species]
        for species in range(len(SPECIES))
    }
    use_anchor_centroid_symmetry = (
        break_anchor_centroid_symmetry
        and not fix_initial_tokens
        and bool(by_species[SPECIES_CODE["fox"]])
    )
    for species, indices in by_species.items():
        if use_anchor_centroid_symmetry and species == SPECIES_CODE["fox"]:
            indices = indices[1:]
        for left, right in itertools.pairwise(indices):
            model.add(coordinate_id[left] < coordinate_id[right])
    if use_anchor_centroid_symmetry:
        model.add(sum(q) >= sum(r))
        model.add(sum(r) >= 0)

    if initial_tokens is not None and use_anchor_centroid_symmetry:
        initial_coordinates = _anchor_centroid_initial_coordinates(
            initial_tokens,
            species_by_token,
        )
        for token, (initial_q, initial_r) in enumerate(initial_coordinates):
            model.add_hint(q[token], initial_q)
            model.add_hint(r[token], initial_r)
        initial_tokens = None
    if initial_tokens is not None:
        rows_by_species = {
            species: sorted(
                (
                    (int(row["q"]), int(row["r"]))
                    for row in initial_tokens
                    if str(row["wildlife"]) == SPECIES[species]
                ),
                key=lambda coord: coord[0] * width + coord[1],
            )
            for species in range(len(SPECIES))
        }
        if any(len(rows_by_species[species]) != counts[species] for species in range(len(SPECIES))):
            raise ValueError("initial token counts do not match model counts")
        anchor_species = species_by_token[0]
        anchor_q, anchor_r = rows_by_species[anchor_species][0]
        translated = {
            species: [(coord_q - anchor_q, coord_r - anchor_r) for coord_q, coord_r in rows]
            for species, rows in rows_by_species.items()
        }
        for token, species in enumerate(species_by_token):
            offset = sum(value == species for value in species_by_token[:token])
            initial_q, initial_r = translated[species][offset]
            if fix_initial_tokens:
                model.add(q[token] == initial_q)
                model.add(r[token] == initial_r)
            else:
                model.add_hint(q[token], initial_q)
                model.add_hint(r[token], initial_r)

    adjacency = adjacency_variables(model, q, r)
    if enforce_connectivity:
        depth = [
            model.new_int_var(0, TOKEN_COUNT - 1, f"depth_{token}") for token in range(TOKEN_COUNT)
        ]
        model.add(depth[0] == 0)
        for child in range(1, TOKEN_COUNT):
            parents = []
            for parent in range(TOKEN_COUNT):
                if parent == child:
                    continue
                selected = model.new_bool_var(f"parent_{child}_{parent}")
                model.add(selected <= adjacent(adjacency, child, parent))
                model.add(depth[child] > depth[parent]).only_enforce_if(selected)
                parents.append(selected)
            model.add_exactly_one(parents)

    scores = [
        _bear_score(model, adjacency, by_species[SPECIES_CODE["bear"]], cards[0]),
        _elk_score(
            model,
            q,
            r,
            adjacency,
            by_species[SPECIES_CODE["elk"]],
            cards[1],
        ),
        _salmon_score(
            model,
            adjacency,
            by_species[SPECIES_CODE["salmon"]],
            list(range(TOKEN_COUNT)),
            cards[2],
        ),
        _hawk_score(model, q, r, adjacency, by_species, cards[3]),
        _fox_score(model, adjacency, by_species, cards[4]),
    ]
    total_score = model.new_int_var(minimum_score, upper, "total_score")
    model.add(total_score == sum(scores))
    if maximize:
        model.maximize(total_score)
    return model, ExactVariables(q, r, total_score, species_by_token)


@dataclass
class SolveResult:
    status: str
    objective: int | None
    best_bound: int | None
    elapsed_seconds: float
    branches: int
    conflicts: int
    tokens: list[dict[str, int | str]] | None
    score_breakdown: tuple[int, int, int, int, int] | None


def solve_counts(
    ruleset: str,
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    *,
    time_limit_seconds: float,
    workers: int,
    initial_tokens: list[dict[str, Any]] | None = None,
    fix_initial_tokens: bool = False,
    enforce_connectivity: bool = True,
    maximize: bool = True,
    break_anchor_centroid_symmetry: bool = False,
) -> SolveResult:
    started = time.monotonic()
    model, variables = build_model(
        ruleset,
        counts,
        minimum_score,
        initial_tokens=initial_tokens,
        fix_initial_tokens=fix_initial_tokens,
        enforce_connectivity=enforce_connectivity,
        maximize=maximize,
        break_anchor_centroid_symmetry=break_anchor_centroid_symmetry,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = 20260723
    status_code = solver.solve(model)
    status = solver.status_name(status_code)
    objective = None
    best_bound = None
    tokens = None
    breakdown = None
    if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        objective = solver.value(variables.total_score)
        tokens = rules.normalized_tokens(
            [
                {
                    "q": solver.value(variables.q[token]),
                    "r": solver.value(variables.r[token]),
                    "wildlife": SPECIES[variables.species_by_token[token]],
                }
                for token in range(TOKEN_COUNT)
            ]
        )
        breakdown = rules.score_tokens(tokens, ruleset)
        if sum(breakdown) < objective:
            raise AssertionError(f"unsound certificate: model={objective}, independent={breakdown}")
    if maximize and status_code != cp_model.MODEL_INVALID:
        best_bound = int(solver.best_objective_bound)
    return SolveResult(
        status=status,
        objective=objective,
        best_bound=best_bound,
        elapsed_seconds=time.monotonic() - started,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        tokens=tokens,
        score_breakdown=breakdown,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ruleset")
    parser.add_argument("counts", help="comma-separated bear,elk,salmon,hawk,fox")
    parser.add_argument("--minimum-score", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--fix-initial", action="store_true")
    parser.add_argument("--no-connectivity", action="store_true")
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.counts.split(","))
    if len(counts) != len(SPECIES):
        parser.error("counts must have five values")
    initial_tokens = None
    if args.initial:
        payload = json.loads(args.initial.read_text())
        initial_tokens = payload.get("tokens", payload)
    result = solve_counts(
        args.ruleset,
        counts,  # type: ignore[arg-type]
        args.minimum_score,
        time_limit_seconds=args.seconds,
        workers=args.workers,
        initial_tokens=initial_tokens,
        fix_initial_tokens=args.fix_initial,
        enforce_connectivity=not args.no_connectivity,
    )
    print(json.dumps(result.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
