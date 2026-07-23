#!/usr/bin/env python3
"""Threshold-specific exact relaxation for the hard AAAAA catalog tail.

The full coordinate model spends most of its time rediscovering scoring
structures that are already forced by a high target.  This model keeps the
coordinates of all twenty tokens, exact non-overlap, every scoring Bear-A
pair, Elk-A line, Salmon-A component, and every Fox-A observation.  It drops
constraints that can only lower the score: whole-board connectivity, bear
pair isolation, separation between salmon components, and hawk isolation.

Consequently every legal board meeting ``target`` is feasible here, but the
converse need not hold.  An INFEASIBLE result is therefore a sound upper-bound
certificate; FEASIBLE or UNKNOWN proves nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools.aaaaa_wildlife_motif_certificate import (
    DIHEDRAL_TRANSFORMS,
    DIRECTIONS,
    LINE_DIRECTIONS,
    free_polyhexes,
    integer_partitions,
)

Coord = tuple[int, int]


@cache
def elk_partitions(count: int) -> tuple[tuple[int, ...], ...]:
    """All token-covering Elk-A line partitions for ``count`` elk."""
    return tuple(integer_partitions(count, 4))


@cache
def salmon_partitions(count: int) -> tuple[tuple[int, ...], ...]:
    """All possible multisets of scored Salmon-A component sizes.

    Tokens not included in the partition represent salmon in invalid
    components.  The empty partition is therefore required.
    """
    result: set[tuple[int, ...]] = {()}
    for used in range(1, count + 1):
        result.update(integer_partitions(used, count))
    return tuple(sorted(result, key=lambda row: (sum(row), row)))


@cache
def salmon_component_offsets(size: int) -> tuple[tuple[int, ...], ...]:
    """Every oriented valid Salmon-A component, anchored canonically."""
    rows: set[tuple[int, ...]] = set()
    for free_shape in free_polyhexes(size):
        shape = set(free_shape)
        if any(
            sum((other[0] - cell[0], other[1] - cell[1]) in DIRECTIONS for other in shape)
            > 2
            for cell in shape
        ):
            continue
        for (qq, qr), (rq, rr) in DIHEDRAL_TRANSFORMS:
            image = {(qq * q + qr * r, rq * q + rr * r) for q, r in shape}
            anchor_q, anchor_r = min(image)
            offsets = sorted((q - anchor_q, r - anchor_r) for q, r in image)
            flattened = tuple(value for coord in offsets[1:] for value in coord)
            rows.add(flattened)
    return tuple(sorted(rows))


def _conditional_offsets(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
    members: list[int],
    allowed: Iterable[tuple[int, ...]],
    selected: cp_model.IntVar,
    name: str,
) -> None:
    if len(members) <= 1:
        return
    differences: list[cp_model.IntVar] = []
    anchor = members[0]
    for offset, token in enumerate(members[1:], 1):
        dq = model.new_int_var(
            -2 * base.GLOBAL_RADIUS, 2 * base.GLOBAL_RADIUS, f"{name}_dq_{offset}"
        )
        dr = model.new_int_var(
            -2 * base.GLOBAL_RADIUS, 2 * base.GLOBAL_RADIUS, f"{name}_dr_{offset}"
        )
        model.add(dq == q[token] - q[anchor])
        model.add(dr == r[token] - r[anchor])
        differences.extend((dq, dr))
    model.add_allowed_assignments(differences, list(allowed)).only_enforce_if(selected)


def _line_offsets(length: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(value for step in range(1, length) for value in (step * dq, step * dr))
        for dq, dr in LINE_DIRECTIONS
    )


def _ordered_line(group: set[Coord]) -> list[Coord]:
    if len(group) == 1:
        return list(group)
    for start in group:
        for dq, dr in LINE_DIRECTIONS:
            row = [(start[0] + step * dq, start[1] + step * dr) for step in range(len(group))]
            if set(row) == group:
                return row
    raise ValueError(f"not a straight elk line: {sorted(group)}")


def _optimal_elk_groups(coords: list[Coord]) -> list[list[Coord]]:
    """Recover one optimal token-covering Elk-A packing for witness tests."""
    index = {coord: position for position, coord in enumerate(coords)}
    groups: list[tuple[int, int, list[Coord]]] = []
    for coord in coords:
        groups.append((1 << index[coord], 2, [coord]))
    for length, score in ((2, 5), (3, 9), (4, 13)):
        seen: set[frozenset[Coord]] = set()
        for start in coords:
            for dq, dr in LINE_DIRECTIONS:
                row = [(start[0] + step * dq, start[1] + step * dr) for step in range(length)]
                shape = frozenset(row)
                if shape in seen or not shape <= index.keys():
                    continue
                seen.add(shape)
                groups.append(
                    (
                        sum(1 << index[cell] for cell in shape),
                        score,
                        _ordered_line(set(shape)),
                    )
                )
    best: list[tuple[int, list[list[Coord]]]] = [(-1, []) for _ in range(1 << len(coords))]
    best[0] = (0, [])
    for state in range(1, len(best)):
        first = (state & -state).bit_length() - 1
        for mask, score, row in groups:
            if mask & (1 << first) and mask & state == mask:
                previous_score, previous = best[state & ~mask]
                if previous_score >= 0 and previous_score + score > best[state][0]:
                    best[state] = (previous_score + score, [*previous, row])
    if coords and best[-1][0] < 0:
        raise RuntimeError("failed to reconstruct Elk-A packing")
    return sorted(best[-1][1], key=lambda row: -len(row)) if coords else []


def relabel_fixed_witness(
    counts: tuple[int, int, int, int, int],
    tokens: list[dict[str, int | str]],
) -> list[tuple[int, int]]:
    """Relabel identical species so the fixed witness matches motif slots."""
    by_name: dict[str, list[Coord]] = {species: [] for species in base.SPECIES}
    for row in tokens:
        by_name[str(row["wildlife"])].append((int(row["q"]), int(row["r"])))
    if tuple(len(by_name[name]) for name in base.SPECIES) != counts:
        raise ValueError("fixed witness counts do not match")
    if len({coord for values in by_name.values() for coord in values}) != base.TOKEN_COUNT:
        raise ValueError("fixed witness tokens overlap")

    bears = set(by_name["bear"])
    bear_pairs = [sorted(component) for component in base.components(bears) if len(component) == 2]
    ordered_bears: list[Coord] = []
    for pair in bear_pairs:
        left, right = pair
        if (right[0] - left[0], right[1] - left[1]) not in LINE_DIRECTIONS:
            left, right = right, left
        ordered_bears.extend((left, right))
    ordered_bears.extend(sorted(bears - set(ordered_bears)))

    elk_groups = _optimal_elk_groups(by_name["elk"])
    ordered_elk = [coord for group in elk_groups for coord in group]

    salmon = set(by_name["salmon"])
    valid_salmon = []
    invalid_salmon: set[Coord] = set()
    for component in base.components(salmon):
        if all(len(base.neighbors(coord) & component) <= 2 for coord in component):
            valid_salmon.append(sorted(component))
        else:
            invalid_salmon.update(component)
    valid_salmon.sort(key=lambda row: -len(row))
    ordered_salmon = [coord for component in valid_salmon for coord in component]
    ordered_salmon.extend(sorted(invalid_salmon))

    ordered = (
        ordered_bears
        + ordered_elk
        + ordered_salmon
        + sorted(by_name["hawk"])
        + sorted(by_name["fox"])
    )
    anchor_q, anchor_r = sorted(by_name["fox"])[0]
    return [(q - anchor_q, r - anchor_r) for q, r in ordered]


@dataclass(frozen=True)
class RelaxationVariables:
    q: list[cp_model.IntVar]
    r: list[cp_model.IntVar]
    species_by_token: list[int]
    non_fox_score: cp_model.IntVar
    fox_score: cp_model.IntVar


def build_model(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    fixed_tokens: list[dict[str, int | str]] | None = None,
) -> tuple[cp_model.CpModel, RelaxationVariables]:
    if sum(counts) != base.TOKEN_COUNT or any(not 0 <= count <= base.COUNT_CAP for count in counts):
        raise ValueError(f"invalid counts: {counts}")
    if counts[base.SPECIES_CODE["fox"]] == 0:
        raise ValueError("the motif relaxation requires a fox anchor")
    if target > base.count_relaxation(counts):
        raise ValueError("target exceeds the standalone relaxation")

    model = cp_model.CpModel()
    species_by_token = [species for species, count in enumerate(counts) for _ in range(count)]
    by_species = {
        species: [index for index, value in enumerate(species_by_token) if value == species]
        for species in range(len(base.SPECIES))
    }
    q = [
        model.new_int_var(-base.GLOBAL_RADIUS, base.GLOBAL_RADIUS, f"q_{token}")
        for token in range(base.TOKEN_COUNT)
    ]
    r = [
        model.new_int_var(-base.GLOBAL_RADIUS, base.GLOBAL_RADIUS, f"r_{token}")
        for token in range(base.TOKEN_COUNT)
    ]
    width = 2 * base.GLOBAL_RADIUS + 1
    coordinate_id = [
        model.new_int_var(0, width * width - 1, f"coordinate_{token}")
        for token in range(base.TOKEN_COUNT)
    ]
    for token in range(base.TOKEN_COUNT):
        model.add(q[token] + r[token] >= -base.GLOBAL_RADIUS)
        model.add(q[token] + r[token] <= base.GLOBAL_RADIUS)
        model.add(
            coordinate_id[token]
            == (q[token] + base.GLOBAL_RADIUS) * width + r[token] + base.GLOBAL_RADIUS
        )
    model.add_all_different(coordinate_id)

    foxes = by_species[base.SPECIES_CODE["fox"]]
    model.add(q[foxes[0]] == 0)
    model.add(r[foxes[0]] == 0)

    # Bear A: a real score determines the number of isolated pairs.  We keep
    # those pair edges but deliberately drop isolation from other bears.
    bear_choices = []
    bear_score_terms = []
    bears = by_species[base.SPECIES_CODE["bear"]]
    for pair_count in range(len(bears) // 2 + 1):
        selected = model.new_bool_var(f"bear_choice_{pair_count}")
        bear_choices.append(selected)
        bear_score_terms.append(base.BEAR_SCORES[pair_count] * selected)
        for pair_index in range(pair_count):
            members = bears[2 * pair_index : 2 * pair_index + 2]
            _conditional_offsets(
                model,
                q,
                r,
                members,
                ((dq, dr) for dq, dr in DIRECTIONS[:3]),
                selected,
                f"bear_{pair_count}_{pair_index}",
            )
    model.add_exactly_one(bear_choices)

    # Elk A: one choice represents the disjoint line packing that realizes
    # the board's optimal Elk-A score.  Singles need no geometry.
    elk_choices = []
    elk_score_terms = []
    elk_score_by_length = {1: 2, 2: 5, 3: 9, 4: 13}
    elks = by_species[base.SPECIES_CODE["elk"]]
    for choice_index, partition in enumerate(elk_partitions(len(elks))):
        selected = model.new_bool_var(f"elk_choice_{choice_index}")
        elk_choices.append(selected)
        elk_score_terms.append(sum(elk_score_by_length[size] for size in partition) * selected)
        cursor = 0
        for group_index, size in enumerate(partition):
            members = elks[cursor : cursor + size]
            cursor += size
            _conditional_offsets(
                model,
                q,
                r,
                members,
                _line_offsets(size),
                selected,
                f"elk_{choice_index}_{group_index}",
            )
    model.add_exactly_one(elk_choices)

    # Salmon A: scored valid components are represented exactly internally.
    # Cross-component nonadjacency is dropped, and unused salmon are free.
    salmon_choices = []
    salmon_score_terms = []
    salmon = by_species[base.SPECIES_CODE["salmon"]]
    for choice_index, partition in enumerate(salmon_partitions(len(salmon))):
        selected = model.new_bool_var(f"salmon_choice_{choice_index}")
        salmon_choices.append(selected)
        salmon_score_terms.append(sum(base.SALMON_SCORES[size] for size in partition) * selected)
        cursor = 0
        for component_index, size in enumerate(partition):
            members = salmon[cursor : cursor + size]
            cursor += size
            _conditional_offsets(
                model,
                q,
                r,
                members,
                salmon_component_offsets(size),
                selected,
                f"salmon_{choice_index}_{component_index}",
            )
    model.add_exactly_one(salmon_choices)

    hawk_scores = sorted(set(base.HAWK_SCORES[: counts[base.SPECIES_CODE["hawk"]] + 1]))
    hawk_score = model.new_int_var(min(hawk_scores), max(hawk_scores), "hawk_score")
    model.add_allowed_assignments([hawk_score], [(score,) for score in hawk_scores])

    non_fox_upper = sum(base.STANDALONE_SCORES[i][counts[i]] for i in range(4))
    non_fox_score = model.new_int_var(0, non_fox_upper, "non_fox_score")
    model.add(
        non_fox_score
        == sum(bear_score_terms) + sum(elk_score_terms) + sum(salmon_score_terms) + hawk_score
    )

    # Fox A observations are exact positive witnesses: an observation can be
    # selected only when at least one token of that species is geometrically
    # adjacent.  Negative adjacency is irrelevant to an upper-bound model.
    coverage_terms = []
    for fox_order, fox in enumerate(foxes):
        for species, targets in by_species.items():
            relevant = [target for target in targets if target != fox]
            if not relevant:
                continue
            covered = model.new_bool_var(f"fox_{fox_order}_sees_{species}")
            adjacency_terms = []
            for target_token in relevant:
                adjacent = model.new_bool_var(f"fox_{fox_order}_adjacent_{target_token}")
                dq = model.new_int_var(
                    -2 * base.GLOBAL_RADIUS,
                    2 * base.GLOBAL_RADIUS,
                    f"fox_{fox_order}_dq_{target_token}",
                )
                dr = model.new_int_var(
                    -2 * base.GLOBAL_RADIUS,
                    2 * base.GLOBAL_RADIUS,
                    f"fox_{fox_order}_dr_{target_token}",
                )
                model.add(dq == q[target_token] - q[fox])
                model.add(dr == r[target_token] - r[fox])
                model.add_allowed_assignments([dq, dr], DIRECTIONS).only_enforce_if(adjacent)
                adjacency_terms.append(adjacent)
            model.add(covered <= sum(adjacency_terms))
            coverage_terms.append(covered)
    fox_upper = counts[base.SPECIES_CODE["fox"]] * (
        sum(count > 0 for count in counts[:4]) + int(counts[4] >= 2)
    )
    fox_score = model.new_int_var(0, fox_upper, "fox_score")
    model.add(fox_score == sum(coverage_terms))
    model.add(non_fox_score + fox_score >= target)

    if fixed_tokens is not None:
        fixed_coordinates = relabel_fixed_witness(counts, fixed_tokens)
        for token, (token_q, token_r) in enumerate(fixed_coordinates):
            model.add(q[token] == token_q)
            model.add(r[token] == token_r)

    return model, RelaxationVariables(q, r, species_by_token, non_fox_score, fox_score)


def solve_relaxation(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    workers: int,
    time_limit: float,
    seed: int,
    fixed_tokens: list[dict[str, int | str]] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    model, variables = build_model(counts, target, fixed_tokens=fixed_tokens)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    payload: dict[str, Any] = {
        "counts": list(counts),
        "target": target,
        "status": solver.status_name(status),
        "proof_complete": status == cp_model.INFEASIBLE,
        "wall_seconds": solver.wall_time,
        "elapsed_seconds": time.monotonic() - started,
        "branches": solver.num_branches,
        "conflicts": solver.num_conflicts,
        "workers": workers,
        "time_limit_seconds": time_limit,
        "seed": seed,
    }
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        payload["relaxation_non_fox_score"] = solver.value(variables.non_fox_score)
        payload["relaxation_fox_score"] = solver.value(variables.fox_score)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, help="comma-separated B,E,S,H,F counts")
    parser.add_argument("--target", required=True, type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    counts = tuple(int(value) for value in args.counts.split(","))
    if len(counts) != len(base.SPECIES):
        raise SystemExit("--counts needs exactly five integers")
    payload = solve_relaxation(
        counts,
        args.target,
        workers=args.workers,
        time_limit=args.time_limit,
        seed=args.seed,
    )
    source = Path(__file__).resolve()
    payload.update(
        {
            "schema": "aaaaa-motif-coordinate-relaxation-v1",
            "ortools_version": ORTOOLS_VERSION,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "relaxation": {
                "whole_board_connectivity_required": False,
                "bear_pair_isolation_required": False,
                "salmon_component_separation_required": False,
                "hawk_isolation_required": False,
                "token_nonoverlap_exact": True,
                "forced_scoring_motifs_exact": True,
                "fox_positive_observations_exact": True,
            },
        }
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["proof_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
