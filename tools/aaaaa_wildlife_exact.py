#!/usr/bin/env python3
"""Exact labeled-token CP-SAT solver for the AAAAA wildlife maximum.

Unlike a cell-indexed board model, this formulation has exactly twenty token
positions. Pairwise adjacency is derived from their axial coordinates, so the
model is globally complete inside radius 19 around a canonical token at the
origin. Count allocations are solved independently; every allocation whose
standalone relaxation can beat the incumbent must be proven infeasible before
the incumbent is declared optimal.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model

SPECIES = ("bear", "elk", "salmon", "hawk", "fox")
SPECIES_CODE = {name: index for index, name in enumerate(SPECIES)}
DIRECTIONS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
TOKEN_COUNT = 20
COUNT_CAP = 6
SALMON_SCORES = (0, 2, 5, 8, 12, 16, 20)
HAWK_SCORES = (0, 2, 5, 8, 11, 14, 18)
BEAR_SCORES = (0, 4, 11, 19)
STANDALONE_SCORES = (
    (0, 0, 4, 4, 11, 11, 19),
    (0, 2, 5, 9, 13, 15, 18),
    (0, 2, 5, 8, 12, 16, 20),
    (0, 2, 5, 8, 11, 14, 18),
)

GLOBAL_RADIUS = TOKEN_COUNT - 1
KNOWN_INCUMBENT_SCORE = 68
KNOWN_INCUMBENT_TOKENS: list[dict[str, int | str]] = [
    {"q": 3, "r": 0, "wildlife": "elk"},
    {"q": 1, "r": 1, "wildlife": "bear"},
    {"q": 2, "r": 1, "wildlife": "fox"},
    {"q": 3, "r": 1, "wildlife": "elk"},
    {"q": 0, "r": 2, "wildlife": "bear"},
    {"q": 1, "r": 2, "wildlife": "salmon"},
    {"q": 2, "r": 2, "wildlife": "fox"},
    {"q": 3, "r": 2, "wildlife": "elk"},
    {"q": 1, "r": 3, "wildlife": "salmon"},
    {"q": 2, "r": 3, "wildlife": "bear"},
    {"q": 3, "r": 3, "wildlife": "elk"},
    {"q": 0, "r": 4, "wildlife": "salmon"},
    {"q": 1, "r": 4, "wildlife": "bear"},
    {"q": 2, "r": 4, "wildlife": "fox"},
    {"q": 3, "r": 4, "wildlife": "fox"},
    {"q": 0, "r": 5, "wildlife": "salmon"},
    {"q": 1, "r": 5, "wildlife": "salmon"},
    {"q": 2, "r": 5, "wildlife": "salmon"},
    {"q": 3, "r": 5, "wildlife": "bear"},
    {"q": 2, "r": 6, "wildlife": "bear"},
]


def count_relaxation(counts: tuple[int, int, int, int, int]) -> int:
    """Score upper bound that ignores all geometric interference."""
    non_fox_types = sum(count > 0 for count in counts[:4])
    fox_types = non_fox_types + int(counts[SPECIES_CODE["fox"]] >= 2)
    non_fox_score = sum(STANDALONE_SCORES[species][counts[species]] for species in range(4))
    return non_fox_score + counts[SPECIES_CODE["fox"]] * fox_types


def count_vectors(
    minimum_score: int | None = None,
) -> list[tuple[tuple[int, int, int, int, int], int]]:
    vectors = []
    for counts in itertools.product(range(COUNT_CAP + 1), repeat=len(SPECIES)):
        if sum(counts) != TOKEN_COUNT:
            continue
        bound = count_relaxation(counts)
        if minimum_score is None or bound >= minimum_score:
            vectors.append((counts, bound))
    vectors.sort(key=lambda item: (-item[1], item[0]))
    return vectors


def neighbors(coord: tuple[int, int]) -> set[tuple[int, int]]:
    q, r = coord
    return {(q + dq, r + dr) for dq, dr in DIRECTIONS}


def components(coords: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    result = []
    remaining = set(coords)
    while remaining:
        component = {remaining.pop()}
        frontier = list(component)
        while frontier:
            found = neighbors(frontier.pop()) & remaining
            component.update(found)
            frontier.extend(found)
            remaining.difference_update(found)
        result.append(component)
    return result


def score_tokens(tokens: list[dict[str, int | str]]) -> tuple[int, int, int, int, int]:
    """Independent executable specification used to validate solver witnesses."""
    occupants = {(int(token["q"]), int(token["r"])): str(token["wildlife"]) for token in tokens}
    if len(occupants) != len(tokens):
        raise ValueError("tokens overlap")
    positions = {
        species: {coord for coord, wildlife in occupants.items() if wildlife == species}
        for species in SPECIES
    }

    bear_pairs = sum(len(component) == 2 for component in components(positions["bear"]))
    bear = BEAR_SCORES[min(bear_pairs, len(BEAR_SCORES) - 1)]

    elk_coords = sorted(positions["elk"])
    groups = [(1 << index, 2) for index in range(len(elk_coords))]
    elk_index = {coord: index for index, coord in enumerate(elk_coords)}
    for start_index, (q, r) in enumerate(elk_coords):
        for dq, dr in DIRECTIONS[:3]:
            mask = 1 << start_index
            for step, score in ((1, 5), (2, 9), (3, 13)):
                coord = (q + step * dq, r + step * dr)
                if coord not in elk_index:
                    break
                mask |= 1 << elk_index[coord]
                groups.append((mask, score))
    elk_dp = [0] * (1 << len(elk_coords))
    for state in range(1, len(elk_dp)):
        first = (state & -state).bit_length() - 1
        for group, score in groups:
            if group & (1 << first) and group & state == group:
                elk_dp[state] = max(elk_dp[state], score + elk_dp[state & ~group])
    elk = elk_dp[-1]

    salmon = sum(
        SALMON_SCORES[len(component)]
        for component in components(positions["salmon"])
        if all(len(neighbors(coord) & component) <= 2 for coord in component)
    )
    isolated_hawks = sum(not (neighbors(coord) & positions["hawk"]) for coord in positions["hawk"])
    hawk = HAWK_SCORES[isolated_hawks]
    fox = sum(
        len({occupants[other] for other in neighbors(coord) if other in occupants})
        for coord in positions["fox"]
    )
    return bear, elk, salmon, hawk, fox


def render_tokens(tokens: list[dict[str, int | str]]) -> str:
    occupants = {
        (int(token["q"]), int(token["r"])): str(token["wildlife"])[0].upper() for token in tokens
    }
    minimum_q = min(q for q, _ in occupants)
    maximum_q = max(q for q, _ in occupants)
    minimum_r = min(r for _, r in occupants)
    maximum_r = max(r for _, r in occupants)
    lines = []
    for r in range(minimum_r, maximum_r + 1):
        cells = " ".join(occupants.get((q, r), ".") for q in range(minimum_q, maximum_q + 1))
        lines.append(f"r={r:>2} {' ' * (r - minimum_r)}{cells}")
    return "\n".join(lines)


def add_optimum_to_complete_proof(payload: dict[str, object]) -> None:
    if payload["minimum_score"] != KNOWN_INCUMBENT_SCORE + 1 or not payload["proof_complete"]:
        return
    breakdown = score_tokens(KNOWN_INCUMBENT_TOKENS)
    if sum(breakdown) != KNOWN_INCUMBENT_SCORE:
        raise RuntimeError("bundled incumbent no longer scores 68")
    payload["optimal_score"] = KNOWN_INCUMBENT_SCORE
    payload["optimal_score_breakdown"] = list(breakdown)
    payload["optimal_counts"] = [6, 4, 6, 0, 4]
    payload["optimal_configuration"] = KNOWN_INCUMBENT_TOKENS
    payload["optimal_configuration_ascii"] = render_tokens(KNOWN_INCUMBENT_TOKENS)


def add_provenance(payload: dict[str, object]) -> None:
    model_source = Path(__file__).resolve()
    production_verifier = (
        model_source.parents[1]
        / "crates"
        / "cascadia-game"
        / "src"
        / "bin"
        / "aaaaa_wildlife_solver.rs"
    )
    payload["model"] = "labeled-token-cp-sat-v2"
    payload["ortools_version"] = ORTOOLS_VERSION
    payload["model_source_sha256"] = hashlib.sha256(model_source.read_bytes()).hexdigest()
    payload["production_verifier_sha256"] = hashlib.sha256(
        production_verifier.read_bytes()
    ).hexdigest()
    payload["assumptions"] = {
        "occupied_connected_hexes": TOKEN_COUNT,
        "maximum_per_species": COUNT_CAP,
        "scoring_cards": "AAAAA",
        "other_game_mechanics": "ignored",
        "global_hex_radius": GLOBAL_RADIUS,
    }


@dataclass
class ExactVariables:
    q: list[cp_model.IntVar]
    r: list[cp_model.IntVar]
    total_score: cp_model.IntVar
    species_by_token: list[int]


def species_tokens(counts: tuple[int, int, int, int, int]) -> list[int]:
    # Prefer fox as the anchor because it dominates the high-score proof. For
    # fox-free allocations, the first present species becomes the anchor.
    # The lexicographically first token of that species is translated to zero.
    order = (
        SPECIES_CODE["fox"],
        SPECIES_CODE["bear"],
        SPECIES_CODE["elk"],
        SPECIES_CODE["salmon"],
        SPECIES_CODE["hawk"],
    )
    return [species for species in order for _ in range(counts[species])]


def adjacency_variables(
    model: cp_model.CpModel,
    q: list[cp_model.IntVar],
    r: list[cp_model.IntVar],
) -> dict[tuple[int, int], cp_model.IntVar]:
    """Create exact adjacency variables using compact coordinate tables."""
    adjacency: dict[tuple[int, int], cp_model.IntVar] = {}
    for left in range(TOKEN_COUNT):
        for right in range(left + 1, TOKEN_COUNT):
            dq = model.new_int_var(-2 * GLOBAL_RADIUS, 2 * GLOBAL_RADIUS, f"dq_{left}_{right}")
            dr = model.new_int_var(-2 * GLOBAL_RADIUS, 2 * GLOBAL_RADIUS, f"dr_{left}_{right}")
            model.add(dq == q[right] - q[left])
            model.add(dr == r[right] - r[left])
            adjacent = model.new_bool_var(f"adj_{left}_{right}")
            model.add_allowed_assignments([dq, dr], DIRECTIONS).only_enforce_if(adjacent)
            model.add_forbidden_assignments([dq, dr], DIRECTIONS).only_enforce_if(
                adjacent.negated()
            )
            adjacency[(left, right)] = adjacent
    return adjacency


def adjacent(
    adjacency: dict[tuple[int, int], cp_model.IntVar], left: int, right: int
) -> cp_model.IntVar:
    return adjacency[(left, right) if left < right else (right, left)]


def build_model(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
) -> tuple[cp_model.CpModel, ExactVariables]:
    if sum(counts) != TOKEN_COUNT or any(count < 0 or count > COUNT_CAP for count in counts):
        raise ValueError(f"invalid counts: {counts}")
    if count_relaxation(counts) < minimum_score:
        raise ValueError(f"count relaxation cannot reach {minimum_score}: {counts}")

    model = cp_model.CpModel()
    species_by_token = species_tokens(counts)
    q = [
        model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"q_{token}")
        for token in range(TOKEN_COUNT)
    ]
    r = [
        model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"r_{token}")
        for token in range(TOKEN_COUNT)
    ]
    coordinate_id = [
        model.new_int_var(0, (2 * GLOBAL_RADIUS + 1) ** 2 - 1, f"coord_{token}")
        for token in range(TOKEN_COUNT)
    ]
    width = 2 * GLOBAL_RADIUS + 1
    for token in range(TOKEN_COUNT):
        diagonal = model.new_int_var(-GLOBAL_RADIUS, GLOBAL_RADIUS, f"diag_{token}")
        model.add(diagonal == q[token] + r[token])
        model.add(
            coordinate_id[token] == (q[token] + GLOBAL_RADIUS) * width + r[token] + GLOBAL_RADIUS
        )
    model.add_all_different(coordinate_id)
    model.add(q[0] == 0)
    model.add(r[0] == 0)

    # Permuting tokens of one species changes nothing. Token 0 is the
    # lexicographically first token of the anchor species before translation;
    # every other same-species group is likewise sorted by coordinate id.
    by_species = {
        species: [index for index, value in enumerate(species_by_token) if value == species]
        for species in range(len(SPECIES))
    }
    for indices in by_species.values():
        for left, right in itertools.pairwise(indices):
            model.add(coordinate_id[left] < coordinate_id[right])

    adjacency = adjacency_variables(model, q, r)

    if enforce_connectivity:
        # A rooted arborescence certifies connectivity without the many
        # equivalent integer-flow assignments of the original formulation.
        depth = [
            model.new_int_var(0, TOKEN_COUNT - 1, f"depth_{token}") for token in range(TOKEN_COUNT)
        ]
        model.add(depth[0] == 0)
        for child in range(1, TOKEN_COUNT):
            parents = []
            for parent in range(TOKEN_COUNT):
                if child == parent:
                    continue
                chosen = model.new_bool_var(f"parent_{child}_{parent}")
                model.add(chosen <= adjacent(adjacency, child, parent))
                model.add(depth[child] > depth[parent]).only_enforce_if(chosen)
                parents.append(chosen)
            model.add_exactly_one(parents)

    # Bear A isolated pairs.
    bears = by_species[SPECIES_CODE["bear"]]
    bear_pairs = []
    for left, right in itertools.combinations(bears, 2):
        pair = model.new_bool_var(f"bear_pair_{left}_{right}")
        model.add(pair <= adjacent(adjacency, left, right))
        for other in bears:
            if other in (left, right):
                continue
            model.add(pair + adjacent(adjacency, left, other) <= 1)
            model.add(pair + adjacent(adjacency, right, other) <= 1)
        bear_pairs.append(pair)
    bear_pair_count = model.new_int_var(0, len(bears) // 2, "bear_pair_count")
    bear_score = model.new_int_var(0, max(BEAR_SCORES), "bear_score")
    model.add(bear_pair_count == sum(bear_pairs))
    model.add_allowed_assignments(
        [bear_pair_count, bear_score],
        [[count, BEAR_SCORES[min(count, 3)]] for count in range(len(bears) // 2 + 1)],
    )

    # Elk A disjoint straight-line group packing. Elk labels are ordered by
    # coordinate id, so a combination has exactly one possible order along
    # each of the three positive-id line directions. The old permutation
    # encoding represented the same physical line up to 24 times.
    elk = by_species[SPECIES_CODE["elk"]]
    elk_groups: list[tuple[tuple[int, ...], int, cp_model.IntVar]] = []
    for token in elk:
        elk_groups.append(((token,), 2, model.new_bool_var(f"elk_single_{token}")))
    positive_id_directions = ((1, 0), (1, -1), (0, 1))
    for length, score in ((2, 5), (3, 9), (4, 13)):
        for ordered in itertools.combinations(elk, length):
            for direction_index, (dq, dr) in enumerate(positive_id_directions):
                group = model.new_bool_var(
                    f"elk_line_{length}_{'_'.join(map(str, ordered))}_{direction_index}"
                )
                for step in range(1, length):
                    model.add(q[ordered[step]] == q[ordered[0]] + step * dq).only_enforce_if(group)
                    model.add(r[ordered[step]] == r[ordered[0]] + step * dr).only_enforce_if(group)
                elk_groups.append((tuple(ordered), score, group))
    for token in elk:
        model.add(sum(group for members, _, group in elk_groups if token in members) <= 1)
    elk_score = model.new_int_var(0, 18, "elk_score")
    model.add(elk_score == sum(score * group for _, score, group in elk_groups))

    # Salmon A exact valid connected components. Every nonempty token subset
    # is a possible component; cut constraints enforce connectivity, boundary
    # constraints make it the full component, and degree<=2 rejects branches.
    salmon = by_species[SPECIES_CODE["salmon"]]
    salmon_components: list[tuple[tuple[int, ...], int, cp_model.IntVar]] = []
    for size in range(1, len(salmon) + 1):
        for members in itertools.combinations(salmon, size):
            member_set = set(members)
            component = model.new_bool_var(f"salmon_component_{'_'.join(map(str, members))}")
            for member in members:
                model.add(
                    sum(adjacent(adjacency, member, other) for other in members if other != member)
                    <= 2 + len(members) * (1 - component)
                )
                for outside in salmon:
                    if outside not in member_set:
                        model.add(component + adjacent(adjacency, member, outside) <= 1)
            first = members[0]
            rest = members[1:]
            for mask in range(1 << len(rest)):
                left = {first}
                left.update(rest[index] for index in range(len(rest)) if mask & (1 << index))
                if len(left) == len(members):
                    continue
                right = member_set - left
                model.add(
                    sum(adjacent(adjacency, one, other) for one in left for other in right)
                    >= component
                )
            salmon_components.append((members, SALMON_SCORES[size], component))
    for token in salmon:
        model.add(
            sum(component for members, _, component in salmon_components if token in members) <= 1
        )
    salmon_score = model.new_int_var(0, 20, "salmon_score")
    model.add(salmon_score == sum(score * component for _, score, component in salmon_components))

    # Hawk A isolated-count table.
    hawks = by_species[SPECIES_CODE["hawk"]]
    hawk_isolated = []
    for hawk in hawks:
        isolated = model.new_bool_var(f"hawk_isolated_{hawk}")
        for other in hawks:
            if other != hawk:
                model.add(isolated + adjacent(adjacency, hawk, other) <= 1)
        hawk_isolated.append(isolated)
    hawk_count = model.new_int_var(0, len(hawks), "hawk_isolated_count")
    hawk_score = model.new_int_var(0, 18, "hawk_score")
    model.add(hawk_count == sum(hawk_isolated))
    model.add_allowed_assignments(
        [hawk_count, hawk_score],
        [[count, HAWK_SCORES[count]] for count in range(len(hawks) + 1)],
    )

    # Fox A distinct adjacent species.
    foxes = by_species[SPECIES_CODE["fox"]]
    fox_distinct: dict[tuple[int, int], cp_model.IntVar] = {}
    for fox in foxes:
        for species in range(len(SPECIES)):
            distinct = model.new_bool_var(f"fox_distinct_{fox}_{species}")
            targets = by_species[species]
            model.add(
                distinct
                <= sum(adjacent(adjacency, fox, target) for target in targets if target != fox)
            )
            fox_distinct[(fox, species)] = distinct

    # Two distinct hexes have at most two common neighbors; three or more
    # distinct hexes have at most one. Assigning each fox that sees a set of
    # non-fox species to one target tuple gives these aggregate overlap cuts.
    # They are redundant consequences of the coordinate model, but expose the
    # most important high-fox-score geometry directly to the SAT relaxation.
    present_non_fox = [species for species in range(4) if counts[species] > 0]
    for size in range(2, len(present_non_fox) + 1):
        for species_group in itertools.combinations(present_non_fox, size):
            target_tuples = 1
            for species in species_group:
                target_tuples *= counts[species]
            capacity = min(len(foxes), (2 if size == 2 else 1) * target_tuples)
            if capacity == len(foxes):
                continue
            overlaps = []
            for fox in foxes:
                overlap = model.new_bool_var(
                    f"fox_overlap_{fox}_{'_'.join(map(str, species_group))}"
                )
                members = [fox_distinct[(fox, species)] for species in species_group]
                for member in members:
                    model.add(overlap <= member)
                model.add(overlap >= sum(members) - len(members) + 1)
                overlaps.append(overlap)
            model.add(sum(overlaps) <= capacity)

    fox_score = model.new_int_var(0, len(foxes) * len(SPECIES), "fox_score")
    model.add(fox_score == sum(fox_distinct.values()))

    upper = count_relaxation(counts)
    if maximum_score is not None:
        upper = min(upper, maximum_score)
    if minimum_score > upper:
        raise ValueError(f"score interval is empty: [{minimum_score}, {upper}]")
    total_score = model.new_int_var(minimum_score, upper, "total_score")
    model.add(total_score == bear_score + elk_score + salmon_score + hawk_score + fox_score)
    if maximize:
        model.maximize(total_score)
    return model, ExactVariables(
        q=q, r=r, total_score=total_score, species_by_token=species_by_token
    )


class Progress(cp_model.CpSolverSolutionCallback):
    def __init__(self, total_score: cp_model.IntVar, started: float) -> None:
        super().__init__()
        self.total_score = total_score
        self.started = started
        self.best = -1

    def on_solution_callback(self) -> None:
        score = self.value(self.total_score)
        if score > self.best:
            self.best = score
            print(
                f"candidate={score} bound={self.best_objective_bound:.3f} "
                f"elapsed={time.monotonic() - self.started:.2f}s",
                flush=True,
            )


def solve_counts(
    counts: tuple[int, int, int, int, int],
    minimum_score: int,
    time_limit: float,
    workers: int,
    seed: int,
    log_search: bool,
    *,
    maximize: bool = True,
    maximum_score: int | None = None,
    enforce_connectivity: bool = True,
) -> dict[str, object]:
    started = time.monotonic()
    model, variables = build_model(
        counts,
        minimum_score,
        maximize=maximize,
        maximum_score=maximum_score,
        enforce_connectivity=enforce_connectivity,
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.log_search_progress = log_search
    if maximize:
        callback = Progress(variables.total_score, started)
        status = solver.solve(model, callback)
    else:
        status = solver.solve(model)
    tokens: list[dict[str, int | str]] = []
    objective: int | None = None
    model_score: int | None = None
    score_breakdown: list[int] | None = None
    if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        model_score = int(solver.value(variables.total_score))
        for token, species in enumerate(variables.species_by_token):
            tokens.append(
                {
                    "q": int(solver.value(variables.q[token])),
                    "r": int(solver.value(variables.r[token])),
                    "wildlife": SPECIES[species],
                }
            )
        tokens.sort(key=lambda row: (int(row["r"]), int(row["q"]), str(row["wildlife"])))
        score_breakdown = list(score_tokens(tokens))
        objective = sum(score_breakdown)
        if maximize and objective != model_score:
            raise RuntimeError(
                f"model witness scored {score_breakdown}, not claimed objective {model_score}"
            )
        if not maximize and objective < minimum_score:
            raise RuntimeError(f"feasibility witness scored {objective}, below {minimum_score}")
        occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
        if enforce_connectivity and len(components(occupied)) != 1:
            raise RuntimeError("model witness is disconnected")
    return {
        "counts": list(counts),
        "count_relaxation": count_relaxation(counts),
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
        },
        "tokens": tokens,
    }


def solve(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def write_payload(payload: dict[str, object]) -> None:
        add_provenance(payload)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)

    if args.counts:
        counts = tuple(int(value) for value in args.counts.split(","))
        if len(counts) != len(SPECIES):
            raise SystemExit("--counts requires bear,elk,salmon,hawk,fox")
        result = solve_counts(
            counts, args.minimum_score, args.time_limit, args.workers, args.seed, args.log_search
        )
        payload: dict[str, object] = {
            "minimum_score": args.minimum_score,
            "allocation_count": 1,
            "proof_complete": False,
            "allocation_proof_complete": result["model_status"] == "INFEASIBLE",
            "counterexample_found": result["objective"] is not None,
            "results": [result],
        }
    else:
        allocations = count_vectors(args.minimum_score)
        results: list[dict[str, object]] = []
        if args.resume and output.exists():
            prior = json.loads(output.read_text(encoding="utf-8"))
            if prior.get("minimum_score") != args.minimum_score:
                raise SystemExit("resume file has a different minimum_score")
            results = list(prior.get("results", []))
            expected_prefix = [list(counts) for counts, _ in allocations[: len(results)]]
            if [result.get("counts") for result in results] != expected_prefix:
                raise SystemExit("resume file allocations do not match this solver ordering")
            if results and results[-1].get("model_status") == "UNKNOWN":
                incomplete = results.pop()
                print(f"retrying incomplete allocation {incomplete['counts']}", flush=True)
            if any(result.get("model_status") != "INFEASIBLE" for result in results):
                raise SystemExit("resume file contains a non-infeasible completed allocation")
            print(f"resuming after {len(results)}/{len(allocations)} allocations", flush=True)

        for index, (counts, bound) in enumerate(allocations):
            if index < len(results):
                continue
            print(
                f"allocation {index + 1}/{len(allocations)} counts={counts} bound={bound}",
                flush=True,
            )
            result = solve_counts(
                counts,
                args.minimum_score,
                args.time_limit,
                args.workers,
                args.seed + index,
                args.log_search,
            )
            results.append(result)
            payload = {
                "minimum_score": args.minimum_score,
                "allocation_count": len(allocations),
                "proof_complete": len(results) == len(allocations)
                and all(item["model_status"] == "INFEASIBLE" for item in results),
                "counterexample_found": any(item["objective"] is not None for item in results),
                "results": results,
            }
            add_optimum_to_complete_proof(payload)
            write_payload(payload)
            if result["objective"] is not None:
                break
            if result["model_status"] != "INFEASIBLE":
                break
        payload = {
            "minimum_score": args.minimum_score,
            "allocation_count": len(allocations),
            "proof_complete": len(results) == len(allocations)
            and all(item["model_status"] == "INFEASIBLE" for item in results),
            "counterexample_found": any(item["objective"] is not None for item in results),
            "results": results,
        }
    add_optimum_to_complete_proof(payload)
    write_payload(payload)
    if "optimal_score" in payload:
        print(
            f"PROVED OPTIMAL: {payload['optimal_score']} wildlife points\n"
            f"breakdown [bear, elk, salmon, hawk, fox]: "
            f"{payload['optimal_score_breakdown']}\n"
            f"counts    [bear, elk, salmon, hawk, fox]: {payload['optimal_counts']}\n\n"
            f"{payload['optimal_configuration_ascii']}\n",
            flush=True,
        )
    if args.print_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"proof record: {output}", flush=True)
    statuses = {
        str(result["model_status"])
        for result in payload["results"]  # type: ignore[index]
    }
    return 0 if statuses <= {"INFEASIBLE", "OPTIMAL", "FEASIBLE"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", help="fix bear,elk,salmon,hawk,fox counts")
    parser.add_argument("--minimum-score", type=int, default=69)
    parser.add_argument("--time-limit", type=float, default=120.0, help="seconds per allocation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--log-search", action="store_true")
    parser.add_argument("--print-json", action="store_true", help="also print the complete ledger")
    parser.add_argument("--output", default="target/aaaaa-wildlife-solver/exact-result.json")
    parser.add_argument("--resume", action="store_true", help="resume an infeasible result prefix")
    return parser


def main() -> int:
    return solve(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
