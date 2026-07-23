#!/usr/bin/env python3
"""Layered local-packing certificates for AAAAA vectors with a unique anchor.

For a count vector containing exactly one token of at least one non-fox
species, every fox that observes that species must occupy one of the six cells
around the unique token.  This module:

1. enumerates every near-target non-fox scoring profile;
2. enumerates the fox subset around the unique token up to the hexagon's
   dihedral symmetry; and
3. solves a finite local set-packing relaxation for the remaining scoring
   motifs.

Motifs that cover no explicit fox may be placed abstractly, and the model
drops board connectivity plus Bear/Hawk/Salmon separation constraints.  It is
therefore a strict superset of legal boards: an exact local upper below the
challenged score is a sound infeasibility certificate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

from ortools import __version__ as ORTOOLS_VERSION
from ortools.sat.python import cp_model

from tools import aaaaa_wildlife_exact as base
from tools.aaaaa_wildlife_motif_certificate import (
    DIHEDRAL_TRANSFORMS,
    DIRECTIONS,
    Shape,
    adjacent,
    boundary,
    free_polyhexes,
    integer_partitions,
    normalize,
)

Coord = tuple[int, int]


@dataclass(frozen=True)
class SpeciesStructure:
    """One relaxed realization of a species' exact card score."""

    score: int
    groups: tuple[int, ...]


@dataclass(frozen=True)
class ScoreProfile:
    """One combination of non-fox scoring structures."""

    structures: tuple[SpeciesStructure, ...]

    @property
    def nonfox_score(self) -> int:
        return sum(structure.score for structure in self.structures)

    @property
    def group_signature(self) -> tuple[tuple[int, ...], ...]:
        return tuple(structure.groups for structure in self.structures)


@dataclass(frozen=True)
class LocalBound:
    status: str
    local_coverage: int | None
    fox_upper: int | None
    branches: int
    conflicts: int
    wall_seconds: float


def _partitions(total: int, maximum: int) -> tuple[tuple[int, ...], ...]:
    if total == 0:
        return ((),)
    return tuple(integer_partitions(total, maximum))


@cache
def species_structures(
    species: int,
    count: int,
    maximum_loss: int,
) -> tuple[SpeciesStructure, ...]:
    """Enumerate every near-maximum score structure for one species.

    Salmon tokens in branched (zero-scoring) components are represented as
    independent leftover singleton groups.  This deliberately drops their
    mutual adjacency and therefore enlarges the feasible set.
    """

    maximum = base.STANDALONE_SCORES[species][count]
    minimum = maximum - maximum_loss
    rows: set[SpeciesStructure] = set()
    if species == base.SPECIES_CODE["bear"]:
        for pairs in range(count // 2 + 1):
            score = base.BEAR_SCORES[min(pairs, len(base.BEAR_SCORES) - 1)]
            if score >= minimum:
                groups = tuple(sorted((2,) * pairs + (1,) * (count - 2 * pairs), reverse=True))
                rows.add(SpeciesStructure(score, groups))
    elif species == base.SPECIES_CODE["elk"]:
        for partition in _partitions(count, 4):
            score = sum({1: 2, 2: 5, 3: 9, 4: 13}[size] for size in partition)
            if score >= minimum:
                rows.add(SpeciesStructure(score, partition))
    elif species == base.SPECIES_CODE["salmon"]:
        for scored_count in range(count + 1):
            for partition in _partitions(scored_count, scored_count or 1):
                score = sum(base.SALMON_SCORES[size] for size in partition)
                if score < minimum:
                    continue
                leftovers = count - scored_count
                groups = tuple(sorted((*partition, *((1,) * leftovers)), reverse=True))
                rows.add(SpeciesStructure(score, groups))
    elif species == base.SPECIES_CODE["hawk"]:
        for isolated in range(count + 1):
            score = base.HAWK_SCORES[isolated]
            if score >= minimum:
                rows.add(SpeciesStructure(score, (1,) * count))
    else:
        raise ValueError(f"unsupported non-fox species index: {species}")
    return tuple(sorted(rows, key=lambda row: (-row.score, row.groups)))


def anchor_score(species: int) -> int:
    """Return the forced card score of a unique non-fox token."""

    if species == base.SPECIES_CODE["bear"]:
        return 0
    if species in (
        base.SPECIES_CODE["elk"],
        base.SPECIES_CODE["salmon"],
        base.SPECIES_CODE["hawk"],
    ):
        return 2
    raise ValueError("the anchor must be a non-fox species")


def choose_anchor(counts: tuple[int, int, int, int, int]) -> int:
    """Choose a deterministic unique non-fox species."""

    candidates = [species for species, count in enumerate(counts[:4]) if count == 1]
    if not candidates:
        raise ValueError("single-anchor bound requires a non-fox count of one")
    return min(candidates)


def score_profiles(
    counts: tuple[int, int, int, int, int],
    target: int,
    anchor: int,
) -> tuple[ScoreProfile, ...]:
    """Enumerate every non-fox structure that could reach ``target``."""

    count_upper = base.count_relaxation(counts)
    maximum_loss = count_upper - target
    if maximum_loss < 0:
        return ()
    choices: list[tuple[SpeciesStructure, ...]] = []
    for species, count in enumerate(counts[:4]):
        if species == anchor:
            choices.append((SpeciesStructure(anchor_score(species), ()),))
        else:
            choices.append(species_structures(species, count, maximum_loss))
    fox_maximum = counts[base.SPECIES_CODE["fox"]] * sum(count > 0 for count in counts)
    rows = {
        ScoreProfile(tuple(structures))
        for structures in itertools.product(*choices)
        if sum(structure.score for structure in structures) + fox_maximum >= target
    }
    return tuple(
        sorted(
            rows,
            key=lambda row: (-row.nonfox_score, row.group_signature),
        )
    )


def _transform(coord: Coord, transform: tuple[tuple[int, int], tuple[int, int]]) -> Coord:
    (qq, qr), (rq, rr) = transform
    q, r = coord
    return qq * q + qr * r, rq * q + rr * r


def _canonical_ring_subset(cells: Iterable[Coord]) -> tuple[Coord, ...]:
    source = tuple(cells)
    return min(
        tuple(sorted(_transform(cell, transform) for cell in source))
        for transform in DIHEDRAL_TRANSFORMS
    )


@cache
def ring_layouts(size: int) -> tuple[tuple[Coord, ...], ...]:
    """Return all size-``size`` subsets of a six-cell ring up to D6."""

    if not 0 <= size <= len(DIRECTIONS):
        raise ValueError("ring subset size must be between zero and six")
    rows = {
        _canonical_ring_subset(cells)
        for cells in itertools.combinations(DIRECTIONS, size)
    }
    return tuple(sorted(rows))


@cache
def _oriented_shapes(species: int, size: int) -> tuple[Shape, ...]:
    if size < 1:
        raise ValueError("group size must be positive")
    if species in (
        base.SPECIES_CODE["bear"],
        base.SPECIES_CODE["hawk"],
    ):
        if size == 1:
            sources = {((0, 0),)}
        elif species == base.SPECIES_CODE["bear"] and size == 2:
            sources = {((0, 0), (1, 0))}
        else:
            raise ValueError("invalid Bear/Hawk group")
    elif species == base.SPECIES_CODE["elk"]:
        if size > 4:
            raise ValueError("Elk-A lines have length at most four")
        sources = {tuple((step, 0) for step in range(size))}
    elif species == base.SPECIES_CODE["salmon"]:
        sources = {
            shape
            for shape in free_polyhexes(size)
            if all(
                sum(adjacent(cell, other) for other in shape) <= 2
                for cell in shape
            )
        }
    else:
        raise ValueError(f"unsupported local-placement species: {species}")

    result = set()
    for source in sources:
        for transform in DIHEDRAL_TRANSFORMS:
            result.add(frozenset(normalize(_transform(cell, transform) for cell in source)))
    return tuple(sorted(result, key=lambda shape: tuple(sorted(shape))))


@cache
def local_placements(
    foxes: tuple[Coord, ...],
    species: int,
    size: int,
) -> tuple[Shape, ...]:
    """Enumerate every group placement touching an explicit fox."""

    blocked = {(0, 0), *foxes}
    touching_cells = boundary(foxes) - blocked
    result = set()
    for shape in _oriented_shapes(species, size):
        for source_cell in shape:
            for target_cell in touching_cells:
                dq = target_cell[0] - source_cell[0]
                dr = target_cell[1] - source_cell[1]
                placed = frozenset((q + dq, r + dr) for q, r in shape)
                if placed.isdisjoint(blocked):
                    result.add(placed)
    return tuple(sorted(result, key=lambda shape: tuple(sorted(shape))))


def _selected_groups(
    model: cp_model.CpModel,
    foxes: tuple[Coord, ...],
    species: int,
    groups: tuple[int, ...],
) -> tuple[list[tuple[Shape, cp_model.IntVar]], dict[Coord, list[cp_model.IntVar]]]:
    terms: list[tuple[Shape, cp_model.IntVar]] = []
    occupants: dict[Coord, list[cp_model.IntVar]] = {}
    for size, required in sorted(collections.Counter(groups).items()):
        placements = local_placements(foxes, species, size)
        variables = [
            model.new_bool_var(f"s{species}_g{size}_p{index}")
            for index in range(len(placements))
        ]
        abstract = model.new_int_var(0, required, f"s{species}_g{size}_abstract")
        model.add(sum(variables) + abstract == required)
        for shape, variable in zip(placements, variables, strict=True):
            terms.append((shape, variable))
            for cell in shape:
                occupants.setdefault(cell, []).append(variable)
    return terms, occupants


def solve_local_profile(
    counts: tuple[int, int, int, int, int],
    profile: ScoreProfile,
    anchor: int,
    foxes: tuple[Coord, ...],
    *,
    workers: int,
    time_limit: float,
) -> LocalBound:
    """Maximize Fox-A coverage in one finite local relaxation."""

    model = cp_model.CpModel()
    all_terms: dict[int, list[tuple[Shape, cp_model.IntVar]]] = {}
    occupants: dict[Coord, list[cp_model.IntVar]] = {}
    for species in range(4):
        if species == anchor or counts[species] == 0:
            continue
        terms, species_occupants = _selected_groups(
            model,
            foxes,
            species,
            profile.structures[species].groups,
        )
        all_terms[species] = terms
        for cell, variables in species_occupants.items():
            occupants.setdefault(cell, []).extend(variables)
    for variables in occupants.values():
        model.add(sum(variables) <= 1)

    coverages: list[cp_model.IntVar] = []
    for fox_index, fox in enumerate(foxes):
        for species in range(4):
            if species == anchor or counts[species] == 0:
                continue
            covered = model.new_bool_var(f"fox{fox_index}_sees_s{species}")
            neighbors = [
                variable
                for shape, variable in all_terms[species]
                if any(adjacent(fox, cell) for cell in shape)
            ]
            model.add(covered <= sum(neighbors))
            coverages.append(covered)

    fox_count = counts[base.SPECIES_CODE["fox"]]
    abstract_foxes = fox_count - len(foxes)
    if abstract_foxes:
        self_coverage = len(foxes)
    else:
        self_coverage = sum(
            any(left != right and adjacent(left, right) for right in foxes)
            for left in foxes
        )
    local_upper = len(coverages) + self_coverage
    local_score = model.new_int_var(0, local_upper, "local_coverage")
    model.add(local_score == sum(coverages) + self_coverage)
    model.maximize(local_score)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status = solver.solve(model)
    if status != cp_model.OPTIMAL:
        return LocalBound(
            status=solver.status_name(status),
            local_coverage=None,
            fox_upper=None,
            branches=solver.num_branches,
            conflicts=solver.num_conflicts,
            wall_seconds=solver.wall_time,
        )

    local_value = int(solver.value(local_score))
    present_types = sum(count > 0 for count in counts)
    # Explicit foxes see the anchor. Abstract foxes are defined to be the
    # foxes that do not, and optimistically receive every other type.
    fox_upper = (
        len(foxes)
        + abstract_foxes * (present_types - 1)
        + local_value
    )
    return LocalBound(
        status="OPTIMAL",
        local_coverage=local_value,
        fox_upper=fox_upper,
        branches=solver.num_branches,
        conflicts=solver.num_conflicts,
        wall_seconds=solver.wall_time,
    )


def relaxed_upper_bound(
    counts: tuple[int, int, int, int, int],
    target: int,
    *,
    anchor: int | None = None,
    workers: int = 1,
    per_case_time_limit: float = 30.0,
) -> dict[str, Any]:
    """Try to exclude ``target`` with the single-anchor relaxation."""

    if sum(counts) != base.TOKEN_COUNT:
        raise ValueError("counts must sum to twenty")
    anchor = choose_anchor(counts) if anchor is None else anchor
    if counts[anchor] != 1 or anchor >= base.SPECIES_CODE["fox"]:
        raise ValueError("anchor must identify a unique non-fox token")
    count_upper = base.count_relaxation(counts)
    if target > count_upper:
        return {
            "status": "INFEASIBLE",
            "upper_bound": count_upper,
            "target": target,
            "profiles": 0,
            "cases": 0,
        }

    profiles = score_profiles(counts, target, anchor)
    fox_count = counts[base.SPECIES_CODE["fox"]]
    present_types = sum(count > 0 for count in counts)
    cache_by_geometry: dict[
        tuple[tuple[tuple[int, ...], ...], tuple[Coord, ...]],
        LocalBound,
    ] = {}
    cases = 0
    solver_wall = 0.0
    best_upper = -1
    best_case: dict[str, Any] | None = None

    for profile_index, profile in enumerate(profiles):
        for local_fox_count in range(fox_count + 1):
            theoretical_fox = fox_count * (present_types - 1) + local_fox_count
            theoretical_total = profile.nonfox_score + theoretical_fox
            if theoretical_total < target:
                best_upper = max(best_upper, theoretical_total)
                continue
            for layout_index, foxes in enumerate(ring_layouts(local_fox_count)):
                key = (profile.group_signature, foxes)
                result = cache_by_geometry.get(key)
                if result is None:
                    result = solve_local_profile(
                        counts,
                        profile,
                        anchor,
                        foxes,
                        workers=workers,
                        time_limit=per_case_time_limit,
                    )
                    cache_by_geometry[key] = result
                    cases += 1
                    solver_wall += result.wall_seconds
                case = {
                    "profile_index": profile_index,
                    "nonfox_score": profile.nonfox_score,
                    "structures": [asdict(row) for row in profile.structures],
                    "local_fox_count": local_fox_count,
                    "layout_index": layout_index,
                    "foxes": [list(cell) for cell in foxes],
                    "result": asdict(result),
                }
                if result.status != "OPTIMAL":
                    return {
                        "status": "UNKNOWN",
                        "upper_bound": None,
                        "target": target,
                        "anchor_species": base.SPECIES[anchor],
                        "profiles": len(profiles),
                        "cases": cases,
                        "failed_case": case,
                        "aggregate_solver_wall_seconds": solver_wall,
                    }
                total_upper = profile.nonfox_score + int(result.fox_upper)
                if total_upper > best_upper:
                    best_upper = total_upper
                    best_case = case
                if total_upper >= target:
                    return {
                        "status": "RELAXATION_FEASIBLE",
                        "upper_bound": total_upper,
                        "target": target,
                        "anchor_species": base.SPECIES[anchor],
                        "profiles": len(profiles),
                        "cases": cases,
                        "best_case": case,
                        "aggregate_solver_wall_seconds": solver_wall,
                    }
    return {
        "status": "INFEASIBLE",
        "upper_bound": min(target - 1, best_upper),
        "target": target,
        "anchor_species": base.SPECIES[anchor],
        "profiles": len(profiles),
        "cases": cases,
        "best_case": best_case,
        "aggregate_solver_wall_seconds": solver_wall,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proof_counts(payload: dict[str, Any]) -> set[tuple[int, ...]]:
    result = set()
    if payload.get("proof_complete") and "counts" in payload:
        result.add(tuple(int(value) for value in payload["counts"]))
    for row in payload.get("results", []):
        if row.get("proof_complete") and "counts" in row:
            result.add(tuple(int(value) for value in row["counts"]))
    return result


def _validated_incumbent(
    catalog_row: dict[str, Any],
    candidate_row: dict[str, Any] | None,
) -> dict[str, Any]:
    source = catalog_row
    if candidate_row is not None and int(candidate_row["score"]) >= int(catalog_row["optimum"]):
        source = candidate_row
    tokens = list(source["tokens"])
    counts = tuple(int(value) for value in catalog_row["counts"])
    observed_counts = tuple(
        sum(str(token["wildlife"]) == species for token in tokens)
        for species in base.SPECIES
    )
    occupied = {(int(token["q"]), int(token["r"])) for token in tokens}
    breakdown = list(base.score_tokens(tokens))
    if (
        observed_counts != counts
        or len(tokens) != base.TOKEN_COUNT
        or len(occupied) != base.TOKEN_COUNT
        or len(base.components(occupied)) != 1
    ):
        raise ValueError(f"invalid incumbent for {counts}")
    claimed = int(source.get("score", source.get("optimum", -1)))
    if sum(breakdown) != claimed:
        raise ValueError(f"incumbent score mismatch for {counts}: {sum(breakdown)} != {claimed}")
    return {
        "score": claimed,
        "score_breakdown": breakdown,
        "tokens": tokens,
    }


def run_batch(
    catalog_path: Path,
    candidates_path: Path,
    proof_ledgers: tuple[Path, ...],
    *,
    requested_counts: set[tuple[int, ...]] | None,
    workers: int,
    per_case_time_limit: float,
    challenge_offset: int = 1,
) -> dict[str, Any]:
    """Run the single-anchor relaxation over applicable unproven rows."""

    started = time.monotonic()
    catalog_payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    candidate_payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    catalog_rows = {
        tuple(int(value) for value in row["counts"]): row
        for row in catalog_payload["results"]
    }
    candidate_rows = {
        tuple(int(value) for value in row["counts"]): row
        for row in candidate_payload["candidates"]
    }
    proven = _proof_counts(catalog_payload)
    for ledger in proof_ledgers:
        proven.update(_proof_counts(json.loads(ledger.read_text(encoding="utf-8"))))

    selected = []
    for counts in sorted(catalog_rows):
        if requested_counts is not None and counts not in requested_counts:
            continue
        if requested_counts is None and counts in proven:
            continue
        if not any(count == 1 for count in counts[:4]):
            continue
        selected.append(counts)

    results = []
    for index, counts in enumerate(selected, start=1):
        incumbent = _validated_incumbent(catalog_rows[counts], candidate_rows.get(counts))
        target = int(incumbent["score"]) + challenge_offset
        bound = relaxed_upper_bound(
            counts,
            target,
            workers=workers,
            per_case_time_limit=per_case_time_limit,
        )
        proof_complete = (
            challenge_offset == 1
            and
            bound["status"] == "INFEASIBLE"
            and int(bound["upper_bound"]) == int(incumbent["score"])
        )
        results.append(
            {
                "counts": list(counts),
                "excluded_score": target,
                "certified_upper_bound": int(incumbent["score"]) if proof_complete else None,
                "proof_complete": proof_complete,
                "proof_method": (
                    "single_anchor_relaxed_local_packing_infeasible"
                    if proof_complete
                    else None
                ),
                "bound": bound,
                "incumbent": incumbent,
            }
        )
        print(
            f"case={index}/{len(selected)} counts={counts} target={target} "
            f"status={bound['status']} cases={bound['cases']}",
            flush=True,
        )

    source = Path(__file__).resolve()
    exact_source = source.with_name("aaaaa_wildlife_exact.py")
    motif_source = source.with_name("aaaaa_wildlife_motif_certificate.py")
    return {
        "schema": "aaaaa-single-anchor-local-packing-batch-v1",
        "proof_complete": bool(results) and all(row["proof_complete"] for row in results),
        "configuration": {
            "workers": workers,
            "per_case_time_limit_seconds": per_case_time_limit,
            "challenge_offset": challenge_offset,
            "requested_counts": (
                [list(counts) for counts in sorted(requested_counts)]
                if requested_counts is not None
                else None
            ),
        },
        "scope": {
            "selected_count": len(selected),
            "previously_proven_count": len(proven),
            "single_anchor_required": True,
        },
        "relaxation": {
            "arithmetic_score_profiles_exhaustive_through_target": True,
            "anchor_fox_ring_enumerated_modulo_dihedral_symmetry": True,
            "whole_board_connectivity_required": False,
            "noncovering_groups_may_be_abstract": True,
            "bear_isolation_dropped": True,
            "hawk_isolation_dropped": True,
            "salmon_component_separation_dropped": True,
            "forced_local_cells_must_not_overlap": True,
            "abstract_fox_coverage_optimistic": True,
        },
        "ortools_version": ORTOOLS_VERSION,
        "source_sha256": _sha256(source),
        "exact_scorer_source_sha256": _sha256(exact_source),
        "motif_support_source_sha256": _sha256(motif_source),
        "catalog": {"path": str(catalog_path), "sha256": _sha256(catalog_path)},
        "candidates": {"path": str(candidates_path), "sha256": _sha256(candidates_path)},
        "proof_ledgers": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in proof_ledgers
        ],
        "elapsed_seconds": time.monotonic() - started,
        "results": results,
    }


def _parse_counts(value: str) -> tuple[int, int, int, int, int]:
    counts = tuple(int(part) for part in value.split(","))
    if len(counts) != 5:
        raise argparse.ArgumentTypeError("counts must contain five comma-separated integers")
    return counts  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--proof-ledger", action="append", default=[], type=Path)
    parser.add_argument("--counts", action="append", type=_parse_counts)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--per-case-time-limit", type=float, default=30.0)
    parser.add_argument("--challenge-offset", type=int, default=1, choices=(0, 1))
    args = parser.parse_args()
    requested = set(args.counts) if args.counts else None
    payload = run_batch(
        args.catalog,
        args.candidates,
        tuple(args.proof_ledger),
        requested_counts=requested,
        workers=args.workers,
        per_case_time_limit=args.per_case_time_limit,
        challenge_offset=args.challenge_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "selected_count": payload["scope"]["selected_count"],
                "certified_count": sum(row["proof_complete"] for row in payload["results"]),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
