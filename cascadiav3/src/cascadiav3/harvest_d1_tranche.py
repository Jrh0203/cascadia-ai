"""Harvest the preregistered D1 relabel tranche from a hard-roots census.

Implements the frozen sampling design (EXPERIMENT_LOG 2026-07-16 09:00):

- eligibility: ``hard == true`` census rows, exact-K1 rows excluded (the
  generation census only emits searched roots, and rows carry an explicit
  ``exact_endgame``/hardness contract when present);
- phase split 6,000 opening / 6,000 mid / 3,000 late using the repository
  phase proxy (tile count: late >= 13, mid >= 8, else opening);
- within each phase, stratification over fixed deciles of the hardness
  ratio (top-two gap / pairwise SE) computed once from the full hard pool,
  uniform sampling within cells via deterministic salted-hash order;
- correlation control: at most 12 roots per game in the first pass; a
  deterministic phase-stratified top-up may raise a game's count to at
  most 16 when a phase falls short;
- sentinel: 1,500 phase-matched non-hard roots, same machinery, disjoint
  from the tranche, never in any training arm.

Outputs probe-roots JSONL masks consumable by the exporter's
``--probe-roots`` plus a registry JSON recording strata, counts, and the
exact selection provenance. Everything is a pure function of the census
file and the frozen constants: rerunning must reproduce byte-identical
masks (the registry records a content hash for that check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRANCHE_PHASE_TARGETS = {"opening": 6000, "mid": 6000, "late": 3000}
SENTINEL_TOTAL = 1500
DECILES = 10
FIRST_PASS_PER_GAME_CAP = 12
TOP_UP_PER_GAME_CAP = 16
HASH_SALT = b"cascadia-d1-tranche-2026-07-16"
LATE_TILE_COUNT = 13
MID_TILE_COUNT = 8


@dataclass(frozen=True)
class CensusRoot:
    seed: int
    ply: int
    phase: str
    hard: bool
    gap: float
    pairwise_se: float

    @property
    def ratio(self) -> float:
        if self.pairwise_se <= 0.0:
            return float("inf")
        return self.gap / self.pairwise_se


def phase_for_tile_count(tile_count: int) -> str:
    if tile_count >= LATE_TILE_COUNT:
        return "late"
    if tile_count >= MID_TILE_COUNT:
        return "mid"
    return "opening"


def _selection_key(seed: int, ply: int) -> bytes:
    digest = hashlib.blake2b(salt=b"", person=b"", digest_size=16)
    digest.update(HASH_SALT)
    digest.update(seed.to_bytes(8, "little"))
    digest.update(ply.to_bytes(4, "little"))
    return digest.digest()


def read_census(path: Path) -> list[CensusRoot]:
    roots: list[CensusRoot] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                if row.get("type") != "hard_root":
                    raise ValueError(f"unexpected row type {row.get('type')!r}")
                if not bool(row.get("eligible", False)):
                    # <2 visited actions: no hardness measurement exists and
                    # the root is outside both pools by the frozen design.
                    continue
                tile_count = row.get("tile_count")
                if tile_count is None:
                    # Generation census fallback: ply-derived proxy used by
                    # analyze_label_movement (3 + ply // 4 tiles placed).
                    tile_count = 3 + int(row["ply"]) // 4
                roots.append(
                    CensusRoot(
                        seed=int(row["seed"]),
                        ply=int(row["ply"]),
                        phase=phase_for_tile_count(int(tile_count)),
                        hard=bool(row["hard"]),
                        gap=float(row["top1_top2_gap"]),
                        pairwise_se=float(row["pairwise_se"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: bad census row: {error}") from error
    if not roots:
        raise ValueError(f"{path}: census is empty")
    return roots


def _decile_edges(ratios: list[float]) -> list[float]:
    finite = sorted(ratio for ratio in ratios if ratio != float("inf"))
    if not finite:
        return []
    edges = []
    for decile in range(1, DECILES):
        position = decile * len(finite) // DECILES
        edges.append(finite[min(position, len(finite) - 1)])
    return edges


def _decile_for(ratio: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if ratio <= edge:
            return index
    return len(edges)


def _stratified_select(
    pool: list[CensusRoot],
    phase_targets: dict[str, int],
    per_game_counts: dict[int, int],
    first_pass_cap: int,
    top_up_cap: int,
    edges: list[float],
) -> tuple[list[CensusRoot], dict[str, Any]]:
    """Deterministic stratified sampling honoring per-game caps.

    First pass fills each (phase, decile) cell round-robin in salted-hash
    order under the first-pass cap; the top-up pass relaxes only the cap
    (never the phase quota) in the same deterministic order.
    """
    by_cell: dict[tuple[str, int], list[CensusRoot]] = defaultdict(list)
    for root in pool:
        by_cell[(root.phase, _decile_for(root.ratio, edges))].append(root)
    for cell in by_cell.values():
        cell.sort(key=lambda root: _selection_key(root.seed, root.ply))

    # Phases fill INTERLEAVED (one cell visit per phase per sweep), never
    # sequentially: the per-game budget is shared across phases, and a
    # sequential fill would let earlier phases starve later ones of games.
    phases = sorted(phase_targets)
    chosen_by_phase: dict[str, list[CensusRoot]] = {phase: [] for phase in phases}
    taken: set[tuple[int, int]] = set()

    def fill_pass(cap: int) -> None:
        cursors = {
            (phase, decile): 0 for phase in phases for decile in range(DECILES)
        }
        progress = True
        while progress:
            progress = False
            for decile in range(DECILES):
                for phase in phases:
                    if len(chosen_by_phase[phase]) >= phase_targets[phase]:
                        continue
                    cell = by_cell.get((phase, decile), [])
                    cursor = cursors[(phase, decile)]
                    while cursor < len(cell):
                        candidate = cell[cursor]
                        cursor += 1
                        key = (candidate.seed, candidate.ply)
                        if key in taken:
                            continue
                        if per_game_counts[candidate.seed] < cap:
                            per_game_counts[candidate.seed] += 1
                            taken.add(key)
                            chosen_by_phase[phase].append(candidate)
                            progress = True
                            break
                    cursors[(phase, decile)] = cursor

    fill_pass(first_pass_cap)
    if any(
        len(chosen_by_phase[phase]) < phase_targets[phase] for phase in phases
    ):
        fill_pass(top_up_cap)

    selected: list[CensusRoot] = []
    shortfall: dict[str, int] = {}
    cell_counts: dict[str, dict[int, int]] = defaultdict(dict)
    for phase in phases:
        chosen = chosen_by_phase[phase]
        if len(chosen) < phase_targets[phase]:
            shortfall[phase] = phase_targets[phase] - len(chosen)
        for root in chosen:
            decile = _decile_for(root.ratio, edges)
            cell_counts[phase][decile] = cell_counts[phase].get(decile, 0) + 1
        selected.extend(chosen)

    provenance = {
        "phase_targets": phase_targets,
        "selected": len(selected),
        "shortfall": shortfall,
        "cell_counts": {
            phase: {str(decile): count for decile, count in sorted(counts.items())}
            for phase, counts in cell_counts.items()
        },
    }
    return selected, provenance


def harvest(census_path: Path, output_dir: Path) -> dict[str, Any]:
    roots = read_census(census_path)
    hard_pool = [root for root in roots if root.hard]
    soft_pool = [root for root in roots if not root.hard]
    if not hard_pool:
        raise ValueError("census contains no hard roots")
    edges = _decile_edges([root.ratio for root in hard_pool])

    per_game_counts: dict[int, int] = defaultdict(int)
    tranche, tranche_provenance = _stratified_select(
        hard_pool,
        TRANCHE_PHASE_TARGETS,
        per_game_counts,
        FIRST_PASS_PER_GAME_CAP,
        TOP_UP_PER_GAME_CAP,
        edges,
    )

    # Sentinel: phase-matched to the TRANCHE's realized phase mix, drawn
    # from the non-hard pool with its own per-game budget (the tranche's
    # caps must not starve the descriptive sentinel).
    tranche_phase_mix = defaultdict(int)
    for root in tranche:
        tranche_phase_mix[root.phase] += 1
    sentinel_targets = {
        phase: round(SENTINEL_TOTAL * count / max(1, len(tranche)))
        for phase, count in sorted(tranche_phase_mix.items())
    }
    sentinel_game_counts: dict[int, int] = defaultdict(int)
    sentinel, sentinel_provenance = _stratified_select(
        soft_pool,
        sentinel_targets,
        sentinel_game_counts,
        FIRST_PASS_PER_GAME_CAP,
        TOP_UP_PER_GAME_CAP,
        edges,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tranche_path = output_dir / "d1_tranche_probe_roots.jsonl"
    sentinel_path = output_dir / "d1_sentinel_probe_roots.jsonl"
    for path, selection in ((tranche_path, tranche), (sentinel_path, sentinel)):
        with path.open("w") as handle:
            for root in sorted(selection, key=lambda root: (root.seed, root.ply)):
                handle.write(json.dumps({"seed": root.seed, "ply": root.ply}) + "\n")

    registry = {
        "type": "d1_tranche_registry",
        "census": str(census_path),
        "census_rows": len(roots),
        "hard_pool": len(hard_pool),
        "non_hard_pool": len(soft_pool),
        "decile_edges": edges,
        "constants": {
            "phase_targets": TRANCHE_PHASE_TARGETS,
            "sentinel_total": SENTINEL_TOTAL,
            "first_pass_per_game_cap": FIRST_PASS_PER_GAME_CAP,
            "top_up_per_game_cap": TOP_UP_PER_GAME_CAP,
            "hash_salt": HASH_SALT.decode(),
            "phase_proxy": "tile_count (late >= 13, mid >= 8)",
        },
        "tranche": tranche_provenance,
        "sentinel": {**sentinel_provenance, "targets": sentinel_targets},
        "tranche_mask_sha256": hashlib.sha256(tranche_path.read_bytes()).hexdigest(),
        "sentinel_mask_sha256": hashlib.sha256(sentinel_path.read_bytes()).hexdigest(),
    }
    registry_path = output_dir / "d1_tranche_registry.json"
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, required=True, help="hard-roots census JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    registry = harvest(args.census, args.output_dir)
    print(json.dumps({key: registry[key] for key in ("hard_pool", "non_hard_pool")}, indent=2))
    print(f"tranche: {registry['tranche']['selected']} roots; sentinel: {registry['sentinel']['selected']}")


if __name__ == "__main__":
    main()
