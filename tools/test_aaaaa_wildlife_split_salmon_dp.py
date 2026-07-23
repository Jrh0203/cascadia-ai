from __future__ import annotations

import itertools
import random

from tools.aaaaa_wildlife_gap_two_salmon_pair_bound import split_singleton_shapes
from tools.aaaaa_wildlife_motif_certificate import adjacent
from tools.aaaaa_wildlife_split_salmon_dp import (
    _cover_configurations,
    _minimal_masks,
    _pack_disjoint,
    _self_observations_hold,
    fox_layouts,
)


def test_minimal_masks_remove_only_occupancy_supersets() -> None:
    assert _minimal_masks([0b0011, 0b0111, 0b1010, 0b1010]) == (0b0011, 0b1010)


def test_subset_index_matches_naive_dominance() -> None:
    generator = random.Random(20260723)
    for _ in range(100):
        masks = [generator.randrange(1 << 12) for _ in range(40)]
        expected = []
        for mask in sorted(set(masks), key=lambda value: (value.bit_count(), value)):
            if not any(previous & mask == previous for previous in expected):
                expected.append(mask)
        assert _minimal_masks(masks) == tuple(expected)


def test_disjoint_packing_finds_and_rejects_exactly() -> None:
    feasible, chosen, _ = _pack_disjoint([(0b001, 0b010), (0b100,)])
    assert feasible
    assert chosen[0] & chosen[1] == 0
    assert not _pack_disjoint([(0b001,), (0b001,)])[0]


def test_cover_enumerator_matches_small_exhaustive_set_cover() -> None:
    foxes = ((0, 0), (2, 0))
    placements = (
        frozenset({(1, 0)}),
        frozenset({(-1, 0)}),
        frozenset({(3, 0)}),
    )
    cells = sorted({cell for shape in placements for cell in shape})
    index = {cell: offset for offset, cell in enumerate(cells)}
    observed = _cover_configurations(foxes, foxes, ((placements, 2),), index)
    exhaustive = set()
    for count in range(3):
        for chosen in itertools.combinations(placements, count):
            occupied = set().union(*chosen) if chosen else set()
            if len(occupied) != sum(map(len, chosen)):
                continue
            if all(any(adjacent(fox, cell) for cell in occupied) for fox in foxes):
                exhaustive.add(sum(1 << index[cell] for cell in occupied))
    assert observed == _minimal_masks(exhaustive)


def test_split_fox_layouts_have_exact_local_outer_counts() -> None:
    salmon = split_singleton_shapes()[0]
    local = {
        cell
        for cell in set().union(
            *(
                {
                    (q + dq, r + dr)
                    for dq, dr in ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
                }
                for q, r in salmon
            )
        )
        if cell not in salmon
    }
    assert len(fox_layouts(salmon, 0)) == 924
    assert len(fox_layouts(salmon, 1)) == 19_008
    assert all(sum(cell in local for cell in row) == 6 for row in fox_layouts(salmon, 0))
    assert all(sum(cell in local for cell in row) == 5 for row in fox_layouts(salmon, 1))


def test_self_observation_allows_only_registered_deficit() -> None:
    foxes = ((0, 0), (1, 0), (5, 0))
    assert not _self_observations_hold(foxes, None)
    assert _self_observations_hold(foxes, (5, 0))
