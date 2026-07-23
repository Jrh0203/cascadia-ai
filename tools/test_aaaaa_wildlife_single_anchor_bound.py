from tools import aaaaa_wildlife_exact as base
from tools.aaaaa_wildlife_single_anchor_bound import (
    SpeciesStructure,
    _oriented_shapes,
    anchor_score,
    choose_anchor,
    local_placements,
    ring_layouts,
    score_profiles,
    species_structures,
)


def test_ring_layouts_are_complete_modulo_dihedral_symmetry() -> None:
    assert [len(ring_layouts(size)) for size in range(7)] == [1, 1, 3, 3, 3, 1, 1]
    assert sum(len(row) for row in ring_layouts(6)) == 6


def test_unique_species_anchor_scores_are_forced() -> None:
    assert anchor_score(base.SPECIES_CODE["bear"]) == 0
    assert anchor_score(base.SPECIES_CODE["elk"]) == 2
    assert anchor_score(base.SPECIES_CODE["salmon"]) == 2
    assert anchor_score(base.SPECIES_CODE["hawk"]) == 2
    assert choose_anchor((1, 6, 6, 1, 6)) == base.SPECIES_CODE["bear"]


def test_near_maximum_structures_include_scoring_and_leftover_tokens() -> None:
    bears = species_structures(base.SPECIES_CODE["bear"], 5, 7)
    assert SpeciesStructure(11, (2, 2, 1)) in bears
    assert SpeciesStructure(4, (2, 1, 1, 1)) in bears

    salmon = species_structures(base.SPECIES_CODE["salmon"], 4, 12)
    assert SpeciesStructure(12, (4,)) in salmon
    assert SpeciesStructure(5, (2, 1, 1)) in salmon
    assert SpeciesStructure(0, (1, 1, 1, 1)) in salmon


def test_score_profile_filter_keeps_the_standalone_maximum() -> None:
    counts = (1, 6, 6, 1, 6)
    profiles = score_profiles(counts, base.count_relaxation(counts), choose_anchor(counts))
    assert len(profiles) == 2
    expected = sum(
        base.STANDALONE_SCORES[species][counts[species]]
        for species in range(4)
    )
    assert all(profile.nonfox_score == expected for profile in profiles)


def test_local_shape_generation_covers_all_card_motif_families() -> None:
    foxes = ring_layouts(3)[0]
    for species, size in (
        (base.SPECIES_CODE["bear"], 2),
        (base.SPECIES_CODE["elk"], 4),
        (base.SPECIES_CODE["salmon"], 6),
        (base.SPECIES_CODE["hawk"], 1),
    ):
        assert _oriented_shapes(species, size)
        placements = local_placements(foxes, species, size)
        assert placements
        assert all(shape.isdisjoint({(0, 0), *foxes}) for shape in placements)
