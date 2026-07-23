import random

from tools import aaaaa_wildlife_exact as aaaaa
from tools import all_wildlife_rules as rules
from tools import cbddb_wildlife_exact as cbddb


def random_connected_board(seed: int) -> list[dict[str, int | str]]:
    return random_connected_board_with_counts(seed, (4, 4, 4, 4, 4))


def random_connected_board_with_counts(
    seed: int,
    counts: tuple[int, int, int, int, int],
) -> list[dict[str, int | str]]:
    rng = random.Random(seed)
    occupied = {(0, 0)}
    while len(occupied) < 20:
        frontier = {
            other
            for coord in occupied
            for other in rules.neighbors(coord)
            if other not in occupied
        }
        occupied.add(rng.choice(sorted(frontier)))
    wildlife = [
        species
        for species, count in zip(rules.SPECIES, counts, strict=True)
        for _ in range(count)
    ]
    rng.shuffle(wildlife)
    return [
        {"q": q, "r": r, "wildlife": species}
        for (q, r), species in zip(sorted(occupied), wildlife, strict=True)
    ]


def test_ruleset_and_count_spaces_are_complete() -> None:
    assert len(rules.rulesets()) == 1024
    assert rules.rulesets()[0] == "AAAAA"
    assert rules.rulesets()[-1] == "DDDDD"
    assert len(set(rules.rulesets())) == 1024
    assert len(rules.count_vectors()) == 826


def test_aaaaa_scorer_and_bound_match_existing_implementation() -> None:
    for seed in range(40):
        board = random_connected_board(seed)
        assert rules.score_tokens(board, "AAAAA") == aaaaa.score_tokens(board)
    for counts, expected in aaaaa.count_vectors():
        assert rules.count_upper(counts, "AAAAA") <= expected


def test_cbddb_scorer_and_bound_match_existing_implementation() -> None:
    for seed in range(40, 80):
        board = random_connected_board(seed)
        assert rules.score_tokens(board, "CBDDB") == cbddb.score_tokens(board)
    for counts, expected in cbddb.count_vectors():
        assert rules.count_upper(counts, "CBDDB") <= expected


def test_all_rulesets_have_sound_nonnegative_count_bounds() -> None:
    representative_counts = (
        (0, 2, 6, 6, 6),
        (2, 6, 6, 0, 6),
        (4, 4, 4, 4, 4),
        (6, 6, 6, 2, 0),
    )
    for ruleset in rules.rulesets():
        upper, maximizing = rules.global_count_upper(ruleset)
        assert upper >= 0
        assert maximizing
        for counts in maximizing:
            assert counts in rules.count_vectors()
        for counts in representative_counts:
            assert rules.count_upper(counts, ruleset) >= 0


def test_known_boards_score_identically_under_mixed_cards() -> None:
    board = random_connected_board(20260723)
    component_scores = {
        species: {
            variant: rules.score_tokens(
                board,
                "".join(
                    variant if index == species else "A"
                    for index in range(len(rules.SPECIES))
                ),
            )[species]
            for variant in rules.VARIANTS
        }
        for species in range(len(rules.SPECIES))
    }
    for ruleset in ("ABCDD", "DCBAC", "BADCD", "CDDAB"):
        score = rules.score_tokens(board, ruleset)
        assert score == tuple(
            component_scores[species][ruleset[species]]
            for species in range(len(rules.SPECIES))
        )


def test_hawk_c_uses_the_tight_cap_six_visibility_bound() -> None:
    expected = (0, 0, 3, 9, 15, 21, 27)
    hawk_a = (0, 2, 5, 8, 11, 14, 18)
    for hawks, hawk_score in enumerate(expected):
        counts = (6, 6, 6 - hawks, hawks, 2)
        a_upper = rules.count_upper(counts, "AAAAA")
        c_upper = rules.count_upper(counts, "AAACA")
        assert c_upper - a_upper + hawk_a[hawks] == hawk_score


def test_fox_c_uses_exact_bipartite_hex_edge_bound() -> None:
    assert rules._fox_c_upper(6, (4, 4, 4, 2)) <= 24
    assert rules._fox_c_upper(6, (6, 4, 2, 2)) <= 24


def test_exact_bipartite_hex_edge_table_is_symmetric_and_tighter_than_planarity() -> None:
    for left in range(rules.COUNT_CAP + 1):
        for right in range(rules.COUNT_CAP + 1):
            exact = rules._bipartite_hex_edge_upper(left, right)
            assert exact == rules._bipartite_hex_edge_upper(right, left)
            if not left or not right:
                assert exact == 0
                continue
            planar = 1 if left + right == 2 else 2 * (left + right) - 4
            assert exact <= min(left * right, 6 * left, 6 * right, planar)
    assert rules._bipartite_hex_edge_upper(6, 6) == 17


def test_fox_b_uses_target_pair_common_neighbor_capacity() -> None:
    assert rules._fox_b_upper(6, (6, 4, 2, 2)) <= 38
    assert rules._fox_b_upper(6, (4, 4, 4, 2)) <= 42


def test_fox_a_uses_common_neighbor_overlap_capacities() -> None:
    assert rules._fox_a_upper(6, (6, 6, 1, 1)) == 26
    assert rules._fox_a_upper(6, (6, 4, 2, 2)) == 30


def test_every_count_bound_dominates_frozen_board_scores() -> None:
    for seed in (101, 202, 303, 404):
        board = random_connected_board(seed)
        counts = (4, 4, 4, 4, 4)
        for ruleset in rules.rulesets():
            assert sum(rules.score_tokens(board, ruleset)) <= rules.count_upper(
                counts, ruleset
            )


def test_coupled_edge_bound_is_sound_on_mixed_count_boards() -> None:
    count_vectors = (
        (0, 2, 6, 6, 6),
        (2, 6, 6, 0, 6),
        (4, 4, 4, 4, 4),
        (6, 6, 6, 2, 0),
    )
    for seed, counts in enumerate(count_vectors, start=700):
        board = random_connected_board_with_counts(seed, counts)
        for ruleset in rules.rulesets():
            coupled = rules._coupled_count_upper(counts, ruleset)
            assert coupled <= rules.count_upper(counts, ruleset)
            assert sum(rules.score_tokens(board, ruleset)) <= coupled


def test_coupled_edge_flow_handles_zero_and_impossible_demands() -> None:
    counts = (4, 4, 4, 4, 4)
    assert rules._cross_demands_feasible(counts, (0, 0, 0, 0, 0), 0, 0)
    assert not rules._cross_demands_feasible(
        counts,
        (0, 0, 0, 0, 0),
        25,
        0,
    )
    assert not rules._cross_demands_feasible(
        counts,
        (0, 0, 0, 0, 0),
        0,
        25,
    )


def test_score_domains_contain_mixed_count_board_components() -> None:
    count_vectors = (
        (0, 2, 6, 6, 6),
        (2, 6, 6, 0, 6),
        (4, 4, 4, 4, 4),
        (6, 6, 6, 2, 0),
    )
    for seed, counts in enumerate(count_vectors, start=900):
        board = random_connected_board_with_counts(seed, counts)
        for ruleset in rules.rulesets():
            score = rules.score_tokens(board, ruleset)
            domains = rules.count_score_domains(counts, ruleset)
            assert all(
                component in domain
                for component, domain in zip(score, domains, strict=True)
            )
            assert score in rules.count_score_profiles(
                counts,
                ruleset,
                sum(score),
                sum(score),
            )
