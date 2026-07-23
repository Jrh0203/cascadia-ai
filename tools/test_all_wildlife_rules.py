import random

from tools import aaaaa_wildlife_exact as aaaaa
from tools import all_wildlife_rules as rules
from tools import cbddb_wildlife_exact as cbddb


def random_connected_board(seed: int) -> list[dict[str, int | str]]:
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
    wildlife = [species for species in rules.SPECIES for _ in range(4)]
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
        assert rules.count_upper(counts, "AAAAA") == expected


def test_cbddb_scorer_and_bound_match_existing_implementation() -> None:
    for seed in range(40, 80):
        board = random_connected_board(seed)
        assert rules.score_tokens(board, "CBDDB") == cbddb.score_tokens(board)
    for counts, expected in cbddb.count_vectors():
        assert rules.count_upper(counts, "CBDDB") == expected


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
