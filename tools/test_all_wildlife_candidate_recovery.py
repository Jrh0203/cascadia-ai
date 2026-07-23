import pytest

from tools.all_wildlife_candidate_recovery import _validate_board
from tools.test_all_wildlife_rules import random_connected_board


def test_validate_board_accepts_independently_scored_board() -> None:
    tokens = random_connected_board(303)
    from tools import all_wildlife_rules as rules

    breakdown = rules.score_tokens(tokens, "CBDDB")
    counts = [
        sum(token["wildlife"] == species for token in tokens)
        for species in rules.SPECIES
    ]
    row = {
        "tokens": tokens,
        "counts": counts,
        "score_breakdown": list(breakdown),
        "score": sum(breakdown),
    }
    assert _validate_board(row, "CBDDB") == rules.normalized_tokens(tokens)


def test_validate_board_rejects_score_mismatch() -> None:
    tokens = random_connected_board(404)
    row = {
        "tokens": tokens,
        "counts": [4, 4, 4, 4, 4],
        "score_breakdown": [0, 0, 0, 0, 0],
        "score": 0,
    }
    with pytest.raises(ValueError, match="independent score mismatch"):
        _validate_board(row, "AAAAA")
