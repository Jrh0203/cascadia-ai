from __future__ import annotations

from tools.all_wildlife_candidate_catalog import _card_matrix, _score_from_matrix
from tools.all_wildlife_rules import rulesets, score_tokens
from tools.test_all_wildlife_rules import random_connected_board


def test_card_matrix_reconstructs_every_ruleset_score() -> None:
    board = random_connected_board(202)
    matrix = _card_matrix(board)
    for ruleset in rulesets():
        assert _score_from_matrix(matrix, ruleset) == score_tokens(board, ruleset)
