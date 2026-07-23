from pathlib import Path

import pytest

from tools.aaaaa_wildlife_split_salmon_dp_collect import _validate_candidate


def test_candidate_validation_rejects_disconnected_board() -> None:
    tokens = [
        {"q": index * 2, "r": 0, "wildlife": species}
        for index, species in enumerate(
            ["bear"] * 4 + ["elk"] * 5 + ["salmon"] * 2 + ["hawk"] * 3 + ["fox"] * 6
        )
    ]
    row = {"tokens": tokens, "score_breakdown": [0, 0, 0, 0, 0], "score": 0}
    with pytest.raises(ValueError, match="disconnected"):
        _validate_candidate(row, (4, 5, 2, 3, 6), 0)


def test_production_oracle_exists() -> None:
    assert Path("target/release/all_wildlife_score_oracle").is_file()
