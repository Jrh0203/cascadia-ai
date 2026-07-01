from __future__ import annotations

import pytest
from cascadia_mlx.graded_oracle_frontier_calibrated_neural import (
    GROUPS,
    run_group,
)


def test_neural_stage_has_four_groups() -> None:
    assert GROUPS == 4


def test_out_of_range_group_is_rejected_before_loading_inputs() -> None:
    with pytest.raises(ValueError, match="outside 0-3"):
        run_group(None, None, None, 4)  # type: ignore[arg-type]
