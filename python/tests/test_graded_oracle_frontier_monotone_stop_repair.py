from __future__ import annotations

import pytest
from cascadia_mlx.graded_oracle_frontier_monotone_stop_repair import (
    FROZEN_GROUPS,
    REPAIR_GROUPS,
    run_repair_group,
)


def test_repair_and_frozen_groups_partition_24_groups() -> None:
    assert set(REPAIR_GROUPS).isdisjoint(FROZEN_GROUPS)
    assert set(REPAIR_GROUPS) | set(FROZEN_GROUPS) == set(range(24))


def test_nonrepair_group_is_rejected_before_loading_inputs() -> None:
    with pytest.raises(ValueError, match="repair set"):
        run_repair_group(None, None, None, 1)  # type: ignore[arg-type]
