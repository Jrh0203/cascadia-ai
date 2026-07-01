from __future__ import annotations

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_local_geometry_feasibility import (
    _equivalence_summary,
    run_group,
)


def test_equivalence_summary_detects_mixed_exact_rows() -> None:
    features = np.asarray(
        [[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    )
    summary = _equivalence_summary(
        features,
        np.asarray([True, False, True]),
        np.asarray([7.0, 7.0, 8.0]),
    )
    assert summary["classes"] == 2
    assert summary["duplicate_classes"] == 1
    assert summary["mixed_target_classes"] == 1
    assert summary["exact_score_tie_conflict_classes"] == 1


def test_out_of_range_group_is_rejected_before_loading_evidence() -> None:
    with pytest.raises(ValueError, match="outside 0-3"):
        run_group(None, None, None, None, 4)  # type: ignore[arg-type]
