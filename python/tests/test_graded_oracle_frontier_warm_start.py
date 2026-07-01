from __future__ import annotations

from cascadia_mlx.graded_oracle_frontier_warm_start import (
    EXPECTED_WARM_START_CHECKPOINT,
    EXPECTED_WARM_START_MANIFEST_BLAKE3,
    EXPECTED_WARM_START_MODEL_BLAKE3,
)


def test_frontier_warm_start_identity_is_frozen() -> None:
    assert EXPECTED_WARM_START_CHECKPOINT == (
        "step-000003592-epoch-0008-batch-000000"
    )
    assert len(EXPECTED_WARM_START_MANIFEST_BLAKE3) == 64
    assert len(EXPECTED_WARM_START_MODEL_BLAKE3) == 64
