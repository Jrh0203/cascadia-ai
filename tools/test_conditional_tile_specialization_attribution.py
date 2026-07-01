from __future__ import annotations

import numpy as np
from conditional_tile_specialization_attribution import (
    BLOCKS,
    EXPERIMENT_ID,
    combine,
    permute_block,
)


def _arm(block: str, contribution: float, pipeline: bool = True) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "arm": block,
        "scientific": {
            "pipeline_passed": pipeline,
            "specialization_contribution": contribution,
        },
    }


def test_permutation_changes_only_selected_columns() -> None:
    items = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    arrays = {
        "tile_item_features": items,
        "tile_query_offsets": np.asarray([0, 4], dtype=np.int64),
    }
    changed = permute_block(
        arrays,
        left_column=1,
        right_column=3,
        query_base=0,
    )
    assert np.array_equal(changed["tile_item_features"][:, 0], items[:, 0])
    assert np.array_equal(changed["tile_item_features"][:, 3:], items[:, 3:])
    assert not np.array_equal(
        changed["tile_item_features"][:, 1:3],
        items[:, 1:3],
    )


def test_clear_winner_identifies_targeted_block() -> None:
    report = combine(
        [
            _arm("tile_factor", 0.01),
            _arm("local_geometry", 0.09),
            _arm("descendant_summary", 0.04),
        ]
    )
    assert report["scientific"]["classification"] == "specialization_block_identified"
    assert report["scientific"]["selected_block"] == "local_geometry"


def test_close_contributions_are_distributed() -> None:
    report = combine(
        [
            _arm("tile_factor", 0.06),
            _arm("local_geometry", 0.05),
            _arm("descendant_summary", 0.04),
        ]
    )
    assert (
        report["scientific"]["classification"]
        == "specialization_distributed_across_blocks"
    )
    assert report["scientific"]["selected_block"] is None


def test_invalid_arm_invalidates_combined_result() -> None:
    arms = [
        _arm("tile_factor", 0.10),
        _arm("local_geometry", 0.01, pipeline=False),
        _arm("descendant_summary", 0.00),
    ]
    report = combine(arms)
    assert (
        report["scientific"]["classification"]
        == "specialization_attribution_pipeline_invalid"
    )
    assert set(BLOCKS) == {arm["arm"] for arm in arms}
