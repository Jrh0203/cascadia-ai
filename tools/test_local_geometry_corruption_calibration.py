from __future__ import annotations

import numpy as np
from local_geometry_corruption_calibration import (
    EXPERIMENT_ID,
    RATES,
    combine,
    corrupt_local_geometry,
)


def _arm(rate: float, feasible: bool, pipeline: bool = True) -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "scientific": {
            "rate": rate,
            "feasible": feasible,
            "pipeline_passed": pipeline,
        },
    }


def test_corruption_changes_only_local_geometry_for_selected_items() -> None:
    width = 10
    items = np.arange(width * 249, dtype=np.float32).reshape(width, 249)
    hashes = np.arange(width * 16, dtype=np.uint8).reshape(width, 16)
    arrays = {
        "tile_item_features": items,
        "tile_item_hash": hashes,
        "tile_query_offsets": np.asarray([0, width], dtype=np.int64),
    }
    changed = corrupt_local_geometry(arrays, rate=0.25)
    assert np.array_equal(changed["tile_item_features"][:, :8], items[:, :8])
    assert np.array_equal(changed["tile_item_features"][:, 188:], items[:, 188:])
    assert not np.array_equal(
        changed["tile_item_features"][:, 8:188],
        items[:, 8:188],
    )


def test_smallest_feasible_rate_is_selected() -> None:
    report = combine(
        [
            _arm(0.10, False),
            _arm(0.25, True),
            _arm(0.50, True),
        ]
    )
    assert report["scientific"]["classification"] == (
        "local_geometry_corruption_calibrated"
    )
    assert report["scientific"]["selected_rate"] == 0.25


def test_no_feasible_rate_is_reported() -> None:
    report = combine([_arm(rate, False) for rate in RATES])
    assert (
        report["scientific"]["classification"]
        == "local_geometry_corruption_not_calibrated"
    )
    assert report["scientific"]["selected_rate"] is None


def test_invalid_arm_invalidates_calibration() -> None:
    report = combine(
        [
            _arm(0.10, True),
            _arm(0.25, False, pipeline=False),
            _arm(0.50, False),
        ]
    )
    assert (
        report["scientific"]["classification"]
        == "local_geometry_corruption_pipeline_invalid"
    )
