from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from cascadia_mlx.p1_relational_pointer_evaluate import (
    CLASSIFICATION_PIPELINE_INVALID,
    CLASSIFICATION_PROPOSAL_INSUFFICIENT,
    CLASSIFICATION_SELECTOR_INSUFFICIENT,
    CLASSIFICATION_SUFFICIENT,
    CLASSIFICATION_TILE_INSUFFICIENT,
    _SelectionAccumulator,
    classify_pointer_integration,
    load_selected_stage,
)
from cascadia_mlx.p1_relational_pointer_train import (
    EXPECTED_PARENT_PARAMETER_BLAKE3,
)

SMOKE_ROOT = Path(
    "artifacts/experiments/"
    "p1-relational-selected-prefix-pointer-pilot-v1/smoke"
)


def test_selection_accumulator_tracks_target_winner_confidence_and_regret() -> None:
    accumulator = _SelectionAccumulator()
    accumulator.add(
        retained=np.array([0, 2], dtype=np.int32),
        target=np.array([True, False, True]),
        source_flags=np.zeros(3, dtype=np.uint32),
        winner=0,
        r4800_mean=np.array([10.0, 9.0, 8.0]),
        r4800_stddev=np.array([1.0, 1.0, 1.0]),
        r4800_samples=np.array([100, 100, 100]),
        r4800_mask=np.ones(3, dtype=np.bool_),
        action_hashes=np.arange(48, dtype=np.uint8).reshape(3, 16),
    )
    report = accumulator.report()
    assert report["target_positive_recall"] == 1.0
    assert report["target_set_exact_fraction"] == 1.0
    assert report["r4800_winner_retention"] == 1.0
    assert report["top64_confidence_set_coverage_95"] == 1.0
    assert report["mean_retained_r4800_regret"] == 0.0


def test_pointer_classification_precedence_is_fail_closed() -> None:
    base = {
        "pipeline_passed": True,
        "tile_stage_passed": True,
        "proposal_passed": True,
        "selector_passed": True,
    }
    assert classify_pointer_integration(base) == CLASSIFICATION_SUFFICIENT
    assert classify_pointer_integration(
        {**base, "selector_passed": False}
    ) == CLASSIFICATION_SELECTOR_INSUFFICIENT
    assert classify_pointer_integration(
        {**base, "proposal_passed": False}
    ) == CLASSIFICATION_PROPOSAL_INSUFFICIENT
    assert classify_pointer_integration(
        {**base, "tile_stage_passed": False}
    ) == CLASSIFICATION_TILE_INSUFFICIENT
    assert classify_pointer_integration(
        {**base, "pipeline_passed": False}
    ) == CLASSIFICATION_PIPELINE_INVALID


@pytest.mark.parametrize("stage", ("draft", "tile", "wildlife"))
def test_bounded_smoke_selected_stage_reloads_exactly(stage: str) -> None:
    run_dir = SMOKE_ROOT / f"{stage}-one-batch/run"
    if not (run_dir / "final-report.json").is_file():
        pytest.skip("pointer stage smoke artifact is not installed")
    model, identity, checkpoint = load_selected_stage(
        stage=stage,
        run_dir=run_dir,
        require_production=False,
    )
    assert identity["mode"] == "bounded-smoke"
    assert checkpoint["name"].startswith("step-")
    assert (
        identity["model"]["frozen_parent_parameter_tensor_blake3"]
        == EXPECTED_PARENT_PARAMETER_BLAKE3
    )
    assert model.config.stage == stage
