from __future__ import annotations

import pytest
from conditional_tile_local_geometry_dropout_authorize import (
    EXPERIMENT_ID,
    PREFLIGHT_CLASSIFICATION,
    PREFLIGHT_EXPERIMENT_ID,
    SOURCE_EXPERIMENT_ID,
    authorize_branch,
)


def _manifest() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "contingently_authorized",
        "branch_authorization": {},
    }


def _source(classification: str, *, pipeline: bool = True) -> dict[str, object]:
    return {
        "experiment_id": SOURCE_EXPERIMENT_ID,
        "scientific_blake3": "source-hash",
        "scientific": {
            "classification": classification,
            "gates": {"pipeline_passed": pipeline},
        },
    }


def _preflight() -> dict[str, object]:
    return {
        "experiment_id": PREFLIGHT_EXPERIMENT_ID,
        "scientific_blake3": "preflight-hash",
        "scientific": {"classification": PREFLIGHT_CLASSIFICATION},
    }


def test_insufficient_source_authorizes_training() -> None:
    manifest, report = authorize_branch(
        source_combined=_source("optimizer_schedule_tile_insufficient"),
        preflight_combined=_preflight(),
        manifest=_manifest(),
        now_ms=123,
    )
    assert manifest["status"] == "authorized"
    assert manifest["branch_authorization"]["authorized_unix_ms"] == 123
    assert report["scientific"]["training_authorized"]
    assert not report["scientific"]["training_cancelled"]


def test_sufficient_source_cancels_training() -> None:
    manifest, report = authorize_branch(
        source_combined=_source("optimizer_schedule_tile_sufficient"),
        preflight_combined=_preflight(),
        manifest=_manifest(),
        now_ms=456,
    )
    assert manifest["status"] == "cancelled"
    assert manifest["branch_authorization"]["cancelled_unix_ms"] == 456
    assert report["scientific"]["training_cancelled"]
    assert not report["scientific"]["training_authorized"]


def test_invalid_source_pipeline_does_not_resolve_branch() -> None:
    with pytest.raises(ValueError, match="pipeline is invalid"):
        authorize_branch(
            source_combined=_source(
                "optimizer_schedule_pipeline_invalid",
                pipeline=False,
            ),
            preflight_combined=_preflight(),
            manifest=_manifest(),
            now_ms=789,
        )
