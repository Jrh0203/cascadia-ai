from __future__ import annotations

from local_geometry_dropout_preflight import (
    ARMS,
    EXPERIMENT_ID,
    FROZEN_SELECTION_BLAKE3,
    PREFLIGHT_ID,
    classify_preflight,
)


def _arm(name: str, *, passed: bool = True) -> dict[str, object]:
    scientific: dict[str, object] = {
        "passed": passed,
        "train_cache_payload_blake3": "cache",
        "validation_opened": False,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    if name in ("contract", "coverage"):
        scientific["epoch_one_selection_blake3"] = FROZEN_SELECTION_BLAKE3
    return {
        "experiment_id": PREFLIGHT_ID,
        "treatment_experiment_id": EXPERIMENT_ID,
        "arm": name,
        "scientific": scientific,
    }


def test_complete_preflight_passes() -> None:
    classification, gates = classify_preflight([_arm(name) for name in ARMS])
    assert classification == "local_geometry_dropout_preflight_passed"
    assert all(gates.values())


def test_failed_arm_invalidates_preflight() -> None:
    arms = [_arm(name, passed=name != "gradient") for name in ARMS]
    classification, gates = classify_preflight(arms)
    assert classification == "local_geometry_dropout_preflight_invalid"
    assert not gates["all_arms_passed"]


def test_selection_mismatch_invalidates_preflight() -> None:
    arms = [_arm(name) for name in ARMS]
    arms[1]["scientific"]["epoch_one_selection_blake3"] = "different"
    classification, gates = classify_preflight(arms)
    assert classification == "local_geometry_dropout_preflight_invalid"
    assert not gates["cross_host_epoch_one_selection_exact"]
