from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from p1_relational_pointer_foundation import (
    ADR_ID,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    _d6_pointer_checks,
    _decode_coordinates,
    _distribution,
    _pointer_key,
    canonical_blake3,
    classify_reports,
)


def _d6() -> dict:
    matrices = [
        [[1, 0], [0, 1]],
        [[0, -1], [1, 1]],
        [[-1, -1], [1, 0]],
        [[-1, 0], [0, -1]],
        [[0, 1], [-1, -1]],
        [[1, 1], [-1, 0]],
    ]
    matrices.extend(
        [
            [[1, 1], [0, -1]],
            [[0, 1], [1, 0]],
            [[-1, 0], [1, 1]],
            [[-1, -1], [0, 1]],
            [[0, -1], [-1, 0]],
            [[1, 0], [-1, -1]],
        ]
    )
    inverse = [0, 5, 4, 3, 2, 1, 6, 7, 8, 9, 10, 11]
    dual = []
    for transform_id in range(12):
        if transform_id < 6:
            dual.append([(value + transform_id) % 6 for value in range(6)])
        else:
            rotation = transform_id - 6
            dual.append([(rotation - value) % 6 for value in range(6)])
    return {
        "coordinate_matrices": matrices,
        "inverse_table": inverse,
        "dual_tile_rotation_tables": dual,
        "single_tile_rotation_tables": [[0] * 6 for _ in range(12)],
    }


def _report(split: str, host: str, *, passed: bool = True) -> dict:
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": ADR_ID,
        "kind": "complete-open-split-pointer-alignment",
        "split": split,
        "source": {
            "bundle_id": "a" * 64,
            "script_blake3": "b" * 64,
        },
        "passed": passed,
        "classification": (
            "p1_relational_pointer_foundation_passed"
            if passed
            else "p1_relational_pointer_foundation_failed"
        ),
    }
    return {
        "schema_version": 1,
        "scientific_identity": identity,
        "scientific_blake3": canonical_blake3(identity),
        "runtime": {"host": host},
    }


def test_normalized_coordinates_decode_exactly() -> None:
    values = np.asarray([[0.0, 1.0 / 24.0], [-2.0 / 24.0, 3.0 / 24.0]])
    np.testing.assert_array_equal(
        _decode_coordinates(values, "test"),
        [[0, 1], [-2, 3]],
    )
    with pytest.raises(ValueError, match="non-integral"):
        _decode_coordinates(np.asarray([[0.01, 0.0]]), "test")


def test_d6_roundtrip_checks_coordinates_and_rotation() -> None:
    checks, failures = _d6_pointer_checks(
        np.asarray([[1, 2], [-3, 1]], dtype=np.int16),
        rotations=np.asarray([4, 0]),
        dual_terrain=np.asarray([True, False]),
        d6=_d6(),
    )
    assert checks == 48
    assert failures == 0


def test_pointer_key_preserves_complete_action_identity() -> None:
    draft = bytes(range(16))
    baseline = _pointer_key(draft, (1, 2), 3, 2, (-1, 0))
    assert baseline != _pointer_key(draft, (1, 2), 4, 2, (-1, 0))
    assert baseline != _pointer_key(draft, (1, 2), 3, 1, (-1, 0))
    assert baseline != _pointer_key(draft, (1, 2), 3, 2, (-1, 1))


def test_distribution_uses_frozen_quantiles() -> None:
    assert _distribution([1, 2, 3, 4]) == {
        "count": 4,
        "minimum": 1,
        "mean": 2.5,
        "p50": 2.5,
        "p90": 3.7,
        "p99": 3.9699999999999998,
        "maximum": 4,
    }


def test_classifier_requires_cross_host_identical_split_pairs() -> None:
    result = classify_reports(
        [
            _report("train", "john2"),
            _report("train", "john4"),
            _report("validation", "john4"),
            _report("validation", "john2"),
        ]
    )
    assert (
        result["scientific_identity"]["classification"]
        == "p1_relational_pointer_foundation_passed"
    )
    assert result["scientific_identity"]["authorized_successor"] is not None


def test_classifier_rejects_same_host_replay() -> None:
    result = classify_reports(
        [
            _report("train", "john2"),
            _report("train", "john2"),
            _report("validation", "john4"),
            _report("validation", "john2"),
        ]
    )
    assert (
        result["scientific_identity"]["classification"]
        == "p1_relational_pointer_foundation_cross_host_inconsistent"
    )


def test_classifier_rejects_split_failure() -> None:
    result = classify_reports(
        [
            _report("train", "john2", passed=False),
            _report("train", "john4", passed=False),
            _report("validation", "john4"),
            _report("validation", "john2"),
        ]
    )
    assert (
        result["scientific_identity"]["classification"]
        == "p1_relational_pointer_foundation_failed"
    )


def test_classifier_rejects_scientific_replay_drift() -> None:
    drifted = _report("train", "john4")
    identity = deepcopy(drifted["scientific_identity"])
    identity["source"] = {**identity["source"], "script_blake3": "c" * 64}
    drifted["scientific_identity"] = identity
    drifted["scientific_blake3"] = canonical_blake3(identity)
    result = classify_reports(
        [
            _report("train", "john2"),
            drifted,
            _report("validation", "john4"),
            _report("validation", "john2"),
        ]
    )
    assert (
        result["scientific_identity"]["classification"]
        == "p1_relational_pointer_foundation_cross_host_inconsistent"
    )
