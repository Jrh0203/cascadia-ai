from __future__ import annotations

from copy import deepcopy

import numpy as np
from exact_r2_vectorized_materialization import (
    EXPERIMENT_ID,
    LEGACY,
    PROTOCOL_ID,
    VECTORIZED,
    _array_digest,
    _latency_summary,
    _report,
    _source_identity,
    classify_reports,
)


def _comparison(
    order: str,
    *,
    split: str = "validation",
    host: str | None = None,
) -> dict:
    groups, actions = {
        "train": (560, 280_012),
        "validation": (240, 860_203),
    }[split]
    identity = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "adr": "0167",
        "kind": "complete-materialization-parity-and-performance",
        "split": split,
        "order": order,
        "complete_split": True,
        "groups_compared": groups,
        "actions_compared": actions,
        "parity": {"pass": True},
        "performance": {
            LEGACY: {"latency_milliseconds": {"p99": 4100.0}},
            VECTORIZED: {"latency_milliseconds": {"p99": 200.0}},
            "p99_speedup": 20.5,
        },
        "memory": {
            "peak_process_rss_bytes": 1024,
            "system_swap_delta_bytes": 0,
        },
        "open_data_verification_id": "proof",
        "source": _source_identity(),
        "runtime": {
            "host": host
            or ("john1" if order == "legacy-first" else "john2"),
        },
    }
    return _report(identity)


def _prediction(*, host: str = "john3") -> dict:
    return _report(
        {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "protocol_id": PROTOCOL_ID,
            "adr": "0167",
            "kind": "frozen-c0-prediction-parity",
            "complete_validation": True,
            "actions_compared": 860_203,
            "parity": {"pass": True},
            "memory": {
                "peak_process_rss_bytes": 1024,
                "system_swap_delta_bytes": 0,
            },
            "open_data_verification_id": "proof",
            "source": _source_identity(),
            "runtime": {"host": host},
        }
    )


def _comparisons() -> list[dict]:
    return [
        _comparison("legacy-first", host="john1"),
        _comparison("vectorized-first", host="john2"),
        _comparison("legacy-first", split="train", host="john4"),
    ]


def test_latency_summary_uses_frozen_quantiles() -> None:
    summary = _latency_summary([100.0, 200.0, 300.0, 400.0])
    assert summary == {
        "mean": 250.0,
        "p50": 250.0,
        "p95": 384.99999999999994,
        "p99": 397.0,
        "maximum": 400.0,
    }


def test_array_digest_binds_label_dtype_shape_and_values() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert _array_digest("x", values) == _array_digest(
        "x",
        np.asfortranarray(values),
    )
    assert _array_digest("x", values) != _array_digest(
        "y",
        values,
    )
    assert _array_digest("x", values) != _array_digest(
        "x",
        values.astype(np.float64),
    )
    assert _array_digest("x", values) != _array_digest(
        "x",
        values.reshape(2, 6),
    )
    changed = values.copy()
    changed[-1, -1] += 1
    assert _array_digest("x", values) != _array_digest(
        "x",
        changed,
    )


def test_classifier_promotes_two_crossed_exact_reports() -> None:
    report = classify_reports(
        _comparisons(),
        _prediction(),
    )
    identity = report["scientific_identity"]
    assert identity["promoted"] is True
    assert identity["classification"] == "exact_r2_vectorized_materialization_promoted"


def test_classifier_rejects_speed_failure() -> None:
    slow = _comparison("vectorized-first")
    identity = deepcopy(slow["scientific_identity"])
    identity["performance"]["p99_speedup"] = 9.99
    slow = _report(identity)
    report = classify_reports(
        [
            _comparison("legacy-first", host="john1"),
            slow,
            _comparison("legacy-first", split="train", host="john4"),
        ],
        _prediction(),
    )
    assert (
        report["scientific_identity"]["classification"]
        == "exact_r2_vectorized_materialization_speed_failure"
    )


def test_classifier_reports_parity_failure_separately_from_structure() -> None:
    failed = _comparison("vectorized-first")
    identity = deepcopy(failed["scientific_identity"])
    identity["parity"]["pass"] = False
    failed = _report(identity)
    report = classify_reports(
        [
            _comparison("legacy-first", host="john1"),
            failed,
            _comparison("legacy-first", split="train", host="john4"),
        ],
        _prediction(),
    )
    scientific = report["scientific_identity"]
    assert scientific["gates"]["structural"] is True
    assert scientific["gates"]["feature_and_prediction_parity"] is False
    assert (
        scientific["classification"]
        == "exact_r2_vectorized_materialization_parity_failure"
    )


def test_classifier_rejects_incomplete_prediction_as_structural() -> None:
    prediction = _prediction()
    identity = deepcopy(prediction["scientific_identity"])
    identity["complete_validation"] = False
    prediction = _report(identity)
    report = classify_reports(
        _comparisons(),
        prediction,
    )
    scientific = report["scientific_identity"]
    assert scientific["gates"]["structural"] is False
    assert (
        scientific["classification"]
        == "exact_r2_vectorized_materialization_structurally_invalid"
    )


def test_classifier_requires_complete_train_report() -> None:
    report = classify_reports(
        [
            _comparison("legacy-first", host="john1"),
            _comparison("vectorized-first", host="john2"),
            _comparison("legacy-first", split="validation", host="john4"),
        ],
        _prediction(),
    )
    assert (
        report["scientific_identity"]["classification"]
        == "exact_r2_vectorized_materialization_structurally_invalid"
    )


def test_classifier_rejects_source_drift_as_cross_host_inconsistent() -> None:
    drifted = _comparison("vectorized-first", host="john2")
    identity = deepcopy(drifted["scientific_identity"])
    identity["source"] = {**identity["source"], "unexpected.py": "0" * 64}
    drifted = _report(identity)
    report = classify_reports(
        [
            _comparison("legacy-first", host="john1"),
            drifted,
            _comparison("legacy-first", split="train", host="john4"),
        ],
        _prediction(),
    )
    scientific = report["scientific_identity"]
    assert scientific["gates"]["structural"] is True
    assert scientific["gates"]["cross_host_consistent"] is False
    assert (
        scientific["classification"]
        == "exact_r2_vectorized_materialization_cross_host_inconsistent"
    )


def test_classifier_requires_four_distinct_runtime_hosts() -> None:
    report = classify_reports(
        _comparisons(),
        _prediction(host="john1"),
    )
    assert (
        report["scientific_identity"]["classification"]
        == "exact_r2_vectorized_materialization_cross_host_inconsistent"
    )
