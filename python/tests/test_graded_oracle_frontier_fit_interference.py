from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from cascadia_mlx.graded_oracle_frontier_fit_interference import (
    CohortGroup,
    _arm_pipeline_passed,
    _cosine_report,
    classify_fit_interference,
    cohort_digest,
    width_bucket,
)


def test_width_bucket_boundaries() -> None:
    assert width_bucket(1) == "at_most_2048"
    assert width_bucket(2048) == "at_most_2048"
    assert width_bucket(2049) == "2049_to_4096"
    assert width_bucket(4096) == "2049_to_4096"
    assert width_bucket(4097) == "above_4096"


def test_cohort_digest_is_stable_and_order_sensitive() -> None:
    first = CohortGroup(11, 0, 1024, "at_most_2048", 0, 0)
    second = CohortGroup(12, 2, 5000, "above_4096", 1, 3)
    assert cohort_digest([first, second]) == cohort_digest([first, second])
    assert cohort_digest([first, second]) != cohort_digest([second, first])


def test_cosine_report_captures_exact_conflict_geometry() -> None:
    gradients = np.asarray(
        [
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    report = _cosine_report(gradients, [(0, 2)])
    matrix = np.asarray(report["cosine_matrix"])
    np.testing.assert_allclose(
        matrix,
        np.asarray(
            [
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    assert report["off_diagonal_negative_fraction"] == pytest.approx(1 / 3)
    assert report["off_diagonal_at_most_negative_0_10_fraction"] == pytest.approx(
        1 / 3
    )


def _arm_report(arm: str, gates: dict[str, bool]) -> dict[str, Any]:
    return {
        "experiment_id": "complete-action-frontier-fit-interference-audit-v1",
        "scientific": {
            "arm": arm,
            "gates": gates,
            "test_split_opened": False,
            "gameplay_opened": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
        },
        "telemetry": {
            "peak_process_rss_bytes": 1024,
            "process_swaps": 0,
            "system_swap_delta_bytes": 0,
        },
    }


def _reports(
    *,
    local: bool = True,
    collapse: bool = True,
    capacity: bool = False,
    gradient: bool = False,
    empirical: bool = False,
) -> dict[str, dict[str, Any]]:
    return {
        "nested-subset": _arm_report(
            "nested-subset",
            {
                "size1_local_fit": local,
                "size4_local_fit": local,
                "scaling_collapse_material": collapse,
                "all_nested_sizes_completed": True,
                "all_exposure_checkpoints_completed": True,
                "all_variants_finite": True,
            },
        ),
        "capacity-scaling": _arm_report(
            "capacity-scaling",
            {
                "recall_monotonic_with_tolerance": capacity,
                "capacity_material": capacity,
                "all_capacity_widths_completed": True,
                "all_exposure_checkpoints_completed": True,
                "all_variants_finite": True,
            },
        ),
        "gradient-conflict": _arm_report(
            "gradient-conflict",
            {
                "gradient_interference_material": gradient,
                "all_gradient_groups_completed": True,
                "all_gradient_norms_positive": True,
            },
        ),
        "error-anatomy": _arm_report(
            "error-anatomy",
            {
                "independent_local_recovery": local,
                "empirical_interference_material": empirical,
                "all_error_groups_completed": True,
                "all_error_scores_finite": True,
            },
        ),
    }


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"local": False},
            "local_optimization_or_representation_insufficient",
        ),
        (
            {"capacity": True},
            "shared_capacity_bottleneck",
        ),
        (
            {"gradient": True, "empirical": True},
            "cross_group_gradient_interference",
        ),
        (
            {"capacity": True, "gradient": True, "empirical": True},
            "mixed_capacity_and_interference",
        ),
        (
            {},
            "shared_model_scaling_failure_unresolved",
        ),
        (
            {"collapse": False},
            "no_material_fit_scaling_failure",
        ),
    ],
)
def test_classification_precedence(
    kwargs: dict[str, bool],
    expected: str,
) -> None:
    classification, _gates = classify_fit_interference(_reports(**kwargs))
    assert classification == expected


def test_pipeline_invalidity_has_highest_precedence() -> None:
    reports = _reports(capacity=True, gradient=True, empirical=True)
    reports["capacity-scaling"]["telemetry"]["process_swaps"] = 1
    classification, gates = classify_fit_interference(reports)
    assert classification == "fit_interference_pipeline_invalid"
    assert not gates["pipeline_passed"]


def test_pipeline_gate_rejects_missing_completion_or_swap_growth() -> None:
    report = _reports()["nested-subset"]
    assert _arm_pipeline_passed(report)
    report["scientific"]["gates"]["all_nested_sizes_completed"] = False
    assert not _arm_pipeline_passed(report)
    report["scientific"]["gates"]["all_nested_sizes_completed"] = True
    report["telemetry"]["system_swap_delta_bytes"] = 1
    assert not _arm_pipeline_passed(report)
