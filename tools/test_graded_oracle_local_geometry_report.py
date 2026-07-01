from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/graded_oracle_local_geometry_report.py"
SPEC = importlib.util.spec_from_file_location(
    "graded_oracle_local_geometry_report",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def test_frozen_replica_selection_order() -> None:
    replicas = [
        {
            "host": "john1",
            "seed": 3,
            "selection_loss": 0.1,
            "top64_r4800_winner_recall": 0.8,
            "r4800_residual_mae": 1.0,
        },
        {
            "host": "john2",
            "seed": 2,
            "selection_loss": 0.1,
            "top64_r4800_winner_recall": 0.9,
            "r4800_residual_mae": 2.0,
        },
        {
            "host": "john3",
            "seed": 1,
            "selection_loss": 0.09,
            "top64_r4800_winner_recall": 0.1,
            "r4800_residual_mae": 9.0,
        },
    ]
    assert reporter.select_replica(replicas)["host"] == "john3"
    replicas[2]["selection_loss"] = 0.1
    assert reporter.select_replica(replicas)["host"] == "john2"
    replicas[1]["top64_r4800_winner_recall"] = 0.8
    assert reporter.select_replica(replicas)["host"] == "john1"
    replicas[1]["r4800_residual_mae"] = 1.0
    replicas[1]["seed"] = 2
    assert reporter.select_replica(replicas)["host"] == "john2"


def test_cross_host_payload_mismatch_is_rejected() -> None:
    origin = {
        "host": "john1",
        "scientific_blake3": "same",
        "scientific": {"test_split_opened": False},
        "performance": {"passed": True},
    }
    cross = copy.deepcopy(origin)
    cross["host"] = "john3"
    cross["scientific"]["extra"] = 1
    with pytest.raises(ValueError, match="scientific payload drifted"):
        reporter.validate_report_pair(
            origin,
            cross,
            origin_host="john1",
            cross_host="john3",
        )


def test_frozen_artifacts_produce_rejection_without_opening_test() -> None:
    experiment_root = (
        ROOT
        / "artifacts/experiments/complete-action-local-geometry-ranker-v1"
    )
    report = reporter.build_report(experiment_root)
    assert report["status"] == "rejected_on_validation"
    assert report["replica_selection"]["selected_host"] == "john2"
    assert report["selected_validation"]["metrics"][
        "mean_top64_retained_r4800_regret"
    ] == pytest.approx(0.0937572161356608)
    assert report["selected_validation"]["metrics"][
        "top64_r4800_winner_recall"
    ] == pytest.approx(0.7416666666666667)
    assert report["selected_validation"]["confidence_top64"][
        "confidence_set_coverage_95"
    ] == pytest.approx(0.8791666666666667)
    assert not report["selected_validation"]["quality_passed"]
    assert report["cross_host_portability"]["performance_passed"]
    assert report["sealed_test"]["passed"]
    assert not report["sealed_test"]["test_authorization_exists"]
    assert not report["sealed_test"]["test_or_gameplay_output_exists"]
    assert report["execution"]["all_three_hosts_started_concurrently"]
    assert not report["passed"]


def test_markdown_states_scientific_failure_and_closed_boundary() -> None:
    experiment_root = (
        ROOT
        / "artifacts/experiments/complete-action-local-geometry-ranker-v1"
    )
    markdown = reporter.render_markdown(reporter.build_report(experiment_root))
    assert "failure is scientific" in markdown
    assert "sealed test and gameplay closed unopened" in markdown
    assert "Local geometry alone is not the missing mechanism" in markdown
