from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/graded_oracle_validation_report.py"
SPEC = importlib.util.spec_from_file_location("graded_oracle_validation_report", MODULE_PATH)
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


def test_host_aliases_are_explicit() -> None:
    assert reporter.normalize_host("Johns-Mac-mini") == "john1"
    assert reporter.normalize_host("john2") == "john2"
    with pytest.raises(ValueError, match="unknown Cascadia cluster host"):
        reporter.normalize_host("unregistered")


def test_frozen_artifacts_produce_rejection_without_opening_test() -> None:
    experiment_root = (
        ROOT
        / "artifacts/experiments/complete-action-graded-oracle-ranker-v1"
    )
    report = reporter.build_report(experiment_root)
    assert report["status"] == "rejected_before_test"
    assert report["replica_selection"]["selected_host"] == "john2"
    assert report["selected_validation"]["metrics"][
        "mean_top64_retained_r4800_regret"
    ] == pytest.approx(0.09018354415893555)
    assert report["selected_validation"]["metrics"][
        "top64_r4800_winner_recall"
    ] == pytest.approx(0.7333333333333333)
    assert not report["selected_validation"]["quality_passed"]
    assert report["performance_passed_on_all_hosts"]
    assert report["sealed_test"]["passed"]
    assert not report["sealed_test"]["test_authorization_exists"]
    assert not report["sealed_test"]["test_evaluation_output_exists"]
    assert not report["sealed_test"]["test_groups_read_by_reporter"]
    assert report["integrity"]["cross_host_metrics_bit_identical_to_selection"]
    assert not report["passed"]


def test_markdown_states_the_unopened_protocol_boundary() -> None:
    experiment_root = (
        ROOT
        / "artifacts/experiments/complete-action-graded-oracle-ranker-v1"
    )
    markdown = reporter.render_markdown(reporter.build_report(experiment_root))
    assert "sealed test and gameplay closed unopened" in markdown
    assert "ADR 0082: closed unopened" in markdown
    assert "K2048 and a large self-play launch remain closed" in markdown
