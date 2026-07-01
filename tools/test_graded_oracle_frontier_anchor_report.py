from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/graded_oracle_frontier_anchor_report.py"
SPEC = importlib.util.spec_from_file_location(
    "graded_oracle_frontier_anchor_report",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
reporter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def replica(
    host: str,
    *,
    seed: int,
    loss: float,
    coverage: float,
    regret: float,
) -> dict[str, object]:
    return {
        "host": host,
        "seed": seed,
        "selection_loss": loss,
        "top64_confidence_set_coverage_95": coverage,
        "mean_top64_retained_r4800_regret": regret,
    }


def test_frozen_replica_selection_order() -> None:
    replicas = [
        replica("john1", seed=3, loss=0.1, coverage=0.9, regret=0.2),
        replica("john2", seed=2, loss=0.1, coverage=0.95, regret=0.3),
        replica("john3", seed=1, loss=0.09, coverage=0.1, regret=9.0),
    ]
    assert reporter.select_replica(replicas)["host"] == "john3"
    replicas[2]["selection_loss"] = 0.1
    assert reporter.select_replica(replicas)["host"] == "john2"
    replicas[1]["top64_confidence_set_coverage_95"] = 0.9
    assert reporter.select_replica(replicas)["host"] == "john1"
    replicas[1]["mean_top64_retained_r4800_regret"] = 0.2
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


def test_cross_host_performance_failure_remains_reportable() -> None:
    origin = {
        "host": "john1",
        "scientific_blake3": "same",
        "scientific": {"test_split_opened": False},
        "performance": {"passed": True},
    }
    cross = copy.deepcopy(origin)
    cross["host"] = "john3"
    cross["performance"]["passed"] = False
    reporter.validate_report_pair(
        origin,
        cross,
        origin_host="john1",
        cross_host="john3",
    )


def test_sealed_state_rejects_authorization_or_outputs(tmp_path: Path) -> None:
    manifest = {"datasets": {"test": {"model_access": "sealed", "opened": False}}}
    assert reporter.verify_sealed_state(tmp_path, manifest)["passed"]
    (tmp_path / "test-authorization.json").write_text("{}\n")
    assert not reporter.verify_sealed_state(tmp_path, manifest)["passed"]
    (tmp_path / "test-authorization.json").unlink()
    (tmp_path / "gameplay-report-invalid.json").write_text("{}\n")
    assert not reporter.verify_sealed_state(tmp_path, manifest)["passed"]


def test_protocol_manifest_hash_ignores_closure_bookkeeping(tmp_path: Path) -> None:
    root = tmp_path
    path = root / "manifest.json"
    base = {
        "status": "active",
        "training": {"status": "active", "seed": 7},
        "treatment": {"width": 64},
    }
    path.write_text(json.dumps(base))
    first_inputs: dict[str, str] = {}
    reporter.load_protocol_manifest(path, first_inputs, root)
    base["status"] = "closed"
    base["training"]["status"] = "completed"
    base["closure"] = {"decision": "rejected"}
    path.write_text(json.dumps(base))
    second_inputs: dict[str, str] = {}
    reporter.load_protocol_manifest(path, second_inputs, root)
    assert first_inputs == second_inputs
    base["treatment"]["width"] = 32
    path.write_text(json.dumps(base))
    third_inputs: dict[str, str] = {}
    reporter.load_protocol_manifest(path, third_inputs, root)
    assert first_inputs != third_inputs


def test_execution_summary_accounts_for_each_host(tmp_path: Path) -> None:
    manifest = {
        "training": {
            "launched_at_unix_seconds": {
                "john1": 10.0,
                "john2": 11.0,
                "john3": 12.0,
                "john4": 13.0,
            },
            "launch_skew_seconds": 3.0,
        }
    }
    lines = []
    for index, host in enumerate(("john1", "john2", "john3", "john4")):
        queued = 100.0 + index
        lines.extend(
            (
                {
                    "event": "started",
                    "host": host,
                    "name": f"train-{host}",
                    "queued_unix_seconds": queued,
                    "started_unix_seconds": queued + 0.5,
                    "queued_seconds": 0.5,
                },
                {
                    "event": "finished",
                    "host": host,
                    "name": f"train-{host}",
                    "queued_unix_seconds": queued,
                    "started_unix_seconds": queued + 0.5,
                    "ended_unix_seconds": queued + 10.5,
                    "elapsed_seconds": 10.0,
                    "return_code": 0,
                },
            )
        )
    event_path = tmp_path / "events-test.jsonl"
    event_path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    summary = reporter.summarize_execution(tmp_path, manifest, {})
    assert summary["all_four_hosts_started_concurrently"]
    assert summary["total_jobs_completed"] == 4
    assert summary["total_jobs_failed"] == 0
    assert summary["aggregate_productive_wall_seconds"] == 40.0
    assert summary["by_host"]["john4"]["idle_with_work_queued_seconds"] == 0.5


def test_frozen_artifacts_produce_rejection_without_opening_test() -> None:
    experiment_root = (
        ROOT
        / "artifacts/experiments/complete-action-frontier-anchored-set-ranker-v1"
    )
    report = reporter.build_report(experiment_root)
    assert report["status"] == "rejected_on_validation"
    assert report["replica_selection"]["selected_host"] == "john2"
    assert report["selected_validation"]["metrics"][
        "top64_r4800_winner_recall"
    ] == pytest.approx(0.7666666666666667)
    assert report["selected_validation"]["metrics"][
        "top64_confidence_set_coverage_95"
    ] == pytest.approx(0.9041666666666667)
    assert report["selected_validation"]["metrics"][
        "target_positive_recall"
    ] == pytest.approx(0.2621323529411765)
    assert not report["selected_validation"]["quality_passed"]
    assert report["cross_host_portability"]["performance_passed"]
    assert report["integrity"]["all_replica_mlx_runtime_sources_identical"]
    assert report["execution"]["total_jobs_completed"] == 20
    assert report["execution"]["total_jobs_failed"] == 0
    assert report["sealed_test"]["passed"]
    assert not report["passed"]


def test_markdown_records_scientific_and_scheduling_failure() -> None:
    experiment_root = (
        ROOT
        / "artifacts/experiments/complete-action-frontier-anchored-set-ranker-v1"
    )
    markdown = reporter.render_markdown(reporter.build_report(experiment_root))
    assert "failure is scientific" in markdown
    assert "single-host MLX pilots plus independent experiments" in markdown
    assert "sealed test and gameplay closed unopened" in markdown
