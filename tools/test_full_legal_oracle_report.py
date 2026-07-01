from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/full_legal_oracle_report.py"
SPEC = importlib.util.spec_from_file_location("full_legal_oracle_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
oracle = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = oracle
SPEC.loader.exec_module(oracle)


def frozen_input() -> oracle.OracleInput:
    path = (
        ROOT
        / "artifacts/experiments/full-legal-public-oracle-v1"
        / "john1/shard-62020-62023.json"
    )
    return oracle.OracleInput(
        path=path,
        host="john1",
        sha256=oracle.sha256_file(path),
        report=oracle.json.loads(path.read_text()),
    )


def test_expected_shard_parser() -> None:
    shard = oracle.parse_expected_shard("john2:62024:4")
    assert shard.host == "john2"
    assert list(shard.seeds) == [62_024, 62_025, 62_026, 62_027]
    with pytest.raises(oracle.argparse.ArgumentTypeError):
        oracle.parse_expected_shard("broken")


def test_host_count_parser() -> None:
    assert oracle.parse_host_count("john3:2") == ("john3", 2)
    with pytest.raises(oracle.argparse.ArgumentTypeError):
        oracle.parse_host_count("john3:-1")


def test_smoke_report_recomputes_raw_scores_and_integrity() -> None:
    item = frozen_input()
    expected = oracle.ExpectedShard("john1", 62_020, 4)
    oracle.validate_report(item, expected)
    report = oracle.analyze(
        [item],
        [expected],
        treatment_mean_minimum=100.0,
        paired_delta_minimum=3.0,
        bootstrap_samples=500,
        host_retries={"john1": 1},
    )
    assert report["baseline"]["mean"] == 96.375
    assert report["treatment"]["mean"] == 99.875
    assert report["paired_delta"]["mean"] == 3.5
    assert report["decision_summary"]["decisions"] == 320
    assert report["decision_summary"]["action_change_rate"] == 0.55
    assert report["gates"]["all_integrity_checks_passed"]
    assert report["gates"]["paired_delta_at_least_threshold"]
    assert not report["gates"]["treatment_mean_at_least_threshold"]
    assert report["host_utilization"]["john1"]["failures_or_retries_observed"] == 1
    assert report["cluster_utilization"]["total_failures_or_retries_observed"] == 1
    assert report["decision_latency"]["treatment"]["mean_milliseconds"] > 1_000.0
    assert not report["passed"]


def test_confirmation_stage_enforces_strict_registered_gates() -> None:
    item = frozen_input()
    expected = oracle.ExpectedShard("john1", 62_020, 4)
    report = oracle.analyze(
        [item],
        [expected],
        treatment_mean_minimum=102.0,
        paired_delta_minimum=6.0,
        bootstrap_samples=500,
        stage="confirmation",
        require_positive_delta_confidence=True,
        require_positive_host_deltas=True,
    )

    assert report["stage"] == "confirmation"
    assert report["status"] == "confirmation_failed"
    assert report["gates"]["paired_delta_bootstrap_lower_bound_positive"]
    assert report["gates"]["every_host_paired_delta_positive"]
    assert not report["gates"]["treatment_mean_at_least_threshold"]
    assert not report["gates"]["paired_delta_at_least_threshold"]


def test_screen_limit_is_part_of_the_frozen_report_contract() -> None:
    item = frozen_input()
    expected = oracle.ExpectedShard("john1", 62_020, 4)
    oracle.validate_report(item, expected, screen_limit=64)

    k1024_report = copy.deepcopy(item.report)
    k1024_report["config"]["screen_limit"] = 1_024
    k1024_item = oracle.OracleInput(
        path=item.path,
        host=item.host,
        sha256=item.sha256,
        report=k1024_report,
    )
    oracle.validate_report(k1024_item, expected, screen_limit=1_024)
    with pytest.raises(ValueError, match="configuration drifted"):
        oracle.validate_report(k1024_item, expected, screen_limit=64)


def test_positive_integer_rejects_invalid_screen_widths() -> None:
    assert oracle.positive_integer("1024") == 1_024
    with pytest.raises(oracle.argparse.ArgumentTypeError):
        oracle.positive_integer("0")
    with pytest.raises(oracle.argparse.ArgumentTypeError):
        oracle.positive_integer("not-an-integer")


def test_markdown_uses_correct_loss_plural() -> None:
    item = frozen_input()
    expected = oracle.ExpectedShard("john1", 62_020, 4)
    report = oracle.analyze(
        [item],
        [expected],
        treatment_mean_minimum=100.0,
        paired_delta_minimum=3.0,
        bootstrap_samples=500,
    )
    report["game_losses"] = 2
    markdown = oracle.render_markdown(report)
    assert "2 losses" in markdown
    assert "losss" not in markdown


def test_artifact_checksum_manifest_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("complete\n")
    digest = oracle.sha256_file(artifact)
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  ./artifact.txt\n")

    manifest_digest = oracle.verify_artifact_checksum_manifest(
        tmp_path,
        {"artifact.txt"},
    )
    assert manifest_digest == oracle.sha256_file(tmp_path / "SHA256SUMS")

    artifact.write_text("tampered\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        oracle.verify_artifact_checksum_manifest(tmp_path, {"artifact.txt"})


def test_frozen_identity_drift_is_rejected() -> None:
    item = frozen_input()
    expected = oracle.ExpectedShard("john1", 62_020, 4)
    drifted_report = copy.deepcopy(item.report)
    drifted_report["executable_blake3"] = "0" * 64
    drifted = oracle.OracleInput(
        path=item.path,
        host=item.host,
        sha256=item.sha256,
        report=drifted_report,
    )
    with pytest.raises(ValueError, match="frozen executable_blake3 identity drifted"):
        oracle.validate_report(drifted, expected)


def test_markdown_uses_dynamic_screen_width_and_singular_counts() -> None:
    item = frozen_input()
    expected = oracle.ExpectedShard("john1", 62_020, 4)
    report = oracle.analyze(
        [item],
        [expected],
        treatment_mean_minimum=100.0,
        paired_delta_minimum=3.0,
        bootstrap_samples=100,
        experiment_id="full-legal-public-oracle-k1024-v1",
    )
    report["configuration"] = copy.deepcopy(report["configuration"])
    report["configuration"]["screen_limit"] = 1_024
    report["game_losses"] = 1
    markdown = oracle.render_markdown(report)
    assert "Top-1024 winner rate" in markdown
    assert "1 loss" in markdown
    assert "1 losses" not in markdown


def test_score_validation_rejects_habitat_bonus() -> None:
    score = {
        "habitat": [1, 1, 1, 1, 1],
        "wildlife": [1, 1, 1, 1, 1],
        "nature_tokens": 1,
        "habitat_bonus": [1, 0, 0, 0, 0],
        "base_total": 11,
        "total": 11,
    }
    with pytest.raises(ValueError, match="habitat bonuses"):
        oracle.validate_score(score, "test")
