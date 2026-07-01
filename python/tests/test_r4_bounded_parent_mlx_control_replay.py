from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import r4_bounded_parent_mlx_control_replay as replay
from cascadia_mlx.r4_bounded_parent_mlx_cache import CONTROL_ARM
from cascadia_mlx.r4_bounded_parent_mlx_train import TRAINING_STEPS


def _checkpoint(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "checkpoint.json").write_text('{"schema_version":1}\n')
    (root / "model.safetensors").write_bytes(b"exact-c0-model")
    return root


def _authorization() -> dict:
    identity = {
        "protocol_id": replay.PROTOCOL_ID,
        "open_data_verification": {"open_data": True},
    }
    identity["open_data_verification_id"] = replay.open_data_verification_id(
        identity["open_data_verification"]
    )
    return {
        "schema_version": 1,
        "experiment_id": replay.EXPERIMENT_ID,
        "protocol_id": replay.PROTOCOL_ID,
        "adr": replay.ADR_ID,
        "approved": True,
        "identity": identity,
        "authorization_id": replay._canonical_blake3(identity),
    }


def _control_report(checkpoint: Path) -> dict:
    identity = {
        "experiment_id": replay.EXPERIMENT_ID,
        "protocol_id": replay.PROTOCOL_ID,
        "adr": replay.ADR_ID,
        "mode": "production",
        "arm": CONTROL_ARM,
        "host": "john1",
    }
    return {
        "schema_version": 1,
        **identity,
        "r3_cache_id": "1" * 64,
        "parent_cache_id": "2" * 64,
        "s1_cache_id": "3" * 64,
        "optimization": {"global_step": TRAINING_STEPS},
        "metrics": {
            "groups": replay.VALIDATION_GROUPS,
            "expected_groups": replay.VALIDATION_GROUPS,
            "candidates": replay.VALIDATION_ACTIONS,
            "expected_candidates": replay.VALIDATION_ACTIONS,
            "all_groups_scored_once": True,
            "all_candidates_scored_once": True,
        },
        "checkpoint": {
            "manifest_blake3": replay._checksum(checkpoint / "checkpoint.json"),
            "model_blake3": replay._checksum(checkpoint / "model.safetensors"),
        },
        "scientific_identity": identity,
        "report_id": replay._canonical_blake3(identity),
    }


def _performance(checkpoint: Path, *, host: str = "john2") -> dict:
    return {
        "measurement": {
            "isolated_process": True,
            "checkpoint_model_blake3": replay._checksum(checkpoint / "model.safetensors"),
            "open_data_verification_id": _authorization()["identity"]["open_data_verification_id"],
            "verification_source": "cluster-preflight",
            "request_id": "4" * 64,
            "result_id": "5" * 64,
            "worker_runtime": {"host": host},
        },
        "complete_decisions": {
            "groups": replay.VALIDATION_GROUPS,
            "actions": replay.VALIDATION_ACTIONS,
            "parent_encodes": replay.VALIDATION_GROUPS,
            "parent_encode_count_exact": True,
        },
        "fixed_chunk": {"action_scores_per_second": 50_000.0},
        "parent_encode": {"latency_milliseconds": {"p50": 2.0}},
        "memory": {"process_swaps": 0},
    }


def test_same_host_control_replay_binds_exact_c0_bytes(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    report = replay.build_replay_report(
        control_report=_control_report(checkpoint),
        authorization=_authorization(),
        performance=_performance(checkpoint),
        checkpoint=checkpoint,
        treatment_arm="q1-seat-marginal-parent",
        replay_host="john2",
    )
    assert report["host"] == "john2"
    assert report["control_arm"] == CONTROL_ARM
    assert report["scientific_identity"]["assertions"]["open_data_reverified"] is True
    assert report["replay_id"] == replay._canonical_blake3(report["scientific_identity"])


def test_control_replay_rejects_wrong_treatment_host(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    with pytest.raises(ValueError, match="assignment"):
        replay.build_replay_report(
            control_report=_control_report(checkpoint),
            authorization=_authorization(),
            performance=_performance(checkpoint, host="john3"),
            checkpoint=checkpoint,
            treatment_arm="q1-seat-marginal-parent",
            replay_host="john3",
        )


def test_control_replay_rejects_checkpoint_drift(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    origin = _control_report(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint bytes"):
        replay.build_replay_report(
            control_report=origin,
            authorization=_authorization(),
            performance=_performance(checkpoint),
            checkpoint=checkpoint,
            treatment_arm="q1-seat-marginal-parent",
            replay_host="john2",
        )


def test_control_replay_rejects_partial_validation(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    performance = _performance(checkpoint)
    performance["complete_decisions"]["groups"] = 239
    with pytest.raises(ValueError, match="coverage"):
        replay.build_replay_report(
            control_report=_control_report(checkpoint),
            authorization=_authorization(),
            performance=performance,
            checkpoint=checkpoint,
            treatment_arm="q1-seat-marginal-parent",
            replay_host="john2",
        )


def test_complete_validation_rows_are_derived_from_signed_control_report(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    rows = replay._complete_validation_rows(_control_report(checkpoint))
    np.testing.assert_array_equal(rows, np.arange(replay.VALIDATION_GROUPS))


def test_complete_validation_rows_reject_incomplete_origin_report(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    control = _control_report(checkpoint)
    control["metrics"]["groups"] -= 1
    with pytest.raises(ValueError, match="complete validation coverage"):
        replay._complete_validation_rows(control)


def test_main_requests_full_coverage_without_mutating_control_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_name = "step-000003000-epoch-0000-batch-003000"
    checkpoint = _checkpoint(run_dir / "checkpoints" / checkpoint_name)
    (run_dir / "latest.json").write_text(json.dumps({"checkpoint": checkpoint_name}) + "\n")
    control_path = tmp_path / "control.json"
    authorization_path = tmp_path / "authorization.json"
    control_path.write_text(json.dumps(_control_report(checkpoint)) + "\n")
    authorization_path.write_text(json.dumps(_authorization()) + "\n")
    output = tmp_path / "reports" / "paired-c0-q1.json"
    observed: dict[str, object] = {}

    def fake_benchmark(**kwargs: object) -> dict:
        observed.update(kwargs)
        return _performance(checkpoint)

    monkeypatch.setattr(replay, "run_isolated_serving_benchmark", fake_benchmark)
    monkeypatch.setattr(replay.socket, "gethostname", lambda: "john2")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "r4_bounded_parent_mlx_control_replay.py",
            "--control-report",
            str(control_path),
            "--authorization",
            str(authorization_path),
            "--treatment-arm",
            "q1-seat-marginal-parent",
            "--train-dataset",
            str(tmp_path / "train"),
            "--validation-dataset",
            str(tmp_path / "validation"),
            "--r3-cache",
            str(tmp_path / "r3"),
            "--parent-cache",
            str(tmp_path / "parent"),
            "--s1-cache",
            str(tmp_path / "s1"),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ],
    )

    assert replay.main() == 0
    np.testing.assert_array_equal(
        observed["decision_rows"],
        np.arange(replay.VALIDATION_GROUPS),
    )
    assert observed["artifact_dir"] == (
        output.parent / "paired-control-benchmarks" / "q1-seat-marginal-parent"
    )
    assert not list(run_dir.glob("serving-benchmark-*.json"))
