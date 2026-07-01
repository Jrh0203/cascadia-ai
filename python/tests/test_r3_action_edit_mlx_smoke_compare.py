from __future__ import annotations

import copy
import json
from pathlib import Path

import blake3
import mlx.core as mx
import pytest
import r3_action_edit_mlx_smoke_compare as smoke_tool


def _checkpoint(path: Path, values: list[float]) -> None:
    mx.save_safetensors(path, {"weight": mx.array(values, dtype=mx.float32)})


def _report(host: str, checkpoint: Path, *, offset: float = 0.0) -> dict:
    trace = [
        {
            "schema_version": 1,
            "step": step,
            "batch_blake3": blake3.blake3(step.to_bytes(8, "little")).hexdigest(),
            "loss": 20.0 + step + offset,
            "candidates": 2_000 + step,
            "elapsed_seconds": 0.1,
        }
        for step in range(1, 11)
    ]
    report = {
        "schema_version": 1,
        "experiment_id": smoke_tool.EXPERIMENT_ID,
        "protocol_id": smoke_tool.PROTOCOL_ID,
        "adr": smoke_tool.ADR_ID,
        "mode": "bounded-smoke",
        "arm": smoke_tool.SMOKE_ARM,
        "host": host,
        "cache_id": "1" * 64,
        "s1_cache_id": "2" * 64,
        "protocol": {"seed": 2026061708},
        "model": {
            "parameter_count": 3,
            "parameter_layout_blake3": "3" * 64,
            "initial_parameter_tensor_blake3": "4" * 64,
            "final_parameter_tensor_blake3": "5" * 64,
        },
        "optimization": {
            "global_step": 10,
            "candidates": sum(event["candidates"] for event in trace),
            "loss_trace": trace,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_blake3": "6" * 64,
            "model_blake3": smoke_tool._checksum(checkpoint),
        },
        "metrics": {
            "prediction_panel": {
                "action_hashes": [
                    blake3.blake3(index.to_bytes(4, "little")).hexdigest() for index in range(3)
                ],
                "scores": [3.0 + offset, 2.0 + offset, 1.0 + offset],
                "standard_errors": [
                    0.5 + offset,
                    0.6 + offset,
                    0.7 + offset,
                ],
            },
        },
        "performance": {},
        "runtime": {},
        "source": {},
        "controls": None,
        "information_boundary": {
            "sealed_test_opened": False,
            "gameplay_run": False,
        },
        "claims": {
            "bounded_smoke_complete": True,
            "offline_comparison_complete": False,
            "gameplay_strength_measured": False,
            "promotion_authorized": False,
            "progress_to_100_claimed": False,
        },
    }
    return _reseal(report)


def _reseal(report: dict) -> dict:
    report["scientific_identity"] = smoke_tool._report_scientific_identity(report)
    report["report_id"] = smoke_tool._canonical_blake3(report["scientific_identity"])
    return report


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    left_checkpoint = tmp_path / "left.safetensors"
    right_checkpoint = tmp_path / "right.safetensors"
    _checkpoint(left_checkpoint, [1.0, 2.0, 3.0])
    _checkpoint(right_checkpoint, [1.0 + 1e-6, 2.0, 3.0])
    left_report = tmp_path / "left.json"
    right_report = tmp_path / "right.json"
    _write(left_report, _report("john1", left_checkpoint))
    _write(right_report, _report("john4", right_checkpoint, offset=1e-6))
    return left_report, right_report, left_checkpoint, right_checkpoint


def test_numerically_equivalent_smoke_passes_with_exact_scientific_identity(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = smoke_tool.compare_smoke(*paths)
    assert result["classification"] == smoke_tool.PASS
    assert result["scientific_identity"]["checks"]["batch_identity_exact"] is True
    assert result["scientific_identity"]["checks"]["panel_stable_ranking_exact"] is True
    assert result["claims"]["production_training_started"] is False


def test_batch_drift_fails_even_when_numeric_values_are_close(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    right = json.loads(paths[1].read_text())
    right["optimization"]["loss_trace"][5]["batch_blake3"] = "9" * 64
    _reseal(right)
    _write(paths[1], right)
    with pytest.raises(smoke_tool.SmokeParityError, match="batch identity"):
        smoke_tool.compare_smoke(*paths)


def test_parameter_drift_outside_tolerance_fails(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _checkpoint(paths[3], [1.01, 2.0, 3.0])
    right = json.loads(paths[1].read_text())
    right["checkpoint"]["model_blake3"] = smoke_tool._checksum(paths[3])
    _reseal(right)
    _write(paths[1], right)
    with pytest.raises(smoke_tool.SmokeParityError, match="numerical parity"):
        smoke_tool.compare_smoke(*paths)


def test_unsealed_report_mutation_fails_closed(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    right = json.loads(paths[1].read_text())
    mutated = copy.deepcopy(right)
    mutated["host"] = "john1"
    _write(paths[1], mutated)
    with pytest.raises(smoke_tool.SmokeParityError, match="malformed"):
        smoke_tool.compare_smoke(*paths)
