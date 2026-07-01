from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
import s4_candidate_set_mlx_smoke_compare as compare


def _report(host: str, model_path: Path) -> dict[str, object]:
    panel = {
        "action_hashes": ["a" * 64, "b" * 64],
        "scores": [2.0, 1.0],
        "standard_errors": [0.1, 0.2],
    }
    warm_start = {
        "checkpoint_id": "c" * 64,
        "global_step": 3000,
        "model_config": {"arm": "t3-r3-radius1-global"},
        "manifest_blake3": "d" * 64,
        "model_blake3": "e" * 64,
    }
    parity = {
        "row": 0,
        "candidates": 2,
        "scores_byte_identical": True,
        "standard_errors_byte_identical": True,
        "prediction_blake3": "f" * 64,
    }
    model = {
        "parameter_count": 2,
        "parameter_layout_blake3": "1" * 64,
        "initial_parameter_tensor_blake3": "2" * 64,
    }
    trace = [
        {
            "step": step,
            "batch_blake3": f"{step:064x}",
            "loss": 1.0 / step,
            "candidates": 2,
            "elapsed_seconds": 0.1,
        }
        for step in range(1, compare.SMOKE_STEPS + 1)
    ]
    scientific = {
        "experiment_id": compare.EXPERIMENT_ID,
        "protocol_id": compare.PROTOCOL_ID,
        "adr": compare.ADR_ID,
        "mode": "bounded-smoke",
        "arm": compare.SMOKE_ARM,
        "host": host,
        "cache_id": "3" * 64,
        "s1_cache_id": "4" * 64,
        "context_cache_id": "5" * 64,
        "protocol": {"seed": 1},
        "warm_start": warm_start,
        "initial_prediction_parity": parity,
        "model": model,
        "optimization": {
            "global_step": compare.SMOKE_STEPS,
            "loss_trace": trace,
        },
        "checkpoint": {
            "model_blake3": compare._checksum(model_path),
        },
        "metrics": {"prediction_panel": panel},
        "performance": {},
        "runtime": {},
        "source": {},
        "controls": None,
        "information_boundary": {},
        "claims": {
            "bounded_smoke_complete": True,
            "offline_comparison_complete": False,
        },
    }
    return {
        "schema_version": 1,
        **scientific,
        "scientific_identity": scientific,
        "report_id": compare._canonical_blake3(scientific),
    }


def test_identical_cross_host_smoke_passes(tmp_path: Path) -> None:
    left_model = tmp_path / "left.safetensors"
    right_model = tmp_path / "right.safetensors"
    tensors = {"weight": mx.array([1.0, 2.0], dtype=mx.float32)}
    mx.save_safetensors(str(left_model), tensors)
    mx.save_safetensors(str(right_model), tensors)
    left = _report("john1", left_model)
    right = _report("john4", right_model)
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left))
    right_path.write_text(json.dumps(right))

    result = compare.compare_smoke(
        left_path,
        right_path,
        left_model,
        right_model,
    )
    assert result["classification"] == compare.PASS
    assert all(result["scientific_identity"]["checks"].values())


def test_batch_drift_is_rejected(tmp_path: Path) -> None:
    left_model = tmp_path / "left.safetensors"
    right_model = tmp_path / "right.safetensors"
    tensors = {"weight": mx.array([1.0, 2.0], dtype=mx.float32)}
    mx.save_safetensors(str(left_model), tensors)
    mx.save_safetensors(str(right_model), tensors)
    left = _report("john1", left_model)
    right = _report("john4", right_model)
    right["optimization"]["loss_trace"][0]["batch_blake3"] = "9" * 64
    right["scientific_identity"] = {
        key: right[key]
        for key in right["scientific_identity"]
    }
    right["report_id"] = compare._canonical_blake3(
        right["scientific_identity"]
    )
    left_path = tmp_path / "left.json"
    right_path = tmp_path / "right.json"
    left_path.write_text(json.dumps(left))
    right_path.write_text(json.dumps(right))

    with pytest.raises(compare.SmokeParityError, match="batch identity"):
        compare.compare_smoke(
            left_path,
            right_path,
            left_model,
            right_model,
        )
