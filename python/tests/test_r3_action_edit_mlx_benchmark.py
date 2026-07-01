from __future__ import annotations

from pathlib import Path

import pytest
from cascadia_mlx.r3_action_edit_mlx_benchmark import (
    R3ServingBenchmarkError,
    _validate_request,
    create_serving_benchmark_request,
)
from cascadia_mlx.r3_action_edit_mlx_cache import ARMS


def _checkpoint(root: Path) -> Path:
    checkpoint = root / "checkpoints" / "step-000000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "checkpoint.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    return checkpoint


def test_serving_benchmark_request_binds_checkpoint_data_and_protocol(
    tmp_path: Path,
) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    open_data = {
        "cache_id": "1" * 64,
        "cache_manifest_blake3": "2" * 64,
        "s1_cache_id": "3" * 64,
        "s1_cache_manifest_blake3": "4" * 64,
        "datasets": {
            "train": {
                "dataset_id": "5" * 64,
                "manifest_blake3": "6" * 64,
            },
            "validation": {
                "dataset_id": "7" * 64,
                "manifest_blake3": "8" * 64,
            },
        },
    }
    request = create_serving_benchmark_request(
        train_dataset=tmp_path / "train",
        validation_dataset=tmp_path / "validation",
        cache=tmp_path / "cache",
        s1_cache=tmp_path / "s1",
        run_dir=tmp_path / "run",
        checkpoint=checkpoint,
        arm=ARMS[0],
        global_step=1,
        open_data_verification=open_data,
        verification_source="cluster-preflight",
        warmup_iterations=5,
        steady_iterations=30,
    )

    identity = _validate_request(request)
    assert identity["arm"] == ARMS[0]
    assert identity["checkpoint"]["global_step"] == 1
    assert identity["decision_rows"] is None

    request["scientific_identity"]["steady_iterations"] = 29
    with pytest.raises(R3ServingBenchmarkError, match="malformed"):
        _validate_request(request)


def test_serving_benchmark_rejects_unproved_verification_source(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path / "run")
    with pytest.raises(ValueError, match="verification source"):
        create_serving_benchmark_request(
            train_dataset=tmp_path / "train",
            validation_dataset=tmp_path / "validation",
            cache=tmp_path / "cache",
            s1_cache=tmp_path / "s1",
            run_dir=tmp_path / "run",
            checkpoint=checkpoint,
            arm=ARMS[0],
            global_step=1,
            open_data_verification={},
            verification_source="trust-me",
            warmup_iterations=1,
            steady_iterations=1,
        )
