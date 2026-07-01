from __future__ import annotations

import json
import struct
from pathlib import Path

import v3_recovery_receipt as recovery


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _run(root: Path, name: str, payload: bytes) -> Path:
    run = root / name
    checkpoint = run / "checkpoints" / "step-final"
    checkpoint.mkdir(parents=True)
    _write(run / "latest.json", {"checkpoint": "step-final"})
    _write(run / "run-manifest.json", {"canonical_blake3": "a" * 64})
    _write(
        run / "engineering-training-report.json",
        {
            "state": {"global_step": 5, "epoch": 1},
            "metrics": {"skipped_resume_batches": 2.0 if name == "resumed" else 0.0},
            "run_manifest_blake3": "a" * 64,
            "training_config": {"examples": 5, "logical_batch_size": 1, "epochs": 1},
        },
    )
    _write_safetensors(checkpoint / "model.safetensors", payload)
    _write_safetensors(checkpoint / "optimizer.safetensors", payload[::-1])
    return run


def _write_safetensors(path: Path, payload: bytes) -> None:
    header = json.dumps(
        {"value": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}},
        separators=(",", ":"),
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_exact_twin_is_certified(tmp_path: Path) -> None:
    resumed = _run(tmp_path, "resumed", b"model")
    control = _run(tmp_path, "control", b"model")
    interrupted = tmp_path / "interrupted.json"
    _write(
        interrupted,
        {"state": {"global_step": 2}, "metrics": {"interrupted": True}},
    )
    result = recovery.certify(interrupted, resumed, control, tmp_path / "receipt.json")
    assert result["passed"] is True


def test_different_model_is_rejected(tmp_path: Path) -> None:
    resumed = _run(tmp_path, "resumed", b"model")
    control = _run(tmp_path, "control", b"other")
    interrupted = tmp_path / "interrupted.json"
    _write(
        interrupted,
        {"state": {"global_step": 2}, "metrics": {"interrupted": True}},
    )
    result = recovery.certify(interrupted, resumed, control, tmp_path / "receipt.json")
    assert result["passed"] is False
