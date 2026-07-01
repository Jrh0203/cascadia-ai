from __future__ import annotations

import json
from pathlib import Path

import blake3
import pytest
import r2_map_freeze_candidate as subject


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _artifact(tmp_path: Path, *, swaps: int = 0) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    checkpoint_id = "run.main.step-000000007"
    checkpoint = run / "checkpoints" / checkpoint_id
    verification_dir = run / "verifications"
    checkpoint.mkdir(parents=True)
    verification_dir.mkdir()
    payloads = {
        "fixed-prediction-panel.safetensors": b"panel",
        "model.safetensors": b"model",
        "optimizer.safetensors": b"optimizer",
        "state.json": b'{"dataset_contract":{},"next_batch_identity":"next"}',
    }
    for name, payload in payloads.items():
        (checkpoint / name).write_bytes(payload)
    manifest = {
        "schema_version": 2,
        "schema_id": subject.CHECKPOINT_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "identity": {"run_id": "run", "model_config_blake3": "a" * 64},
        "model_config": {},
        "files": {
            name: {"bytes": len(payload), "blake3": blake3.blake3(payload).hexdigest()}
            for name, payload in payloads.items()
        },
    }
    manifest["manifest_identity_blake3"] = blake3.blake3(_canonical(manifest)).hexdigest()
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode()
    (checkpoint / "checkpoint.json").write_bytes(manifest_bytes)
    receipt = {
        "schema_version": 2,
        "schema_id": subject.VERIFICATION_SCHEMA,
        "checkpoint_id": checkpoint_id,
        "checkpoint_manifest_blake3": blake3.blake3(manifest_bytes).hexdigest(),
        "exact_prediction_match": True,
        "exact_next_batch_match": True,
    }
    receipt["verification_id"] = blake3.blake3(_canonical(receipt)).hexdigest()
    (verification_dir / f"{checkpoint_id}.json").write_text(json.dumps(receipt))
    training = {
        "schema_version": 1,
        "schema_id": subject.TRAINING_RECEIPT_SCHEMA,
        "run_id": "run",
        "final_step": 7,
        "best_validation_checkpoint": checkpoint_id,
        "resource_receipt": {
            "process_swaps": 0,
            "system_swap_delta_bytes": swaps,
            "sample_count": 3,
        },
        "checkpoints": [
            {
                "checkpoint": str(checkpoint),
                "checkpoint_manifest_blake3": receipt["checkpoint_manifest_blake3"],
                "verification_id": receipt["verification_id"],
                "validation": {"total": 0.1},
            }
        ],
    }
    training["receipt_blake3"] = blake3.blake3(_canonical(training)).hexdigest()
    (run / "training-command-receipt.json").write_text(json.dumps(training))
    (run / "last_verified.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint_id,
                "manifest_blake3": receipt["checkpoint_manifest_blake3"],
            }
        )
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema_id": subject.REFERENCE_SCHEMA,
                "implementation_identity": {
                    "replay_pinecone_panel_sha256": "1" * 64,
                    "source_bundle_sha256": "2" * 64,
                    "serving_protocol_schema_sha256": "3" * 64,
                },
            }
        )
    )
    backend_parity = {
        "schema_id": "cascadia.r2-map.mlx-numpy-checkpoint-parity.v1",
        "checkpoint_id": checkpoint_id,
        "checkpoint_manifest_blake3": receipt["checkpoint_manifest_blake3"],
        "model_weights_blake3": manifest["files"]["model.safetensors"]["blake3"],
        "verification_id": receipt["verification_id"],
        "maximum_absolute_error": 1e-6,
        "tolerance": 2e-5,
        "finite": True,
        "passed": True,
    }
    backend_parity["receipt_blake3"] = subject._backend_parity_identity(backend_parity)
    parity_path = tmp_path / "backend-parity.json"
    parity_path.write_text(json.dumps(backend_parity))
    return run, reference, parity_path


def test_freeze_produces_portable_container_bundle(tmp_path: Path) -> None:
    run, reference, parity = _artifact(tmp_path)
    output = tmp_path / "frozen"
    receipt = subject.freeze_candidate(
        run_dir=run,
        output=output,
        reference_manifest=reference,
        backend_parity_receipt=parity,
        expected_step=7,
    )
    bundle = json.loads((output / "r2-run/bundle.json").read_text())
    assert receipt["checkpoint_id"] == "run.main.step-000000007"
    assert bundle["entries"][0]["checkpoint_path"].startswith("/input/r2-run/checkpoints/")
    assert bundle["entries"][0]["pinned"] is True
    assert bundle["protocols"]["collector_hash"] == [17] * 32
    assert (output / "r2-backend-parity.json").is_file()


def test_freeze_rejects_training_swap(tmp_path: Path) -> None:
    run, reference, parity = _artifact(tmp_path, swaps=4096)
    with pytest.raises(subject.FreezeCandidateError, match="zero-swap"):
        subject.freeze_candidate(
            run_dir=run,
            output=tmp_path / "frozen",
            reference_manifest=reference,
            backend_parity_receipt=parity,
            expected_step=7,
        )


def test_freeze_rejects_backend_parity_tampering(tmp_path: Path) -> None:
    run, reference, parity = _artifact(tmp_path)
    value = json.loads(parity.read_text())
    value["maximum_absolute_error"] = 1.0
    parity.write_text(json.dumps(value))
    with pytest.raises(subject.FreezeCandidateError, match="backend parity"):
        subject.freeze_candidate(
            run_dir=run,
            output=tmp_path / "frozen",
            reference_manifest=reference,
            backend_parity_receipt=parity,
            expected_step=7,
        )
