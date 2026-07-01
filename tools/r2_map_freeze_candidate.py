#!/usr/bin/env python3
"""Freeze a completed John1 R2-MAP run into one portable serving input."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import blake3

CHECKPOINT_FILES = (
    "checkpoint.json",
    "fixed-prediction-panel.safetensors",
    "model.safetensors",
    "optimizer.safetensors",
    "state.json",
)
TRAINING_RECEIPT_SCHEMA = "r2-map-training-command-receipt-v1"
CHECKPOINT_SCHEMA = "r2-map-checkpoint-v2"
VERIFICATION_SCHEMA = "r2-map-checkpoint-verification-v2"
REFERENCE_SCHEMA = "cascadia.r2-map.reference-panel-manifest.v1.1"


class FreezeCandidateError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _blake3_bytes(value: bytes) -> str:
    return blake3.blake3(value).hexdigest()


def _blake3_file(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeCandidateError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreezeCandidateError(f"JSON artifact is not an object: {path}")
    return value


def _verify_content_identity(value: dict[str, Any], field: str) -> None:
    claimed = value.get(field)
    payload = dict(value)
    payload.pop(field, None)
    if not isinstance(claimed, str) or claimed != _blake3_bytes(_canonical(payload)):
        raise FreezeCandidateError(f"{field} does not bind the canonical artifact")


def _verify_checkpoint(run_dir: Path, checkpoint_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint = run_dir / "checkpoints" / checkpoint_id
    manifest = _read_json(checkpoint / "checkpoint.json")
    if (
        manifest.get("schema_id") != CHECKPOINT_SCHEMA
        or manifest.get("schema_version") != 2
        or manifest.get("checkpoint_id") != checkpoint_id
    ):
        raise FreezeCandidateError("terminal checkpoint manifest identity differs")
    _verify_content_identity(manifest, "manifest_identity_blake3")
    files = manifest.get("files")
    expected_names = set(CHECKPOINT_FILES) - {"checkpoint.json"}
    if not isinstance(files, dict) or set(files) != expected_names:
        raise FreezeCandidateError("terminal checkpoint file manifest differs")
    for name in expected_names:
        descriptor = files[name]
        path = checkpoint / name
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("bytes") != path.stat().st_size
            or descriptor.get("blake3") != _blake3_file(path)
        ):
            raise FreezeCandidateError(f"terminal checkpoint file differs: {name}")

    receipt = _read_json(run_dir / "verifications" / f"{checkpoint_id}.json")
    if (
        receipt.get("schema_id") != VERIFICATION_SCHEMA
        or receipt.get("schema_version") != 2
        or receipt.get("checkpoint_id") != checkpoint_id
        or receipt.get("checkpoint_manifest_blake3") != _blake3_file(checkpoint / "checkpoint.json")
        or receipt.get("exact_prediction_match") is not True
        or receipt.get("exact_next_batch_match") is not True
    ):
        raise FreezeCandidateError("terminal checkpoint verification receipt differs")
    _verify_content_identity(receipt, "verification_id")
    return manifest, receipt


def _backend_parity_identity(value: dict[str, Any]) -> str:
    digest = blake3.blake3(b"r2-map-mlx-numpy-checkpoint-parity-receipt-v1")
    for name in (
        "schema_id",
        "checkpoint_id",
        "checkpoint_manifest_blake3",
        "model_weights_blake3",
        "verification_id",
    ):
        encoded = str(value.get(name, "")).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(bytes([bool(value.get("finite")), bool(value.get("passed"))]))
    return digest.hexdigest()


def _verify_backend_parity(
    path: Path,
    *,
    checkpoint_id: str,
    manifest: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    receipt = _read_json(path)
    maximum = receipt.get("maximum_absolute_error")
    tolerance = receipt.get("tolerance")
    if (
        receipt.get("schema_id") != "cascadia.r2-map.mlx-numpy-checkpoint-parity.v1"
        or receipt.get("checkpoint_id") != checkpoint_id
        or receipt.get("checkpoint_manifest_blake3") != verification["checkpoint_manifest_blake3"]
        or receipt.get("model_weights_blake3") != manifest["files"]["model.safetensors"]["blake3"]
        or receipt.get("verification_id") != verification["verification_id"]
        or receipt.get("finite") is not True
        or receipt.get("passed") is not True
        or not isinstance(maximum, (int, float))
        or not isinstance(tolerance, (int, float))
        or isinstance(maximum, bool)
        or isinstance(tolerance, bool)
        or tolerance <= 0.0
        or maximum > tolerance
        or receipt.get("receipt_blake3") != _backend_parity_identity(receipt)
    ):
        raise FreezeCandidateError("MLX/NumPy backend parity receipt differs")
    return receipt


def _terminal_checkpoint(run_dir: Path, expected_step: int) -> str:
    training = _read_json(run_dir / "training-command-receipt.json")
    if (
        training.get("schema_id") != TRAINING_RECEIPT_SCHEMA
        or training.get("schema_version") != 1
        or training.get("final_step") != expected_step
    ):
        raise FreezeCandidateError("training receipt is absent, incomplete, or at the wrong step")
    _verify_content_identity(training, "receipt_blake3")
    resources = training.get("resource_receipt")
    if (
        not isinstance(resources, dict)
        or resources.get("process_swaps") != 0
        or resources.get("system_swap_delta_bytes") != 0
        or not isinstance(resources.get("sample_count"), int)
        or resources["sample_count"] <= 0
    ):
        raise FreezeCandidateError("training resource receipt failed its zero-swap gate")
    pointer = _read_json(run_dir / "last_verified.json")
    checkpoint_id = pointer.get("checkpoint")
    if (
        not isinstance(checkpoint_id, str)
        or re.search(rf"\.step-{expected_step:09d}$", checkpoint_id) is None
        or training.get("best_validation_checkpoint") != checkpoint_id
    ):
        raise FreezeCandidateError(
            "terminal, last-verified, and best-validation checkpoints differ"
        )
    manifest, receipt = _verify_checkpoint(run_dir, checkpoint_id)
    if pointer.get("manifest_blake3") != receipt["checkpoint_manifest_blake3"]:
        raise FreezeCandidateError("last-verified pointer hash differs")
    checkpoints = training.get("checkpoints")
    final_entry = checkpoints[-1] if isinstance(checkpoints, list) and checkpoints else None
    if (
        not isinstance(final_entry, dict)
        or Path(str(final_entry.get("checkpoint"))).name != checkpoint_id
        or final_entry.get("checkpoint_manifest_blake3") != receipt["checkpoint_manifest_blake3"]
        or final_entry.get("verification_id") != receipt["verification_id"]
        or not isinstance(final_entry.get("validation"), dict)
        or manifest.get("identity", {}).get("run_id") != training.get("run_id")
    ):
        raise FreezeCandidateError("terminal checkpoint is not the validated training terminus")
    return checkpoint_id


def freeze_candidate(
    *,
    run_dir: Path,
    output: Path,
    reference_manifest: Path,
    backend_parity_receipt: Path,
    expected_step: int,
) -> dict[str, Any]:
    checkpoint_id = _terminal_checkpoint(run_dir, expected_step)
    manifest, verification = _verify_checkpoint(run_dir, checkpoint_id)
    backend_parity = _verify_backend_parity(
        backend_parity_receipt,
        checkpoint_id=checkpoint_id,
        manifest=manifest,
        verification=verification,
    )
    reference = _read_json(reference_manifest)
    implementation = reference.get("implementation_identity")
    if reference.get("schema_id") != REFERENCE_SCHEMA or not isinstance(implementation, dict):
        raise FreezeCandidateError("reference implementation manifest differs")
    protocol_names = {
        "collector_hash": "replay_pinecone_panel_sha256",
        "source_hash": "source_bundle_sha256",
        "serving_protocol_hash": "serving_protocol_schema_sha256",
    }
    protocols: dict[str, list[int]] = {}
    for target, source in protocol_names.items():
        try:
            decoded = bytes.fromhex(implementation[source])
        except (KeyError, TypeError, ValueError) as error:
            raise FreezeCandidateError("reference protocol identity differs") from error
        if len(decoded) != 32:
            raise FreezeCandidateError("reference protocol digest is not 32 bytes")
        protocols[target] = list(decoded)

    if output.exists():
        raise FreezeCandidateError(f"refusing to replace existing freeze: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        portable_run = temporary / "r2-run"
        portable_checkpoint = portable_run / "checkpoints" / checkpoint_id
        portable_verifications = portable_run / "verifications"
        portable_checkpoint.mkdir(parents=True)
        portable_verifications.mkdir(parents=True)
        source_checkpoint = run_dir / "checkpoints" / checkpoint_id
        for name in CHECKPOINT_FILES:
            shutil.copyfile(source_checkpoint / name, portable_checkpoint / name)
        shutil.copyfile(
            run_dir / "verifications" / f"{checkpoint_id}.json",
            portable_verifications / f"{checkpoint_id}.json",
        )
        shutil.copyfile(backend_parity_receipt, temporary / "r2-backend-parity.json")

        model = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_manifest_blake3": verification["checkpoint_manifest_blake3"],
            "model_config_blake3": manifest["identity"]["model_config_blake3"],
            "model_weights_blake3": manifest["files"]["model.safetensors"]["blake3"],
            "verification_id": verification["verification_id"],
        }
        bundle = {
            "schema_version": 2,
            "schema_id": "r2-map-local-serving-bundle-v2",
            "protocols": protocols,
            "entries": [
                {
                    "manifest_identity_blake3": manifest["manifest_identity_blake3"],
                    "run_dir": "/input/r2-run",
                    "checkpoint_path": f"/input/r2-run/checkpoints/{checkpoint_id}",
                    "model": model,
                    "pinned": True,
                }
            ],
        }
        (portable_run / "bundle.json").write_text(
            json.dumps(bundle, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        copied_manifest, copied_verification = _verify_checkpoint(portable_run, checkpoint_id)
        if copied_manifest != manifest or copied_verification != verification:
            raise FreezeCandidateError("portable checkpoint differs after copy")
        receipt = {
            "schema_id": "cascadia.r2-map.portable-candidate-freeze.v1",
            "schema_version": 1,
            "checkpoint_id": checkpoint_id,
            "checkpoint_manifest_blake3": verification["checkpoint_manifest_blake3"],
            "model_weights_blake3": model["model_weights_blake3"],
            "verification_id": verification["verification_id"],
            "bundle_blake3": _blake3_file(portable_run / "bundle.json"),
            "backend_parity_receipt_blake3": _blake3_file(temporary / "r2-backend-parity.json"),
            "backend_parity_identity": backend_parity["receipt_blake3"],
            "source_run_dir": str(run_dir),
            "container_run_dir": "/input/r2-run",
            "expected_step": expected_step,
        }
        receipt["receipt_blake3"] = _blake3_bytes(_canonical(receipt))
        (temporary / "freeze-receipt.json").write_text(
            json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        os.replace(temporary, output)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--backend-parity-receipt", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    arguments = parser.parse_args()
    try:
        receipt = freeze_candidate(
            run_dir=arguments.run_dir.resolve(strict=True),
            output=arguments.output,
            reference_manifest=arguments.reference_manifest.resolve(strict=True),
            backend_parity_receipt=arguments.backend_parity_receipt.resolve(strict=True),
            expected_step=arguments.expected_step,
        )
    except (FreezeCandidateError, OSError, ValueError) as error:
        parser.exit(2, f"R2-MAP candidate freeze refused: {error}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
